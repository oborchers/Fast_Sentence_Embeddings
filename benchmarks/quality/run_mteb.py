"""Measure the quality of fse sentence vectors on MTEB against other static embedding models.

The task selection follows what the compared libraries report themselves:

- ``MTEB(eng, v2)``, the 41 tasks of the current English MTEB leaderboard, which the
  sentence-transformers documentation uses for its MTEB examples;
- ``NanoBEIR``, the 13 small BEIR retrieval tasks of the sentence-transformers static models.

SIF and uSIF learn their principal components once from a fixed corpus that is unrelated to
every task (the training split of WikiText-103) and are frozen afterwards: every task only calls
``infer``, like the compared models, which are frozen too. Average is trained on the same corpus,
which has no effect on its vectors.

All fse models tokenize as ``notebooks/STS-Benchmarks.ipynb`` does: NLTK ``word_tokenize``,
lowercased, punctuation dropped, hyphenated words split. The compared models use their own
tokenizers.

For every task the script records the main score, the wall time of the ``encode`` calls (for fse
including tokenization), the number of encoded texts and the load average before the task.
Encoding speed depends on the machine and its load; compare it only within one run.

Setup, from the repository root (the benchmark dependencies stay out of the project)::

    uv venv -p 3.12 .venv-quality
    uv pip install -p .venv-quality/bin/python -e . -r benchmarks/quality/requirements.txt

Examples::

    .venv-quality/bin/python benchmarks/quality/run_mteb.py run --models fse/uSIF-paranmt-300-float32 --tasks STSBenchmark
    .venv-quality/bin/python benchmarks/quality/run_mteb.py run --models all
    .venv-quality/bin/python benchmarks/quality/run_mteb.py summarize
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import statistics
import sys
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mteb
import numpy as np
from mteb.cache import ResultCache
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.jsonl"
CACHE = Path(os.environ.get("FSE_QUALITY_CACHE", Path.home() / ".cache" / "fse-quality"))

BENCHMARKS = ["MTEB(eng, v2)", "NanoBEIR"]
WIKITEXT = ("Salesforce/wikitext", "wikitext-103-raw-v1", "b08601e04326c79dfdd32d625aee71d232d685c3")

FSE_VECTORS = [
    "glove-wiki-gigaword-300",
    "word2vec-google-news-300",
    "fasttext-wiki-news-subwords-300",
    "paranmt-300",
]
# The parameters of the STS table in the README.
FSE_ALGOS = {
    "Average": {},
    "SIF": {"components": 10, "lang_freq": "en"},
    "uSIF": {"length": 11, "lang_freq": "en"},
}
FSE_DTYPES = ["float32", "int8"]
FSE_MODELS = [f"fse/{a}-{v}-{d}" for a in FSE_ALGOS for v in FSE_VECTORS for d in FSE_DTYPES]
# Registered in mteb with pinned revisions.
OTHER_MODELS = [
    "minishlab/potion-base-8M",
    "minishlab/potion-base-32M",
    "sentence-transformers/static-retrieval-mrl-en-v1",
    "sentence-transformers/all-MiniLM-L6-v2",
]

logger = logging.getLogger("run_mteb")

_NOT_PUNCTUATION = re.compile(".*[A-Za-z0-9].*")


def _prep_token(token: str) -> list[str]:
    t = token.lower().strip("';.:()").strip('"')
    t = "not" if t == "n't" else t
    return re.split(r"[-]", t)


def tokenize(text: str) -> list[str]:
    """Tokenize like notebooks/STS-Benchmarks.ipynb."""
    from nltk import word_tokenize

    tokens: list[str] = []
    for token in word_tokenize(text):
        if _NOT_PUNCTUATION.match(token):
            tokens.extend(_prep_token(token))
    return tokens


def fit_corpus() -> Path:
    """Write the tokenized WikiText-103 training paragraphs once, one per line."""
    path = CACHE / f"wikitext-103-{WIKITEXT[2][:8]}.txt"
    if path.exists():
        return path
    from datasets import load_dataset

    CACHE.mkdir(parents=True, exist_ok=True)
    data = load_dataset(WIKITEXT[0], WIKITEXT[1], split="train", revision=WIKITEXT[2])
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for line in data["text"]:
            line = line.strip()
            if not line or line.startswith("="):  # empty lines and section headings
                continue
            tokens = tokenize(line)
            if tokens:
                f.write(" ".join(tokens) + "\n")
    tmp.rename(path)
    return path


class FseEncoder(AbsEncoder):
    """Encode with a frozen fse model: trained once on the fit corpus, then only ``infer``."""

    def __init__(self, model_name: str, revision: str | None = None, **kwargs: Any) -> None:
        from fse import SIF, Average, CorpusFile, Vectors, uSIF

        algo, rest = model_name.removeprefix("fse/").split("-", 1)
        vectors_name, dtype = rest.rsplit("-", 1)
        cls = {"Average": Average, "SIF": SIF, "uSIF": uSIF}[algo]
        vectors = Vectors.from_pretrained(vectors_name, mmap="r")
        self.model = cls(vectors, wv_dtype=dtype, workers=os.cpu_count(), **FSE_ALGOS[algo])
        start = time.perf_counter()
        self.model.train(CorpusFile(str(fit_corpus())))
        logger.info("fitted %s in %.1f s", model_name, time.perf_counter() - start)

    def encode(self, inputs: Any, *, task_metadata: Any, hf_split: str, hf_subset: str, prompt_type: Any = None, **kwargs: Any) -> np.ndarray:
        from fse import IndexedList

        sentences = [tokenize(text) for batch in inputs for text in batch["text"]]
        return self.model.infer(IndexedList(sentences))


def fse_meta(name: str) -> ModelMeta:
    return ModelMeta(
        loader=FseEncoder,
        name=name,
        revision=version("fse"),
        release_date=None,
        languages=["eng-Latn"],
        n_parameters=None,
        memory_usage_mb=None,
        max_tokens=np.inf,
        embed_dim=300,
        license="gpl-3.0",
        open_weights=True,
        public_training_code="https://github.com/oborchers/Fast_Sentence_Embeddings",
        public_training_data="https://huggingface.co/datasets/Salesforce/wikitext",
        framework=["NumPy"],
        reference="https://huggingface.co/fse",
        similarity_fn_name=ScoringFunction.COSINE,
        use_instructions=False,
        training_datasets=set(),
    )


class EncodeTimer:
    """Wrap a model's ``encode`` to record wall time, texts and load average per task."""

    def __init__(self, model: Any) -> None:
        self.stats: dict[str, dict[str, float]] = defaultdict(lambda: {"seconds": 0.0, "texts": 0})
        encode = model.encode

        def timed(inputs: Any, *args: Any, task_metadata: Any, **kwargs: Any) -> Any:
            entry = self.stats[task_metadata.name]
            entry.setdefault("load_before", os.getloadavg()[0])
            start = time.perf_counter()
            out = encode(inputs, *args, task_metadata=task_metadata, **kwargs)
            entry["seconds"] += time.perf_counter() - start
            entry["texts"] += len(out)
            return out

        model.encode = timed


def tasks_for(names: list[str] | None) -> list[Any]:
    tasks = [t for b in BENCHMARKS for t in mteb.get_benchmark(b).tasks]
    if names:
        tasks = [t for t in tasks if t.metadata.name in names]
        missing = set(names) - {t.metadata.name for t in tasks}
        if missing:
            raise SystemExit(f"unknown tasks: {sorted(missing)}")
    return tasks


def run(models: list[str], task_names: list[str] | None) -> None:
    from nltk import download

    download("punkt_tab", quiet=True)
    tasks = tasks_for(task_names)
    benchmark_of = {t.metadata.name: b for b in BENCHMARKS for t in mteb.get_benchmark(b).tasks}
    cache = ResultCache(CACHE / "mteb")
    for name in models:
        meta = fse_meta(name) if name.startswith("fse/") else mteb.get_model_meta(name)
        model = meta.load_model()
        timer = EncodeTimer(model)
        result = mteb.evaluate(model, tasks, cache=cache, overwrite_strategy="always", show_progress_bar=False)
        with RESULTS.open("a", encoding="utf-8") as f:
            for task_result in result.task_results:
                task = task_result.task_name
                f.write(
                    json.dumps(
                        {
                            "model": name,
                            "revision": meta.revision,
                            "benchmark": benchmark_of[task],
                            "task": task,
                            "type": mteb.get_task(task).metadata.type,
                            "main_score": task_result.get_score(),
                            **timer.stats.get(task, {}),
                            "machine": platform.platform(),
                            "python": sys.version.split()[0],
                            "mteb": version("mteb"),
                            "fse": version("fse"),
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        }
                    )
                    + "\n"
                )
        logger.info("finished %s", name)


def summarize() -> None:
    """Print a Markdown table of the latest result per model and task."""
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        latest[row["model"], row["task"]] = row
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (model, _), row in latest.items():
        by_model[model].append(row)
    eng = {t.metadata.name for t in mteb.get_benchmark("MTEB(eng, v2)").tasks}
    types = sorted({t.metadata.type for t in mteb.get_benchmark("MTEB(eng, v2)").tasks})
    print("| Model | Mean (Task) | Mean (TaskType) | " + " | ".join(types) + " | NanoBEIR | texts/s |")
    print("|---" * (len(types) + 5) + "|")
    for model, rows in sorted(by_model.items()):
        main = [r for r in rows if r["task"] in eng]
        nano = [r["main_score"] for r in rows if r["benchmark"] == "NanoBEIR"]
        per_type = {ty: [r["main_score"] for r in main if r["type"] == ty] for ty in types}
        complete = len(main) == len(eng)
        mean_task = statistics.mean(r["main_score"] for r in main) if complete else float("nan")
        mean_type = statistics.mean(statistics.mean(v) for v in per_type.values()) if complete else float("nan")
        texts = sum(r.get("texts", 0) for r in rows)
        seconds = sum(r.get("seconds", 0.0) for r in rows)
        cells = [f"{100 * statistics.mean(v):.2f}" if v else "" for v in per_type.values()]
        print(
            f"| {model} | {100 * mean_task:.2f} | {100 * mean_type:.2f} | "
            + " | ".join(cells)
            + f" | {100 * statistics.mean(nano):.2f} | {texts / seconds:,.0f} |"
            if nano and seconds
            else f"| {model} | {100 * mean_task:.2f} | {100 * mean_type:.2f} | " + " | ".join(cells) + " | | |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--models", nargs="+", required=True, help="model names, 'fse', 'others' or 'all'")
    run_parser.add_argument("--tasks", nargs="+", help="restrict to these task names")
    sub.add_parser("summarize")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if args.command == "summarize":
        summarize()
        return
    groups = {"fse": FSE_MODELS, "others": OTHER_MODELS, "all": FSE_MODELS + OTHER_MODELS}
    models = [m for name in args.models for m in groups.get(name, [name])]
    run(models, args.tasks)


if __name__ == "__main__":
    main()
