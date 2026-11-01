"""Measure the quality of fse sentence vectors on MTEB against other static embedding models.

The task selection follows what the compared libraries report themselves:

- ``MTEB(eng, v2)``, the 41 tasks of the current English MTEB leaderboard, which the
  sentence-transformers documentation uses for its MTEB examples;
- ``NanoBEIR``, the 13 small BEIR retrieval tasks of the sentence-transformers static models.

The fse models are finished models (``fse.Encoder`` files, one ``.safetensors`` each) in
--fse-models (default ``$FSE_QUALITY_CACHE/models``), named ``fse/<file name without
.safetensors>``; fse 2.0 trains nothing, so SIF and uSIF tables come from the offline recipe.
Every task only encodes, like the compared models, which are frozen too. A model with the
``external`` tokenizer gets its texts tokenized as the 1.1.0 tables were (NLTK
``word_tokenize``, lowercased, punctuation dropped, hyphenated words split); any other model
tokenizes the raw text itself. The compared models use their own tokenizers. The results of
fse 1.1.0 in results.jsonl were measured by this script before 2.0 and stay as they were.

For every task the script records the main score, the wall time of the ``encode`` calls (for fse
including tokenization), the number of encoded texts and the load average before the task.
Encoding speed depends on the machine and its load; compare it only within one run.

Setup, from the repository root (the benchmark dependencies stay out of the project)::

    uv venv -p 3.12 .venv-quality
    uv pip install -p .venv-quality/bin/python -e . -r benchmarks/quality/requirements.txt

Examples::

    .venv-quality/bin/python benchmarks/quality/run_mteb.py run --models fse/uSIF-paranmt-300 --tasks STSBenchmark
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
from collections import defaultdict, deque
from contextlib import closing
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mteb
import numpy as np
from artifact_manifest import verified_source
from model2vec_adapter import NativeModel2VecEncoder
from mteb.cache import ResultCache
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from protocol import compatible_results, measurement_identity, source_identity
from resume import pending_tasks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quality_coordination import paused, task_reservation  # isort: skip
from bench_pretrained import COMMITS  # isort: skip
from compare_inputs import identity  # isort: skip
from compare_models import MODELS  # isort: skip

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.jsonl"  # the default of --results
CACHE = Path(os.environ.get("FSE_QUALITY_CACHE", Path.home() / ".cache" / "fse-quality"))

BENCHMARKS = ["MTEB(eng, v2)", "NanoBEIR"]
# A selection of both that runs in about 8 minutes per model on an M1 Pro, for baselines:
# every task type of MTEB(eng, v2) without its slowest tasks, and all of NanoBEIR.
BASELINE_TASKS = [
    "BIOSSES", "SICK-R", "STS12", "STS13", "STS14", "STS15", "STS17", "STS22.v2", "STSBenchmark",
    "SprintDuplicateQuestions", "TwitterSemEval2015", "TwitterURLCorpus",
    "Banking77Classification", "MTOPDomainClassification", "MassiveScenarioClassification",
    "MedrxivClusteringS2S.v2", "MedrxivClusteringP2P.v2", "BiorxivClusteringP2P.v2",
    "ArXivHierarchicalClusteringS2S", "AskUbuntuDupQuestions", "ArguAna", "SummEvalSummarization.v2",
    "NanoArguAnaRetrieval", "NanoClimateFeverRetrieval", "NanoDBPediaRetrieval", "NanoFEVERRetrieval",
    "NanoFiQA2018Retrieval", "NanoHotpotQARetrieval", "NanoMSMARCORetrieval", "NanoNFCorpusRetrieval",
    "NanoNQRetrieval", "NanoQuoraRetrieval", "NanoSCIDOCSRetrieval", "NanoSciFactRetrieval",
    "NanoTouche2020Retrieval",
]  # fmt: skip
# Registered in mteb with pinned revisions.
OTHER_MODELS = [
    *sorted(COMMITS),
    *[f"model2vec/{name}" for name in sorted(COMMITS) if name.startswith("minishlab/")],
    "sentence-transformers/all-MiniLM-L6-v2",
]


logger = logging.getLogger("run_mteb")

_NOT_PUNCTUATION = re.compile(".*[A-Za-z0-9].*")


def _prep_token(token: str) -> list[str]:
    t = token.lower().strip("';.:()").strip('"')
    t = "not" if t == "n't" else t
    return re.split(r"[-]", t)


def tokenize(text: str) -> list[str]:
    """Tokenize like notebooks/STS-Benchmarks.ipynb on the branch 1.x."""
    from nltk import word_tokenize

    tokens: list[str] = []
    for token in word_tokenize(text):
        if _NOT_PUNCTUATION.match(token):
            tokens.extend(_prep_token(token))
    return tokens


def fse_models(folder: Path) -> list[str]:
    """The names of the finished fse models in ``folder``; raises if there are none, so a
    run of the fse group never ends without measuring anything."""
    names = [f"fse/{p.stem}" for p in sorted(folder.glob("*.safetensors"))]
    if not names:
        raise SystemExit(
            f"no fse models (*.safetensors) in {folder}: put finished models there or pass "
            "--fse-models"
        )
    return names


class FseEncoder(AbsEncoder):
    """Encode with a finished fse model (``fse.Encoder.load``)."""

    folder = CACHE / "models"  # set by --fse-models
    workers = 2
    batch_tasks = 8
    coordination: Path | None = None

    def __init__(self, model_name: str, revision: str | None = None, **kwargs: Any) -> None:
        import fse

        path = self.folder / f"{model_name.removeprefix('fse/')}.safetensors"
        self.model = fse.Encoder.load(path)
        self.provenance = verified_source(path, self.model)
        logger.info("loaded %s from %s", self.model, path)

    def encode(
        self,
        inputs: Any,
        *,
        task_metadata: Any,
        hf_split: str,
        hf_subset: str,
        prompt_type: Any = None,
        **kwargs: Any,
    ) -> np.ndarray:
        texts = [text for batch in inputs for text in batch["text"]]
        if self.model.tokenizer == "external":
            texts = [tokenize(text) for text in texts]
        return self.model.encode(texts, workers=self.workers)


def vector_license(name: str) -> str:
    """The source vectors' terms, independent of fse's code license.

    Legacy converted files may have no provenance metadata. Unrecognised sources,
    including paranmt and paragram, have unknown terms, never an assumed license.
    """
    for source, license_id in (
        ("glove-", "pddl"),
        ("word2vec-google-news-", "apache-2.0"),
        ("fasttext-", "cc-by-sa-3.0"),
        ("potion-", "mit"),
    ):
        if source in name.lower():
            return license_id
    return "unknown"


def fse_meta(name: str) -> ModelMeta:
    encoder = FseEncoder(name).model
    return ModelMeta(
        loader=FseEncoder,
        name=name,
        revision=version("fse"),
        release_date=None,
        languages=["eng-Latn"],
        n_parameters=None,
        memory_usage_mb=None,
        max_tokens=encoder._content.tokenizer.max_tokens or np.inf,
        embed_dim=encoder.output_dim,
        license=vector_license(name),
        open_weights=True,
        public_training_code="https://github.com/oborchers/Fast_Sentence_Embeddings",
        public_training_data=None,
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


def metadata(name: str) -> ModelMeta:
    if name.startswith("fse/"):
        return fse_meta(name)
    if name.startswith("model2vec/"):
        repo = name.removeprefix("model2vec/")
        return mteb.get_model_meta(repo).model_copy(
            update={
                "name": name,
                "revision": COMMITS[repo],
                "loader": NativeModel2VecEncoder,
            }
        )
    return mteb.get_model_meta(name).model_copy(update={"revision": MODELS[name]})


def task_batches(model, tasks, cache):
    pending = deque(tasks)
    while pending:
        with task_reservation(FseEncoder.coordination):
            for _ in range(FseEncoder.batch_tasks):
                if not pending or (FseEncoder.coordination and paused(FseEncoder.coordination)):
                    break
                task = pending.popleft()
                result = mteb.evaluate(
                    model, [task], cache=cache, overwrite_strategy="always", show_progress_bar=False
                )
                yield task, result


def run(
    models: list[str],
    task_names: list[str] | None,
    results: Path = RESULTS,
    resume: bool = False,
    defer_mind: bool = False,
) -> None:
    from nltk import download

    download("punkt_tab", quiet=True)
    tasks = tasks_for(task_names)
    benchmark_of = {t.metadata.name: b for b in BENCHMARKS for t in mteb.get_benchmark(b).tasks}
    cache = ResultCache(CACHE / "mteb")
    environment = {**identity(), **measurement_identity(tasks_for(None))}
    for name in models:
        with task_reservation(FseEncoder.coordination):
            meta = metadata(name)
            model = meta.load_model(device="cpu")
        specification = {
            "model": name,
            "revision": meta.revision,
            "loader": f"{type(model).__module__}.{type(model).__name__}",
            "encoder_fingerprint": getattr(getattr(model, "model", None), "fingerprint", None),
            "workers": FseEncoder.workers,
            "device": "cpu",
            "versions": environment["versions"],
            **{
                key: environment[key]
                for key in ("method_fingerprint", "runtime_fingerprint", "suite_fingerprint")
            },
        }
        specification.update(getattr(model, "provenance", source_identity(name)))
        specification.setdefault("float32_parent_fingerprint", None)
        pending = pending_tasks(results, specification, tasks) if resume else tasks
        logger.info("%s: %d pending tasks of %d", name, len(pending), len(tasks))
        if defer_mind:
            pending = defer_mind_task(pending, environment, specification, results)
        timer = EncodeTimer(model)
        with closing(task_batches(model, pending, cache)) as batches:
            for task_to_run, result in batches:
                for task_result in result.task_results:
                    task = task_result.task_name
                    with results.open("a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    **environment,
                                    "model": name,
                                    **getattr(model, "provenance", source_identity(name)),
                                    "loader": f"{type(model).__module__}.{type(model).__name__}",
                                    "workers": FseEncoder.workers,
                                    "device": "cpu",
                                    "task_metadata": task_to_run.metadata.model_dump(mode="json"),
                                    "encoder_fingerprint": getattr(
                                        getattr(model, "model", None), "fingerprint", None
                                    ),
                                    "revision": meta.revision,
                                    "encoder_description": str(
                                        getattr(model, "model", type(model).__name__)
                                    ),
                                    "benchmark": benchmark_of[task],
                                    "task": task,
                                    "type": mteb.get_task(task).metadata.type,
                                    "main_score": task_result.get_score(),
                                    "status": "completed",
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


def defer_mind_task(tasks, environment, specification, results):
    remaining = []
    for task in tasks:
        if task.metadata.name != "MindSmallReranking":
            remaining.append(task)
            continue
        row = {
            **environment,
            **specification,
            **source_identity(specification["model"]),
            "task": task.metadata.name,
            "task_metadata": task.metadata.model_dump(mode="json"),
            "status": "deferred",
            "main_score": None,
            "reason": "deferred: needs more than 16 GiB RAM on the M1",
            "benchmark": "MTEB(eng, v2)",
            "mteb": version("mteb"),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with results.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        logger.info("MindSmallReranking: %s", row["reason"])
    return remaining


def summarize(results: Path = RESULTS) -> None:
    """Print a Markdown table of the latest result per model and task."""
    rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
    by_model = compatible_results(rows, tasks_for(None))
    eng = {t.metadata.name for t in mteb.get_benchmark("MTEB(eng, v2)").tasks}
    types = sorted({t.metadata.type for t in mteb.get_benchmark("MTEB(eng, v2)").tasks})
    print(
        "| Model | Mean (Task) | Mean (TaskType) | "
        + " | ".join(types)
        + " | NanoBEIR (13 tasks required) | coverage | texts/s |"
    )
    print("|---" * (len(types) + 6) + "|")
    for key, rows in sorted(by_model.items()):
        config = json.loads(key)
        model = (
            f"{config['model']} ({config.get('method_fingerprint') or config.get('legacy_commit')}, "
            f"{config.get('runtime_fingerprint')}, {config.get('workers')} workers)"
        )
        main = [r for r in rows if r["task"] in eng]
        nano = [r["main_score"] for r in rows if r["benchmark"] == "NanoBEIR"]
        per_type = {ty: [r["main_score"] for r in main if r["type"] == ty] for ty in types}
        complete = len(main) == len(eng)
        mean_task = statistics.mean(r["main_score"] for r in main) if complete else float("nan")
        mean_type = (
            statistics.mean(statistics.mean(v) for v in per_type.values())
            if complete
            else float("nan")
        )
        texts = sum(r.get("texts", 0) for r in rows)
        seconds = sum(r.get("seconds", 0.0) for r in rows)
        cells = [f"{100 * statistics.mean(v):.2f}" if v else "" for v in per_type.values()]
        print(
            f"| {model} | {100 * mean_task:.2f} | {100 * mean_type:.2f} | "
            + " | ".join(cells)
            + f" | {100 * statistics.mean(nano) if len(nano) == 13 else float('nan'):.2f} | {len(main)}/41 + {len(nano)}/13 | {texts / seconds:,.0f} |"
            if nano and seconds
            else f"| {model} | {100 * mean_task:.2f} | {100 * mean_type:.2f} | "
            + " | ".join(cells)
            + f" | | {len(main)}/41 + {len(nano)}/13 | |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument(
        "--models", nargs="+", required=True, help="model names, 'fse', 'others' or 'all'"
    )
    run_parser.add_argument(
        "--tasks", nargs="+", help="restrict to these task names, or 'baseline'"
    )
    run_parser.add_argument("--results", type=Path, default=RESULTS)
    run_parser.add_argument(
        "--fse-models", type=Path, default=FseEncoder.folder, help="the finished fse models"
    )
    run_parser.add_argument(
        "--defer-mind", action="store_true", help="record the authorized RAM deferral"
    )
    run_parser.add_argument("--resume", action="store_true", help="reuse matching completed tasks")
    run_parser.add_argument("--workers", type=int, default=2)
    run_parser.add_argument("--batch-tasks", type=int, default=8)
    run_parser.add_argument("--coordination", type=Path, help="shared scratchpad lock directory")
    summary = sub.add_parser("summarize")
    summary.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if args.command == "summarize":
        summarize(args.results)
        return
    if args.workers < 1 or args.batch_tasks < 1:
        parser.error("workers and batch-tasks must be positive")
    import torch

    torch.set_num_threads(args.workers)
    os.environ["RAYON_NUM_THREADS"] = str(args.workers)
    FseEncoder.batch_tasks = args.batch_tasks
    FseEncoder.workers = args.workers
    FseEncoder.coordination = args.coordination
    FseEncoder.folder = args.fse_models
    ours = fse_models(args.fse_models) if {"fse", "all"} & set(args.models) else []
    groups = {"fse": ours, "others": OTHER_MODELS, "all": ours + OTHER_MODELS}
    models = [m for name in args.models for m in groups.get(name, [name])]
    tasks = BASELINE_TASKS if args.tasks == ["baseline"] else args.tasks
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=args.workers):
        run(models, tasks, args.results, args.resume, args.defer_mind)


if __name__ == "__main__":
    main()
