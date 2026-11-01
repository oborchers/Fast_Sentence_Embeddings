"""Fetch pinned MiniLM weights and audit POTION conversion on the complete WikiText corpus.

Run with nice -n 10. Uses the heavy-job lock, never the timing lock, because its
outputs are numerical-quality diagnostics rather than speed measurements.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from compare_inputs import corpus_identity, identity, texts
from compare_models import MODELS, artifact_name
from compare_numerics import compare
from quality_coordination import task_reservation


def diagnose(models: Path, results: Path) -> None:
    from huggingface_hub import snapshot_download
    from model2vec import StaticModel

    from fse import Encoder

    repo = "minishlab/potion-base-8M"
    sentences = texts("wikitext")
    provenance = {
        **identity(),
        **corpus_identity("wikitext", sentences),
        "model": repo,
        "revision": MODELS[repo],
        "corpus": "wikitext",
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
    }
    baseline = Encoder.load(models / artifact_name(repo, "float32"))
    reference = baseline.encode(sentences, workers=1)
    folder = snapshot_download(repo, revision=MODELS[repo], local_files_only=True)
    native = StaticModel.from_pretrained(folder).encode(sentences, use_multiprocessing=False)
    comparisons = [("model2vec-native", "float32", native, None)]
    for dtype in ("float16", "int8"):
        candidate = Encoder.load(models / artifact_name(repo, dtype))
        comparisons.append(
            ("fse", dtype, candidate.encode(sentences, workers=1), candidate.fingerprint)
        )
    for kind, dtype, vectors, fingerprint in comparisons:
        before = os.getloadavg()[0]
        row = {
            **provenance,
            **compare(reference, vectors),
            "reference_kind": "fse",
            "reference_table_dtype": "float32",
            "reference_fingerprint": baseline.fingerprint,
            "candidate_kind": kind,
            "candidate_table_dtype": dtype,
            "candidate_fingerprint": fingerprint,
            "output_dtype": str(vectors.dtype),
            "load_before": before,
            "load_after": os.getloadavg()[0],
            "time_utc": datetime.now(UTC).isoformat(),
        }
        with results.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(
            json.dumps(
                {
                    "candidate": kind,
                    "table_dtype": dtype,
                    "recall_at_10": row["recall_at_k"],
                    "minimum_cosine": row["minimum_cosine"],
                }
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordination", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise RuntimeError("run diagnostics with nice -n 10")
    with task_reservation(args.coordination):
        from huggingface_hub import snapshot_download
        from threadpoolctl import threadpool_limits

        repo = "sentence-transformers/all-MiniLM-L6-v2"
        snapshot_download(
            repo,
            revision=MODELS[repo],
            max_workers=2,
            allow_patterns=["*.json", "*.txt", "*.safetensors", "1_Pooling/*"],
        )
        with threadpool_limits(limits=2):
            diagnose(args.models, args.results)


if __name__ == "__main__":
    main()
