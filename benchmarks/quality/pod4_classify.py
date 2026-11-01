"""Full held-out accuracy and long-call inference on balanced held-out repetitions."""

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from compare_artifacts import file_digest
from compare_models import MODELS
from compare_parallelism import child_environment
from folded_classifier_prepare import REPO, prepare, prepare_mlp
from pod_artifacts import emit
from pod_pairs import append, series
from prepare_fineweb import digest
from prepare_native_classifier import inputs
from prepare_setfit import SOURCES


def expanded_input(folder, texts, labels, requested):
    if not texts or len(texts) != len(labels):
        raise ValueError("held-out texts and labels must have equal nonzero lengths")
    count = math.ceil(requested / len(texts)) * len(texts)
    if count > 10_000_000:
        raise MemoryError("classification input exceeds ten million complete held-out copies")
    path = folder / "pipeline-inputs" / f"timing-{count}-seed0.jsonl"
    receipt = path.with_suffix(".identity.json")
    if receipt.exists():
        return receipt, json.loads(receipt.read_text())
    import shutil

    expected = count * (sum(len(text.encode("utf-8")) + 40 for text in texts) / len(texts))
    if shutil.disk_usage(folder).free < expected + 25 * 1024**3:
        raise OSError("timing repetitions would leave less than 25 GiB free")
    from pod4_memory import admit

    admit(count * 24, "classification repetition index workspace")
    indices = np.arange(count, dtype=np.int64) % len(texts)
    np.random.default_rng(0).shuffle(indices)
    with path.open("x", encoding="utf-8") as stream:
        for index in indices:
            stream.write(
                json.dumps({"text": texts[int(index)], "label": int(labels[int(index)])}) + "\n"
            )
    record = {
        "file": str(path.relative_to(folder)),
        "checksum": file_digest(path),
        "rows": count,
        "input_digest": digest(texts[int(index)] for index in indices),
        "label": "balanced repetitions of the complete held-out split, shuffled for inference only",
        "seed": 0,
        "numpy_version": np.__version__,
        "source_rows": len(texts),
        "repetition_factor": count // len(texts),
        "accuracy_scope": "original complete held-out accuracy remains in the training manifest",
    }
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    return receipt, record


def command(folder, workers, kind, receipt, passes=3):
    return [
        sys.executable,
        str(Path(__file__).with_name("pod_folded_child.py")),
        "--folder",
        str(folder),
        "--workers",
        str(workers),
        "--kind",
        kind,
        "--evaluation-record",
        str(receipt),
        "--minimum-seconds",
        "2",
        "--passes",
        str(passes),
    ]


def calibrate(args, folder, texts, labels, workers):
    requested = max(len(texts), 100_000)
    while True:
        receipt, record = expanded_input(folder, texts, labels, requested)
        done = subprocess.run(
            command(folder, workers, "folded", receipt, 1),
            capture_output=True,
            text=True,
            check=False,
            env=child_environment(workers),
            timeout=max(0.01, args.deadline - time.monotonic()),
        )
        if done.returncode:
            append(
                args.output / "calibration.jsonl",
                {"status": "failed", "stderr": done.stderr[-12000:]},
            )
            raise RuntimeError("folded-classifier calibration failed")
        row = json.loads(done.stdout.strip().splitlines()[-1])
        append(
            args.output / "calibration.jsonl", {"scope": "input sizing only", "measurement": row}
        )
        smallest = min(row["first_predict_s"], *row["warm_seconds"])
        if smallest >= 2.5:
            return receipt, record
        if not math.isfinite(smallest) or smallest <= 0:
            raise ValueError("nonpositive classification calibration duration")
        requested = math.ceil(record["rows"] * 2.5 / smallest * 1.1)


def timed(args, folder, texts, labels, suite, name):
    failed, expected = [], []
    for workers in (1, 32):
        receipt, _ = calibrate(args, folder, texts, labels, workers)
        base = {label: command(folder, workers, "folded", receipt) for label in ("fse-a", "fse-b")}
        pairs = {
            "same-linear-head": {
                "encode-matmul": command(folder, workers, "encode-matmul", receipt),
                "model2vec-same-linear": command(folder, workers, "model2vec-linear", receipt),
            },
            "mlp": {"model2vec-mlp": command(folder, workers, "mlp", receipt)},
        }
        for label, references in pairs.items():
            cell = f"folded/{suite}/{name}/{workers}/{label}"
            expected.append(cell)
            done = series(
                {**base, **references},
                workers,
                args.output,
                args.deadline,
                cell,
                abort_on_timeout=True,
            )
            if not done["complete"]:
                failed.append(cell)
    from pod_folded_summary import publish_summary

    result = publish_summary(args.output)
    from pod4_receipts import check_cells

    failed.extend(check_cells(expected, result))
    if failed:
        raise RuntimeError("classification has short, failed or noisy cells: " + ", ".join(failed))


def dataset(args, suite, name, data, provenance):
    from pod4_memory import admit

    training = len(data[0]) * 256 * 4 * 8 + len(data[2]) * 512 * 4 * 4 + 4 * 1024**3
    admit(
        training, "classifier fitting features, token indices, weights and evaluation intermediates"
    )
    folder = args.output / (suite + "-" + name)
    manifest = folder / "manifest.json"
    record = (
        json.loads(manifest.read_text())
        if manifest.exists()
        else prepare(suite, name, data, provenance, folder)
    )
    if "mlp" not in record:
        record = prepare_mlp(data, provenance, folder, record)
    append(args.output / "classification-quality.jsonl", {**record, **args.hardware})
    emit("pod4-prepared-" + suite + "-" + name, record)
    timed(args, folder, data[2], data[3], suite, name)


def run(args):
    import torch
    from huggingface_hub import snapshot_download

    snapshot_download(
        REPO,
        revision=MODELS[REPO],
        allow_patterns=["*.json", "*.safetensors", "README.md"],
        max_workers=2,
    )
    torch.set_num_threads(16)
    names = [
        r["dataset"].removeprefix("SetFit/") for r in json.loads(SOURCES.read_text())["datasets"]
    ]
    selected = ["ag_news", "subj", *[name for name in names if name not in ("ag_news", "subj")]]
    order = [("setfit", selected), ("fineweb", None)]
    expected = [
        f"folded/{suite}/{name}/{workers}/{label}"
        for suite, name in [("setfit", name) for name in selected] + [("fineweb", "fineweb")]
        for workers in (1, 32)
        for label in ("same-linear-head", "mlp")
    ]
    (args.output / "classification-plan.json").write_text(
        json.dumps({"expected_cells": expected}, indent=2) + "\n"
    )
    failed = []
    from pod4_memory import admit

    for suite, selection in order:
        for name, data, provenance in inputs(suite, args.cache / suite, selection, admission=admit):
            if time.monotonic() >= args.deadline:
                raise TimeoutError("classification stage deadline reached")
            try:
                dataset(args, suite, name, data, provenance)
            except (TimeoutError, subprocess.TimeoutExpired):
                raise
            except Exception as error:
                failed.append(suite + "/" + name)
                append(
                    args.output / "invocations.jsonl",
                    {"suite": suite, "dataset": name, "status": "failed", "error": repr(error)},
                )
            del data
    from pod4_receipts import check_cells
    from pod_folded_summary import publish_summary

    failed.extend(check_cells(expected, publish_summary(args.output)))
    if failed:
        raise RuntimeError("classification datasets failed: " + ", ".join(failed))
