"""One unmodified, preloaded MIND task, eleven sequential float32 models."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from importlib.metadata import distribution, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pod_artifacts import emit, environment, export_jsonl  # isort: skip


def mteb_files():
    from compare_artifacts import file_digest

    package = distribution("mteb")
    names = ("mteb/evaluate.py", "mteb/tasks/reranking/eng/mind_small_reranking.py")
    return {name: file_digest(Path(package.locate_file(name))) for name in names}


def append(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")


def evaluate_one(name, task, shared_seconds, hardware, output, workers):
    import mteb
    from backend_runtime import model_runtime
    from compare_inputs import identity
    from compare_runtime import verify_backend_identity
    from protocol import measurement_identity, source_identity
    from run_mteb import EncodeTimer, metadata

    start = time.monotonic()
    meta = metadata(name)
    model = meta.load_model(device="cpu")
    load_seconds = time.monotonic() - start
    context = {**identity(), **measurement_identity([task])}
    runtime = model_runtime(model, context)
    timer = EncodeTimer(model)
    begin = time.monotonic()
    result = mteb.evaluate(
        model, [task], cache=None, overwrite_strategy="always", show_progress_bar=False, num_proc=1
    )
    elapsed = time.monotonic() - begin
    if not task.data_loaded:
        raise RuntimeError("pinned MTEB unexpectedly unloaded the preloaded task")
    verify_backend_identity(runtime["backend_kind"], runtime)
    encode_seconds = timer.stats.get(task.metadata.name, {}).get("seconds", 0.0)
    row = {
        **context,
        **runtime,
        **source_identity(name),
        **getattr(model, "provenance", {}),
        **hardware,
        "model": name,
        "revision": meta.revision,
        "task": task.metadata.name,
        "task_metadata": task.metadata.model_dump(mode="json"),
        "benchmark": "MTEB(eng, v2)",
        "status": "completed",
        "main_score": result.task_results[0].get_score(),
        "encoder_fingerprint": getattr(getattr(model, "model", None), "fingerprint", None),
        "workers": workers,
        "device": "cpu",
        "mteb": version("mteb"),
        "timing": {
            "shared_data_load_seconds": shared_seconds,
            "incremental_data_load_seconds": 0,
            "model_load_seconds": load_seconds,
            "encode_seconds": encode_seconds,
            "evaluation_wall_seconds": elapsed,
            "evaluation_nonencode_seconds": max(0.0, elapsed - encode_seconds),
        },
        "timing_definition": "evaluation_nonencode includes scoring and all evaluate overhead outside model.encode",
    }
    append(output / "mind-results.jsonl", row)
    append(
        output / "invocations.jsonl",
        {
            "model": name,
            "status": "completed",
            "seconds": time.monotonic() - start,
            "timing": row["timing"],
            **hardware,
        },
    )
    export_jsonl(
        "mind-results-" + str(len(list((output).glob("model-done-*")))),
        output / "mind-results.jsonl",
    )
    (output / ("model-done-" + name.replace("/", "_"))).touch()
    del timer, model, result
    gc.collect()


def run(models, cache, output, workers):
    import mteb
    from pod_storage import require_free, snapshot
    from run_mteb import FseEncoder

    if version("mteb") != "2.26.25":
        raise ValueError("MIND requires pinned, unmodified mteb 2.26.25")
    FseEncoder.folder = models
    FseEncoder.workers = workers
    FseEncoder.mmap = True
    hardware = environment()
    import torch

    torch.set_num_threads(workers)
    task = mteb.get_task("MindSmallReranking")
    if task.metadata.dataset["revision"] != "227478e3235572039f4f7661840e059f31ef6eb1":
        raise ValueError("MIND task dataset differs from its pinned revision")
    before = mteb_files()
    snapshot(cache, models, output, "before-mind-load")
    require_free(cache, 120, "before loading unmodified MIND")
    begin = time.monotonic()
    try:
        task.load_data(num_proc=1)
    except Exception:
        try:
            snapshot(cache, models, output, "mind-load-failed")
        except OSError as diagnostic_error:
            emit("mind-load-diagnostic-failed", {"errno": diagnostic_error.errno})
        raise
    shared_seconds = time.monotonic() - begin
    snapshot(cache, models, output, "mind-loaded")
    if not task.data_loaded:
        raise RuntimeError("MIND task did not finish loading")
    emit(
        "mind-load",
        {
            "seconds": shared_seconds,
            "variant": "one public load_data, retained by evaluate(data_preloaded=True)",
            "mteb_files": before,
            "task": task.metadata.model_dump(mode="json"),
            **hardware,
        },
    )
    order = json.loads((output / "model-order.json").read_text())
    for name in order:
        try:
            evaluate_one(name, task, shared_seconds, hardware, output, workers)
        except Exception as error:
            append(
                output / "invocations.jsonl",
                {"model": name, "status": "failed", "error": repr(error), **hardware},
            )
            emit("model-error-" + name.replace("/", "_"), {"model": name, "error": repr(error)})
        if mteb_files() != before:
            raise RuntimeError("installed MTEB files changed")
    task.unload_data()
    export_jsonl("mind-results-final", output / "mind-results.jsonl")
    export_jsonl("mind-invocations", output / "invocations.jsonl")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "evaluate"])
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if sys.platform != "linux":
        raise RuntimeError("MIND pod execution requires Linux; local MIND remains deferred")
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        from pod_models import archive_fit_receipts, prepare_mind
        from pod_storage import cleanup_preparation_sources, require_free, snapshot

        snapshot(args.cache, args.models, args.output, "before-preparation")
        require_free(args.cache, 140, "before model preparation")
        order = prepare_mind(args.models, args.cache, args.workers)
        if len(order) != 11:
            raise ValueError("MIND plan must contain all eleven float32 configurations")
        (args.output / "model-order.json").write_text(json.dumps(order))
        emit("fitted-artifacts", archive_fit_receipts(args.models))
        emit("model-order", order)
        snapshot(args.cache, args.models, args.output, "preparation-saved")
        cleanup_preparation_sources(args.cache, args.models, args.output)
        snapshot(args.cache, args.models, args.output, "preparation-cleaned")
    else:
        run(args.models, args.cache, args.output, args.workers)


if __name__ == "__main__":
    main()
