"""CUDA quality planning, sealed model verdicts and deadline-bounded task children."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from importlib.metadata import version
from pathlib import Path

from _harness.budget import select_billing
from _harness.ledger import Ledger, watching
from _harness.records import atomic, document, sha256
from _measurement import write_environment
from _measurement_config import CONFIG, validate_cell
from _speed_gpu_study import probe_verdicts, verdict

MODELS = ("sentence-transformers/all-mpnet-base-v2", "Qwen/Qwen3-Embedding-0.6B")


def select(spec, phase, *, unpriced=False):
    spec = select_billing(spec, "gpu_quality_billing", allow_unpriced=unpriced)
    spec["cells"] = [
        c for c in spec["cells"] if c.get("functional_only", False) == (phase == "probe")
    ]
    if {c["model"] for c in spec["cells"]} != set(MODELS) or len(spec["cells"]) != 2:
        raise ValueError("GPU quality requires exactly mpnet and Qwen3 in each phase")
    tasks = (
        ["MTEB(eng, v2)/STSBenchmark", "NanoBEIR/NanoQuoraRetrieval"]
        if phase == "probe"
        else ["MTEB(eng, v2)", "NanoBEIR"]
    )
    for cell in spec["cells"]:
        validate_cell(cell)
        if cell.get("tasks") != tasks or cell["id"].startswith("smoke-ref-") != (phase == "probe"):
            raise ValueError("GPU quality requires its full declared tasks and smoke prompt checks")
        if (
            cell.get("device") != "cuda"
            or cell.get("torch_dtype") != "float32"
            or not cell.get("mteb_loader")
            or type(cell.get("workers")) is not int
            or cell["workers"] != 1
        ):
            raise ValueError("GPU quality requires the float32 CUDA MTEB loader in one process")
    return spec


def identity(args, spec):
    return {
        "allocation_root": str(args.allocation_root.resolve()),
        "allocated_at": args.allocated_at,
        "quality_sha256": sha256(args.input),
        "lock_sha256": sha256(CONFIG / "speed-gpu.lock.txt"),
        "models": {m: spec["models"][m] for m in MODELS},
    }


def environment():
    import torch
    from _speed_gpu import observation, require_cuda

    require_cuda(torch)
    for line in (CONFIG / "speed-gpu.lock.txt").read_text().splitlines():
        if "==" in line and not line.startswith((" ", "#")):
            name, pin = line.split()[0].split("==")
            if version(name) != pin:
                raise RuntimeError("GPU quality package differs from lock: " + name)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    return {"torch_version": torch.__version__, **observation(torch)}


def child(path):
    """One full, unsampled MTEB task, through the central quality entry."""
    from _quality.mteb import evaluate

    job = document(path)
    cell = job["cell"]
    validate_cell(cell)
    # Infrastructure failures are fatal, never converted into a model rejection.
    gpu = environment()
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        if not nltk.download("punkt_tab", quiet=True):
            raise RuntimeError("pinned NLTK tokenizer resources unavailable") from None
    context = {"results": Path(job["results"]), "fits": {}}
    try:
        row = evaluate(cell, context)
    except Exception as error:
        row = {"status": "failed", "reason": repr(error)}
    import torch
    from _speed_gpu import observation

    gpu.update(observation(torch))
    atomic(
        Path(job["output"]), {**row, "gpu": gpu, "class": "correctness", "speed_evidence": False}
    )
    return 0


def launch(cell, args, ledger):
    """A separate process group makes the allocation work deadline enforceable."""
    output = args.results / (cell["id"] + "-receipt.json")
    job = args.work / (cell["id"] + "-job.json")
    atomic(job, {"cell": cell, "results": str(args.results), "output": str(output)})
    command = [sys.executable, str(CONFIG.parent / "quality.py"), "--gpu-task", str(job)]
    with (args.results / (cell["id"] + ".log")).open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
        try:
            process.wait(timeout=max(0.001, ledger.budget.remaining()))
        except subprocess.TimeoutExpired:
            ledger.stopping = True
            return {"status": "aborted", "reason": "GPU quality hard work deadline reached"}
        finally:
            # Also remove surviving descendants after a normal leader exit.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    if process.returncode != 0:
        raise RuntimeError(
            f"GPU quality infrastructure child exited {process.returncode}; see task log"
        )
    return document(output)


def run(args, spec):
    if args.allocation_root is None:
        raise ValueError("GPU quality requires --allocation-root")
    context = {"allocation_root": args.allocation_root, "allocated_at": args.allocated_at}
    expected = identity(args, spec)
    allowed = {} if args.gpu_phase == "probe" else probe_verdicts(context, spec, expected)
    args.results.mkdir(parents=True, exist_ok=False)
    args.work.mkdir(parents=True, exist_ok=False)
    ledger = Ledger(args.results, spec, args.allocated_at)
    write_environment(args, spec)
    os.environ["MEASURE_WORK"] = str(args.work)
    model_verdicts = {}
    with watching(ledger, restore=False):
        try:
            process_cells(args, ledger, allowed, model_verdicts)
        except Exception as error:
            from _harness.records import append

            append(args.results / "fatal.jsonl", {"error": repr(error)})
        finally:
            if args.gpu_phase == "probe":
                atomic(
                    args.results / "gpu-verdicts.json",
                    {"identity": expected, "verdicts": model_verdicts},
                )
            ledger.fill_missing("GPU quality preparation failed or work deadline reached")
    return ledger.done()


def record_probe(cell, result, verdicts):
    """Every smoke task must pass. One model rejection leaves the other model eligible."""
    previous = verdicts.setdefault(
        cell["model"], {"verdict": "passed", "reason": None, "tasks": []}
    )
    previous["tasks"].append(cell["task"])
    if result["status"] != "valid":
        previous.update(verdict="failed", reason=result["reason"])
    # A checked rejection completes the probe successfully, as in the throughput probe.
    return {
        **result,
        "task_status": result["status"],
        "status": "valid",
        "verdict": previous["verdict"],
    }


def process_cells(args, ledger, allowed, model_verdicts):
    for cell in ledger.contract["cells"]:
        ledger.current, ledger.stage = cell["id"], "gpu-quality"
        started = time.monotonic()
        admission = verdict(allowed, {"id": cell["model"]})
        if args.gpu_phase == "measure" and admission["verdict"] != "passed":
            ledger.result(cell, started, status="not_started", reason=admission["reason"])
            continue
        if not ledger.admit(cell["estimated_seconds"]):
            ledger.result(cell, started, status="not_started", reason="budget or stop signal")
            continue
        result = launch(cell, args, ledger)
        if args.gpu_phase == "probe":
            result = record_probe(cell, result, model_verdicts)
        ledger.result(cell, started, **result)
