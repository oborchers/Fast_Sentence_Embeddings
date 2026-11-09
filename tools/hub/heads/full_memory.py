"""No-run admission for the full-width Bekko comparison on this macOS host."""

from __future__ import annotations

import importlib.metadata
import platform
import re
import shutil
import subprocess
from pathlib import Path

from inputs import load, read

PROTOCOL = Path(__file__).with_name("full-bekko.json")


def settings(parent):
    original = load(parent / "config.json")
    full = read(PROTOCOL)
    if full["schema"] != "civil-comments-full-bekko@1":
        raise ValueError("unsupported full Bekko protocol")
    if importlib.metadata.version("scikit-learn") != full["scikit_learn_version"]:
        raise ValueError("full float32 L-BFGS requires the reviewed sklearn 1.9.1 implementation")
    if read(parent / "selection.json")["full_train"]["rows"] != full["train_rows"]:
        raise ValueError("full training row count differs from the reviewed input")
    return {**original, **full}, full


def admission(parent, config):
    if platform.system() != "Darwin":
        raise RuntimeError("this memory admission requires macOS")
    total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    pressure = subprocess.check_output(["memory_pressure", "-Q"], text=True)
    found = re.search(r"System-wide memory free percentage:\s*(\d+)%", pressure)
    if found is None:
        raise ValueError("macOS memory availability could not be read")
    percent = int(found[1])
    available = total * percent // 100
    matrix = config["train_rows"] * config["dimension"] * 4
    peak = matrix + config["memory_reserve_bytes"]
    disk = shutil.disk_usage(parent).free
    reasons = []
    if peak > config["maximum_plan_bytes"] or peak > config["maximum_rss_bytes"]:
        reasons.append("estimated peak exceeds the configured plan or RSS limit")
    if available < peak:
        reasons.append("macOS reported availability is below the conservative working-set plan")
    if disk < matrix + config["minimum_free_bytes"]:
        reasons.append("working matrix would breach the disk reserve")
    if not (parent / "test-hold.json").is_file() or not read(parent / "test-hold.json")["active"]:
        reasons.append("the official test hold must remain active")
    return {
        "admitted": not reasons,
        "reasons": reasons,
        "train_rows": config["train_rows"],
        "dimension": config["dimension"],
        "working_dtype": "float32",
        "matrix_bytes": matrix,
        "estimated_peak_bytes": peak,
        "maximum_plan_bytes": config["maximum_plan_bytes"],
        "maximum_rss_bytes": config["maximum_rss_bytes"],
        "physical_memory_bytes": total,
        "reported_free_percent": percent,
        "estimated_available_bytes": available,
        "free_disk_bytes": disk,
        "swap": subprocess.check_output(["sysctl", "vm.swapusage"], text=True).strip(),
        "phases": {
            "encode": "stream parquet, map/write/flush/unmap one output batch, retain no corpus texts",
            "scale": "one float32 memmap, StandardScaler partial_fit and transform in batches",
            "fit": "one read-only float32 memmap, one OVR fit, no scaler or matrix copies",
            "validate": "unmap training matrix before loading encoder and validation texts",
        },
        "dtype_evidence": {
            "sklearn_version": importlib.metadata.version("scikit-learn"),
            "input": "LogisticRegression.fit validates float64 or float32",
            "loss": "LinearModelLoss.weight_intercept_raw casts weights to X.dtype",
            "gradient": "X.T @ grad_pointwise uses the same float32 training matrix",
        },
    }
