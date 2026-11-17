"""Same-build paired runtime screens on an explicitly approved Linux machine."""

from __future__ import annotations


def validate(cell):
    _single_operation(cell)
    if type(cell.get("runtime_phased_memory", False)) is not bool:
        raise ValueError("runtime_phased_memory must be boolean")
    if type(cell.get("runtime_paired", False)) is not bool:
        raise ValueError("runtime_paired must be boolean")
    _memory_profile(cell)
    if not cell.get("runtime_paired", False):
        return
    if (
        cell["stage"] not in ("encode-allocate", "encode-memory", "encode-calls")
        or cell["workers"] != 8
    ):
        raise ValueError("paired runtime screens require encode and eight workers")
    if cell["stage"] != "encode-calls" and cell["iterations"] != 1:
        raise ValueError("paired runtime screens require one complete encode per observation")
    if (
        cell["decision_gate"]
        or cell["table_dtype"] != "float32"
        or cell["output_dtype"] != "float32"
    ):
        raise ValueError("paired runtime screens require float32 without an optimization gate")


def _memory_profile(cell):
    if not cell.get("runtime_phased_memory", False):
        return
    if not cell.get("runtime_paired", False) and not (
        cell["stage"] == "encode-memory"
        and cell.get("single_operation", False)
        and cell["runtime_variant"] == "baseline"
        and not cell["runtime_timeline"]
        and cell["workers"] == 8
        and cell["table_dtype"] == cell["output_dtype"] == "float32"
        and cell["chunk_bytes"] is None
        and cell["base_doc_ids"] == cell["candidate_doc_ids"] == "none"
    ):
        raise ValueError("phased memory requires a runtime screen or ordinary single encode")
    if cell["profile"] != "standard@1" or cell["candidate_lengths"] != "none":
        raise ValueError(
            "phased runtime memory requires the standard profile without length reports"
        )


def _single_operation(cell):
    if type(cell.get("single_operation", False)) is not bool:
        raise ValueError("single_operation must be boolean")
    if not cell.get("single_operation", False):
        return
    if (
        cell["stage"] not in ("encode-memory", "text-join")
        or cell["iterations"] != 1
        or cell["max_calibration_steps"] != 1
        or cell["minimum_seconds"] != 2
        or cell["target_seconds"] != 2
        or cell["max_target_seconds"] != 2
        or cell["runtime_paired"]
    ):
        raise ValueError(
            "single_operation requires ordinary encode-memory or text-join, one iteration, "
            "one frozen two-second calibration tier and no runtime-paired substitution"
        )


def configure(job, cell, name, control):
    if cell.get("runtime_paired", False) and (control or name == "base"):
        job.update(runtime_variant="baseline", runtime_baseline=None)
        if cell["stage"] != "encode-calls":
            job["stage"] = "encode-allocate"


def compare(cell, rows):
    if not cell.get("runtime_paired", False):
        return None
    if cell.get("runtime_timeline", False) and any(
        not (row.get("runtime_evidence") or {}).get("valid", False) for row in rows
    ):
        return {"valid": False, "reason": "paired runtime diagnostic incomplete"}
    if cell["runtime_variant"] != "window-4x":
        return {
            "valid": True,
            "preliminary": True,
            "variant": cell["runtime_variant"],
            "stage": cell["stage"],
        }
    tasks = []
    for row in rows:
        evidence = row.get("runtime_evidence")
        if not evidence or not evidence.get("valid"):
            return {"valid": False, "reason": "paired window diagnostic incomplete"}
        if not evidence["summary"].get("task_layout_consistent", True):
            return {"valid": False, "reason": "window probe task layouts differ across calls"}
        tasks.append({key: evidence["summary"][key] for key in ("task_rows", "task_bytes")})
    equal = bool(tasks) and all(task == tasks[0] for task in tasks)
    return {
        "valid": equal,
        "task_equality": equal,
        "preliminary": True,
        "reason": None if equal else "paired window task sizes differ",
    }


def fixed_size(cell, rows, functional):
    if cell["stage"] == "encode-calls":
        if not cell.get("runtime_fixed_iterations", False):
            return False
        if not functional and any(row["seconds"] < 2 for row in rows):
            raise ValueError("frozen short-call block is below two seconds; no automatic retry")
        return True
    if not (cell.get("runtime_paired", False) or cell.get("single_operation", False)):
        return False
    if not functional and any(row["seconds_per_iteration"] < 2 for row in rows):
        raise ValueError(
            "fixed runtime workload is below two seconds per encode; "
            "do not substitute repeated short calls or retry automatically"
        )
    return True


def pair_observations(cell, rows):
    if not (cell.get("runtime_paired") or cell.get("single_operation")):
        return {}
    return {
        "runtime_pair_observations": [
            {
                "pair": number,
                "first_pair": number == 0,
                "reduced_warmup": cell["warmup_calls"] == 1 and cell["warmup_seconds"] <= 6,
                "arms": [_pair_arm(row, cell) for row in rows if row["pair"] == number],
            }
            for number in range(cell["pairs"])
        ],
        "runtime_pair_scope": "all pairs retained; inspect first pair separately after single-block warmup",
    }


def _pair_arm(row, cell):
    calls = row["operation_repetitions"] * (cell.get("call_repeats") or 1)
    return {
        "arm": row["arm"],
        "seconds": row["seconds"],
        "process_cpu_seconds": row["process_cpu_seconds"],
        "mean_seconds_per_call": row["seconds"] / calls,
    }
