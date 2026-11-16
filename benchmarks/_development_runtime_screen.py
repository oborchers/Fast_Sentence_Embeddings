"""Same-build paired runtime screens on an explicitly approved Linux machine."""

from __future__ import annotations


def validate(cell):
    if type(cell.get("runtime_phased_memory", False)) is not bool:
        raise ValueError("runtime_phased_memory must be boolean")
    if type(cell.get("runtime_paired", False)) is not bool:
        raise ValueError("runtime_paired must be boolean")
    _memory_profile(cell)
    if not cell.get("runtime_paired", False):
        return
    if cell["stage"] not in ("encode-allocate", "encode-memory") or cell["workers"] != 8:
        raise ValueError("paired runtime screens require encode and eight workers")
    if cell["iterations"] != 1:
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
    if not cell.get("runtime_paired", False):
        raise ValueError("runtime_phased_memory requires the paired runtime study")
    if cell["profile"] != "standard@1" or cell["candidate_lengths"] != "none":
        raise ValueError(
            "phased runtime memory requires the standard profile without length reports"
        )


def configure(job, cell, name, control):
    if cell.get("runtime_paired", False) and (control or name == "base"):
        job.update(stage="encode-allocate", runtime_variant="baseline", runtime_baseline=None)


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
        tasks.append({key: evidence["summary"][key] for key in ("task_rows", "task_bytes")})
    equal = bool(tasks) and all(task == tasks[0] for task in tasks)
    return {
        "valid": equal,
        "task_equality": equal,
        "preliminary": True,
        "reason": None if equal else "paired window task sizes differ",
    }


def fixed_size(cell, rows, functional):
    if not cell.get("runtime_paired", False):
        return False
    if not functional and any(row["seconds_per_iteration"] < 2 for row in rows):
        raise ValueError(
            "fixed runtime workload is below two seconds per encode; "
            "do not substitute repeated short calls or retry automatically"
        )
    return True
