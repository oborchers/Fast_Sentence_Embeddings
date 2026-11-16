"""Same-build paired runtime screens on an explicitly approved Linux machine."""

from __future__ import annotations


def validate(cell):
    if type(cell.get("runtime_paired", False)) is not bool:
        raise ValueError("runtime_paired must be boolean")
    if not cell.get("runtime_paired", False):
        return
    if cell["stage"] not in ("encode-allocate", "encode-memory") or cell["workers"] != 8:
        raise ValueError("paired runtime screens require encode and eight workers")
    if (
        cell["decision_gate"]
        or cell["table_dtype"] != "float32"
        or cell["output_dtype"] != "float32"
    ):
        raise ValueError("paired runtime screens require float32 without an optimization gate")


def configure(job, cell, name, control):
    if cell.get("runtime_paired", False) and (control or name == "base"):
        job.update(stage="encode-allocate", runtime_variant="baseline", runtime_baseline=None)


def compare(cell, rows):
    if not cell.get("runtime_paired", False):
        return None
    if cell["runtime_variant"] != "window-4x":
        return {"preliminary": True, "variant": cell["runtime_variant"], "stage": cell["stage"]}
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
