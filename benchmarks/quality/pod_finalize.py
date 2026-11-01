"""Stdlib-only artifact export, including partial JSONL files after a hard timeout."""

import base64
import gzip
import hashlib
import json
import sys
from pathlib import Path


def emit(name, value):
    compressed = gzip.compress(json.dumps(value, ensure_ascii=True).encode(), mtime=0)
    encoded = base64.b64encode(compressed).decode()
    chunks = [encoded[start : start + 160] for start in range(0, len(encoded), 160)]
    print(
        f"DR190_BEGIN {name} {len(compressed)} {hashlib.sha256(compressed).hexdigest()}", flush=True
    )
    for index, chunk in enumerate(chunks):
        print(f"DR190_DATA {name} {index} {chunk}", flush=True)
    print(f"DR190_END {name} {len(chunks)}", flush=True)


def jsonl_value(path, content):
    rows, errors = [], []
    for number, line in enumerate(content.splitlines()):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            errors.append({"line": number + 1, "content": line})
    return {"path": str(path), "format": "jsonl", "rows": rows, "parse_errors": errors}


def failures(value):
    result = []
    if value.get("parse_errors"):
        result.append({"path": value["path"], "reason": "partial JSONL record"})
    for row in value.get("rows", []):
        if not isinstance(row, dict):
            continue
        if row.get("event") == "series_complete" and not row["complete"]:
            result.append({"cell": row.get("cell"), "reason": "incomplete or noisy cell"})
        if row.get("status") == "failed":
            result.append({"model": row.get("model"), "reason": "failed invocation"})
    return result


def final(root, returncode, *, sentinel=True):
    steps, incomplete = [], []
    if (root / "pod4-plan.json").exists():
        try:
            sys.path.insert(0, "/dr151-repo/benchmarks/quality")
            from pod4_receipts import validate

            incomplete.extend(validate(root))
        except (ImportError, OSError, ValueError) as error:
            incomplete.append({"reason": "cannot validate Pod4 receipts: " + str(error)})
    for index, path in enumerate(sorted(root.rglob("*"))):
        if not path.is_file() or path.suffix not in (".json", ".jsonl", ".log", ".txt"):
            continue
        if "pipeline-inputs" in path.parts:
            continue
        name = f"final-file-{index}"
        content = path.read_text(errors="replace")
        value = (
            jsonl_value(path, content)
            if path.suffix == ".jsonl"
            else {"path": str(path), "format": "text", "content": content}
        )
        emit(name, value)
        incomplete.extend(failures(value))
        steps.append({"name": name, "path": str(path)})
    emit(
        "manifest",
        {
            "complete": returncode == 0 and not incomplete,
            "returncode": returncode,
            "status": "complete" if returncode == 0 and not incomplete else "partial or failed",
            "incomplete": incomplete,
            "steps": steps,
        },
    )
    if sentinel:
        print("ALL_DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--failure":
        emit(
            "manifest",
            {
                "complete": False,
                "returncode": int(sys.argv[2]),
                "status": "export or setup failed",
                "reason": sys.argv[3],
            },
        )
    else:
        final(Path(sys.argv[1]), int(sys.argv[2]), sentinel="--no-sentinel" not in sys.argv)
