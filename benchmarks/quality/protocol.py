"""Content identities for evaluation methods and the code actually executed."""

from __future__ import annotations

import json
import math
from pathlib import Path


def files_digest(root: Path, paths: list[Path]) -> str:
    from fse._container._hash import Hasher

    hasher = Hasher()
    for path in sorted(paths):
        name = str(path.relative_to(root)).encode("utf-8")
        content = path.read_bytes()
        for data in (name, content):
            hasher.update(len(data).to_bytes(8, "little"))
            hasher.update(data)
    return hasher.checksum()


def measurement_identity(tasks: list) -> dict:
    import fse
    from fse._container._hash import checksum

    folder = Path(__file__).resolve().parent
    method = [
        folder / name
        for name in ("run_mteb.py", "model2vec_adapter.py", "resume.py", "protocol.py")
    ]
    runtime = Path(fse.__file__).resolve().parent
    code = [*runtime.rglob("*.py"), *runtime.glob("_native*.so"), *runtime.glob("_native*.pyd")]
    suite = [task.metadata.model_dump(mode="json") for task in tasks]
    return {
        "method_fingerprint": files_digest(folder, method),
        "runtime_fingerprint": files_digest(runtime, code),
        "suite_fingerprint": checksum(json.dumps(suite, sort_keys=True).encode("utf-8")),
    }


def configuration_key(row: dict) -> str:
    from resume import KEYS

    fields = {key: row.get(key) for key in (*KEYS, "machine", "python")}
    if not row.get("method_fingerprint") or not row.get("runtime_fingerprint"):
        fields["legacy_commit"] = row.get("commit", "unknown")
    return json.dumps(fields, sort_keys=True)


def compatible_results(rows: list[dict], tasks: list) -> dict[str, list[dict]]:
    expected = {task.metadata.name: task.metadata.model_dump(mode="json") for task in tasks}
    groups: dict[str, dict[str, dict]] = {}
    for row in rows:
        key = configuration_key(row)
        group = groups.setdefault(key, {})
        if (
            row.get("task_metadata") == expected.get(row["task"])
            and row["task"] in expected
            and isinstance(row.get("main_score"), (int, float))
            and math.isfinite(row["main_score"])
        ):
            group[row["task"]] = row
    return {key: list(tasks.values()) for key, tasks in groups.items()}
