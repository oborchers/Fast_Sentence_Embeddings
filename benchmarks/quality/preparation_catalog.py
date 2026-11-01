"""Register each produced artifact without rereading unrelated cache entries."""

from __future__ import annotations

import json
from pathlib import Path

from artifact_manifest import header


def register(folder: Path, path: Path, record: dict, parent: str) -> None:
    metadata = header(path)
    source_file = folder / "sources.jsonl"
    records = (
        [json.loads(s) for s in source_file.read_text().splitlines()]
        if source_file.exists()
        else []
    )
    matches = [
        r
        for r in records
        if r.get("source") == record["source"] and r.get("recipe") == record["recipe"]
    ]
    if len(matches) > 1:
        raise ValueError("ambiguous preparation records")
    if matches:
        selected = matches[0]
        if path.name not in selected["artifacts"]:
            selected["artifacts"].append(path.name)
    else:
        records.append({**record, "artifacts": [path.name]})
    source_file.write_text("".join(json.dumps(r) + "\n" for r in records))
    manifest = folder / "artifact-identities.json"
    document = json.loads(manifest.read_text()) if manifest.exists() else {"artifacts": []}
    if any(r["artifact"] == path.name for r in document["artifacts"]):
        raise ValueError("artifact already registered")
    document["artifacts"].append(
        {
            "artifact": path.name,
            "source": record["source"],
            "revision": record["revision"],
            "table_dtype": metadata["fse.dtype"],
            "file_bytes": path.stat().st_size,
            "encoder_fingerprint": metadata["fse.fingerprint"],
            "float32_parent_fingerprint": parent,
            "embedded_source": json.loads(metadata.get("fse.source", "null")),
            "preparation_record": record,
            "provenance_verification": "pinned source and offline preparation record",
        }
    )
    manifest.write_text(json.dumps(document, indent=2) + "\n")
