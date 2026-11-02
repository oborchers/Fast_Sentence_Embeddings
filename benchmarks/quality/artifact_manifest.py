"""Bind converted artifacts to their preparation records and float32 parents."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def header(path: Path) -> dict:
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        return json.loads(stream.read(size))["__metadata__"]


def preparation_record(entry, records, pins):
    name = entry["artifact"]
    matches = [record for record in records if name in record["artifacts"]]
    if len(matches) != 1:
        raise ValueError(f"{name}: need one unambiguous preparation record")
    record = matches[0]
    repo, revision = record["source"], record["revision"]
    if pins.get(repo) != revision or entry["source"] != repo or entry["revision"] != revision:
        raise ValueError(f"{name}: preparation source differs from the pin")
    return record


def checked_header(folder, entry, record):
    name = entry["artifact"]
    metadata = header(folder / name)
    embedded = json.loads(metadata.get("fse.source", "null"))
    if embedded is not None and (
        embedded["repo"] != record["source"] or embedded["revision"] != record["revision"]
    ):
        raise ValueError(f"{name}: embedded source differs from the preparation record")
    if embedded is None and not record["source"].startswith("fse/"):
        raise ValueError(f"{name}: converted Hub encoder lacks its embedded source")
    if metadata["fse.dtype"] != entry["table_dtype"]:
        raise ValueError(f"{name}: table dtype differs from the artifact manifest")
    return metadata, embedded


def bind(folder: Path, artifacts: dict, pins: dict) -> dict:
    records = [json.loads(line) for line in (folder / "sources.jsonl").read_text().splitlines()]
    bound = []
    for entry in artifacts["artifacts"]:
        name = entry["artifact"]
        record = preparation_record(entry, records, pins)
        metadata, embedded = checked_header(folder, entry, record)
        parents = [
            file for file in record["artifacts"] if header(folder / file)["fse.dtype"] == "float32"
        ]
        if len(parents) != 1:
            raise ValueError(f"{name}: need exactly one float32 parent in the preparation record")
        bound.append(
            {
                **entry,
                "encoder_fingerprint": metadata["fse.fingerprint"],
                "float32_parent_fingerprint": header(folder / parents[0])["fse.fingerprint"],
                "embedded_source": embedded,
                "preparation_record": record,
                "provenance_verification": "embedded source and preparation record"
                if embedded
                else "external word-vector preparation record",
            }
        )
    return {**artifacts, "artifacts": bound}


def verified_source(path: Path, encoder) -> dict:
    manifest = json.loads((path.parent / "artifact-identities.json").read_text())
    matches = [entry for entry in manifest["artifacts"] if entry["artifact"] == path.name]
    if len(matches) != 1:
        raise ValueError(f"{path.name}: no unique bound artifact identity")
    entry = matches[0]
    metadata = header(path)
    embedded = json.loads(metadata.get("fse.source", "null"))
    if (
        encoder.fingerprint != entry["encoder_fingerprint"]
        or encoder.word_vectors.dtype != entry["table_dtype"]
        or embedded != entry["embedded_source"]
        or path.stat().st_size != entry["file_bytes"]
    ):
        raise ValueError(
            f"{path.name}: loaded artifact differs from its bound preparation identity"
        )
    return {
        "source_model": entry["source"],
        "source_revision": entry["revision"],
        "table_dtype": entry["table_dtype"],
        "float32_parent_fingerprint": entry["float32_parent_fingerprint"],
        "artifact_file_bytes": entry["file_bytes"],
        "provenance_verification": entry["provenance_verification"],
        "fit_provenance": entry["preparation_record"].get("fit"),
        "offline_recipe": entry["preparation_record"].get("recipe"),
        "timing_proxy": entry["preparation_record"].get("timing_proxy", False),
    }


def main():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from compare_models import MODELS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = bind(args.folder, json.loads(args.artifacts.read_text()), MODELS)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
