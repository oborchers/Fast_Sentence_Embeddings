"""Recoverable publication of small derived card files and their build manifest.

Model/config bytes never change. A staged journal survives interrupted writes.
Recovery accepts only known before/after hashes, never arbitrary edited files.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from hub_classic_delivery import CARD_RECIPE, read_json
from hub_classic_evidence import digest

from fse._container._common import _sync_directory


def _spec(path):
    return {"bytes": path.stat().st_size, "xxh3": digest(path)}


def journal_path(folder):
    return folder.parent / f".{folder.name}.card-transaction.json"


def _write_json(path, value):
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.writing-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Exclusive, atomic publication of complete journal bytes.
        _sync_directory(path.parent)
    finally:
        temporary.unlink()


def publish_card(folder, record_path, cards, text, fit):
    journal = journal_path(folder)
    if journal.exists():
        raise FileExistsError("recover the existing card transaction first")
    before = read_json(record_path)
    stage = Path(tempfile.mkdtemp(prefix=f".{folder.name}.card-stage-", dir=folder.parent))
    try:
        (stage / "README.md").write_text(text)
        if fit is not None:
            (stage / "preparation.json").write_text(json.dumps(fit, indent=2) + "\n")
        after = dict(before, files=dict(before["files"]), card_recipe=CARD_RECIPE)
        for path in stage.iterdir():
            after["files"][path.name] = _spec(path)
        _write_json(stage / "manifest.json", after)
        _sync_directory(stage)
        _write_json(journal, {"stage": stage.name, "before": before, "after": after})
    except BaseException:
        if not journal.exists():
            shutil.rmtree(stage)
        raise
    recover_card(folder, record_path, cards)


def _transaction(folder):
    journal = journal_path(folder)
    transaction = read_json(journal)
    name = transaction["stage"]
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or not name.startswith(f".{folder.name}.card-stage-")
    ):
        raise ValueError("invalid card transaction stage")
    stage = folder.parent / name
    if not stage.is_dir() or stage.is_symlink():
        raise ValueError("card transaction stage is not an owned directory")
    expected = f"fse/{folder.name}"
    if transaction["before"]["repo"] != expected or transaction["after"]["repo"] != expected:
        raise ValueError("card transaction repository mismatch")
    return journal, stage, transaction


def _check_existing(path, allowed):
    if not path.exists():
        return
    if path.is_symlink() or not path.is_file() or _spec(path) not in allowed:
        raise ValueError(f"card transaction refuses edited file {path.name}")


def _check_model(folder, transaction, stage):
    before, after = transaction["before"], transaction["after"]
    managed = {"README.md", "preparation.json"}
    for name, spec in before["files"].items():
        if name not in managed and _spec(folder / name) != spec:
            raise ValueError("model changed during card transaction")
    for name in managed & set(after["files"]):
        if _spec(stage / name) != after["files"][name]:
            raise ValueError("staged card checksum mismatch")
        allowed = [after["files"][name]]
        if name in before["files"]:
            allowed.append(before["files"][name])
        _check_existing(folder / name, allowed)
    if read_json(stage / "manifest.json") != after:
        raise ValueError("staged card manifest mismatch")


def _replace(source, target, transaction_id):
    temporary = target.parent / f".{target.name}.{transaction_id}.writing"
    if temporary.exists():
        if temporary.is_symlink() or not temporary.is_file():
            raise ValueError("invalid owned card-writing intermediate")
        temporary.unlink()
    try:
        with source.open("rb") as incoming, temporary.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        os.replace(temporary, target)
        _sync_directory(target.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def recover_card(folder, record_path, cards):
    journal, stage, transaction = _transaction(folder)
    _check_model(folder, transaction, stage)
    before, after = transaction["before"], transaction["after"]
    current = read_json(record_path)
    if current not in (before, after):
        raise ValueError("build manifest changed during card transaction")
    cards.mkdir(parents=True, exist_ok=True)
    preview = cards / f"{folder.name}.md"
    _check_existing(preview, [before["files"]["README.md"], after["files"]["README.md"]])
    for name in ("README.md", "preparation.json"):
        if (stage / name).exists():
            _replace(stage / name, folder / name, stage.name)
    _replace(stage / "README.md", preview, stage.name)
    _replace(stage / "manifest.json", record_path, stage.name)
    journal.unlink()
    _sync_directory(folder.parent)
    shutil.rmtree(stage)


def recover_pending(models, cards):
    for journal in sorted(models.glob(".*.card-transaction.json")):
        name = journal.name[1 : -len(".card-transaction.json")]
        folder = models / name
        recover_card(folder, models / f"{name}.json", cards)
