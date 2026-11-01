"""Durable small JSON receipts, published without truncating earlier evidence."""

import json
import os
import tempfile
from pathlib import Path

from fse._container._common import _sync_directory


def write_atomic(path, value):
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.writing-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def mkdir_durable(path):
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    if not current.is_dir():
        raise NotADirectoryError(current)
    for directory in reversed(missing):
        directory.mkdir()
        _sync_directory(directory.parent)
