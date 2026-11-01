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
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def mkdir_durable(path, *, exist_ok=False):
    if not path.parent.exists():
        mkdir_durable(path.parent, exist_ok=True)
    path.mkdir(exist_ok=exist_ok)
    _sync_directory(path.parent)
