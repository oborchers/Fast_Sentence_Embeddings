"""Release regenerable pod inputs while preserving their immutable identity receipts."""

import os
from pathlib import Path

from compare_artifacts import file_digest


def preserve_identity(options, path):
    path = Path(path)
    if path.parent.resolve() != options.inputs.resolve() or path.suffix != ".txt":
        raise ValueError("refuse a file outside the owned prepared input folder")
    receipt = path.with_suffix(".identity.json")
    import json

    data = receipt.read_bytes()
    record = json.loads(data)
    if record["checksum"] != file_digest(path):
        raise ValueError("prepared input changed before identity preservation")
    target = options.output / "input-identities" / receipt.name
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        if target.read_bytes() != data:
            raise ValueError("regenerated input differs from its preserved identity")
    else:
        with target.open("xb") as stream:
            stream.write(data)


def release(options, path):
    path = Path(path)
    if path in options.pinned_inputs:
        return
    preserve_identity(options, path)
    path.unlink()
    path.with_suffix(".identity.json").unlink()
    if options.input_file == path:
        options.input_file = None


def release_current(options):
    if getattr(options, "input_file", None) is not None:
        release(options, options.input_file)


def discard_validation_cache(path):
    """Flush then advise away only this temporary output, outside encoding timers."""
    if not hasattr(os, "posix_fadvise"):
        raise RuntimeError("large validation cache management requires Linux posix_fadvise")
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
