"""Small real stores shared by document-ID tests."""

import shutil
from pathlib import Path

import numpy as np

import fse
from fse._container import NpyContainer
from fse._container._hash import Hasher
from support.encoder import finished
from support.runtime import texts


def write_store(
    path, ids, *, dtype="float32", shard_rows=7, batches=False, durability="full", sentences=None
):
    encoder = finished(dim=17)
    sentences = texts(len(ids)) if sentences is None else sentences
    with fse.ShardSink(
        path, encoder, dtype=dtype, shard_rows=shard_rows, doc_ids=True, durability=durability
    ) as sink:
        if batches:
            parts = ((sentences[i : i + 3], ids[i : i + 3]) for i in range(0, len(ids), 3))
            list(encoder.encode_batches(parts, sink=sink, workers=2, doc_ids=True))
        else:
            encoder.encode(sentences, doc_ids=ids, sink=sink, workers=2)
    return encoder


def column(store, name):
    return np.concatenate([t[name] for _, t in store._shards])


def prerelease_shard(directory):
    """Rewrite one shard as a pre-release of 2.1 wrote it: tensor ``row_ids`` and the keys
    ``fse.row_ids*``, with valid checksums and fingerprint."""
    directory = Path(directory)
    tensors, metadata = NpyContainer().open(directory, mmap=False)
    tensors = {("row_ids" if k == "doc_ids" else k): v for k, v in tensors.items()}
    metadata = {
        k.replace("fse.doc_ids", "fse.row_ids"): v
        for k, v in metadata.items()
        if not k.startswith("fse.checksum.") and k != "fse.fingerprint"
    }
    replacement = directory.parent / f".replacement-{directory.name}"
    NpyContainer().write(replacement, tensors, metadata)
    for old in directory.iterdir():
        old.unlink()
    for new in replacement.iterdir():
        shutil.move(new, directory / new.name)
    replacement.rmdir()


def prerelease_store(path, ids, *, shards=None, **options):
    """An ID store written with the removed names: ``store.json`` and the shards ``shards``
    (all by default)."""
    path = Path(path)
    write_store(path, ids, **options)
    file = path / "store.json"
    file.write_text(file.read_text().replace('"fse.doc_ids"', '"fse.row_ids"'))
    for directory in sorted(path.glob("shard-*")):
        if shards is None or directory.name in shards:
            prerelease_shard(directory)


def tree(path):
    """Every entry under ``path`` with its kind, size, mtime and content checksum."""
    entries = []
    for entry in sorted(Path(path).rglob("*")):
        stat = entry.stat()
        digest = None
        if entry.is_file():
            hasher = Hasher()
            hasher.update(entry.read_bytes())
            digest = hasher.checksum()
        entries.append(
            (str(entry.relative_to(path)), entry.is_dir(), stat.st_size, stat.st_mtime_ns, digest)
        )
    return entries
