"""Tensors, metadata and raw safetensors files for the container tests."""

import json

import numpy as np

F32 = np.float32
FINGERPRINT = "xxh3-128:18063af8c78a3bc9bb2ca036e75551d5"


def _tensors():
    return {
        "vectors": np.array([[1.0, -0.0], [0.5, 2.0]], dtype=F32),
        "vocab.bytes": np.frombuffer("aé".encode(), dtype=np.uint8).copy(),
        "vocab.offsets": np.array([0, 1, 3], dtype=np.int64),
    }


_METADATA = {
    "fse.kind": "word_vectors",
    "fse.format_version": "2",
    "fse.dim": "2",
    "fse.dtype": "float32",
}


def _assemble(path, header, data, pad_to=64, raw_header=None):
    text = raw_header if raw_header is not None else json.dumps(header).encode()
    text += b" " * (-(8 + len(text)) % pad_to)
    path.write_bytes(len(text).to_bytes(8, "little") + text + data)


def fake_fingerprint(char: str) -> str:
    """A well-formed fingerprint that belongs to no content."""
    return "xxh3-128:" + char * 32


LEGACY = "sha256:" + "b" * 64  # a fingerprint of the pre-release format


def _replaced_hashes(make):
    """An edit of metadata that replaces every checksum and the fingerprint by ``make(key)``."""

    def edit(metadata):
        for key in metadata:
            if key.startswith("fse.checksum.") or key == "fse.fingerprint":
                metadata[key] = make(key)

    return edit


#: Edits of the hashes that the containers reject, with the error and its message.
ALGORITHM_CASES = {
    "sha256 values": (
        _replaced_hashes(lambda k: LEGACY if k == "fse.fingerprint" else "a" * 64),
        "unsupported",
        "sha256 checksums of a pre-release",
    ),
    "another algorithm": (
        _replaced_hashes(lambda _: "blake3:" + "a" * 64),
        "unsupported",
        "does not read",
    ),
    "two other algorithms": (
        _replaced_hashes(
            lambda k: "xxh3-64:" + "a" * 16 if k == "fse.fingerprint" else "blake3:" + "a" * 64
        ),
        "corrupt",
        "malformed",
    ),
    "sha256 and xxh3-128 mixed": (
        _replaced_hashes(lambda k: LEGACY if k == "fse.fingerprint" else fake_fingerprint("a")),
        "corrupt",
        "malformed",
    ),
}


def _small_chunks(monkeypatch, size=8):
    """Hash and write in blocks of ``size`` bytes in every module of the container package."""
    from fse._container import _common, _npy, _safetensors

    for module in (_common, _npy, _safetensors):
        monkeypatch.setattr(module, "_CHUNK", size)
