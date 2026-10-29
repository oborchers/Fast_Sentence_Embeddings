"""Word vectors for the tests and edits of their saved files."""

import json

import numpy as np

from fse._container import SafetensorsContainer
from fse._wordvectors import FastTextParams, WordVectors

F32 = np.float32
WORDS = ["the", "é", "é", "", "naïve", "日本"]


def _wv(dtype="float32", fasttext=False, counts=True, n=None, dim=5, seed=0):
    rng = np.random.default_rng(seed)
    words = WORDS if n is None else [f"w{i}" for i in range(n)]
    rows = rng.standard_normal((len(words), dim)).astype(F32)
    kwargs = {"counts": np.arange(len(words), dtype=np.int64)} if counts else {}
    if fasttext:
        kwargs["fasttext"] = FastTextParams(3, 5, 7)
        kwargs["ngrams"] = rng.standard_normal((7, dim)).astype(F32)
    wv = WordVectors.from_arrays(rows, words=words, **kwargs)
    return wv if dtype == "float32" else wv.quantize(dtype)


def _same(a: WordVectors, b: WordVectors):
    for name in (
        "stored_vectors",
        "scales",
        "counts",
        "vocab_bytes",
        "vocab_offsets",
        "stored_ngrams",
        "ngram_scales",
    ):
        x, y = getattr(a, name), getattr(b, name)
        assert (x is None) == (y is None), name
        if x is not None:
            assert x.dtype == y.dtype, name
            assert x.shape == y.shape, name
            assert np.asarray(x).tobytes() == np.asarray(y).tobytes(), name
    assert a.fasttext == b.fasttext
    assert a.fingerprint == b.fingerprint


def _tensor_end(path, name):
    """The file offset after the data of tensor ``name`` of a safetensors file."""
    raw = path.read_bytes()
    length = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + length])
    return 8 + length + header[name]["data_offsets"][1]


def _overwrite(path, name, data: bytes):
    """Overwrite the last bytes of tensor ``name`` in place (a mapped file sees it)."""
    with path.open("r+b") as f:
        f.seek(_tensor_end(path, name) - len(data))
        f.write(data)


def _flip_last_byte(path, name):
    with path.open("rb") as f:
        f.seek(_tensor_end(path, name) - 1)
        last = f.read(1)
    _overwrite(path, name, bytes([last[0] ^ 1]))


def _rewrite(path, edit_tensors=None, edit_metadata=None, name="edited"):
    """Rewrite a saved artifact with edited tensors or metadata (fresh checksums)."""
    tensors, metadata = SafetensorsContainer().open(path, mmap=False)
    tensors = {k: np.array(v) for k, v in tensors.items()}
    metadata = {
        k: v
        for k, v in metadata.items()
        if k != "fse.fingerprint" and not k.startswith("fse.checksum.")
    }
    if edit_tensors:
        edit_tensors(tensors)
    if edit_metadata:
        edit_metadata(metadata)
    target = path.parent / f"{name}.safetensors"
    SafetensorsContainer().write(target, tensors, metadata)
    return target


def _saved(wv, tmp_path):
    path = tmp_path / "saved.safetensors"
    wv.save(path)
    return path
