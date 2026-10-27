"""Tensors, metadata and raw safetensors files for the container tests."""

import json

import numpy as np

F32 = np.float32
FINGERPRINT = "sha256:98a86ce422b1d61979da4b088934ebbb5c4f128463ca20123f7508d0f74f32db"


def _tensors():
    return {
        "vectors": np.array([[1.0, -0.0], [0.5, 2.0]], dtype=F32),
        "vocab.bytes": np.frombuffer("aé".encode(), dtype=np.uint8).copy(),
        "vocab.offsets": np.array([0, 1, 3], dtype=np.int64),
    }


_METADATA = {
    "fse.kind": "word_vectors",
    "fse.format_version": "1",
    "fse.dim": "2",
    "fse.dtype": "float32",
}


def _assemble(path, header, data, pad_to=64, raw_header=None):
    text = raw_header if raw_header is not None else json.dumps(header).encode()
    text += b" " * (-(8 + len(text)) % pad_to)
    path.write_bytes(len(text).to_bytes(8, "little") + text + data)
