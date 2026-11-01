"""Real local release-runtime checks, including actual FastText subword tensors."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from support.encoder import finished
from support.runtime import texts

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"
pytestmark = pytest.mark.skipif(not BENCHMARKS.exists(), reason="benchmarks absent")


@pytest.fixture
def runtime_case(monkeypatch, tmp_path, request):
    pytest.importorskip("scipy")
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    release = importlib.import_module("hub_release_cards")
    readback = importlib.import_module("hub_public_readback")
    metrics = importlib.import_module("hub_classic_metrics")
    hashes = importlib.import_module("fse._container._hash")
    dtype, fasttext = getattr(request, "param", ("float32", True))
    encoder = finished(dtype, fasttext=fasttext)
    assert (encoder.word_vectors.fasttext is not None) is fasttext
    sentences = texts(512, seed=7)
    labels = np.arange(256, dtype=np.float32) % 6
    monkeypatch.setattr(readback, "probe_inputs", lambda _: (labels, sentences))
    vectors = encoder.encode(sentences, workers=2)
    probe = {
        "pair_indices": list(range(256)),
        "sentences": 512,
        "pearson_x100": metrics.score(labels, vectors),
        "tolerance_points": 0.001,
        "vectors_xxh3": hashes.checksum(vectors),
    }
    folder = tmp_path / "model"
    encoder.save_pretrained(folder)
    entry = {
        "local_folder": str(folder),
        "method": "raw",
        "metadata": release.metadata_of(encoder, "raw", None),
        "probe": probe,
    }
    return release, entry


@pytest.mark.parametrize(
    "runtime_case",
    [(dtype, fasttext) for dtype in ("float32", "float16", "int8") for fasttext in (False, True)],
    indirect=True,
)
def test_real_runtime_load_metadata_and_probe_pass(runtime_case):
    release, entry = runtime_case
    release.check_runtime(entry)


@pytest.mark.parametrize("failure", ["metadata", "tolerance", "bitwise"])
def test_real_runtime_rejects_each_independent_mismatch(runtime_case, failure):
    release, entry = runtime_case
    if failure == "metadata":
        entry["metadata"]["dimension"] += 1
    elif failure == "tolerance":
        entry["probe"]["pearson_x100"] += 1.0
    else:
        entry["probe"]["vectors_xxh3"] = "xxh3-128:" + "0" * 32
    with pytest.raises(ValueError, match=r"metadata differs|does not reproduce"):
        release.check_runtime(entry)
