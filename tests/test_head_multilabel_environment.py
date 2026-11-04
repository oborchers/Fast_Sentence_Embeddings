"""Every public multilabel route compares subnormals in the kernel environment."""

import numpy as np
import pytest

import fse
from fse import _native
from fse._backends._cpu import CpuBackend
from support.encoder import finished

TEXTS = ["", "w1"]


def _model(tmp_path, route, dtype):
    encoder = finished(normalize=route == "normalized")
    model = fse.Classifier.from_arrays(
        encoder,
        np.zeros((4, encoder.output_dim), np.float32),
        np.array([-100, -104, 0, 104], np.float32),
        labels=["tiny", "zero", "tie", "one"],
        link="multilabel",
        thresholds=[0, 0, 0.5, 1],
    )
    if dtype != "float32":
        model = model.quantize(dtype)
    if route in ("loaded", "standalone"):
        path = tmp_path / "model.safetensors"
        model.save(path, standalone=route == "standalone")
        model = fse.Classifier.load(path, mmap=True)
    return model


def _predict(model, tmp_path, route):
    if route == "store":
        with fse.ShardSink(tmp_path / "vectors", model.encoder) as sink:
            model.encoder.encode(TEXTS, workers=1, sink=sink)
        with fse.VectorStore.open(tmp_path / "vectors") as store:
            return model.predict(store, workers=1)
    if route == "vectors":
        batches = model.predict_batches([TEXTS], workers=1, return_vectors=True)
        return next(batches).outputs
    return model.predict(TEXTS, workers=1)


@pytest.mark.parametrize(
    "route", ["folded", "normalized", "vectors", "loaded", "standalone", "store"]
)
@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_public_decisions_use_and_restore_a_protected_environment(
    tmp_path, monkeypatch, route, dtype
):
    model = _model(tmp_path, route, dtype)
    decisions = CpuBackend.decisions
    checked = []

    def in_foreign_environment(self, probabilities, thresholds, out):
        saved = _native._test_fpenv_set(flush=True, round_up=True)
        try:
            before = _native._test_fpenv_get()
            decisions(self, probabilities, thresholds, out)
            checked.append(_native._test_fpenv_get() == before)
        finally:
            _native._test_fpenv_restore(saved)

    monkeypatch.setattr(CpuBackend, "decisions", in_foreign_environment)
    got = _predict(model, tmp_path, route)
    assert got.tolist() == [[True, False, False, False]] * len(TEXTS)
    assert checked and all(checked)
