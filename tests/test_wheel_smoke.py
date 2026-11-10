"""The smoke test of the Windows wheels (tools/wheel_smoke.py, DR-344) passes and fails."""

import numpy as np
import pytest
import wheel_smoke

import fse
from fse._runtime._tasks import INLINE_MAX_BYTES


def test_the_smoke_test_passes_on_this_build(capsys):
    assert wheel_smoke.main() == 0
    assert "smoke test passed" in capsys.readouterr().out


def test_the_smoke_test_fails_when_the_kernel_differs_from_the_reference(monkeypatch):
    real = fse.Encoder.encode

    def altered(self, texts, **kwargs):
        out = real(self, texts, **kwargs)
        return out + np.float32(1e-3) if kwargs.get("backend") == "numpy" else out

    monkeypatch.setattr(fse.Encoder, "encode", altered)
    with pytest.raises(SystemExit, match="differs from the reference"):
        wheel_smoke.main()


def test_the_text_is_large_enough_for_worker_threads():
    """Up to INLINE_MAX_BYTES of text the tasks run in the calling thread."""
    assert sum(len(text.encode()) for text in wheel_smoke._texts()) > INLINE_MAX_BYTES
