"""fse._backends._mlx: importing fse never imports MLX, and without it the backend says why."""

import subprocess
import sys

import pytest

from fse._backends import BACKEND_NAMES, MlxBackend, get_backend


def _run(code):
    subprocess.run([sys.executable, "-c", code], check=True)


def test_fse_does_not_import_mlx():
    _run(
        "import sys\n"
        "import fse\n"
        "from fse import _backends, _runtime\n"
        "from fse._backends import _mlx\n"
        "assert not [m for m in sys.modules if m.split('.')[0] == 'mlx'], 'mlx was imported'\n"
    )


def test_without_mlx_the_backend_names_the_extra():
    """MLX blocked as if it were not installed: the backend and an encoder refuse clearly."""
    _run(
        "import sys\n"
        "sys.modules['mlx'] = None\n"
        "sys.modules['mlx.core'] = None\n"
        "import numpy as np, fse\n"
        "from fse._backends import UnsupportedByBackendError, get_backend\n"
        "for make in (lambda: get_backend('mlx'), lambda: fse.Average().fit(\n"
        "        fse.WordVectors.from_arrays(np.ones((2, 3), np.float32), words=['a', 'b'])\n"
        "    ).encode(['a b'], backend='mlx')):\n"
        "    try:\n"
        "        make()\n"
        "    except UnsupportedByBackendError as err:\n"
        "        assert 'pip install \"fse[mlx]\"' in str(err), err\n"
        "    else:\n"
        "        raise AssertionError('the mlx backend was made without mlx')\n"
    )


def test_mlx_is_a_backend_name():
    assert "mlx" in BACKEND_NAMES
    assert MlxBackend.name == "mlx"


def test_isa_belongs_to_the_cpu_backend():
    with pytest.raises(ValueError, match="cpu backend only"):
        get_backend("mlx", isa="portable")
