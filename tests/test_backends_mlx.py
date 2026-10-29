"""fse._backends._mlx: importing fse never imports MLX, and without it the backend says why."""

import subprocess
import sys


def _run(code):
    subprocess.run([sys.executable, "-c", code], check=True)


def test_fse_does_not_import_mlx():
    _run(
        "import sys\n"
        "import fse\n"
        "from fse import _backends, _runtime\n"
        "from fse._backends import _mlx\n"
        "from fse._backends._mlx import _load\n"
        "assert not [m for m in sys.modules if m.split('.')[0] == 'mlx'], 'mlx was imported'\n"
    )
