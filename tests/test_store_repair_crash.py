"""Every repair mutation boundary is rerunnable, with append attempted before repair."""

import subprocess
import sys
from pathlib import Path

import pytest

from support.repair import after_crash, damaged_store

CHILD = """
import os, sys
from pathlib import Path
import fse
from fse._store import _repair

point = int(sys.argv[2])
count = 0
real_mkdir, real_rename, real_sync = os.mkdir, _repair._rename_exclusive, _repair._sync_directory

def tick():
    global count
    count += 1
    if count == point:
        os._exit(77)

def mkdir(*args, **kwargs):
    result = real_mkdir(*args, **kwargs)
    tick()
    return result

def rename(*args):
    result = real_rename(*args)
    tick()
    return result

def sync(*args, **kwargs):
    tick()
    result = real_sync(*args, **kwargs)
    tick()
    return result

os.mkdir = mkdir
_repair._rename_exclusive, _repair._sync_directory = rename, sync
assert fse.repair_store(Path(sys.argv[1])) == 4
"""


@pytest.mark.parametrize("point", range(1, 19))
def test_process_crash_at_every_move_and_directory_sync(tmp_path, point):
    path = tmp_path / "store"
    damaged_store(path)
    child = subprocess.run(
        [sys.executable, "-c", CHILD, str(path), str(point)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert child.returncode == 77, child.stderr
    after_crash(path)


def test_repair_finishes_without_a_crash(tmp_path):
    import fse

    path = tmp_path / "store"
    damaged_store(path)
    child = subprocess.run(
        [sys.executable, "-c", CHILD, str(path), "100"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert child.returncode == 0, child.stderr
    assert fse.repair_store(path) == 4
    after_crash(path)
