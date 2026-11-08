"""A store or view maps its files until it is closed (docs/architecture/store.md,
"Concurrency"), checked with the Windows emulation of ``windows_file_maps``: a file a live
map holds cannot be deleted. The real Windows case with another process is in
tests/test_store_vectorstore_processes.py."""

import gc

import pytest

import fse
from support import scores as sc
from support.views import score_parts, vector_parts

pytestmark = pytest.mark.usefixtures("windows_file_maps")


def _tensors(*paths):
    """Every tensor file of every shard of the stores ``paths``."""
    files = [file for path in paths for file in sorted(path.glob("shard-*/*.npy"))]
    assert {file.parent.parent for file in files} == set(paths)
    return files


def _released_on_close(store, files):
    """``files`` cannot be deleted while ``store`` is open, and can be once it is closed."""
    for file in files:
        with pytest.raises(PermissionError, match="mapped"):
            file.unlink()
    store.close()
    gc.collect()
    for file in files:
        file.unlink()


def test_a_score_store_maps_its_files_until_it_is_closed(tmp_path):
    path = tmp_path / "scores"
    sc.make_scores(path)
    store = fse.ScoreStore.open(path)
    _released_on_close(store, _tensors(path))


def test_a_vector_view_maps_the_files_of_every_member_until_it_is_closed(tmp_path):
    _, paths = vector_parts(tmp_path)
    view = fse.VectorStore.open_view(paths)
    _released_on_close(view, _tensors(*paths))


def test_a_score_view_maps_the_files_of_every_member_until_it_is_closed(tmp_path):
    _, paths, _ = score_parts(tmp_path)
    view = fse.ScoreStore.open_view(paths)
    _released_on_close(view, _tensors(*paths))
