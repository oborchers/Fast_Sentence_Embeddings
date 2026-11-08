"""A store or view maps its files until it is closed (docs/architecture/store.md,
"Concurrency"), checked with the Windows emulation of ``windows_file_maps``: a file a live
map holds cannot be deleted. The real Windows case with another process is in
tests/test_store_vectorstore_processes.py."""

import gc

import pytest

import fse
from fse._store._layout import shard_name
from support import scores as sc
from support.views import score_parts, vector_parts

pytestmark = pytest.mark.usefixtures("windows_file_maps")


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
    _released_on_close(store, [path / shard_name(row) / "scores.npy" for row in (0, 4, 8)])


def test_a_vector_view_maps_the_files_of_every_member_until_it_is_closed(tmp_path):
    _, paths = vector_parts(tmp_path)
    view = fse.VectorStore.open_view(paths)
    _released_on_close(view, [path / shard_name(0) / "vectors.npy" for path in paths])


def test_a_score_view_maps_the_files_of_every_member_until_it_is_closed(tmp_path):
    _, paths, _ = score_parts(tmp_path)
    view = fse.ScoreStore.open_view(paths)
    _released_on_close(view, [path / shard_name(0) / "scores.npy" for path in paths])
