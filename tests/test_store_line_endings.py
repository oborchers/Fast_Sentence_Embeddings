"""A store written with CRLF manifests (2.0.0 on Windows) stays readable and repairable."""

import fse
from fse._store._layout import shard_name
from support.store import make_store


def _crlf(path):
    for file in [path / "store.json", *path.glob("shard-*/manifest.json")]:
        file.write_bytes(file.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))


def test_a_store_with_crlf_manifests_opens_verifies_and_repairs(tmp_path):
    path = tmp_path / "store"
    make_store(path, sizes=(4, 4, 2))
    with fse.VectorStore.open(path) as store:
        want = store.get_vectors(start=0)
    _crlf(path)
    assert b"\r\n" in (path / shard_name(0) / "manifest.json").read_bytes()
    before = {p: p.read_bytes() for p in path.rglob("*") if p.is_file()}
    with fse.VectorStore.open(path) as store:
        assert len(store) == 10
        assert (store.get_vectors(start=0) == want).all()
        store.verify()
    assert fse.repair_store(path) == 10
    assert {p: p.read_bytes() for p in path.rglob("*") if p.is_file()} == before
