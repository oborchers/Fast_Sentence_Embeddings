"""The persistence names of document IDs, and the refusal of the removed pre-release names."""

import json
import shutil

import pytest

import fse
from fse._container import NpyContainer
from support.document_ids import prerelease_shard, prerelease_store, tree, write_store

IDS = [5, 3, 3, 9, -(2**63), 7]
STORE_KEYS = {
    "fse.kind",
    "fse.format_version",
    "fse.transform_version",
    "fse.encoder",
    "fse.dtype",
    "fse.output_dim",
    "fse.doc_ids",
    "fse.resume",
}


@pytest.mark.parametrize("dtype", ["float32", "int8"])
def test_golden_names_and_keys(tmp_path, dtype):
    write_store(tmp_path / "s", IDS, dtype=dtype, shard_rows=4)
    store = json.loads((tmp_path / "s" / "store.json").read_text())
    assert store["store_version"] == 2
    assert set(store["metadata"]) == STORE_KEYS
    assert store["metadata"]["fse.doc_ids"] == "int64"
    files = {"manifest.json", "vectors.npy", "inv_norms.npy", "doc_ids.npy"}
    if dtype == "int8":
        files.add("scales.npy")
    shards = sorted((tmp_path / "s").glob("shard-*"))
    assert len(shards) == 2
    for shard, rows in zip(shards, ([5, 3, 3, 9], [-(2**63), 7]), strict=True):
        assert {p.name for p in shard.iterdir()} == files
        manifest = json.loads((shard / "manifest.json").read_text())
        metadata = manifest["metadata"]
        assert metadata["fse.doc_ids"] == "int64"
        assert metadata["fse.doc_ids_min"] == str(min(rows))
        assert metadata["fse.doc_ids_max"] == str(max(rows))
        assert metadata["fse.doc_ids_sorted"] == str(rows == sorted(rows)).lower()
        assert not any("row_ids" in key for key in metadata)
        entry = manifest["tensors"]["doc_ids"]
        assert entry["dtype"] == "I64"
        assert entry["shape"] == [len(rows)]


def test_store_without_ids_stays_version_one(tmp_path):
    encoder = fse_encoder()
    with fse.ShardSink(tmp_path / "s", encoder, shard_rows=4) as sink:
        encoder.encode(["w1", "w2"], sink=sink)
    store = json.loads((tmp_path / "s" / "store.json").read_text())
    assert store["store_version"] == 1
    assert not any("doc_ids" in key for key in store["metadata"])
    assert not list((tmp_path / "s").glob("shard-*/doc_ids.npy"))


def fse_encoder():
    from support.encoder import finished

    return finished(dim=17)


def refused(action, root):
    before = tree(root)
    with pytest.raises(fse.UnsupportedArtifactError, match="pre-release"):
        action()
    assert tree(root) == before


def append(root):
    return lambda: fse.ShardSink(root, fse_encoder(), doc_ids=True, shard_rows=4)


def test_old_store_is_refused_everywhere(tmp_path):
    root = tmp_path / "s"
    prerelease_store(root, IDS, shard_rows=4)
    refused(lambda: fse.VectorStore.open(root), root)
    refused(append(root), root)
    refused(lambda: fse.repair_store(root), root)


def test_old_shard_under_a_new_store_is_refused(tmp_path):
    root = tmp_path / "s"
    write_store(root, IDS, shard_rows=4)
    prerelease_shard(root / "shard-0000000000000004")
    refused(lambda: fse.VectorStore.open(root), root)
    refused(append(root), root)
    refused(lambda: fse.repair_store(root), root)
    assert not list(root.glob(".broken-*")) and not list(root.glob(".shard-*"))


def test_repair_refuses_before_quarantining_a_broken_early_shard(tmp_path):
    root = tmp_path / "s"
    write_store(root, list(range(10)), shard_rows=4)
    with (root / "shard-0000000000000000" / "vectors.npy").open("r+b") as stream:
        stream.seek(-1, 2)
        byte = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([byte[0] ^ 1]))
    prerelease_shard(root / "shard-0000000000000008")
    refused(lambda: fse.repair_store(root), root)
    assert not list(root.glob(".broken-*"))


def test_verify_refuses_a_replaced_manifest(tmp_path):
    root = tmp_path / "s"
    write_store(root, IDS, shard_rows=4)
    with fse.VectorStore.open(root) as store:
        store.verify()
        prerelease_shard(root / "shard-0000000000000004")
        refused(store.verify, root)


def test_verify_refuses_a_rewritten_store_file(tmp_path):
    root = tmp_path / "s"
    write_store(root, IDS, shard_rows=4)
    with fse.VectorStore.open(root) as store:
        file = root / "store.json"
        file.write_text(file.read_text().replace('"fse.doc_ids"', '"fse.row_ids"'))
        refused(store.verify, root)


def test_old_tensor_name_with_new_keys_is_corrupt(tmp_path):
    root = tmp_path / "s"
    write_store(root, IDS, shard_rows=8)
    directory = root / "shard-0000000000000000"
    tensors, metadata = NpyContainer().open(directory, mmap=False)
    tensors["row_ids"] = tensors.pop("doc_ids")
    metadata = {
        k: v
        for k, v in metadata.items()
        if not k.startswith("fse.checksum.") and k != "fse.fingerprint"
    }
    NpyContainer().write(tmp_path / "r", tensors, metadata)
    shutil.rmtree(directory)
    shutil.move(tmp_path / "r", directory)
    with pytest.raises(fse.CorruptArtifactError):
        fse.VectorStore.open(root)
