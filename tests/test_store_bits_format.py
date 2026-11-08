"""The companion artifact of the sign bits: its names, its metadata, and how a store opens it."""

import shutil

import numpy as np
import pytest

import fse
from fse._container import NpyContainer, checksum
from fse._store import _bits_format
from fse._store._bits_format import PREFIX, bits_name, first_orphan, width
from fse._store._layout import shard_name
from support.bits import signed_store, stored_bits
from support.store import change_metadata

# The persistence names of the artifact freeze with release 2.1; this test is their register.
NAMES = {
    "prefix": "sign_bits-",
    "tensor": "sign_bits",
    "rule": "sign_bits@1",
    "keys": {
        "fse.kind",
        "fse.format_version",
        "fse.bits",
        "fse.encoder",
        "fse.dtype",
        "fse.output_dim",
        "fse.first_row",
        "fse.rows",
        "fse.shard_fingerprint",
        "fse.created_by",
    },
}


def made(tmp_path, **kwargs):
    path = tmp_path / "store"
    info, rows_of = signed_store(path, **kwargs)
    fse.add_sign_bits(path)
    return path, info, rows_of


def test_the_persistence_names():
    assert (
        NAMES["prefix"],
        NAMES["tensor"],
        NAMES["rule"],
    ) == (PREFIX, _bits_format.TENSOR, _bits_format.RULE)
    assert bits_name(7) == "sign_bits-0000000000000007"
    assert (width(1), width(8), width(9), width(256)) == (1, 1, 2, 32)


def test_the_metadata_holds_exactly_the_named_keys(tmp_path):
    path, _, _ = made(tmp_path)
    tensors, metadata = NpyContainer().open(path / bits_name(0), mmap=False)
    keys = {k for k in metadata if not k.startswith("fse.checksum.") and k != "fse.fingerprint"}
    assert keys == NAMES["keys"]
    assert (metadata["fse.kind"], metadata["fse.format_version"]) == ("sign_bits", "1")
    assert metadata["fse.bits"] == "sign_bits@1"
    assert (
        metadata["fse.shard_fingerprint"]
        == NpyContainer().metadata(path / shard_name(0))["fse.fingerprint"]
    )
    assert list(tensors) == ["sign_bits"]
    assert tensors["sign_bits"].dtype == np.uint8
    assert tensors["sign_bits"].shape == (5, 2)


def test_a_store_without_bits_opens_with_none(tmp_path):
    path = tmp_path / "store"
    signed_store(path)
    with fse.VectorStore.open(path) as store:
        assert stored_bits(store) == {0: None, 5: None}


def test_the_bits_are_mapped_read_only(tmp_path):
    path, _, _ = made(tmp_path)
    with fse.VectorStore.open(path) as store:
        bits = stored_bits(store)[0]
        assert not bits.flags.writeable


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"fse.shard_fingerprint": "xxh3-128:" + "0" * 32}, "bound to another shard"),
        ({"fse.first_row": "1"}, "fse.first_row"),
        ({"fse.rows": "4"}, "fse.rows"),
        ({"fse.output_dim": "11"}, "fse.output_dim"),
        ({"fse.dtype": "int8"}, "fse.dtype"),
        ({"fse.encoder": "xxh3-128:" + "1" * 32}, "fse.encoder"),
        ({"fse.kind": "other"}, "fse.kind"),
        ({"fse.extra": "1"}, "exactly"),
    ],
)
def test_bits_that_do_not_fit_their_shard_are_damage(tmp_path, changes, message):
    path, _, _ = made(tmp_path)
    change_metadata(path / bits_name(0), changes)
    with pytest.raises(fse.CorruptArtifactError, match=message) as caught:
        fse.VectorStore.open(path)
    assert bits_name(0) in str(caught.value)
    assert "fse.repair_store(path)" in str(caught.value)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"fse.format_version": "2"}, "format '2'"),
        ({"fse.bits": "sign_bits@2"}, "sign_bits@2"),
        ({"fse.bits": "centered_bits@1"}, "centered_bits@1"),
        ({"fse.durability": "sometimes"}, "fse.durability"),
    ],
)
def test_an_unknown_version_or_rule_is_never_skipped(tmp_path, changes, message):
    path, _, _ = made(tmp_path)
    change_metadata(path / bits_name(0), changes)
    with pytest.raises(fse.UnsupportedArtifactError, match=message):
        fse.VectorStore.open(path)


def test_the_tensor_must_be_the_uint8_bits_of_the_shard(tmp_path):
    path, _, _ = made(tmp_path)
    directory = path / bits_name(0)
    tensors, metadata = NpyContainer().open(directory, mmap=False)
    for bad in (
        {"sign_bits": tensors["sign_bits"][:4]},
        {"sign_bits": tensors["sign_bits"].astype(np.int8)},
        {"sign_bits": np.ascontiguousarray(tensors["sign_bits"][:, :1])},
        {**tensors, "other": tensors["sign_bits"]},
    ):
        shutil.rmtree(directory)
        NpyContainer().write(directory, bad, metadata_without_integrity(metadata))
        with pytest.raises(fse.CorruptArtifactError, match="it must hold exactly sign_bits"):
            fse.VectorStore.open(path)


def metadata_without_integrity(metadata):
    return {
        k: v
        for k, v in metadata.items()
        if k != "fse.fingerprint" and not k.startswith("fse.checksum.")
    }


def test_a_regular_directory_is_required(tmp_path):
    path, _, _ = made(tmp_path)
    shutil.rmtree(path / bits_name(0))
    (path / bits_name(0)).write_bytes(b"not a directory")
    with pytest.raises(fse.CorruptArtifactError, match="not a regular directory"):
        fse.VectorStore.open(path)


def test_untagged_bits_are_hashed_by_verify_and_tagged_ones_by_open(tmp_path):
    path, _, _ = made(tmp_path)
    file = path / bits_name(0) / "sign_bits.npy"
    data = bytearray(file.read_bytes())
    data[-1] ^= 0x80
    file.write_bytes(bytes(data))
    with fse.VectorStore.open(path) as store, pytest.raises(fse.CorruptArtifactError):
        store.verify()  # the hash of an untagged artifact is not read when it is opened
    tagged = tmp_path / "tagged"
    signed_store(tagged)
    fse.add_sign_bits(tagged, durability="none")
    file = tagged / bits_name(0) / "sign_bits.npy"
    data = bytearray(file.read_bytes())
    data[-1] ^= 0x80
    file.write_bytes(bytes(data))
    with pytest.raises(fse.CorruptArtifactError, match="checksum"):
        fse.VectorStore.open(tagged)


def test_bits_that_are_not_those_of_the_vectors_fail_verify(tmp_path):
    path, _, _ = made(tmp_path)
    directory = path / bits_name(0)
    tensors, metadata = NpyContainer().open(directory, mmap=False)
    bits = np.array(tensors["sign_bits"])
    bits[0, 0] ^= 0xFF
    shutil.rmtree(directory)
    NpyContainer().write(directory, {"sign_bits": bits}, metadata_without_integrity(metadata))
    with (
        fse.VectorStore.open(path) as store,
        pytest.raises(fse.CorruptArtifactError, match="not those of the vectors"),
    ):
        store.verify()


def test_set_padding_bits_fail_verify(tmp_path):
    path, _, _ = made(tmp_path)
    directory = path / bits_name(0)
    tensors, metadata = NpyContainer().open(directory, mmap=False)
    bits = np.array(tensors["sign_bits"])
    bits[:, 1] |= 0x01  # dimension 15 does not exist (dim 10)
    shutil.rmtree(directory)
    NpyContainer().write(directory, {"sign_bits": bits}, metadata_without_integrity(metadata))
    with (
        fse.VectorStore.open(path) as store,
        pytest.raises(fse.CorruptArtifactError, match="padding bits"),
    ):
        store.verify()


def test_bits_without_a_shard_are_an_orphan(tmp_path):
    path, _, _ = made(tmp_path)
    shutil.copytree(path / bits_name(5), path / bits_name(1))  # inside the chain
    with pytest.raises(fse.CorruptArtifactError, match="no shard in the published chain"):
        fse.VectorStore.open(path)
    shutil.rmtree(path / bits_name(1))
    shutil.copytree(path / bits_name(5), path / bits_name(8))  # at the end, no shard follows
    with pytest.raises(fse.CorruptArtifactError, match="sign_bits-"):
        fse.VectorStore.open(path)


def test_bits_of_a_shard_published_after_the_snapshot_are_passed_over(tmp_path):
    path, _, _ = made(tmp_path)
    assert first_orphan(path, {0, 5}, 8) is None
    shutil.copytree(path / bits_name(5), path / bits_name(8))
    assert first_orphan(path, {0, 5}, 8) == (8, bits_name(8))  # its shard is not there
    shutil.copytree(path / shard_name(5), path / shard_name(8))  # now a writer has published it
    assert first_orphan(path, {0, 5}, 8) is None
    assert first_orphan(path, {0}, 8) == (5, bits_name(5))  # inside the snapshot: an orphan


def test_entries_that_are_not_canonical_are_ignored(tmp_path):
    path, _, _ = made(tmp_path)
    for name in (".sign_bits-0000000000000009.writing-ab", "sign_bits-9", "sign_bits-x"):
        (path / name).mkdir()
    with fse.VectorStore.open(path) as store:
        assert len(store) == 8


def test_the_hash_of_the_bits_is_the_checksum_of_their_bytes(tmp_path):
    path, _, rows_of = made(tmp_path)
    metadata = NpyContainer().metadata(path / bits_name(0))
    with fse.VectorStore.open(path) as store:
        assert metadata["fse.checksum.sign_bits"] == checksum(stored_bits(store)[0])
    assert len(rows_of) == 2
