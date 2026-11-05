"""The kind sentence_vectors: root, identifiers and the values of one shard."""

import numpy as np
import pytest

from fse._container import CorruptArtifactError
from fse._contract import UnsupportedArtifactError
from fse._store._layout import open_shard, read_store
from fse._store._vectors_format import VECTORS, StoreInfo, shard_metadata
from support.container import fake_fingerprint
from support.store import change_metadata, make_store

INFO = StoreInfo(fake_fingerprint("a"), "int8", 3)


def test_the_metadata_of_a_root_and_a_shard_share_the_kind_keys():
    root = INFO.metadata()
    shard = shard_metadata(INFO, 4, 2, "test", unsynced=True)
    assert root["fse.kind"] == "sentence_vectors"
    assert {key: shard[key] for key in root} == root
    assert shard["fse.durability"] == "none"
    assert shard["fse.quantization"] == "quantize_rows@1"
    assert "fse.quantization" not in shard_metadata(
        StoreInfo(INFO.encoder, "float32", 3), 0, 1, "t"
    )


def test_a_root_must_agree_with_its_store_version():
    with pytest.raises(CorruptArtifactError, match="store and sentence_vectors versions differ"):
        VECTORS.parse_root(2, INFO.metadata(), "store.json")
    assert VECTORS.parse_root(1, INFO.metadata(), "store.json") == INFO


@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        ({"fse.format_version": "9"}, UnsupportedArtifactError, "unsupported format '9'"),
        ({"fse.durability": "future"}, UnsupportedArtifactError, "unsupported fse.durability"),
        ({"fse.quantization": "quantize_rows@7"}, UnsupportedArtifactError, "quantize_rows@7"),
        ({"fse.quantization": "!"}, CorruptArtifactError, "must be quantize_rows@1"),
    ],
)
def test_a_future_identifier_is_recognized_before_the_structure(change, error, match):
    metadata = {**shard_metadata(INFO, 0, 1, "t"), **change}
    with pytest.raises(error, match=match):
        VECTORS.check_identifiers(metadata, "manifest.json", INFO)


def test_no_shard_constrains_its_neighbours(tmp_path):
    make_store(tmp_path / "s", sizes=(4,))
    shard, tensors = open_shard(tmp_path / "s", read_store(tmp_path / "s"), 0)
    assert VECTORS.summary(shard, tensors) is None
    assert VECTORS.follows(None, None) is None


@pytest.mark.parametrize(
    ("dtype", "tensor", "value", "match"),
    [
        ("float32", "vectors", np.nan, "vectors hold values that are not finite"),
        ("float16", "vectors", np.inf, "vectors hold values that are not finite"),
        ("int8", "vectors", -128, "an int8 code is -128"),
        ("float32", "inv_norms", -1.0, "inv_norms holds values that are not finite and at least 0"),
        ("int8", "scales", np.nan, "scales holds values that are not finite and at least 0"),
    ],
)
def test_the_values_of_a_shard_are_checked(tmp_path, dtype, tensor, value, match):
    path = tmp_path / "s"
    make_store(path, dtype=dtype, sizes=(4,))
    shard, mapped = open_shard(path, read_store(path), 0)
    mapped = {name: np.array(array) for name, array in mapped.items()}
    mapped[tensor][0] = value
    with pytest.raises(CorruptArtifactError, match=match):
        VECTORS.check_values(shard, mapped)


def test_a_shard_that_does_not_match_its_store_is_corrupt(tmp_path):
    path = tmp_path / "s"
    make_store(path, sizes=(4,))
    change_metadata(path / "shard-0000000000000000", {"fse.rows": "5"})
    with pytest.raises(CorruptArtifactError):
        open_shard(path, read_store(path), 0)
