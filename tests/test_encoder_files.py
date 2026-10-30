"""What the readers of other libraries share: configs, the one float32 table, the module
types of sentence-transformers."""

import json

import numpy as np
import pytest

from fse._container import CorruptArtifactError
from fse._contract import UnsupportedArtifactError
from fse._encoder import _files
from fse._encoder._pretrained import LocalFiles

safetensors = pytest.importorskip("safetensors.numpy")
NAMES = frozenset({"a", "b"})


def _files_with(tmp_path, tensors):
    safetensors.save_file(tensors, tmp_path / "t.safetensors")
    return LocalFiles(tmp_path)


def test_a_table_of_one_of_its_names_is_read(tmp_path):
    rows = np.arange(12, dtype=np.float32).reshape(4, 3)
    for name in sorted(NAMES):
        files = _files_with(tmp_path, {name: rows})
        table = _files.read_table(files, "t.safetensors", NAMES, 4)
        assert table.tobytes() == rows.tobytes()
        assert not table.flags.writeable


@pytest.mark.parametrize(
    ("tensors", "error", "match"),
    [
        (
            {"a": np.zeros((4, 3), np.float32), "b": np.zeros(1, np.float32)},
            CorruptArtifactError,
            r"exactly one tensor, a or b, not \['a', 'b'\]",
        ),
        ({"c": np.zeros((4, 3), np.float32)}, CorruptArtifactError, "exactly one tensor"),
        ({"a": np.zeros(4, np.float32)}, CorruptArtifactError, "must be a matrix"),
        ({"a": np.zeros((4, 3), np.float16)}, UnsupportedArtifactError, "a float16 table"),
        ({"a": np.zeros((4, 3), np.int8)}, UnsupportedArtifactError, "an? int8 table"),
        (
            {"a": np.zeros((5, 3), np.float32)},
            CorruptArtifactError,
            "5 rows, but the tokenizer has 4",
        ),
        (
            {"a": np.zeros((4, 3), np.float32), "mapping": np.zeros(4, np.int64)},
            UnsupportedArtifactError,
            "vocabulary quantization",
        ),
    ],
)
def test_the_header_is_checked(tmp_path, tensors, error, match):
    with pytest.raises(error, match=match):
        _files.read_table(_files_with(tmp_path, tensors), "t.safetensors", NAMES, 4)


def test_a_check_sees_the_dtype_and_shape_first(tmp_path):
    seen = []

    def check(path, dtype, shape):
        seen.append((path.name, dtype, shape))
        raise CorruptArtifactError("refused by the check")

    files = _files_with(tmp_path, {"a": np.zeros((4, 3), np.float16)})
    with pytest.raises(CorruptArtifactError, match="refused by the check"):
        _files.read_table(files, "t.safetensors", NAMES, 4, check)
    assert seen == [("t.safetensors", "float16", (4, 3))]


def test_a_single_name_is_named_alone(tmp_path):
    files = _files_with(tmp_path, {"c": np.zeros((4, 3), np.float32)})
    with pytest.raises(CorruptArtifactError, match=r"exactly the tensor a, not \['c'\]"):
        _files.read_table(files, "t.safetensors", frozenset({"a"}), 4)


def test_configs_are_strict_and_bounded(tmp_path):
    files = LocalFiles(tmp_path)
    (tmp_path / "c.json").write_text(json.dumps({"a": [1, 2]}))
    assert _files.read_config(files, "c.json") == {"a": [1, 2]}
    (tmp_path / "c.json").write_text('{"a": 1, "a": 2}')
    with pytest.raises(CorruptArtifactError, match="duplicate key"):
        _files.read_config(files, "c.json")
    with pytest.raises(CorruptArtifactError, match="more than"):
        _files.parse_config(b" " * (_files.MAX_CONFIG_BYTES + 1), tmp_path / "c.json")
    (tmp_path / "c.json").write_bytes(b" " * (_files.MAX_CONFIG_BYTES + 1))
    with pytest.raises(CorruptArtifactError, match="more than the 1048576 it may have"):
        _files.read_config(files, "c.json")


def test_the_module_types_of_each_version():
    """The class paths sentence-transformers writes into modules.json: up to 5.3 the
    module, from 5.4 the module and the class; Normalize moved again in 6.0."""
    assert {
        "sentence_transformers.models.StaticEmbedding",
        "sentence_transformers.sentence_transformer.modules.static_embedding.StaticEmbedding",
    } == _files.STATIC_EMBEDDING
    assert "sentence_transformers.base.modules.normalize.Normalize" in _files.NORMALIZE
    assert len(_files.NORMALIZE) == 3
    assert not (_files.STATIC_EMBEDDING | _files.NORMALIZE) & _files.DENSE
