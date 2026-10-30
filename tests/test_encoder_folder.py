"""Hugging Face folders of fse models: their layout and config, the round trip through
from_pretrained, extra files, and writes that are atomic and never replace anything."""

import json
import os
import threading
from pathlib import Path

import numpy as np
import pytest

import fse
from fse._container import (
    CorruptArtifactError,
    PublishedNotDurableError,
    SafetensorsContainer,
    _npy_inplace,
)
from fse._contract import UnsupportedArtifactError
from fse._encoder import _card, _folder, _pretrained
from support import model2vec as mv
from support.encoder import finished
from support.runtime import texts

FINISHED = Path(__file__).parent / "test_data" / "v2" / "finished"
GOLDEN = {
    "encoder_float32": fse.Encoder,
    "encoder_int8": fse.Encoder,
    "classifier_float32": fse.Classifier,
    "classifier_standalone": fse.Classifier,
    "regressor_int8": fse.Regressor,
}
SENTENCES = texts(60)
F32 = np.float32


def _classifier(normalize=False):
    encoder = finished(normalize=normalize)
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((3, encoder.output_dim)).astype(F32)
    return fse.Classifier.from_arrays(
        encoder, weight, rng.standard_normal(3), labels=["a", "b", "c"]
    )


def _fingerprint_of_file(path):
    return SafetensorsContainer().open(path, mmap=False)[1]["fse.fingerprint"]


def _leftovers(folder):
    return sorted(p.name for p in folder.iterdir() if p.name.startswith("."))


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_a_folder_holds_the_model_file_its_config_and_its_card(name, tmp_path):
    cls = GOLDEN[name]
    model = cls.load(FINISHED / f"{name}.safetensors")
    folder = model.save_pretrained(tmp_path / name)
    assert folder == tmp_path / name
    assert sorted(p.name for p in folder.iterdir()) == [
        "README.md",
        "config.json",
        "model.safetensors",
    ]
    kind = "encoder" if cls is fse.Encoder else "linear_head"
    want = {
        "fingerprint": model.fingerprint,
        "fse_config_version": 1,
        "fse_format_version": 1,
        "fse_kind": kind,
        "model": "model.safetensors",
        "model_type": "fse",
    }
    assert (folder / "config.json").read_bytes() == (json.dumps(want, indent=2) + "\n").encode()
    assert _fingerprint_of_file(folder / "model.safetensors") == model.fingerprint
    again = cls.from_pretrained(folder)
    assert again.fingerprint == model.fingerprint
    assert _leftovers(tmp_path) == []


def test_an_encoder_folder_gives_the_rows_of_the_encoder(tmp_path):
    encoder = finished(fasttext=True)
    folder = encoder.save_pretrained(tmp_path / "enc", license="apache-2.0", language=["en"])
    want = encoder.encode(SENTENCES, workers=1)
    for mmap in (False, True):
        got = fse.Encoder.from_pretrained(folder, mmap=mmap).encode(SENTENCES, workers=1)
        assert got.view(np.uint32).tolist() == want.view(np.uint32).tolist()
    card = (folder / "README.md").read_text(encoding="utf-8")
    assert '- "apache-2.0"' not in card
    assert 'license: "apache-2.0"' in card


def test_a_standalone_head_folder_describes_the_folded_file(tmp_path):
    clf = _classifier()
    folder = clf.save_pretrained(tmp_path / "clf", standalone=True)
    config = json.loads((folder / "config.json").read_bytes())
    written = _fingerprint_of_file(folder / "model.safetensors")
    assert config["fingerprint"] == written != clf.fingerprint
    assert "standalone: the head is folded into the table" in (folder / "README.md").read_text(
        encoding="utf-8"
    )
    loaded = fse.Classifier.from_pretrained(folder)
    assert loaded.standalone
    assert loaded.fingerprint == written
    assert (
        loaded.predict(SENTENCES, workers=1).tolist() == clf.predict(SENTENCES, workers=1).tolist()
    )
    for mmap in (False, True):
        assert fse.Classifier.from_pretrained(folder, mmap=mmap).fingerprint == written
    with pytest.raises(ValueError, match="does not fold into the table"):
        _classifier(normalize=True).save_pretrained(tmp_path / "n", standalone=True)
    assert not (tmp_path / "n").exists()
    assert _leftovers(tmp_path) == []


def test_a_regressor_folder_round_trips(tmp_path):
    encoder = finished(normalize=False)
    reg = fse.Regressor.from_arrays(encoder, np.ones((2, 6), F32), np.zeros(2, F32))
    folder = reg.save_pretrained(tmp_path / "reg")
    got = fse.Regressor.from_pretrained(folder).predict(SENTENCES, workers=1)
    assert got.tobytes() == reg.predict(SENTENCES, workers=1).tobytes()


@pytest.mark.parametrize("mmap", [False, True])
def test_a_file_swapped_after_the_config_was_read_is_caught(mmap, tmp_path, monkeypatch):
    encoder, other = finished(seed=0), finished(seed=1)
    folder = encoder.save_pretrained(tmp_path / "enc")
    other.save(tmp_path / "other.safetensors")
    real = _pretrained.fse_model

    def swap_after_the_check(*args, **kwargs):
        found = real(*args, **kwargs)
        os.replace(tmp_path / "other.safetensors", folder / "model.safetensors")
        return found

    monkeypatch.setattr(_pretrained, "fse_model", swap_after_the_check)
    with pytest.raises(CorruptArtifactError, match="do not belong together"):
        fse.Encoder.from_pretrained(folder, mmap=mmap)


def test_a_head_file_swapped_after_the_config_was_read_is_caught(tmp_path, monkeypatch):
    clf = _classifier()
    folder = clf.save_pretrained(tmp_path / "clf")
    _classifier(normalize=True).save(tmp_path / "other.safetensors")
    real = _pretrained.fse_model

    def swap_after_the_check(*args, **kwargs):
        found = real(*args, **kwargs)
        os.replace(tmp_path / "other.safetensors", folder / "model.safetensors")
        return found

    monkeypatch.setattr(_pretrained, "fse_model", swap_after_the_check)
    with pytest.raises(CorruptArtifactError, match="do not belong together"):
        fse.Classifier.from_pretrained(folder)


def test_heads_are_read_only_from_fse_models_of_their_task(tmp_path):
    encoder = finished(normalize=False)
    reg = fse.Regressor.from_arrays(encoder, np.ones(6, F32)).save_pretrained(tmp_path / "reg")
    with pytest.raises(
        ValueError, match=r"open it with fse\.Regressor\.load or fse\.Regressor\.from"
    ):
        fse.Classifier.from_pretrained(reg)
    enc = encoder.save_pretrained(tmp_path / "enc")
    with pytest.raises(
        UnsupportedArtifactError, match=r"open it with fse\.Encoder\.from_pretrained"
    ):
        fse.Classifier.from_pretrained(enc)
    with pytest.raises(UnsupportedArtifactError, match="no fse folder"):
        fse.Regressor.from_pretrained(mv.save_model(tmp_path / "m2v"))
    _classifier().save(tmp_path / "c.safetensors")
    assert fse.Classifier.from_pretrained(tmp_path / "c.safetensors").n_outputs == 3
    with pytest.raises(ValueError, match="revision, token, cache_dir and local_files_only"):
        fse.Classifier.from_pretrained(tmp_path / "c.safetensors", revision="main")


@pytest.mark.parametrize("kind", ["empty", "full", "file", "link"])
def test_an_existing_path_is_never_written_into(kind, tmp_path):
    path = tmp_path / "x"
    if kind in ("empty", "full"):
        path.mkdir()
        if kind == "full":
            (path / "a").write_text("a")
    elif kind == "file":
        path.write_text("a")
    else:
        path.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError, match="never written into or replaced"):
        finished().save_pretrained(path)
    assert _leftovers(tmp_path) == []
    with pytest.raises(FileNotFoundError, match="is no directory"):
        finished().save_pretrained(tmp_path / "no" / "x")


def test_extra_files_are_checked_and_copied_as_read(tmp_path):
    (tmp_path / "LICENSE.txt").write_bytes(b"MIT License\n")
    (tmp_path / "NOTICE").write_bytes("Notice é\r\n".encode())  # bytes, copied as they are
    extra = {"LICENSE": tmp_path / "LICENSE.txt", "NOTICE": tmp_path / "NOTICE"}
    folder = finished().save_pretrained(tmp_path / "enc", extra_files=extra)
    assert (folder / "LICENSE").read_bytes() == b"MIT License\n"
    assert (folder / "NOTICE").read_bytes() == "Notice é\r\n".encode()
    (tmp_path / "bin").write_bytes(b"\xff\xfe")
    (tmp_path / "big").write_bytes(b"a" * (2**20 + 1))
    (tmp_path / "link").symlink_to(tmp_path / "NOTICE")
    cases = [
        ({"a/b": tmp_path / "NOTICE"}, ValueError, "no plain file name"),
        ({"x*": tmp_path / "NOTICE"}, ValueError, "no plain file name"),
        ({".gitattributes": tmp_path / "NOTICE"}, ValueError, "no plain file name"),
        ({"config.json": tmp_path / "NOTICE"}, ValueError, "a file of the model itself"),
        ({"BIN": tmp_path / "bin"}, ValueError, "no UTF-8 text"),
        ({"BIG": tmp_path / "big"}, CorruptArtifactError, "more than"),
        ({"LINK": tmp_path / "link"}, CorruptArtifactError, "not a regular file"),
    ]
    for files, error, match in cases:
        with pytest.raises(error, match=match):
            finished().save_pretrained(tmp_path / "bad", extra_files=files)
        assert not (tmp_path / "bad").exists()
    assert _leftovers(tmp_path) == []


def _fail(*_args, **_kwargs):
    raise OSError("injected")


@pytest.mark.parametrize("where", ["model", "card", "config", "rename", "interrupt"])
def test_a_failed_write_leaves_nothing(where, tmp_path, monkeypatch):
    encoder = finished()
    if where == "model":
        monkeypatch.setattr(fse.Encoder, "save", _fail)
    elif where == "card":
        monkeypatch.setattr(_card, "card", _fail)
    elif where == "config":
        real = _folder._write_new
        monkeypatch.setattr(
            _folder, "_write_new", lambda p, d: _fail() if p.name == "README.md" else real(p, d)
        )
    elif where == "rename":
        monkeypatch.setattr(_npy_inplace, "_rename_exclusive", _fail)
    else:

        def interrupt(*_args, **_kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(_card, "card", interrupt)
    with pytest.raises((OSError, KeyboardInterrupt)):
        encoder.save_pretrained(tmp_path / "enc")
    assert sorted(os.listdir(tmp_path)) == []


def test_two_writers_of_one_path_give_one_folder(tmp_path, monkeypatch):
    meet, real = threading.Barrier(2, timeout=10), _folder.publish_directory

    def publish(tmp, path):
        meet.wait()
        real(tmp, path)

    monkeypatch.setattr(_folder, "publish_directory", publish)
    errors = []

    def save(seed):
        try:
            finished(seed=seed).save_pretrained(tmp_path / "enc")
        except FileExistsError as error:
            errors.append(error)

    threads = [threading.Thread(target=save, args=(seed,)) for seed in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(errors) == 1
    assert "another writer published it first" in str(errors[0])
    assert sorted(os.listdir(tmp_path)) == ["enc"]
    fse.Encoder.from_pretrained(tmp_path / "enc")


def test_a_failed_sync_after_the_rename_keeps_the_complete_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(_npy_inplace, "_sync_directory", _fail)
    encoder = finished()
    with pytest.raises(PublishedNotDurableError, match="is published, but syncing"):
        encoder.save_pretrained(tmp_path / "enc")
    monkeypatch.undo()
    assert sorted(os.listdir(tmp_path)) == ["enc"]
    assert fse.Encoder.from_pretrained(tmp_path / "enc").fingerprint == encoder.fingerprint


def test_the_config_schema_is_strict():
    where = Path("config.json")
    config = {
        "fingerprint": "xxh3-128:" + "0" * 32,
        "fse_config_version": 1,
        "fse_format_version": 1,
        "fse_kind": "encoder",
        "model": "model.safetensors",
        "model_type": "fse",
    }
    assert _folder.parse_config(dict(config), where) == config
    with pytest.raises(UnsupportedArtifactError, match="lacks \\['model'\\]"):
        _folder.parse_config({k: v for k, v in config.items() if k != "model"}, where)
    with pytest.raises(UnsupportedArtifactError, match="format version True"):
        _folder.parse_config({**config, "fse_format_version": True}, where)
    with pytest.raises(
        CorruptArtifactError, match="names the kind encoder, the file holds a linear"
    ):
        _folder.check_loaded(config, config["fingerprint"], "linear_head", where)


@pytest.mark.parametrize("name", ["", "  ", "two\nlines"])
def test_an_empty_model_name_is_refused_not_replaced(name, tmp_path):
    with pytest.raises(ValueError, match="model_name must be"):
        finished().save_pretrained(tmp_path / "enc", model_name=name)
    with pytest.raises(ValueError, match="model_name must be"):
        _classifier().save_pretrained(tmp_path / "clf", model_name=name)
    assert sorted(os.listdir(tmp_path)) == []
    folder = finished().save_pretrained(tmp_path / "enc")  # None: the folder's name
    assert "\n# enc\n" in (folder / "README.md").read_text(encoding="utf-8")
