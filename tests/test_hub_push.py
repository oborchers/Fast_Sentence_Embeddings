"""push_to_hub: one create_commit with exact operations on a checked parent commit, and the
check of every file at the commit it returns.

huggingface_hub is replaced by a fake HfApi that keeps the files of each commit; the
operations are huggingface_hub's own classes, so their checks run. Nothing reaches the
network: HF_HUB_OFFLINE is set and the token is removed for every test.
"""

import hashlib
import inspect
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import fse
import fse._hub as hub
from fse import _hub_push
from fse._container import CorruptArtifactError
from support.encoder import finished

huggingface_hub = pytest.importorskip("huggingface_hub")
REPO = "acme/model"
TOKEN = "hf_secret_token"


def _sha1(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class FakeApi:
    """create_repo, model_info and create_commit of one repository, commit by commit."""

    def __init__(self):
        self.commits = {"0" * 40: {".gitattributes": b"*.safetensors filter=lfs\n"}}
        self.head = "0" * 40
        self.calls = []
        self.after_commit = None

    def __call__(self, token=None):  # huggingface_hub.HfApi(token=...)
        self.calls.append(("HfApi", token))
        return self

    def create_repo(self, repo_id, *, repo_type, private, exist_ok):
        self.calls.append(("create_repo", repo_id, repo_type, private, exist_ok))

    def model_info(self, repo_id, *, revision, files_metadata):
        assert repo_id == REPO
        assert files_metadata
        commit = revision if revision in self.commits else self.head  # a commit, or a branch
        files = self.commits[commit]
        siblings = [
            SimpleNamespace(
                rfilename=name,
                lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest(), size=len(data))
                if name.endswith(".safetensors")
                else None,
                blob_id=_sha1(b"pointer " + data) if name.endswith(".safetensors") else _sha1(data),
            )
            for name, data in files.items()
        ]
        return SimpleNamespace(sha=commit, siblings=siblings)

    def create_commit(self, repo_id, operations, **options):
        self.calls.append(("create_commit", repo_id, list(operations), options))
        if options["parent_commit"] != self.head:
            raise RuntimeError("the branch moved since the parent commit")
        files = dict(self.commits[self.head])
        for op in operations:
            if isinstance(op, huggingface_hub.CommitOperationDelete):
                assert op.is_folder is False
                del files[op.path_in_repo]  # exactly this name, no pattern
            else:
                files[op.path_in_repo] = Path(op.path_or_fileobj).read_bytes()
        oid = hashlib.sha1(repr(sorted(files.items())).encode()).hexdigest()
        self.commits[oid] = files
        self.head = oid
        if self.after_commit is not None:
            self.after_commit(files)
        url = f"https://huggingface.co/{repo_id}/commit/{oid}"
        pr = "https://huggingface.co/acme/model/discussions/1" if options["create_pr"] else None
        return SimpleNamespace(oid=oid, commit_url=url, pr_url=pr)

    def files(self):
        return self.commits[self.head]

    def commit_calls(self):
        return [c for c in self.calls if c[0] == "create_commit"]


@pytest.fixture
def api(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    fake = FakeApi()
    library = SimpleNamespace(
        HfApi=fake,
        CommitOperationAdd=huggingface_hub.CommitOperationAdd,
        CommitOperationDelete=huggingface_hub.CommitOperationDelete,
        constants=SimpleNamespace(ENDPOINT="https://huggingface.co"),
    )
    monkeypatch.setattr(hub, "_library", lambda: library)
    fake.tmp = tmp_path / "tmp"
    return fake


def _push(model, **options):
    return model.push_to_hub(REPO, **{"license": "mit", "token": TOKEN} | options)


def test_a_push_is_one_commit_of_the_folder_on_its_parent(api):
    encoder = finished()
    url = _push(encoder, private=True, revision="main")
    (call,) = api.commit_calls()
    _, repo, operations, options = call
    assert repo == REPO
    assert sorted(op.path_in_repo for op in operations) == [
        "README.md",
        "config.json",
        "model.safetensors",
    ]
    assert all(isinstance(op, huggingface_hub.CommitOperationAdd) for op in operations)
    assert options == {
        "commit_message": "Upload encoder with fse",
        "commit_description": None,
        "repo_type": "model",
        "revision": "main",
        "create_pr": False,
        "parent_commit": "0" * 40,
    }
    assert ("create_repo", REPO, "model", True, True) in api.calls
    assert ("HfApi", TOKEN) in api.calls
    assert url == f"https://huggingface.co/{REPO}/commit/{api.head}"
    assert sorted(api.files()) == [
        ".gitattributes",
        "README.md",
        "config.json",
        "model.safetensors",
    ]
    loaded = fse.Encoder.from_pretrained(_pushed(api))
    assert loaded.fingerprint == encoder.fingerprint
    assert list(api.tmp.iterdir()) == []  # the temporary folder is gone


def _pushed(api):
    """The files of the head commit, as a local folder."""
    folder = Path(tempfile.mkdtemp(dir=api.tmp.parent))
    for name, data in api.files().items():
        (folder / name).write_bytes(data)
    return folder


def test_a_model_without_a_license_is_refused_before_the_hub_is_asked(api):
    with pytest.raises(ValueError, match="records no license; pass license="):
        finished().push_to_hub(REPO, token=TOKEN)
    assert api.calls == []
    assert list(api.tmp.iterdir()) == []
    with pytest.raises(ValueError, match="no Hub repository id"):
        finished().push_to_hub("a/b/c", license="mit")


def test_options_pass_through_and_a_pull_request_returns_its_url(api):
    url = _push(finished(), create_pr=True, commit_message="First", revision="dev")
    (_, _, _, options) = api.commit_calls()[0]
    assert options["create_pr"] is True
    assert options["revision"] == "dev"
    assert options["commit_message"] == "First"
    assert url == "https://huggingface.co/acme/model/discussions/1"


def test_the_same_content_makes_no_commit(api):
    encoder = finished()
    first = _push(encoder)
    assert _push(encoder) == first
    assert len(api.commit_calls()) == 1


def test_other_content_under_the_same_names_needs_replace(api):
    _push(finished(seed=0))
    with pytest.raises(ValueError, match=r"other content under \['README.md', 'config.json'"):
        _push(finished(seed=1))
    assert len(api.commit_calls()) == 1
    _push(finished(seed=1), replace=True)
    (_, _, operations, options) = api.commit_calls()[1]
    assert sorted(op.path_in_repo for op in operations) == [
        "README.md",
        "config.json",
        "model.safetensors",
    ]
    assert "Replaces model.safetensors." in options["commit_description"]
    pushed = fse.Encoder.from_pretrained(_pushed(api))
    assert pushed.fingerprint == finished(seed=1).fingerprint


def test_other_files_are_deleted_by_their_exact_names_and_licenses_are_kept(api):
    files = api.commits["0" * 40]
    for name in ("onnx/model.onnx", "w*", "w1", "[LN]x", "Lx"):
        files[name] = b"x"
    files["LICENSE"] = b"MIT License, kept\n"
    files["NOTICE.md"] = b"notice\n"
    with pytest.raises(ValueError, match="files of another model"):
        _push(finished())
    _push(finished(), replace=True)
    (_, _, operations, options) = api.commit_calls()[0]
    deletes = [op for op in operations if isinstance(op, huggingface_hub.CommitOperationDelete)]
    assert sorted(op.path_in_repo for op in deletes) == [
        "Lx",
        "[LN]x",
        "onnx/model.onnx",
        "w*",
        "w1",
    ]
    assert all(op.is_folder is False for op in deletes)
    assert "Removes w*." in options["commit_description"]
    assert api.files()["LICENSE"] == b"MIT License, kept\n"
    assert "NOTICE.md" in api.files()
    assert "onnx/model.onnx" not in api.files()


def test_a_license_file_is_replaced_only_when_passed_and_replaced(api, tmp_path):
    api.commits["0" * 40]["LICENSE"] = b"old\n"
    (tmp_path / "LICENSE").write_bytes(b"new\n")
    with pytest.raises(ValueError, match=r"other content under \['LICENSE'\]"):
        _push(finished(), extra_files={"LICENSE": tmp_path / "LICENSE"})
    _push(finished(), extra_files={"LICENSE": tmp_path / "LICENSE"}, replace=True)
    assert api.files()["LICENSE"] == b"new\n"


def test_a_branch_that_moved_meanwhile_makes_no_commit(api):
    real = api.model_info

    def moved(*args, **kwargs):
        info = real(*args, **kwargs)
        api.head = "f" * 40
        api.commits[api.head] = dict(api.commits["0" * 40])
        return info

    api.model_info = moved
    with pytest.raises(RuntimeError, match="the branch moved"):
        _push(finished())
    assert api.head == "f" * 40
    assert list(api.tmp.iterdir()) == []


@pytest.mark.parametrize("change", ["missing", "lfs", "no-lfs", "blob", "extra"])
def test_every_file_is_checked_at_the_returned_commit(api, change):
    def tamper(files):
        if change == "missing":
            del files["README.md"]
        elif change == "lfs":
            files["model.safetensors"] += b"x"
        elif change == "blob":
            files["config.json"] = b"{}"
        elif change == "extra":
            files["modules.json"] = b"[]"

    api.after_commit = tamper
    if change == "no-lfs":
        real = api.model_info

        def without_lfs(*args, **kwargs):
            info = real(*args, **kwargs)
            for s in info.siblings:
                s.lfs = None
            return info

        api.after_commit = lambda _files: setattr(api, "model_info", without_lfs)
    match = {
        "missing": r"lacks \['README.md'\]",
        "lfs": "model.safetensors of commit .* differs",
        "no-lfs": "no LFS file with an lfs.sha256",
        "blob": "config.json of commit .* differs",
        "extra": r"holds \['modules.json'\]",
    }[change]
    with pytest.raises(CorruptArtifactError, match=match) as raised:
        _push(finished())
    assert "the commit stays as it is" in str(raised.value)
    assert TOKEN not in str(raised.value)


def test_heads_are_pushed_complete_or_standalone(api):
    encoder = finished(normalize=False)
    clf = fse.Classifier.from_arrays(encoder, np.ones((2, 6)), labels=["a", "b"])
    _push(clf, standalone=True)
    (_, _, _, options) = api.commit_calls()[0]
    assert options["commit_message"] == "Upload classifier with fse"
    pushed = fse.Classifier.from_pretrained(_pushed(api))
    assert pushed.standalone
    reg = fse.Regressor.from_arrays(encoder, np.ones(6))
    _push(reg, replace=True)
    assert api.commit_calls()[1][3]["commit_message"] == "Upload regressor with fse"


def test_the_installed_huggingface_hub_takes_every_argument_fse_passes():
    """The mock cannot see an API change; the signatures of the installed version can (the
    minimum and the latest job run this)."""
    api = huggingface_hub.HfApi
    want = {
        "create_repo": {"repo_id", "repo_type", "private", "exist_ok"},
        "model_info": {"repo_id", "revision", "files_metadata"},
        "create_commit": {
            "repo_id",
            "operations",
            "commit_message",
            "commit_description",
            "repo_type",
            "revision",
            "create_pr",
            "parent_commit",
        },
    }
    for method, names in want.items():
        assert names <= set(inspect.signature(getattr(api, method)).parameters), method
    assert "is_folder" in inspect.signature(huggingface_hub.CommitOperationDelete).parameters
    assert {"path_in_repo", "path_or_fileobj"} <= set(
        inspect.signature(huggingface_hub.CommitOperationAdd).parameters
    )
    assert isinstance(huggingface_hub.constants.ENDPOINT, str)


def test_the_plan_of_a_push():
    local = {"a": ("s1", "b1"), "b": ("s2", "b2")}
    remote = {"a": (None, "b1"), ".gitattributes": (None, "g"), "COPYING": (None, "c")}
    assert _hub_push.plan(local, remote, False) == (["b"], [], {".gitattributes", "COPYING"})
    assert _hub_push.plan(local, {"a": ("s1", "x")}, False) == (["b"], [], set())
    with pytest.raises(ValueError, match=r"other content under \['a'\]"):
        _hub_push.plan(local, {"a": ("other", "x")}, False)
