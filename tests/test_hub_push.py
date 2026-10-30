"""push_to_hub: one create_commit with exact operations on a checked parent commit, and the
check of every file at the commit it returns.

huggingface_hub is replaced by a fake HfApi that keeps the files of each commit; the
operations are huggingface_hub's own classes, so their checks run. Nothing reaches the
network: HF_HUB_OFFLINE is set and the token is removed for every test.
"""

import fnmatch
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
    """create_repo, model_info and create_commit of one repository, commit by commit.

    How a file is stored is state of the commit, not a rule of its name: ``lfs[commit]``
    holds the names stored as LFS files. A commit stores what it adds as an LFS file when a
    ``filter=lfs`` rule of the .gitattributes of its parent matches it, drops an addition a
    rule of its .gitignore matches (as huggingface_hub does, without an error), and keeps
    how the other files were stored.
    A pull request (``create_pr``) is a commit on top of ``parent_commit`` under a ref of
    its own; the branch does not move, as the Hub does it."""

    def __init__(self):
        self.commits = {"0" * 40: {".gitattributes": b"*.safetensors filter=lfs\n"}}
        self.lfs = {"0" * 40: set()}
        self.head = "0" * 40
        self.prs = []
        self.calls = []
        self.after_commit = None

    def __call__(self, token=None):  # huggingface_hub.HfApi(token=...)
        self.calls.append(("HfApi", token))
        return self

    def create_repo(self, repo_id, *, repo_type, private, exist_ok):
        self.calls.append(("create_repo", repo_id, repo_type, private, exist_ok))

    def put(self, name, data, *, lfs=False):
        """A file of the first commit, stored as an LFS file or as a plain git blob."""
        self.commits["0" * 40][name] = data
        if lfs:
            self.lfs["0" * 40].add(name)

    def model_info(self, repo_id, *, revision, files_metadata):
        assert repo_id == REPO
        assert files_metadata
        commit = revision if revision in self.commits else self.head  # a commit, or a branch
        siblings = []
        for name, data in self.commits[commit].items():
            if name in self.lfs[commit]:
                lfs = SimpleNamespace(sha256=hashlib.sha256(data).hexdigest(), size=len(data))
                siblings.append(
                    SimpleNamespace(rfilename=name, lfs=lfs, blob_id=_sha1(b"p" + data))
                )
            else:
                siblings.append(SimpleNamespace(rfilename=name, lfs=None, blob_id=_sha1(data)))
        return SimpleNamespace(sha=commit, siblings=siblings)

    def create_commit(self, repo_id, operations, **options):
        self.calls.append(("create_commit", repo_id, list(operations), options))
        parent = options["parent_commit"]
        if options["create_pr"]:
            assert parent in self.commits  # a pull request starts from the parent commit
        elif parent != self.head:
            raise RuntimeError("the branch moved since the parent commit")
        files, lfs = dict(self.commits[parent]), set(self.lfs[parent])
        for op in operations:
            if isinstance(op, huggingface_hub.CommitOperationDelete):
                assert op.is_folder is False
                del files[op.path_in_repo]  # exactly this name, no pattern
                lfs.discard(op.path_in_repo)
            elif not _matches(files.get(".gitignore", b""), op.path_in_repo, None):
                files[op.path_in_repo] = Path(op.path_or_fileobj).read_bytes()
                if _matches(files.get(".gitattributes", b""), op.path_in_repo, "filter=lfs"):
                    lfs.add(op.path_in_repo)
                else:
                    lfs.discard(op.path_in_repo)
        oid = hashlib.sha1(repr((sorted(files.items()), sorted(lfs))).encode()).hexdigest()
        self.commits[oid], self.lfs[oid] = files, lfs
        if self.after_commit is not None:
            self.after_commit(files)
        url = f"https://huggingface.co/{repo_id}/commit/{oid}"
        if options["create_pr"]:
            self.prs.append(oid)
            number = len(self.prs)
            pr = f"https://huggingface.co/{repo_id}/discussions/{number}"
            return SimpleNamespace(
                oid=oid, commit_url=url, pr_url=pr, pr_revision=f"refs/pr/{number}"
            )
        self.head = oid
        return SimpleNamespace(oid=oid, commit_url=url, pr_url=None, pr_revision=None)

    def download(self, repo_id, name, *, revision, token, cache_dir):
        """hf_hub_download: the file of a commit, into ``cache_dir``."""
        assert repo_id == REPO
        self.calls.append(("download", name, revision, token))
        path = Path(cache_dir) / revision / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.commits[revision][name])
        return str(path)

    def files(self):
        return self.commits[self.head]

    def commit_calls(self):
        return [c for c in self.calls if c[0] == "create_commit"]


def _matches(rules: bytes, name: str, attribute: str | None) -> bool:
    """Whether a line of .gitattributes (with ``attribute``) or .gitignore matches."""
    for line in rules.decode().splitlines():
        parts = line.split()
        if (
            parts
            and fnmatch.fnmatchcase(name, parts[0])
            and (attribute is None or attribute in parts[1:])
        ):
            return True
    return False


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
        hf_hub_download=fake.download,
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


def test_a_pull_request_starts_from_the_parent_and_leaves_the_branch(api):
    first = _push(finished(seed=0))
    branch = api.head
    real = api.model_info

    def moved(*args, **kwargs):  # the branch moves after the push read it
        info = real(*args, **kwargs)
        api.head = "f" * 40
        api.commits[api.head], api.lfs[api.head] = dict(api.commits[branch]), set()
        return info

    api.model_info = moved
    url = _push(finished(seed=1), create_pr=True, replace=True)
    api.model_info = real
    (pr,) = api.prs
    assert url == "https://huggingface.co/acme/model/discussions/1"
    assert api.head == "f" * 40  # the branch is as the other writer left it
    assert api.commit_calls()[-1][3]["parent_commit"] == branch
    assert api.commits[pr]["model.safetensors"] != api.commits[branch]["model.safetensors"]
    assert "model.safetensors" in api.lfs[pr]
    assert first.endswith(branch)


def test_a_model_file_stored_as_a_plain_blob_is_never_taken_as_pushed(api):
    encoder = finished()
    _push(encoder)
    api.lfs[api.head].discard("model.safetensors")  # the same bytes, but no LFS file
    with pytest.raises(ValueError, match=r"\['model.safetensors'\] stored as plain git blobs"):
        _push(encoder)
    assert len(api.commit_calls()) == 1
    _push(encoder, replace=True)
    assert len(api.commit_calls()) == 2
    assert "model.safetensors" in api.lfs[api.head]


def test_an_unchanged_commit_is_checked_too():
    sha256, sha1 = "a" * 64, "b" * 40
    with pytest.raises(CorruptArtifactError, match=r"no LFS file with an lfs\.sha256"):
        _hub_push.check_files(
            {"model.safetensors": (None, sha1)}, {"model.safetensors": (sha256, sha1)}, "c"
        )
    _hub_push.check_files(
        {"model.safetensors": (sha256, "x")}, {"model.safetensors": (sha256, sha1)}, "c"
    )


def _snapshot(api):
    return api.head, {k: dict(v) for k, v in api.commits.items()}


@pytest.mark.parametrize(
    ("replace", "match"),
    [
        (False, r"files of another model \['\.gitignore', 'old\.safetensors'\]"),
        (True, r"has a \.gitignore, by which the Hub may skip"),  # it would delete it, too late
    ],
)
def test_a_gitignore_in_the_repository_stops_the_push_before_a_commit(api, replace, match):
    api.put(".gitignore", b"*.safetensors\n")
    api.put("old.safetensors", b"old weights", lfs=True)
    before = _snapshot(api)
    with pytest.raises(ValueError, match=match):
        _push(finished(), replace=replace)
    assert api.commit_calls() == []
    assert _snapshot(api) == before  # the branch and every commit as they were


@pytest.mark.parametrize(
    "rules",
    [
        b"*.bin filter=lfs diff=lfs merge=lfs -text\n",
        b"*.safetensors filter=lfs diff=lfs merge=lfs -text\nmodel.safetensors -filter\n",
        None,  # no .gitattributes at all
    ],
)
def test_rules_that_would_store_the_model_as_a_plain_blob_stop_the_push(api, rules):
    if rules is None:
        del api.commits["0" * 40][".gitattributes"]
    else:
        api.put(".gitattributes", rules)
    before = _snapshot(api)
    with pytest.raises(ValueError, match=r"does not store \['model\.safetensors'\] as LFS files"):
        _push(finished(), replace=True)
    assert api.commit_calls() == []
    assert _snapshot(api) == before


def test_the_lfs_rules_of_a_gitattributes():
    default = "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
    assert _hub_push.stored_as_lfs(default, "model.safetensors")
    assert not _hub_push.stored_as_lfs(default + "model.safetensors -filter\n", "model.safetensors")
    assert not _hub_push.stored_as_lfs(default + "*.safetensors !filter\n", "model.safetensors")
    assert not _hub_push.stored_as_lfs(
        default + "*.safetensors filter=other\n", "model.safetensors"
    )
    assert _hub_push.stored_as_lfs("/model.safetensors filter=lfs\n", "model.safetensors")
    assert _hub_push.stored_as_lfs(
        "# a comment\n\n*.safetensors filter=lfs\n", "a/model.safetensors"
    )
    assert not _hub_push.stored_as_lfs("other/*.safetensors filter=lfs\n", "model.safetensors")
    assert not _hub_push.stored_as_lfs("", "model.safetensors")


def test_the_rules_are_read_at_the_commit_of_the_push(api):
    _push(finished(), token=TOKEN)
    (download,) = [c for c in api.calls if c[0] == "download"]
    assert download == ("download", ".gitattributes", "0" * 40, TOKEN)


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
    for name in ("onnx/model.onnx", "w*", "w1", "[LN]x", "Lx"):
        api.put(name, b"x")
    api.put("LICENSE", b"MIT License, kept\n")
    api.put("NOTICE.md", b"notice\n")
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
    api.put("LICENSE", b"old\n")
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
        api.lfs[api.head] = set()
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
    names = {"repo_id", "filename", "revision", "token", "cache_dir"}
    assert names <= set(inspect.signature(huggingface_hub.hf_hub_download).parameters)
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
