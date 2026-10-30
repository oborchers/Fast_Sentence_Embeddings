"""The rules of a push that need no Hub: which files a .gitattributes stores as LFS
files, what a push plans to add, delete and keep, and the API of the installed
huggingface_hub that push_to_hub relies on."""

import inspect

import pytest

from fse import _hub_push

huggingface_hub = pytest.importorskip("huggingface_hub")


def test_the_lfs_rules_of_a_gitattributes():
    default = "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
    assert _hub_push.stored_as_lfs(default, "model.safetensors")
    assert not _hub_push.stored_as_lfs(default + "model.safetensors -filter\n", "model.safetensors")
    assert not _hub_push.stored_as_lfs(default + "*.safetensors !filter\n", "model.safetensors")
    assert not _hub_push.stored_as_lfs(
        default + "*.safetensors filter=other\n", "model.safetensors"
    )
    assert _hub_push.stored_as_lfs("/model.safetensors filter=lfs\n", "model.safetensors")
    assert _hub_push.stored_as_lfs("# a comment\n\n*.safetensors filter=lfs\n", "model.safetensors")
    assert _hub_push.stored_as_lfs("**/model.safetensors filter=lfs\n", "model.safetensors")
    assert not _hub_push.stored_as_lfs("other/*.safetensors filter=lfs\n", "model.safetensors")
    assert not _hub_push.stored_as_lfs(default + "**/*.safetensors -filter\n", "model.safetensors")
    assert not _hub_push.stored_as_lfs("", "model.safetensors")
    with pytest.raises(ValueError, match="no file of the root"):
        _hub_push.stored_as_lfs(default, "a/model.safetensors")


# The .gitattributes a new repository of the Hub gets (excerpt, as create_repo writes it).
HUB_DEFAULT = """*.7z filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
saved_model/**/* filter=lfs diff=lfs merge=lfs -text
*tfevents* filter=lfs diff=lfs merge=lfs -text
"""


def test_lines_the_rule_reader_does_not_read_are_named():
    assert _hub_push.unsupported_rules(HUB_DEFAULT) == []
    assert _hub_push.stored_as_lfs(HUB_DEFAULT, "model.safetensors")
    assert not _hub_push.stored_as_lfs(HUB_DEFAULT, "config.json")
    lines = [
        '"model.safetensors" -filter',
        '"*.safetensors" filter=lfs',
        "[attr]weights filter=lfs diff=lfs",
        "!*.safetensors filter",
        "model\\.safetensors -filter",
    ]
    text = HUB_DEFAULT + "# a comment\n" + "\n".join(lines) + "\n"
    assert _hub_push.unsupported_rules(text) == lines


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
