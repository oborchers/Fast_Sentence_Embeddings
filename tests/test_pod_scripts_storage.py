"""Pod scripts preserve SSH access, hard deadlines and full-data disk requirements."""

import importlib
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.syspath_prepend(str(root / "quality"))
    return importlib.import_module("pod_storage")


@pytest.mark.skipif(os.name == "nt", reason="pod shells require a POSIX host")
@pytest.mark.usefixtures("modules")
@pytest.mark.parametrize(("module", "limit"), [("make_mind_pod", 10800), ("make_folded_pod", 9000)])
def test_pod_ssh_precedes_environment_and_is_inside_deadline(module, limit, tmp_path):
    script = importlib.import_module(module).script("a" * 40)
    outer = tmp_path / "pod.sh"
    outer.write_text(script)
    subprocess.run(["bash", "-n", str(outer)], check=True)
    work = script.split("<<'DR151_WORK'\n", 1)[1].split("DR151_WORK\n", 1)[0]
    inner = tmp_path / "work.sh"
    inner.write_text(work)
    subprocess.run(["bash", "-n", str(inner)], check=True)
    assert work.index("run_stage ssh") < work.index("set -euxo pipefail")
    ssh = work.split("<<'DR151_SSH'\n", 1)[1].split("DR151_SSH\n", 1)[0]
    assert "set -x" not in ssh
    assert "printf '%s\\n' \"$PUBLIC_KEY\"" in ssh
    assert "/usr/bin/chmod 600 /root/.ssh/authorized_keys" in ssh
    assert "/usr/sbin/sshd -t\n/usr/sbin/sshd" in ssh
    assert f"started + {limit} - " in script
    assert "status=${PIPESTATUS[0]}" in script
    assert "/dr151-results/bootstrap.log" in script
    assert "stage_start" in script
    assert "stage_end" in script
    assert "/usr/bin/df -h /dr151-cache" in script
    assert "ALL_DONE" in script
    assert "/usr/bin/sleep infinity" in script


@pytest.mark.usefixtures("modules")
def test_mind_cache_and_disk_preflight_precede_downloads():
    script = importlib.import_module("make_mind_pod").script("a" * 40)
    assert "200 GB container disk" in script
    assert "HF_DATASETS_CACHE=/dr151-cache/huggingface/datasets" in script
    assert "HF_HUB_DISABLE_XET=1" in script
    assert script.index("u.free < 140*2**30") < script.index("git clone")
    assert "10800 - 90" in script


def test_disk_budget_fails_clearly_without_sampling(modules, monkeypatch, tmp_path):
    monkeypatch.setattr(modules.shutil, "disk_usage", lambda _: SimpleNamespace(free=60 * 2**30))
    with pytest.raises(RuntimeError, match="200 GB container disk"):
        modules.require_free(tmp_path, 120, "before loading MIND")


@pytest.mark.skipif(os.name == "nt", reason="Windows symlinks can require privileges")
def test_disk_inventory_does_not_double_count_hub_snapshot_symlinks(modules, tmp_path):
    (tmp_path / "blob").write_bytes(b"abc")
    (tmp_path / "snapshot-link").symlink_to(tmp_path / "blob")
    assert modules.tree_bytes(tmp_path) == {"regular_file_bytes": 3, "files": 1}


def test_cleanup_requires_complete_verified_finished_models(modules, tmp_path):
    with pytest.raises(ValueError, match="exclusively created"):
        modules.cleanup_preparation_sources(tmp_path, tmp_path, tmp_path)


def test_cleanup_rejects_reused_or_mismatched_cache(modules, tmp_path):
    import json

    cache = tmp_path / "cache"
    output = tmp_path / "results"
    cache.mkdir()
    output.mkdir()
    owner = {"cache": str(cache.resolve()), "created_exclusively": True, "owner_token": "a" * 32}
    (cache / ".dr151-owner.json").write_text(json.dumps(owner))
    (output / "cache-ownership.json").write_text(json.dumps(owner))
    modules.verify_owned_cache(cache, output)
    (output / "cache-ownership.json").write_text(json.dumps({**owner, "owner_token": "b" * 32}))
    with pytest.raises(ValueError, match="ownership does not match"):
        modules.verify_owned_cache(cache, output)


@pytest.mark.usefixtures("modules")
def test_legacy_generator_delegates_to_revised_mind_script():
    original = importlib.import_module("make_pod_scripts")
    revised = importlib.import_module("make_mind_pod")
    assert original.script("a" * 40, 1) == revised.script("a" * 40)
    assert "if ! /usr/local/bin/python - <<'DR151_CACHE_OWNER'" in revised.script("a" * 40)
