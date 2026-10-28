import os
import sys

import pytest

from fse import _cpus
from fse._cpus import _cgroup_quota, usable_cpus

V2_MOUNTS = "30 25 0:26 / /sys/fs/cgroup rw,nosuid shared:4 - cgroup2 cgroup2 rw,nsdelegate\n"
V1_MOUNTS = (
    "31 25 0:27 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n"
    "32 25 0:28 / /sys/fs/cgroup/cpu,cpuacct rw - cgroup cgroup rw,cpu,cpuacct\n"
)


def _proc(tmp_path, cgroup, mountinfo):
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "cgroup").write_text(cgroup)
    (proc / "mountinfo").write_text(mountinfo)
    return proc


def _write(path, name, text):
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text(text)


@pytest.mark.parametrize(
    ("value", "want"), [("max 100000", None), ("200000 100000", 2), ("150000 100000", 2)]
)
def test_v2_quota_of_the_cgroup(tmp_path, value, want):
    proc = _proc(tmp_path, "0::/app.slice/job\n", V2_MOUNTS)
    _write(tmp_path / "sys/fs/cgroup/app.slice/job", "cpu.max", value)
    assert _cgroup_quota(proc, tmp_path) == want


def test_v2_quota_of_an_ancestor_counts(tmp_path):
    proc = _proc(tmp_path, "0::/app.slice/job\n", V2_MOUNTS)
    _write(tmp_path / "sys/fs/cgroup/app.slice", "cpu.max", "300000 100000")
    _write(tmp_path / "sys/fs/cgroup/app.slice/job", "cpu.max", "max 100000")
    assert _cgroup_quota(proc, tmp_path) == 3
    _write(tmp_path / "sys/fs/cgroup/app.slice/job", "cpu.max", "100000 100000")
    assert _cgroup_quota(proc, tmp_path) == 1


def test_v2_namespace_root(tmp_path):
    """In a container the process sees its cgroup as / at the mount root."""
    proc = _proc(tmp_path, "0::/\n", V2_MOUNTS)
    _write(tmp_path / "sys/fs/cgroup", "cpu.max", "400000 100000")
    assert _cgroup_quota(proc, tmp_path) == 4


def test_v1_quota(tmp_path):
    proc = _proc(tmp_path, "4:memory:/docker/x\n3:cpu,cpuacct:/docker/x\n", V1_MOUNTS)
    directory = tmp_path / "sys/fs/cgroup/cpu,cpuacct/docker/x"
    _write(directory, "cpu.cfs_quota_us", "250000\n")
    _write(directory, "cpu.cfs_period_us", "100000\n")
    assert _cgroup_quota(proc, tmp_path) == 3
    _write(directory, "cpu.cfs_quota_us", "-1\n")
    assert _cgroup_quota(proc, tmp_path) is None


@pytest.mark.parametrize(
    ("cgroup", "mountinfo"),
    [
        ("0::/a\n", ""),  # no cgroup2 mount
        ("0::/../outside\n", V2_MOUNTS),  # not under the mount
        ("garbage\n", V2_MOUNTS),
        ("2:memory:/a\n", V1_MOUNTS),  # no cpu controller
    ],
)
def test_no_quota(tmp_path, cgroup, mountinfo):
    proc = _proc(tmp_path, cgroup, mountinfo)
    assert _cgroup_quota(proc, tmp_path) is None


def test_missing_and_broken_files(tmp_path):
    assert _cgroup_quota(tmp_path / "nothing", tmp_path) is None
    proc = _proc(tmp_path, "0::/job\n", V2_MOUNTS)
    _write(tmp_path / "sys/fs/cgroup/job", "cpu.max", "lots\n")
    assert _cgroup_quota(proc, tmp_path) is None
    _write(tmp_path / "sys/fs/cgroup/job", "cpu.max", "100 0\n")
    assert _cgroup_quota(proc, tmp_path) is None


def test_a_small_quota_rounds_up_to_one(tmp_path):
    proc = _proc(tmp_path, "0::/job\n", V2_MOUNTS)
    _write(tmp_path / "sys/fs/cgroup/job", "cpu.max", "1000 100000")
    assert _cgroup_quota(proc, tmp_path) == 1


def test_usable_cpus_takes_the_smaller_of_affinity_and_quota(monkeypatch):
    monkeypatch.delenv("PYTHON_CPU_COUNT", raising=False)
    monkeypatch.setattr(sys, "_xoptions", {})
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(8)), raising=False)
    monkeypatch.setattr(_cpus, "_process_quota", lambda: 3)
    assert usable_cpus() == 3
    monkeypatch.setattr(_cpus, "_process_quota", lambda: None)
    assert usable_cpus() == 8


@pytest.mark.parametrize("value", ["6", "1"])
def test_the_override_wins(monkeypatch, value):
    monkeypatch.setattr(sys, "_xoptions", {})
    monkeypatch.setenv("PYTHON_CPU_COUNT", value)
    monkeypatch.setattr(_cpus, "_process_quota", lambda: 2)
    assert usable_cpus() == int(value)


def test_the_x_option_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("PYTHON_CPU_COUNT", "6")
    monkeypatch.setattr(sys, "_xoptions", {"cpu_count": "5"})
    assert usable_cpus() == 5


@pytest.mark.parametrize("value", ["0", "-1", "two", "1.5", ""])
def test_an_invalid_override_raises(monkeypatch, value):
    monkeypatch.setattr(sys, "_xoptions", {})
    monkeypatch.setenv("PYTHON_CPU_COUNT", value)
    with pytest.raises(ValueError, match="PYTHON_CPU_COUNT"):
        usable_cpus()


def test_default_means_no_override(monkeypatch):
    monkeypatch.setattr(sys, "_xoptions", {})
    monkeypatch.setenv("PYTHON_CPU_COUNT", "default")
    assert usable_cpus() >= 1


def test_usable_cpus_on_this_machine():
    assert 1 <= usable_cpus() <= (os.cpu_count() or 1) or "PYTHON_CPU_COUNT" in os.environ
