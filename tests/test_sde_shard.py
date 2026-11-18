"""SDE shards prove their complete union and cannot hide missing execution witnesses."""

import copy
import json
import os
import runpy
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / "tools/check_sde.py"))


def pair(profile="spr", compiler="gcc", layout=0):
    names = MODULE["REQUIRED"] | {"test_avx512_selection_reaches_every_vector_dispatch_site"}
    names |= {
        prefix + str(index) + "]"
        for prefix, count in zip(MODULE["CHILD_PREFIXES"], (5, 7), strict=True)
        for index in range(count)
    }
    nodeids = sorted(
        [path + "::test_example" for path in MODULE["TESTS"]]
        + ["tests/test_native_isa.py::" + name for name in names]
    )
    return [
        {
            "compiler": compiler,
            "layout": layout,
            "profile": profile,
            "source_ref": "head",
            "shard": shard,
            "all": nodeids,
            "selected": nodeids[shard::2],
            "cases": [
                {
                    "classname": MODULE["junit_identity"](nodeid)[0],
                    "name": MODULE["junit_identity"](nodeid)[1],
                    "skipped": False,
                }
                for nodeid in nodeids[shard::2]
            ],
        }
        for shard in (0, 1)
    ]


@pytest.mark.parametrize("profile", ["spr", "hsw"])
def test_merge_requires_every_selected_test_and_all_profile_witnesses(profile):
    rows = pair(profile)
    result = MODULE["merge_pair"](rows, "head")
    assert result["tests"] == len(rows[0]["all"])
    assert result["child_imports"] == 12
    assert result["profile"] == profile


@pytest.mark.parametrize("damage", ["missing", "duplicate", "overlap", "source", "collection"])
def test_merge_rejects_incomplete_or_inconsistent_halves(damage):
    rows = pair()
    if damage == "missing":
        rows[0]["cases"].pop()
    elif damage == "duplicate":
        rows[0]["cases"].append(rows[0]["cases"][0])
    elif damage == "overlap":
        rows[1] = copy.deepcopy(rows[0])
    elif damage == "source":
        rows[1]["source_ref"] = "stale"
    else:
        rows[1]["all"] = rows[1]["all"][1:]
    with pytest.raises(RuntimeError):
        MODULE["merge_pair"](rows, "head")


@pytest.mark.parametrize("name", [*sorted(MODULE["REQUIRED"]), MODULE["CHILD_PREFIXES"][0] + "0]"])
def test_merged_required_witnesses_cannot_be_skipped(name):
    rows = pair()
    for row in rows:
        for case in row["cases"]:
            if case["name"] == name:
                case["skipped"] = True
    with pytest.raises(RuntimeError, match=r"execute|child-import"):
        MODULE["merge_pair"](rows, "head")


@pytest.mark.parametrize("damage", [None, "missing", "duplicate", "unexpected"])
def test_complete_gate_requires_exactly_every_compiler_layout_profile_shard(tmp_path, damage):
    rows = [
        row
        for compiler in ("gcc", "clang", "msvc")
        for layout in (0, 1)
        for profile in ("spr", "hsw")
        for row in pair(profile, compiler, layout)
    ]
    if damage == "missing":
        rows.pop()
    elif damage == "duplicate":
        rows.append(rows[0])
    elif damage == "unexpected":
        rows[0]["compiler"] = "other"
    for index, row in enumerate(rows):
        (tmp_path / f"{index}.json").write_text(json.dumps(row))
    if damage:
        with pytest.raises(RuntimeError, match="twenty-four"):
            MODULE["merge_receipts"](tmp_path, "head")
    else:
        assert len(MODULE["merge_receipts"](tmp_path, "head")) == 12


@pytest.mark.parametrize("damage", [None, "missing", "failure"])
def test_shard_checks_actual_junit_against_its_selected_nodes(tmp_path, damage):
    row = pair()[0]
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    for case in row["cases"]:
        element = ET.SubElement(suite, "testcase", classname=case["classname"], name=case["name"])
        if damage == "failure":
            ET.SubElement(element, "failure")
    if damage == "missing":
        suite.remove(suite[-1])
    report = tmp_path / "report.xml"
    ET.ElementTree(root).write(report)
    arguments = (report, row, "spr", "gcc", 0, "head")
    if damage:
        with pytest.raises(RuntimeError, match=r"failed tests|selected node"):
            MODULE["shard_receipt"](*arguments)
    else:
        assert MODULE["shard_receipt"](*arguments) == row


def test_real_pytest_partition_respects_selection_and_preserves_skips(tmp_path):
    test = tmp_path / "test_example.py"
    test.write_text(
        "import pytest\n"
        "@pytest.mark.parametrize('value', range(8))\n"
        "def test_example(value):\n    assert value >= 0\n"
        "@pytest.mark.skip(reason='platform')\n"
        "def test_platform(): pass\n"
        "def test_excluded(): pass\n"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    rows = []
    for shard in (0, 1):
        manifest = tmp_path / f"{shard}.json"
        report = tmp_path / f"{shard}.xml"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "tools.sde_shard",
                str(test),
                "-k",
                "not excluded",
                f"--fse-sde-shard={shard}",
                f"--fse-sde-manifest={manifest}",
                f"--junitxml={report}",
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        row = json.loads(manifest.read_text())
        observed = ET.parse(report).getroot().findall(".//testcase")
        assert sorted((case.get("classname"), case.get("name")) for case in observed) == sorted(
            MODULE["junit_identity"](nodeid) for nodeid in row["selected"]
        )
        rows.append(row)
    assert rows[0]["all"] == rows[1]["all"]
    assert len(rows[0]["all"]) == 9
    assert not set(rows[0]["selected"]) & set(rows[1]["selected"])
    assert sorted(rows[0]["selected"] + rows[1]["selected"]) == rows[0]["all"]


def test_sharded_launcher_requires_execution_before_writing_artifact(monkeypatch, tmp_path):
    run = MODULE["run_profile"]
    row = pair()[1]
    args = SimpleNamespace(
        shard=1, compiler="gcc", layout=0, source_ref="head", receipts=tmp_path / "artifacts"
    )

    def execute(command, *, env, check):
        assert check
        assert command[:4] == ["sde", "-spr", "--", sys.executable]
        assert set(MODULE["TESTS"]).issubset(command)
        assert "tools.sde_shard" in command
        assert "--fse-sde-shard=1" in command
        assert str(ROOT) in env["PYTHONPATH"]
        assert env["FSE_REQUIRE_ISA"] == "avx512"
        (tmp_path / "collection.json").write_text(json.dumps(row))
        root = ET.Element("testsuites")
        suite = ET.SubElement(root, "testsuite")
        for case in row["cases"]:
            ET.SubElement(suite, "testcase", classname=case["classname"], name=case["name"])
        ET.ElementTree(root).write(tmp_path / "spr.xml")

    monkeypatch.setitem(run.__globals__, "emulator_prefix", lambda _profile: ["sde", "-spr", "--"])
    monkeypatch.setitem(run.__globals__, "subprocess", SimpleNamespace(run=execute))
    run(args, "spr", tmp_path)
    artifact = args.receipts / "gcc-0-spr-1.json"
    assert json.loads(artifact.read_text()) == row
