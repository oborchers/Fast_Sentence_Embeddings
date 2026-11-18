"""A zero-exit emulator dry run cannot satisfy the conformance execution gate."""

import runpy
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/check_sde.py"))
REQUIRE = MODULE["require_execution"]


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_chip_check_scope_keeps_the_exact_native_image(monkeypatch, platform):
    prefix = MODULE["emulator_prefix"]
    monkeypatch.setenv("FSE_SDE_EXE", "sde")
    monkeypatch.setitem(prefix.__globals__, "sys", SimpleNamespace(platform=platform))
    util = SimpleNamespace(
        find_spec=lambda _name: SimpleNamespace(origin="/wheel/_native.cp313.pyd")
    )
    monkeypatch.setitem(prefix.__globals__, "importlib", SimpleNamespace(util=util))
    expected = ["sde", "-hsw"]
    if platform == "win32":
        expected += ["-chip-check-image", "_native.cp313.pyd"]
    assert prefix("hsw") == [*expected, "--"]


@pytest.mark.parametrize("origin", [None, "/wheel/_native.py"])
def test_windows_chip_check_cannot_silently_select_no_extension(monkeypatch, origin):
    prefix = MODULE["emulator_prefix"]
    monkeypatch.setenv("FSE_SDE_EXE", "sde")
    monkeypatch.setitem(prefix.__globals__, "sys", SimpleNamespace(platform="win32"))
    util = SimpleNamespace(find_spec=lambda _name: SimpleNamespace(origin=origin))
    monkeypatch.setitem(prefix.__globals__, "importlib", SimpleNamespace(util=util))
    with pytest.raises(RuntimeError, match="native image"):
        prefix("hsw")


def receipt(path, profile="spr", omitted=None, skipped=None, failed=False):
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    names = MODULE["REQUIRED"] | (
        {"test_avx512_selection_reaches_every_vector_dispatch_site"} if profile == "spr" else set()
    )
    names |= {
        prefix + str(index) + "]"
        for prefix, count in zip(MODULE["CHILD_PREFIXES"], (5, 7), strict=True)
        for index in range(count)
    }
    for name in sorted(names - {omitted}):
        case = ET.SubElement(suite, "testcase", name=name, classname="tests.test_native_isa")
        if name == skipped:
            ET.SubElement(case, "skipped")
    if failed:
        case = ET.SubElement(suite, "testcase", name="another_test", classname="tests.other")
        ET.SubElement(case, "failure")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


@pytest.mark.parametrize("profile", ["spr", "hsw"])
def test_requires_real_successful_profile_and_child_execution(tmp_path, profile):
    path = tmp_path / "receipt.xml"
    receipt(path, profile)
    result = REQUIRE(path, profile)
    assert result["profile"] == profile
    assert result["child_imports"] == 12
    assert result["tests"] >= 14


def test_zero_exit_without_execution_report_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="execution receipt"):
        REQUIRE(tmp_path / "never-created.xml", "spr")


@pytest.mark.parametrize(
    "name",
    [*sorted(MODULE["REQUIRED"]), "test_avx512_selection_reaches_every_vector_dispatch_site"],
)
@pytest.mark.parametrize("kind", ["omitted", "skipped"])
def test_required_kernel_and_snapshot_proofs_cannot_be_skipped(tmp_path, name, kind):
    path = tmp_path / "receipt.xml"
    receipt(path, **{kind: name})
    with pytest.raises(RuntimeError, match="did not execute"):
        REQUIRE(path, "spr")


@pytest.mark.parametrize("kind", ["omitted", "skipped"])
def test_child_import_proofs_cannot_be_omitted_or_skipped(tmp_path, kind):
    path = tmp_path / "receipt.xml"
    receipt(path, **{kind: MODULE["CHILD_PREFIXES"][0] + "0]"})
    with pytest.raises(RuntimeError, match="child-import"):
        REQUIRE(path, "spr")


def test_a_failure_elsewhere_still_fails_the_receipt(tmp_path):
    path = tmp_path / "receipt.xml"
    receipt(path, failed=True)
    with pytest.raises(RuntimeError, match="failed tests"):
        REQUIRE(path, "spr")


@pytest.mark.parametrize("profile", ["spr", "hsw"])
def test_selected_profile_runs_the_full_manifest_and_requires_its_receipt(monkeypatch, profile):
    main = MODULE["main"]
    calls = []

    def run(command, *, env, check):
        assert check is True
        assert env["FSE_SDE_PROFILE"] == profile
        assert env["FSE_REQUIRE_ISA"] == ("avx512" if profile == "spr" else "avx2")
        assert set(MODULE["TESTS"]).issubset(command)
        report = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml="))
        receipt(Path(report), profile)
        calls.append(command)

    monkeypatch.setitem(
        main.__globals__,
        "sys",
        SimpleNamespace(argv=["check_sde.py", "--profile", profile], executable="python"),
    )
    monkeypatch.setitem(
        main.__globals__, "emulator_prefix", lambda value: ["sde", f"-{value}", "--"]
    )
    monkeypatch.setitem(main.__globals__, "subprocess", SimpleNamespace(run=run))
    main()
    assert len(calls) == 1
    assert calls[0][:4] == ["sde", f"-{profile}", "--", "python"]
