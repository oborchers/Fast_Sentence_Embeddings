"""A zero-exit emulator dry run cannot satisfy the conformance execution gate."""

import runpy
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/check_sde.py"))
REQUIRE = MODULE["require_execution"]


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
    if failed:
        ET.SubElement(ET.SubElement(suite, "testcase", name="other"), "failure")
    ET.ElementTree(root).write(path)


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
    "name", [*MODULE["REQUIRED"], "test_avx512_selection_reaches_every_vector_dispatch_site"]
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
