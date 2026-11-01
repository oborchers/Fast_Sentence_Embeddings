"""A card/report must describe one complete source family and one corpus decision."""

import importlib
import json
import os
import stat
from pathlib import Path

import pytest

from support.hub_classics import delivery_fixture

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"
pytestmark = pytest.mark.skipif(
    not (BENCHMARKS / "hub_classic_delivery.py").exists(), reason="benchmarks absent"
)


@pytest.fixture
def family(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    return delivery_fixture(monkeypatch, tmp_path)


def test_all_six_models_are_bound_to_the_same_fit_and_selection(family):
    delivery = importlib.import_module("hub_classic_delivery")
    models, records, _, _, selection = family
    actual, _, _ = delivery.validated_source(models, "glove-wiki-gigaword-300", selection)
    assert {r["repo"] for r in actual} == {r["repo"] for r in records}
    selection.write_text(selection.read_text() + " ")
    with pytest.raises(ValueError, match="does not reference this corpus selection"):
        delivery.validated_source(models, "glove-wiki-gigaword-300", selection)


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "nan", "benchmark", "difference", "pairs"]
)
def test_invalid_or_incompatible_verification_cannot_reach_cards(family, mutation):
    checks = importlib.import_module("hub_classic_verification")
    _, _, verification, _, _ = family
    row = verification["rows"][0]
    if mutation == "missing":
        verification["rows"].pop()
    elif mutation == "duplicate":
        verification["rows"][1] = dict(row)
    elif mutation == "nan":
        row["pearson"] = float("nan")
    elif mutation == "benchmark":
        row["benchmark"] = {}
    elif mutation == "difference":
        row["difference_1x_points"] = 1.0
    else:
        row["pairs"] = 1
    with pytest.raises(ValueError, match=r"verification|finite|identity mismatch|does not match"):
        checks.verification_rows(verification, "glove-wiki-gigaword-300")


def test_untouched_template_cards_are_not_complete_delivery_evidence(family):
    delivery = importlib.import_module("hub_classic_delivery")
    models, _, _, _, selection = family
    with pytest.raises(ValueError, match="card enrichment is missing"):
        delivery.validated_source(models, "glove-wiki-gigaword-300", selection, enriched=True)


@pytest.mark.parametrize("step", [1, 2, 3, 4])
def test_interrupted_card_publication_can_recover_without_changing_model_bytes(
    family, monkeypatch, tmp_path, step
):
    publisher = importlib.import_module("hub_classic_card_publish")
    evidence = importlib.import_module("hub_classic_evidence")
    models, records, _, fit, _ = family
    record = next(r for r in records if r["method"] == "sif10" and r["dtype"] == "float32")
    folder = models / record["repo"].split("/")[1]
    record_path = models / f"{folder.name}.json"
    cards = tmp_path / "cards"
    original = publisher._replace
    calls = 0

    def interrupted(source, target, transaction_id):
        nonlocal calls
        original(source, target, transaction_id)
        calls += 1
        if calls == step:
            raise OSError("injected publication interruption")

    monkeypatch.setattr(publisher, "_replace", interrupted)
    with pytest.raises(OSError, match="injected publication interruption"):
        publisher.publish_card(folder, record_path, cards, "derived card\n", fit)
    assert publisher.journal_path(folder).exists()
    monkeypatch.setattr(publisher, "_replace", original)
    publisher.recover_card(folder, record_path, cards)
    after = json.loads(record_path.read_text())
    assert (
        evidence.digest(folder / "model.safetensors")
        == record["files"]["model.safetensors"]["xxh3"]
    )
    assert (cards / f"{folder.name}.md").read_bytes() == (folder / "README.md").read_bytes()
    assert json.loads((folder / "preparation.json").read_text()) == fit
    assert after["files"]["README.md"]["xxh3"] == evidence.digest(folder / "README.md")
    assert not publisher.journal_path(folder).exists()
    assert not list(models.glob(f".{folder.name}.card-stage-*"))


def test_publication_sync_uses_writable_handles_and_exact_utf8_lf(family, monkeypatch, tmp_path):
    publisher = importlib.import_module("hub_classic_card_publish")
    models, records, _, fit, _ = family
    record = next(r for r in records if r["method"] == "sif10" and r["dtype"] == "float32")
    folder = models / record["repo"].split("/")[1]
    model_before = (folder / "model.safetensors").read_bytes()
    original = os.fsync

    def writable_sync(descriptor):
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            # Windows rejects fsync on read-only descriptors. Test that requirement on POSIX too.
            os.write(descriptor, b"")
        original(descriptor)

    monkeypatch.setattr(publisher.os, "fsync", writable_sync)
    text = "derived café card\nsecond line\n"
    publisher.publish_card(folder, models / f"{folder.name}.json", tmp_path / "cards", text, fit)
    assert (folder / "README.md").read_bytes() == text.encode("utf-8")
    assert (folder / "model.safetensors").read_bytes() == model_before
    assert not publisher.journal_path(folder).exists()
