"""Published head names put the encoder first and keep both head roles explicit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hub.heads.aegis_cards import render


@pytest.mark.parametrize(
    ("encoder", "head", "expected_id"),
    [
        ("bekko", "safety", "fse/bekko-aegis2-prompt-safety-classifier"),
        ("bekko", "categories", "fse/bekko-aegis2-prompt-categories-classifier"),
        ("potion", "safety", "fse/potion-8m-aegis2-prompt-safety-classifier"),
        ("potion", "categories", "fse/potion-8m-aegis2-prompt-categories-classifier"),
    ],
)
def test_head_names_put_the_encoder_first(encoder, head, expected_id):
    receipt = Path(__file__).resolve().parents[1] / "benchmarks/baseline/dr318/metrics.json"
    record = json.loads(receipt.read_text())
    record["selection"]["encoder"] = encoder
    card = render(record, head)
    assert f"\n# {expected_id}\n" in card
    assert [line for line in card.splitlines() if line.startswith("# ")] == [f"# {expected_id}"]
    assert "Both complete heads use the same encoder" in card
    assert "/path/to/" not in card
    assert "has not been uploaded" not in card
    prefix = "bekko" if encoder == "bekko" else "potion-8m"
    for role in ("safety", "categories"):
        assert f'from_pretrained("fse/{prefix}-aegis2-prompt-{role}-classifier")' in card
    assert 'heads.predict(["How do I bake sourdough bread?"], output="proba")' in card


@pytest.mark.parametrize("head", ["safety", "categories"])
def test_card_version_notice_matches_verified_head_compatibility(head):
    receipt = Path(__file__).resolve().parents[1] / "benchmarks/baseline/dr318/metrics.json"
    record = json.loads(receipt.read_text())
    assert record["compatibility"]["safety_probability_bits"]
    assert record["compatibility"]["safety_predictions"]
    assert record["compatibility"]["categories_unsupported"]
    card = render(record, head)
    if head == "safety":
        assert "This safety artifact loads with fse 2.0.0." in card
        assert "fse 2.0.0 cannot load this artifact." not in card
    else:
        assert (
            "Requires an fse master build with multilabel support, or fse 2.1 when released."
            in card
        )
        assert "fse 2.0.0 cannot load this artifact." in card
    assert "The example groups both heads and requires an fse master build" in card
