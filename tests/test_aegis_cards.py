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
