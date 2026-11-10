"""Focused checks for the Aegis deliverable scan, without fitting or model imports."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pytest
from hub.heads.aegis_scan import CONFIG, compact_json, text_scan


class AegisScanTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.work = Path(self.directory.name)
        (self.work / "raw").mkdir()
        for split in ("train", "validation", "test"):
            (self.work / "raw" / (split + ".json")).write_text("[]")

    def train(self, prompt):
        (self.work / "raw/train.json").write_text(json.dumps([{"prompt": prompt}]))

    def test_card_prompt_copy_is_rejected(self):
        example = "An invented amber telescope watches twelve quiet planets."
        self.train(example)
        diagnosis = self.work / "diagnosis.json"
        with pytest.raises(ValueError, match="source prompt substring"):
            text_scan(self.work, {"card.md": example}, diagnosis=diagnosis)
        report = json.loads(diagnosis.read_text())
        assert report["rejected"]
        assert report["rejected"][0]["split"] == "train"
        assert report["rejected"][0]["row_index_zero_based"] == 0

    def test_own_url_is_audited_but_adjacent_prompt_text_is_rejected(self):
        config = json.loads(CONFIG.read_text())
        url = config["sources"]["aart-LICENSE"]["url"]
        self.train(url)
        report = text_scan(self.work, {"licenses.json": json.dumps({"url": url})})
        assert not report["rejected"]
        assert report["ignored"]
        for entry in report["ignored"]:
            assert entry["window_from_deliverable"] in url
            assert entry["row_index_zero_based"] == 0
        adjacent = url + " invented prompt material outside the approved literal"
        self.train(adjacent)
        with pytest.raises(ValueError, match="source prompt substring"):
            text_scan(self.work, {"card.md": adjacent})

    def test_escaped_json_prompt_values_and_keys_are_rejected(self):
        path = self.work / "evidence.json"
        for marker in ("Ä", "\n", '"', "\\"):
            prompt = marker + "q" * 29
            self.train(prompt)
            for value in ({"nested": [{"example": prompt}]}, {prompt: "value"}):
                path.write_text(json.dumps(value, ensure_ascii=False))
                compact_json(path)
                with pytest.raises(ValueError, match="source prompt substring"):
                    text_scan(self.work, {"evidence.json": path.read_text()})

    def test_compaction_preserves_values_and_negative_zero(self):
        path = self.work / "metrics.json"
        value = {"nested": [{"float": -0.0, "number": 1.2345678901234567}], "url": "https://x/y"}
        path.write_text(json.dumps(value, indent=2))
        before = json.dumps(json.loads(path.read_text()), sort_keys=True)
        compact_json(path)
        assert json.dumps(json.loads(path.read_text()), sort_keys=True) == before
        assert "\\/" not in path.read_text()
        assert path.read_text().count("\n") == 1


if __name__ == "__main__":
    unittest.main()
