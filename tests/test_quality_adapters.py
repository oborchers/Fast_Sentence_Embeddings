"""Run directly with .venv-quality/bin/python to check the optional MTEB adapters."""

import importlib
import importlib.util
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("mteb"), "requires the pinned quality environment")
class TestQualityAdapters(unittest.TestCase):
    def setUp(self):
        quality = Path(__file__).resolve().parents[1] / "benchmarks/quality"
        with patch.object(sys, "path", [str(quality), *sys.path]):
            self.runner = importlib.import_module("run_mteb")

    def test_native_loader_has_a_distinct_name_and_pinned_revision(self):
        name = "model2vec/minishlab/potion-base-8M"
        meta = self.runner.metadata(name)
        assert meta.name == name
        assert meta.loader is self.runner.NativeModel2VecEncoder
        assert meta.revision == self.runner.COMMITS["minishlab/potion-base-8M"]

    def test_native_adapter_preserves_texts_and_disables_all_core_joblib(self):
        adapter = object.__new__(self.runner.NativeModel2VecEncoder)
        calls = []

        def encode(texts, **kwargs):
            calls.append((texts, kwargs))
            return np.zeros((len(texts), 2), np.float32)

        adapter.model = SimpleNamespace(encode=encode)
        out = adapter.encode(
            [{"text": ["a", "b"]}, {"text": ["c"]}],
            task_metadata=None,
            hf_split="test",
            hf_subset="default",
        )
        assert out.shape == (3, 2)
        assert calls == [(["a", "b", "c"], {"use_multiprocessing": False})]

    def test_authorized_deferral_keeps_every_other_task(self):
        def task(name):
            return SimpleNamespace(
                metadata=SimpleNamespace(name=name, model_dump=lambda **_: {"name": name})
            )

        mind, other = task("MindSmallReranking"), task("STSBenchmark")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            remaining = self.runner.defer_mind_task([mind, other], {}, {"model": "fse/test"}, path)
            import json

            record = json.loads(path.read_text())
        assert remaining == [other]
        assert record["status"] == "deferred"
        assert record["main_score"] is None
        assert record["reason"] == "deferred: needs more than 16 GiB RAM on the M1"

    def test_precision_report_treats_python_version_formats_consistently(self):
        report = importlib.import_module("report_precision")
        assert report.setup({"python": "3.13.16"}) == report.setup(
            {"python": "3.13.16 (build details)"}
        )

    def test_precision_delta_needs_matching_known_source_revisions(self):
        report = importlib.import_module("report_precision")
        left = {
            "task_metadata": {"main_score": "accuracy"},
            "main_score": 0.5,
            "benchmark": "test",
            "source_revision": "new",
        }
        right = {**left, "main_score": 0.4, "source_revision": "old"}
        assert (
            report.task_deltas({"task": left}, {"task": right})[0]["delta_vs_float32_points"]
            is None
        )

    def test_batch_yields_lock_when_timing_marker_appears(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            marker = folder / "timing-window.marker"
            actions = []
            calls = []

            @contextmanager
            def reservation(_folder):
                actions.append("acquire")
                if marker.exists():
                    marker.unlink()
                yield
                actions.append("release")

            def evaluate(_model, tasks, **_kwargs):
                calls.extend(tasks)
                if len(calls) == 1:
                    marker.touch()
                return tasks[0]

            with (
                patch.object(self.runner, "task_reservation", reservation),
                patch.object(self.runner.FseEncoder, "coordination", folder),
                patch.object(self.runner.mteb, "evaluate", evaluate),
            ):
                got = list(self.runner.task_batches(None, [1, 2, 3], None))
            assert got == [(1, 1), (2, 2), (3, 3)]
            assert actions == ["acquire", "release", "acquire", "release"]


if __name__ == "__main__":
    unittest.main()
