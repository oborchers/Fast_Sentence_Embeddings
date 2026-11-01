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
            self.report = importlib.import_module("report_precision")

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
        report = self.report
        assert report.setup({"python": "3.13.16"}) == report.setup(
            {"python": "3.13.16 (build details)"}
        )

    def test_precision_delta_needs_matching_known_source_revisions(self):
        report = self.report
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

    def test_precision_delta_requires_the_same_executing_backend_code(self):
        left = {
            "task_metadata": {"main_score": "accuracy"},
            "main_score": 0.5,
            "benchmark": "test",
            "source_revision": "pinned",
            "float32_parent_fingerprint": "parent",
            "runtime_package": "fse",
            "runtime_package_version": "installed-version",
            "runtime_fingerprint": "code-before",
            "fse_reference_runtime_fingerprint": "common-reference",
        }
        right = {**left, "main_score": 0.4, "encoder_fingerprint": "parent"}
        delta = self.report.task_deltas({"task": left}, {"task": right})[0]
        assert abs(delta["delta_vs_float32_points"] - 10) < 1e-12
        for key in ("runtime_package", "runtime_package_version", "runtime_fingerprint"):
            with self.subTest(key=key):
                other = {**right, key: "different"}
                delta = self.report.task_deltas({"task": left}, {"task": other})[0]
                assert delta["float32_score"] is None
                assert delta["delta_vs_float32_points"] is None

    def test_non_fse_written_result_round_trips_through_resume(self):
        import json
        from contextlib import nullcontext

        runner = self.runner
        task_metadata = {"name": "example", "type": "STS"}
        task = SimpleNamespace(
            metadata=SimpleNamespace(
                name="example", type="STS", model_dump=lambda **_: task_metadata
            )
        )
        result = SimpleNamespace(
            task_results=[SimpleNamespace(task_name="example", get_score=lambda: 0.5)]
        )
        meta = SimpleNamespace(revision="pinned", load_model=lambda **_: SimpleNamespace())
        environment = {"versions": {"mteb": "pinned"}}
        fingerprints = {
            key: key for key in ("method_fingerprint", "runtime_fingerprint", "suite_fingerprint")
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.jsonl"
            with (
                patch("nltk.download"),
                patch.object(runner, "tasks_for", return_value=[task]),
                patch.object(
                    runner.mteb, "get_benchmark", return_value=SimpleNamespace(tasks=[task])
                ),
                patch.object(runner.mteb, "get_task", return_value=task),
                patch.object(runner, "metadata", return_value=meta),
                patch.object(runner, "identity", return_value=environment),
                patch.object(runner, "measurement_identity", return_value=fingerprints),
                patch.object(
                    runner,
                    "model_runtime",
                    return_value={
                        "backend_kind": "st",
                        "runtime_package": "sentence-transformers",
                        "runtime_package_version": "pinned",
                        "package_location": "/counterpart/st",
                        "runtime_fingerprint": "counterpart-content",
                        "fse_reference_runtime_fingerprint": "common-fse",
                    },
                ),
                patch.object(runner, "verify_backend_identity"),
                patch.object(runner, "verify_reference_runtime"),
                patch.object(runner, "ResultCache"),
                patch.object(runner, "EncodeTimer", return_value=SimpleNamespace(stats={})),
                patch.object(runner, "task_reservation", side_effect=lambda *_: nullcontext()),
                patch.object(
                    runner, "task_batches", side_effect=lambda *_: (r for r in [(task, result)])
                ),
            ):
                runner.run(["sentence-transformers/all-MiniLM-L6-v2"], None, path)
            row = json.loads(path.read_text())
            specification = {key: row[key] for key in importlib.import_module("resume").KEYS}
            assert row["float32_parent_fingerprint"] is None
            assert row["runtime_package"] == "sentence-transformers"
            assert row["runtime_fingerprint"] == "counterpart-content"
            assert row["fse_reference_runtime_fingerprint"] == "common-fse"
            assert runner.pending_tasks(path, specification, [task]) == []

    def test_comparison_context_separates_own_code_from_fse_reference(self):
        left = {"runtime_fingerprint": "fse-own", "fse_reference_runtime_fingerprint": "shared"}
        right = {"runtime_fingerprint": "st-own", "fse_reference_runtime_fingerprint": "shared"}
        assert self.report.setup(left) == self.report.setup(right)
        right["fse_reference_runtime_fingerprint"] = "different-fse-reference"
        assert self.report.setup(left) != self.report.setup(right)

    def test_all_selected_fse_model_metadata_passes_pinned_mteb_validation(self):
        model = SimpleNamespace(
            _content=SimpleNamespace(tokenizer=SimpleNamespace(max_tokens=512)), output_dim=256
        )
        names = [
            "potion-base-8M",
            "potion-base-32M",
            "potion-retrieval-32M",
            "potion-multilingual-128M",
            "static-retrieval-mrl-en-v1",
            "static-similarity-mrl-multilingual-v1",
            "glove-wiki-gigaword-300",
            "word2vec-google-news-300",
        ]
        with patch.object(self.runner, "FseEncoder", return_value=SimpleNamespace(model=model)):
            for name in names:
                for dtype in ("float32", "float16", "int8"):
                    meta = self.runner.fse_meta(f"fse/{name}-{dtype}")
                    assert meta.name == f"fse/{name}-{dtype}"
        assert self.runner.vector_license("fse/unmapped-float32") == "not specified"

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
