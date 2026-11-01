"""Large jobs preserve source multiplicities, storage modes and streamed inputs."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.syspath_prepend(str(root / "quality"))
    return importlib.import_module("pod_large_inputs")


def test_repetition_is_seeded_balanced_and_uses_every_source(modules):
    indices = modules.shuffled_indices(3, 8, 0)
    np.testing.assert_array_equal(np.bincount(indices), [3, 3, 2])
    np.testing.assert_array_equal(indices, modules.shuffled_indices(3, 8, 0))
    assert not np.array_equal(indices, modules.shuffled_indices(3, 8, 1))
    with pytest.raises(ValueError, match="empty"):
        modules.shuffled_indices(0, 8, 0)


def test_weighted_statistics_equal_explicit_repetition(modules):
    lengths = importlib.import_module("text_lengths")
    source = ["one two", "three", "four five six"]
    model = SimpleNamespace(
        max_length=None, tokenize=lambda rows: [list(range(len(r.split()))) for r in rows]
    )
    weighted = modules.repeated_lengths(source, model, 8)
    explicit = lengths.measure_lengths([source[i % 3] for i in range(8)], model)
    assert weighted["texts"] == explicit["texts"] == 8
    for key in ("words", "characters", "potion_tokens"):
        assert weighted[key] == explicit[key]


@pytest.mark.usefixtures("modules")
def test_recipe_chunk_rule_keeps_source_order_and_minimum_words():
    chunker = importlib.import_module("wiki_chunks")
    assert list(chunker.split_sentences(" = heading = ")) == []
    assert list(
        chunker.split_sentences("one two three four five. Six seven eight nine ten! tiny.")
    ) == ["one two three four five.", "Six seven eight nine ten!"]


@pytest.mark.usefixtures("modules")
def test_matrix_retains_every_storage_dtype_and_durability():
    matrix = importlib.import_module("pod_large")
    for workers in (10, 32):
        commands = matrix.commands(["python"], workers)
        for dtype in ("float32", "float16", "int8"):
            for durability in ("none", "full"):
                label = (
                    "fse-a"
                    if (dtype, durability) == ("float32", "none")
                    else f"shard-{dtype}-{durability}"
                )
                assert dtype in commands[label]
                assert durability in commands[label]
        assert {"encode", "memory", "rust-npy-none", "rust-npy-full"} <= commands.keys()
    single = matrix.commands(["python"], 1)
    assert set(single) == {"fse-a", "fse-b", "encode", "memory", "rust-npy-none"}


@pytest.mark.usefixtures("modules")
@pytest.mark.parametrize("mode", ["encode", "memory", "shard"])
def test_native_paths_stream_the_file_and_do_not_build_a_text_list(monkeypatch, tmp_path, mode):
    import fse

    paths = importlib.import_module("large_write_paths")
    rows = tmp_path / "example.txt"
    rows.write_text("red blue\ngreen\n")
    encoder = fse.Encoder.from_arrays(np.eye(3, dtype=np.float32), words=["red", "blue", "green"])
    artifact = tmp_path / "encoder.safetensors"
    encoder.save(artifact)
    monkeypatch.setattr(paths, "artifact_path", lambda *_: artifact)
    monkeypatch.setattr(paths, "checked_encoder", lambda *_: {})
    clocks = iter([1.0, 2.0, 3.0])
    monkeypatch.setattr(paths, "time", SimpleNamespace(perf_counter=lambda: next(clocks)))
    args = SimpleNamespace(
        models=tmp_path,
        inputs=tmp_path,
        corpus="example",
        mode=mode,
        workers=1,
        storage_dtype="int8",
        durability="none",
    )
    record, vectors = paths.fse_run(args, tmp_path / "store")
    assert record["p50_s"] == (
        2.0 if mode == "shard" else 1.0
    )  # Stub clock; no local benchmark was timed.
    if mode == "shard":
        assert vectors is None
        assert record["published_rows"] == 2
        assert record["durable_rows"] == 0
        with fse.VectorStore.open(tmp_path / "store") as store:
            assert len(store) == 2
            assert store.dtype == "int8"
    else:
        assert vectors.shape == (2, 3)


@pytest.mark.usefixtures("modules")
def test_showcase_is_exactly_one_run_and_one_aa_repeat(monkeypatch, tmp_path):
    showcase = importlib.import_module("pod_showcase")
    calls = []
    monkeypatch.setattr(showcase, "prepare_one", lambda _, count: {"texts": count})
    monkeypatch.setattr(showcase, "ensure_model", lambda *_: None)
    monkeypatch.setattr(showcase, "emit", lambda *_: None)
    monkeypatch.setattr(showcase, "environment", dict)

    def operation(command, label, *_):
        calls.append(command)
        return {
            "label": label,
            "p50_s": 1.0 if label == "fse-a" else 1.02,
            "output_digest": "same",
            "runtime_package": "fse",
            "runtime_package_version": "test",
            "package_location": "/installed/fse",
            "runtime_fingerprint": "same-code",
        }

    monkeypatch.setattr(showcase, "operation", operation)
    showcase.run(SimpleNamespace(output=tmp_path, cache=tmp_path, deadline=100))
    result = json.loads((tmp_path / "showcase-100m/summary.json").read_text())
    assert len(calls) == 2
    assert result["required_rounds"] == 1
    assert result["complete"]
    assert result["source"]["texts"] == 100_000_000
    for command in calls:
        assert command[command.index("--storage-dtype") + 1] == "int8"
        assert command[command.index("--durability") + 1] == "none"
        assert command[command.index("--workers") + 1] == "32"


@pytest.mark.usefixtures("modules")
@pytest.mark.parametrize("workers", [1, 10, 32])
def test_rust_parallelism_overrides_inherited_serial_setting(monkeypatch, workers):
    import os

    paths = importlib.import_module("large_write_paths")
    observed = []
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setattr(
        paths,
        "rust_pipeline",
        lambda *_: observed.append(
            (os.environ["TOKENIZERS_PARALLELISM"], os.environ["RAYON_NUM_THREADS"])
        ),
    )
    paths.run(SimpleNamespace(kind="rust", workers=workers), Path("unused"))
    assert observed == [("true" if workers > 1 else "false", str(workers))]
