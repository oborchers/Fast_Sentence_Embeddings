"""Model2vec's tokenizer setting and its optional batch pool are distinct controls."""

import importlib
from pathlib import Path


def test_wikitext_default_does_not_claim_ten_tokenizer_threads(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    settings = importlib.import_module("compare_parallelism").settings
    default = settings("model2vec", 10, 5352)
    assert default["threads"] is None
    assert default["tokenizer_thread_count"] is None
    assert default["tokenizer_parallelism"] == "on"
    assert default["parallel_batch_pool"] == "disabled"
    assert settings("model2vec", 1, 5352)["tokenizer_parallelism"] == "off"
    assert settings("fse", 10, 5352)["fse_workers"] == 10


def test_large_default_batch_uses_pool_and_disables_tokenizer_parallelism(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    settings = importlib.import_module("compare_parallelism").settings
    default = settings("model2vec", 10, 151674)
    assert default["parallel_batch_pool"] == "default"
    assert default["tokenizer_parallelism"] == "off"


def test_accelerate_limit_is_set_before_a_fresh_interpreter_starts(monkeypatch):
    import json
    import os
    import subprocess
    import sys

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    module = importlib.import_module("compare_parallelism")
    monkeypatch.setenv("VECLIB_MAXIMUM_THREADS", "999")
    environment = module.child_environment(1)
    output = subprocess.check_output(
        [sys.executable, "-c", "import os; print(os.environ['VECLIB_MAXIMUM_THREADS'])"],
        env=environment,
        text=True,
    )
    assert output.strip() == "1"
    assert os.environ["VECLIB_MAXIMUM_THREADS"] == "999"
    assert {key: environment[key] for key in module.NATIVE_THREAD_VARIABLES} == dict.fromkeys(
        module.NATIVE_THREAD_VARIABLES, "1"
    )
    assert json.loads(json.dumps(module.native_thread_limits()))["VECLIB_MAXIMUM_THREADS"] == "999"
