"""Locale membership of the non-Latin Unigram timing cell."""

import gzip
import importlib
import json
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"
pytestmark = pytest.mark.skipif(
    not (BENCHMARKS / "bench_unigram.py").exists(),
    reason="the local benchmark checkout is absent",
)


@pytest.mark.parametrize("corpus", ["massive", "massive-nonlatin"])
def test_massive_subset_filters_locale_metadata_without_filtering_text(
    corpus, monkeypatch, tmp_path
):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    bench = importlib.import_module("bench_unigram")
    # Mixed locales in one unrelated filename prove membership comes from each row.
    path = tmp_path / "mixed.json.gz"
    rows = [
        {"lang": lang, "text": f"ASCII payload for {lang}"}
        for lang in ("en", "zh-CN", "zh-TW", "ja", "ja-JP", "ko")
    ]
    with gzip.open(path, "wt", encoding="utf-8") as out:
        out.write("\n".join(json.dumps(row) for row in rows))
    monkeypatch.setattr(bench, "_one", lambda _pattern: [path])
    expected = rows if corpus == "massive" else rows[1:5]
    assert bench.texts(corpus) == [row["text"] for row in expected]
