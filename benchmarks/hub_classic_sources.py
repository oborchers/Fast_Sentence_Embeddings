"""Pinned classic sources and offline recipe names for the maintainer's quality scope."""

from __future__ import annotations

SOURCES = {
    "paragram-300-sl999": "c99818c1470d5c4e454eb9d59dfce2b179dba455",
    "paranmt-300": "a08e15771891207a22f840a3091a00a1657d888b",
    "glove-wiki-gigaword-300": "4e5ad402664236b85be45a0d107b95259ba8a909",
    "word2vec-google-news-300": "4274c2efa75c99a64e3f83eabfc5b34d1acabe70",
    "fasttext-crawl-subwords-300": "552b66a16850576275f894ed738f9e953b45a594",
}
FIT_REPO = "Salesforce/wikitext"
FIT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
FIT_LINES = 859_346

DTYPES = ("float32", "float16", "int8")

BASELINES = {
    "glove-wiki-gigaword-300": (44.46, 71.43),
    "word2vec-google-news-300": (61.54, 71.18),
    "paranmt-300": (79.82, 76.74),
    "paragram-300-sl999": (51.46, 74.29),
    "fasttext-crawl-subwords-300": (48.53, 73.58),
}

LICENSES = {
    "glove-wiki-gigaword-300": "pddl",
    "word2vec-google-news-300": "other",
    "paranmt-300": "unknown",
    "paragram-300-sl999": "unknown",
    "fasttext-crawl-subwords-300": "cc-by-sa-3.0",
}
