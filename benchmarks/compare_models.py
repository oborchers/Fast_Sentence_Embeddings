"""Immutable source revisions for the complete comparison model selection."""

from bench_pretrained import COMMITS


MODELS = {
    **COMMITS,
    "fse/paranmt-300": "a08e15771891207a22f840a3091a00a1657d888b",
    "fse/fasttext-crawl-subwords-300": "552b66a16850576275f894ed738f9e953b45a594",
    "fse/glove-wiki-gigaword-300": "4e5ad402664236b85be45a0d107b95259ba8a909",
    "fse/word2vec-google-news-300": "4274c2efa75c99a64e3f83eabfc5b34d1acabe70",
    "sentence-transformers/all-MiniLM-L6-v2": "8b3219a92973c328a8e22fadcfa821b5dc75636a",
}


def artifact_name(repo: str, dtype: str) -> str:
    stem = repo.rsplit("/", maxsplit=1)[-1]
    if repo.startswith("fse/"):
        stem = "Average-" + stem
    return f"{stem}-{dtype}.safetensors"
