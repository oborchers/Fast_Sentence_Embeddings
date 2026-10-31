"""Pinned real-model cells for the MLX gate; no downloads or synthetic substitutes."""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import fse

BASE = "c754efd382f7b42504d140cabf5aab6e40ad51bb"
MODELS = {
    "static-retrieval-mrl-en-v1": (
        "sentence-transformers/static-retrieval-mrl-en-v1",
        "f60985c706f192d45d218078e49e5a8b6f15283a",
        (1024, 512, 256),
    ),
    "potion-base-8M": (
        "minishlab/potion-base-8M",
        "bf8b056651a2c21b8d2565580b8569da283cab23",
        (256,),
    ),
    "potion-base-32M": (
        "minishlab/potion-base-32M",
        "1e5a03f8eeb2c98b928fbbd846f22f816360919f",
        (512,),
    ),
}
DATASETS = {
    "wikitext": ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3", 5352),
    "massive": ("mteb/amazon_massive_intent", "940fd47a81eaa7f2cc7b129674d945d618ac38c2", 151674),
}


@dataclass(frozen=True)
class Cell:
    model: str
    dim: int
    corpus: str
    mode: str
    dtype: str

    @property
    def key(self) -> str:
        return f"{self.model}-{self.dim}-{self.corpus}-{self.mode}-{self.dtype}"

    @property
    def deciding(self) -> bool:
        return (self.model, self.dim, self.corpus, self.dtype) == (
            "static-retrieval-mrl-en-v1",
            1024,
            "wikitext",
            "float32",
        )


def matrix() -> list[Cell]:
    return [
        Cell(model, dim, corpus, mode, dtype)
        for model, (_, _, dims) in MODELS.items()
        for dim in dims
        for corpus in DATASETS
        for mode in ("raw", "norm")
        for dtype in ("float32", "float16", "int8")
    ]


def snapshot(receipt: list[dict], repo: str, revision: str, kind: str) -> Path:
    matches = [r for r in receipt if (r["repo"], r["type"]) == (repo, kind)]
    if len(matches) != 1 or matches[0]["revision"] != revision:
        raise ValueError(f"cache receipt does not pin {repo} at {revision}")
    path = Path(matches[0]["path"])
    name = f"{kind}s--{repo.replace('/', '--')}"
    if path.parts[-3:] != (name, "snapshots", revision) or not path.is_dir():
        raise ValueError(f"missing or unexpected snapshot: {path}")
    return path


def variant(enc, mode: str, dtype: str):
    """Keep every conversion rule and provenance field; change only steps and table dtype."""
    if mode not in ("raw", "norm") or dtype not in ("float32", "float16", "int8"):
        raise ValueError("unknown mode or table dtype")
    content = enc._content
    out = fse.Encoder._converted(
        enc.word_vectors,
        content.tokenizer,
        normalize=mode == "norm",
        source=content.source,
        license=content.license,
    )
    out = out if dtype == "float32" else out.quantize(dtype)
    if out.word_vectors.dtype != dtype or bool(out.steps) != (mode == "norm"):
        raise AssertionError("encoder does not match the requested cell")
    return out


def encoder(cell: Cell, receipt: list[dict]):
    repo, revision, dims = MODELS[cell.model]
    if cell.dim not in dims:
        raise ValueError("dimension outside the gate matrix")
    path = snapshot(receipt, repo, revision, "model")
    options = {"truncate_dim": cell.dim} if cell.model.startswith("static-") else {}
    enc = fse.Encoder.from_pretrained(
        repo, revision=revision, local_files_only=True, cache_dir=path.parents[2], **options
    )
    if enc._content.source.revision != revision or enc.output_dim != cell.dim:
        raise AssertionError("converted encoder has unexpected provenance or dimension")
    return variant(enc, cell.mode, cell.dtype)


def corpus(cell: Cell, receipt: list[dict]) -> tuple[list[str], dict]:
    """The same selection and ordering as bench_wordpiece.texts, at one exact snapshot."""
    repo, revision, expected = DATASETS[cell.corpus]
    path = snapshot(receipt, repo, revision, "dataset")
    files, texts = _wikitext(path) if cell.corpus == "wikitext" else _massive(path)
    if len(texts) != expected:
        raise ValueError(f"expected {expected} texts, got {len(texts)}")
    from fse._container._hash import Hasher

    digest = Hasher()
    for text in texts:
        data = text.encode("utf-8")
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return texts, {
        "repo": repo,
        "revision": revision,
        "files": list(map(str, files)),
        "texts": len(texts),
        "checksum": digest.checksum(),
    }


def _wikitext(path: Path) -> tuple[list[Path], list[str]]:
    import pyarrow.parquet as pq

    files, texts = [], []
    for split in ("validation", "test"):
        shards = sorted(path.glob(f"wikitext-103-raw-v1/{split}-*"))
        if len(shards) != 1:
            raise ValueError(f"expected one {split} parquet shard")
        files.extend(shards)
        texts.extend(pq.read_table(shards[0]).column("text").to_pylist())
    return files, [s for s in texts if s.strip()]


def _massive(path: Path) -> tuple[list[Path], list[str]]:
    files, texts = sorted(path.glob("test/*.json.gz")), []
    if len(files) != 51:
        raise ValueError("expected all 51 MASSIVE test languages")
    for file in files:
        with gzip.open(file, "rt", encoding="utf-8") as stream:
            texts.extend(json.loads(line)["text"] for line in stream)
    return files, texts


def describe(cell: Cell, enc) -> dict:
    return {
        **asdict(cell),
        "key": cell.key,
        "output_dtype": "float32",
        "encoder_fingerprint": enc.fingerprint,
        "table_fingerprint": enc.word_vectors.fingerprint,
        "tokenizer": asdict(enc._content.tokenizer),
        "source": asdict(enc._content.source),
        "dtype_verified": str(enc.word_vectors.stored_vectors.dtype),
        "normalize_verified": bool(enc.steps),
    }
