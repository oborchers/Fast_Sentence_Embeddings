#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The prepared files of bench_length.py and the checks before every reuse.

Every artifact (corpus, word vectors, model file) is checked by its size and xxh3 against
what was recorded when it was made. ``Prepared`` with ``verify`` hashes every file it
uses and records a stamp (size, modification time, xxh3) in ``<cache>/verified.json``;
without ``verify`` (the profiled process, which would otherwise profile the hashing) a
file must have a stamp of the same size and time, which ``bench_length.py verify`` (or
any prepare, compare or matrix) wrote. A model is reused only if it was made of the same
inputs (``model_meta``): the corpus 1.1 trains on, the counts, the hub revision of the
vectors and the xxh3 of the word vectors 2.0 builds its table of, a quantized copy bound to its
float32 origin. Every JSON file is written atomically; one that cannot be read counts as
missing, so its artifact is made again.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import bench_length_corpus as corpora

CASES = Path(__file__).with_name("bench_length_cases.py")
MODEL_VERSION = 3  # of the model preparation; bump it to make every model again


def write_json(path: Path, value: object) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict | None:
    """The object in ``path``, or None if it is missing or not a JSON object."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def model_files(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]


def record_path(path: Path) -> Path:
    return path.with_name(path.name + ".json")


def hub_revision(vectors: str) -> str:
    """The snapshot of the gensim model in the hub cache, which both sides start from."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(f"fse/{vectors}", local_files_only=True)).name


def vocabulary_words(path: Path) -> set[str]:
    import fse

    wv = fse.WordVectors.load(path, mmap=True)
    data, offsets = wv.vocab_bytes.tobytes(), wv.vocab_offsets.tolist()
    return {data[a:b].decode("utf-8") for a, b in itertools.pairwise(offsets)}


class Prepared:
    """The prepared files of one --tokens and --seed: corpora, manifests and models."""

    def __init__(self, args, identity, verify: bool = True):
        self.args = args
        self.identity = identity
        self.verify = verify
        self.cache = args.cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.stamp_file = self.cache / "verified.json"
        self.stamps = read_json(self.stamp_file) or {}
        self._digests: dict[str, str] = {}
        self.wv_float32 = self.cache / f"{args.vectors}.safetensors"
        self.counts = self.cache / "counts-en.npy"
        self.source = self.xxh3(corpora.CACHED)
        self.folder = corpora.folder(self.cache, self.source, args.tokens, args.seed)
        self.models = self.folder / "models"
        self.generator = {
            "tokens": args.tokens,
            "seed": args.seed,
            "sentence_mean": corpora.SENTENCE_MEAN,
            "synthetic": list(corpora.SYNTHETIC),
            "parts": list(corpora.PARTS),
            "skew_share": corpora.SKEW_SHARE,
        }
        self.manifests: dict[str, dict] = {}
        self._vocabulary: str | None = None
        self._revision: str | None = None

    # --- the checks -------------------------------------------------------------------

    def xxh3(self, path: Path) -> str:
        """The xxh3 of ``path``: hashed with ``verify`` (and stamped), else its stamp."""
        key = str(path)
        if key in self._digests:
            return self._digests[key]
        stat = path.stat()
        stamp = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if self.verify:
            stamp["xxh3"] = self.identity(path)["xxh3"]
            self.stamps[key] = stamp
            write_json(self.stamp_file, self.stamps)
        else:
            recorded = self.stamps.get(key) or {}
            if {k: recorded.get(k) for k in stamp} != stamp:
                raise SystemExit(f"{path} is not verified: run bench_length.py verify first")
            stamp["xxh3"] = recorded["xxh3"]
        self._digests[key] = stamp["xxh3"]
        return stamp["xxh3"]

    def matches(self, path: Path, recorded: dict) -> bool:
        """Whether ``path`` is the file ``recorded`` (its size and xxh3)."""
        if not path.exists() or path.stat().st_size != recorded.get("bytes"):
            return False
        return self.xxh3(path) == recorded.get("xxh3")

    def sources(self) -> dict:
        """The sources of the corpora: the cached corpus and the raw parquet files."""
        parquet = sorted(corpora.PARQUET.glob("train-*.parquet"))
        return {
            "cached": self.source,
            "parquet": [{"name": p.name, "bytes": p.stat().st_size} for p in parquet],
        }

    def vocabulary(self) -> str:
        """The vocabulary fingerprint of the word vectors (equal for every dtype)."""
        if self._vocabulary is None:
            import fse

            wv = fse.WordVectors.load(self.wv_float32, mmap=True)
            self._vocabulary = wv.vocabulary_fingerprint
        return self._vocabulary

    def revision(self) -> str:
        if self._revision is None:
            self._revision = hub_revision(self.args.vectors)
        return self._revision

    # --- the corpora ------------------------------------------------------------------

    def corpus(self, name: str) -> Path:
        return self.folder / f"{name}.txt"

    def manifest(self, name: str) -> dict:
        if name not in self.manifests:
            expected = {
                "kind": name,
                "generator": self.generator,
                "tokenizer": corpora.tokenizer_of(name),
                "vocabulary": self.vocabulary(),
                "sources": self.sources(),
            }
            path = self.corpus(name)
            self.manifests[name] = corpora.load_manifest(path, expected, self.matches)
        return self.manifests[name]

    def stats(self, name: str, tokenizer: str) -> dict:
        """The statistics of a corpus as ``tokenizer`` splits it, computed (with
        ``verify``) and kept in the manifest the first time."""
        manifest = self.manifest(name)
        if tokenizer not in manifest["stats"]:
            if not self.verify:
                raise SystemExit(f"{name} has no {tokenizer} statistics: run verify first")
            lines = self.corpus(name).read_text(encoding="utf-8").splitlines()
            texts = [line.split() for line in lines] if tokenizer == "external" else lines
            words = vocabulary_words(self.wv_float32)
            manifest["stats"][tokenizer] = corpora.stats(texts, tokenizer, words)
            corpora.save_manifest(self.corpus(name), manifest)
        return manifest["stats"][tokenizer]

    # --- the vectors and models -------------------------------------------------------

    def wv(self, dtype: str) -> Path:
        if dtype == "float32":
            return self.wv_float32
        return self.cache / f"{self.args.vectors}-{dtype}.safetensors"

    def wv_meta(self, dtype: str) -> dict:
        """What word vectors are made of: the hub revision, and for a quantized copy the
        xxh3 of its float32 origin."""
        meta = {"vectors": self.args.vectors, "revision": self.revision(), "dtype": dtype}
        if dtype != "float32":
            meta["origin"] = self.xxh3(self.wv_float32)
        return meta

    def v2_model(self, key: tuple[str, str, str]) -> Path:
        model, tokenizer, wv_dtype = key
        return self.models / f"v2-{model}-{tokenizer.replace('@', '')}-{wv_dtype}.safetensors"

    def v1_model(self, key: tuple[str, str]) -> Path:
        return self.models / f"v1-{key[0]}-{key[1]}"

    def model_meta(self, side: str, key: tuple) -> dict:
        """What a model is made of: its key, the paragraphs (1.1 trains on them), the
        counts, the hub revision of the vectors and, for 2.0, the word vectors of its table."""
        meta = {
            "version": MODEL_VERSION,
            "side": side,
            "key": list(key),
            "paragraphs": self.manifest("paragraphs")["artifact"]["xxh3"],
            "counts": self.xxh3(self.counts),
            "vectors": self.args.vectors,
            "revision": self.revision(),
        }
        if side == "v2":
            meta["word_vectors"] = self.xxh3(self.wv(key[2]))
            meta["origin"] = self.xxh3(self.wv_float32)
        return meta

    def made_of(self, path: Path, meta: dict) -> bool:
        """Whether the artifact ``path`` (a file or a folder) was made of ``meta`` and is
        unchanged since (its record next to it)."""
        recorded = read_json(record_path(path))
        if recorded is None or recorded.get("meta") != meta or not path.exists():
            return False
        files, saved = model_files(path), recorded.get("files") or []
        names = [Path(f.get("path", "")).name for f in saved]
        if names != [f.name for f in files]:
            return False
        return all(self.matches(p, r) for p, r in zip(files, saved, strict=True))

    def forget(self, path: Path) -> None:
        """Hash ``path`` again at its next use (it was just made)."""
        self._digests.pop(str(path), None)

    def artifact(self, path: Path) -> dict:
        """The identity of ``path``, just made: hashed and stamped."""
        self.forget(path)
        return {"path": str(path), "bytes": path.stat().st_size, "xxh3": self.xxh3(path)}

    def record(self, path: Path, meta: dict) -> None:
        """Record what the artifact ``path``, just made, is made of."""
        files = [self.artifact(p) for p in model_files(path)]
        write_json(record_path(path), {"meta": meta, "files": files})


# --- making them -------------------------------------------------------------------


def run_child(python: str, config: dict) -> dict:
    # Offline: 1.1 resolves the hub cache through snapshot_download.
    done = subprocess.run(
        [python, str(CASES), json.dumps(config)],
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode:
        raise SystemExit(f"{config} failed:\n{done.stderr[-3000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def ensure_vectors(ctx: Prepared, dtype: str) -> Path:
    """The word vectors of ``dtype``: float32 converted from the gensim model of the hub
    cache, a quantized copy made from them; made again if their record differs."""
    path, meta = ctx.wv(dtype), ctx.wv_meta(dtype)
    if ctx.made_of(path, meta):
        return path
    path.unlink(missing_ok=True)
    print(f"# making the word vectors {path.name}", flush=True)
    config = {"side": "v2", "vectors": ctx.args.vectors, "tmp": str(ctx.cache)}
    config.update(wv=str(path), wv_float32=str(ctx.wv_float32), wv_dtype=dtype)
    run_child(sys.executable, {**config, "kind": "prepare" if dtype == "float32" else "quantize"})
    ctx.record(path, meta)
    return path


def ensure_model(ctx: Prepared, side: str, key: tuple, python: str) -> Path:
    """The model of ``key``, made (untimed) if missing or made of something else."""
    if side == "v2":
        ensure_vectors(ctx, key[2])
    path = ctx.v2_model(key) if side == "v2" else ctx.v1_model(key)
    meta = ctx.model_meta(side, key)
    if ctx.made_of(path, meta):
        return path
    if path.is_dir():  # a stale 1.1 model folder of this cache
        shutil.rmtree(path)
    ctx.models.mkdir(parents=True, exist_ok=True)
    config = {
        "side": side,
        "kind": "model",
        "model": key[0],
        "vectors": ctx.args.vectors,
        "counts": str(ctx.counts),
        "paragraphs": str(ctx.corpus("paragraphs")),
        "workers": os.cpu_count(),
    }
    if side == "v2":
        config.update(tokenizer=key[1], wv_dtype=key[2], wv=str(ctx.wv(key[2])))
        config["encoder"] = str(path)
    else:
        config.update(dtype=key[1], v1_model=str(path))
    print(f"# making the {side} model {'-'.join(key)}", flush=True)
    run_child(python, config)
    ctx.record(path, meta)
    return path
