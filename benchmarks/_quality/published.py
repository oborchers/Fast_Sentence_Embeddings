"""Private reproduction of the published FineWeb SIF component corpus."""

from __future__ import annotations

import json

from _harness.records import atomic
from _measurement_config import PINS


def corpus(cache):
    from hub_classic_evidence import digest
    from hub_classic_fineweb import REPO, REVISION, sampled_corpus
    from hub_classic_sources import FIT_LINES
    from huggingface_hub import snapshot_download

    spec = PINS["published_sif"]
    if (REPO, REVISION, FIT_LINES, spec["seed"]) != (
        spec["repo"],
        spec["revision"],
        spec["texts"],
        0,
    ):
        raise ValueError("published corpus pins differ from the retained sampling recipe")
    output = cache / "published-fineweb"
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "fineweb-seed0-859346.txt").exists():
        snapshot_download(
            REPO,
            repo_type="dataset",
            revision=REVISION,
            token=False,
            cache_dir=cache / "fineweb" / "hub",
            allow_patterns=["data/*.parquet"],
            max_workers=2,
        )
    path = sampled_corpus(cache, output)
    if digest(path) != spec["corpus_xxh3"]:
        raise ValueError("FineWeb sample differs from the published component corpus")
    return path


def selection(cell, context, receipt, texts):
    """Export a frozen recipe receipt compatible with the retained model builder."""
    from hub_classic_evidence import digest

    spec = PINS["published_sif"]
    fit = {
        "source_repo": cell["model"],
        "source_revision": receipt["source_revision"],
        "corpus": spec["repo"],
        "revision": receipt["fit_revision"],
        "seed": spec["seed"],
        "paragraphs": sum(1 for _ in texts.open()),
        "corpus_xxh3": digest(texts),
        "wordfreq_version": receipt["wordfreq_version"],
        "wordfreq_language": "en",
        "wordfreq_wordlist": "best",
        "a": spec["a"],
        "components": spec["components"],
        "candidate_fingerprint": receipt["encoder_fingerprint"],
        "solver": "scipy.linalg.eigh(driver=evd), float32 block Gram, float64 accumulation",
    }
    row = {
        "selected": "fineweb",
        "corpus_path": str(texts.resolve()),
        "fit": fit,
        "selection_rule": "frozen published recipe; no selection experiment rerun",
        "functional_only": cell.get("functional_only", False),
    }
    if not row["functional_only"] and (
        fit["corpus_xxh3"] != spec["corpus_xxh3"] or fit["paragraphs"] != spec["texts"]
    ):
        raise ValueError("published recipe corpus changed before selection receipt")
    atomic(context["results"] / (cell["id"] + "-selection.json"), row)


def miniature_sample(output):
    """Execute the real two-pass sampler on eight tiny local Parquet shards."""
    import hub_classic_fineweb as recipe
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq

    snapshot = output / "functional-snapshot"
    data = snapshot / "data"
    data.mkdir(parents=True)
    texts = [
        "The man is walking with a happy dog.",
        "A woman is playing with a cat.",
        "The boy and the girl are running.",
    ]
    for number in range(8):
        pq.write_table(
            pa.table({"text": [*texts, "= Heading =", "", "The dog is sitting on the cat."]}),
            data / f"train-{number:05d}-of-00008.parquet",
        )
    original_download, original_count = huggingface_hub.snapshot_download, recipe.FIT_LINES

    def local_snapshot(repo, **kwargs):
        if (
            repo != recipe.REPO
            or kwargs["revision"] != recipe.REVISION
            or not kwargs["local_files_only"]
        ):
            raise ValueError("functional snapshot request differs from the real sampling path")
        return str(snapshot)

    try:
        huggingface_hub.snapshot_download = local_snapshot
        recipe.FIT_LINES = 24
        path = recipe.sampled_corpus(output, output)
        # The second invocation also exercises the retained cached-sample verifier.
        if recipe.sampled_corpus(output, output) != path:
            raise ValueError("functional sampled corpus cache changed")
    finally:
        huggingface_hub.snapshot_download = original_download
        recipe.FIT_LINES = original_count
    manifest = json.loads((output / "fineweb-sample.json").read_text())
    if manifest["texts"] != 24 or manifest["seed"] != 0 or len(path.read_text().splitlines()) != 24:
        raise ValueError("functional sampler row count or seed differs")
    return path
