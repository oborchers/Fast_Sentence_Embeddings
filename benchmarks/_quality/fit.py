"""Private offline SIF calculations, shared with the retained Hub preparation tools."""

from __future__ import annotations

import gc
import tempfile
from importlib.metadata import version
from pathlib import Path

from _measurement import atomic, sha256
from _measurement_config import MODEL_PINS, PINS


def corpus(cell, cache):
    from hub_classic_corpus import fit_corpus

    if cell["corpus"] != "wikitext":
        raise ValueError("SIF fit requires the pinned WikiText training corpus")
    return fit_corpus(cache), PINS["corpus"]["revision"]


def miniature(folder):
    """Small deterministic mathematical fixture, used only by --rehearse."""
    import numpy as np
    from gensim.models import KeyedVectors

    words = [
        "the",
        "a",
        "man",
        "woman",
        "boy",
        "girl",
        "dog",
        "cat",
        "is",
        "was",
        "are",
        "were",
        "in",
        "on",
        "with",
        "to",
        "from",
        "playing",
        "running",
        "walking",
        "sitting",
        "standing",
        "happy",
        "sad",
    ]
    kv = KeyedVectors(vector_size=16)
    kv.add_vectors(words, np.random.default_rng(0).normal(size=(len(words), 16)).astype(np.float32))
    path = folder / "functional-fit.txt"
    path.write_text("\n".join(" ".join(words[i:] + words[:i]) for i in range(len(words))) + "\n")
    return kv, path, "functional-only"


def fit(cell, context):
    import numpy as np
    from hub_classic_corpus import batches
    from hub_classic_math import components, gram_of_batches, wordfreq_weights
    from prepare_hub_classics import fold_encoder, provenance, source_model, weighted_encoder

    import fse

    target = context["work"] / cell["id"]
    target.mkdir()
    name = cell["model"].removeprefix("fse/")
    if cell.get("functional_only"):
        kv, texts, revision = miniature(target)
    else:
        kv = source_model(name)
        texts, revision = corpus(cell, context["cache"])
    if (
        not cell.get("functional_only")
        and MODEL_PINS[cell["model"]] != __import__("hub_classic_sources").SOURCES[name]
    ):
        raise ValueError("quality and Hub source revisions differ; do not fit another source")
    weights = wordfreq_weights(kv.index_to_key)
    with tempfile.TemporaryDirectory(prefix="fit-", dir=target) as temporary:
        directory = Path(temporary)
        weighted = weighted_encoder(kv, weights, directory)
        basis = components(
            gram_of_batches(
                weighted.encode(batch, workers=cell["workers"]) for batch in batches(texts)
            )
        )
        state = context["results"] / (cell["id"] + ".npz")
        np.savez(state, weights=weights, components=basis)
        paths, arrays, params = fold_encoder(weighted, basis, directory)
        encoder = fse.Encoder.from_arrays(
            arrays[0],
            words=list(kv.index_to_key),
            fasttext=params,
            ngrams=arrays[1] if params is not None else None,
            tokenizer="external",
            normalize=True,
            oov_weight=float(weights.max()) if params is not None else None,
        )
        encoder = provenance(encoder, name)
        artifact = target / "encoder.safetensors"
        encoder.save(artifact)
        del arrays, weighted, encoder, kv
        gc.collect()
        for path in paths:
            path.unlink()
    receipt = {
        "source_model": cell["model"],
        "source_revision": MODEL_PINS[cell["model"]],
        "fit_corpus": cell["corpus"],
        "fit_revision": revision,
        "fit_corpus_sha256": sha256(texts),
        "seed": 0,
        "frequency": "wordfreq en best, integer count quantization, a=0.001",
        "wordfreq_version": version("wordfreq"),
        "components": 10,
        "fit_sha256": sha256(state),
        "artifact_sha256": sha256(artifact),
        "artifact": str(artifact.resolve()),
        "functional_only": cell.get("functional_only", False),
    }
    atomic(context["results"] / (cell["id"] + "-fit.json"), receipt)
    context["fits"][cell["id"]] = receipt
    return {"status": "valid", "fit": receipt}


def sts(cell, context):
    from hub_classic_metrics import score
    from scipy.stats import spearmanr

    import fse

    receipt = context["fits"][cell["fit"]]
    encoder = fse.Encoder.load(receipt["artifact"], mmap=True)
    labels, _, tokens = benchmark(context)
    vectors = encoder.encode(tokens, workers=cell["workers"])
    half = len(labels)
    similarities = (vectors[:half] * vectors[half:]).sum(axis=1)
    return {
        "status": "valid",
        "pairs": half,
        "pearson": score(labels, vectors),
        "spearman": float(spearmanr(labels, similarities).statistic * 100),
        "fit": receipt,
        "score_scale": "0 to 100",
        "task": "STSBenchmark test, historical Pearson and Spearman",
    }


def mean(cell, context):
    """ParaNMT uses the unweighted mean, without fitting on evaluation texts."""
    from prepare_hub_classics import provenance, source_model

    import fse
    from fse.gensim import from_gensim

    name = cell["model"].removeprefix("fse/")
    target = context["work"] / cell["id"]
    target.mkdir()
    encoder = fse.Encoder.from_word_vectors(
        from_gensim(source_model(name)), tokenizer="external", normalize=True
    )
    encoder = provenance(encoder, name)
    artifact = target / "encoder.safetensors"
    encoder.save(artifact)
    receipt = {
        "source_model": cell["model"],
        "source_revision": MODEL_PINS[cell["model"]],
        "method": "plain mean",
        "artifact": str(artifact.resolve()),
        "artifact_sha256": sha256(artifact),
        "encoder_fingerprint": encoder.fingerprint,
    }
    context["fits"][cell["id"]] = receipt
    atomic(context["results"] / (cell["id"] + "-mean.json"), receipt)
    return {"status": "valid", "fit": receipt}


def recall(cell, context):
    from hub_classic_evidence import benchmark_identity
    from hub_classic_metrics import neighbors
    from hub_classic_metrics import recall as neighbor_recall

    from .mteb import native_model

    parent = native_model(
        {**cell, "dtype": "float32", "backend": "fse"}, context["fits"].get(cell.get("fit"))
    )
    quantized = parent.quantize(cell["dtype"])
    _, texts, tokens = benchmark(context)
    inputs = tokens if parent.tokenizer == "external" else texts
    a = parent.encode(inputs, workers=cell["workers"])
    b = quantized.encode(inputs, workers=cell["workers"])
    return {
        "status": "valid",
        "recall_at_10": neighbor_recall(neighbors(a), neighbors(b)),
        "rows": len(inputs),
        "benchmark": benchmark_identity(),
        "float32_fingerprint": parent.fingerprint,
        "quantized_fingerprint": quantized.fingerprint,
        "backend_identity": __import__("_measurement").package_identity("fse"),
    }


def benchmark(context):
    """The retained historical helper reads a fixed Git object, not the caller's cwd."""
    from contextlib import chdir

    from verify_hub_classics import benchmark as historical_benchmark

    with chdir(context["fixture_repo"]):
        return historical_benchmark()
