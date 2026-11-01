"""Offline SIF/uSIF equations, following the references in recipe 01."""

import numpy as np
from collections import Counter
from scipy.linalg import eigh


def probabilities(words, corpus):
    counts = Counter(token for tokens in corpus for token in tokens)
    values = np.array([counts[word] for word in words], dtype=np.float64)
    if values.sum() == 0:
        raise ValueError("corpus has no vocabulary overlap")
    return values / values.sum()


def word_weights(p, method, length):
    if method == "mean":
        return np.ones(len(p), dtype=np.float32)
    if method == "sif":
        a = 1e-3
        return (a / (a + p)).astype(np.float32)
    if method != "usif":
        raise ValueError(method)
    v = len(p)
    threshold = 1 - (1 - 1 / v) ** length
    alpha = np.count_nonzero(p > threshold) / v
    a = (1 - alpha) / (alpha * (v / 2) + np.finfo(np.float32).eps)
    weights = a / (a / 2 + p)
    if not np.isfinite(weights).all():
        raise ValueError("uSIF weights are not finite")
    return weights.astype(np.float32)


def means(table, words, corpus):
    lookup = {word: i for i, word in enumerate(words)}
    rows = np.zeros((len(corpus), table.shape[1]), dtype=np.float32)
    for i, tokens in enumerate(corpus):
        ids = [lookup[t] for t in tokens if t in lookup]
        if ids:
            # Like 1.1.0, the offline mean counts unknown tokens in its denominator.
            rows[i] = table[ids].sum(axis=0, dtype=np.float32) / len(tokens)
    return rows


def components(rows, k, weighted):
    gram = np.zeros((rows.shape[1], rows.shape[1]), dtype=np.float64)
    for start in range(0, len(rows), 16384):
        block = rows[start : start + 16384]
        gram += block.T @ block  # float32 block product, float64 accumulation
    values, basis = eigh(gram, driver="evd")
    values = np.maximum(values[-k:][::-1], 0)
    basis = basis[:, -k:][:, ::-1].T.astype(np.float32)
    factors = (
        (values / values.sum()).astype(np.float32) if weighted else np.ones(k, dtype=np.float32)
    )
    return basis, factors


def remove(rows, basis, factors):
    return rows - ((rows @ basis.T) * factors) @ basis


def unit(rows):
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return np.divide(rows, norms, out=np.zeros_like(rows), where=norms != 0)


def read_pairs(name):
    import csv
    from _helpers import ROOT

    with (ROOT / "evaluation" / f"sts-{name}.csv").open(encoding="utf-8") as stream:
        rows = list(csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE))
    return [r[5] for r in rows], [r[6] for r in rows], np.array([float(r[4]) for r in rows])


def load_tables(recipe):
    import re
    import shutil
    from gensim.models import KeyedVectors
    from safetensors.numpy import load_file
    from tokenizers import Tokenizer

    revision = "4e5ad402664236b85be45a0d107b95259ba8a909"
    folder = recipe.work / "glove"
    folder.mkdir()
    for filename in ("glove-wiki-gigaword-300.model", "glove-wiki-gigaword-300.model.vectors.npy"):
        source = recipe.source_file("fse/glove-wiki-gigaword-300", revision, filename)
        shutil.copyfile(source, folder / filename)
    glove = KeyedVectors.load(str(folder / "glove-wiki-gigaword-300.model"), mmap="r")
    from _helpers import POTION, POTION_REVISION

    table = load_file(recipe.source_file(POTION, POTION_REVISION, "model.safetensors"))[
        "embeddings"
    ]
    tokenizer = Tokenizer.from_file(
        str(recipe.source_file(POTION, POTION_REVISION, "tokenizer.json"))
    )
    words = [tokenizer.id_to_token(i) for i in range(len(table))]
    unknown = tokenizer.token_to_id("[UNK]")

    def glove_tokens(text):
        return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())

    def potion_tokens(text):
        return [
            words[i] for i in tokenizer.encode(text, add_special_tokens=False).ids if i != unknown
        ]

    return [
        ("GloVe-300", glove.vectors, glove.index_to_key, glove_tokens),
        ("POTION-8M", table, words, potion_tokens),
    ]


def prepare(table, words, train_tokens, method):
    p = probabilities(words, train_tokens)
    length = max(1, int(np.mean([len(t) for t in train_tokens])))
    weighted = np.asarray(table * word_weights(p, method, length)[:, None], dtype=np.float32)
    if method == "mean":
        basis = np.empty((0, table.shape[1]), dtype=np.float32)
        factors = np.empty(0, dtype=np.float32)
    else:
        basis, factors = components(
            means(weighted, words, train_tokens), 10 if method == "sif" else 5, method == "usif"
        )
    folded = fold_rows(weighted, basis, factors)
    return folded, weighted, basis, factors


def pearson_score(rows, gold):
    from scipy.stats import pearsonr

    left, right = rows[: len(gold)], rows[len(gold) :]
    return float(pearsonr(gold, np.einsum("ij,ij->i", left, right)).statistic * 100)


def historical_check(recipe, table, words, test_a, test_b, gold):
    import fse

    import nltk
    from wordfreq import get_frequency_dict

    nltk.download(
        "punkt_tab", download_dir=str(recipe.work / "nltk"), quiet=True, raise_on_error=True
    )
    nltk.data.path.insert(0, str(recipe.work / "nltk"))

    historical = [historical_tokens(t) for t in test_a + test_b]
    frequency = get_frequency_dict("en", wordlist="best")
    counts = np.array([int(frequency.get(w, 1e-8) * (2**31 - 1)) for w in words], dtype=np.int64)
    p = counts.astype(np.float32)
    p /= counts.sum()
    legacy_results = []
    for method, published in (("mean", 44.46), ("sif", 71.43), ("usif", 68.82)):
        weighted = table * word_weights(p, method, 11)[:, None]
        rows = means(weighted, words, historical)
        if method != "mean":
            basis, factors = components(rows, 10 if method == "sif" else 5, method == "usif")
            reference = unit(remove(rows, basis, factors))
            folded = fold_rows(weighted, basis, factors)
        else:
            reference, folded = unit(rows), weighted
        model = fse.Encoder.from_arrays(folded, words=words, tokenizer="external", normalize=True)
        # 1.x accepted empty tokens. They contribute no row and only rescale the mean.
        # Drop them at the 2.0 boundary. Cosine direction is unchanged.
        migrated = model.encode([[t for t in tokens if t] for tokens in historical], workers=1)
        np.testing.assert_allclose(migrated, reference, atol=3e-5, rtol=3e-5)
        measured = pearson_score(migrated, gold)
        legacy_results.append(
            {
                "method": method,
                "recipe_pearson_x100": measured,
                "README_1_1_0": published,
                "delta_points": measured - published,
            }
        )
        del model, weighted, folded
    return legacy_results


def historical_tokens(text):
    import re
    from nltk import word_tokenize

    result = []
    for token in word_tokenize(text):
        if re.match(r".*[A-Za-z0-9].*", token):
            token = token.lower().strip("';.:()").strip('"')
            result.extend(re.split("-", "not" if token == "n't" else token))
    return result


def fold_rows(weighted, basis, factors):
    folded = np.empty_like(weighted)
    for start in range(0, len(weighted), 16384):
        folded[start : start + 16384] = remove(weighted[start : start + 16384], basis, factors)
    return folded
