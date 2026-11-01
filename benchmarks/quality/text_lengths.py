"""Exact streaming length statistics, with the pinned POTION inference token rules."""

from collections import Counter
from itertools import islice

from compare_artifacts import file_digest
from compare_models import MODELS
from compare_snapshots import cached_snapshot

MODEL = "minishlab/potion-base-8M"


def counter():
    from model2vec import StaticModel

    folder = cached_snapshot(MODEL, MODELS[MODEL])
    model = StaticModel.from_pretrained(folder)
    from compare_runtime import backend_identity

    provenance = {
        "runtime": backend_identity("model2vec"),
        "model": MODEL,
        "revision": MODELS[MODEL],
        "tokenizer_file_checksum": file_digest(folder / "tokenizer.json"),
        "max_tokens": model.max_length,
        "character_cap": model.max_length * model.median_token_length if model.max_length else None,
        "unknown_id_excluded": model.unk_token_id,
        "tokens_definition": "tokens consumed after configured character/token truncation and unknown-token removal; no special tokens",
    }
    return model, provenance


def token_counts(model, rows):
    if model.max_length is not None:
        cap = model.max_length * model.median_token_length
        rows = [row[:cap] for row in rows]
    return [len(ids) for ids in model.tokenize(rows)]


def median(histogram):
    n = sum(histogram.values())
    targets = ((n - 1) // 2, n // 2)
    found, seen = [], 0
    for value, count in sorted(histogram.items()):
        found.extend(value for target in targets if seen <= target < seen + count)
        seen += count
    return sum(found) / 2 if found else None


class Lengths:
    def __init__(self):
        from fse._container._hash import Hasher

        self.count = 0
        self.histograms = {name: Counter() for name in ("words", "characters", "potion_tokens")}
        self.hasher = Hasher()

    def add(self, rows, tokens):
        for text, count in zip(rows, tokens, strict=True):
            values = (len(text.split()), len(text), count)
            for name, value in zip(self.histograms, values, strict=True):
                self.histograms[name][value] += 1
            data = text.encode("utf-8")
            self.hasher.update(len(data).to_bytes(8, "little"))
            self.hasher.update(data)
            self.count += 1

    def add_weighted_counts(self, words, characters, tokens, weight):
        if weight < 0:
            raise ValueError("negative input multiplicity")
        for name, value in zip(self.histograms, (words, characters, tokens), strict=True):
            self.histograms[name][value] += weight
        self.count += weight

    def record(self):
        if not self.count:
            raise ValueError("length statistics require at least one complete text")
        result = {"texts": self.count, "input_digest": self.hasher.checksum()}
        for name, histogram in self.histograms.items():
            total = sum(value * count for value, count in histogram.items())
            result[name] = {"total": total, "mean": total / self.count, "median": median(histogram)}
        return result


def batches(rows, size=8192):
    source = iter(rows)
    while batch := list(islice(source, size)):
        yield batch


def measure_lengths(rows, model):
    result = Lengths()
    for batch in batches(rows):
        result.add(batch, token_counts(model, batch))
    return result.record()
