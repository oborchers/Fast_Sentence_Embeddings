"""Pinned WordPiece stress inputs and explicit token-limit artifact variants."""

from __future__ import annotations

import gzip
import json
from dataclasses import replace

from _harness.budget import check
from _harness.records import document, sha256
from _measurement_config import CONFIG

CORPORA = ("massive-de-fr-ru", "stsb-added-pieces", "wikitext-long-docs")


def model_key(cell):
    limit = cell.get("token_limit")
    return cell["model"] + (f"@tokens-{limit}" if limit is not None else "")


def validate(cell):
    limit = cell["token_limit"]
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("token_limit must be a positive integer or null")
    if limit is not None and cell["stage"] != "tokenizer":
        raise ValueError("token_limit currently supports the tokenizer stage only")


def variants(model, repo, cells):
    """Make changed length rules before persistence, never inside a timed operation."""
    yield repo, model
    limits = sorted(
        {c["token_limit"] for c in cells if c["model"] == repo and c["token_limit"] is not None}
    )
    for limit in limits:
        if model._content.tokenizer.name != "wordpiece@1":
            raise ValueError("token_limit requires WordPiece")
        tokenizer = replace(model._content.tokenizer, max_tokens=limit, max_chars=None)
        current = type(model)._new(replace(model._content, tokenizer=tokenizer), model.word_vectors)
        yield f"{repo}@tokens-{limit}", current


def save_models(model, repo, cells, work, deadline):
    result = {}
    for key, current in variants(model, repo, cells):
        artifacts = {}
        for dtype in ("float32", "float16", "int8"):
            check(deadline)
            path = work / (key.replace("/", "--") + "-" + dtype + ".safetensors")
            converted = current if dtype == "float32" else current.quantize(dtype)
            converted.save(path)
            artifacts[dtype] = dict(
                path=str(path), sha256=sha256(path), fingerprint=converted.fingerprint
            )
        pool = current._spec.pool_input()
        result[key] = dict(
            artifacts=artifacts,
            dimensions=current.output_dim,
            file_import_alignment_mod128=int(pool.vectors.ctypes.data % 128),
            length_rule=[current._spec.tokenizer.max_tokens, current._spec.tokenizer.max_chars],
        )
    return result


def massive(work, cache, deadline, snapshot):
    pin = document(CONFIG / "wordpiece-corpora.json")["massive"]
    paths = [f"test/{lang}.json.gz" for lang in pin["languages"]]
    root = snapshot(pin["repo"], pin["revision"], work, cache, dataset=True, patterns=paths)
    texts, files = [], []
    for name in paths:
        check(deadline)
        path = root / name
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        if len(rows) != pin["rows_per_language"] or any(
            not isinstance(r.get("text"), str) for r in rows
        ):
            raise ValueError("pinned MASSIVE rows or schema changed")
        texts.extend(row["text"] for row in rows)
        files.append(dict(path=name, sha256=sha256(path), rows=len(rows)))
    return dict(texts=texts, score_slices=[], source={**pin, "files": files})


def corpus(name, work, cache, deadline, snapshot, stsb, wikitext):
    if name == "massive-de-fr-ru":
        data = massive(work, cache, deadline, snapshot)
    elif name == "stsb-added-pieces":
        data = stsb(deadline)
        data["texts"] = [_added(text) for text in data["texts"]]
        # Derived text is a stress workload, not an STS quality measurement.
        data["score_slices"] = []
        data["source"]["derived"] = "insert alternating [MASK]/[SEP] after groups of 1,2,3 words"
    else:
        from _development_lengths import documents

        data = wikitext(work, cache, deadline)
        data["texts"] = documents(data["texts"], 16384)
        data["source"]["derived"] = (
            "join nonempty rows until over 16384 code points; drop final rest"
        )
    data["timing_texts"] = [" ".join(text.split()) for text in data["texts"]]
    data["timing_policy"] = "one-space whitespace; preserve derived row order"
    return data


def _added(text):
    words = text.split()
    parts, pos, group = [], 0, 0
    while pos < len(words):
        width = group % 3 + 1
        parts.extend(words[pos : pos + width])
        parts.append("[MASK]" if group % 2 == 0 else "[SEP]")
        pos += width
        group += 1
    return " ".join(parts)
