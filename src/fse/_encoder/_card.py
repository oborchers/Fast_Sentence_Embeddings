#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The model card (``README.md``) of a Hugging Face folder of an fse model.

ARCHITECTURE.md, "Hub folders". The card is made from the header of the model file
actually written (its metadata and tensor shapes), so it describes exactly that file,
also the folded table of a standalone head. Its YAML front matter holds ``library_name:
fse``, the license, the base model, the language, the task and tags; every string is
written double-quoted, so no YAML 1.1 rule turns a value into a bool or a number. The
text states only facts of the file: no number that was not measured. Written without a
template engine, so the same file gives the same bytes.
"""

from __future__ import annotations

import json
import re
from typing import Any

from fse._hub import REPO_ID

# A license id as the Hub writes it (an SPDX id or "other").
LICENSE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
# A language code of the Hub: ISO 639-1 or 639-3, optionally with a region or script.
_LANGUAGE = re.compile(r"[a-z]{2,3}([-_][A-Za-z0-9]{2,8})*")
_ABOUT = "https://github.com/oborchers/Fast_Sentence_Embeddings"
_MAX_LABELS = 20


def resolve_license(
    metadata: dict[str, str], given: str | list[str] | None
) -> str | tuple[str, ...] | None:
    """The license of the card: the one the model records (``fse.license``, the license of
    the model it was converted from), or ``given`` when it records none. A given license
    that differs from a recorded one raises: fse never replaces the claim of a source."""
    recorded = _recorded_license(metadata)
    if given is None:
        return recorded
    ids = (given,) if isinstance(given, str) else tuple(given)
    if not ids:
        raise ValueError("license must be a license id or a non-empty list of them")
    for value in ids:
        if not isinstance(value, str) or not LICENSE_ID.fullmatch(value):
            raise ValueError(f"{value!r} is no license id (such as mit or apache-2.0)")
    chosen = ids[0] if len(ids) == 1 else ids
    if recorded is not None and chosen != recorded:
        raise ValueError(
            f"the model records the license {recorded!r} of its source; license={given!r} "
            "would replace it"
        )
    return chosen


def _recorded_license(metadata: dict[str, str]) -> str | tuple[str, ...] | None:
    value = metadata.get("fse.license")
    if value is None or not value.startswith("["):
        return value
    return tuple(json.loads(value))


def card(
    metadata: dict[str, str],
    shapes: dict[str, tuple[int, ...]],
    *,
    license: str | tuple[str, ...] | None,
    language: list[str] | None,
    model_name: str,
    location: str,
) -> bytes:
    """The card of the model file with ``metadata`` and tensor ``shapes``: its front matter
    and its text; ``location`` is the repository or folder the usage snippet loads."""
    if not isinstance(model_name, str) or not model_name.strip() or "\n" in model_name:
        raise ValueError(f"model_name must be a non-empty line of text, not {model_name!r}")
    for code in language or ():
        if not isinstance(code, str) or not _LANGUAGE.fullmatch(code):
            raise ValueError(f"{code!r} is no language code (such as en or de)")
    head = metadata.get("fse.head.task")
    lines = ["---", *_front_matter(metadata, head, license, language), "---", ""]
    lines += [f"# {model_name.strip()}", "", _summary(metadata, head), ""]
    lines += ["## Model", "", *_facts(metadata, shapes, head), ""]
    lines += ["## Usage", "", *_usage(head, location), ""]
    lines += ["## Source and license", "", *_source(metadata, license), ""]
    lines += ["## About fse", "", f"Made with [fse]({_ABOUT}), fast sentence embeddings.", ""]
    return "\n".join(lines).encode("utf-8")


def _quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _front_matter(
    metadata: dict[str, str], head: str | None, license: Any, language: list[str] | None
) -> list[str]:
    out = ["library_name: " + _quoted("fse")]
    if isinstance(license, str):
        out.append("license: " + _quoted(license))
    elif license:
        out += ["license:", *(f"- {_quoted(v)}" for v in license)]
    base = base_model(metadata)
    if base is not None:
        out.append("base_model: " + _quoted(base))
    if language:
        out += ["language:", *(f"- {_quoted(v)}" for v in language)]
    task = {None: "feature-extraction", "classification": "text-classification"}.get(head)
    if task is not None:
        out.append("pipeline_tag: " + _quoted(task))
    kind = {None: "sentence-embeddings", "classification": "text-classification"}
    tags = ["fse", "static-embeddings", kind.get(head, "regression")]
    out += ["tags:", *(f"- {_quoted(v)}" for v in tags)]
    return out


def base_model(metadata: dict[str, str]) -> str | None:
    """The Hub repository the model was converted from, if ``fse.source`` names one."""
    if "fse.source" not in metadata:
        return None
    repo = json.loads(metadata["fse.source"]).get("repo")
    return repo if isinstance(repo, str) and REPO_ID.fullmatch(repo) else None


def _summary(metadata: dict[str, str], head: str | None) -> str:
    steps = json.loads(metadata["fse.steps"])
    unit = "normalized to unit length" if "normalize@1" in steps else "not normalized"
    if head is None:
        return f"A static embedding model for fse: the mean of the token rows of a text, {unit}."
    what = "classifier" if head == "classification" else "regressor"
    return f"A linear {what} for fse on the mean of the token rows of a text."


def _facts(metadata: dict[str, str], shapes: dict[str, tuple[int, ...]], head: str | None):
    rows = shapes["vocab.offsets"][0] - 1
    out = [
        f"- rows (vocabulary): {rows}",
        f"- dimensions of the table: {metadata['fse.dim']} ({metadata['fse.dtype']})",
        f"- tokenizer: {_tokenizer(metadata)}",
        f"- steps per row: {', '.join(json.loads(metadata['fse.steps'])) or 'none'}",
    ]
    if "fse.fasttext" in metadata:
        out.append(f"- FastText ngrams: {metadata['fse.fasttext']}")
    if head is not None:
        out += _head_facts(metadata, shapes)
    out.append(f"- fingerprint: `{metadata['fse.fingerprint']}`")
    out.append(f"- written by: {metadata.get('fse.created_by', 'fse')}")
    return out


def _tokenizer(metadata: dict[str, str]) -> str:
    name = metadata["fse.tokenizer"]
    if "fse.unknown" not in metadata:
        return f"`{name}`"
    rules = "unknown ids dropped" if metadata["fse.unknown"] == "drop" else "unknown ids kept"
    if "fse.max_tokens" in metadata:
        rules += f", the first {metadata['fse.max_tokens']} tokens"
    if "fse.max_chars" in metadata:
        rules += f" of the first {metadata['fse.max_chars']} characters"
    return f"`{name}` ({rules})"


def _head_facts(metadata: dict[str, str], shapes: dict[str, tuple[int, ...]]) -> list[str]:
    k = shapes["head.bias"][0]
    standalone = metadata.get("fse.head.standalone") == "true"
    out = [
        f"- outputs: {k}",
        f"- link: {metadata['fse.head.link']}",
        "- trained on "
        + ("normalized" if metadata["fse.head.normalized"] == "true" else "raw")
        + " means",
    ]
    if standalone:
        out.append("- standalone: the head is folded into the table, which holds no vectors")
    if "fse.head.labels" in metadata:
        labels = json.loads(metadata["fse.head.labels"])
        shown = ", ".join(str(v) for v in labels[:_MAX_LABELS])
        more = f" and {len(labels) - _MAX_LABELS} more" if len(labels) > _MAX_LABELS else ""
        out.append(f"- labels: {shown}{more}")
    return out


def _usage(head: str | None, location: str) -> list[str]:
    if head is None:
        call = f"encoder = fse.Encoder.from_pretrained({_quoted(location)})"
        use = 'vectors = encoder.encode(["A sentence."])'
    else:
        name = "Classifier" if head == "classification" else "Regressor"
        call = f"model = fse.{name}.from_pretrained({_quoted(location)})"
        use = 'predictions = model.predict(["A sentence."])'
    return ["```", "pip install fse", "```", "", "```python", "import fse", "", call, use, "```"]


def _source(metadata: dict[str, str], license: Any) -> list[str]:
    out = []
    if "fse.source" in metadata:
        source = json.loads(metadata["fse.source"])
        where = source.get("repo") or "a local folder"
        at = f" at revision `{source['revision']}`" if source.get("revision") else ""
        out.append(f"Converted from the {source['format']} model {where}{at}.")
        if "dims" in source:
            width, kept = source["dims"]
            out.append(f"Its table of {width} dimensions was cut to the first {kept}.")
        out += _differences(source["format"])
        out.append("")
    if license is None:
        out.append("The model records no license.")
    else:
        ids = license if isinstance(license, str) else ", ".join(license)
        applies = " The license of the source model applies to its use." if out else ""
        out.append(f"License: {ids}.{applies}")
    return out


def _differences(format: str) -> list[str]:
    """Where a converted model differs from its source library (ARCHITECTURE.md,
    "encoder 1")."""
    if format == "model2vec":
        return [
            "As model2vec does, fse drops unknown token ids and applies the length rule above; "
            "`normalize@1` adds no small value to the norm, unlike model2vec."
        ]
    if format == "sentence-transformers":
        return [
            "As sentence-transformers does, fse pools every id of the tokenizer, `[UNK]` "
            "included; `normalize@1` is fse's own rule, not the Normalize module."
        ]
    return []
