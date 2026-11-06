"""The length report as a measured candidate mode (DR-163): modes, data cases, checks.

``candidate_lengths`` of a cell is ``none`` (the default), ``summary``, ``per_row``,
``per_row-count`` or ``raise``; only the candidate arm of a cell runs it, and the base arm
and every older cell run exactly as before. Everything here that is not the timed call runs
outside the timer: the declared statistics of the data cases, the reference counts and the
comparison of a report with the independent mode."""

from __future__ import annotations

from _development_budget import check

MODES = ("none", "summary", "per_row", "per_row-count", "raise")
PROFILE = "length-report@1"
# The corpora built from WikiText rows by the rule of one model (see ``documents``). No pinned
# sentence-transformers model has a length rule in its tokenizer.json, so only the rule of
# Model2Vec is measured.
CORPORA = ("wikitext-docs-m2v", "wikitext-docs-at-cut", "wikitext-lines-uncut")
M2V_REPO = "minishlab/potion-base-8M"
REPO_OF = dict.fromkeys(CORPORA, M2V_REPO)


def validate(cell):
    if cell["candidate_lengths"] not in MODES:
        raise ValueError("unknown candidate_lengths mode")
    if cell["profile"] == PROFILE and cell["stage"] not in ("tokenizer", "encode-memory"):
        raise ValueError("length-report@1 measures the stages tokenizer and encode-memory")
    if cell["candidate_lengths"] != "none" and cell["profile"] != PROFILE:
        raise ValueError("candidate_lengths needs the profile length-report@1")
    if cell["corpus"] in REPO_OF and cell["model"] != REPO_OF[cell["corpus"]]:
        raise ValueError(
            f"the corpus {cell['corpus']} is built for the model {REPO_OF[cell['corpus']]}"
        )


def mode_for(cell, candidate=False):
    return cell["candidate_lengths"] if candidate else "none"


def keywords(mode):
    """The keywords of ``encode`` for a mode."""
    return {
        "none": {},
        "summary": {"lengths": "summary"},
        "per_row": {"lengths": "per_row"},
        "per_row-count": {"lengths": "per_row", "count_truncated": True},
        "raise": {"on_truncation": "raise"},
    }[mode]


def workspace(mode):
    """The native workspace of the isolated tokenizer stage for a mode."""
    from fse._text import new_workspace

    if mode == "none":
        return new_workspace()
    watch = 2 if mode == "per_row-count" else 1
    return new_workspace(watch, mode == "raise", mode in ("per_row", "per_row-count"))


def split(result):
    """The rows and the report of the result of ``encode``."""
    return result if isinstance(result, tuple) else (result, None)


def digest(report):
    """A digest of a report, for the receipts."""
    import numpy as np

    from fse._container._hash import checksum

    parts = {
        name: getattr(report, name)
        for name in (
            "first_row",
            "n_rows",
            "n_char_truncated",
            "n_token_truncated",
            "n_truncated",
            "n_empty",
            "tokens_total",
            "dropped_total",
            "truncated_tokens_total",
            "truncated_chars_total",
        )
    }
    arrays = {
        name: checksum(np.ascontiguousarray(getattr(report, name)))
        for name in (
            "char_truncated",
            "token_truncated",
            "empty",
            "tokens",
            "dropped_unknown",
            "truncated_tokens",
            "truncated_chars",
        )
        if getattr(report, name) is not None
    }
    return {"sums": parts, "arrays": arrays}


def check_against_independent(model, texts, mode, report, workers):
    """The report of a mode equals the one of the independent mode (the full per-row report
    with counts), field by field; outside every timer."""
    if mode in ("none", "raise") or report is None:
        return None
    _, full = model.encode(texts, lengths="per_row", count_truncated=True, workers=workers)
    sums = (
        "n_rows",
        "n_char_truncated",
        "n_token_truncated",
        "n_truncated",
        "n_empty",
        "tokens_total",
        "dropped_total",
    )
    for name in sums:
        if getattr(report, name) != getattr(full, name):
            raise ValueError(f"the length report differs from the independent one in {name}")
    if mode == "per_row-count":
        for name in ("truncated_tokens_total", "truncated_chars_total"):
            if getattr(report, name) != getattr(full, name):
                raise ValueError(f"the length report differs from the independent one in {name}")
    _same_arrays(report, full)
    return digest(report)


def _same_arrays(report, full):
    """Every array the report holds equals that of the full report, element by element."""
    import numpy as np

    for name in (
        "char_truncated",
        "token_truncated",
        "empty",
        "tokens",
        "dropped_unknown",
        "truncated_tokens",
        "truncated_chars",
    ):
        got = getattr(report, name)
        if got is not None and not np.array_equal(got, getattr(full, name)):
            raise ValueError(f"the length report differs from the independent one in {name}")


def check_reference(model, texts, mode):
    """The sums of a small report equal the independent reference count of the texts."""
    if mode == "none":
        return None
    from _development_token_reference import reference_lengths

    want = reference_lengths(model, texts)
    _, got = model.encode(texts, lengths="per_row", count_truncated=True, workers=1)
    for name, key in (
        ("n_char_truncated", "char_truncated"),
        ("n_token_truncated", "token_truncated"),
    ):
        if getattr(got, name) != sum(row[key] for row in want):
            raise ValueError(f"the length report differs from the reference count in {name}")
    for name, key in (
        ("tokens_total", "tokens"),
        ("dropped_total", "dropped_unknown"),
        ("truncated_tokens_total", "truncated_tokens"),
        ("truncated_chars_total", "truncated_chars"),
    ):
        if getattr(got, name) != sum(row[key] for row in want):
            raise ValueError(f"the length report differs from the reference count in {name}")
    if mode == "raise" and got.n_truncated:
        raise ValueError("the texts of an on_truncation=raise cell must not be cut")
    return {"texts": len(texts), "n_truncated": got.n_truncated}


def documents(rows, limit):
    """Documents of WikiText rows: the rows that are not empty after ``str.strip()``,
    whitespace-normalized as every timing text, joined with one space until the length in code
    points exceeds ``limit``; the incomplete rest is dropped."""
    docs, current, size = [], [], 0
    for row in rows:
        if not row.strip():
            continue
        text = " ".join(row.split())
        size += len(text) + (1 if current else 0)
        current.append(text)
        if size > limit:
            docs.append(" ".join(current))
            current, size = [], 0
    return docs


def declared(model, texts, expect_uncut=False):
    """The statistics a data case declares, from one summary report (outside every timer):
    the count, the lengths in code points and the shares and sums the rule cuts."""
    import numpy as np

    _, report = model.encode(texts, lengths="per_row", count_truncated=True, workers=1)
    chars = np.array([len(t) for t in texts], dtype=np.int64)
    if expect_uncut and report.n_truncated:
        raise ValueError("an uncut data case holds rows that the length rule cuts")
    return {
        "rows": len(texts),
        "codepoints_min": int(chars.min()) if len(chars) else 0,
        "codepoints_median": float(np.median(chars)) if len(chars) else 0.0,
        "codepoints_max": int(chars.max()) if len(chars) else 0,
        "n_char_truncated": report.n_char_truncated,
        "n_token_truncated": report.n_token_truncated,
        "n_truncated": report.n_truncated,
        "share_truncated": report.n_truncated / max(1, report.n_rows),
        "share_exactly_at_cut": _share_at_cut(report),
        "truncated_tokens_total": report.truncated_tokens_total,
        "truncated_chars_total": report.truncated_chars_total,
        "tokens_total": report.tokens_total,
        "max_tokens": report.max_tokens,
        "max_chars": report.max_chars,
    }


def _share_at_cut(report):
    """The share of rows whose ids end exactly at max_tokens, where the flag reads one more
    word (rows with tokens == max_tokens before the unknown ids dropped)."""
    import numpy as np

    if report.max_tokens is None:
        return 0.0
    at = (report.tokens + report.dropped_unknown == report.max_tokens) & ~report.token_truncated
    return float(np.count_nonzero(at)) / max(1, report.n_rows)


def documents_at_cut(model, rows, deadline=None):
    """Documents of exactly ``max_tokens`` ids: the longest prefix of the words of each
    document of ``rows`` with that many ids (the ids before unknown ids are dropped), where
    one exists. A sentence of this kind ends exactly at the cut, so the flag of the length
    report reads one more word of it (docs/architecture/tokenizers.md)."""
    limit = model._spec.tokenizer.max_tokens

    def ids(text):
        check(deadline)
        _, report = model.encode([text], lengths="per_row", count_truncated=True, workers=1)
        return int(report.tokens[0] + report.dropped_unknown[0] + report.truncated_tokens[0])

    found = []
    for row in rows:
        words = row.split()
        low, high = 0, len(words)  # the smallest prefix with at least limit ids
        if high == 0 or ids(row) < limit:
            continue
        while low < high:
            middle = (low + high) // 2
            if ids(" ".join(words[:middle])) >= limit:
                high = middle
            else:
                low = middle + 1
        text = " ".join(words[:low])
        if ids(text) == limit:
            found.append(text)
    return found


def uncut_mask(model, texts):
    """For each text whether the rule leaves it whole (all, if the model has no rule)."""
    import numpy as np

    _, report = model.encode(texts, lengths="per_row", workers=1)
    return np.asarray(~(report.char_truncated | report.token_truncated))
