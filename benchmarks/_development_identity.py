"""Source identity of a resumed run (DR-248): cells, data preparation and admission.

The stage ``source-identity`` times ``fse._encoder._resume.identify(source, None)``: the
fingerprint, the row count and the line marks of a source. Its two data cases are built
from the pinned WikiText-103 training sentences (the sentence preparation of the speed
studies, in pinned order, one per line): ``wikitext-train-file`` is a text file of
``source_bytes`` bytes (the sentences repeated until the size is reached, the last line cut
at a UTF-8 character boundary), ``wikitext-train-strings`` is the list of its first
``source_rows`` lines. Both live in the work directory, outside the checkout."""

from __future__ import annotations

import shutil
from pathlib import Path

from _development_budget import check
from _development_process import checked
from _measurement import sha256

STAGE = "source-identity"
PROFILE = "source-identity@1"
FORMS = {"wikitext-train-file": "file", "wikitext-train-strings": "strings"}
CHUNK = 1 << 24
PARAGRAPH_STEP = 2048  # how many paragraphs pass between two looks at the wall-time budget


def validate(cell):
    """The cell fields of the stage: a pinned data case, one worker, no decision gate."""
    if cell["profile"] != PROFILE or cell["corpus"] not in FORMS:
        raise ValueError(f"{STAGE} requires the profile {PROFILE} and a corpus of {list(FORMS)}")
    if cell["model"] is not None or cell["workers"] != 1 or cell["decision_gate"]:
        raise ValueError(f"{STAGE} takes no model, one worker and decision_gate=false")
    for name in ("source_bytes", "source_rows", "probe_bytes"):
        if type(cell[name]) is not int or cell[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")


def cut(data, room):
    """The largest UTF-8 character boundary of ``data`` that is at most ``room``."""
    room = min(room, len(data))
    while room < len(data) and data[room] & 0xC0 == 0x80:
        room -= 1
    return room


def _paragraphs(work, cache, deadline, whole):
    import pyarrow.parquet as pq
    from _development_data import _snapshot
    from _measurement_config import PINS
    from _speed_cells import parquet_texts

    pin = PINS["corpus"]
    pattern = f"{pin['configuration']}/train-*.parquet"
    if not whole:  # a small file needs the first part only
        pattern = f"{pin['configuration']}/train-00000-*.parquet"
    snapshot = _snapshot(
        pin["repo"], pin["revision"], work, cache, dataset=True, patterns=[pattern]
    )
    parts = sorted((snapshot / pin["configuration"]).glob("train-*.parquet"))
    if not parts:
        raise ValueError("pinned WikiText training split is missing")
    for part in parts:
        check(deadline)
        yield from parquet_texts(pq.ParquetFile(part))


def _first_cycle(out, target, work, cache, deadline):
    """Write the sentences in pinned order until the cycle ends or ``target`` is reached.
    Return (bytes written, paragraphs, sentences, whether the cycle ended)."""
    from _measurement_config import PINS
    from _speed_cells import collect_sentences

    written, paragraphs, sentences = 0, 0, 0
    for text in _paragraphs(work, cache, deadline, target > 64 * 1024**2):
        if paragraphs % PARAGRAPH_STEP == 0:
            check(deadline)
        found = []
        sentences += collect_sentences(found, text, False)
        paragraphs += 1
        for line in found:
            data = (line + "\n").encode("utf-8")
            if written + len(data) > target:
                out.write(data[: cut(data, target - written)])
                return target, paragraphs, sentences, False
            out.write(data)
            written += len(data)
    spec = PINS["corpus"]
    if paragraphs != spec["expected_paragraphs"] or sentences != spec["expected_source_positions"]:
        raise ValueError("pinned sentence corpus changed")
    return written, paragraphs, sentences, True


def _repeat(path, cycle, target, deadline):
    """Append the first ``cycle`` bytes of the file to it, again and again, until it has
    ``target`` bytes; the last block is cut at a UTF-8 character boundary."""
    written = cycle
    with path.open("ab") as out:
        while written < target:
            with path.open("rb") as source:
                left = cycle
                while left and written < target:
                    check(deadline)
                    block = source.read(min(CHUNK, left))
                    left -= len(block)
                    if written + len(block) > target:
                        block = block[: cut(block, target - written)]
                        written = target
                    else:
                        written += len(block)
                    out.write(block)
    return written


def _rows(path):
    import numpy as np

    count, last = 0, 10
    with path.open("rb") as stream:
        while block := stream.read(CHUNK):
            data = np.frombuffer(block, dtype=np.uint8)
            count += int(np.count_nonzero(data == 10))
            last = block[-1]
    return count + (last != 10)


def build_file(path, cell, work, cache, deadline):
    """The file of the cell (see the module text): its record, checked."""
    target = cell["source_bytes"]
    with path.open("xb") as out:
        written, paragraphs, sentences, ended = _first_cycle(out, target, work, cache, deadline)
    cycle = written
    if written < target:
        _repeat(path, written, target, deadline)
    size = path.stat().st_size
    if not target - 3 <= size <= target:
        raise ValueError("the generated source file has the wrong size")
    digest = sha256(path)
    rows = _rows(path)
    return dict(
        path=str(path),
        sha256=digest,
        bytes=size,
        lines=rows,
        source=dict(
            repo="Salesforce/wikitext",
            sentences_per_cycle=sentences if ended else None,
            paragraphs_read=paragraphs,
            cycle_complete=ended,
            cycle_bytes=cycle if ended else None,
            rule="pinned sentences in pinned order, one per line, repeated to source_bytes; "
            "the last line is cut at a UTF-8 character boundary (so the size is source_bytes "
            "minus at most three bytes)",
        ),
    )


def prepare_cases(cells, work, cache, deadline):
    """The records of the corpora of the stage cells; one file serves every case."""
    work = Path(work)
    cells = [cell for cell in cells if cell["stage"] == STAGE]
    if not cells:
        return {}
    sizes = {(c["source_bytes"], c["source_rows"], c["probe_bytes"]) for c in cells}
    if len(sizes) != 1:
        raise ValueError("cells of one study share source_bytes, source_rows and probe_bytes")
    cell = cells[0]
    check(deadline)
    free = shutil.disk_usage(work).free
    if free < cell["source_bytes"] + 512 * 1024**2:
        raise MemoryError("the source file would leave less than 512 MiB free")
    record = build_file(work / "source-identity-lines.txt", cell, work, cache, deadline)
    if record["lines"] < cell["source_rows"]:
        raise ValueError("the file has fewer lines than source_rows")
    result = {}
    for name in sorted({c["corpus"] for c in cells}):
        extra = dict(form=FORMS[name], rows=record["lines"])
        if FORMS[name] == "strings":
            extra["rows"] = cell["source_rows"]
        result[name] = {**record, **extra, "probe_bytes": cell["probe_bytes"]}
    return result


def verify(corpus):
    """The file of a prepared case is the one that was hashed."""
    checked(corpus["path"], corpus["sha256"])


def admission(cell, corpus, work, limit):
    """Estimated memory and disk: the mapped file is page cache, the list is resident."""
    from _development_system import memory_limit

    limit = min(filter(None, (limit, memory_limit())), default=None)
    if not limit:
        raise ValueError("RAM cannot be detected; supply --memory-limit-bytes")
    rows = corpus["rows"]
    mean = corpus["bytes"] / max(1, corpus["lines"])
    components = {"reserve": 1024**3, "blocks_and_marks": CHUNK * 4 + rows * 2}
    if corpus["form"] == "strings":
        components["python_list_and_source"] = int(rows * (mean + 80) + rows * (mean + 16) * 2)
    free = shutil.disk_usage(work).free
    need = cell["probe_bytes"] + 256 * 1024**2
    plan = dict(
        components=components,
        estimated_peak_bytes=sum(components.values()),
        memory_limit_bytes=limit,
        required_disk_bytes=need,
        available_disk_bytes=free,
        page_cache_bytes=corpus["bytes"] if corpus["form"] == "file" else 0,
    )
    plan["admitted"] = plan["estimated_peak_bytes"] <= limit * 0.85 and free >= need
    return plan
