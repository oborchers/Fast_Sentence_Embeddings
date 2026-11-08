"""Stage-specific admission before allocating any repeated development workload."""

from __future__ import annotations

import shutil
from pathlib import Path

from _development_ids import document_profile
from _development_system import LABEL
from _harness import resources


def disk_reserve(cell):
    return (30 if document_profile(cell) else 1) * 1024**3


def check_write_disk(job):
    if job["stage"] != "encode-sink" or not document_profile(job):
        return
    width = {"float32": 4, "float16": 2, "int8": 1}[job["output_dtype"]]
    row_bytes = job["dimensions"] * width + 4 + 8
    if job["output_dtype"] == "int8":
        row_bytes += 4
    required = job["texts"] * row_bytes * job["iterations"] * 2 + disk_reserve(job)
    if shutil.disk_usage(job["work"]).free < required:
        raise MemoryError("document-ID write group would leave less than 30 GiB free")


def _group_components(cell, prepared, count, output, components):
    """n heads: their folded tables (one shared table or n separate ones) beside the table,
    the probabilities of every head in both live results, and with vectors the full means."""
    if cell["stage"] not in ("head-group-folded", "head-group-mixed", "head-group-score-sink"):
        return
    table = prepared["models"][cell["model"]]["artifacts"][cell["table_dtype"]]
    components["head_table_reserve"] = Path(table["path"]).stat().st_size * cell["heads"]
    components["head_probabilities_and_copies"] = count * 4 * 4 * 2 * cell["heads"]
    components["output_and_copies"] = output * 2 if cell["return_vectors"] else 0
    if cell["stage"] == "head-group-score-sink":
        # The reference run (the logits of every head and their side-by-side copy), the IDs
        # and the sink matrix of the window.
        components["score_reference_and_ids"] = count * (cell["heads"] * 16 * 3 + 16)
        if cell["sinks"] != "scores":
            # The vectors of the reference run beside the sink's own buffers, and with both
            # stores a second set of writer queues (three threads of 64 MiB) outside the window.
            components["vector_reference"] = output
            components["second_writer_queues"] = 3 * 64 * 1024**2 * (cell["sinks"] == "both")


def _score_sink_disk(cell, count, dim):
    """The stores of the stage head-group-score-sink: the scores of every head (four logits of
    four bytes), the vectors (a float32 row, its inverse norm) and the eight-byte IDs of each
    row, before and after the sync."""
    if cell["stage"] != "head-group-score-sink":
        return 0
    row = (cell["heads"] * 16 + 8) * (cell["sinks"] != "vectors")
    row += (dim * 4 + 4 + 8) * (cell["sinks"] != "scores")
    return count * row * cell["iterations"] * 2


def admission(cell, prepared, count, work, requested_limit=None, kind="production"):
    corpus = prepared["corpora"][cell["corpus"]]
    model = prepared["models"][cell["model"]]
    limit = resources.require_limit(requested_limit)
    mean_bytes = corpus["timing_utf8_bytes"] / corpus["texts"]
    dim = model["dimensions"]
    output = count * dim * 4
    components = {
        "reserve": 1024**3,
        "model": Path(model["artifacts"][cell["table_dtype"]]["path"]).stat().st_size * 2,
        "python_texts": count * 80 + corpus["timing_utf8_bytes"] * 4,
        "utf8_and_offsets": int(count * (mean_bytes * 2 + 16)),
        "tokenizer_output": int(count * (mean_bytes * 4 + 16)),
        "output_and_copies": output * 2
        if cell["stage"] in ("pooling", "encode-memory")
        else min(output, 64 * 1024**2),
        "workers": cell["workers"] * 64 * 1024**2,
    }
    if cell["profile"] == "linear-heads@1" and cell["stage"] in (
        "head-folded",
        "head-with-vectors",
        "head-labels",
    ):
        # Four probabilities per row, two live public results plus bounded workspaces.
        # Folded mode retains its prepared word table; vectors mode holds full means.
        table = prepared["models"][cell["model"]]["artifacts"][cell["table_dtype"]]
        components["head_table_reserve"] = Path(table["path"]).stat().st_size
        components["head_probabilities_and_copies"] = count * 4 * 4 * 2
        components["output_and_copies"] = output * 2 if cell["stage"] == "head-with-vectors" else 0
    _group_components(cell, prepared, count, output, components)
    if (cell.get("base_doc_ids", "none"), cell.get("candidate_doc_ids", "none")) != (
        "none",
        "none",
    ):
        components["document_ids_and_verification"] = count * 16
    phases = None
    if document_profile(cell) and cell["stage"] == "encode-sink":
        from _development_file_resources import file_components

        components, phases = file_components(cell, count, dim, mean_bytes, components)
    disk = int(count * (mean_bytes + 1) + disk_reserve(cell))
    if cell["stage"] == "encode-sink":
        disk += (output + count * 8) * cell["iterations"] * 2

    elif cell["stage"] in ("store-open", "store-search"):
        components["mapped_store_and_filter"] = output + count * 48
        disk += (output + count * 8) * 2
    disk += _score_sink_disk(cell, count, dim)
    return resources.assess(
        components,
        limit=limit,
        work=work,
        disk=disk,
        kind=kind,
        **LABEL,
        texts=count,
        operation_repetitions=cell["iterations"],
        sequential_phases=phases,
    )


def plans(args, cells, prepared, kind):
    """The plan of every cell of a study at the size of ``prepared``: the repetitions are the
    declared ones (calibration may grow them, and admits again before every sizing step)."""
    from _development_identity import STAGE
    from _development_identity import admission as identity_admission

    result = {}
    for cell in cells:
        corpus = prepared["corpora"][cell["corpus"]]
        if cell["stage"] == STAGE:
            result[cell["id"]] = identity_admission(
                cell, corpus, args.work, args.memory_limit_bytes, kind
            )
            continue
        count = corpus["texts"] * cell["resident_cycles"]
        result[cell["id"]] = admission(
            cell, prepared, count, args.work, args.memory_limit_bytes, kind
        )
    return result
