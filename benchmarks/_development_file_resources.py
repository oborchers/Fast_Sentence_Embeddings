"""Phased admission for the document-ID profile's bounded FileSource write path."""

from __future__ import annotations


def check_limits():
    from fse._runtime._pipeline import WINDOW_BYTES
    from fse._store._writer import MAX_PENDING_BYTES

    if WINDOW_BYTES > 512 * 1024**2 or MAX_PENDING_BYTES > 64 * 1024**2:
        raise ValueError("installed runtime exceeds the document-ID file admission bounds")


def file_components(cell, count, dim, mean_bytes, components):
    # FileSource maps the input; it does not join or tokenize all rows at once.
    # Pipeline.WINDOW_BYTES is 512 MiB and both writer queues allow 64 MiB.
    # Keep a separate 1 GiB allowance per worker for native token workspaces,
    # lookup scratch and allocator retention. These are estimates, not measured RSS.
    source = int(count * (mean_bytes + 1))
    encoding = source + 512 * 1024**2 + 2 * 64 * 1024**2 + cell["workers"] * 1024**3
    # Every shard is mapped during full verification. Include the complete
    # float32-sized output even for smaller formats, inverse norms, scales and IDs.
    verification = count * (dim * 4 + 16)
    common = {
        name: value
        for name, value in components.items()
        if name not in ("utf8_and_offsets", "tokenizer_output", "output_and_copies")
    }
    common["file_encode_or_full_store_verification"] = max(encoding, verification)
    return common, {
        "scope": "document-ids@1 encode-sink FileSource only",
        "mapped_input_bytes": source,
        "encoding_phase_bytes": encoding,
        "verification_phase_bytes": verification,
        "combination": "common live state plus the larger sequential phase",
        "runtime_window_bytes": 512 * 1024**2,
        "writer_queue_bytes": 2 * 64 * 1024**2,
        "native_workspace_allowance_per_worker": 1024**3,
    }
