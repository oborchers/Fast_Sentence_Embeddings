"""Exact planning cuts from an untimed diagnostic, including dedup and tail splits."""

from unittest.mock import patch


class Cuts:
    def __init__(self):
        self.rows = []

    def install(self, stack):
        from fse._text import _scan

        offset_cuts = _scan.offset_cuts
        split_tail = _scan._Buffered.split_tail

        def cutting(offsets, chunk_bytes, chunk_rows):
            cuts = offset_cuts(offsets, chunk_bytes, chunk_rows)
            total = int(offsets[-1]) - int(offsets[0])
            pieces = max(1, -(-total // chunk_bytes), -(-(len(offsets) - 1) // chunk_rows))
            self.rows.append(
                {
                    "operation": "offset_cuts",
                    "chunk_bytes": chunk_bytes,
                    "chunk_rows": chunk_rows,
                    "requested_pieces": pieces,
                    "byte_targets": [
                        int(offsets[0]) + j * total // pieces for j in range(1, pieces)
                    ],
                    "cuts": cuts.tolist(),
                    "cut_byte_offsets": [int(offsets[i]) for i in cuts],
                    "scope": "actual deduplicated cuts after byte targets and row bounds",
                }
            )
            return cuts

        def splitting(source, chunks, workers):
            result = split_tail(source, chunks, workers)
            self.rows.append(
                {
                    "operation": "split_tail",
                    "workers": workers,
                    "before": _chunks(source, chunks),
                    "after": _chunks(source, result),
                }
            )
            return result

        stack.enter_context(patch.object(_scan, "offset_cuts", cutting))
        stack.enter_context(patch.object(_scan._Buffered, "split_tail", splitting))


def _chunks(source, chunks):
    return [
        {
            "start": c.start,
            "end": c.end,
            "first_row": c.first_row,
            "rows": c.n_sentences,
            "bytes": source._span_bytes(c),
        }
        for c in chunks
    ]
