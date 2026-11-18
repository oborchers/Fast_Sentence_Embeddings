"""Bounded byte-balanced refinement for the per-call native team only."""

from itertools import chain, pairwise

from fse._text import Chunk
from fse._text._plan import byte_cuts

from ._tasks import Task


def refine(source, tasks, workers, limit):
    """Keep original boundaries, or return None before creating an excess task.

    A task targets ceil(total bytes / (4 * workers)) bytes. Byte quantiles skip
    indivisible sentences, including runs of zero-byte rows. Original row bounds
    still apply because refinement never joins tasks. Cuts are generated lazily:
    at most ``limit`` new tasks and constant temporary cut state are retained.
    """
    target = max(1, -(-source.nbytes // (4 * workers)))
    refined = []
    for task in tasks:
        chunk = task.chunk
        offsets = source._offsets[chunk.start : chunk.end + 1]
        size = int(offsets[-1]) - int(offsets[0])
        pieces = max(1, -(-size // target))
        cuts = chain((0,), byte_cuts(offsets, pieces), (task.n_rows,))
        for start, end in pairwise(cuts):
            if len(refined) == limit:
                return None
            refined.append(
                Task(
                    task.source,
                    Chunk(
                        chunk.start + start, chunk.start + end, task.first_row + start, end - start
                    ),
                )
            )
    return refined
