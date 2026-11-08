"""The functional stage: a real reduction of the input, with the facts that make it auditable.

A functional run proves that a study works. It is no speed evidence. It therefore runs on a
smaller input that it actually builds, and it records what it did: the original size, the
reduced size, the rule that chose the rows, the seed and a digest of the chosen positions. The
rule is deterministic and independent of the Python version and of any array library.
Standard library only.
"""

from __future__ import annotations

import hashlib

RULE = "hash-sample@1"
SEED = 0
NOTE = "functional run: no speed evidence"


def select(count, keep, seed=SEED):
    """The sorted positions of ``keep`` rows of ``count``: those with the smallest SHA-256 of
    ``seed:position``. All positions when ``keep`` is not below ``count``."""
    if type(count) is not int or type(keep) is not int or count < 0 or keep <= 0:
        raise ValueError("a row count and a positive number of rows to keep are required")
    if keep >= count:
        return list(range(count))

    def key(position):
        return hashlib.sha256(f"{seed}:{position}".encode()).digest()

    return sorted(sorted(range(count), key=key)[:keep])


def digest(positions):
    value = hashlib.sha256()
    for position in positions:
        value.update(position.to_bytes(8, "little"))
    return value.hexdigest()


def reduction(original, positions, *, seed=SEED, floor=0, reason=None):
    """The metadata of one reduction (``reason`` says why nothing was reduced, else ``None``)."""
    return {
        "rule": RULE,
        "seed": seed,
        "original_rows": original,
        "reduced_rows": len(positions),
        "reduced": len(positions) < original,
        "floor_rows": floor,
        "selection_sha256": digest(positions),
        "reason": reason,
        "note": NOTE,
    }
