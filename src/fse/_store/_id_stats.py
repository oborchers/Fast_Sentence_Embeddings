"""Bounded statistics of a document-ID column, including cross-block ordering."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from fse._container import CorruptArtifactError
from fse._contract import UnsupportedArtifactError

ID_KEYS = ("fse.row_ids_min", "fse.row_ids_max", "fse.row_ids_sorted")
MODE_KEYS = ("fse.row_ids", "fse.resume")
_SIGNED = re.compile(r"(?:0|-?[1-9][0-9]*)\Z")


@dataclass
class IdStats:
    """The statistics of the committed prefix, without keeping its IDs in memory."""

    minimum: int | None = None
    maximum: int | None = None
    sorted: bool = True
    last: int | None = None

    def update(self, ids: np.ndarray) -> None:
        if not len(ids):
            return
        lo, hi = int(ids.min()), int(ids.max())
        self.minimum = lo if self.minimum is None else min(self.minimum, lo)
        self.maximum = hi if self.maximum is None else max(self.maximum, hi)
        self.sorted = self.sorted and (self.last is None or self.last <= int(ids[0]))
        self.sorted = self.sorted and bool(np.all(ids[:-1] <= ids[1:]))
        self.last = int(ids[-1])

    def metadata(self) -> dict[str, str]:
        return dict(
            zip(
                ID_KEYS,
                (str(self.minimum), str(self.maximum), str(self.sorted).lower()),
                strict=True,
            )
        )


def check_mode(metadata: dict[str, Any], file: Any) -> None:
    """Recognize future capabilities before applying a closed known schema."""
    for key, supported in (("fse.row_ids", ("none", "int64")), ("fse.resume", ("none",))):
        value = metadata.get(key)
        if isinstance(value, str) and value not in supported:
            raise UnsupportedArtifactError(f"{file}: unsupported {key} {value!r}")


def parsed_stats(metadata: dict[str, str], file: Any) -> IdStats:
    values = [metadata.get(key, "") for key in ID_KEYS]
    if any(
        len(v) > 20 or not _SIGNED.fullmatch(v) or not -(2**63) <= int(v) < 2**63
        for v in values[:2]
    ):
        raise CorruptArtifactError(f"{file}: row_ids bounds must be canonical signed int64 values")
    lo, hi = map(int, values[:2])
    if lo > hi or values[2] not in ("true", "false"):
        raise CorruptArtifactError(f"{file}: invalid row_ids bounds or sorted flag")
    return IdStats(lo, hi, values[2] == "true")


def verify_stats(ids: np.ndarray, expected: IdStats, name: str) -> None:
    actual = IdStats()
    for begin in range(0, len(ids), (1 << 20) // 8):
        actual.update(ids[begin : begin + (1 << 20) // 8])
    if actual.metadata() != expected.metadata():
        raise CorruptArtifactError(f"{name}: row_ids statistics do not match the stored IDs")
