#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The arithmetic of the throughput bounds of fse (PERFORMANCE.md): no fse import and no
measurement, so the formulas can be checked on their own (tests/test_roofline.py).

Work is counted in tokens, the rows the kernel adds (token ids and FastText buckets). For
an encoder on one corpus at N workers, with seconds per token measured at one worker:
t_ser in the calling thread before and after the run, t_call in the calling thread while
the workers run (it overlaps their work), t_par in the workers, and C(N) the summed speed
of the cores in use:

- per core:  C(N) / t_par                                   (the work roof)
- Amdahl:    1 / (t_ser + max(t_par / C(N), t_call))         (Amdahl 1967; Hill and Marty 2008)
- memory:    BW / bytes_per_token                           (the no-reuse roof; Williams et al. 2009)
- combined:  1 / (t_ser + max(t_par / C(N), t_call, bytes_per_token / BW))

These are bounds under the assumptions PERFORMANCE.md lists, not guarantees: the memory
roof counts every row as read from memory, which cache hits beat, and t_par holds the
memory waits of one core.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROFILES = Path(__file__).with_name("roofline_hardware.toml")
ITEMSIZE = {"float32": 4, "float16": 2, "int8": 1}
# Bytes the kernel reads per token besides the row: the token id in the batch and the
# weight of the word (float32 each); an int8 table adds the float32 scale of the row.
ID_BYTES = 4
WEIGHT_BYTES = 4
SCALE_BYTES = 4
# The finished rows are float32.
OUT_BYTES = 4


@dataclass(frozen=True)
class Bandwidth:
    """One memory roof in GB/s (10**9 bytes), its name and where the number comes from."""

    name: str
    gb_per_s: float
    kind: str  # "spec" (the vendor's peak) or "measured"
    source: str


@dataclass(frozen=True)
class Hardware:
    name: str
    p_cores: int
    e_cores: int
    l2_cluster_bytes: int  # the L2 of one P cluster; 0 where unknown
    bandwidths: tuple[Bandwidth, ...]
    measured: bool = False  # whether numbers of this machine were measured
    e_core_ratio: float = 1.0  # the speed of an E core relative to a P core on this work
    e_core_source: str = "assumed: an E core counted as a P core, the largest possible C(N)"
    notes: tuple[str, ...] = field(default=())


@dataclass(frozen=True)
class Model:
    name: str
    dim: int
    dtype: str
    vocab: int
    normalize: bool

    @property
    def row_bytes(self) -> int:
        return self.dim * ITEMSIZE[self.dtype]

    @property
    def token_bytes(self) -> int:
        """Bytes the kernel reads for one token: its row, weight, scale (int8) and id."""
        scale = SCALE_BYTES if self.dtype == "int8" else 0
        return self.row_bytes + WEIGHT_BYTES + scale + ID_BYTES

    @property
    def table_bytes(self) -> int:
        return self.vocab * (self.token_bytes - ID_BYTES)


@dataclass(frozen=True)
class Dataset:
    """What the tool measures from a corpus with the encoder's own lookup."""

    name: str
    texts: int
    bytes: int
    tokens: int
    distinct_ids: int

    @property
    def tokens_per_text(self) -> float:
        return self.tokens / self.texts


@dataclass(frozen=True)
class Stages:
    """Seconds of one pass over the corpus at one worker, per stage. The first four run
    in the calling thread: join, plan and encode_rest before or after the run (serial),
    caller while the workers run (overlapping them); the others run in the workers."""

    join: float  # the strings joined into one buffer (StringSource; up to 8 threads)
    plan: float  # the scan into chunks
    caller: float  # the rest of infer outside the tasks: reserve, commit, the result
    encode_rest: float  # Encoder.encode around infer
    tokenize: float  # stages 1 and 2: tokenizer and lookup into batches
    pool: float  # the weighted mean
    normalize: float  # normalize@1
    task_rest: float  # the rest of the tasks: checks, views, Python

    @property
    def serial(self) -> float:
        """The calling thread's time before and after the run, which no worker overlaps."""
        return self.join + self.plan + self.encode_rest

    @property
    def parallel(self) -> float:
        return self.tokenize + self.pool + self.normalize + self.task_rest


def bytes_per_token(model: Model, data: Dataset) -> float:
    """Bytes per token if no read is served from a cache: what the kernel reads for the
    token, and the output row of its text written once, spread over the text's tokens
    (normalize reads and writes it again while it is still in the cache)."""
    return model.token_bytes + OUT_BYTES * model.dim / data.tokens_per_text


def cores(hw: Hardware, workers: int) -> float:
    """C(N): the summed speed of the cores ``workers`` threads use, in P cores; the
    threads fill the P cores first, threads beyond all cores add nothing."""
    if workers < 1:
        raise ValueError(f"workers must be positive, not {workers}")
    p = min(workers, hw.p_cores)
    e = min(max(workers - hw.p_cores, 0), hw.e_cores)
    return p + hw.e_core_ratio * e


@dataclass(frozen=True)
class Bounds:
    """Bounds in tokens per second at ``workers``; the memory roofs by their name."""

    workers: int
    core: float
    amdahl: float
    memory: dict[str, float]  # by the name of the roof
    combined: dict[str, float]
    binding: dict[str, str]  # "serial", "work", "caller" or "memory"


def bounds(
    hw: Hardware,
    model: Model,
    data: Dataset,
    stages: Stages,
    workers: int,
    plan: float | None = None,
) -> Bounds:
    """The bounds at ``workers``; ``plan`` replaces the scan time of one worker by the
    one measured at ``workers`` (the scan makes more chunks there)."""
    serial = stages.serial + (0.0 if plan is None else plan - stages.plan)
    t_ser = serial / data.tokens
    t_par = stages.parallel / data.tokens
    t_call = stages.caller / data.tokens
    c = cores(hw, workers)
    work = t_par / c
    memory, combined, binding = {}, {}, {}
    for roof in hw.bandwidths:
        per_token = bytes_per_token(model, data) / (roof.gb_per_s * 1e9)
        memory[roof.name] = 1 / per_token
        terms = {"work": work, "caller": t_call, "memory": per_token}
        combined[roof.name] = 1 / (t_ser + max(terms.values()))
        binding[roof.name] = _binding(t_ser, terms)
    amdahl = 1 / (t_ser + max(work, t_call))
    return Bounds(workers, c / t_par, amdahl, memory, combined, binding)


def _binding(serial: float, terms: dict[str, float]) -> str:
    """What binds: the serial part if it takes longer than the slowest overlapping term."""
    name = max(terms, key=lambda k: terms[k])
    return "serial" if serial >= terms[name] else name


def serial_fraction(stages: Stages) -> float:
    """The share of the time at one worker that no worker overlaps."""
    return stages.serial / (stages.serial + stages.caller + stages.parallel)


def share(measured: float, bound: float) -> float:
    """The measured rate as a share of a bound; above 1 an assumption of it failed."""
    if not bound > 0 or not math.isfinite(bound):
        raise ValueError(f"a bound must be positive and finite, not {bound}")
    return measured / bound


# --- hardware profiles --------------------------------------------------------------------


def load_profiles(path: Path = PROFILES) -> dict[str, Hardware]:
    """The profiles of ``path``; a number without its source raises."""
    with path.open("rb") as f:
        raw = tomllib.load(f)
    return {name: profile(name, entry) for name, entry in raw.items()}


def _need(entry: dict, key: str, name: str):
    if key not in entry:
        raise ValueError(f"hardware profile {name!r} has no {key!r}")
    return entry[key]


def _roof(name: str, roof: dict) -> Bandwidth:
    if not roof.get("name"):
        raise ValueError(f"a bandwidth of hardware profile {name!r} has no name")
    if not roof.get("source"):
        raise ValueError(f"a bandwidth of hardware profile {name!r} has no source")
    if roof.get("kind") not in ("spec", "measured"):
        raise ValueError(f"a bandwidth of hardware profile {name!r} must be spec or measured")
    return Bandwidth(roof["name"], float(roof["gb_per_s"]), roof["kind"], roof["source"])


def profile(name: str, entry: dict) -> Hardware:
    """One profile from its TOML table."""
    for key in ("p_cores", "e_cores", "l2_cluster_bytes", "topology_source"):
        _need(entry, key, name)
    roofs = tuple(_roof(name, roof) for roof in _need(entry, "bandwidth", name))
    if len({r.name for r in roofs}) != len(roofs):
        raise ValueError(f"the bandwidths of hardware profile {name!r} need distinct names")
    ratio = entry.get("e_core_ratio")
    extra = {}
    if ratio is not None:
        if not entry.get("e_core_source"):
            raise ValueError(f"the e_core_ratio of hardware profile {name!r} has no source")
        extra = {"e_core_ratio": float(ratio), "e_core_source": entry["e_core_source"]}
    return Hardware(
        name,
        int(entry["p_cores"]),
        int(entry["e_cores"]),
        int(entry["l2_cluster_bytes"]),
        roofs,
        measured=bool(entry.get("measured", False)),
        notes=(entry["topology_source"], *entry.get("notes", ())),
        **extra,
    )
