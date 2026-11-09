"""The protocol variants of the speed entry, versioned and kept apart.

Each variant fixes how the arms of a pair are compared: whether the output is checked by bits
or by the shared tolerance, and whether a fse-against-fse control (A/A) bounds the noise. The
variant of a cell follows from its kind and its reference; an unknown combination is refused.
The tolerances are the constants of ``_speed_cells.check_output`` and are not changed here.
Standard library only (the head check imports fse).
"""

from __future__ import annotations

LIBRARIES = ("model2vec", "rust", "fastembed", "st")

# (kind, reference) -> name, check, A/A, paired, own memory plan, needs quiet.
VARIANTS = {
    **{
        ("encode", name): ("library-compare@1", "tolerance", False, True, False, True)
        for name in LIBRARIES
    },
    ("dtype", "fse-control"): ("dtype-control@1", "bits", True, True, False, True),
    ("dtype", "fse-base"): ("version-compare@1", "bits", False, True, False, True),
    ("heads", "model2vec"): ("heads-compare@1", "tolerance", False, True, False, True),
    ("heads", "fse-unfolded"): ("heads-compare@1", "tolerance", False, True, False, True),
    ("heads", "fse-separate"): ("heads-compare@1", "bits", False, True, False, True),
    ("write", None): ("write@1", "bits", False, False, True, False),
    ("search", "fse-exact"): ("search-time@1", "bits", False, False, True, True),
    ("predict", None): ("predict@1", "bits", False, False, True, False),
}


def protocol(cell):
    """The variant of one cell: ``name``, ``exact`` (compare bits) and ``control`` (A/A)."""
    try:
        name, check, control, paired, own_memory, needs_quiet = VARIANTS[
            (cell["kind"], cell["reference"])
        ]
    except KeyError:
        raise ValueError(
            f"no protocol for kind {cell['kind']!r} against reference {cell['reference']!r}"
        ) from None
    return {
        "name": name,
        "exact": check == "bits",
        "control": control,
        "paired": paired,
        "own_memory_plan": own_memory,
        "needs_quiet": needs_quiet,
    }


def validate(cells):
    """Refuse a study with a cell no variant covers, before anything runs."""
    for cell in cells:
        protocol(cell)
        if cell["kind"] == "search":
            from _harness.search import validate as validate_search

            validate_search(cell)
        if cell["kind"] == "predict":
            from _speed_predict import validate

            validate(cell)

    from _harness.search import validate_group

    search = [c for c in cells if c["kind"] == "search"]
    for model in {c["model"] for c in search}:
        validate_group([c for c in search if c["model"] == model])


def require(cells, base_python, functional=False):
    """Refuse, before any download or input, a study whose environment cannot run its cells."""
    if any(c["reference"] == "fse-base" for c in cells) and not base_python:
        raise ValueError("fse-base cells need --base-python, the installed base fse")
    if any(c["kind"] in ("heads", "predict") for c in cells):
        import fse

        if not hasattr(fse, "Heads"):
            raise ValueError(
                "head cells need an fse revision with fse.Heads; the pin in pins.json is older"
            )

    if any(c["kind"] == "search" for c in cells):
        import inspect

        from fse import ShardSink, VectorStore
        from fse._backends._cpu import CpuBackend

        if (
            "oversample" not in inspect.signature(VectorStore.search).parameters
            or "sign_bits" not in inspect.signature(ShardSink).parameters
        ):
            raise ValueError(
                "search cells require binary search and persisted sign bits; pin is older"
            )
        if (
            any(
                c["kind"] == "search" and c["prefilter"] == "shipped" and not functional
                for c in cells
            )
            and CpuBackend().prefilter_variant() is None
        ):
            raise ValueError("search-prefilter requires a shipped enabled prefilter variant")


def select(cells, study):
    if study == "search-prefilter":
        cells = [
            dict(c, relative_to="search-exact-int8")
            for c in cells
            if c["id"] in ("search-exact-int8", "search-prefilter-int8")
        ]
    elif study == "all":
        cells = [c for c in cells if c["stage"] != "search-prefilter"]
    else:
        cells = [c for c in cells if c["stage"] == study]
    return cells


def reduce_cells(contract):
    for cell in [*contract["cells"], contract["quiet_cell"]]:
        cell["production_workers"], cell["production_texts"] = cell["workers"], cell["texts"]
        cell.update(
            texts=65_536
            if cell["kind"] == "search"
            else (20_000 if cell["kind"] in ("write", "predict") else 12_000),
            workers=min(cell["workers"], 2),
            functional_only=True,
        )
