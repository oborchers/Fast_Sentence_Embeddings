"""The protocol variants of the speed entry, versioned and kept apart.

Each variant fixes how the arms of a pair are compared: whether the output is checked by bits
or by the shared tolerance, and whether a fse-against-fse control (A/A) bounds the noise. The
variant of a cell follows from its kind and its reference; an unknown combination is refused.
The tolerances are the constants of ``_speed_cells.check_output`` and are not changed here.
Standard library only (the head check imports fse).
"""

from __future__ import annotations

LIBRARIES = ("model2vec", "rust", "fastembed", "st")

# (kind, reference) -> (name, output check, A/A control). A reference not listed is refused.
VARIANTS = {
    **{("encode", name): ("library-compare@1", "tolerance", False) for name in LIBRARIES},
    ("dtype", "fse-control"): ("dtype-control@1", "bits", True),
    ("dtype", "fse-base"): ("version-compare@1", "bits", False),
    ("heads", "model2vec"): ("heads-compare@1", "tolerance", False),
    ("heads", "fse-unfolded"): ("heads-compare@1", "tolerance", False),
    ("heads", "fse-separate"): ("heads-compare@1", "bits", False),
    ("write", None): ("write@1", "bits", False),
}


def protocol(cell):
    """The variant of one cell: ``name``, ``exact`` (compare bits) and ``control`` (A/A)."""
    try:
        name, check, control = VARIANTS[(cell["kind"], cell["reference"])]
    except KeyError:
        raise ValueError(
            f"no protocol for kind {cell['kind']!r} against reference {cell['reference']!r}"
        ) from None
    return {"name": name, "exact": check == "bits", "control": control}


def validate(cells):
    """Refuse a study with a cell no variant covers, before anything runs."""
    for cell in cells:
        protocol(cell)


def require(cells, base_python):
    """Refuse, before any download or input, a study whose environment cannot run its cells."""
    if any(c["reference"] == "fse-base" for c in cells) and not base_python:
        raise ValueError("fse-base cells need --base-python, the installed base fse")
    if any(c["kind"] == "heads" for c in cells):
        import fse

        if not hasattr(fse, "Heads"):
            raise ValueError(
                "head cells need an fse revision with fse.Heads; the pin in pins.json is older"
            )
