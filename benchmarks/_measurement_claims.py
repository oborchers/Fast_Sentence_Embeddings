"""Claim classes of a measurement and the former one-letter names they replace.

Results written before DR-294 (Messklassen mit lesbaren Namen statt Einzelbuchstaben
benennen) store the class as one letter, for example ``"class": "K"``. Those files stay
unchanged. Readers map an explicit class value to its name, so old and new rows of the same
configuration group together. A missing class stays unknown (``None``): no other field, such
as a decision label like ``descriptive_only``, determines a class.
"""

from __future__ import annotations

CLAIM_CLASSES = ("correctness", "descriptive", "regression", "optimization", "publication")
# The former one-letter values and the class each one names (the only aliases).
ALIASES = {
    "K": "correctness",
    "D": "descriptive",
    "R": "regression",
    "O": "optimization",
    "P": "publication",
}


def claim_class(value):
    """The name of an explicit class value, a former one-letter value included.

    ``None`` (no class recorded) stays ``None``; any other value raises ``ValueError``.
    """
    if value is None or value in CLAIM_CLASSES:
        return value
    if isinstance(value, str) and value in ALIASES:
        return ALIASES[value]
    raise ValueError(
        f"unknown claim class {value!r}: expected one of {CLAIM_CLASSES} "
        f"or a former one-letter value of {tuple(ALIASES)}"
    )


def grouping_fields(row, keys):
    """The fields ``keys`` of a result row, with its class by name, as a grouping key.

    Only the class is mapped; every other field (machine, protocol, configuration) is taken
    as recorded, so rows that differ in any of them stay in separate groups.
    """
    fields = {k: row.get(k) for k in keys}
    if "class" in fields:
        fields["class"] = claim_class(fields["class"])
    return fields
