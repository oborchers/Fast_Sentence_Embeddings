"""Private output probes: fse vectors against the native library on multilingual texts.

A functional check of class correctness before any score is read: it shows that fse computes what the
model's own library computes. For a float16 table (bekko-static-embedding-v1) the reference
is Sentence Transformers with its weights widened to float32, because fse accumulates
float16 rows in float32 while Sentence Transformers averages in float16."""

from __future__ import annotations

from _measurement_config import MODEL_PINS

TEXTS = (
    "The weather is lovely today.",
    "Das Wetter ist heute herrlich.",
    "Il fait très beau aujourd'hui.",
    "Hoy hace un tiempo estupendo.",
    "O tempo está ótimo hoje.",
    "Сегодня прекрасная погода.",
    "今日はとても良い天気です。",
    "今天天气很好。",
    "오늘 날씨가 정말 좋네요.",
    "الطقس جميل اليوم.",
    "आज मौसम बहुत अच्छा है।",
    "Bugün hava çok güzel.",
    "Hali ya hewa ni nzuri leo.",
    "Thời tiết hôm nay thật đẹp.",
    "Cuaca hari ini sangat bagus.",
    "A cat sleeps on the warm windowsill while the city wakes up below.",
    "Eine Katze schläft auf der warmen Fensterbank, während unten die Stadt erwacht.",
    "",
    "1234 5678",
    "Ünïcödé çhäräctérs ñ ß ø å",
)
REFERENCES = ("st", "st-float32", "model2vec")
# The library comparison bounds of fse for float32 tables.
RTOL, ATOL = 3e-5, 3e-6


def reference_vectors(cell, texts):
    import numpy as np

    name, revision = cell["model"], MODEL_PINS[cell["model"]]
    if cell["reference"] == "model2vec":
        from _quality.identity import model2vec_pinned

        model = model2vec_pinned(name)
        vectors = model.encode(list(texts), use_multiprocessing=False)
        return np.asarray(vectors, dtype=np.float32), str(model.embedding.dtype)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        name, revision=revision, device="cpu", truncate_dim=cell.get("truncate_dim")
    )
    if cell["reference"] == "st-float32":
        model = model.float()
    vectors = model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
    return np.asarray(vectors, dtype=np.float32), str(next(model.parameters()).dtype)


def agree(ours, theirs):
    """Refuse a probe whose vectors are not finite, whose zero rows differ or whose values
    leave the library bound: a failed probe stops the run."""
    import numpy as np

    if not (np.isfinite(ours).all() and np.isfinite(theirs).all()):
        raise ValueError("probe vectors are not finite")
    zero = np.linalg.norm(ours, axis=1) == 0
    if not np.array_equal(zero, np.linalg.norm(theirs, axis=1) == 0):
        raise ValueError("probe zero vectors differ")
    if not np.allclose(ours, theirs, rtol=RTOL, atol=ATOL):
        raise ValueError("probe vectors leave the library bound")


def probe(cell, _context=None):
    import numpy as np

    from _quality.mteb import native_model

    if cell.get("reference") not in REFERENCES:
        raise ValueError(f"probe reference must be one of {REFERENCES}")
    ours = np.asarray(native_model(cell, None).encode(list(TEXTS), workers=1), dtype=np.float32)
    theirs, weights = reference_vectors(cell, TEXTS)
    if ours.shape != theirs.shape:
        raise ValueError(f"probe shapes differ: fse {ours.shape}, reference {theirs.shape}")
    agree(ours, theirs)
    norms = np.linalg.norm(ours, axis=1) * np.linalg.norm(theirs, axis=1)
    nonzero = norms > 0
    cosine = (ours * theirs).sum(axis=1)[nonzero] / norms[nonzero]
    return {
        "status": "valid",
        "reference": cell["reference"],
        "reference_weight_dtype": weights,
        "texts": len(TEXTS),
        "zero_vectors_fse": int((np.linalg.norm(ours, axis=1) == 0).sum()),
        "zero_vectors_reference": int((np.linalg.norm(theirs, axis=1) == 0).sum()),
        "output_dim": int(ours.shape[1]),
        "max_abs_difference": float(np.abs(ours - theirs).max()),
        "min_cosine": float(cosine.min()) if cosine.size else None,
        "model_revision": MODEL_PINS[cell["model"]],
        "bound": {"rtol": RTOL, "atol": ATOL},
        "scope": "functional output probe, class correctness; no score and no speed value",
    }
