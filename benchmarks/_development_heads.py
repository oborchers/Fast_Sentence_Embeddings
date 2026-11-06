"""PR-1 head adapters inside speed.py's existing local-development-v2 controller.

The profile fixes one four-output softmax head, without training. Both installed
builds support it. Multilabel correctness stays in the product conformance suite.
The stage head-labels times model.predict (int64 labels). Its option rule is linear@1
(no temperature, in both builds) or linear@2 (temperature=1.0: the same weight bits,
only the label rule differs; the candidate only).
"""

from __future__ import annotations

from typing import NamedTuple

PROFILE = "linear-heads@1"
STAGES = ("head-folded", "head-with-vectors", "head-labels")
RULES = ("linear@1", "linear@2")
SPEC = {"outputs": 4, "seed": 17, "link": "softmax", "normalized": False}


BATCH_STAGES = ("head-folded", "head-with-vectors", "head-group-folded", "head-group-mixed")


class HeadOutput(NamedTuple):
    probabilities: object
    vectors: object


def validate_batches(cell):
    """The cell field ``batch_rows`` (DR-243, Umwege über fse.Heads bei einzelnen Köpfen
    entfernen): None (one call on all texts), or the rows of each batch of a stream through
    ``predict_batches``, for the stages that return the outputs and the vectors in memory."""
    rows = cell["batch_rows"]
    if rows is None:
        return
    if type(rows) is not int or rows < 1:
        raise ValueError("batch_rows is a positive integer, or null")
    if cell["stage"] not in BATCH_STAGES:
        raise ValueError(f"batch_rows belongs to the stages {', '.join(BATCH_STAGES)}")
    modes = (cell["base_heads_mode"], cell["candidate_heads_mode"])
    if cell["stage"].startswith("head-group") and modes != ("one-pass", "one-pass"):
        raise ValueError("batch_rows of a head group needs both arms in one-pass mode")


def chunks(texts, rows):
    """``texts`` as the batches of a stream of ``rows`` texts each."""
    return [texts[start : start + rows] for start in range(0, len(texts), rows)]


def operation(encoder, texts, job, *, backend="cpu"):
    import numpy as np

    from fse import Classifier

    if job.get("profile") != PROFILE or job["stage"] not in STAGES:
        raise ValueError("a linear-heads@1 stage is required")
    rng = np.random.default_rng(SPEC["seed"])
    weight = rng.standard_normal((SPEC["outputs"], encoder.output_dim)).astype(np.float32)
    bias = rng.standard_normal(SPEC["outputs"]).astype(np.float32)
    labels_stage = job["stage"] == "head-labels"
    extra = {"temperature": 1.0} if labels_stage and job.get("rule") == "linear@2" else {}
    model = Classifier.from_arrays(
        encoder,
        weight,
        bias,
        # Integer labels keep the labels output a plain int64 array to byte-check.
        labels=list(range(SPEC["outputs"]))
        if labels_stage
        else [f"class_{index}" for index in range(SPEC["outputs"])],
        link=SPEC["link"],
        normalized=SPEC["normalized"],
        **extra,
    )

    def predict():
        if job.get("batch_rows"):
            return _streamed(model, texts, job, backend)
        if labels_stage:
            return HeadOutput(model.predict(texts, workers=job["workers"], backend=backend), None)
        if job["stage"] == "head-folded":
            return HeadOutput(
                model.predict_proba(texts, workers=job["workers"], backend=backend), None
            )
        (batch,) = model.predict_batches(
            (texts,), output="proba", return_vectors=True, workers=job["workers"], backend=backend
        )
        return HeadOutput(batch.outputs, batch.vectors)

    # Prepare the lazy folded table outside every timer. Verify the complete public
    # output shape once here; the controller checks every subsequent output's bytes.
    _check_shapes(predict(), job, len(texts), encoder.output_dim)
    return predict


def _check_shapes(prepared, job, rows, dim):
    if job["stage"] == "head-labels":
        if prepared.probabilities.shape != (rows,) or prepared.probabilities.dtype.kind != "i":
            raise ValueError("head label shape or dtype differs from the fixed profile")
    elif prepared.probabilities.shape != (rows, SPEC["outputs"]):
        raise ValueError("head probability shape differs from the fixed profile")
    if job["stage"] == "head-with-vectors" and prepared.vectors.shape != (rows, dim):
        raise ValueError("head vector shape differs from the fixed profile")


def _streamed(model, texts, job, backend):
    """The same outputs as one call, from ``predict_batches`` over batches of ``batch_rows``."""
    import numpy as np

    vectors = job["stage"] == "head-with-vectors"
    got = list(
        model.predict_batches(
            chunks(texts, job["batch_rows"]),
            output="proba",
            return_vectors=vectors,
            workers=job["workers"],
            backend=backend,
        )
    )
    return HeadOutput(
        np.concatenate([b.outputs for b in got]),
        np.concatenate([b.vectors for b in got]) if vectors else None,
    )


def probe(job, texts):
    """Integrate head stages into the shared untimed CPU/reference adapter preflight."""
    from _development_encode import output_identity
    from _development_process import checked

    from fse import Encoder

    artifact = job["artifact"]
    encoder = Encoder.from_pretrained(
        checked(artifact["path"], artifact["sha256"]), normalize=False, mmap=True
    )
    got = output_identity(operation(encoder, texts, job)())
    expected = output_identity(operation(encoder, texts, job, backend="numpy")())
    if got != expected:
        raise ValueError("preflight head shape, dtype or bits differ from numpy reference")
    return got


def nonregression(cell, result):
    """An explicit noise-limited regression check, never a speedup/keep decision."""
    if cell["profile"] != PROFILE:
        return result
    aa = result["aa"]
    main, confirmation = result["comparison"], result["confirmation"]
    limit = cell["noise_multiplier"] * aa["factor"]["iqr"]
    quiet = aa["factor"]["iqr"] < cell["target_aa_iqr"]
    differences = [main["time_saving"], confirmation["time_saving"]]
    passed = quiet and min(differences) >= -limit
    status = "within_aa_noise" if passed else ("regression" if quiet else "inconclusive_noise")
    return {
        **result,
        "nonregression": {
            "status": status,
            "passed": passed,
            "maximum_time_loss": limit,
            "comparison_saving": differences[0],
            "confirmation_saving": differences[1],
            "claim": "noise-limited local indication, not proof of zero slowdown",
        },
    }


def profile(job):
    """Untimed reproduction for externally coordinated CPU/allocation/RSS profiling."""
    from _development_encode import output_identity
    from _development_stages import inputs
    from _development_system import apply_affinity

    policy = {"affinity": apply_affinity(job.get("cpu_affinity"))}
    encoder, texts = inputs(job)
    predict = operation(encoder, texts, job)
    iterations = job["iterations"]
    if type(iterations) is not int or iterations < 1:
        raise ValueError("positive integral profile iterations required")
    print("profile_ready", flush=True)
    output = None
    for _ in range(iterations):
        output = None
        output = predict()
    return {
        "diagnostic_profile": True,
        "valid_for_comparison": False,
        "profile": PROFILE,
        "head_spec": SPEC,
        "worker_path": job["stage"],
        "policy": policy,
        "operation_repetitions": iterations,
        "output": output_identity(output),
    }
