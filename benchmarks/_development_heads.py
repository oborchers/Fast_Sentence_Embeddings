"""PR-1 head adapters inside speed.py's existing local-development-v2 controller.

The profile fixes one four-output softmax head, without training. Both installed
builds support it. Multilabel correctness stays in the product conformance suite.
"""

from __future__ import annotations

from typing import NamedTuple

PROFILE = "linear-heads@1"
STAGES = ("head-folded", "head-with-vectors")
SPEC = {"outputs": 4, "seed": 17, "link": "softmax", "normalized": False}


class HeadOutput(NamedTuple):
    probabilities: object
    vectors: object


def operation(encoder, texts, job):
    import numpy as np

    from fse import Classifier

    if job.get("profile") != PROFILE or job["stage"] not in STAGES:
        raise ValueError("a linear-heads@1 stage is required")
    rng = np.random.default_rng(SPEC["seed"])
    weight = rng.standard_normal((SPEC["outputs"], encoder.output_dim)).astype(np.float32)
    bias = rng.standard_normal(SPEC["outputs"]).astype(np.float32)
    model = Classifier.from_arrays(
        encoder,
        weight,
        bias,
        labels=[f"class_{index}" for index in range(SPEC["outputs"])],
        link=SPEC["link"],
        normalized=SPEC["normalized"],
    )

    def predict():
        if job["stage"] == "head-folded":
            return HeadOutput(model.predict_proba(texts, workers=job["workers"]), None)
        (batch,) = model.predict_batches(
            (texts,), output="proba", return_vectors=True, workers=job["workers"]
        )
        return HeadOutput(batch.outputs, batch.vectors)

    # Prepare the lazy folded table outside every timer. Verify the complete public
    # output shape once here; the controller checks every subsequent output's bytes.
    prepared = predict()
    if prepared.probabilities.shape != (len(texts), SPEC["outputs"]):
        raise ValueError("head probability shape differs from the fixed profile")
    if job["stage"] == "head-with-vectors" and prepared.vectors.shape != (
        len(texts),
        encoder.output_dim,
    ):
        raise ValueError("head vector shape differs from the fixed profile")
    return predict


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
