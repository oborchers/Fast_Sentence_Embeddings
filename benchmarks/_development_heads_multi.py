"""Several heads in one pass, inside speed.py's local-development-v2 controller.

The profile fixes n linear heads without training (float32 PCG64, seed 17 + the head's
index, four softmax outputs each, the output ``proba``). Both arms run the candidate build:
the arm "separate" calls every head on its own, the arm "one-pass" calls ``fse.Heads`` once,
and the outputs of both are byte-identical per head. The folded tables and the plan of the
group are made before any timer. Stage head-group-folded: n heads on raw means; stage
head-group-mixed: half of them on raw means and half on normalized means. M1 local
indication only, no README speed claims (docs/benchmarks/measurement.md).
"""

from __future__ import annotations

from typing import NamedTuple

PROFILE = "linear-heads-multi@1"
STAGES = ("head-group-folded", "head-group-mixed")
MODES = ("separate", "one-pass")
SPEC = {"outputs": 4, "seed": 17, "link": "softmax", "output": "proba", "dtype": "float32"}


class HeadsOutput(NamedTuple):
    outputs: dict
    vectors: object


def enabled(cell):
    return cell.get("profile") == PROFILE


def validate(cell):
    if enabled(cell) != (cell["stage"] in STAGES):
        raise ValueError("head group stages require linear-heads-multi@1, and that profile them")
    if not enabled(cell):
        if cell["heads"] != 1 or cell["return_vectors"] or cell["practical_saving"] is not None:
            raise ValueError("heads, return_vectors and practical_saving are for the head groups")
        return
    if cell["base_heads_mode"] not in MODES or cell["candidate_heads_mode"] not in MODES:
        raise ValueError("heads modes are separate or one-pass")
    if cell["table_dtype"] != "float32" or cell["output_dtype"] != "float32":
        raise ValueError("linear-heads-multi@1 requires float32")
    _count(cell)
    saving = cell["practical_saving"]
    if saving is not None and not (type(saving) in (int, float) and 0 < saving < 1):
        raise ValueError("practical_saving is a fraction between zero and one")
    if cell["decision_gate"] and saving is None:
        raise ValueError("a head group decision cell needs its practical_saving before the run")


def _count(cell):
    if type(cell["heads"]) is not int or cell["heads"] < 1:
        raise ValueError("a head group needs at least one head")
    if cell["stage"] == "head-group-mixed" and cell["heads"] % 2:
        raise ValueError("a mixed head group needs an even number of heads")
    if type(cell["return_vectors"]) is not bool or type(cell["stop_session_on_reject"]) is not bool:
        raise ValueError("return_vectors and stop_session_on_reject must be booleans")


def configure(job, cell, name, control):
    """The mode of an arm: the candidate arm of a comparison runs ``candidate_heads_mode``,
    every other arm (the base arm, both arms of an A/A control) ``base_heads_mode``."""
    if enabled(cell):
        candidate = not control and name == "candidate"
        job["heads_mode"] = cell["candidate_heads_mode" if candidate else "base_heads_mode"]


def models(encoder, job):
    """The n fixed heads as {name: Classifier}, in order."""
    import numpy as np

    from fse import Classifier

    count, mixed = job["heads"], job["stage"] == "head-group-mixed"
    out = {}
    for index in range(count):
        rng = np.random.default_rng(SPEC["seed"] + index)
        weight = rng.standard_normal((SPEC["outputs"], encoder.output_dim)).astype(np.float32)
        bias = rng.standard_normal(SPEC["outputs"]).astype(np.float32)
        out[f"head_{index}"] = Classifier.from_arrays(
            encoder,
            weight,
            bias,
            labels=[f"class_{i}" for i in range(SPEC["outputs"])],
            link=SPEC["link"],
            normalized=mixed and index >= count // 2,
        )
    return out


def operation(encoder, texts, job, *, backend="cpu"):
    if not enabled(job) or job["stage"] not in STAGES or job.get("heads_mode") not in MODES:
        raise ValueError("a linear-heads-multi@1 stage and a head mode are required")
    group = models(encoder, job)
    options = {"workers": job["workers"], "backend": backend}
    build = _one_pass if job["heads_mode"] == "one-pass" else _separate
    predict = build(group, texts, job["return_vectors"], options)
    # Make every lazy table and plan outside every timer, and check the shape once.
    prepared = predict()
    if list(prepared.outputs) != list(group):
        raise ValueError("head group outputs differ from the fixed profile")
    for array in prepared.outputs.values():
        if array.shape != (len(texts), SPEC["outputs"]) or array.dtype.str != "<f4":
            raise ValueError("head group probabilities differ from the fixed profile")
    if job["return_vectors"] and prepared.vectors.shape != (len(texts), encoder.output_dim):
        raise ValueError("head group vector shape differs from the fixed profile")
    return predict


def _one_pass(group, texts, vectors, options):
    from fse import Heads

    heads = Heads(**group)
    if not vectors:
        return lambda: HeadsOutput(dict(heads.predict(texts, output="proba", **options)), None)

    def predict():
        (batch,) = heads.predict_batches((texts,), output="proba", return_vectors=True, **options)
        return HeadsOutput(dict(batch.outputs), batch.vectors)

    return predict


def _separate(group, texts, vectors, options):
    if not vectors:
        return lambda: HeadsOutput(
            {name: m.predict_proba(texts, **options) for name, m in group.items()}, None
        )

    def predict():
        batches = {
            name: next(m.predict_batches((texts,), output="proba", return_vectors=True, **options))
            for name, m in group.items()
        }
        first = next(iter(batches.values()))
        return HeadsOutput({name: b.outputs for name, b in batches.items()}, first.vectors)

    return predict


def probe(job, texts):
    """The untimed adapter probe: the arm's mode on the cpu backend equals both modes on the
    numpy reference, per head, by bytes."""
    from _development_encode import output_identity
    from _development_process import checked

    from fse import Encoder

    artifact = job["artifact"]
    encoder = Encoder.from_pretrained(
        checked(artifact["path"], artifact["sha256"]), normalize=False, mmap=True
    )
    got = output_identity(operation(encoder, texts, job)())
    for mode in MODES:
        reference = output_identity(
            operation(encoder, texts, {**job, "heads_mode": mode}, backend="numpy")()
        )
        if got != reference:
            raise ValueError(f"preflight head group ({mode}) differs from the numpy reference")
    return got


def practical(cell, result):
    """The practical gate of a decision cell: the saving of the comparison and of the
    confirmation, 1 - 1 / factor as the controller's time_saving, at least the declared
    fraction; kept next to keep, never replacing it."""
    if not enabled(cell) or cell["practical_saving"] is None or "comparison" not in result:
        return result
    needed = cell["practical_saving"]
    savings = [result["comparison"]["time_saving"], result["confirmation"]["time_saving"]]
    passed = min(savings) >= needed
    return {
        **result,
        "practical": {
            "required_saving": needed,
            "comparison_saving": savings[0],
            "confirmation_saving": savings[1],
            "passed": passed,
            "claim": "significantly faster only if keep holds and this passes",
        },
    }


def profile(job):
    """Untimed reproduction for externally coordinated CPU/allocation/RSS profiling."""
    from _development_encode import output_identity
    from _development_stages import inputs
    from _development_system import apply_affinity

    policy = {"affinity": apply_affinity(job.get("cpu_affinity"))}
    encoder, texts = inputs(job)
    predict = operation(encoder, texts, {**job, "heads_mode": job["candidate_heads_mode"]})
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
        "head_spec": {**SPEC, "heads": job["heads"], "return_vectors": job["return_vectors"]},
        "worker_path": job["stage"],
        "heads_mode": job["candidate_heads_mode"],
        "policy": policy,
        "operation_repetitions": iterations,
        "output": output_identity(output),
    }
