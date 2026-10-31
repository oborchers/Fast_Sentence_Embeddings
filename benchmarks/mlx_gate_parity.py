"""Independent per-row acceptance bounds for the gate's finished WordPiece tables.

No backend bounds are used. The formula and exact-row cases follow
docs/architecture/backends.md and tests/support/bound.py. CPU conformance remains
the oracle of CPU arithmetic; quantization error against float32 is a separate issue.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

import fse

U = 2.0**-24


def gamma(k):
    return k * U / (1 - k * U)


def equal(got: np.ndarray, want: np.ndarray) -> None:
    if got.shape != want.shape or got.dtype != want.dtype or not np.isfinite(got).all():
        raise AssertionError("shape, dtype or finiteness differs")
    if not np.array_equal(got.view(np.uint32), want.view(np.uint32)):
        raise AssertionError("output is not bit-identical")


def counts(enc) -> dict:
    if "mlx" not in enc._pipelines:
        return {}
    from fse._backends._mlx._backend import fallbacks

    return fallbacks(enc._pipelines["mlx"].prepared)


def delta(after: dict, before: dict) -> dict:
    return {k: after[k] - before.get(k, 0) for k in after if after[k] != before.get(k, 0)}


@contextmanager
def launches():
    """Observe real device launches only in an untimed pass, restoring even after failure."""
    from fse._backends._mlx import _kernels

    original, seen = _kernels.launch, []

    def observed(*args, **kwargs):
        seen.append(1)
        return original(*args, **kwargs)

    _kernels.launch = observed
    try:
        yield seen
    finally:
        _kernels.launch = original


def magnitude(ids: np.ndarray, enc) -> float:
    """Float64 sum of absolute terms, with the CPU's float32 stage-2 factors."""
    wv = enc.word_vectors
    factors = enc._spec.weights[ids]
    if wv.dtype == "int8":
        factors = np.multiply(factors, wv.scales[ids], dtype=np.float32)
    sums = np.zeros(wv.dim, np.float64)
    # Bound memory for a long sentence, and preserve contribution order in float64.
    for token, factor in zip(ids, factors, strict=True):
        sums += abs(float(factor)) * np.abs(wv.stored_vectors[token].astype(np.float64))
    return float(np.linalg.norm(sums / len(ids))) if len(ids) else 0.0


def acceptance(k: int, dim: int, m: float, radius: float | None) -> tuple[float, bool]:
    """A conservative exact-row mask also covers the device's tiny-norm fallback.

    For normalized rows ||g|| >= r-E. Requiring r-E >= 2**-50 proves the
    device squared norm is not below 2**-100 without observing a different raw run.
    """
    if not k or k >= 2**24:
        return 0.0, True
    error = (1e-5 + 2 * gamma(k)) * m + 2.0**-125 * k * np.sqrt(dim)
    if radius is None:
        return float(error), False
    exact = radius <= error or radius - error < 2.0**-50
    bound = 2 * error / max(radius, 2.0**-63) + 2 * (gamma(dim) / 2 + 4 * U)
    return float(bound), exact


def raw_encoder(enc):
    content = enc._content
    return fse.Encoder._converted(
        enc.word_vectors,
        content.tokenizer,
        normalize=False,
        source=content.source,
        license=content.license,
    )


def check_bound(got, want, bound: float, exact: bool) -> float:
    if exact:
        equal(got, want)
        return 0.0
    error = float(np.linalg.norm(got.astype(np.float64) - want.astype(np.float64)))
    if not np.isfinite(bound) or error > bound:
        raise AssertionError(f"device row exceeds the contract: {error} > {bound}")
    return error / bound if bound else 0.0


def audit(enc, texts: list[str], cpu: np.ndarray, mlx: np.ndarray, guard) -> dict:
    """Check all rows and count actual tokens; compute expensive bounds only where bits differ."""
    from fse._text import StringSource, make_lookup, new_workspace

    if cpu.shape != mlx.shape or cpu.dtype != mlx.dtype or cpu.dtype != np.float32:
        raise AssertionError("shape or float32 output dtype differs")
    if not np.isfinite(cpu).all() or not np.isfinite(mlx).all():
        raise AssertionError("nonfinite output")
    if enc.word_vectors.fasttext is not None:
        raise ValueError("this real-model gate supports tables without FastText only")
    differing = np.flatnonzero((cpu.view(np.uint32) != mlx.view(np.uint32)).any(axis=1))
    radii = _radii(enc, texts, differing, guard)
    source = StringSource(texts)
    lookup, ws = make_lookup(enc.word_vectors, enc._content.tokenizer), new_workspace()
    tokens, empty, worst = 0, 0, 0.0
    different = set(map(int, differing))
    for chunk in source.scan(workers=1).chunks:
        guard.check("parity token chunk")
        batch = source.encode(chunk, lookup, ws)
        tokens += len(batch.tokens)
        for local, (start, end) in enumerate(
            zip(batch.offsets[:-1], batch.offsets[1:], strict=True)
        ):
            row = batch.first_row + local
            if start == end:
                empty += 1
                if cpu[row].view(np.uint32).any() or mlx[row].view(np.uint32).any():
                    raise AssertionError("an empty sentence must give positive zero")
            if row in different:
                ids = batch.tokens[start:end]
                bound, exact = acceptance(
                    len(ids), enc.output_dim, magnitude(ids, enc), radii.get(row)
                )
                worst = max(worst, check_bound(mlx[row], cpu[row], bound, exact))
    return {
        "bit_equal": not len(differing),
        "differing_rows": len(differing),
        "differing_values": int((cpu.view(np.uint32) != mlx.view(np.uint32)).sum()),
        "max_error_over_bound": worst,
        "tokens": tokens,
        "empty_rows": empty,
        "contract_passed": True,
    }


def _radii(enc, texts, differing, guard) -> dict:
    if not enc.steps or not len(differing):
        return {}
    guard.check("CPU raw rows for normalization bound")
    raw = raw_encoder(enc).encode([texts[i] for i in differing], workers=1, backend="cpu")
    guard.check("CPU raw rows complete")
    return dict(
        zip(map(int, differing), np.linalg.norm(raw.astype(np.float64), axis=1), strict=True)
    )
