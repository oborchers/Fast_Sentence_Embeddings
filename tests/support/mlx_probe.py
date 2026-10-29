"""An experiment, not a backend: a weighted mean in a Metal kernel behind the contract.

It shows that the contract fits a device (prepare uploads once, rows writes host rows,
capabilities steer the runtime) and measures where Metal breaks the numerical contract.
PR 9 builds the MLX backend. Run with: uv run --with mlx pytest tests/test_backends_mlx_probe.py
"""

import numpy as np

from fse._backends import Capabilities, PoolInput, Prepared, UnsupportedByBackendError
from fse._backends._base import check_out, check_pool_input, check_prepared, dims
from fse._contract import Aggregation, Batch

_SOURCE = """
    uint d = thread_position_in_grid.x;
    uint s = thread_position_in_grid.y;
    if (d >= DIM) return;
    int begin = offsets[s];
    int end = offsets[s + 1];
    float acc = 0.0f;
    for (int t = begin; t < end; ++t) {
        uint w = tokens[t];
        acc = fma(weights[w], rows[w * DIM + d], acc);
    }
    int n = end - begin;
    out[s * DIM + d] = n ? precise::divide(1.0f, float(n)) * acc : 0.0f;
"""


class MlxProbe:
    name = "mlx-probe"

    def __init__(self):
        import mlx.core as mx

        self.mx = mx
        self.kernel = mx.fast.metal_kernel(
            name="fse_weighted_mean_probe",
            input_names=["rows", "weights", "offsets", "tokens"],
            output_names=["out"],
            source=_SOURCE,
        )

    def capabilities(self) -> Capabilities:
        return Capabilities(
            word_dtypes=frozenset({"float32"}),
            aggregations=frozenset({Aggregation.WEIGHTED_MEAN}),
            steps=frozenset(),
            fasttext=False,
            concurrent=False,
            max_dim=1 << 16,
        )

    def prepare(self, pool: PoolInput, steps) -> Prepared:
        check_pool_input(pool)
        self.capabilities().check(self.name, pool, steps)
        checked, out = dims(pool, steps)
        state = (self.mx.array(pool.vectors), self.mx.array(pool.weights))  # uploaded once
        return Prepared(self.name, pool, checked, out, state)

    def rows(self, batch: Batch, prepared: Prepared, out: np.ndarray) -> None:
        check_prepared(prepared, self.name)
        check_out(out, batch, prepared)
        n, dim = batch.n_sentences, prepared.output_dim
        if n == 0:
            return
        rows, weights = prepared.state
        (result,) = self.kernel(
            inputs=[
                rows,
                weights,
                self.mx.array(batch.offsets.astype(np.int32)),
                self.mx.array(batch.tokens),
            ],
            template=[("DIM", dim)],
            grid=(dim, n, 1),
            threadgroup=(min(dim, 256), 1, 1),
            output_shapes=[(n, dim)],
            output_dtypes=[self.mx.float32],
        )
        out[...] = np.array(result)
        bad = np.flatnonzero(~np.isfinite(out).all(axis=1))
        if len(bad):
            raise ValueError(f"row {batch.first_row + int(bad[0])} contains NaN or infinite values")

    def score(self, *_args):
        raise UnsupportedByBackendError("the mlx-probe backend cannot run score (PR 9)")
