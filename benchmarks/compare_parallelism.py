"""Describe configured parallelism without presenting a hint as a measured thread count."""

import os

NATIVE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def child_environment(threads: int) -> dict[str, str]:
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    # Accelerate can cache this setting on its first call, before threadpoolctl.
    return {**os.environ, **dict.fromkeys(NATIVE_THREAD_VARIABLES, str(threads))}


def native_thread_limits() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in NATIVE_THREAD_VARIABLES}


def settings(kind: str, requested: int, texts: int) -> dict:
    result = {
        "requested_threads": requested,
        "threads": requested if kind == "fse" else None,
        "fse_workers": requested if kind == "fse" else None,
        "tokenizer_thread_count": None,
        "tokenizer_threads_measured": False,
    }
    if kind == "model2vec":
        batch_pool = requested > 1 and texts > 10_000
        result.update(
            {
                "tokenizer_parallelism": "on" if requested > 1 and not batch_pool else "off",
                "parallel_batch_pool": "default" if batch_pool else "disabled",
            }
        )
    return result
