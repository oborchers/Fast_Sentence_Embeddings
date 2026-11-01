"""Describe configured parallelism without presenting a hint as a measured thread count."""


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
