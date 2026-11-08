"""Private resolution of configured MTEB task entries into (benchmark, task) pairs.

A task is always carried together with the benchmark it was resolved from: the same task
name exists in several benchmarks with different language subsets (STS17 in MTEB(eng, v2)
is English only, in MTEB(Multilingual, v2) it holds every language pair). Resolving a name
alone would evaluate the wrong subsets without an error."""

from __future__ import annotations

from importlib.metadata import version

from _measurement_config import PINS

MIND_REASON = "not computed: MTEB 2.26.25 MindSmallReranking needs more memory than the 64 GB machine (two attempts, status 137 and earlier ENOSPC)"

# The pinned benchmarks and their task counts in MTEB 2.26.25. A changed count is an error.
BENCHMARKS = {
    "MTEB(eng, v2)": 41,
    "NanoBEIR": 13,
    "MTEB(Multilingual, v2)": 131,
    "NanoBEIR-multilingual": 13,
}
# MindSmallReranking is excluded only from the English benchmark (DR-271: Englische
# Qualität der statischen Baselines mit MTEB vergleichen); it is no task of the others.
EXCLUDED = {"MTEB(eng, v2)": "MindSmallReranking"}
SEPARATOR = "/"


def pinned():
    if version("mteb") != PINS["versions"]["mteb"]:
        raise ValueError("MTEB differs from the pinned evaluator")


def benchmark_tasks(name):
    """The tasks of a pinned benchmark, with their benchmark language filter applied."""
    import mteb

    pinned()
    tasks = list(mteb.get_benchmark(name).tasks)
    if len(tasks) != BENCHMARKS[name]:
        raise ValueError(f"pinned benchmark task count changed: {name}")
    names = [t.metadata.name for t in tasks]
    if len(set(names)) != len(names):
        raise ValueError(f"pinned benchmark names a task twice: {name}")
    return [t for t in tasks if t.metadata.name != EXCLUDED.get(name)]


def resolve(entry):
    """``[(benchmark, task), ...]`` of one configured entry: a pinned benchmark, the explicit
    form ``"Benchmark/Task"``, or a task name. A task name that more than one pinned benchmark
    holds is refused; name its benchmark explicitly."""
    if entry in BENCHMARKS:
        return [(entry, task) for task in benchmark_tasks(entry)]
    if SEPARATOR in entry and entry.split(SEPARATOR, 1)[0] in BENCHMARKS:
        benchmark, name = entry.split(SEPARATOR, 1)
        return [(benchmark, task_of(benchmark, name))]
    if entry == EXCLUDED["MTEB(eng, v2)"]:
        raise ValueError(MIND_REASON)
    holders = [b for b in BENCHMARKS if any(t.metadata.name == entry for t in benchmark_tasks(b))]
    if len(holders) > 1:
        raise ValueError(
            f"task {entry} is in several benchmarks {holders}; name it as 'Benchmark/{entry}'"
        )
    if holders:
        return [(holders[0], task_of(holders[0], entry))]
    import mteb

    pinned()
    return [(entry, mteb.get_task(entry))]


def task_of(benchmark, name):
    """The task instance ``name`` of ``benchmark`` (a pinned benchmark or a lone task)."""
    if benchmark not in BENCHMARKS:
        import mteb

        pinned()
        if benchmark != name:
            raise ValueError("a lone task is its own benchmark label")
        return mteb.get_task(name)
    found = [t for t in benchmark_tasks(benchmark) if t.metadata.name == name]
    if len(found) != 1:
        raise ValueError(f"task {name} is not in the pinned benchmark {benchmark}")
    return found[0]


def expand(cells):
    result = []
    for cell in cells:
        if cell["kind"] != "mteb":
            result.append(cell)
            continue
        for entry in cell["tasks"]:
            for benchmark, task in resolve(entry):
                result.append(
                    {
                        **cell,
                        "id": cell["id"] + "_" + task.metadata.name.replace(".", "-"),
                        "task": task.metadata.name,
                        "benchmark": benchmark,
                        "evaluation_filters": filters_for(task),
                    }
                )
    if len({c["id"] for c in result}) != len(result):
        raise ValueError("task expansion produced duplicate identifiers")
    return result


def filters_for(task):
    return {
        "hf_subsets": list(task.hf_subsets),
        "eval_splits": list(task.eval_splits),
        "seed": task.seed,
    }
