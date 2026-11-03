"""One continuous timed group with fixed inputs and bounded live result storage."""

from __future__ import annotations

import time

CONTRACT = "repeated-prepared-input@1"


def repeat(operation, iterations):
    output, first_index, read_seconds, open_seconds = None, None, 0.0, 0.0
    started = time.perf_counter()
    for _ in range(iterations):
        # Release the previous tokenizer batch before allocating another one.
        output = None
        output = operation()
        if isinstance(output, dict):
            if first_index is None:
                first_index = output["index"]
            read_seconds += output["seconds"]
            open_seconds += output["including_open_seconds"]
    elapsed = time.perf_counter() - started
    if isinstance(output, dict):
        output["first_index"] = first_index
    return (
        elapsed,
        output,
        {
            "timer_contract": CONTRACT,
            "operation_repetitions": iterations,
            "sink_read_to_close_seconds_sum": read_seconds if first_index is not None else None,
            "sink_including_open_seconds_sum": open_seconds if first_index is not None else None,
        },
    )
