"""Keep optional pod training durations separate from inference measurements."""

import sys
import time


def fitted(args, function, tool_chain):
    collect = getattr(args, "pod_training_times", False)
    if collect and sys.platform != "linux":
        raise RuntimeError("classification timing is restricted to the Linux pod")
    start = time.monotonic() if collect else None
    result = function()
    return result, {
        "training_seconds": time.monotonic() - start if start is not None else None,
        "training_tool_chain": tool_chain,
        "training_comparison": "different tool chains, separate from inference throughput",
    }
