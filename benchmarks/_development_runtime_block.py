"""Same timeline hooks inside timed diagnostic blocks, with bounded rolling storage.

Only unprobed blocks decide. Reset bookkeeping is included in probed timings.
The buffer retains the last call; full per-call traces come from untimed evidence.
"""

from contextlib import ExitStack

from _development_runtime_probe import Timeline, _capacity


class BlockProbe:
    def __init__(self, operation):
        plan = _capacity(operation, {**operation.job, "stage": "encode-allocate"})
        if not plan["admitted"]:
            raise ValueError("timed probe exceeds diagnostic capacity")
        self.timeline = Timeline(plan["capacity"])
        self.stack = ExitStack()
        self.calls = 0
        self.invalid = False

    def __enter__(self):
        try:
            self.timeline.install(self.stack)
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def _check(self):
        trace = self.timeline
        self.invalid |= bool(trace.overflow or trace.dropped_reentrant or trace.future_tasks)
        if self.invalid:
            raise ValueError("timed block probe lost markers or publication callbacks")

    def next_call(self):
        self._check()
        trace = self.timeline
        trace.used = 0
        trace.current_call = self.calls
        self.calls += 1

    def evidence(self):
        self._check()
        return {
            "valid": True,
            "calls_since_child_start": self.calls,
            "capacity": self.timeline.capacity,
            "gc": self.timeline.collections.export(),
            "scope": "same timeline hooks, rolling per-call buffer; reset cost included",
            "claim_class": "descriptive",
            "valid_for_speed_decision": False,
        }
