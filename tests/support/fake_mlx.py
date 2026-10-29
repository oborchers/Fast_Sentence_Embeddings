"""A fake MLX for the tests of the device thread (tests/test_backends_mlx_device*.py): it
evaluates nothing and records what it was given, so they run on every machine.

``async_eval`` (the start of a group's evaluation) records the group in ``evaluated`` and
raises ``fail``; ``eval`` (waiting for it) records in ``waited`` and raises ``fail_wait``,
and ``log`` has both in their order.
"""


class FakeMx:
    gpu = "gpu"

    def __init__(self, keep=True):
        self.evaluated = []
        self.waited = []
        self.log = []
        self.fail = None
        self.fail_wait = None
        self.keep = keep  # False: count evaluations without keeping the arrays alive

    def new_thread_unsafe_stream(self, device):
        return f"stream on {device}"

    def async_eval(self, *arrays):
        if self.fail is not None:
            raise self.fail
        self.evaluated.append(arrays if self.keep else len(arrays))
        self.log.append(("start", arrays if self.keep else len(arrays)))

    def eval(self, *arrays):
        self.log.append(("wait", arrays if self.keep else len(arrays)))
        if self.fail_wait is not None:
            error, self.fail_wait = self.fail_wait, None
            raise error
        self.waited.append(arrays if self.keep else len(arrays))
