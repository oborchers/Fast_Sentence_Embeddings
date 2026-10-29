"""A fake MLX for the tests of the device thread (tests/test_backends_mlx_device*.py): it
evaluates nothing and records what it was given, so they run on every machine."""


class FakeMx:
    gpu = "gpu"

    def __init__(self, keep=True):
        self.evaluated = []
        self.fail = None
        self.keep = keep  # False: count evaluations without keeping the arrays alive

    def new_thread_unsafe_stream(self, device):
        return f"stream on {device}"

    def eval(self, *arrays):
        if self.fail is not None:
            raise self.fail
        self.evaluated.append(arrays if self.keep else len(arrays))
