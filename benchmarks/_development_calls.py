"""Many small encode calls: one pass is every call of a fixed pool, in a fixed order."""

from __future__ import annotations

from _harness.budget import check


class Calls(list):
    """The results of the calls of the last pass, one array per call, in call order."""


def groups(texts, call_texts):
    """The pool cut into the lists of the calls, made before any timer starts; the last
    call holds the remainder."""
    return [texts[i : i + call_texts] for i in range(0, len(texts), call_texts)]


def operation(model, texts, job):
    from _development_encode import output_identity

    check(job.get("wall_deadline"))
    if job.get("call_repeats") is not None:
        return Repeated(model, texts, job)
    calls = groups(texts, job["call_texts"])
    dtype = job["output_dtype"]
    workers = job["workers"]
    reference = model.encode(texts, workers=workers, dtype=dtype)
    expected = output_identity(reference)
    del reference
    encode = model.encode

    def one_pass():
        results = Calls()
        append = results.append
        for call in calls:
            append(encode(call, workers=workers, dtype=dtype))
        return results

    check(job.get("wall_deadline"))
    if output_identity(one_pass()) != expected:  # stacked calls equal one bulk encode
        raise ValueError("small calls differ from one bulk encode")
    return one_pass


def validate(cell):
    repeats = cell.get("call_repeats")
    if cell["stage"] == "encode-calls":
        _validate_variant(cell)
    if repeats is not None:
        if type(repeats) is not int or repeats <= 0:
            raise ValueError("call_repeats must be a positive integer or null")
        if cell["stage"] != "encode-calls" or cell["output_dtype"] != "float32":
            raise ValueError("call_repeats requires float32 encode-calls")
    if cell["stage"] == "encode-calls" and cell["workers"] != 1 and repeats is None:
        raise ValueError("parallel encode-calls requires call_repeats")
    if cell["stage"] == "encode-calls" and cell["runtime_timeline"] and repeats is None:
        raise ValueError("encode-calls timeline requires call_repeats")


def _validate_variant(cell):
    from _development_runtime import CALL_VARIANTS

    if cell["runtime_variant"] not in CALL_VARIANTS:
        raise ValueError("unsupported encode-calls runtime variant")
    if cell["runtime_variant"] != "baseline" and cell.get("call_repeats") is None:
        raise ValueError("encode-calls variants require call_repeats")


class Repeated:
    """Repeat the exact complete input, releasing each result before the next call."""

    def __init__(self, model, texts, job):
        from fse import Pool
        from fse._text import StringSource

        self.model, self.job = model, job
        self.variant = job.get("runtime_variant", "baseline")
        self.pool = None
        self.block_probe = None
        self.texts = StringSource(texts) if "prebuilt" in self.variant else texts
        # Pool construction/start and eventual close are outside timed groups.
        if self.variant in ("pool", "pool+prebuilt"):
            self.pool = Pool(job["workers"], start=True)
        try:
            self._verify(texts)
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _verify(self, texts):
        from _development_encode import output_identity
        from _development_runtime import settings

        expected = output_identity(
            self.model.encode(texts, workers=self.job["workers"], dtype=self.job["output_dtype"])
        )
        with settings(self.variant):
            if output_identity(self.one()) != expected:
                raise ValueError("repeated exact call differs from baseline full output")

    def one(self):
        if self.block_probe is not None:
            self.block_probe.next_call()
        try:
            return self.model.encode(
                self.texts,
                workers=self.job["workers"],
                dtype=self.job["output_dtype"],
                pool=self.pool,
            )
        finally:
            if self.block_probe is not None:
                self.block_probe.end_call()

    def __enter__(self):
        if self.job.get("runtime_probe_blocks", False):
            from _development_runtime_block import BlockProbe

            try:
                self.block_probe = BlockProbe(self)
                self.block_probe.__enter__()
            except BaseException:
                self.__exit__(None, None, None)
                raise
        return self

    def __exit__(self, *args):
        try:
            if self.block_probe is not None:
                self.block_probe.__exit__(*args)
        finally:
            if self.pool is not None:
                self.pool.close()

    def block_evidence(self):
        return None if self.block_probe is None else self.block_probe.evidence()

    def __call__(self):
        from _development_runtime import settings

        with settings(self.variant):
            output = None
            for _ in range(self.job["call_repeats"]):
                output = None
                output = self.one()
            return output

    def evidence(self):
        from _development_runtime import settings
        from _development_runtime_probe import observe

        if not self.job.get("runtime_timeline") or self.job.get("diagnostic_profile", False):
            return None
        # Restore timed hooks before installing the separate full-capture hooks.
        if self.block_probe is not None:
            self.block_probe.__exit__(None, None, None)
            self.block_probe = None
        # The separate probe records consecutive calls with composite task identity.
        with settings(self.variant):
            return observe(_Single(self), self.job)


class _Single:
    def __init__(self, operation):
        self.model = operation.model
        self.variant = operation.variant
        self.operation = operation

    def __call__(self):
        return self.operation.one()
