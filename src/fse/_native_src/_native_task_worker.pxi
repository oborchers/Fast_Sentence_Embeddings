# Included by _native.pyx: the internal StringSource worker. Runtime helpers are
# injected, so the extension continues to import only fse._contract.

cdef class TextTasks:
    """One call's immutable pooling state and owners, shared by its worker threads."""
    cdef object __weakref__
    cdef WeightedMean kernel
    cdef TextLookup lookup
    cdef int isa
    cdef bint normalize
    cdef object text, local, base, helpers
    cdef StringCopy copy

    def __init__(self, WeightedMean kernel not None, TextLookup lookup not None, isa,
                 bint normalize, text, local, base, helpers):
        self.kernel, self.lookup = kernel, lookup
        self.isa = _check_isa(isa)
        self.normalize = normalize
        if isinstance(text, StringCopy):
            self.copy = text
            text = (self.copy.data, self.copy.offsets)
        self.text, self.local, self.base, self.helpers = text, local, base, helpers

    def __call__(self, item):
        # Attempts are task-local: concurrent calls never overwrite another task's state.
        cdef _TextAttempt attempt = _TextAttempt(self, item)
        self.helpers[3](attempt.run, item[0].n_rows)  # the unchanged earliest algorithm
        return self.helpers[5](self.helpers[6], self.local, item[0], item[1])


cdef class _TextAttempt:
    cdef TextTasks owner
    cdef object task, target
    cdef bint copied, reported

    def __init__(self, TextTasks owner not None, item):
        self.owner, self.task = owner, item[0]
        stored = item[1]
        codes, scales = stored if isinstance(stored, tuple) else (stored, None)
        self.target = (codes, scales, None)

    def run(self, Py_ssize_t n):
        cdef TextTasks owner = self.owner
        if owner is None:
            raise TypeError("native text attempt is not initialized")
        # Preserve the order of Pipeline._work and pooled_task for every prefix.
        part = owner.helpers[0](self.target, n)
        task = self.task.prefix(n)
        codes = part[0]
        rows = codes if codes.dtype == np.float32 else owner.helpers[1](
            owner.local, len(codes), owner.kernel.dim)
        cdef Workspace ws = owner.helpers[2](owner.local)
        if ws is None:
            raise TypeError(
                "Argument 'ws' has incorrect type (expected fse._native.Workspace, got NoneType)"
            )
        chunk = task.chunk
        cdef _CopyRange span
        cdef _CopyRange *copy = NULL
        if owner.copy is not None:
            span.rows = owner.copy.rows
            span.offsets = <const int64_t *>np.PyArray_DATA(<np.ndarray>owner.copy.offsets)
            span.out = PyBytes_AS_STRING(owner.copy.data)
            span.first, span.last = self.task.chunk.start, self.task.chunk.end
            if not 0 <= span.first <= chunk.start <= chunk.end <= span.last <= len(owner.copy.snapshot):
                raise AssertionError("copied task range is outside its snapshot")
            span.copied = &self.copied
            copy = &span
        try:
            _pool_text_checked(owner.kernel, owner.lookup, ws, owner.text,
                           chunk.start, chunk.end, chunk.n_sentences,
                               owner.base + chunk.first_row, rows, owner.normalize, None, owner.isa, copy)
        finally:
            if self.copied and not self.reported:
                owner.copy.finished += self.task.chunk.end - self.task.chunk.start
                self.reported = True
        owner.helpers[4](rows, part, owner.base + task.first_row)
