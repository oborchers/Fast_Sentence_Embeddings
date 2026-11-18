# Per-call StringSource task team. Python owns setup; callbacks touch only C state.

cdef extern from "_native_threads.h":
    int64_t fse_atomic_add(fse_atomic *p, int64_t n) noexcept nogil
    void fse_atomic_store(fse_atomic *p, int64_t n) noexcept nogil
    void fse_atomic_min(fse_atomic *p, int64_t n) noexcept nogil
    fse_atomic fse_test_team_live
    ctypedef struct fse_team_slot:
        int started
        int joined
        uint64_t started_id
        uint64_t actual_id
    void fse_team(int n, fse_part fn, void *arg, fse_team_slot *slots,
                  unsigned failed_starts) noexcept nogil
    int fse_team_balanced(int n, fse_team_slot *slots) noexcept nogil
    void fse_team_pause() noexcept nogil


cdef struct _TeamTask:
    _Span span
    _Stages stages
    _Pooled result


cdef struct _Team:
    fse_atomic next
    fse_atomic failed_at
    PyObject *kernel
    PyObject *lookup
    PyObject **workspaces
    _TeamTask *tasks
    fse_team_slot *slots
    Py_ssize_t n, workers, limit
    int corrupt
    Py_ssize_t corrupt_task, status_task, delay_task
    int status, delay_ms
    bint hold
    unsigned failed_starts
    int allocated, freed, started, joined
    bint balanced
    uint64_t started_ids[256]
    uint64_t actual_ids[256]


cdef int _test_team_fail_allocation = 0
cdef unsigned _test_team_failed_starts = 0
cdef long long _team_calls = 0
cdef Py_ssize_t _team_status_task = -1
cdef int _team_status = 0
cdef Py_ssize_t _team_corrupt_task = -1
cdef Py_ssize_t _team_delay_task = -1
cdef int _team_delay_ms = 0
cdef fse_atomic _team_hold
cdef fse_atomic _team_entered


def _test_team_control(hold=None):
    """Atomic test barrier: entry receipt proves native execution before a signal/fork."""
    if hold is not None:
        if hold:
            fse_atomic_store(&_team_entered, 0)
        fse_atomic_store(&_team_hold, bool(hold))
    return fse_atomic_load(&_team_hold), fse_atomic_load(&_team_entered)


def _test_team_calls():
    return _team_calls


def _test_team_resources():
    return fse_atomic_load(&fse_test_team_live)


def _test_team_faults(int allocation=0, unsigned failed_starts=0,
                      Py_ssize_t status_task=-1, int status=0,
                      Py_ssize_t corrupt_task=-1, Py_ssize_t delay_task=-1, int delay_ms=0):
    global _test_team_fail_allocation, _test_team_failed_starts
    global _team_status_task, _team_status, _team_corrupt_task, _team_delay_task, _team_delay_ms
    old = (_test_team_fail_allocation, _test_team_failed_starts, _team_status_task,
           _team_status, _team_corrupt_task, _team_delay_task, _team_delay_ms)
    _test_team_fail_allocation, _test_team_failed_starts = allocation, failed_starts
    _team_status_task, _team_status, _team_corrupt_task = status_task, status, corrupt_task
    _team_delay_task, _team_delay_ms = delay_task, delay_ms
    return old


cdef void *_team_calloc(_Team *t, size_t n, size_t size) noexcept nogil:
    cdef void *p = calloc(n, size)
    if p != NULL:
        t.allocated += 1
        fse_atomic_add(&fse_test_team_live, 1)
    return p


cdef void _team_free(_Team *t, void *p) noexcept nogil:
    if p != NULL:
        free(p)
        t.freed += 1
        fse_atomic_add(&fse_test_team_live, -1)


cdef void _team_release(_Team *t) noexcept nogil:
    cdef Py_ssize_t i
    if t.workspaces != NULL:
        for i in range(t.workers):
            if t.workspaces[i] != NULL:
                (<Workspace>t.workspaces[i])._clear()
    _team_free(t, t.workspaces)
    _team_free(t, t.tasks)
    _team_free(t, t.slots)
    t.workspaces, t.tasks, t.slots = NULL, NULL, NULL
    t.balanced = t.balanced and t.allocated == t.freed


cdef bint _team_failed(_Pooled *r) noexcept nogil:
    return (r.encode != ENCODE_OK or r.built != CHECK_OK or r.kernel != CHECK_OK
            or r.bad >= 0 or r.unnormalizable >= 0 or r.not_finite >= 0)


cdef void _team_worker(void *arg, int slot) noexcept nogil:
    cdef _Team *t = <_Team *>arg
    cdef int64_t i
    cdef _TeamTask *task
    cdef int delay, corrupt
    cdef uint64_t saved = fse_fpenv_push()
    if t.hold:
        fse_atomic_add(&_team_entered, 1)
        while fse_atomic_load(&_team_hold):
            fse_team_pause()
    while True:
        i = fse_atomic_add(&t.next, 1)
        if i >= t.n or i > fse_atomic_load(&t.failed_at):
            break
        task = &t.tasks[i]
        if i == t.delay_task:
            for delay in range(t.delay_ms):
                fse_team_pause()
        _start_result(&task.result)
        if i == t.status_task and t.status == ENCODE_NO_MEMORY:
            task.result.encode = ENCODE_NO_MEMORY
        else:
            task.result.encode = _encode_span_limit(
                <TextLookup>t.lookup, &task.span, <Workspace>t.workspaces[slot], t.limit)
        if task.result.encode == ENCODE_OK:
            corrupt = t.corrupt if t.corrupt_task < 0 or i == t.corrupt_task else 0
            _pool_stages_kind(<WeightedMean>t.kernel, <Workspace>t.workspaces[slot],
                              &task.stages, &task.result, corrupt)
            if i == t.status_task and t.status != ENCODE_OK:
                task.result.built = CHECK_OFFSETS_START  # deterministic mismatch probe
        if _team_failed(&task.result):
            fse_atomic_min(&t.failed_at, i)
    fse_fpenv_pop(saved)


cdef class TaskTeam:
    """Private, single-use per-call owner; no task runs unless preparation succeeds."""
    cdef _Team team
    cdef readonly bint prepared
    cdef object owners, workspaces
    cdef int fail_allocation, preparation_steps
    cdef Py_ssize_t failed_at
    cdef int encode_status

    def __cinit__(self):
        self.team.balanced = True
        self.failed_at = -1

    def __init__(self, WeightedMean kernel not None, TextLookup lookup not None,
                 isa, bint normalize, text, tasks, out, int workers):
        self.fail_allocation = _test_team_fail_allocation
        self.team.failed_starts = _test_team_failed_starts
        self.team.corrupt = _test_corrupt_kind
        self.team.limit = _MAX_SENTENCE_BYTES
        self.team.status_task, self.team.status = _team_status_task, _team_status
        self.team.corrupt_task = _team_corrupt_task
        self.team.delay_task, self.team.delay_ms = _team_delay_task, _team_delay_ms
        self.team.hold = fse_atomic_load(&_team_hold) != 0
        self.workspaces = []
        try:
            self._prepare(kernel, lookup, isa, normalize, text, tasks, out, workers)
        except MemoryError:
            _team_release(&self.team)
            self.workspaces = []
            self.owners = None
            return
        self.prepared = True

    cdef int _allocation(self) except -1:
        self.preparation_steps += 1
        if self.preparation_steps == self.fail_allocation:
            raise MemoryError("injected task-team preparation failure")
        return 0

    cdef int _prepare(self, WeightedMean kernel, TextLookup lookup, isa, bint normalize,
                       text, tasks, out, int workers) except -1:
        cdef _Team *t = &self.team
        cdef Workspace ws
        cdef const unsigned char[::1] data
        cdef Py_ssize_t i, first_row = 0, start = 0, end, rows
        cdef int code = _check_isa(isa)
        error = _out_error(kernel, out, len(out))
        if error is not None:
            raise error
        t.n = len(tasks)
        if not 0 < t.n <= 256 or workers < 2:
            raise ValueError("task team needs 1 to 256 tasks and at least two workers")
        t.workers = min(workers, t.n)
        t.kernel, t.lookup = <PyObject *>kernel, <PyObject *>lookup
        self._allocation()
        ws = Workspace()
        self.workspaces.append(ws)
        self._allocation()
        data = text[0]
        offsets, first, last = lookup._check_encode(data, text[1], 0, len(text[1]) - 1, ws)
        self.owners = (kernel, lookup, data, offsets, out)
        self._allocation()
        t.workspaces = <PyObject **>_team_calloc(t, t.workers, sizeof(PyObject *))
        if t.workspaces == NULL:
            raise MemoryError()
        t.workspaces[0] = <PyObject *>ws
        for i in range(1, t.workers):
            self._allocation()
            ws = Workspace()
            self.workspaces.append(ws)
            t.workspaces[i] = <PyObject *>ws
        self._allocation()
        t.tasks = <_TeamTask *>_team_calloc(t, t.n, sizeof(_TeamTask))
        if t.tasks == NULL:
            raise MemoryError()
        self._allocation()
        t.slots = <fse_team_slot *>_team_calloc(t, t.workers, sizeof(fse_team_slot))
        if t.slots == NULL:
            raise MemoryError()
        for i in range(t.n):
            chunk = tasks[i].chunk
            end, rows = chunk.end, chunk.n_sentences
            if (chunk.start != start or chunk.first_row != first_row or rows <= 0
                    or end - start != rows or end > last or first_row + rows > len(out)):
                raise RuntimeError("native task team requires ordered, disjoint StringSource ranges")
            _span_of(&t.tasks[i].span, data, offsets, start, end)
            _stages_of(&t.tasks[i].stages, kernel, lookup, out, len(out), normalize, code, None)
            t.tasks[i].stages.out += first_row * kernel.dim
            t.tasks[i].stages.rows = rows
            start, first_row = end, first_row + rows
        fse_atomic_store(&t.next, 0)
        fse_atomic_store(&t.failed_at, t.n)
        return 0

    def run(self):
        global _nogil_sections, _team_calls
        if not self.prepared:
            raise RuntimeError("task team is unprepared or has already run")
        cdef _Team *t = &self.team
        cdef Py_ssize_t i
        self.prepared = False
        _team_calls += 1
        _nogil_sections += 1
        with nogil:
            fse_team(<int>t.workers, _team_worker, <void *>t, t.slots, t.failed_starts)
            self.failed_at = fse_atomic_load(&t.failed_at)
            self.encode_status = (t.tasks[self.failed_at].result.encode
                                  if self.failed_at < t.n else ENCODE_OK)
            t.balanced = fse_team_balanced(<int>t.workers, t.slots)
            for i in range(1, t.workers):
                t.started += t.slots[i].started
                t.joined += t.slots[i].joined
                t.started_ids[i] = t.slots[i].started_id
                t.actual_ids[i] = t.slots[i].actual_id
            _team_release(t)
        self.workspaces = []
        self.owners = None
        if not t.balanced:
            raise RuntimeError("native task team resource balance mismatch")
        return self.failed_at, self.encode_status == ENCODE_NO_MEMORY

    @property
    def thread_ids(self):
        return tuple((self.team.started_ids[i], self.team.actual_ids[i])
                     for i in range(1, self.team.workers) if self.team.started_ids[i])

    @property
    def balance(self):
        return (self.team.allocated, self.team.freed, self.team.started,
                self.team.joined, bool(self.team.balanced), self.preparation_steps)

    def __dealloc__(self):
        _team_release(&self.team)
