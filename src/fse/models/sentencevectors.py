#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers


import gc
import logging
import os
import traceback
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, cast

from gensim import matutils, utils
from gensim.models.keyedvectors import KeyedVectors
from numpy import (
    arange,
    argsort,
    array,
    asarray,
    ascontiguousarray,
    concatenate,
    dot,
    dtype,
    empty,
    errstate,
    finfo,
    flatnonzero,
    float64,
    floating,
    int64,
    integer,
    isfinite,
    isnan,
    issubdtype,
    lexsort,
    may_share_memory,
    ndarray,
    newaxis,
    partition,
    prod,
    sqrt,
    vstack,
    where,
    zeros,
)
from numpy import (
    float32 as REAL,
)
from numpy import (
    memmap as np_memmap,
)

from fse._validation import (
    NonNegativeInt,
    PathLike,
    PositiveInt,
    Similarity,
    validate_arguments,
)
from fse.models._quantize import (
    STORAGE_TYPES,
    StorageDtype,
    dequantize,
    has_scales,
    quantize_into,
)
from fse.models.utils import REMOVAL_BLOCK_ROWS

try:
    from fse.models.average_inner import score_rows
except ImportError:
    score_rows = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class _StoredScores:
    """Scores of stored float16 or int8 rows: the compiled kernel in threads, or numpy.

    ``run(codes, scales, weights_t, bias, factors, out)`` sets
    ``out[i] = (codes[i] @ weights_t.T) * scales[i] * factors[i] + bias``. With the
    extension, row ranges run in parallel without the GIL and without converting a block
    to float32 first; without it, blocks are converted and multiplied with numpy.
    """

    def __init__(self, workers: int | None):
        self.workers = (os.cpu_count() or 1) if workers is None else int(workers)
        self._pool: ThreadPoolExecutor | None = None

    def __enter__(self) -> "_StoredScores":
        return self

    def __exit__(self, *exc: object):
        if self._pool is not None:
            self._pool.shutdown()

    def run(
        self,
        codes: ndarray,
        scales: ndarray | None,
        weights_t: ndarray,
        bias: ndarray | None,
        factors: ndarray | None,
        out: ndarray,
    ):
        n = len(codes)
        if score_rows is None:
            for begin in range(0, n, REMOVAL_BLOCK_ROWS):
                end = begin + REMOVAL_BLOCK_ROWS
                # Scaled rows first, so a code times a large weight cannot overflow.
                rows = array(codes[begin:end], dtype=REAL)
                for per_row in (scales, factors):
                    if per_row is not None:
                        rows *= per_row[begin:end, newaxis]
                block = dot(rows, weights_t.T)
                if bias is not None:
                    block += bias
                out[begin:end] = block
            return
        # Ranges of at least one block, about four per worker.
        step = max(REMOVAL_BLOCK_ROWS, -(-n // (self.workers * 4)))
        ranges = [(begin, min(begin + step, n)) for begin in range(0, n, step)]
        if self.workers == 1 or len(ranges) == 1:
            for begin, end in ranges:
                score_rows(codes, scales, weights_t, bias, factors, out, begin, end)
            return
        if self._pool is None:
            self._pool = ThreadPoolExecutor(self.workers)
        futures = [
            self._pool.submit(score_rows, codes, scales, weights_t, bias, factors, out, b, e)
            for b, e in ranges
        ]
        try:
            for future in futures:
                future.result()
        except BaseException:
            # No range may still write into out once this returns, also on a further
            # interrupt while waiting: cancel what did not start, wait for the rest.
            for future in futures:
                future.cancel()
            interrupted: BaseException | None = None
            while not all(future.done() for future in futures):
                try:
                    wait(futures)
                except KeyboardInterrupt as err:
                    interrupted = err
            if interrupted is not None:
                raise interrupted from None
            raise


def _block_dots(
    vectors: ndarray,
    query: ndarray,
    factors: ndarray | None,
    begin: int,
    end: int,
    scores: _StoredScores | None = None,
) -> ndarray:
    """The dot products of rows begin to end with query, in float32.

    For stored float16 or int8 rows, ``factors`` scales the dot product of each row (its
    int8 scale and inverse norm), so no normalized copy of the matrix is needed.
    """
    block = vectors[begin:end]
    if factors is None:
        return dot(block, query)
    dots = empty((len(block), 1), dtype=REAL)
    runner = scores if scores is not None else _StoredScores(1)
    runner.run(
        ascontiguousarray(block),
        None,
        ascontiguousarray(query[newaxis, :], dtype=REAL),
        None,
        ascontiguousarray(factors[begin:end], dtype=REAL),
        dots,
    )
    return dots[:, 0]


def _most_similar_rows(
    vectors: ndarray,
    query: ndarray,
    count: int,
    factors: ndarray | None = None,
    scores: _StoredScores | None = None,
) -> tuple[ndarray, ndarray]:
    """The rows with the ``count`` largest dot products with ``query``, largest first.

    The rows are read in blocks of ``max(REMOVAL_BLOCK_ROWS, count)``, and only the best
    ``count`` candidates are kept between blocks, so the memory of the search grows with
    ``count``, not with the number of rows.
    The dot products equal those of a single ``dot(vectors, query)`` up to float32
    rounding, since BLAS may sum a block differently. Of equal products the lower row
    wins; NaN ranks below every number, as in ``argsort``. With ``factors``, each row's dot
    product is multiplied by its factor (see ``_block_dots``).
    """
    best_rows = empty(0, dtype=int64)
    best_keys = empty(0, dtype=REAL)
    # At least as many rows per block as candidates, so the candidates are copied O(rows)
    # times in total, also for a count close to the number of rows.
    block = max(REMOVAL_BLOCK_ROWS, count)
    if factors is not None and scores is not None:
        # Stored rows are scored in parallel; larger blocks keep every worker busy.
        block = max(block, REMOVAL_BLOCK_ROWS * scores.workers * 4)
    for begin in range(0, len(vectors), block):
        # Negated, so the best rows are the smallest keys and NaN sorts after them.
        dots = _block_dots(vectors, query, factors, begin, begin + block, scores)
        keys = concatenate([best_keys, -dots])
        rows = concatenate([best_rows, arange(begin, begin + len(keys) - len(best_keys))])
        if len(keys) > count:
            kth = partition(keys, count - 1)[count - 1]
            if isnan(kth):
                # Fewer numbers than places: all of them, then NaN rows fill up.
                better, ties = flatnonzero(~isnan(keys)), flatnonzero(isnan(keys))
            else:
                better, ties = flatnonzero(keys < kth), flatnonzero(keys == kth)
            # Of the rows tied with the count-th, the lowest ones fill the remaining places.
            ties = ties[argsort(rows[ties], kind="stable")[: count - len(better)]]
            keep = concatenate([better, ties])
            keys, rows = keys[keep], rows[keep]
        best_keys, best_rows = keys, rows
    order = lexsort((best_rows, best_keys))
    return best_rows[order], -best_keys[order]


class SentenceVectors(utils.SaveLoad):
    """The sentence vectors of a model, stored as float32, float16 or int8.

    ``vectors`` holds the stored rows in ``dtype``; for int8 they are codes and ``scales``
    holds one float32 scale per row (see :mod:`fse.models._quantize`). Every read, such as
    ``sv[i]``, :meth:`get_vector`, :meth:`get_vectors` and :meth:`iter_blocks`, returns
    float32 rows, whatever the storage format.
    """

    vectors_norm: ndarray | None
    mapfile_shape: tuple[int, int] | None
    scales: ndarray | None
    inv_norms: ndarray | None

    @validate_arguments
    def __init__(
        self,
        vector_size: NonNegativeInt,
        mapfile_path: PathLike | None = None,
        dtype: StorageDtype = "float32",
    ):
        """
        Parameters
        ----------
        vector_size : int
            Dimension of the sentence vectors.
        mapfile_path : str, optional
            Store the vectors in memory-mapped files with this path prefix.
        dtype : {"float32", "float16", "int8"}, optional
            Storage format. float16 halves and int8 quarters the memory of the vectors;
            reads always return float32.
        """
        self.vector_size = vector_size  # Size of vectors
        self.dtype = dtype
        self.vectors = zeros((0, vector_size), STORAGE_TYPES[dtype])  # Vectors for sentences
        self.scales = zeros(0, REAL) if has_scales(dtype) else None
        self.vectors_norm = None
        # Inverse norms of the rows, the search cache of float16 and int8 storage.
        self.inv_norms = None

        # File for numpy memmap
        self.mapfile_path = Path(mapfile_path) if mapfile_path is not None else None
        self.mapfile_shape = None

    def __setstate__(self, state: dict):
        # Sentence vectors saved before 1.1.0 are float32 without scales.
        self.__dict__.update(state)
        self.__dict__.setdefault("dtype", "float32")
        self.__dict__.setdefault("scales", None)
        self.__dict__.setdefault("inv_norms", None)

    @property
    def quantized(self) -> bool:
        """Whether the vectors are stored in another format than float32."""
        return self.dtype != "float32"

    def __getitem__(self, entities: int) -> ndarray:
        """Get vector representation of `entities`.

        Parameters
        ----------
        entities : {int, list of int}
            Index or sequence of entities.

        Returns
        -------
        numpy.ndarray
            Vector representation for `entities` (1D if `entities` is int, otherwise - 2D).

        """

        if isinstance(
            entities,
            (
                int,
                integer,
            ),
        ):
            return self.get_vector(entities)

        return vstack([self.get_vector(e) for e in entities])

    def __contains__(self, index: object) -> bool:
        if isinstance(index, bool) or not isinstance(index, (int, integer)):
            return False
        return bool(0 <= index < len(self))

    def __len__(self) -> int:
        return len(self.vectors)

    def _load_all_vectors_from_disk(self, mapfile_path: Path):
        """Reads all vectors (and int8 scales) from disk"""
        path = str(mapfile_path.absolute())
        shape = self.mapfile_shape
        self.vectors = _open_mapfile(Path(f"{path}.vectors"), STORAGE_TYPES[self.dtype], shape)
        if shape is None:
            self.vectors = self.vectors.reshape(-1, self.vector_size)
        if has_scales(self.dtype):
            rows = None if shape is None else (shape[0],)
            self.scales = _open_mapfile(Path(f"{path}.scales"), REAL, rows)

    def save(self, *args, **kwargs):
        """Save object.

        Parameters
        ----------
        fname : str
            Path to the output file.

        See Also
        --------
        :meth:`~fse.models.sentencevectors.SentenceVectors.load`
            Load object.

        """
        self.mapfile_shape = self.vectors.shape
        ignore = ["vectors_norm", "inv_norms", "_saved"]
        # don't bother storing the cached normalized vectors
        if self.mapfile_path is not None:
            ignore += ["vectors", "scales"]
        kwargs["ignore"] = kwargs.get("ignore", ignore)
        super().save(*args, **kwargs)

    @classmethod
    def load(cls, fname_or_handle, **kwargs):
        # TODO: Unittests
        sv = super().load(fname_or_handle, **kwargs)
        path = sv.mapfile_path
        if path is not None:
            sv._load_all_vectors_from_disk(mapfile_path=path)
        return sv

    def get_vector(self, index: int, use_norm: bool = False) -> ndarray:
        """Get sentence representations in vector space, as a 1D numpy array.

        Parameters
        ----------
        index : int
            Input index
        use_norm : bool, optional
            If True - resulting vector will be L2-normalized (unit euclidean length).

        Returns
        -------
        numpy.ndarray
            Vector representation of index.

        Raises
        ------
        KeyError
            If index out of bounds.

        """
        if index in self:
            if self.quantized and use_norm:
                # The direction of the stored row: its int8 scale cancels out.
                self.init_sims()
                result = array(self.vectors[index], dtype=REAL)
                result *= cast(ndarray, self.inv_norms)[index]
            elif self.quantized:
                result = self.get_vectors(int(index), int(index) + 1)[0]
            else:
                if use_norm:
                    self.init_sims()
                result = (
                    cast(ndarray, self.vectors_norm)[index] if use_norm else self.vectors[index]
                )

            result.setflags(write=False)
            return result
        else:
            raise KeyError(f"index {index} not found")

    def get_vectors(self, start: int = 0, stop: int | None = None) -> ndarray:
        """The float32 sentence vectors of rows start to stop, as a new array.

        start and stop follow slice semantics, so negative values count from the end.

        Stored float16 and int8 rows are converted to float32. For many rows, prefer
        :meth:`iter_blocks`, which never holds more than one block in memory.
        """
        for value, name in ((start, "start"), (stop, "stop")):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, integer))
            ):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        rows = slice(start, stop)
        scales = None if self.scales is None else self.scales[rows]
        return dequantize(self.vectors[rows], scales)

    def iter_blocks(
        self, block_rows: PositiveInt = REMOVAL_BLOCK_ROWS
    ) -> Iterator[tuple[int, ndarray]]:
        """Yield ``(first_row, vectors)`` for consecutive blocks of float32 sentence vectors.

        The streaming access to all vectors, for example to score a model over them: each
        block is converted from the storage format on its own, so memory stays at one block
        also for memory-mapped vectors larger than RAM.
        """
        if (
            isinstance(block_rows, bool)
            or not isinstance(block_rows, (int, integer))
            or block_rows < 1
        ):
            raise ValueError(f"block_rows must be a positive int, got {block_rows!r}")
        for begin in range(0, len(self), int(block_rows)):
            yield begin, self.get_vectors(begin, begin + int(block_rows))

    def init_sims(self, replace: bool = False):
        """Precompute L2-normalized vectors.

        Parameters
        ----------
        replace : bool, optional
            If True, normalize ``vectors`` themselves and keep no separate copy. This saves
            memory, but the original vectors are lost.
        """
        if self.quantized:
            self._init_inverse_norms(replace)
            return
        if getattr(self, "vectors_norm", None) is not None and not replace:
            return
        logger.info("precomputing L2-norms of sentence vectors")
        if replace:
            self.vectors = _l2_norm(self.vectors, replace=True)
            self.vectors_norm = self.vectors
        elif self.mapfile_path is not None:
            norm_path = Path(f"{self.mapfile_path}.vectors_norm")
            # Reuse a file of the right size instead of truncating it: another model loaded
            # from the same files may have it mapped, and Windows refuses to truncate then.
            expected_size = int(prod(self.vectors.shape)) * dtype(REAL).itemsize
            exists = norm_path.exists() and norm_path.stat().st_size == expected_size
            self.vectors_norm = np_memmap(
                norm_path,
                dtype=REAL,
                mode="r+" if exists else "w+",
                shape=self.vectors.shape,
            )
            # Normalize in blocks so a memory-mapped matrix never has to fit into RAM.
            block = 100_000
            for start in range(0, len(self.vectors), block):
                self.vectors_norm[start : start + block] = _l2_norm(
                    self.vectors[start : start + block]
                )
        else:
            self.vectors_norm = _l2_norm(self.vectors)

    def _init_inverse_norms(self, replace: bool):
        """The search cache of float16 and int8 storage: one inverse norm per stored row.

        The norms are those of the stored values, the int8 codes without their scales: a
        positive scale does not change the direction, and a code row of int8 values has a
        norm of at least 1, so its inverse fits into float32 however small the vector is.
        With replace, the rows themselves are normalized: for int8 the scales become the
        inverse code norms, float16 rows are rewritten in blocks.
        """
        if self.inv_norms is not None and not replace:
            return
        logger.info("precomputing L2-norms of sentence vectors")
        inv_norms = empty(len(self), dtype=REAL)
        for begin in range(0, len(self), REMOVAL_BLOCK_ROWS):
            stored = asarray(self.vectors[begin : begin + REMOVAL_BLOCK_ROWS], dtype=float64)
            end = begin + len(stored)
            norms = sqrt((stored**2).sum(axis=1))
            # A zero vector has no direction; it stays at zero instead of producing NaN.
            inv = where(norms > 0, 1 / where(norms > 0, norms, 1), 0)
            if replace:
                if self.scales is not None:
                    self.scales[begin:end] = inv
                else:
                    self.vectors[begin:end] = stored * inv[:, newaxis]
            inv_norms[begin:end] = inv
        self.inv_norms = inv_norms
        if replace:
            # The rows are normalized now, up to the rounding of their format.
            self.inv_norms = None
            self._init_inverse_norms(replace=False)

    def _search_factors(self) -> ndarray | None:
        """Per-row factors of the dot products of stored rows, for float16 and int8.

        The inverse norms of the stored values: dot(codes, query) times them is the cosine.
        """
        if not self.quantized:
            return None
        self.init_sims()
        return self.inv_norms

    @validate_arguments
    def score(
        self,
        weights: Any,
        bias: Any = None,
        workers: PositiveInt | None = None,
        out: Any = None,
    ) -> ndarray:
        """``X @ weights + bias`` for the float32 sentence vectors X, in blocks of rows.

        The way to apply a linear model to all sentence vectors, for example a logistic
        regression with ``sigmoid(sv.score(coef.T, intercept))``. float16 and int8 rows are
        scored as stored by the compiled kernel in ``workers`` threads (all cores by
        default), without a float32 copy; float32 rows use BLAS block by block. Memory stays
        at one block besides the result, which may be a memmap passed as ``out``.

        Parameters
        ----------
        weights : ndarray
            Finite float weights of shape (vector_size,) or (vector_size, k).
        bias : float or ndarray, optional
            Finite bias, a scalar or of shape (k,).
        workers : int, optional
            Threads for float16 and int8 rows; all cores by default.
        out : ndarray, optional
            float32 array of shape (n,) for vector weights or (n, k), written in place.

        Returns
        -------
        ndarray
            The scores, of shape (n,) or (n, k), in float32.
        """
        w = _real_array(weights, "weights")
        if w.ndim not in (1, 2) or w.shape[0] != self.vector_size or w.size == 0:
            raise ValueError(
                f"weights must have shape ({self.vector_size},) or ({self.vector_size}, k) "
                f"with k > 0, got {w.shape}"
            )
        k = 1 if w.ndim == 1 else w.shape[1]
        # Copies, so writing out can never change the weights while they are read.
        weights_t = _float32_copy(w.reshape(self.vector_size, k).T, "weights")
        b = None
        if bias is not None:
            b = _real_array(bias, "bias")
            if b.shape not in ((), (k,)):
                raise ValueError(f"bias must be a scalar or have shape ({k},), got {b.shape}")
            b = _float32_copy(b.reshape(-1).repeat(k) if b.ndim == 0 else b, "bias")
        shape = (len(self),) if w.ndim == 1 else (len(self), k)
        if out is None:
            out = empty(shape, dtype=REAL)
        elif not isinstance(out, ndarray) or out.dtype != REAL or out.shape != shape:
            raise ValueError(f"out must be a float32 ndarray of shape {shape}")
        elif not out.flags.c_contiguous or not out.flags.writeable:
            raise ValueError("out must be C-contiguous and writeable")
        for stored, name in ((self.vectors, "the sentence vectors"), (self.scales, "the scales")):
            if stored is not None and may_share_memory(out, stored):
                raise ValueError(f"out must not overlap {name}")
        # A second map of a mapfile has its own memory, but writing it changes the file.
        written = _mapped_file(out)
        if written is not None and self.mapfile_path is not None:
            for suffix in (".vectors", ".scales"):
                path = Path(f"{self.mapfile_path}{suffix}")
                if path.exists() and os.path.samefile(written, path):
                    raise ValueError(f"out must not be a map of the mapfile {path}")
        rows = out.reshape(len(self), k)
        if not self.quantized:
            # Contiguous weights, and a vector for k = 1, so BLAS takes its fast routines.
            matrix = ascontiguousarray(weights_t.T)
            right = matrix[:, 0] if k == 1 else matrix
            for begin in range(0, len(self), REMOVAL_BLOCK_ROWS):
                end = begin + REMOVAL_BLOCK_ROWS
                block = dot(self.vectors[begin:end], right).reshape(-1, k)
                if b is not None:
                    block += b
                rows[begin:end] = block
            return out
        scales = None if self.scales is None else ascontiguousarray(self.scales, dtype=REAL)
        with _StoredScores(workers) as scores:
            scores.run(self.vectors, scales, weights_t, b, None, rows)
        return out

    @validate_arguments
    def convert(self, dtype: StorageDtype):
        """Store the sentence vectors in another format, in place and in blocks of rows.

        Also converts vectors of models saved by earlier versions, and memory-mapped
        vectors, which are written to new files that then replace the old ones. Converting
        to float16 or int8 rounds the vectors; converting back gives the rounded values.
        """
        if dtype == self.dtype:
            return
        _write_rows(
            lambda: _RowWriter(self, len(self), append=False, dtype=dtype, suffix=".converting"),
            lambda writer: [writer.store(begin, block) for begin, block in self.iter_blocks()],
        )

    def _allocate(
        self, dtype: StorageDtype, rows: int, suffix: str = ""
    ) -> tuple[ndarray, ndarray | None, list[tuple[Path, Path]]]:
        """Arrays for rows in a storage format, as memmaps next to the mapfile or in RAM.

        Returns the arrays and, for memmaps, pairs of (written file, final file).
        """
        shape = (rows, self.vector_size)
        if self.mapfile_path is None:
            codes = empty(shape, dtype=STORAGE_TYPES[dtype])
            return codes, (empty(rows, dtype=REAL) if has_scales(dtype) else None), []
        files = [
            (Path(f"{self.mapfile_path}.vectors{suffix}"), Path(f"{self.mapfile_path}.vectors"))
        ]
        codes = np_memmap(str(files[0][0]), dtype=STORAGE_TYPES[dtype], mode="w+", shape=shape)
        scales = None
        if has_scales(dtype):
            files.append(
                (Path(f"{self.mapfile_path}.scales{suffix}"), Path(f"{self.mapfile_path}.scales"))
            )
            scales = np_memmap(str(files[1][0]), dtype=REAL, mode="w+", shape=(rows,))
        return codes, scales, files

    def _install(self, dtype: StorageDtype, codes: ndarray, scales: ndarray | None):
        """Make arrays in RAM the stored vectors."""
        self.dtype = dtype
        self.vectors, self.scales = codes, scales
        self.vectors_norm = None
        self.inv_norms = None

    def _replace_files(self, dtype: StorageDtype, files: list[tuple[Path, Path]]):
        """Make written files the mapfiles, replacing the old ones, and open them.

        The caller has flushed and released the written arrays. Every map of the old files
        is released first, because Windows refuses to replace a mapped file. The old files
        move aside and come back if any step fails, including opening the new ones; vectors
        in RAM before (a first training into a mapfile) are kept for that.
        """
        previous: tuple[StorageDtype, ndarray | None, ndarray | None] = (
            self.dtype,
            self.vectors,
            self.scales,
        )
        mapped = isinstance(self.vectors, np_memmap)
        self.vectors = self.scales = None  # type: ignore[assignment]
        self.vectors_norm = None
        self.inv_norms = None
        self.__dict__.pop("_saved", None)
        if mapped:
            # Maps cannot survive the replacement; the restored files are opened again.
            previous = (previous[0], None, None)
        _close_released_maps()
        finals = [final for _, final in files]
        stale = Path(f"{self.mapfile_path}.scales")
        if not has_scales(dtype) and stale.exists():
            finals.append(stale)
        backups: list[tuple[Path, Path]] = []
        installed: list[Path] = []
        try:
            for final in finals:
                if final.exists():
                    backup = Path(f"{final}.previous")
                    os.replace(final, backup)
                    backups.append((backup, final))
            for written, final in files:
                os.replace(written, final)
                installed.append(final)
            self.dtype = dtype
            self.mapfile_shape = None
            self._open_files()
        except BaseException as err:
            traceback.clear_frames(err.__traceback__)
            self.vectors = self.scales = None  # type: ignore[assignment]
            _close_released_maps()
            for final in installed:
                final.unlink(missing_ok=True)
            for backup, final in backups:
                os.replace(backup, final)
            self.dtype = previous[0]
            if previous[1] is not None:
                self.vectors, self.scales = previous[1], previous[2]
            else:
                self._open_files()
            raise
        for backup, _ in backups:
            try:
                backup.unlink()
            except OSError as err:
                # The new vectors are in place; a leftover copy of the old ones is harmless.
                logger.warning(f"could not remove the previous mapfile {backup}: {err}")

    def _open_files(self, rows: int | None = None):
        """Open the mapfiles in the current format, with rows rows or as large as they are.

        Both maps are opened before either is installed, so a failure changes nothing.
        """
        shape = None if rows is None else (rows, self.vector_size)
        vectors: ndarray = np_memmap(
            f"{self.mapfile_path}.vectors", dtype=STORAGE_TYPES[self.dtype], mode="r+", shape=shape
        )
        if rows is None:
            vectors = vectors.reshape(-1, self.vector_size)
        scales = (
            np_memmap(
                f"{self.mapfile_path}.scales",
                dtype=REAL,
                mode="r+",
                shape=None if rows is None else (rows,),
            )
            if has_scales(self.dtype)
            else None
        )
        if scales is not None and len(scales) < len(vectors):
            raise ValueError(
                f"the mapfile {self.mapfile_path}.scales holds {len(scales)} scales for "
                f"{len(vectors)} sentence vectors"
            )
        self.vectors, self.scales = vectors, scales

    def _stage(self, rows: int):
        """Replace the stored vectors by a float32 staging matrix for rows new rows.

        Training of float16 and int8 vectors writes into it, as into float32 vectors; then
        :meth:`_store_staged` stores its rows in the format, or :meth:`_unstage` drops
        it. The stored vectors wait in ``_saved``, which only exists meanwhile, so no other
        reference keeps their maps open.
        """
        shape = (rows, self.vector_size)
        if self.mapfile_path is None:
            staging = empty(shape, dtype=REAL)
        else:
            try:
                staging = np_memmap(str(self._staging_path()), dtype=REAL, mode="w+", shape=shape)
            except BaseException as err:
                traceback.clear_frames(err.__traceback__)
                self._drop_staging_file()
                raise
        self._saved = (self.vectors, self.scales)
        self.vectors = staging
        self.scales = None
        self.vectors_norm = None
        self.inv_norms = None

    def _staging_path(self) -> Path:
        return Path(f"{self.mapfile_path}.staging.vectors")

    def _store_staged(self, append: bool):
        """Store the staged float32 rows in the format, after or instead of the saved ones."""
        # Held in a list the cleanup empties, so no reference keeps the staging map open.
        staged = [self.vectors]
        self.vectors, self.scales = self.__dict__.pop("_saved")
        rows = len(staged[0])
        try:
            _write_rows(
                lambda: self._begin_writing(rows, append),
                lambda writer: writer.store(0, staged[0]),
            )
        finally:
            staged.clear()
            self._drop_staging_file()
            self.vectors_norm = None
            self.inv_norms = None

    def _begin_writing(self, rows: int, append: bool) -> "_RowWriter":
        """A writer that stores float32 rows of a new training in the format.

        Rows are stored in any order with ``store(first, rows)``; ``commit`` makes them the
        vectors (after the existing ones with append), ``abort`` drops them and leaves the
        existing vectors as they were.
        """
        return _RowWriter(self, rows, append)

    def _unstage(self):
        """Drop the staging matrix and restore the saved vectors, after a failure."""
        if "_saved" in self.__dict__:
            self.vectors, self.scales = self.__dict__.pop("_saved")
        self.vectors_norm = None
        self.inv_norms = None
        self._drop_staging_file()

    def _shrink_files(self, rows: int):
        """Cut the mapfiles back to rows rows and reopen them with rows rows.

        Another map of the files (Windows refuses to cut a mapped file) can make cutting
        fail; the files then stay larger, which loading accepts, and the vectors are still
        reopened with exactly rows rows, so the model stays consistent.
        """
        self.vectors = self.scales = None  # type: ignore[assignment]
        _close_released_maps()
        paths = [
            (
                Path(f"{self.mapfile_path}.vectors"),
                dtype(STORAGE_TYPES[self.dtype]).itemsize * self.vector_size,
            )
        ]
        if has_scales(self.dtype):
            paths.append((Path(f"{self.mapfile_path}.scales"), dtype(REAL).itemsize))
        for path, row_bytes in paths:
            try:
                os.truncate(path, rows * row_bytes)
            except OSError as err:
                logger.warning(f"could not cut {path} back to {rows} rows: {err}")
        self._open_files(rows)

    def _drop_staging_file(self):
        if self.mapfile_path is not None:
            _close_released_maps()
            self._staging_path().unlink(missing_ok=True)

    def similarity(self, d1: int, d2: int) -> float:
        """Compute cosine similarity between two sentences from the training set.

        Parameters
        ----------
        d1 : int
            index of sentence
        d2 : int
            index of sentence

        Returns
        -------
        float
            The cosine similarity between the vectors of the two sentences.

        """
        return dot(matutils.unitvec(self[d1]), matutils.unitvec(self[d2]))

    def distance(self, d1: int, d2: int) -> float:
        """Compute cosine similarity between two sentences from the training set.

        Parameters
        ----------
        d1 : int
            index of sentence
        d2 : int
            index of sentence

        Returns
        -------
        float
            The cosine distance between the vectors of the two sentences.

        """
        return 1 - self.similarity(d1, d2)

    @validate_arguments
    def most_similar(
        self,
        positive: Any = None,
        negative: Any = None,
        indexable: Any = None,
        topn: NonNegativeInt | None = 10,
        restrict_size: NonNegativeInt | tuple[NonNegativeInt, NonNegativeInt] | None = None,
        min_similarity: Similarity | None = None,
    ) -> list[tuple[Any, ...]] | ndarray:
        """Find the top-N most similar sentences.
        Positive sentences contribute positively towards the similarity, negative sentences negatively.

        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given sentences and the vectors for each sentence in the model.
        It is an exact search: every query reads all normalized sentence vectors in blocks.
        With a positive `topn`, the search needs memory for one block and `topn` candidates,
        not for a similarity per sentence; the normalized vectors themselves are computed
        once by :meth:`init_sims`, in RAM unless the vectors are memory-mapped. Of equally
        similar sentences the lower
        index comes first. For many queries over very many sentences, build an approximate
        nearest neighbor index (for example FAISS) from the vectors instead.

        Parameters
        ----------
        positive : list of int, optional
            List of indices that contribute positively.
        negative : list of int, optional
            List of indices that contribute negatively.
        indexable: list, IndexedList, IndexedLineDocument
            Provides an indexable object from where the most similar sentences are read
        topn : int or None, optional
            Number of top-N similar sentences to return, when `topn` is a positive int. When
            `topn` is 0 or None, the similarities to all sentences are returned as an array.
        restrict_size : int or Tuple(int,int), optional
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_size=10000 would
            only check the first 10000 sentence vectors.
            restrict_size=(500, 1000) would search the sentence vectors with indices between
            500 and 1000.
        min_similarity : float, optional
            Only return sentences whose cosine similarity is at least this value, in [-1, 1].
            Fewer than `topn` sentences are returned when fewer reach it. Requires a positive
            `topn`.

        Returns
        -------
        list of (int, float) or list of (str, int, float)
            A sequence of (index, similarity) is returned.
            When an indexable is provided, returns (str, index, similarity)
            When `topn` is None, then similarities for all words are returned as a
            one-dimensional numpy array with the size of the vocabulary.

        """
        if indexable is not None and not hasattr(indexable, "__getitem__"):
            raise RuntimeError("Indexable must provide __getitem__")
        if positive is None:
            positive = []
        if negative is None:
            negative = []

        self.init_sims()

        if isinstance(positive, (int, integer)) and not negative:
            positive = [positive]
        if isinstance(positive, (ndarray)) and not negative and len(positive.shape) == 1:
            positive = [positive]

        positive = [
            (sent, 1.0) if isinstance(sent, (int, integer, ndarray)) else sent for sent in positive
        ]
        negative = [
            (sent, -1.0) if isinstance(sent, (int, integer, ndarray)) else sent for sent in negative
        ]

        all_sents, mean = set(), []
        for sent, weight in positive + negative:
            if isinstance(sent, ndarray):
                mean.append(weight * sent)
            else:
                mean.append(weight * self.get_vector(index=sent, use_norm=True))
                if sent in self:
                    all_sents.add(sent)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

        if isinstance(restrict_size, (int, integer)):
            lo, hi = 0, restrict_size
        elif isinstance(restrict_size, tuple):
            lo, hi = restrict_size
        else:
            lo, hi = 0, None
        if hi is not None and hi < lo:
            raise ValueError(
                f"restrict_size {restrict_size} has an upper bound below its lower bound"
            )

        if min_similarity is not None and not topn:
            raise ValueError(
                "min_similarity filters the top-N list; it needs a positive topn, not "
                f"topn={topn}, which returns the similarities to all sentences"
            )

        factors = self._search_factors()
        if factors is None:
            vectors_norm = cast(ndarray, self.vectors_norm)
            limited = vectors_norm if restrict_size is None else vectors_norm[lo:hi]
        else:
            # float16 and int8 rows are searched as stored, scaled by their factors.
            limited = self.vectors if restrict_size is None else self.vectors[lo:hi]
            factors = factors if restrict_size is None else factors[lo:hi]
        if not topn:
            if factors is None:
                return dot(limited, mean)
            with _StoredScores(None) as scores:
                return _block_dots(limited, mean, factors, 0, len(limited), scores)
        with _StoredScores(None) as scores:
            best, similarities = _most_similar_rows(
                limited, mean, topn + len(all_sents), factors=factors, scores=scores
            )
        best_off = best + lo

        result: list[tuple[Any, ...]]
        if indexable is not None:
            result = [
                (indexable[off_idx], off_idx, float(sim))
                for off_idx, sim in zip(best_off.tolist(), similarities, strict=True)
                if off_idx not in all_sents
            ]
        else:
            result = [
                (off_idx, float(sim))
                for off_idx, sim in zip(best_off.tolist(), similarities, strict=True)
                if off_idx not in all_sents
            ]
        result = result[:topn]
        if min_similarity is not None:
            # result is sorted by descending similarity.
            result = [item for item in result if item[-1] >= min_similarity]
        return result

    @validate_arguments
    def similar_by_word(
        self,
        word: str,
        wv: KeyedVectors,
        indexable: Any = None,
        topn: NonNegativeInt | None = 10,
        restrict_size: NonNegativeInt | tuple[NonNegativeInt, NonNegativeInt] | None = None,
        min_similarity: Similarity | None = None,
    ) -> list[tuple[Any, ...]] | ndarray:
        """Find the top-N most similar sentences to a given word.

        Parameters
        ----------
        word : str
            Word
        wv : :class:`~gensim.models.keyedvectors.KeyedVectors`
            This object essentially contains the mapping between words and embeddings.
        indexable: list, IndexedList, IndexedLineDocument
            Provides an indexable object from where the most similar sentences are read
        topn : int or None, optional
            Number of top-N similar sentences to return, when `topn` is a positive int. When
            `topn` is 0 or None, the similarities to all sentences are returned as an array.
        restrict_size : int or Tuple(int,int), optional
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_size=10000 would
            only check the first 10000 sentence vectors.
            restrict_size=(500, 1000) would search the sentence vectors with indices between
            500 and 1000.
        min_similarity : float, optional
            Only return sentences whose cosine similarity is at least this value, in [-1, 1].
            Fewer than `topn` sentences are returned when fewer reach it. Requires a positive
            `topn`.

        Returns
        -------
        list of (int, float) or list of (str, int, float)
            A sequence of (index, similarity) is returned.
            When an indexable is provided, returns (str, index, similarity)
            When `topn` is None, then similarities for all words are returned as a
            one-dimensional numpy array with the size of the vocabulary.

        """
        return self.most_similar(
            positive=wv[word],
            indexable=indexable,
            topn=topn,
            restrict_size=restrict_size,
            min_similarity=min_similarity,
        )

    @validate_arguments
    def similar_by_sentence(
        self,
        sentence: Any,
        model: Any,
        indexable: Any = None,
        topn: NonNegativeInt | None = 10,
        restrict_size: NonNegativeInt | tuple[NonNegativeInt, NonNegativeInt] | None = None,
        min_similarity: Similarity | None = None,
    ) -> list[tuple[Any, ...]] | ndarray:
        """Find the top-N most similar sentences to a given sentence.

        Parameters
        ----------
        sentence : list of str
            Sentence as list of strings
        model : :class:`~fse.models.base_s2v.BaseSentence2VecModel`
            This object essentially provides the infer method used to transform .
        indexable: list, IndexedList, IndexedLineDocument
            Provides an indexable object from where the most similar sentences are read
        topn : int or None, optional
            Number of top-N similar sentences to return, when `topn` is a positive int. When
            `topn` is 0 or None, the similarities to all sentences are returned as an array.
        restrict_size : int or Tuple(int,int), optional
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_size=10000 would
            only check the first 10000 sentence vectors.
            restrict_size=(500, 1000) would search the sentence vectors with indices between
            500 and 1000.
        min_similarity : float, optional
            Only return sentences whose cosine similarity is at least this value, in [-1, 1].
            Fewer than `topn` sentences are returned when fewer reach it. Requires a positive
            `topn`.

        Returns
        -------
        list of (int, float) or list of (str, int, float)
            A sequence of (index, similarity) is returned.
            When an indexable is provided, returns (str, index, similarity)
            When `topn` is None, then similarities for all words are returned as a
            one-dimensional numpy array with the size of the vocabulary.

        """
        infer_op = getattr(model, "infer", None)
        if not callable(infer_op):
            raise RuntimeError(
                "Model does not have infer method. Make sure to pass a BaseSentence2VecModel"
            )

        vector = model.infer([(sentence, 0)])
        return self.most_similar(
            positive=vector,
            indexable=indexable,
            topn=topn,
            restrict_size=restrict_size,
            min_similarity=min_similarity,
        )

    @validate_arguments
    def similar_by_vector(
        self,
        vector: ndarray,
        indexable: Any = None,
        topn: NonNegativeInt | None = 10,
        restrict_size: NonNegativeInt | tuple[NonNegativeInt, NonNegativeInt] | None = None,
        min_similarity: Similarity | None = None,
    ) -> list[tuple[Any, ...]] | ndarray:
        """Find the top-N most similar sentences to a given vector.

        Parameters
        ----------
        vector : ndarray
            Vectors
        indexable: list, IndexedList, IndexedLineDocument
            Provides an indexable object from where the most similar sentences are read
        topn : int or None, optional
            Number of top-N similar sentences to return, when `topn` is a positive int. When
            `topn` is 0 or None, the similarities to all sentences are returned as an array.
        restrict_size : int or Tuple(int,int), optional
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_size=10000 would
            only check the first 10000 sentence vectors.
            restrict_size=(500, 1000) would search the sentence vectors with indices between
            500 and 1000.
        min_similarity : float, optional
            Only return sentences whose cosine similarity is at least this value, in [-1, 1].
            Fewer than `topn` sentences are returned when fewer reach it. Requires a positive
            `topn`.

        Returns
        -------
        list of (int, float) or list of (str, int, float)
            A sequence of (index, similarity) is returned.
            When an indexable is provided, returns (str, index, similarity)
            When `topn` is None, then similarities for all words are returned as a
            one-dimensional numpy array with the size of the vocabulary.

        """
        return self.most_similar(
            positive=vector,
            indexable=indexable,
            topn=topn,
            restrict_size=restrict_size,
            min_similarity=min_similarity,
        )


def _l2_norm(m, replace=False):
    """Return an L2-normalized version of a matrix.

    Parameters
    ----------
    m : np.array
        The matrix to normalize.
    replace : boolean, optional
        If True, modifies the existing matrix.

    Returns
    -------
    The normalized matrix. Rows of zeros stay zero. If replace=True and m holds floats, m is
    normalized in place and returned; other dtypes cannot be normalized in place, so a new
    float32 matrix is returned.
    """
    dist = sqrt((m**2).sum(-1))[..., newaxis]
    # A zero vector has no direction; leave it at zero instead of producing NaN.
    dist[dist == 0] = 1
    if replace and issubdtype(m.dtype, floating):
        m /= dist
        return m
    return (m / dist).astype(REAL)


def _open_mapfile(path: Path, kind: Any, shape: tuple[int, ...] | None) -> np_memmap:
    """Open a mapfile of the sentence vectors; raise unless its size fits the shape.

    Without a recorded shape (models saved before 1.1.0), the file is opened flat.
    """
    if not path.exists():
        raise FileNotFoundError(f"the mapfile {path} of the sentence vectors is missing")
    if shape is None:
        return np_memmap(str(path), dtype=kind, mode="r+")
    size = path.stat().st_size
    expected = int(prod(shape)) * dtype(kind).itemsize
    # Larger is accepted: an append that failed may not have been able to cut it back.
    if size < expected:
        raise ValueError(
            f"the mapfile {path} has {size} bytes, but sentence vectors of shape "
            f"{tuple(shape)} in {dtype(kind)} need {expected}"
        )
    return np_memmap(str(path), dtype=kind, mode="r+", shape=shape)


class _RowWriter:
    """Stores float32 rows of a training (or a conversion) in a format; see _begin_writing."""

    def __init__(
        self,
        sv: SentenceVectors,
        rows: int,
        append: bool,
        dtype: StorageDtype | None = None,
        suffix: str = ".writing",
    ):
        self.sv = sv
        self.rows = rows
        self.dtype: StorageDtype = sv.dtype if dtype is None else dtype
        self.before = len(sv.vectors) if append else 0
        self.files: list[tuple[Path, Path]] = []
        self.codes: ndarray | None
        self.scales: ndarray | None
        self.codes = self.scales = None
        self.offset = 0
        if self.before and sv.mapfile_path is not None:
            # A memmap grows in place, like float32 vectors do; abort cuts it back.
            self.mode = "grow"
            total = self.before + rows
            try:
                self.codes = np_memmap(
                    f"{sv.mapfile_path}.vectors",
                    dtype=STORAGE_TYPES[self.dtype],
                    mode="r+",
                    shape=(total, sv.vector_size),
                )
                if has_scales(self.dtype):
                    self.scales = np_memmap(
                        f"{sv.mapfile_path}.scales", dtype=REAL, mode="r+", shape=(total,)
                    )
            except BaseException as err:
                # A file may have grown already; cut both back before raising.
                traceback.clear_frames(err.__traceback__)
                self.abort()
                raise
            self.offset = self.before
        else:
            self.mode = "concatenate" if self.before else "replace"
            if sv.mapfile_path is not None:
                # Known before allocating, so abort can delete what was created.
                self.files = [
                    (Path(f"{sv.mapfile_path}.vectors{suffix}"), Path(f"{sv.mapfile_path}.vectors"))
                ]
                if has_scales(self.dtype):
                    self.files.append(
                        (
                            Path(f"{sv.mapfile_path}.scales{suffix}"),
                            Path(f"{sv.mapfile_path}.scales"),
                        )
                    )
            try:
                self.codes, self.scales, _ = sv._allocate(self.dtype, rows, suffix=suffix)
            except BaseException as err:
                traceback.clear_frames(err.__traceback__)
                self.abort()
                raise

    def store(self, first: int, rows: ndarray):
        """Store float32 rows as rows first to first + len(rows) of this training."""
        begin, end = self.offset + first, self.offset + first + len(rows)
        quantize_into(
            rows,
            cast(ndarray, self.codes)[begin:end],
            None if self.scales is None else self.scales[begin:end],
            first_row=self.before + first,
        )

    def _release(self):
        """Flush and drop the written arrays; their maps must close before files move."""
        for written in (self.codes, self.scales):
            if isinstance(written, np_memmap):
                written.flush()
        self.codes = self.scales = None

    def commit(self):
        sv = self.sv
        if self.mode == "grow":
            self._release()
            sv._open_files(self.before + self.rows)
        elif self.mode == "concatenate":
            # Both arrays exist before either is installed, so a failure changes nothing.
            vectors = vstack([sv.vectors, self.codes])
            scales = None if sv.scales is None else concatenate([sv.scales, self.scales])
            self._release()
            sv.vectors, sv.scales = vectors, scales
        elif self.files:
            self._release()
            sv._replace_files(self.dtype, self.files)
        else:
            codes, scales = self.codes, self.scales
            self._release()
            sv._install(self.dtype, cast(ndarray, codes), scales)
        sv.vectors_norm = None
        sv.inv_norms = None

    def abort(self):
        self.codes = self.scales = None
        if self.mode == "grow":
            # Cut the files back, so they still match the existing vectors.
            self.sv._shrink_files(self.before)
        if self.files:
            _close_released_maps()
        for written, _ in self.files:
            written.unlink(missing_ok=True)


def _float32_copy(value: ndarray, name: str) -> ndarray:
    """A C-contiguous float32 copy of value; raise if a value does not fit into float32."""
    with errstate(over="ignore"):
        # An overflow is reported below as a clear error, not as a numpy warning.
        converted = array(value, dtype=REAL, order="C", copy=True)
    if not isfinite(converted).all():
        raise ValueError(f"{name} must be finite in float32, beyond +-{float(finfo(REAL).max):g}")
    return converted


def _real_array(value: Any, name: str) -> ndarray:
    """value as a float array; raise for other types and for NaN or infinite values."""
    array = asarray(value)
    if array.dtype.kind != "f":
        raise TypeError(f"{name} must hold floats, got dtype {array.dtype}")
    if not isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _close_released_maps():
    """Close maps that are only waiting for the garbage collector.

    A memmap in a reference cycle keeps its file mapped until the cycle is collected, and
    Windows refuses to replace, cut or delete a mapped file.
    """
    gc.collect()


def _write_rows(make_writer: Any, store: Any):
    """Create a writer, store rows with it and commit; on any failure, abort.

    Finished frames of a failure may still hold slices of the written memmaps, so they are
    cleared before abort deletes or cuts files (Windows refuses to while they are mapped).
    """
    writer = None
    try:
        writer = make_writer()
        store(writer)
        writer.commit()
    except BaseException as err:
        traceback.clear_frames(err.__traceback__)
        if writer is not None:
            writer.abort()
        raise


def _mapped_file(value: Any) -> str | None:
    """The file that value or an array it is a view of maps, if any."""
    while isinstance(value, ndarray):
        if isinstance(value, np_memmap) and value.filename is not None:
            return str(value.filename)
        value = value.base
    return None
