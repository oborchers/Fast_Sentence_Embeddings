#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The arguments of ``StaticModel.encode`` and ``from_pretrained`` that change no row, and
the refusal of the ones that would (docs/architecture/hub.md, "model2vec compatibility (fse.StaticModel)").

model2vec ignores every keyword it does not know (``**kwargs``), and several it knows
change nothing about the rows. fse accepts a closed list of names, each with its neutral
value or a value that cannot matter, checks every value strictly, and raises where the
caller asked for something fse does not do: a name outside the list is a ``TypeError`` of
the signature, a neutral name with another value a ``ValueError`` that says what to use.
"""

from __future__ import annotations

import numpy as np

from fse._encoder._model2vec_rules import Unset

_DTYPES = ("float16", "int8", "float64")


def flag(name: str, value: object) -> bool:
    """``value`` if it is a bool."""
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool, not {type(value).__name__}")
    return value


def count(name: str, value: object, low: int) -> int:
    """``value`` as an int of at least ``low`` (numpy integers too, bools never)."""
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an int, not {type(value).__name__}")
    if value < low:
        raise ValueError(f"{name} must be at least {low}, not {value}")
    return int(value)


def length(name: str, value: object) -> int | Unset | None:
    """A ``max_length``: not given, None (no cut) or a positive int."""
    if value is None or isinstance(value, Unset):
        return value
    return count(name, value, 1)


def texts(sentences: object, hint: str = "") -> tuple[object, bool]:
    """``sentences`` as a list, tuple or numpy array of texts, and whether it was one str
    (model2vec gives a vector or one array for it, a matrix or a list for a list)."""
    if isinstance(sentences, str):
        return [sentences], True
    if not isinstance(sentences, list | tuple | np.ndarray):
        raise TypeError(
            f"sentences must be a str or a list, tuple or numpy array of str, not "
            f"{type(sentences).__name__}{hint}"
        )
    return sentences, False


def sequence_arguments(
    *,
    show_progress_bar: object,
    batch_size: object,
    use_multiprocessing: object,
    multiprocessing_threshold: object,
) -> bool:
    """Check the arguments that change no row (fse sizes its own tasks and shows no bar);
    whether ``use_multiprocessing`` is on."""
    flag("show_progress_bar", show_progress_bar)
    count("batch_size", batch_size, 1)
    parallel = flag("use_multiprocessing", use_multiprocessing)
    count("multiprocessing_threshold", multiprocessing_threshold, 0)
    return parallel


def encode_arguments(
    *,
    show_progress_bar: object,
    batch_size: object,
    use_multiprocessing: object,
    multiprocessing_threshold: object,
    convert_to_numpy: object,
    convert_to_tensor: object,
    device: object,
) -> int | None:
    """Check the arguments of ``encode`` that change no row and give the ``workers`` the
    others imply (1 without multiprocessing, else None: fse's own)."""
    parallel = sequence_arguments(
        show_progress_bar=show_progress_bar,
        batch_size=batch_size,
        use_multiprocessing=use_multiprocessing,
        multiprocessing_threshold=multiprocessing_threshold,
    )
    if flag("convert_to_numpy", convert_to_numpy) is not True:
        raise ValueError(
            "convert_to_numpy=False: encode gives a numpy array; convert it with "
            "torch.from_numpy where a tensor is needed"
        )
    if flag("convert_to_tensor", convert_to_tensor):
        raise ValueError(
            "convert_to_tensor=True: encode gives a numpy array; convert it with "
            "torch.from_numpy where a tensor is needed"
        )
    if not (device is None or (isinstance(device, str) and device == "cpu")):
        raise ValueError(
            f"device={device!r}: encode runs on the CPU (use device=None or 'cpu'; "
            "backend='mlx' of fse.Encoder.encode uses the GPU of Apple silicon)"
        )
    return None if parallel else 1


def loading_arguments(quantize_to: object, vocabulary_quantization: object) -> None:
    """Check the arguments of ``from_pretrained`` that fse does not offer: model2vec
    quantizes the table (and computes the mean of a float16 table in float16), clusters
    the vocabulary; only ``quantize_to='float32'`` (what fse has) is accepted."""
    if quantize_to is not None:
        if not isinstance(quantize_to, str):
            raise TypeError(f"quantize_to must be a str or None, not {type(quantize_to).__name__}")
        if quantize_to != "float32":
            what = (
                "model2vec computes the mean in that dtype, which fse does not reproduce"
                if quantize_to in _DTYPES
                else "the dtypes of model2vec are float16, float32, float64 and int8"
            )
            raise ValueError(
                f"quantize_to={quantize_to!r}: fse.StaticModel has float32 tables only ({what}); "
                "fse.Encoder.encode takes dtype= for float16 or int8 rows"
            )
    if vocabulary_quantization is not None:
        raise ValueError(
            "vocabulary_quantization: fse.StaticModel does not cluster the vocabulary "
            "(models that did it are refused when loaded)"
        )
