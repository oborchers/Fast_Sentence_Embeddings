"""The small public classifier fixture shared by prediction tests."""

import numpy as np

import fse
from support.encoder import finished

F32 = np.float32


def model(k=3, labels=None, normalize=False, fasttext=True, **kwargs):
    encoder = finished("float16", fasttext=fasttext, normalize=normalize)
    rng = np.random.default_rng(k)
    weight = rng.standard_normal((k, encoder.output_dim)).astype(F32)
    bias = rng.standard_normal(k).astype(F32)
    names = labels or (["no", "yes"] if k == 1 else [f"c{i}" for i in range(k)])
    return fse.Classifier.from_arrays(encoder, weight, bias, labels=names, **kwargs)
