import os

import numpy as np
import pytest

from fse import Encoder


@pytest.fixture
def encoder():
    return Encoder.from_arrays(
        np.array([[1, 0, 2], [0, 2, 1]], dtype=np.float32),
        words=["hello", "world"],
    )


os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "false"
