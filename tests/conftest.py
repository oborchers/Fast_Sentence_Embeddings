import sys

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip memmap tests on Windows.

    These tests delete memory-mapped files while the mapping is still open,
    which Windows does not allow. The leftover files then break later tests
    that reuse the same paths.
    """
    if sys.platform != "win32":
        return
    skip = pytest.mark.skip(
        reason="memory-mapped test files cannot be deleted on Windows"
    )
    for item in items:
        if "memmap" in item.keywords:
            item.add_marker(skip)
