"""Partition the complete emulated collection deterministically, without dropping tests."""

import json
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--fse-sde-shard", type=int, choices=(0, 1), required=True)
    parser.addoption("--fse-sde-manifest", required=True)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    ordered = sorted(items, key=lambda item: item.nodeid)
    nodeids = [item.nodeid for item in ordered]
    if not nodeids or len(nodeids) != len(set(nodeids)):
        raise pytest.UsageError("SDE requires a nonempty collection with unique node IDs")
    shard = config.getoption("--fse-sde-shard")
    selected = ordered[shard::2]
    config.hook.pytest_deselected(items=ordered[1 - shard :: 2])
    items[:] = selected
    Path(config.getoption("--fse-sde-manifest")).write_text(
        json.dumps({"shard": shard, "all": nodeids, "selected": nodeids[shard::2]}),
        encoding="utf-8",
    )
