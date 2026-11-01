"""Run the guide's actual 2.0 code blocks, with local stand-ins for downloads."""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

from fse import Encoder

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "migrating-from-1x.md"


def _blocks():
    return [
        block
        for block in re.findall(r"```python\n(.*?)```", GUIDE.read_text(), re.S)
        if block.startswith("# 2.0:")
    ]


def test_migration_guide_offline(tmp_path, monkeypatch):
    if sys.version_info >= (3, 14):
        pytest.importorskip("gensim")
    from gensim.models import KeyedVectors

    monkeypatch.chdir(tmp_path)
    table = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.float32)
    words = ["cat", "dog", "sleep"]
    np.savez("folded.npz", words=np.array(words), vectors=table)
    kv = KeyedVectors(3)
    kv.add_vectors(words, table)
    kv.save("glove-wiki-gigaword-50.model")
    hub_calls = []

    def local_pretrained(repo, *, revision):
        hub_calls.append((repo, revision))
        return Encoder.from_arrays(table, words=words)

    monkeypatch.setattr(Encoder, "from_pretrained", local_pretrained)
    namespace = {}
    blocks = _blocks()
    assert len(blocks) == 9
    for block in blocks:
        exec(compile(block, str(GUIDE), "exec"), namespace)
    assert hub_calls == [("minishlab/potion-base-8M", "bf8b056651a2c21b8d2565580b8569da283cab23")]
    assert namespace["legacy_rows"].shape == (1, 3)
    assert namespace["folded_rows"].shape == (2, 3)


@pytest.mark.network
def test_migration_guide_hub():
    block = next(block for block in _blocks() if block.startswith("# 2.0: hub (network)"))
    namespace = {"Encoder": Encoder}
    exec(compile(block, str(GUIDE), "exec"), namespace)
    assert np.isfinite(namespace["hub_rows"]).all()
