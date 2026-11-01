"""Run the guide's actual 2.0 code blocks, with local stand-ins for downloads."""

import re
from pathlib import Path

import numpy as np
import pytest

from fse import Encoder

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "migrating-from-1x.md"
GENSIM_BLOCKS = {"gensim", "legacy-hub"}


def _blocks():
    blocks = {}
    for block in re.findall(r"```python\n(.*?)```", GUIDE.read_text(), re.S):
        if block.startswith("# 2.0:"):
            name = block.splitlines()[0].removeprefix("# 2.0: ")
            blocks[name] = block
    assert len(blocks) == 9
    return blocks


def test_migration_guide_offline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    table = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.float32)
    words = ["cat", "dog", "sleep"]
    np.savez("folded.npz", words=np.array(words), vectors=table)
    hub_calls = []

    def local_pretrained(repo, *, revision):
        hub_calls.append((repo, revision))
        return Encoder.from_arrays(table, words=words)

    monkeypatch.setattr(Encoder, "from_pretrained", local_pretrained)
    namespace = {}
    for name, block in _blocks().items():
        if name not in GENSIM_BLOCKS:
            exec(compile(block, str(GUIDE), "exec"), namespace)
    assert hub_calls == [("minishlab/potion-base-8M", "bf8b056651a2c21b8d2565580b8569da283cab23")]
    assert namespace["folded_rows"].shape == (2, 3)


def test_migration_guide_gensim(tmp_path, monkeypatch):
    pytest.importorskip("gensim")  # optional on musllinux and Python 3.14+
    from gensim.models import KeyedVectors

    monkeypatch.chdir(tmp_path)
    namespace = {}
    blocks = _blocks()
    exec(compile(blocks["average"], str(GUIDE), "exec"), namespace)
    kv = KeyedVectors(3)
    kv.add_vectors(namespace["words"], namespace["table"])
    kv.save("glove-wiki-gigaword-50.model")
    for name, block in blocks.items():
        if name in GENSIM_BLOCKS:
            exec(compile(block, str(GUIDE), "exec"), namespace)
    assert namespace["legacy_rows"].shape == (1, 3)


@pytest.mark.network
def test_migration_guide_hub():
    block = _blocks()["hub (network)"]
    namespace = {"Encoder": Encoder}
    exec(compile(block, str(GUIDE), "exec"), namespace)
    assert np.isfinite(namespace["hub_rows"]).all()
