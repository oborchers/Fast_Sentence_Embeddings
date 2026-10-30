"""fse.StaticModel on minishlab/potion-multilingual-128M (a Unigram tokenizer, a table of
512 MB) at a pinned commit, against model2vec 0.10.0.

Marked ``network`` (``make test-network``); a file of its own, as it is the largest download.
"""

import pytest

import fse
from support import static_model as sm
from support import unigram as uni

pytestmark = pytest.mark.network
REPO = "minishlab/potion-multilingual-128M"
REVISION = "73908c3438cf03b6a01bcb9611d62b23d0726f08"
FILES = ["config.json", "model.safetensors", "tokenizer.json", "modules.json", "README.md"]


@pytest.fixture(autouse=True)
def online(monkeypatch):
    """The tests of the fake Hub set the offline mode, and huggingface_hub reads it once:
    these tests need the network, whatever ran before them in the process."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)


def test_potion_multilingual_is_the_static_model_of_model2vec():
    model2vec = pytest.importorskip("model2vec")
    huggingface_hub = pytest.importorskip("huggingface_hub")

    folder = huggingface_hub.snapshot_download(REPO, revision=REVISION, allow_patterns=FILES)
    ours = fse.StaticModel.from_pretrained(REPO, revision=REVISION)
    theirs = model2vec.StaticModel.from_pretrained(folder)
    assert ours.encoder.tokenizer == "unigram@1"
    texts = [*uni.corpus(2000, seed=13), *sm.hub_texts(ours)[:3000]]
    worst = sm.assert_the_model_of_model2vec(ours, theirs, texts)
    print(f"{REPO}: {len(texts)} texts, the largest error is {worst:.2e} of the bound")
    assert worst <= 1
