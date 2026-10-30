"""Tiny sentence-transformers models, written by sentence-transformers itself or by hand
in its layout.

``save_model`` writes a model of one StaticEmbedding with sentence-transformers 6.2.0
(``SentenceTransformer.save_pretrained``): ``modules.json`` with the module at the root
(``path ""``), ``model.safetensors`` with ``embedding.weight``, ``tokenizer.json`` and
``config_sentence_transformers.json``. ``relayout`` turns it into the layouts other
versions write: the module in ``0_StaticEmbedding/`` and the type spellings of ST up to
5.3 (both target repos) or of 5.4 to 5.7, and the tensor named as model2vec names it.
``write_model`` writes the same layout by hand, without sentence-transformers, for the
tests of refusals.
"""

import json
import shutil

import numpy as np

from support import wordpiece as wp

DIM = 16
LEGACY = "sentence_transformers.models.StaticEmbedding"
# ST 5.4 to 5.7 and 6.x write this spelling.
V5 = "sentence_transformers.sentence_transformer.modules.static_embedding.StaticEmbedding"


def table(n_tokens: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n_tokens, DIM)).astype(np.float32)


def tokenizer(tokens=None, *, truncation=None):
    """A BERT tokenizer of ``tokens`` with the template of the target models."""
    from tokenizers.processors import TemplateProcessing

    tokens = wp.vocabulary() if tokens is None else tokens
    hf = wp.hf_tokenizer(tokens, wp.added_tokens(tokens))
    cls, sep = tokens.index("[CLS]"), tokens.index("[SEP]")
    hf.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls), ("[SEP]", sep)],
    )
    if truncation is not None:
        hf.enable_truncation(**truncation)
    return hf


def save_model(folder, rows=None, *, tokens=None, truncation=None, truncate_dim=None):
    """A model of one StaticEmbedding saved by sentence-transformers into ``folder``."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import StaticEmbedding

    tokens = wp.vocabulary() if tokens is None else tokens
    rows = table(len(tokens)) if rows is None else rows
    module = StaticEmbedding(tokenizer(tokens, truncation=truncation), embedding_weights=rows)
    SentenceTransformer(modules=[module], truncate_dim=truncate_dim).save_pretrained(str(folder))
    return folder


def relayout(folder, *, spelling=LEGACY, subfolder="0_StaticEmbedding", tensor=None):
    """Move the module of a saved model into ``subfolder`` and spell its type as another
    version does; ``tensor`` renames the table."""
    modules = json.loads((folder / "modules.json").read_text(encoding="utf-8"))
    if subfolder:
        (folder / subfolder).mkdir()
        for name in ("model.safetensors", "tokenizer.json"):
            shutil.move(folder / name, folder / subfolder / name)
    modules[0] |= {"path": subfolder, "type": spelling}
    (folder / "modules.json").write_text(json.dumps(modules, indent=2))
    if tensor is not None:
        from safetensors.numpy import load_file, save_file

        path = folder / subfolder / "model.safetensors"
        save_file({tensor: load_file(path)["embedding.weight"]}, path)
    return folder


def write_model(folder, rows=None, *, tokens=None, modules=None, config=None, tensors=None):
    """A model in the layout of sentence-transformers 3.3 (the target repos), written
    without it: ``0_StaticEmbedding/`` with ``model.safetensors`` and ``tokenizer.json``."""
    from safetensors.numpy import save_file

    tokens = wp.vocabulary() if tokens is None else tokens
    rows = table(len(tokens)) if rows is None else rows
    module = folder / "0_StaticEmbedding"
    module.mkdir(parents=True)
    save_file(
        {"embedding.weight": rows} if tensors is None else tensors, module / "model.safetensors"
    )
    tokenizer(tokens).save(str(module / "tokenizer.json"))
    if modules is None:
        modules = [{"idx": 0, "name": "0", "path": "0_StaticEmbedding", "type": LEGACY}]
    (folder / "modules.json").write_text(json.dumps(modules))
    if config is None:
        config = {"prompts": {}, "default_prompt_name": None, "similarity_fn_name": "cosine"}
    if config is not False:
        (folder / "config_sentence_transformers.json").write_text(json.dumps(config))
    return folder
