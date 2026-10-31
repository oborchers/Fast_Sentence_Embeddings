"""Prepare mean-pooling GloVe and word2vec quality models from pinned cached tables.

No fitting occurs. The external tokenizer is the NLTK recipe in run_mteb.py.
Both quantized tables derive from the same float32 encoder without retraining.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_inputs import identity  # isort: skip
from quality_coordination import task_reservation  # isort: skip

SOURCES = {
    "glove-wiki-gigaword-300": "4e5ad402664236b85be45a0d107b95259ba8a909",
    "word2vec-google-news-300": "4274c2efa75c99a64e3f83eabfc5b34d1acabe70",
}


def prepare(folder: Path, coordination: Path) -> None:
    from gensim.models import KeyedVectors

    import fse

    folder.mkdir(parents=True, exist_ok=True)
    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    for name, revision in SOURCES.items():
        paths = [
            folder / f"Average-{name}-{dtype}.safetensors"
            for dtype in ("float32", "float16", "int8")
        ]
        if any(path.exists() for path in paths):
            raise FileExistsError(f"artifacts already exist for {name}")
        source = hub / f"models--fse--{name}" / "snapshots" / revision / f"{name}.model"
        with task_reservation(coordination):
            kv = KeyedVectors.load(str(source), mmap="r")
            encoder = fse.Encoder.from_arrays(
                kv.vectors, words=kv.index_to_key, tokenizer="external", normalize=True
            )
            encoder.save(paths[0])
            for dtype, path in zip(("float16", "int8"), paths[1:], strict=True):
                encoder.quantize(dtype).save(path)
            record = {
                **identity(),
                "source": f"fse/{name}",
                "revision": revision,
                "recipe": "unweighted mean, external NLTK tokens, L2 normalization",
                "artifacts": [path.name for path in paths],
                "output_dim": encoder.output_dim,
            }
            with (folder / "sources.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            del encoder, kv
        print(json.dumps(record), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--coordination", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.folder, args.coordination)


if __name__ == "__main__":
    main()
