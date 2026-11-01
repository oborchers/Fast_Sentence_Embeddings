"""Compare a tiny 1.x export with 2.0, without downloads or timing.

Run `export PATH` with origin/1.x installed, then `compare PATH` with 2.0.
The NPZ contains only numeric/string arrays; no pickles are loaded by 2.0.
"""

import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

WORDS = ["cat", "dog", "sleep", "run", "the", "bird"]
TRAIN = [["cat", "sleep"], ["dog", "run"], ["the", "bird"], ["sleep", "run", "bird"]]
PROBES = [["cat", "sleep"], ["cat", "missing"], ["missing"], []]


def revision(path):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def export(path, source_commit):
    from gensim.models import KeyedVectors

    import fse

    rng = np.random.default_rng(197)
    table = rng.normal(size=(len(WORDS), 4)).astype(np.float32)
    kv = KeyedVectors(4)
    kv.add_vectors(WORDS, table)
    for word, count in zip(WORDS, [50, 30, 20, 10, 100, 5], strict=True):
        kv.set_vecattr(word, "count", count)
    arrays = {"words": np.array(WORDS), "table": table}
    for name, cls in [("average", fse.Average), ("sif", fse.SIF), ("usif", fse.uSIF)]:
        count = {"average": 0, "sif": 1, "usif": 2}[name]
        kwargs = {} if name == "average" else {"components": count}
        model = cls(kv, workers=1, **kwargs)
        model.train(fse.IndexedList(TRAIN))
        folded = model.wv.vectors * model.word_weights[:, None]
        factors = np.empty(0, np.float32)
        if name != "average":
            basis = model.svd_res[1]
            factors = model.svd_weights if name == "usif" else np.ones(len(basis), np.float32)
            folded -= ((folded @ basis.T) * factors) @ basis
        if name == "usif":
            assert len(factors) == 2
            assert np.all((factors > 0) & (factors < 1))
        arrays[f"{name}_components"] = np.array(count)
        arrays[f"{name}_factors"] = factors
        arrays[f"{name}_table"] = folded
        for norm in [False, True]:
            arrays[f"{name}_{norm}"] = model.infer(fse.IndexedList(PROBES), use_norm=norm)
    arrays["legacy_version"] = np.array(fse.__version__)
    arrays["legacy_commit"] = np.array(source_commit)
    np.savez(path, **arrays)


def compare(path):
    from fse import Encoder

    report = {
        "probes": PROBES,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "model_details": {},
        "results": {},
    }
    with np.load(path, allow_pickle=False) as data:
        report["legacy_version"] = str(data["legacy_version"])
        report["legacy_commit"] = str(data["legacy_commit"])
        for name in ["average", "sif", "usif"]:
            report["model_details"][name] = {
                "components": int(data[f"{name}_components"]),
                "factors": data[f"{name}_factors"].tolist(),
            }
            for norm in [False, True]:
                encoder = Encoder.from_arrays(
                    data[f"{name}_table"],
                    words=data["words"].tolist(),
                    tokenizer="external",
                    normalize=norm,
                )
                actual = encoder.encode(PROBES, workers=1)
                expected = data[f"{name}_{norm}"]
                report["results"][f"{name}_normalize_{norm}"] = {
                    "legacy": expected.tolist(),
                    "v2": actual.tolist(),
                    "bitwise_equal": bool(
                        np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
                    ),
                    "max_abs_error_by_row": np.max(np.abs(actual - expected), axis=1).tolist(),
                }
    report["v2_commit"] = revision(Path(__file__).resolve().parents[1])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["export", "compare"])
    parser.add_argument("path", type=Path)
    parser.add_argument("--legacy-commit", help="source commit of the installed 1.x branch")
    args = parser.parse_args()
    if args.mode == "export":
        if not args.legacy_commit:
            parser.error("export requires --legacy-commit")
        export(args.path, args.legacy_commit)
    else:
        compare(args.path)
