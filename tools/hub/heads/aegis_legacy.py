"""Run only in a fresh fse 2.0.0 environment, against validation probes."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np

import fse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    if importlib.metadata.version("fse") != "2.0.0":
        raise ValueError("compatibility probe requires exactly fse 2.0.0")
    texts = json.loads((args.work / "compatibility-texts.json").read_text())
    safety = fse.Classifier.from_pretrained(args.work / "safety-complete")
    actual = safety.predict_proba(texts)
    expected = np.load(args.work / "compatibility-proba.npy")
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(
        safety.predict(texts), np.load(args.work / "compatibility-predict.npy")
    )
    np.testing.assert_array_equal(safety.classes_, ["safe", "unsafe"])
    try:
        fse.Classifier.from_pretrained(args.work / "categories-complete")
    except fse.UnsupportedArtifactError:
        pass
    else:
        raise AssertionError("fse 2.0.0 unexpectedly accepted the multilabel artifact")
    result = {
        "fse": "2.0.0",
        "rows": len(texts),
        "safety_probability_bits": True,
        "safety_predictions": True,
        "categories_unsupported": True,
        "summary": "Safety loads with identical probability bits and predictions; "
        "categories raises UnsupportedArtifactError.",
    }
    (args.work / "compatibility.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
