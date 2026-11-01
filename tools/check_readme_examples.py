"""Execute every README Python fence unchanged with real Hub models, never mocks.

Needs the test dependency group, network access or the normal Hub cache. Run as a
heavy job under the shared lock when applicable. This is a correctness check, not a
benchmark. Files are saved only in an owned temporary working directory.
"""

import argparse
import contextlib
import gc
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from fse import Classifier, Encoder, StaticModel

ROOT = Path(__file__).resolve().parents[1]


PYTHON_TAGS = {"python", "python3", "py"}


def fence(line):
    # CommonMark 0.31.2, section 4.5: matching fence characters and minimum length.
    match = re.fullmatch(r"( {0,3})(`{3,}|~{3,})([^\r\n]*)", line.rstrip("\r\n"))
    if match is None:
        return None
    indent, marker, info = match.groups()
    if marker[0] == "`" and "`" in info:
        return None
    return len(indent), marker, info.strip()


def opening(line, heading, number):
    value = fence(line)
    if value is None:
        if re.search(r"(?:`{3,}|~{3,})[ \t]*(?:python3?|py)(?:[ \t]|$)", line):
            raise ValueError(
                f"Unsupported nested Python fence at line {number}; use a top-level fence"
            )
        return None
    indent, marker, info = value
    language = info.split()[0] if info else ""
    return {
        "indent": indent,
        "marker": marker,
        "python": language in PYTHON_TAGS,
        "heading": heading,
        "line": number + 1,
        "code": [],
    }


def closing(line, block):
    value = fence(line)
    return (
        value is not None
        and value[1][0] == block["marker"][0]
        and len(value[1]) >= len(block["marker"])
        and not value[2]
    )


def append_line(block, line):
    # Strip only the fence's Markdown indentation, preserving code and blank lines.
    spaces = len(line) - len(line.lstrip(" "))
    block["code"].append(line[min(block["indent"], spaces) :])


def snippets(readme):
    """Read top-level Python fences; reject unsupported nesting instead of skipping it.

    https://spec.commonmark.org/0.31.2/#fenced-code-blocks
    """
    found, block, heading = [], None, "README"
    for number, line in enumerate(readme.read_text().splitlines(keepends=True), 1):
        if block is None:
            match = re.match(r"^#{1,6} (.+)", line)
            heading = match.group(1) if match else heading
            block = opening(line, heading, number)
        elif closing(line, block):
            if block["python"]:
                found.append((block["heading"], block["line"], "".join(block["code"])))
            block = None
        else:
            append_line(block, line)
    if block is not None and block["python"]:
        raise ValueError("Unclosed Python fence")
    if not found:
        raise ValueError("README contains no Python examples")
    return found


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def verify_vectors(namespace):

    if "vectors" not in namespace:
        return {}
    vectors = namespace["vectors"]
    if vectors.ndim != 2 or vectors.shape[0] != 2 or vectors.dtype != np.float32:
        raise AssertionError("Encoding example must produce two float32 vector rows")
    if not np.isfinite(vectors).all():
        raise AssertionError("Encoding example produced non-finite values")
    return {"vector_shape": list(vectors.shape), "output_dtype": str(vectors.dtype)}


def encoder_record(encoder):
    source = encoder._content.source
    return {
        "model": source.repo,
        "revision": source.revision,
        "steps": list(encoder.steps),
    }


def verify_reload_without_sklearn(path, expected):
    # A fresh interpreter has not imported sklearn during training. Refuse any attempt.
    code = """
import importlib.abc,json,sys
class RefuseSklearn(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'sklearn' or fullname.startswith('sklearn.'):
            raise AssertionError('Reload/prediction must not import scikit-learn')
sys.meta_path.insert(0, RefuseSklearn())
from fse import Classifier
model=Classifier.load(sys.argv[1])
assert model.normalized is False
print(json.dumps(model.predict(['I need a new password'], workers=1).tolist()))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(path.resolve())],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    if json.loads(result.stdout) != expected:
        raise AssertionError("Saved classifier changed predictions")


def verify_normalization(namespace):
    """Check both the folded raw path and the explicit normalized-head claim."""

    encoder, classifier, head = (namespace[k] for k in ("encoder", "classifier", "head"))
    if encoder.steps or classifier.normalized is not False:
        raise AssertionError("README classifier must train on raw means")
    if classifier._pipeline("cpu", with_vectors=False).folded is not True:
        raise AssertionError("README classifier does not use folding")
    repo = encoder._content.source.repo
    normalized = StaticModel.from_pretrained(repo, normalize=True).encoder
    normal_head = type(head)().fit(
        normalized.encode(namespace["texts"], workers=1),
        np.resize(head.classes_, len(namespace["texts"])),
    )
    explicit = Classifier.from_sklearn(normalized, normal_head)
    if explicit.normalized is not True or explicit._pipeline("cpu", with_vectors=False).folded:
        raise AssertionError("Normalized heads must use the explicit path")
    explicit.predict(["I need a new password"], workers=1)
    try:
        Encoder.from_pretrained(repo, normalize=False)
    except ValueError as exc:
        if "holds its own rules" not in str(exc):
            raise
    else:
        raise AssertionError("Generic POTION loading unexpectedly accepted normalize=False")


def verify_classifier(namespace):

    verify_normalization(namespace)
    classifier = namespace["classifier"]
    texts = ["I need a new password"]
    expected = namespace["labels"].tolist()
    classifier.save("standalone.safetensors", standalone=True)
    for path in (Path("router.safetensors"), Path("standalone.safetensors")):
        restored = Classifier.load(path)
        np.testing.assert_array_equal(restored.predict(texts), namespace["labels"])
        np.testing.assert_array_equal(
            restored.predict_proba(texts).view(np.uint32),
            classifier.predict_proba(texts).view(np.uint32),
        )
        verify_reload_without_sklearn(path, expected)
    return {
        "classifier_normalized": classifier.normalized,
        "folded": True,
        "reload_without_sklearn": True,
    }


def execute_example(readme, line, code):

    namespace = {"__name__": "__main__"}
    exec(compile("\n" * (line - 1) + code, str(readme), "exec"), namespace)
    result = verify_vectors(namespace)
    if "classifier" in namespace:
        result.update(verify_classifier(namespace))
    if "compact" in namespace:
        compact = Encoder.load("compact.safetensors")
        if compact.word_vectors.dtype != "int8":
            raise AssertionError("Quantization example must save an int8 table")
        compact.encode(["the cat sleeps"], workers=1)
        result["saved_table_dtype"] = compact.word_vectors.dtype
    encoder = (
        namespace.get("encoder")
        or getattr(namespace.get("model"), "encoder", None)
        or namespace.get("compact")
    )
    if encoder is not None:
        result.update(encoder_record(encoder))
    del namespace
    gc.collect()
    return result


def run_examples(readme):
    readme = readme.resolve()
    results = []
    with tempfile.TemporaryDirectory(prefix="fse-readme-") as folder:
        for index, (heading, line, code) in enumerate(snippets(readme), 1):
            work = Path(folder) / str(index)
            work.mkdir()
            print(f"Running README Python example {index}: {heading} (line {line})", flush=True)
            with working_directory(work):
                result = execute_example(readme, line, code)
            results.append({"heading": heading, "line": line, "status": "passed", **result})
    return {"status": "passed", "python": sys.version, "examples": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = run_examples(args.readme)
    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
