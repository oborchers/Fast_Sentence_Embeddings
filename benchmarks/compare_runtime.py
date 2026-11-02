"""Identify the executing backend, including installed code and native binaries."""

from __future__ import annotations

from importlib.metadata import distribution
from importlib.util import find_spec
from pathlib import Path

from compare_artifacts import file_digest, runtime_identity

PACKAGES = {
    "fse": ("fse", "fse"),
    "model2vec": ("model2vec", "model2vec"),
    "st": ("sentence_transformers", "sentence-transformers"),
    "fastembed": ("fastembed", "fastembed"),
    "faiss": ("faiss", "faiss-cpu"),
    "hnswlib": ("hnswlib", "hnswlib"),
    "fasttext": ("fasttext", "fasttext-numpy2"),
    "tfidf": ("sklearn", "scikit-learn"),
}


def stable_file(path: Path) -> bool:
    return (
        path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo")
    )


def recorded_files(package) -> dict[str, Path]:
    if package.files is None:
        raise ValueError("backend distribution has no installed-file manifest")
    paths = {
        "distribution/" + str(p): Path(package.locate_file(p)).resolve() for p in package.files
    }
    for path in paths.values():
        if (
            path.suffix not in (".pyc", ".pyo")
            and "__pycache__" not in path.parts
            and not path.is_file()
        ):
            raise ValueError("backend distribution has a missing installed file")
    return paths


def installed_files(package, spec) -> dict[str, Path]:
    paths = recorded_files(package)
    root = Path(spec.origin).resolve()
    if spec.submodule_search_locations is not None:
        root = root.parent
        paths.update({"module/" + str(p.relative_to(root)): p.resolve() for p in root.rglob("*")})
    else:
        paths["module/" + root.name] = root
    result, seen = {}, set()
    for name, path in sorted(
        paths.items(), key=lambda item: (not item[0].startswith("module/"), item[0])
    ):
        if stable_file(path) and path not in seen:
            result[name] = path
            seen.add(path)
    return result


def package_identity(module_name: str, distribution_name: str) -> dict:
    from fse._container._hash import Hasher

    spec = find_spec(module_name)
    if spec is None or spec.origin is None:
        raise ValueError(f"backend module {module_name} has no installed origin")
    package = distribution(distribution_name)
    paths = installed_files(package, spec)
    hasher = Hasher()
    for name, path in sorted(paths.items()):
        for value in (name.encode("utf-8"), file_digest(path).encode("ascii")):
            hasher.update(len(value).to_bytes(8, "little"))
            hasher.update(value)
    return {
        "runtime_package": distribution_name,
        "runtime_package_version": package.version,
        "package_location": str(Path(spec.origin).resolve()),
        "runtime_fingerprint": hasher.checksum(),
        "runtime_fingerprint_scope": "installed distribution files and module tree, excluding Python caches",
        "runtime_file_count": len(paths),
    }


def backend_identity(kind: str) -> dict:
    if kind == "rust":
        import tomllib

        from compare_rust_build import ROOT, verified_binary

        binary, build = verified_binary()
        packages = tomllib.loads((ROOT / "Cargo.lock").read_text())["package"]
        package = next(p for p in packages if p["name"] == "model2vec-rs")
        return {
            "runtime_package": "model2vec-rs",
            "runtime_package_version": package["version"],
            "package_location": str(binary.resolve()),
            "runtime_fingerprint": build["binary_digest"],
            "runtime_fingerprint_scope": "locked Rust worker binary, including linked backend code",
            "runtime_file_count": 1,
            "rust_worker_build": build,
        }
    module, name = PACKAGES[kind]
    if kind == "fse":
        spec = find_spec(module)
        if spec is None or spec.origin is None:
            raise ValueError("fse has no installed origin")
        return {
            "runtime_package": name,
            "runtime_package_version": distribution(name).version,
            "package_location": str(Path(spec.origin).resolve()),
            "runtime_fingerprint": runtime_identity(),
            "runtime_fingerprint_scope": "fse Python sources and native extension (existing scheme)",
        }
    return package_identity(module, name)


def expected_package(kind: str) -> str:
    return "model2vec-rs" if kind == "rust" else PACKAGES[kind][1]


def verify_backend_identity(kind: str, before: dict) -> None:
    if before.get("runtime_package") != expected_package(kind):
        raise ValueError("counterpart row cannot carry another backend's runtime identity")
    after = backend_identity(kind)
    keys = ("runtime_package", "runtime_package_version", "package_location", "runtime_fingerprint")
    if any(before.get(key) != after.get(key) for key in keys):
        raise ValueError("backend installed code changed during the measured operation")
