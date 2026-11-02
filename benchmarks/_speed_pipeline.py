"""Private backend adapters for the fixed library-only comparison."""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
import tomllib
from importlib.metadata import distributions
from pathlib import Path

from _measurement import atomic, package_identity, sha256, telemetry
from _measurement_config import PINS
from _speed_cells import configure
from _speed_write import headline_child

RUST_SOURCE = r"""
use model2vec_rs::model::StaticModel;
use serde_json::json;
use std::{env,fs,io::{BufRead,BufReader,Write},time::Instant};
fn main()->Result<(),Box<dyn std::error::Error>> {
 let a:Vec<String>=env::args().collect();let workers:usize=a[5].parse()?;
 rayon::ThreadPoolBuilder::new().num_threads(workers).build_global()?;
 let texts:Vec<String>=BufReader::new(fs::File::open(&a[2])?).lines().collect::<Result<_,_>>()?;
 let model=StaticModel::from_pretrained(&a[1],None,None,None)?;
 let indices:Vec<usize>=serde_json::from_slice(&fs::read(&a[6])?)?;
 let warmed=model.encode_flat(&texts);drop(warmed);
 let start=Instant::now();let output=model.encode_flat(&texts);let seconds=start.elapsed().as_secs_f64();
 assert_eq!(output.len(),texts.len()*model.dim());assert!(cfg!(target_endian="little"));
 let mut file=fs::OpenOptions::new().write(true).create_new(true).open(&a[4])?;
 for row in &indices {let first=row*model.dim();let slice=&output[first..first+model.dim()];
  let bytes=unsafe {std::slice::from_raw_parts(slice.as_ptr() as *const u8,slice.len()*4)};file.write_all(bytes)?;}
 file.sync_all()?;
 println!("{}",json!({"seconds":seconds,"actual_threads":rayon::current_num_threads(),"shape":[indices.len(),model.dim()],"full_shape":[texts.len(),model.dim()]}));Ok(())
}
"""


def fse_encoder(job):
    from fse import Encoder

    model = Encoder.load(job["artifact"], mmap=True)
    if model.word_vectors.dtype != job["dtype"]:
        raise ValueError("finished table dtype differs from the requested cell")
    return lambda texts: model.encode(texts, workers=job["workers"])


def model2vec_encoder(job):
    from model2vec import StaticModel

    model = StaticModel.from_pretrained(job["snapshot"], normalize=True)
    return lambda texts: model.encode(texts, use_multiprocessing=job["workers"] > 1)


def fastembed_encoder(job):
    import numpy as np
    from fastembed import TextEmbedding

    model = TextEmbedding(
        model_name=job["model"], specific_model_path=job["snapshot"], local_files_only=True
    )
    return lambda texts: np.stack(
        list(model.embed(texts, parallel=job["workers"] if job["workers"] > 1 else None))
    )


def interpreter(backend, reference_python):
    return reference_python if ADAPTERS[backend]["environment"] == "reference" else sys.executable


def build_rust(folder, lock):
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "src").mkdir()
    (folder / "src/main.rs").write_text(RUST_SOURCE)
    (folder / "Cargo.lock").write_bytes(lock.read_bytes())
    (folder / "Cargo.toml").write_text(
        """[package]
name="fse-model2vec-rs-compare"
version="0.1.0"
edition="2024"
[dependencies]
model2vec-rs={git="https://github.com/MinishLab/model2vec-rs",rev="@REVISION@",default-features=false,features=["onig","local-only"]}
serde_json="=1.0.145"
rayon="=1.12.0"
[profile.release]
codegen-units=1
lto="thin"
""".replace("@REVISION@", PINS["rust_revision"])
    )
    subprocess.run(
        [
            str(Path.home() / ".cargo/bin/cargo"),
            "build",
            "--release",
            "--locked",
            "--manifest-path",
            str(folder / "Cargo.toml"),
        ],
        check=True,
    )
    binary = folder / "target/release/fse-model2vec-rs-compare"
    receipt = dict(
        package_version=next(
            p["version"]
            for p in tomllib.loads((folder / "Cargo.lock").read_text())["package"]
            if p["name"] == "model2vec-rs"
        ),
        binary=str(binary),
        binary_sha256=sha256(binary),
        source_sha256=sha256(folder / "src/main.rs"),
        lock_sha256=sha256(lock),
        revision=PINS["rust_revision"],
        compiler=subprocess.check_output(
            [str(Path.home() / ".cargo/bin/rustc"), "--version"], text=True
        ).strip(),
    )
    atomic(folder / "build.json", receipt)
    return receipt


def rust_child(job):
    import resource

    configure(job["workers"])
    binary = Path(job["binary"])
    if (
        sha256(binary) != job["binary_sha256"]
        or sha256(Path(job["input_file"])) != job["input_sha256"]
    ):
        raise ValueError("Rust binary or input differs from its frozen record")
    before = telemetry()
    record = json.loads(
        subprocess.check_output(
            [
                str(binary),
                job["snapshot"],
                job["input_file"],
                "1",
                job["vectors_file"],
                str(job["workers"]),
                job["sample_file"],
            ],
            text=True,
        )
    )
    if record["actual_threads"] != job["workers"] or sha256(binary) != job["binary_sha256"]:
        raise ValueError("Rust worker count or code changed")
    record.update(
        label=job["label"],
        round=job["round"],
        identity=dict(
            package="model2vec-rs",
            version=job["package_version"],
            revision=job["rust_revision"],
            path=str(binary.resolve()),
            compiler=job["compiler"],
            source_sha256=job["source_sha256"],
            lock_sha256=job["lock_sha256"],
            binary_sha256=job["binary_sha256"],
        ),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        output_checked=False,
        vector_format="raw",
        telemetry_before=before,
        telemetry_after=telemetry(),
    )
    atomic(Path(job["receipt"]), record)


def prepare_models(root, contract, reuse=None):
    from huggingface_hub import snapshot_download

    from fse import Encoder, StaticModel

    result = {}
    needed = {c["model"] for c in contract["cells"]}
    for repo, revision in contract["models"].items():
        if repo not in needed:
            continue
        snapshot = snapshot_download(
            repo,
            revision=revision,
            cache_dir=(reuse or root) / "hub",
            local_files_only=reuse is not None,
            allow_patterns=["*.json", "*.safetensors", "*/*.json", "*/*.safetensors"],
        )
        encoder = (
            Encoder.from_pretrained(snapshot, normalize=True, truncate=False)
            if repo.startswith("sentence-transformers/")
            else StaticModel.from_pretrained(snapshot, normalize=True).encoder
        )
        artifacts = {}
        dtypes = {
            "float32",
            *(c["dtype"] for c in contract["cells"] if c["model"] == repo and c["kind"] == "dtype"),
        }
        for dtype in sorted(dtypes):
            artifact = root / (repo.rsplit("/", 1)[-1] + "-" + dtype + ".safetensors")
            current = encoder if dtype == "float32" else encoder.quantize(dtype)
            current.save(artifact)
            artifacts[dtype] = str(artifact)
        result[repo] = {
            "snapshot": snapshot,
            "artifact": artifacts["float32"],
            "artifacts": artifacts,
            "revision": revision,
        }
    return result


def measure_child(job):
    if job.get("kind") == "write":
        return headline_child(job)
    if job["backend"] == "rust":
        return rust_child(job)
    adapter = ADAPTERS[job["backend"]]
    configure(job["workers"], native_threads=adapter["native_threads"])
    import resource

    import numpy as np

    package = adapter["package"]
    identity = package_identity(package)
    with Path(job["input_file"]).open() as stream:
        texts = [line.rstrip("\n") for line in stream]
    if len(texts) != job["texts"] or sha256(Path(job["input_file"])) != job["input_sha256"]:
        raise ValueError("input changed")
    indices = np.asarray(json.loads(Path(job["sample_file"]).read_text()), dtype=np.int64)
    encode = adapter["factory"](job)
    try:
        warm = encode(texts)
        if warm.shape != (len(texts), job["dimensions"]):
            raise ValueError("warm output shape changed")
        del warm
        gc.collect()
        before = telemetry()
        started = time.perf_counter()
        vectors = encode(texts)
        seconds = time.perf_counter() - started
        after = telemetry()
        if vectors.dtype != np.float32 or vectors.shape != (len(texts), job["dimensions"]):
            raise ValueError("backend output contract changed")
        sample = np.ascontiguousarray(vectors[indices])
        with Path(job["vectors_file"]).open("xb") as stream:
            np.save(stream, sample, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        if package_identity(package) != identity:
            raise ValueError("executing backend changed")
        atomic(
            Path(job["receipt"]),
            {
                "identity": identity,
                "versions": {d.metadata["Name"]: d.version for d in distributions()},
                "seconds": seconds,
                "shape": list(sample.shape),
                "full_shape": list(vectors.shape),
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "telemetry_before": before,
                "telemetry_after": after,
                "affinity": sorted(os.sched_getaffinity(0)),
                "workers": job["workers"],
                "worker_path": adapter["factory"].__name__,
            },
        )
    finally:
        if hasattr(encode, "close"):
            encode.close()


def st_pool_worker(*args):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    SentenceTransformer._multi_process_worker(*args)


def st_encoder(job):
    import torch
    from sentence_transformers import SentenceTransformer

    workers = job["workers"]
    if workers > 1:
        st_file_limits()
    torch.set_num_threads(1 if workers > 1 else workers)
    model = SentenceTransformer(job["snapshot"], device="cpu", local_files_only=True)
    if workers == 1:
        return lambda texts: model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
    torch.set_num_interop_threads(1)
    original = SentenceTransformer.__dict__.get("_multi_process_worker")
    SentenceTransformer._multi_process_worker = staticmethod(st_pool_worker)
    try:
        pool = model.start_multi_process_pool(target_devices=["cpu"] * workers)
    finally:
        if original is None:
            del SentenceTransformer._multi_process_worker
        else:
            SentenceTransformer._multi_process_worker = original

    def encode(texts):
        return model.encode(
            texts,
            pool=pool,
            batch_size=1024,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    encode.close = lambda: model.stop_multi_process_pool(pool)
    return encode


def st_file_limits():
    """The documented CPU process pool transports many shared tensor descriptors."""
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = 524288
    if hard != resource.RLIM_INFINITY and hard < target:
        raise RuntimeError(f"ST pool needs nofile hard limit >= {target}; got {hard}")
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, target), hard))
    if __import__("shutil").disk_usage("/dev/shm").free < 512 * 1024**2:
        raise RuntimeError("ST pool needs at least 512 MiB free shared memory")


# A new Python library adds one factory and one registry entry, not another runner.
ADAPTERS = {
    "fse": {
        "factory": fse_encoder,
        "package": "fse",
        "environment": "main",
        "native_threads": None,
    },
    "model2vec": {
        "factory": model2vec_encoder,
        "package": "model2vec",
        "environment": "main",
        "native_threads": None,
    },
    "fastembed": {
        "factory": fastembed_encoder,
        "package": "fastembed",
        "environment": "reference",
        "native_threads": 1,
    },
    "st": {
        "factory": st_encoder,
        "package": "sentence-transformers",
        "environment": "main",
        "native_threads": 1,
    },
    "rust": {
        "factory": rust_child,
        "package": "model2vec-rs",
        "environment": "main",
        "native_threads": None,
    },
}
