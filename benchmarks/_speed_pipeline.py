"""Private backend adapters for the fixed library-only comparison."""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
import tomllib
from functools import partial
from importlib.metadata import distributions
from pathlib import Path

from _speed_cells import admit, atomic, configure, package_identity, sha256, telemetry


def setup_command(*arguments, allocated_at, contract):
    if not admit(contract, allocated_at, 60):
        raise RuntimeError("setup exhausted allocation budget")
    subprocess.run([str(x) for x in arguments], check=True)


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


def setup(root, contract, allocated_at):
    """Install the two frozen CPU environments without starting measurement."""
    root.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    tools, source = root / "tools", root / "fse-source"
    tools.mkdir()
    execute = partial(setup_command, allocated_at=allocated_at, contract=contract)
    execute("apt-get", "update")
    packages = ("build-essential", "pkg-config", "libssl-dev", "curl", "git", "ca-certificates")
    execute("apt-get", "install", "-y", *packages)
    execute(sys.executable, "-m", "pip", "install", "--prefix", tools, "uv==0.13.0")
    uv = tools / "bin/uv"
    for name, lock in (("main", "speed-main.lock.txt"), ("reference", "speed-reference.lock.txt")):
        destination = root / name
        execute(uv, "venv", "--python", sys.executable, destination)
        execute(
            uv,
            "pip",
            "install",
            "--python",
            destination / "bin/python",
            "--no-deps",
            "--require-hashes",
            "-r",
            here / lock,
            "--extra-index-url",
            "https://download.pytorch.org/whl/cpu",
            "--index-strategy=unsafe-best-match",
        )
        execute(uv, "pip", "check", "--python", destination / "bin/python")
    execute("git", "clone", "https://github.com/oborchers/Fast_Sentence_Embeddings.git", source)
    execute("git", "-C", source, "checkout", "--detach", contract["fse_revision"])
    flags = ("--no-deps", "--no-build-isolation")
    execute(uv, "pip", "install", "--python", root / "main/bin/python", *flags, source)
    execute(uv, "pip", "check", "--python", root / "main/bin/python")
    installer = tools / "rustup-init"
    execute(
        "curl",
        "--fail",
        "--location",
        "--output",
        installer,
        "https://static.rust-lang.org/rustup/archive/1.28.2/x86_64-unknown-linux-gnu/rustup-init",
    )
    installer.chmod(0o700)
    execute(
        installer, "-y", "--no-modify-path", "--profile", "minimal", "--default-toolchain", "1.98.0"
    )
    atomic(
        root / "setup.json",
        dict(
            rustup_installer_sha256=sha256(installer),
            locks={
                name: sha256(here / name)
                for name in ("speed-main.lock.txt", "speed-reference.lock.txt", "speed-rust.lock")
            },
            fse_revision=contract["fse_revision"],
            python=sys.version,
            base_image=contract["machine"]["image"],
            system_compiler=subprocess.check_output(["cc", "--version"], text=True).splitlines()[0],
        ),
    )


def encoder(job):
    kind, workers = job["backend"], job["workers"]
    if kind == "fse":
        from fse import Encoder

        model = Encoder.load(job["artifact"], mmap=True)
        if model.word_vectors.dtype != job["dtype"]:
            raise ValueError("finished table dtype differs from the requested cell")
        return lambda texts: model.encode(texts, workers=workers)
    if kind == "model2vec":
        from model2vec import StaticModel

        model = StaticModel.from_pretrained(job["snapshot"], normalize=True)
        return lambda texts: model.encode(texts, use_multiprocessing=workers > 1)
    if kind == "st":
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(workers)
        model = SentenceTransformer(job["snapshot"], device="cpu", local_files_only=True)
        return lambda texts: model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
    if kind == "fastembed":
        import numpy as np
        from fastembed import TextEmbedding

        model = TextEmbedding(
            model_name=job["model"], specific_model_path=job["snapshot"], local_files_only=True
        )
        return lambda texts: np.stack(
            list(model.embed(texts, parallel=workers if workers > 1 else None))
        )
    raise ValueError(f"unsupported Python backend: {kind}")


def build_rust(folder, lock):
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "src").mkdir()
    (folder / "src/main.rs").write_text(RUST_SOURCE)
    (folder / "Cargo.lock").write_bytes(lock.read_bytes())
    (folder / "Cargo.toml").write_text("""[package]
name="fse-model2vec-rs-compare"
version="0.1.0"
edition="2024"
[dependencies]
model2vec-rs={git="https://github.com/MinishLab/model2vec-rs",rev="19f58bef6ce145f2bc08de7016498f28d521feea",default-features=false,features=["onig","local-only"]}
serde_json="=1.0.145"
rayon="=1.12.0"
[profile.release]
codegen-units=1
lto="thin"
""")
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
        revision="19f58bef6ce145f2bc08de7016498f28d521feea",
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
        artifact = root / (repo.rsplit("/", 1)[-1] + ".safetensors")
        encoder.save(artifact)
        result[repo] = {"snapshot": snapshot, "artifact": str(artifact), "revision": revision}
    return result


def measure_child(job):
    if job.get("kind") == "write":
        return headline_child(job)
    if job["backend"] == "rust":
        return rust_child(job)
    configure(job["workers"], native_threads=1 if job["backend"] == "fastembed" else None)
    import resource

    import numpy as np

    package = {
        "fse": "fse",
        "model2vec": "model2vec",
        "st": "sentence-transformers",
        "fastembed": "fastembed",
    }[job["backend"]]
    identity = package_identity(package)
    with Path(job["input_file"]).open() as stream:
        texts = [line.rstrip("\n") for line in stream]
    if len(texts) != job["texts"] or sha256(Path(job["input_file"])) != job["input_sha256"]:
        raise ValueError("input changed")
    indices = np.asarray(json.loads(Path(job["sample_file"]).read_text()), dtype=np.int64)
    encode = encoder(job)
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
        },
    )


def headline_tokens(snapshot, source):
    import numpy as np

    from fse import StaticModel

    model = StaticModel.from_pretrained(snapshot)
    if model.max_length != 512:
        raise ValueError("headline requires the pinned 512-token policy")
    counts = np.empty(len(source), dtype=np.int64)
    for start in range(0, len(source), 2048):
        batch = [
            text[: model.max_length * model.median_token_length]
            for text in source[start : start + 2048]
        ]
        rows = model.tokenizer.encode_batch(batch, add_special_tokens=False)
        counts[start : start + len(rows)] = [len(row.ids[: model.max_length]) for row in rows]
    return counts


def headline_child(job):
    import resource

    import numpy as np

    from fse import Encoder, ShardSink, VectorStore

    configure(job["workers"])
    model = Encoder.load(job["artifact"], mmap=True)
    path = Path(job["vectors_file"])
    if path.exists() or path.is_symlink():
        raise ValueError("headline output must be fresh")
    if sha256(Path(job["input_file"])) != job["input_sha256"]:
        raise ValueError("headline input changed")
    identity = package_identity("fse")
    before = telemetry()
    disk_before = __import__("shutil").disk_usage(path.parent)._asdict()
    started = time.perf_counter()
    sink = ShardSink(path, model, dtype="int8", durability="none")
    first_read = time.perf_counter()
    with sink:
        model.encode(Path(job["input_file"]), workers=job["workers"], sink=sink)
    seconds = time.perf_counter() - first_read
    including_open = time.perf_counter() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    after = telemetry()
    if sink.published_rows != job["texts"]:
        raise ValueError("headline row count differs")
    files, synced_at, sync_telemetry = sync_headline(path)
    durable_seconds = synced_at - first_read
    indices = json.loads(Path(job["sample_file"]).read_text())
    sample_path = Path(job["sample_texts_file"])
    if sha256(sample_path) != job["sample_texts_sha256"]:
        raise ValueError("headline sample texts changed")
    texts = __import__("json").loads(sample_path.read_text())
    reference = model.encode(texts, dtype="int8", workers=job["workers"])
    with VectorStore.open(path, encoder=model) as store:
        if len(store) != job["texts"]:
            raise ValueError("reopened headline row count differs")
        store.verify()
        codes, scales = stored_codes(store, indices)
    if not np.array_equal(reference.codes.view(np.uint8), codes.view(np.uint8)):
        raise ValueError("headline int8 sample codes differ")
    if not np.array_equal(reference.scales.view(np.uint8), scales.view(np.uint8)):
        raise ValueError("headline int8 sample scales differ")
    if package_identity("fse") != identity:
        raise ValueError("executing headline backend changed")
    atomic(
        Path(job["receipt"]),
        dict(
            seconds=seconds,
            affinity=sorted(os.sched_getaffinity(0)),
            requested_worker_limit=job["workers"],
            including_sink_open_seconds=including_open,
            time_until_durable_seconds=durable_seconds,
            external_fsync=True,
            durability="none",
            published_rows=sink.published_rows,
            durable_rows=sink.durable_rows,
            output_dtype="int8",
            table_dtype=str(model.word_vectors.dtype),
            sample_checked_rows=len(indices),
            sample_bitwise_equal=True,
            peak_rss_bytes=peak,
            identity=identity,
            files=files,
            bytes_written=sum(row["bytes"] for row in files),
            telemetry_before=before,
            telemetry_after=after,
            telemetry_after_sync=sync_telemetry,
            disk_before=disk_before,
            disk_with_output=__import__("shutil").disk_usage(path.parent)._asdict(),
        ),
    )


def sync_headline(path):
    files = []
    for p in sorted(path.rglob("*")):
        if p.is_symlink():
            raise ValueError("unexpected symlink in owned headline output")
        if p.is_file():
            with p.open("rb") as stream:
                os.fsync(stream.fileno())
            files.append(p)
    for p in [*sorted((p for p in path.rglob("*") if p.is_dir()), reverse=True), path, path.parent]:
        descriptor = os.open(p, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    synced_at, sync_telemetry = time.perf_counter(), telemetry()
    inventory = [
        dict(path=str(p.relative_to(path)), bytes=p.stat().st_size, sha256=sha256(p)) for p in files
    ]
    return inventory, synced_at, sync_telemetry


def stored_codes(store, indices):
    import numpy as np

    indices = np.asarray(indices, dtype=np.int64)
    codes = np.empty((len(indices), store.dim), dtype=np.int8)
    scales = np.empty(len(indices), dtype=np.float32)
    for shard, tensors in store._shards:
        mask = (indices >= shard.first_row) & (indices < shard.first_row + shard.rows)
        rows = indices[mask] - shard.first_row
        codes[mask] = tensors["vectors"][rows]
        scales[mask] = tensors["scales"][rows]
    return codes, scales
