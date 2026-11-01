"""Private inference and input primitives for the single CPU comparison."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from _speed_cells import atomic, configure, package_identity, sha256


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
        return lambda texts: np.stack(list(model.embed(texts, parallel=None)))
    raise ValueError(f"unsupported Python backend: {kind}")


def capture_output(job, vectors, count, first):
    import numpy as np

    dtype, ndim = (np.int64, 1) if job.get("kind") == "classify" else (np.float32, 2)
    if vectors.dtype != dtype or vectors.ndim != ndim or len(vectors) != count:
        raise ValueError("backend returned a different output contract")
    if not vectors.flags.c_contiguous:
        raise ValueError("backend output must be a contiguous array")
    digest = hashlib.sha256(memoryview(vectors).cast("B")).hexdigest()
    if first:
        mapped = np.lib.format.open_memmap(
            job["vectors_file"], mode="w+", dtype=dtype, shape=vectors.shape
        )
        for start in range(0, len(vectors), 4096):
            mapped[start : start + 4096] = vectors[start : start + 4096]
        mapped.flush()
        del mapped
    return digest


def measure_child(job):
    if job["backend"] == "rust":
        return rust_child(job)
    configure(job["workers"])
    import resource

    package = {
        "fse": "fse",
        "model2vec": "model2vec",
        "st": "sentence-transformers",
        "fastembed": "fastembed",
        "folded": "fse",
        "unfolded": "fse",
        "model2vec-linear": "model2vec",
    }[job["backend"]]
    identity = package_identity(package)
    with Path(job["input_file"]).open() as stream:
        texts = [line.rstrip("\n") for line in stream]
    if len(texts) != job["texts"] or sha256(Path(job["input_file"])) != job["input_sha256"]:
        raise ValueError("input row count or digest changed")
    started = time.perf_counter()
    encode = class_encoder(job) if job.get("kind") == "classify" else encoder(job)
    load_seconds = time.perf_counter() - started
    durations, outputs = [], []
    for _ in range(1 + job.get("warm_passes", 1)):
        started = time.perf_counter()
        vectors = encode(texts)
        durations.append(time.perf_counter() - started)
        digest = capture_output(job, vectors, len(texts), len(outputs) == 0)
        outputs.append(digest)
        del vectors
    if len(set(outputs)) != 1 or package_identity(package) != identity:
        raise ValueError("output or executing backend changed between calls")
    atomic(
        Path(job["receipt"]),
        {
            "label": job["label"],
            "round": job["round"],
            "identity": identity,
            "first_seconds": durations[0],
            "warm_seconds": durations[1:],
            "output_sha256": outputs[0],
            "output_checked": False,
            "load_seconds": load_seconds,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
        },
    )


RUST_SOURCE = r"""
use model2vec_rs::model::StaticModel;
use serde_json::json;
use std::{env,fs,io::{BufRead,BufReader,Write},time::Instant,hash::{Hash,Hasher}};
fn main()->Result<(),Box<dyn std::error::Error>> {
 let a:Vec<String>=env::args().collect();
 if a.get(1).map(String::as_str)==Some("--startup") {
  let start=Instant::now();let n:usize=a[2].parse()?;
  rayon::ThreadPoolBuilder::new().num_threads(n).build_global()?;rayon::broadcast(|_|());
  println!("{}",json!({"pool_seconds":start.elapsed().as_secs_f64(),"threads":rayon::current_num_threads()}));return Ok(());
 }
 let workers:usize=a[5].parse()?;
 let start=Instant::now();
 rayon::ThreadPoolBuilder::new().num_threads(workers).build_global()?;
 let pool=start.elapsed().as_secs_f64();
 let texts:Vec<String>=BufReader::new(fs::File::open(&a[2])?).lines().collect::<Result<_,_>>()?;
 let start=Instant::now();let model=StaticModel::from_pretrained(&a[1],None,None,None)?;
 let load=start.elapsed().as_secs_f64();
 let passes:usize=a[3].parse()?;let mut seconds=Vec::new();let mut hashes=Vec::new();
 for pass in 0..=passes {
  let start=Instant::now();let output=model.encode_flat(&texts);
  seconds.push(start.elapsed().as_secs_f64());
  let mut digest=std::collections::hash_map::DefaultHasher::new();
  for value in &output {value.to_bits().hash(&mut digest);}hashes.push(digest.finish().to_string());
  if pass==0 {
   assert!(cfg!(target_endian="little"));
   let bytes=unsafe {std::slice::from_raw_parts(output.as_ptr() as *const u8,output.len()*4)};
   let mut file=fs::OpenOptions::new().write(true).create_new(true).open(&a[4])?;
   file.write_all(bytes)?;file.sync_all()?;
  }
 }
 println!("{}",json!({"load_seconds":load,"pool_start_seconds":pool,"actual_threads":rayon::current_num_threads(),"first_seconds":seconds[0],"warm_seconds":&seconds[1..],"output_hashes":hashes,"shape":[texts.len(),model.dim()]}));Ok(())
}
"""


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
        ["cargo", "build", "--release", "--locked", "--manifest-path", str(folder / "Cargo.toml")],
        check=True,
    )
    binary = folder / "target/release/fse-model2vec-rs-compare"
    receipt = dict(
        binary=str(binary),
        binary_sha256=sha256(binary),
        source_sha256=sha256(folder / "src/main.rs"),
        lock_sha256=sha256(lock),
        revision="19f58bef6ce145f2bc08de7016498f28d521feea",
        compiler=subprocess.check_output(["rustc", "--version"], text=True).strip(),
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
    record = json.loads(
        subprocess.check_output(
            [
                str(binary),
                job["snapshot"],
                job["input_file"],
                str(job.get("warm_passes", 1)),
                job["vectors_file"],
                str(job["workers"]),
            ],
            text=True,
        )
    )
    if len(set(record.pop("output_hashes"))) != 1 or record["actual_threads"] != job["workers"]:
        raise ValueError("Rust output or worker budget changed")
    if sha256(binary) != job["binary_sha256"]:
        raise ValueError("Rust binary changed during measurement")
    record.update(
        label=job["label"],
        round=job["round"],
        identity=dict(
            package="model2vec-rs",
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
    )
    atomic(Path(job["receipt"]), record)


def class_encoder(job):
    import numpy as np

    from fse import Classifier, Encoder

    head = np.load(job["head_file"])
    weight, bias = head["weight"], head["bias"]
    if job["backend"] == "folded":
        model = Classifier.load(job["folded_file"], mmap=True)
        if not model.standalone or model.normalized:
            raise ValueError("classification requires a folded raw-mean table")
        return lambda texts: model.predict(texts, workers=job["workers"])
    if job["backend"] == "model2vec-linear":
        from model2vec import StaticModel

        model = StaticModel.from_pretrained(job["snapshot"], normalize=False)

        def encode(texts):
            return model.encode(texts, use_multiprocessing=job["workers"] > 1)
    else:
        model = Encoder.load(job["raw_encoder"], mmap=True)
        if model.steps:
            raise ValueError("unfolded arm must use raw unnormalized means")

        def encode(texts):
            return model.encode(texts, workers=job["workers"])

    return lambda texts: np.argmax(encode(texts) @ weight.T + bias, axis=1).astype(np.int64)


def prepare_models(root, contract):
    from huggingface_hub import snapshot_download

    from fse import StaticModel

    models = {}
    for repo, revision in contract["models"].items():
        snapshot = snapshot_download(
            repo,
            revision=revision,
            cache_dir=root / "hub",
            allow_patterns=[
                "model.safetensors",
                "tokenizer.json",
                "config.json",
                "modules.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "*/model.safetensors",
                "*/tokenizer.json",
                "*/config.json",
            ],
        )
        normal = StaticModel.from_pretrained(snapshot, normalize=True).encoder
        item = dict(snapshot=snapshot, revision=revision, artifacts={})
        for dtype in ("float32", "float16", "int8"):
            destination = root / (repo.split("/")[-1] + "-" + dtype + ".safetensors")
            (normal if dtype == "float32" else normal.quantize(dtype)).save(destination)
            item["artifacts"][dtype] = str(destination)
        models[repo] = item
    return models


def prepare_head(root, contract, models):
    import numpy as np
    from datasets import load_dataset
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    from fse import Classifier, StaticModel

    spec = contract["classification"]
    data = load_dataset(spec["repo"], revision=spec["revision"], cache_dir=str(root / "datasets"))

    def subset(split, count):
        rows = data[split]
        a, _, b, _ = train_test_split(
            rows["text"],
            rows["label"],
            train_size=count,
            stratify=rows["label"],
            random_state=spec["seed"],
        )
        return a, b

    texts, labels = subset("train", spec["train_selection"])
    fit, _, target, _ = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=spec["seed"]
    )
    sample, _ = subset("test", spec["evaluation_selection"])
    model = StaticModel.from_pretrained(
        models["minishlab/potion-base-8M"]["snapshot"], normalize=False
    ).encoder
    head = LogisticRegression(
        C=spec["C"], max_iter=spec["max_iter"], random_state=spec["seed"]
    ).fit(model.encode(fit, workers=1), target)
    weight, bias = head.coef_.astype(np.float32), head.intercept_.astype(np.float32)
    folder = root / "head.npz"
    np.savez(folder, weight=weight, bias=bias)
    raw, folded = root / "raw.safetensors", root / "folded.safetensors"
    model.save(raw)
    classifier = Classifier.from_arrays(
        model, weight, bias, labels=np.arange(4, dtype=np.int64), normalized=False
    )
    classifier.save(folded, standalone=True)
    return sample, dict(
        head_file=str(folder),
        raw_encoder=str(raw),
        folded_file=str(folded),
        head_sha256=sha256(folder),
        accuracy="historical recipe only; not measured",
    )


def stream_child(job):
    import resource

    configure(job["workers"])
    from fse import Encoder, ShardSink, VectorStore

    source, output = Path(job["input_file"]), Path(job["vectors_file"])
    if sha256(source) != job["input_sha256"] or output.exists():
        raise ValueError("streaming input changed or output already exists")
    model = Encoder.load(job["artifact"], mmap=True)
    identity = package_identity("fse")
    disk_before = shutil.disk_usage(output.parent)._asdict()
    started = time.perf_counter()
    with ShardSink(output, model, dtype="int8", durability="none") as sink:
        model.encode(source, workers=job["workers"], sink=sink)
    seconds = time.perf_counter() - started
    store = VectorStore.open(output)
    if len(store) != job["texts"]:
        raise ValueError("stream output row count differs")
    files = [
        dict(path=str(p.relative_to(output)), bytes=p.stat().st_size, sha256=sha256(p))
        for p in sorted(output.rglob("*"))
        if p.is_file()
    ]
    if package_identity("fse") != identity:
        raise ValueError("executing backend changed during streaming")
    atomic(
        Path(job["receipt"]),
        dict(
            label=job["label"],
            round=job["round"],
            identity=identity,
            df_before=disk_before,
            df_after=shutil.disk_usage(output.parent)._asdict(),
            first_seconds=seconds,
            warm_seconds=[seconds],
            output_checked=True,
            rows=len(store),
            files=files,
            bytes_written=sum(p["bytes"] for p in files),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            durability="none",
            time_until_durable=None,
        ),
    )
