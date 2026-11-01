//! Native Rust worker. Corpus IO and output serialization stay outside encode timing.
use model2vec_rs::model::StaticModel;
use serde_json::json;
use std::{env, fs, io::{BufRead, BufReader, BufWriter, Write}, time::Instant};

fn pipeline(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.len() != 6 || !["full", "none"].contains(&args[5].as_str()) {
        return Err("expected: --pipeline MODEL_FOLDER TEXT_FILE OUTPUT_NPY full|none".into());
    }
    let load = Instant::now();
    let model = StaticModel::from_pretrained(&args[2], None, None, None)?;
    let load_s = load.elapsed().as_secs_f64();
    let start = Instant::now();
    let parent = std::path::Path::new(&args[4]).parent().ok_or("output has no parent")?;
    fs::create_dir(parent)?;
    let sentences: Vec<String> = BufReader::new(fs::File::open(&args[3])?).lines().collect::<Result<_, _>>()?;
    let vectors = model.encode(&sentences);
    // Library API plus a standard NumPy v1 writer. The backend has no persisted-vector API.
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}", sentences.len(), model.dim());
    let padding = (64 - (10 + header.len() + 1) % 64) % 64;
    header.push_str(&" ".repeat(padding));
    header.push('\n');
    let mut file = BufWriter::new(fs::File::create_new(&args[4])?);
    file.write_all(b"\x93NUMPY\x01\x00")?;
    file.write_all(&(header.len() as u16).to_le_bytes())?;
    file.write_all(header.as_bytes())?;
    for row in &vectors {
        for value in row { file.write_all(&value.to_le_bytes())?; }
    }
    file.flush()?;
    if args[5] == "full" {
        file.get_ref().sync_all()?;
        fs::File::open(parent)?.sync_all()?;
        fs::File::open(parent.parent().ok_or("output has no grandparent")?)?.sync_all()?;
    }
    let seconds = start.elapsed().as_secs_f64();
    println!("{}", json!({"load_s": load_s, "p50_s": seconds, "warm_seconds": [seconds],
        "texts": sentences.len(), "shape": [sentences.len(), model.dim()],
        "time_until_durable_s": if args[5] == "full" { Some(seconds) } else { None },
        "writer": "model2vec-rs library + BufWriter NumPy v1 float32, flush, optional file and parent fsync"}));
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.get(1).map(String::as_str) == Some("--pipeline") { return pipeline(&args); }
    if args.len() != 5 {
        return Err("expected: MODEL_FOLDER CORPUS_JSON PASSES VECTORS_F32".into());
    }
    let sentences: Vec<String> = serde_json::from_slice(&fs::read(&args[2])?)?;
    let passes: usize = args[3].parse()?;
    if passes == 0 || sentences.is_empty() {
        return Err("passes and corpus must be nonempty".into());
    }
    let start = Instant::now();
    let model = StaticModel::from_pretrained(&args[1], None, None, None)?;
    let load_s = start.elapsed().as_secs_f64();
    let mut seconds = Vec::new();
    let mut vectors = Vec::new();
    for _ in 0..=passes {
        let start = Instant::now();
        let out = model.encode(&sentences);
        seconds.push(start.elapsed().as_secs_f64());
        vectors = out;
    }
    let mut file = fs::File::create_new(&args[4])?;
    for row in &vectors {
        for value in row {
            file.write_all(&value.to_le_bytes())?;
        }
    }
    println!("{}", json!({"load_s": load_s, "first_encode_s": seconds[0],
        "warm_seconds": &seconds[1..], "shape": [sentences.len(), model.dim()]}));
    Ok(())
}
