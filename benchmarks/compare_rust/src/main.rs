//! Native Rust worker. Corpus IO and output serialization stay outside encode timing.
use model2vec_rs::model::StaticModel;
use serde_json::json;
use std::{env, fs, io::Write, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
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
