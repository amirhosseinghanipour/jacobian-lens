//! Content identities for reproducible model, tokenizer and corpus artifacts.
use crate::Result;
use sha2::{Digest, Sha256};
use std::{io::Read, path::Path};
pub fn sha256_file(path: impl AsRef<Path>) -> Result<String> {
    let mut input = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut block = vec![0u8; 1024 * 1024];
    loop {
        let n = input.read(&mut block)?;
        if n == 0 {
            break;
        }
        hash.update(&block[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
/// Hash a local Hugging Face checkpoint, including every indexed weight shard
/// and tokenizer/configuration file. Paths in shard indices must stay inside
/// the supplied directory. The runtime remains responsible for loading weights.
pub fn model_identity(directory: impl AsRef<Path>) -> Result<String> {
    use crate::LensError;
    use std::collections::BTreeSet;
    let dir = directory.as_ref();
    let mut names = BTreeSet::from(["config.json".to_string()]);
    let mut found = false;
    for (single, index) in [
        ("model.safetensors", "model.safetensors.index.json"),
        ("pytorch_model.bin", "pytorch_model.bin.index.json"),
    ] {
        if dir.join(index).exists() {
            names.insert(index.into());
            let value: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join(index))?)
                .map_err(|e| LensError::Invalid(e.to_string()))?;
            let map = value["weight_map"]
                .as_object()
                .ok_or_else(|| LensError::Invalid("missing shard weight_map".into()))?;
            if map.is_empty() {
                return Err(LensError::Invalid("empty weight_map".into()));
            }
            for shard in map.values() {
                let shard = shard
                    .as_str()
                    .ok_or_else(|| LensError::Invalid("invalid shard filename".into()))?;
                if shard.is_empty()
                    || Path::new(shard)
                        .components()
                        .any(|c| !matches!(c, std::path::Component::Normal(_)))
                {
                    return Err(LensError::Invalid(
                        "shard filename must be a relative path without traversal".into(),
                    ));
                }
                names.insert(shard.into());
            }
            found = true;
            break;
        } else if dir.join(single).exists() {
            names.insert(single.into());
            found = true;
            break;
        }
    }
    if !found {
        return Err(LensError::Invalid(
            "no supported local weight file/index found".into(),
        ));
    }
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "generation_config.json",
    ] {
        if dir.join(name).exists() {
            names.insert(name.into());
        }
    }
    let mut hash = Sha256::new();
    hash.update(b"jlens-model-identity-v1\0");
    for name in names {
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update(sha256_file(dir.join(name))?.as_bytes());
    }
    Ok(format!("{:x}", hash.finalize()))
}
