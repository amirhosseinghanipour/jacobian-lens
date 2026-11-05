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
/// Identity binds the exact Qwen weights, architecture config and tokenizer.
pub fn qwen_identity(directory: impl AsRef<Path>) -> Result<String> {
    let mut hash = Sha256::new();
    for name in ["model.safetensors", "config.json", "tokenizer.json"] {
        hash.update(name.as_bytes());
        hash.update(sha256_file(directory.as_ref().join(name))?.as_bytes());
    }
    Ok(format!("{:x}", hash.finalize()))
}
