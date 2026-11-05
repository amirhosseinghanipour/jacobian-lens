//! Optional persistent Python runtime. Requires PyTorch, Transformers, safetensors
//! and the Anthropic `jlens` package in the selected Python environment.
//!
//! Tensor exchange uses private safetensors files, never pickle. Python owns the
//! model and its autograd graph; Rust owns lens aggregation and analysis. Calls
//! are blocking and serialized per worker. Use one worker per device/queue and
//! bounded blocking tasks in async applications. A timeout kills that worker.
use crate::{
    FittingConfig, JacobianEstimator, LensError, LensModel, ModelDimensions, ModelTrace,
    PromptJacobian, ResidualReplacement, Result, Unembed,
};
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap},
    io::{BufRead, BufReader, Write},
    path::Path,
    process::{Child, ChildStdin, Command, Stdio},
    sync::{Mutex, mpsc},
    thread::JoinHandle,
    time::Duration,
};
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfLayout {
    pub path: String,
    pub layers: String,
    pub norm: String,
    pub embed: String,
    pub lm_head: String,
}
/// Model loading is delegated to the installed runtime, not a Rust model list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfOptions {
    pub model: String,
    pub revision: Option<String>,
    /// Optional separate tokenizer directory or Hub identifier.
    pub tokenizer: Option<String>,
    pub device: String,
    pub dtype: String,
    pub local_files_only: bool,
    pub trust_remote_code: bool,
    pub force_bos: bool,
    pub layout: Option<HfLayout>,
    /// Transformers auto class: AutoModelForCausalLM or AutoModelForImageTextToText.
    pub auto_class: String,
    /// Optional `module:function`, called with this options dictionary. Return
    /// a Python jlens LensModel with `unembedding_rows(ids)` for J-space frames.
    /// This supports runtimes and layouts outside Transformers without a port.
    pub factory: Option<String>,
    pub threads: usize,
}
impl HfOptions {
    pub fn local(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            revision: None,
            tokenizer: None,
            device: "cpu".into(),
            dtype: "float32".into(),
            local_files_only: true,
            trust_remote_code: false,
            force_bos: false,
            layout: None,
            auto_class: "AutoModelForCausalLM".into(),
            factory: None,
            threads: 2,
        }
    }
}
struct Worker {
    child: Child,
    input: ChildStdin,
    responses: mpsc::Receiver<std::result::Result<Value, String>>,
    reader: Option<JoinHandle<()>>,
    directory: tempfile::TempDir,
    timeout: Duration,
    sequence: u64,
    failed: bool,
}
impl Drop for Worker {
    fn drop(&mut self) {
        // Let Python release CUDA/runtime resources before resorting to kill.
        if !self.failed {
            let message = format!("{{\"op\":\"shutdown\",\"id\":{}}}\n", self.sequence + 1);
            let _ = self.input.write_all(message.as_bytes());
            let _ = self.input.flush();
            let deadline = std::time::Instant::now() + Duration::from_secs(1);
            while std::time::Instant::now() < deadline {
                if matches!(self.child.try_wait(), Ok(Some(_))) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}
impl Worker {
    fn call(&mut self, mut request: Value) -> Result<Value> {
        if self.failed {
            return Err(LensError::Backend(
                "worker is no longer usable; launch a new worker".into(),
            ));
        }
        self.sequence += 1;
        request["id"] = json!(self.sequence);
        let bytes = serde_json::to_vec(&request).map_err(|e| LensError::Backend(e.to_string()))?;
        if let Err(e) = self
            .input
            .write_all(&bytes)
            .and_then(|_| self.input.write_all(b"\n"))
            .and_then(|_| self.input.flush())
        {
            self.failed = true;
            return Err(LensError::Backend(format!("worker input: {e}")));
        }
        let response = match self.responses.recv_timeout(self.timeout) {
            Ok(Ok(response)) => response,
            other => {
                self.failed = true;
                let _ = self.child.kill();
                return Err(LensError::Backend(format!(
                    "worker stopped or timed out: {other:?}"
                )));
            }
        };
        if response["id"].as_u64() != Some(self.sequence) {
            self.failed = true;
            return Err(LensError::Backend("worker response ID mismatch".into()));
        }
        if let Some(error) = response.get("error") {
            return Err(LensError::Backend(
                error.as_str().unwrap_or("unknown worker error").into(),
            ));
        }
        Ok(response)
    }
    fn exchange(
        &mut self,
        request: Value,
        tensors: HashMap<String, Tensor>,
        device: &Device,
    ) -> Result<(Value, HashMap<String, Tensor>)> {
        let input = self.directory.path().join("input.safetensors");
        let output = self.directory.path().join("output.safetensors");
        // Each call is under the worker mutex; no request can overwrite another.
        let tensors = tensors
            .into_iter()
            .map(|(k, v)| Ok((k, v.to_device(&Device::Cpu)?.contiguous()?)))
            .collect::<Result<HashMap<_, _>>>()?;
        safetensors::serialize_to_file(&tensors, None, &input)?;
        if output.exists() {
            std::fs::remove_file(&output)?;
        }
        let response = self.call(request)?;
        let tensors = candle_core::safetensors::load(&output, device)?;
        Ok((response, tensors))
    }
}

pub struct HfModel {
    dimensions: ModelDimensions,
    worker: Mutex<Worker>,
    device: Device,
    tokenizer: Option<Tokenizer>,
    metadata: Value,
}
impl HfModel {
    /// Launch one resident model. `timeout` applies to each request, including
    /// loading and one prompt's complete Jacobian. No automatic retry occurs.
    /// Results use F32 on `output_device`; Python's model device is independent.
    pub fn launch(
        python: impl AsRef<Path>,
        options: &HfOptions,
        output_device: &Device,
        timeout: Duration,
    ) -> Result<Self> {
        if timeout.is_zero() || options.threads == 0 {
            return Err(LensError::Invalid(
                "positive timeout and thread count required".into(),
            ));
        }
        let directory = tempfile::tempdir()?;
        let mut child = Command::new(python.as_ref())
            .arg("-u")
            .arg("-c")
            .arg(include_str!("hf_worker.py"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let input = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        let (sender, responses) = mpsc::channel();
        let reader = std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) => {
                        let _ = sender.send(Err("worker exited".into()));
                        break;
                    }
                    Ok(_) => {
                        let parsed = serde_json::from_str(&line)
                            .map_err(|e| format!("invalid worker response: {e}"));
                        if sender.send(parsed).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = sender.send(Err(e.to_string()));
                        break;
                    }
                }
            }
        });
        let mut worker = Worker {
            child,
            input,
            responses,
            reader: Some(reader),
            directory,
            timeout,
            sequence: 0,
            failed: false,
        };
        let reply = worker
            .call(json!({"op":"init", "options":options, "directory":worker.directory.path()}))?;
        let dimensions: ModelDimensions = serde_json::from_value(reply["dimensions"].clone())
            .map_err(|e| LensError::Backend(e.to_string()))?;
        dimensions.validate()?;
        let tokenizer_path = worker.directory.path().join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            Some(
                Tokenizer::from_file(tokenizer_path)
                    .map_err(|e| LensError::Tokenizer(e.to_string()))?,
            )
        } else {
            None
        };
        Ok(Self {
            dimensions,
            worker: Mutex::new(worker),
            device: output_device.clone(),
            tokenizer,
            metadata: reply["metadata"].clone(),
        })
    }
    pub fn tokenizer(&self) -> Result<&Tokenizer> {
        self.tokenizer.as_ref().ok_or_else(|| LensError::Tokenizer("backend has no Rust-compatible fast tokenizer; use encode or supply a tokenizer explicitly".into()))
    }
    pub fn metadata(&self) -> &Value {
        &self.metadata
    }
    fn worker(&self) -> Result<std::sync::MutexGuard<'_, Worker>> {
        self.worker
            .lock()
            .map_err(|_| LensError::Backend("worker lock poisoned".into()))
    }
    /// Use the backend's tokenizer, including its explicit BOS policy.
    pub fn encode(&self, text: &str, max_length: usize) -> Result<Tensor> {
        if max_length == 0 {
            return Err(LensError::Invalid("zero token limit".into()));
        }
        let (_, mut tensors) = self.worker()?.exchange(
            json!({"op":"encode","text":text,"max_length":max_length}),
            HashMap::new(),
            &self.device,
        )?;
        take(&mut tensors, "input_ids")?
            .to_dtype(DType::U32)
            .map_err(Into::into)
    }
}
fn validate_tensor(tensor: &Tensor, shape: &[usize], name: &str) -> Result<()> {
    if tensor.dims() != shape || tensor.dtype() != DType::F32 || !tensor.is_contiguous() {
        return Err(LensError::Backend(format!(
            "{name}: expected contiguous F32 {shape:?}, received {:?} {:?}",
            tensor.dtype(),
            tensor.dims()
        )));
    }
    Ok(())
}

fn take(tensors: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    tensors
        .remove(name)
        .ok_or_else(|| LensError::Backend(format!("worker omitted tensor {name}")))
}
impl Unembed for HfModel {
    fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        if hidden.dims().last() != Some(&self.dimensions.hidden_size) {
            return Err(crate::shape("readout", "[..., hidden_size]", hidden));
        }
        let (_, mut tensors) = self.worker()?.exchange(
            json!({"op":"unembed"}),
            HashMap::from([("hidden".into(), hidden.to_dtype(DType::F32)?)]),
            &self.device,
        )?;
        let logits = take(&mut tensors, "logits")?;
        let mut expected = hidden.dims().to_vec();
        *expected.last_mut().expect("validated nonempty shape") = self.dimensions.vocab_size;
        validate_tensor(&logits, &expected, "logits")?;
        Ok(logits)
    }
}
impl LensModel for HfModel {
    fn dimensions(&self) -> ModelDimensions {
        self.dimensions
    }
    fn capture(
        &self,
        ids: &Tensor,
        layers: &[usize],
        replacements: &[ResidualReplacement],
    ) -> Result<ModelTrace> {
        crate::model::validate_capture(self.dimensions, layers, replacements)?;
        let (batch, seq) = ids.dims2()?;
        if batch == 0 || seq == 0 || !matches!(ids.dtype(), DType::U32 | DType::I64) {
            return Err(LensError::Invalid(
                "nonempty integer token IDs required".into(),
            ));
        }
        let mut input = HashMap::from([("input_ids".into(), ids.clone())]);
        for edit in replacements {
            if edit.hidden.dims() != [batch, seq, self.dimensions.hidden_size]
                || edit.hidden.dtype() != DType::F32
                || !edit.hidden.device().same_device(&self.device)
            {
                return Err(LensError::Invalid(
                    "replacement must match trace shape, F32 dtype and output device".into(),
                ));
            }
            input.insert(format!("replacement.{}", edit.layer), edit.hidden.clone());
        }
        let (_, mut tensors) = self.worker()?.exchange(
            json!({"op":"capture", "layers":layers}),
            input,
            &self.device,
        )?;
        let final_residual = take(&mut tensors, "final_residual")?;
        let shape = [batch, seq, self.dimensions.hidden_size];
        validate_tensor(&final_residual, &shape, "final residual")?;
        let mut residuals = BTreeMap::new();
        for &layer in layers {
            if let std::collections::btree_map::Entry::Vacant(entry) = residuals.entry(layer) {
                let residual = take(&mut tensors, &format!("residual.{layer}"))?;
                validate_tensor(&residual, &shape, "captured residual")?;
                entry.insert(residual);
            }
        }
        Ok(ModelTrace {
            residuals,
            final_residual,
        })
    }
    fn unembedding_rows(&self, tokens: &[u32]) -> Result<Tensor> {
        if tokens.is_empty()
            || tokens
                .iter()
                .any(|&t| t as usize >= self.dimensions.vocab_size)
        {
            return Err(LensError::Invalid(
                "invalid unembedding token selection".into(),
            ));
        }
        let (_, mut tensors) = self.worker()?.exchange(
            json!({"op":"head", "tokens":tokens}),
            HashMap::new(),
            &self.device,
        )?;
        let rows = take(&mut tensors, "rows")?;
        validate_tensor(
            &rows,
            &[tokens.len(), self.dimensions.hidden_size],
            "unembedding rows",
        )?;
        Ok(rows)
    }
}
impl JacobianEstimator for HfModel {
    fn fitting_dimensions(&self) -> (usize, usize) {
        (self.dimensions.hidden_size, self.dimensions.layer_count)
    }
    fn estimate_prompt(&self, ids: &Tensor, config: &FittingConfig) -> Result<PromptJacobian> {
        config.validate(self.dimensions.layer_count)?;
        let (batch, length) = ids.dims2()?;
        let length = length.min(config.max_seq_len);
        if batch != 1
            || length.saturating_sub(1) <= config.skip_first
            || !matches!(ids.dtype(), DType::U32 | DType::I64)
        {
            return Err(LensError::Invalid("invalid fitting token IDs".into()));
        }
        let (reply, mut tensors) = self.worker()?.exchange(
            json!({"op":"jacobian", "config":config}),
            HashMap::from([("input_ids".into(), ids.clone())]),
            &self.device,
        )?;
        let mut jacobians = BTreeMap::new();
        for &layer in &config.source_layers {
            jacobians.insert(layer, take(&mut tensors, &format!("jacobians.{layer}"))?);
        }
        let number = |key: &str| {
            reply[key]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| LensError::Backend(format!("invalid {key}")))
        };
        Ok(PromptJacobian {
            jacobians,
            seq_len: number("seq_len")?,
            valid_positions: number("valid_positions")?,
        })
    }
}
