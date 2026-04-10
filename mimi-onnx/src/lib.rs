//! ONNX Runtime backend for Mimi audio codec.
//!
//! Provides both batch and streaming encode/decode using ONNX Runtime,
//! with optional NNAPI execution provider for Android NPU/DSP acceleration.
//!
//! v2 streaming models carry both conv state buffers (11 per encoder/decoder)
//! and KV cache state as explicit tensor inputs/outputs.

use anyhow::Result;
use ort::memory::Allocator;
use ort::session::Session;
use ort::value::Tensor;

const NUM_LAYERS: usize = 8;
const NUM_HEADS: usize = 8;
const HEAD_DIM: usize = 64;

/// A conv state buffer: flat f32 data of shape [1, channels, temporal_size].
#[derive(Clone)]
struct ConvState {
    name: String,
    channels: usize,
    temporal_size: usize,
    data: Vec<f32>,
}

impl ConvState {
    fn new(name: String, channels: usize, temporal_size: usize) -> Self {
        Self {
            name,
            channels,
            temporal_size,
            data: vec![0.0; channels * temporal_size],
        }
    }

    fn to_tensor(&self) -> Result<Tensor<f32>> {
        Tensor::from_array((
            [1usize, self.channels, self.temporal_size],
            self.data.clone().into_boxed_slice(),
        ))
        .map_err(|e| anyhow::anyhow!("Failed to create conv state tensor '{}': {e}", self.name))
    }

    fn update_from(&mut self, data: &[f32], shape: &[i64]) {
        let new_channels = shape[1] as usize;
        let new_temporal = shape[2] as usize;
        self.channels = new_channels;
        self.temporal_size = new_temporal;
        self.data = data.to_vec();
    }
}

/// Create a KV cache tensor. For seq_len=0, uses allocator API (supports zero dims).
/// For seq_len>0, uses from_array with the provided data.
fn make_cache_tensor(seq_len: usize, data: &[f32]) -> Result<Tensor<f32>> {
    if seq_len == 0 {
        let allocator = Allocator::default();
        Tensor::<f32>::new(&allocator, [1i64, NUM_HEADS as i64, 0i64, HEAD_DIM as i64])
            .map_err(|e| anyhow::anyhow!("Failed to create empty cache tensor: {e}"))
    } else {
        Tensor::from_array(([1usize, NUM_HEADS, seq_len, HEAD_DIM], data.to_vec().into_boxed_slice()))
            .map_err(|e| anyhow::anyhow!("Failed to create cache tensor: {e}"))
    }
}

/// Parse state_spec.txt to get conv state shapes.
fn parse_state_spec(spec_path: &str) -> Result<(Vec<(String, usize, usize)>, Vec<(String, usize, usize)>)> {
    let content = std::fs::read_to_string(spec_path)
        .map_err(|e| anyhow::anyhow!("Failed to read state spec '{spec_path}': {e}"))?;

    let mut encoder_states = Vec::new();
    let mut decoder_states = Vec::new();
    let mut section = "";

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "[encoder]" {
            section = "encoder";
            continue;
        }
        if line == "[decoder]" {
            section = "decoder";
            continue;
        }
        // Format: "conv name channels temporal_size" or "conv_tr name channels temporal_size"
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 4 {
            let name = parts[1].to_string();
            let channels: usize = parts[2].parse().map_err(|e| anyhow::anyhow!("Bad channels: {e}"))?;
            let temporal: usize = parts[3].parse().map_err(|e| anyhow::anyhow!("Bad temporal: {e}"))?;
            match section {
                "encoder" => encoder_states.push((name, channels, temporal)),
                "decoder" => decoder_states.push((name, channels, temporal)),
                _ => {}
            }
        }
    }
    Ok((encoder_states, decoder_states))
}

pub struct OnnxMimiCodec {
    encoder_session: Session,
    decoder_session: Session,
    num_codebooks: usize,
    streaming: bool,
    // v2 conv state buffers
    encoder_conv_states: Vec<ConvState>,
    decoder_conv_states: Vec<ConvState>,
    // KV cache state for streaming mode.
    encoder_kv_cache: Vec<Vec<f32>>,
    decoder_kv_cache: Vec<Vec<f32>>,
    encoder_cache_seq_len: usize,
    decoder_cache_seq_len: usize,
}

impl OnnxMimiCodec {
    /// Create a new ONNX-based Mimi codec.
    ///
    /// For streaming v2 models, reads `state_spec.txt` from the model directory
    /// to initialize conv state buffers.
    pub fn new(
        encoder_path: &str,
        decoder_path: &str,
        num_codebooks: usize,
        use_nnapi: bool,
        streaming: bool,
    ) -> Result<Self> {
        let encoder_session = build_session(encoder_path, use_nnapi)
            .map_err(|e| anyhow::anyhow!("Failed to create encoder ONNX session: {e}"))?;
        let decoder_session = build_session(decoder_path, use_nnapi)
            .map_err(|e| anyhow::anyhow!("Failed to create decoder ONNX session: {e}"))?;

        let num_cache_tensors = NUM_LAYERS * 2;

        // Try to load state_spec.txt for v2 conv states
        let model_dir = std::path::Path::new(encoder_path)
            .parent()
            .unwrap_or(std::path::Path::new("."));
        let spec_path = model_dir.join("state_spec.txt");

        let (encoder_conv_states, decoder_conv_states) = if streaming && spec_path.exists() {
            let spec_str = spec_path.to_string_lossy().to_string();
            let (enc_specs, dec_specs) = parse_state_spec(&spec_str)?;
            let enc_states: Vec<ConvState> = enc_specs
                .into_iter()
                .map(|(name, ch, ts)| ConvState::new(name, ch, ts))
                .collect();
            let dec_states: Vec<ConvState> = dec_specs
                .into_iter()
                .map(|(name, ch, ts)| ConvState::new(name, ch, ts))
                .collect();
            eprintln!(
                "[mimi-onnx] v2 streaming: {} enc conv states, {} dec conv states",
                enc_states.len(),
                dec_states.len()
            );
            (enc_states, dec_states)
        } else {
            (Vec::new(), Vec::new())
        };

        Ok(Self {
            encoder_session,
            decoder_session,
            num_codebooks,
            streaming,
            encoder_conv_states,
            decoder_conv_states,
            encoder_kv_cache: vec![Vec::new(); num_cache_tensors],
            decoder_kv_cache: vec![Vec::new(); num_cache_tensors],
            encoder_cache_seq_len: 0,
            decoder_cache_seq_len: 0,
        })
    }

    /// Reset streaming state (KV caches and conv states).
    pub fn reset_state(&mut self) {
        for cache in &mut self.encoder_kv_cache {
            cache.clear();
        }
        for cache in &mut self.decoder_kv_cache {
            cache.clear();
        }
        self.encoder_cache_seq_len = 0;
        self.decoder_cache_seq_len = 0;
        for state in &mut self.encoder_conv_states {
            state.data.fill(0.0);
        }
        for state in &mut self.decoder_conv_states {
            state.data.fill(0.0);
        }
    }

    /// Streaming encode: process one chunk of PCM audio and return codec tokens.
    pub fn encode_step(&mut self, pcm: &[f32]) -> Result<Option<Vec<Vec<u32>>>> {
        if !self.streaming {
            anyhow::bail!("encode_step requires streaming mode");
        }
        let len = pcm.len();
        let input_tensor =
            Tensor::from_array(([1usize, 1, len], pcm.to_vec().into_boxed_slice()))
                .map_err(|e| anyhow::anyhow!("Failed to create input tensor: {e}"))?;

        let mut inputs: Vec<ort::session::SessionInputValue> = vec![input_tensor.into()];

        // Add conv state inputs (v2)
        for state in &self.encoder_conv_states {
            inputs.push(state.to_tensor()?.into());
        }

        // Add KV cache inputs
        let seq_len = self.encoder_cache_seq_len;
        for cache_data in &self.encoder_kv_cache {
            inputs.push(make_cache_tensor(seq_len, cache_data)?.into());
        }

        let outputs = self
            .encoder_session
            .run(inputs.as_slice())
            .map_err(|e| anyhow::anyhow!("Encoder streaming inference failed: {e}"))?;

        // Output[0]: audio_codes [batch=1, num_codebooks, codes_length]
        let (shape, codes_data) = outputs[0]
            .try_extract_tensor::<i64>()
            .map_err(|e| anyhow::anyhow!("Failed to extract encoder output: {e}"))?;
        let n_codebooks = shape[1] as usize;
        let n_time = shape[2] as usize;

        // Extract updated conv states from outputs[1..1+n_conv]
        let n_conv = self.encoder_conv_states.len();
        for i in 0..n_conv {
            let (conv_shape, conv_data) = outputs[1 + i]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("Failed to extract encoder conv state: {e}"))?;
            self.encoder_conv_states[i].update_from(&conv_data, &conv_shape);
        }

        // Extract updated KV cache from outputs[1+n_conv..]
        let mut new_cache_seq_len = 0;
        for i in 0..(NUM_LAYERS * 2) {
            let (cache_shape, cache_data) = outputs[1 + n_conv + i]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("Failed to extract cache output: {e}"))?;
            new_cache_seq_len = cache_shape[2] as usize;
            self.encoder_kv_cache[i] = cache_data.to_vec();
        }
        self.encoder_cache_seq_len = new_cache_seq_len;

        if n_time == 0 {
            return Ok(None);
        }

        let used_cb = n_codebooks.min(self.num_codebooks);
        let mut codes = vec![vec![0u32; n_time]; used_cb];
        for cb in 0..used_cb {
            for t in 0..n_time {
                codes[cb][t] = codes_data[cb * n_time + t] as u32;
            }
        }
        Ok(Some(codes))
    }

    /// Streaming decode: process codec tokens and return PCM audio.
    pub fn decode_step(&mut self, codes: &[Vec<u32>]) -> Result<Option<Vec<f32>>> {
        if !self.streaming {
            anyhow::bail!("decode_step requires streaming mode");
        }
        let n_codebooks = codes.len();
        let n_time = if n_codebooks > 0 { codes[0].len() } else { 0 };

        let mut flat = vec![0i64; n_codebooks * n_time];
        for cb in 0..n_codebooks {
            for t in 0..n_time {
                flat[cb * n_time + t] = codes[cb][t] as i64;
            }
        }
        let input_tensor =
            Tensor::from_array(([1usize, n_codebooks, n_time], flat.into_boxed_slice()))
                .map_err(|e| anyhow::anyhow!("Failed to create codes tensor: {e}"))?;

        let mut inputs: Vec<ort::session::SessionInputValue> = vec![input_tensor.into()];

        // Add conv state inputs (v2)
        for state in &self.decoder_conv_states {
            inputs.push(state.to_tensor()?.into());
        }

        // Add KV cache inputs
        let seq_len = self.decoder_cache_seq_len;
        for cache_data in &self.decoder_kv_cache {
            inputs.push(make_cache_tensor(seq_len, cache_data)?.into());
        }

        let outputs = self
            .decoder_session
            .run(inputs.as_slice())
            .map_err(|e| anyhow::anyhow!("Decoder streaming inference failed: {e}"))?;

        // Output[0]: audio_values [batch=1, channels=1, seq_len]
        let (shape, pcm_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract decoder output: {e}"))?;
        let seq_len_out = shape[2] as usize;

        // Extract updated conv states
        let n_conv = self.decoder_conv_states.len();
        for i in 0..n_conv {
            let (conv_shape, conv_data) = outputs[1 + i]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("Failed to extract decoder conv state: {e}"))?;
            self.decoder_conv_states[i].update_from(&conv_data, &conv_shape);
        }

        // Extract updated KV cache
        let mut new_cache_seq_len = 0;
        for i in 0..(NUM_LAYERS * 2) {
            let (cache_shape, cache_data) = outputs[1 + n_conv + i]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("Failed to extract cache output: {e}"))?;
            new_cache_seq_len = cache_shape[2] as usize;
            self.decoder_kv_cache[i] = cache_data.to_vec();
        }
        self.decoder_cache_seq_len = new_cache_seq_len;

        if seq_len_out == 0 {
            return Ok(None);
        }

        Ok(Some(pcm_data[..seq_len_out].to_vec()))
    }

    /// Batch encode PCM audio to codec tokens (non-streaming).
    pub fn encode(&mut self, pcm: &[f32]) -> Result<Vec<Vec<u32>>> {
        if self.streaming {
            self.reset_state();
            let result = self.encode_step(pcm)?;
            self.reset_state();
            return result.ok_or_else(|| anyhow::anyhow!("Batch encode produced no output"));
        }

        let len = pcm.len();
        let input_tensor = Tensor::from_array(([1usize, 1, len], pcm.to_vec().into_boxed_slice()))
            .map_err(|e| anyhow::anyhow!("Failed to create input tensor: {e}"))?;

        let outputs = self
            .encoder_session
            .run(ort::inputs![input_tensor])
            .map_err(|e| anyhow::anyhow!("Encoder inference failed: {e}"))?;

        let (shape, codes_data) = outputs[0]
            .try_extract_tensor::<i64>()
            .map_err(|e| anyhow::anyhow!("Failed to extract encoder output as i64: {e}"))?;
        let total_codebooks = shape[1] as usize;
        let n_codebooks = total_codebooks.min(self.num_codebooks);
        let n_time = shape[2] as usize;

        let mut codes = vec![vec![0u32; n_time]; n_codebooks];
        for cb in 0..n_codebooks {
            for t in 0..n_time {
                codes[cb][t] = codes_data[cb * n_time + t] as u32;
            }
        }
        Ok(codes)
    }

    /// Batch decode codec tokens back to PCM audio (non-streaming).
    pub fn decode(&mut self, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
        if self.streaming {
            self.reset_state();
            let result = self.decode_step(codes)?;
            self.reset_state();
            return result.ok_or_else(|| anyhow::anyhow!("Batch decode produced no output"));
        }

        let n_input_codebooks = codes.len();
        let n_time = if n_input_codebooks > 0 {
            codes[0].len()
        } else {
            0
        };

        let mut flat = vec![0i64; n_input_codebooks * n_time];
        for cb in 0..n_input_codebooks {
            for t in 0..n_time {
                flat[cb * n_time + t] = codes[cb][t] as i64;
            }
        }
        let input_tensor =
            Tensor::from_array(([1usize, n_input_codebooks, n_time], flat.into_boxed_slice()))
                .map_err(|e| anyhow::anyhow!("Failed to create input tensor: {e}"))?;

        let outputs = self
            .decoder_session
            .run(ort::inputs![input_tensor])
            .map_err(|e| anyhow::anyhow!("Decoder inference failed: {e}"))?;

        let (shape, pcm_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract decoder output as f32: {e}"))?;
        let seq_len = shape[2] as usize;

        Ok(pcm_data[..seq_len].to_vec())
    }

    pub fn num_codebooks(&self) -> usize {
        self.num_codebooks
    }

    pub fn is_streaming(&self) -> bool {
        self.streaming
    }
}

fn build_session(model_path: &str, use_nnapi: bool) -> Result<Session> {
    let mut builder = Session::builder()
        .map_err(|e| anyhow::anyhow!("Failed to create session builder: {e}"))?;

    if use_nnapi {
        #[cfg(feature = "nnapi")]
        {
            builder = builder.with_execution_providers([
                ort::execution_providers::NNAPIExecutionProvider::default().build(),
                ort::execution_providers::CPUExecutionProvider::default().build(),
            ])
            .map_err(|e| anyhow::anyhow!("Failed to set execution providers: {e}"))?;
        }
        #[cfg(not(feature = "nnapi"))]
        {
            eprintln!("[mimi-onnx] NNAPI requested but 'nnapi' feature not enabled, using CPU");
            let _ = use_nnapi;
        }
    }

    let session = builder
        .commit_from_file(model_path)
        .map_err(|e| anyhow::anyhow!("Failed to load ONNX model '{model_path}': {e}"))?;

    Ok(session)
}
