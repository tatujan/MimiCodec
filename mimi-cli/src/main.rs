use anyhow::{bail, Context, Result};
use candle::Device;
use clap::{Parser, Subcommand};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

const DEFAULT_HF_REPO: &str = "kyutai/moshiko-pytorch-bf16";
const DEFAULT_MODEL_FILE: &str = "tokenizer-e351c8d8-checkpoint125.safetensors";
const FRAME_SIZE: usize = 1920; // 80ms at 24kHz

#[derive(Debug, Parser)]
#[command(name = "mimi", about = "Mimi audio codec CLI")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Encode a WAV file to tokens, then decode back to WAV
    Roundtrip {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 8)]
        num_codebooks: usize,
        #[arg(long)]
        model_file: Option<PathBuf>,
        #[arg(long)]
        cpu: bool,
        /// Use ONNX backend instead of Candle (requires --encoder-model and --decoder-model)
        #[arg(long)]
        backend: Option<String>,
        /// Path to encoder ONNX model (for --backend onnx)
        #[arg(long)]
        encoder_model: Option<PathBuf>,
        /// Path to decoder ONNX model (for --backend onnx)
        #[arg(long)]
        decoder_model: Option<PathBuf>,
        /// Use streaming ONNX models with KV cache (for --backend onnx)
        #[arg(long)]
        streaming: bool,
    },
    /// Encode a WAV file to tokens
    Encode {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 8)]
        num_codebooks: usize,
        #[arg(long)]
        model_file: Option<PathBuf>,
        #[arg(long)]
        cpu: bool,
    },
    /// Decode tokens back to a WAV file
    Decode {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 8)]
        num_codebooks: usize,
        #[arg(long)]
        model_file: Option<PathBuf>,
        #[arg(long)]
        cpu: bool,
    },
}

fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if candle::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if candle::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        eprintln!("Warning: neither CUDA nor Metal available, falling back to CPU");
        Ok(Device::Cpu)
    }
}

fn resolve_model_file(model_file: Option<PathBuf>) -> Result<String> {
    if let Some(path) = model_file {
        return Ok(path.to_string_lossy().to_string());
    }
    eprintln!("No --model-file specified, downloading from HuggingFace...");
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(DEFAULT_HF_REPO.to_string());
    let path = repo.get(DEFAULT_MODEL_FILE)?;
    Ok(path.to_string_lossy().to_string())
}

fn read_wav_pcm(path: &PathBuf) -> Result<Vec<f32>> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path).context("opening input WAV")?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;

    let mut format = probed.format;
    let track = format
        .tracks()
        .first()
        .context("no audio tracks found")?;

    let sample_rate = track
        .codec_params
        .sample_rate
        .context("no sample rate")?;
    let channels = track
        .codec_params
        .channels
        .context("no channel info")?
        .count();

    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())?;

    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(e.into()),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = decoder.decode(&packet)?;
        let spec = *decoded.spec();
        let num_frames = decoded.capacity();
        let mut sample_buf = SampleBuffer::<f32>::new(num_frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        let samples = sample_buf.samples();

        if channels == 1 {
            all_samples.extend_from_slice(samples);
        } else {
            // Mix to mono
            for chunk in samples.chunks(channels) {
                let mono: f32 = chunk.iter().sum::<f32>() / channels as f32;
                all_samples.push(mono);
            }
        }
    }

    if sample_rate != 24000 {
        bail!(
            "Input sample rate is {sample_rate}Hz, but Mimi requires 24000Hz. \
             Please resample your audio first (e.g., ffmpeg -i input.wav -ar 24000 output.wav)"
        );
    }

    Ok(all_samples)
}

fn run_encode(
    mimi: &mut mimi_core::mimi::Mimi,
    pcm: &[f32],
    device: &Device,
) -> Result<Vec<Vec<Vec<Vec<u32>>>>> {
    mimi.reset_state();
    let mut all_codes: Vec<Vec<Vec<Vec<u32>>>> = Vec::new();
    let num_frames = pcm.len() / FRAME_SIZE;

    for i in 0..num_frames {
        let chunk = &pcm[i * FRAME_SIZE..(i + 1) * FRAME_SIZE];
        let tensor = candle::Tensor::from_slice(chunk, (1, 1, FRAME_SIZE), device)?;
        let codes = mimi.encode_step(
            &mimi_core::StreamTensor::from_tensor(tensor),
            &().into(),
        )?;
        if let Some(codes) = codes.as_option() {
            let codes_vec = codes.to_vec3::<u32>()?;
            all_codes.push(codes_vec);
        }
    }

    Ok(all_codes)
}

fn run_decode(
    mimi: &mut mimi_core::mimi::Mimi,
    all_codes: &[Vec<Vec<Vec<u32>>>],
    device: &Device,
) -> Result<Vec<f32>> {
    mimi.reset_state();
    let mut all_pcm: Vec<f32> = Vec::new();

    for frame_codes in all_codes {
        // frame_codes is [batch=1, codebooks, time]
        let batch = &frame_codes[0]; // [codebooks, time]
        let codes = candle::Tensor::new(batch.clone(), device)?.unsqueeze(0)?;
        let pcm = mimi.decode_step(
            &mimi_core::StreamTensor::from_tensor(codes),
            &().into(),
        )?;
        if let Some(pcm) = pcm.as_option() {
            let pcm_vec = pcm.to_vec3::<f32>()?;
            all_pcm.extend_from_slice(&pcm_vec[0][0]);
        }
    }

    Ok(all_pcm)
}

fn write_tokens(path: &PathBuf, all_codes: &[Vec<Vec<Vec<u32>>>]) -> Result<()> {
    let mut file = std::fs::File::create(path)?;
    let num_frames = all_codes.len() as u32;
    let num_codebooks = if num_frames > 0 {
        all_codes[0][0].len() as u32
    } else {
        0
    };
    // Simple header: magic, num_codebooks, num_frames
    file.write_all(b"MIMI")?;
    file.write_all(&num_codebooks.to_le_bytes())?;
    file.write_all(&num_frames.to_le_bytes())?;
    for frame in all_codes {
        // frame[0] = [codebooks][time]
        for codebook_times in &frame[0] {
            for &val in codebook_times {
                file.write_all(&val.to_le_bytes())?;
            }
        }
    }
    Ok(())
}

fn read_tokens(path: &PathBuf) -> Result<(usize, Vec<Vec<Vec<Vec<u32>>>>)> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != b"MIMI" {
        bail!("Invalid token file (bad magic)");
    }
    let mut buf4 = [0u8; 4];
    file.read_exact(&mut buf4)?;
    let num_codebooks = u32::from_le_bytes(buf4) as usize;
    file.read_exact(&mut buf4)?;
    let num_frames = u32::from_le_bytes(buf4) as usize;

    // Each frame: [batch=1, codebooks, time=1]
    let mut all_codes = Vec::with_capacity(num_frames);
    for _ in 0..num_frames {
        let mut codebooks = Vec::with_capacity(num_codebooks);
        for _ in 0..num_codebooks {
            file.read_exact(&mut buf4)?;
            let val = u32::from_le_bytes(buf4);
            codebooks.push(vec![val]); // time dim = 1
        }
        all_codes.push(vec![codebooks]); // batch dim = 1 -> [1, K, 1]
    }
    Ok((num_codebooks, all_codes))
}

fn main() -> Result<()> {
    let args = Args::parse();

    match args.command {
        Command::Roundtrip {
            input,
            output,
            num_codebooks,
            model_file,
            cpu,
            backend,
            encoder_model,
            decoder_model,
            streaming,
        } => {
            eprintln!("Reading: {}", input.display());
            let pcm = read_wav_pcm(&input)?;
            let usable_samples = (pcm.len() / FRAME_SIZE) * FRAME_SIZE;
            let pcm = &pcm[..usable_samples];
            let duration_secs = usable_samples as f64 / 24000.0;
            eprintln!(
                "Input: {:.2}s ({} samples, truncated to {} frames)",
                duration_secs,
                pcm.len(),
                usable_samples / FRAME_SIZE
            );

            let is_onnx = backend.as_deref() == Some("onnx");

            if is_onnx {
                #[cfg(feature = "onnx")]
                {
                    let enc_path = encoder_model
                        .as_ref()
                        .context("--encoder-model required for ONNX backend")?;
                    let dec_path = decoder_model
                        .as_ref()
                        .context("--decoder-model required for ONNX backend")?;

                    eprintln!("Loading ONNX models...");
                    eprintln!("  encoder: {}", enc_path.display());
                    eprintln!("  decoder: {}", dec_path.display());
                    let mut codec = mimi_onnx::OnnxMimiCodec::new(
                        &enc_path.to_string_lossy(),
                        &dec_path.to_string_lossy(),
                        num_codebooks,
                        false, // no NNAPI on desktop
                        streaming,
                    )?;
                    let mode = if streaming { "streaming" } else { "batch" };
                    eprintln!("ONNX models loaded ({num_codebooks} codebooks, {mode})");

                    let (all_codes_flat, encode_time) = if streaming {
                        // Streaming encode: process frame by frame
                        let t0 = Instant::now();
                        codec.reset_state();
                        let mut all_codes: Vec<Vec<u32>> = Vec::new();
                        for i in 0..(pcm.len() / FRAME_SIZE) {
                            let chunk = &pcm[i * FRAME_SIZE..(i + 1) * FRAME_SIZE];
                            if let Some(codes) = codec.encode_step(chunk)? {
                                // Accumulate: extend each codebook's time series
                                if all_codes.is_empty() {
                                    all_codes = codes;
                                } else {
                                    for (cb, new_times) in codes.iter().enumerate() {
                                        all_codes[cb].extend_from_slice(new_times);
                                    }
                                }
                            }
                        }
                        (all_codes, t0.elapsed())
                    } else {
                        let t0 = Instant::now();
                        let codes = codec.encode(pcm)?;
                        (codes, t0.elapsed())
                    };
                    eprintln!(
                        "Encoded in {:.3}s ({:.1}x realtime)",
                        encode_time.as_secs_f64(),
                        duration_secs / encode_time.as_secs_f64()
                    );

                    let (decoded, decode_time) = if streaming {
                        // Streaming decode: process token by token
                        let t0 = Instant::now();
                        codec.reset_state();
                        let n_time = if !all_codes_flat.is_empty() { all_codes_flat[0].len() } else { 0 };
                        let mut all_pcm: Vec<f32> = Vec::new();
                        for t in 0..n_time {
                            let frame_codes: Vec<Vec<u32>> = all_codes_flat.iter()
                                .map(|cb| vec![cb[t]])
                                .collect();
                            if let Some(pcm_out) = codec.decode_step(&frame_codes)? {
                                all_pcm.extend_from_slice(&pcm_out);
                            }
                        }
                        (all_pcm, t0.elapsed())
                    } else {
                        let t0 = Instant::now();
                        let decoded = codec.decode(&all_codes_flat)?;
                        (decoded, t0.elapsed())
                    };
                    eprintln!(
                        "Decoded in {:.3}s ({:.1}x realtime)",
                        decode_time.as_secs_f64(),
                        duration_secs / decode_time.as_secs_f64()
                    );

                    let out_file = std::fs::File::create(&output)?;
                    let mut out_buf = std::io::BufWriter::new(out_file);
                    mimi_core::wav::write_pcm_as_wav(&mut out_buf, &decoded, 24000)?;
                    eprintln!("Written: {}", output.display());
                }
                #[cfg(not(feature = "onnx"))]
                {
                    bail!("ONNX backend requested but 'onnx' feature not enabled. Rebuild with: cargo build --features onnx");
                }
            } else {
                // Candle backend (default)
                let dev = device(cpu)?;
                eprintln!("Device: {:?}", dev);
                let model_path = resolve_model_file(model_file)?;
                eprintln!("Loading model from: {model_path}");
                let mut mimi =
                    mimi_core::mimi::load(&model_path, Some(num_codebooks), &dev)?;
                eprintln!("Model loaded ({num_codebooks} codebooks)");

                let t0 = Instant::now();
                let codes = run_encode(&mut mimi, pcm, &dev)?;
                let encode_time = t0.elapsed();
                eprintln!(
                    "Encoded in {:.3}s ({:.1}x realtime)",
                    encode_time.as_secs_f64(),
                    duration_secs / encode_time.as_secs_f64()
                );

                let t0 = Instant::now();
                let decoded = run_decode(&mut mimi, &codes, &dev)?;
                let decode_time = t0.elapsed();
                eprintln!(
                    "Decoded in {:.3}s ({:.1}x realtime)",
                    decode_time.as_secs_f64(),
                    duration_secs / decode_time.as_secs_f64()
                );

                let out_file = std::fs::File::create(&output)?;
                let mut out_buf = std::io::BufWriter::new(out_file);
                mimi_core::wav::write_pcm_as_wav(&mut out_buf, &decoded, 24000)?;
                eprintln!("Written: {}", output.display());
            }
        }
        Command::Encode {
            input,
            output,
            num_codebooks,
            model_file,
            cpu,
        } => {
            let dev = device(cpu)?;
            let model_path = resolve_model_file(model_file)?;
            eprintln!("Loading model from: {model_path}");
            let mut mimi =
                mimi_core::mimi::load(&model_path, Some(num_codebooks), &dev)?;

            let pcm = read_wav_pcm(&input)?;
            let usable_samples = (pcm.len() / FRAME_SIZE) * FRAME_SIZE;
            let pcm = &pcm[..usable_samples];
            eprintln!("Encoding {} frames...", usable_samples / FRAME_SIZE);

            let codes = run_encode(&mut mimi, pcm, &dev)?;
            write_tokens(&output, &codes)?;
            eprintln!("Tokens written to: {}", output.display());
        }
        Command::Decode {
            input,
            output,
            num_codebooks: _,
            model_file,
            cpu,
        } => {
            let dev = device(cpu)?;
            let model_path = resolve_model_file(model_file)?;
            eprintln!("Loading model from: {model_path}");
            let (nc, all_codes) = read_tokens(&input)?;
            let mut mimi =
                mimi_core::mimi::load(&model_path, Some(nc), &dev)?;

            eprintln!("Decoding {} frames...", all_codes.len());
            let decoded = run_decode(&mut mimi, &all_codes, &dev)?;

            let out_file = std::fs::File::create(&output)?;
            let mut out_buf = std::io::BufWriter::new(out_file);
            mimi_core::wav::write_pcm_as_wav(&mut out_buf, &decoded, 24000)?;
            eprintln!("Written: {}", output.display());
        }
    }

    Ok(())
}
