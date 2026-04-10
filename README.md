# Mimi Codec

Standalone Rust implementation of the [Mimi](https://github.com/kyutai-labs/moshi) streaming neural audio codec, extracted from the Moshi monorepo. Includes a native Candle backend, ONNX Runtime backend, JNI bridge for Android, and Python bindings.

24 kHz mono audio, 12.5 Hz frame rate, configurable 1–32 codebooks.

## Crates

| Crate | Description |
|-------|-------------|
| `mimi-core` | Candle-based codec: SEANet encoder/decoder, transformer, RVQ |
| `mimi-onnx` | ONNX Runtime backend with streaming state management |
| `mimi-cli` | CLI tool for encode/decode/roundtrip |
| `mimi-jni` | JNI bridge for Android (used by [MimiDemo](https://github.com/tatujan/MimiDemo)) |
| `mimi-pyo3` | Python bindings via PyO3 |

## ONNX Streaming Models

Pre-exported streaming ONNX models are available on Hugging Face: [BMekiker/mimi-onnx-streaming](https://huggingface.co/BMekiker/mimi-onnx-streaming)

| Variant | Encoder | Decoder | Bitrate |
|---------|---------|---------|---------|
| 8 codebooks | 194 MB | 170 MB | ~1.1 kbps |
| 16 codebooks | 242 MB | 186 MB | ~2.2 kbps |

### Exporting ONNX models

```bash
pip install torch transformers onnx onnxruntime

# 8 codebooks, 320ms chunks
python scripts/export_streaming_onnx.py --num-codebooks 8 --chunk-ms 320 --output-dir onnx-models/streaming-8cb

# 16 codebooks
python scripts/export_streaming_onnx.py --num-codebooks 16 --chunk-ms 320 --output-dir onnx-models/streaming-16cb
```

## CLI Usage

```bash
# Build
cargo build --release -p mimi-cli

# Roundtrip (encode + decode)
mimi-cli roundtrip --input test.wav --output out.wav --num-codebooks 8

# ONNX backend
mimi-cli roundtrip --input test.wav --output out.wav \
  --backend onnx \
  --encoder-model onnx-models/streaming-8cb/encoder_model.onnx \
  --decoder-model onnx-models/streaming-8cb/decoder_model.onnx \
  --num-codebooks 8 --streaming
```

## Building for Android

```bash
cargo install cargo-ndk

cd mimi-jni
cargo ndk -t arm64-v8a -P 24 build --release

# Copy .so to the Android project
cp target/aarch64-linux-android/release/libmimi_jni.so \
   ../MimiDemo/app/src/main/jniLibs/arm64-v8a/
```

## Scripts

| Script | Description |
|--------|-------------|
| `export_streaming_onnx.py` | Export PyTorch Mimi to streaming ONNX with causal attention masking |
| `export_onnx.py` | Export batch (non-streaming) ONNX models |
| `compare_backends.py` | Numerical comparison between PyTorch, ONNX batch, and ONNX streaming |
| `convert_to_gguf.py` | Convert safetensors to GGUF (Q4_0, Q8_0) |

## License

This project contains code derived from [Moshi](https://github.com/kyutai-labs/moshi) by [Kyutai](https://kyutai.ai/).

- **Code**: MIT License — see [LICENSE](LICENSE)
- **Model weights** (Mimi codec): [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) by Kyutai
