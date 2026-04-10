# Mimi Audio Codec — Extraction & Build Report

## Objective

Extract the Mimi streaming neural audio codec from the Moshi monorepo
(kyutai-labs/moshi) into a standalone Rust project, build and test it on
macOS (Apple M4), then cross-compile a JNI library for Android.

---

## Background

We initially explored Stability AI's Stable Codec (`stable-codec-speech-16k`)
for low-bitrate speech compression. It was ruled out because it has a hard
dependency on FlashAttention/CUDA — no CPU, no Metal, no mobile support.

We then identified Mimi, the neural audio codec inside Kyutai's Moshi
speech-to-speech repo. Mimi is a streaming codec (80ms latency, ~1.1 kbps)
with implementations in PyTorch, MLX, and Rust. The Rust implementation
uses the Candle ML framework and supports Metal (macOS GPU) and CPU backends,
making it viable for both Mac testing and Android deployment.

---

## Phase 1: Extraction from Moshi

### Source repo

Cloned `https://github.com/kyutai-labs/moshi.git` to `~/dev/moshi/`.

The Moshi repo is a monorepo containing the full Moshi speech-to-speech
model (7B parameter language model), the Mimi codec, server infrastructure,
web client, and MLX/Rust/PyTorch implementations. We needed only the Mimi
codec from the Rust implementation.

### File analysis

The Rust source lives in `rust/moshi-core/src/`. We catalogued every source
file and classified it as Mimi-specific or Moshi-LM-specific:

**Mimi-specific (copied):**
- `mimi.rs` — main codec struct, Config, encode/decode, streaming, model loading
- `seanet.rs` — SEANet encoder/decoder (convolutional audio backbone)
- `quantization.rs` — SplitResidualVectorQuantizer (codebook quantization)
- `transformer.rs` — Transformer with RoPE positional embeddings, KV caching
- `streaming.rs` — StreamTensor, StreamMask, StreamingModule trait
- `conv.rs` — StreamableConv1d, ConvDownsample1d, ConvTrUpsample1d
- `kv_cache.rs` — key-value cache for attention layers
- `nn.rs` — MaybeQuantizedLinear (supports both f32 and GGML quantized weights)
- `wav.rs` — WAV file I/O

**Moshi-LM-specific (skipped):**
- `lm.rs`, `asr.rs`, `lm_generate.rs`, `lm_generate_multistream.rs`
- `batched_transformer.rs`, `conditioner.rs`, `tts.rs`, `tts_streaming.rs`

### Modification required

The original `transformer.rs` had a `Transformer` enum with two variants:
`Standard(ProjectedTransformer)` and `Batched(crate::batched_transformer::ProjectedTransformer)`.
The `Batched` variant referenced the `batched_transformer` module which is
Moshi-LM-specific and was not copied.

Mimi only uses the `Standard` variant (it always passes `batch_size = None`
to `Transformer::new()`). The `Batched` variant is used by the Moshi LM
server for batched inference across multiple concurrent sessions.

**Fix:** Replaced the `Transformer` enum with a simple struct wrapping
`ProjectedTransformer` directly. The `new()` method ignores `batch_size`
and always constructs the standard transformer. This was the only code
change to any of the 9 copied source files.

A new `lib.rs` was authored that exports only the Mimi modules and defines
the `NormType` enum (referenced by `transformer.rs` and `mimi.rs` as
`crate::NormType`).

---

## Phase 2: macOS Build

### Workspace structure

Created `~/dev/mimi-codec/` with three crates:

```
~/dev/mimi-codec/
  Cargo.toml                  workspace root
  mimi-core/                  codec library (9 copied files + new lib.rs)
  mimi-cli/                   command-line encode/decode tool (new)
  mimi-pyo3/                  Python bindings (ported from moshi)
```

### Rust toolchain

Rust was not installed on the system. Installed via `rustup`:
- rustc 1.94.1
- cargo 1.94.1

### Dependencies

Pinned to the same versions as the Moshi repo:
- `candle-core` 0.9.1 (ML tensor framework, Rust equivalent of PyTorch)
- `candle-nn` 0.9.1 (neural network layers)
- `candle-transformers` 0.9.1 (transformer utilities, quantized NN)
- `serde` 1.0, `rayon` 1.8.1, `tracing` 0.1.40

CLI-specific: `clap` 4.4.12, `hf-hub` 0.4.3, `symphonia` 0.5.3, `tokio` 1.35.1.

### Feature configuration

`mimi-core` features:
```toml
[features]
default = ["metal"]
metal = ["candle/metal", "candle-nn/metal"]
cuda = ["candle/cuda", "candle-nn/cuda"]
flash-attn = ["cuda", "dep:candle-flash-attn"]
```

`default = ["metal"]` so the M4 Mac uses GPU acceleration without flags.

### mimi-cli

Wrote a new CLI binary with three subcommands:
- `mimi roundtrip --input in.wav --output out.wav` — encode then decode
- `mimi encode --input in.wav --output tokens.bin` — encode to token file
- `mimi decode --input tokens.bin --output out.wav` — decode from tokens

Features:
- Automatic device selection: Metal GPU if available, else CPU
- HuggingFace auto-download if `--model-file` not specified
- WAV reading via symphonia (handles various formats and sample rates)
- Streaming encode/decode in 1920-sample chunks (80ms at 24kHz)
- Timing statistics (encode/decode speed as realtime multiple)

### mimi-pyo3

Copied from `moshi/rust/mimi-pyo3/src/lib.rs` (386 lines). Only change:
`use ::moshi as mm;` → `use ::mimi_core as mm;`

Exposes the `rustymimi` Python module with `Tokenizer` and `StreamTokenizer`
classes, matching the original Moshi Python bindings.

### Build result

```
$ cargo build --release -p mimi-core -p mimi-cli
```

Clean build, no errors. The `batched_transformer` removal was the only
compilation issue encountered.

### Model weights

Downloaded Mimi safetensors from HuggingFace:
- Repository: `kyutai/moshiko-pytorch-bf16`
- File: `tokenizer-e351c8d8-checkpoint125.safetensors`
- Size: 367 MB
- Stored at: `~/dev/mimi-codec/models/`

### Test results

Generated a 3-second 440Hz sine wave test WAV (24kHz mono).

**First run (cold):**
```
Device: Metal(MetalDevice(DeviceId(1)))
Input: 2.96s (71040 samples, truncated to 37 frames)
Encoded in 4.127s (0.7x realtime)
Decoded in 0.349s (8.5x realtime)
```

**Second run (warmed up):**
```
Encoded in 0.324s (9.1x realtime)
Decoded in 0.267s (11.1x realtime)
```

The first run is slow due to Metal shader compilation. Subsequent runs
show the codec comfortably exceeds realtime on the M4 GPU.

---

## Phase 3: Android Cross-Compilation

### Challenge: Feature unification

Cargo's workspace feature resolver unifies features across all crates.
When `mimi-cli` enables the `metal` feature on `mimi-core`, that feature
propagates to `mimi-jni` even though `mimi-jni` specifies
`default-features = false`. This causes `candle-core` to pull in `objc2`
(Apple Objective-C bindings), which fails to compile for Android targets.

**Solution:** Excluded `mimi-jni` from the workspace and gave it a
standalone `Cargo.toml` with explicit dependency versions (not workspace
references). This breaks the feature unification chain:

```toml
# mimi-jni/Cargo.toml (standalone, not workspace member)
[dependencies]
candle = { version = "0.9.1", package = "candle-core", default-features = false }
candle-nn = { version = "0.9.1", default-features = false }
jni = "0.21"
mimi-core = { path = "../mimi-core", default-features = false }
```

The workspace `Cargo.toml` was updated:
```toml
exclude = ["mimi-jni"]
```

### Prerequisites installed

1. **Android NDK** 27.0.12077973 — installed via `sdkmanager` (which
   required installing OpenJDK via Homebrew first, and the Android
   command-line tools via `brew install --cask android-commandlinetools`)

2. **Rust Android targets:**
   ```
   rustup target add aarch64-linux-android   (ARM64 phones)
   rustup target add x86_64-linux-android    (emulator)
   ```

3. **cargo-ndk** 4.1.2 — Cargo plugin that configures NDK toolchains
   for cross-compilation.

### API level flag

cargo-ndk v4 changed the API level flag from `-p` (lowercase) to `-P`
(uppercase). The lowercase `-p` now means `--package`, matching Cargo's
convention. First build attempt failed with a panic because `-p 24` was
interpreted as "package named 24". Fixed by using `-P 24` or omitting it
(defaults to API 21).

### Build commands

```bash
export ANDROID_NDK_HOME="$HOME/Library/Android/sdk/ndk/27.0.12077973"

# ARM64 (production phones)
cd ~/dev/mimi-codec/mimi-jni
cargo ndk -t arm64-v8a build --release

# x86_64 (emulator)
cargo ndk -t x86_64 build --release
```

Both compiled cleanly after the workspace exclusion fix.

### JNI bridge design

`mimi-jni/src/lib.rs` exposes 5 functions to Kotlin via JNI:

| Function | Purpose |
|----------|---------|
| `nativeCreate(modelPath: String, numCodebooks: Int): Long` | Load model from file, return opaque handle |
| `nativeEncodeStep(handle: Long, pcm: FloatArray): IntArray?` | Encode 1920 samples → token array |
| `nativeDecodeStep(handle: Long, codes: IntArray): FloatArray?` | Decode token array → 1920 samples |
| `nativeReset(handle: Long)` | Clear KV caches between utterances |
| `nativeDestroy(handle: Long)` | Free the model (prevent memory leak) |

The handle pattern: Rust allocates a `Mimi` struct on the heap with
`Box::new()`, converts to a raw pointer with `Box::into_raw()`, and
returns it as a `jlong`. Kotlin holds this as an opaque `Long`. On
subsequent calls, the `Long` is cast back to a `*mut Mimi` reference.
`nativeDestroy` converts it back to a `Box` which is dropped, freeing
memory.

Token layout is time-major: for 8 codebooks, `encodeStep` returns an
`IntArray` of length `8 * T` where each group of 8 ints represents one
80ms frame's codebook values.

### Output

```
android-libs/
  arm64-v8a/libmimi_jni.so    5.4 MB
  x86_64/libmimi_jni.so       5.8 MB
```

All 5 JNI symbols verified as exported via `llvm-nm`:
```
Java_com_example_mimi_MimiCodec_nativeCreate
Java_com_example_mimi_MimiCodec_nativeDecodeStep
Java_com_example_mimi_MimiCodec_nativeDestroy
Java_com_example_mimi_MimiCodec_nativeEncodeStep
Java_com_example_mimi_MimiCodec_nativeReset
```

Binary confirmed as `ELF 64-bit LSB shared object, ARM aarch64`.

---

## Final Project Layout

```
~/dev/mimi-codec/
  Cargo.toml                    workspace (excludes mimi-jni)
  Cargo.lock
  .gitignore
  ANDROID_PORT_PLAN.md          detailed Android integration guide
  REPORT.md                     this document
  models/
    tokenizer-e351c8d8-checkpoint125.safetensors   (367 MB, gitignored)
  mimi-core/                    codec library
    Cargo.toml                  (default feature: metal)
    src/
      lib.rs                    module exports + NormType enum
      mimi.rs                   main codec struct
      seanet.rs                 convolutional encoder/decoder
      quantization.rs           vector quantizer
      transformer.rs            transformer (modified: removed Batched variant)
      streaming.rs              streaming primitives
      conv.rs                   convolution layers
      kv_cache.rs               KV cache
      nn.rs                     quantized/unquantized linear layers
      wav.rs                    WAV I/O
  mimi-cli/                     CLI tool
    Cargo.toml
    src/main.rs                 roundtrip/encode/decode subcommands
  mimi-jni/                     Android JNI bridge (standalone, excluded from workspace)
    Cargo.toml                  (no default features — CPU only)
    src/lib.rs                  5 JNI functions
    target/
      aarch64-linux-android/release/libmimi_jni.so
      x86_64-linux-android/release/libmimi_jni.so
  mimi-pyo3/                    Python bindings
    Cargo.toml
    src/lib.rs
  android-libs/                 staged .so files for Android project
    arm64-v8a/libmimi_jni.so
    x86_64/libmimi_jni.so
```

---

## Codec Specifications

| Parameter | Value |
|-----------|-------|
| Sample rate | 24,000 Hz |
| Frame size | 1,920 samples (80 ms) |
| Frame rate | 12.5 Hz |
| Codebooks | 1–32 (configurable, default 8) |
| Codebook size | 2,048 entries per codebook |
| Bitrate at 8 codebooks | ~1,100 bps |
| Bitrate at 4 codebooks | ~550 bps |
| Model size (f32) | 367 MB |
| Native library size | 5.4 MB (ARM64) |
| Encode speed (M4 Metal) | 9.1x realtime |
| Decode speed (M4 Metal) | 11.1x realtime |
| Streaming latency | 80 ms (one frame) |

---

## Issues Encountered and Resolutions

| Issue | Cause | Resolution |
|-------|-------|------------|
| `batched_transformer` not found | Copied Mimi files reference Moshi-LM module | Replaced `Transformer` enum with struct, removed Batched variant |
| Nested Vec type mismatches in CLI | `to_vec3` returns 3 levels, collected into 4 | Fixed type signatures to `Vec<Vec<Vec<Vec<u32>>>>` throughout |
| `objc2` fails on Android target | Cargo workspace feature unification enables Metal on all crates | Excluded `mimi-jni` from workspace, standalone Cargo.toml with `default-features = false` |
| `cargo ndk -p 24` panics | cargo-ndk v4 changed `-p` to mean `--package`, not platform | Use `-P` (uppercase) for API level |
| `sdkmanager` requires Java | Android SDK tools need JDK | Installed OpenJDK via `brew install openjdk` |
| `huggingface-cli` not in PATH | pip installed to Python 3.14 system without PATH setup | Used `hf_hub_download()` directly from Python instead |
