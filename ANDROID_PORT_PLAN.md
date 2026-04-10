# Porting Mimi Codec to Android — Detailed Plan

## Big Picture

We have a working Rust audio codec (`mimi-core`) that compresses speech to ~1.1 kbps.
The goal: run it on Android for **real-time streaming** — mic input goes in, compressed
tokens come out (and vice versa).

The architecture looks like this:

```
┌─────────────────────────────────────────────────┐
│  Android App (Kotlin)                           │
│                                                 │
│  ┌──────────┐   ┌──────────┐   ┌────────────┐  │
│  │ AudioRec │──▶│ MimiCodec│──▶│ Network /  │  │
│  │ (mic)    │   │ (Kotlin) │   │ Storage    │  │
│  └──────────┘   └────┬─────┘   └────────────┘  │
│                      │ JNI                      │
│  ┌───────────────────┴──────────────────────┐   │
│  │  libmimi_jni.so  (Rust, compiled for     │   │
│  │  Android ARM64)                          │   │
│  │  ┌─────────────────────────────────┐     │   │
│  │  │  mimi-core (the codec engine)   │     │   │
│  │  └─────────────────────────────────┘     │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

**In plain English:** Your Kotlin app captures audio from the mic, passes raw PCM
samples down to a Rust native library via JNI (Java Native Interface), and gets back
compressed tokens (or vice versa for playback). The Rust code does all the heavy
neural network math. Your Kotlin code handles the Android UI, audio I/O, and networking.

---

## Concepts You Need to Know

### What is JNI?
JNI (Java Native Interface) is how Android/Java/Kotlin code calls native C/C++/Rust
code. You've probably seen it with NDK libraries. It works like this:

```kotlin
// Kotlin side — declare a native method
external fun encodeAudio(pcmData: FloatArray): IntArray

// Load the shared library in a companion object
companion object {
    init { System.loadLibrary("mimi_jni") }
}
```

```rust
// Rust side — implement the function with a matching JNI signature
#[no_mangle]
pub extern "C" fn Java_com_example_mimi_MimiCodec_encodeAudio(
    env: JNIEnv, _: JClass, pcm_data: JFloatArray
) -> JIntArray { ... }
```

Think of it like a `suspend fun` bridge — Kotlin calls a function name, JNI routes it
to the native `.so` library, Rust does the work, and returns the result.

### What is `cargo-ndk`?
A Cargo plugin that cross-compiles Rust code for Android. It handles all the NDK
toolchain setup. You run `cargo ndk -t arm64-v8a build --release` and get an
`libmimi_jni.so` that runs on Android ARM64 devices.

### What is a `.so` file?
A shared library — the Linux/Android equivalent of a `.dll` on Windows. Your APK
bundles this file in `jniLibs/arm64-v8a/` and Android loads it at runtime.

---

## Step-by-Step Plan

### Phase 1: Rust Side — Create JNI Bindings

**Goal:** Create a new Rust crate `mimi-jni` that wraps `mimi-core` and exposes
functions callable from Kotlin via JNI.

#### 1.1 Install Android cross-compilation tools

```bash
# Install the Android NDK (if not already via Android Studio)
# Android Studio → SDK Manager → SDK Tools → NDK (Side by side)

# Install Rust Android targets
rustup target add aarch64-linux-android    # ARM64 (99% of modern phones)
rustup target add armv7-linux-androideabi  # ARMv7 (older phones, optional)
rustup target add x86_64-linux-android     # x86_64 (emulator)

# Install cargo-ndk (handles NDK toolchain linking)
cargo install cargo-ndk
```

#### 1.2 Create `mimi-jni` crate

New crate in the workspace:

```
mimi-jni/
  Cargo.toml
  src/
    lib.rs       ← JNI bridge functions
```

**Cargo.toml:**
```toml
[package]
name = "mimi-jni"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]   # Produces libmimi_jni.so

[dependencies]
mimi-core = { path = "../mimi-core", default-features = false }  # no Metal!
jni = "0.21"               # JNI bindings for Rust
```

Key detail: `default-features = false` disables Metal (macOS-only). On Android,
the codec runs on CPU. This is fine — we measured 9-11x realtime on Mac CPU,
and modern Android chips (Snapdragon 8 Gen 2/3) are comparable.

#### 1.3 Write the JNI bridge (`mimi-jni/src/lib.rs`)

This file is the "translator" between Kotlin and Rust. It needs to expose
these functions:

| JNI Function | What it does |
|---|---|
| `createCodec(modelBytes: ByteArray, numCodebooks: Int): Long` | Load model, return a handle (pointer) |
| `encodeStep(handle: Long, pcmData: FloatArray): IntArray?` | Encode 80ms of audio → tokens |
| `decodeStep(handle: Long, codes: IntArray): FloatArray?` | Decode tokens → 80ms of audio |
| `resetState(handle: Long)` | Clear internal buffers (between utterances) |
| `destroyCodec(handle: Long)` | Free memory |

The `Long` handle is a pattern you may have seen in Android NDK code — it's a raw
pointer cast to a 64-bit integer. Kotlin holds onto it as an opaque handle.

```rust
use jni::JNIEnv;
use jni::objects::{JByteArray, JClass, JFloatArray, JIntArray};
use jni::sys::jlong;
use mimi_core::mimi::{Config, Mimi};
use candle::{Device, DType, Tensor};
use candle_nn::VarBuilder;

// Called from Kotlin: MimiCodec.createCodec(modelBytes, numCodebooks)
#[no_mangle]
pub extern "C" fn Java_com_example_mimi_MimiCodec_createCodec(
    mut env: JNIEnv,
    _class: JClass,
    model_bytes: JByteArray,
    num_codebooks: jni::sys::jint,
) -> jlong {
    // 1. Copy the model bytes from Java heap
    // 2. Load safetensors from bytes
    // 3. Create Mimi instance
    // 4. Box it and return as raw pointer (jlong)
    // ... (implementation details)
}
```

The pattern: Rust allocates the `Mimi` struct on the heap with `Box::new()`,
converts it to a raw pointer with `Box::into_raw()`, and passes the pointer
as a `Long` to Kotlin. When Kotlin calls other functions, it passes the `Long`
back, and Rust converts it back to a reference. Think of it like an object ID.

#### 1.4 Model loading strategy

The model is 367MB. Options for Android:

| Strategy | Where model lives | Pros | Cons |
|---|---|---|---|
| **Bundle in APK assets** | `assets/` folder | Simple, offline | +367MB APK size |
| **Download on first launch** | App internal storage | Small APK | Needs network, progress UI |
| **Split APK / Play Asset Delivery** | Google Play handles it | Best UX | More setup |

**Recommendation:** Download on first launch. Store in `context.filesDir`.
The JNI function takes a file path string instead of byte array:

```kotlin
val modelPath = File(context.filesDir, "mimi_model.safetensors").absolutePath
val handle = MimiCodec.createCodecFromFile(modelPath, numCodebooks = 8)
```

---

### Phase 2: Build for Android

#### 2.1 Set NDK path

```bash
# Find your NDK path (Android Studio installs it here)
export ANDROID_NDK_HOME="$HOME/Library/Android/sdk/ndk/27.0.12077973"
# (adjust version number to what you have installed)
```

#### 2.2 Cross-compile

```bash
cd ~/dev/mimi-codec

# Build for ARM64 (production phones)
cargo ndk -t arm64-v8a -p 24 build --release -p mimi-jni

# The output .so file will be at:
# target/aarch64-linux-android/release/libmimi_jni.so

# For emulator testing:
cargo ndk -t x86_64 -p 24 build --release -p mimi-jni
```

The `-p 24` sets the minimum Android API level (Android 7.0).

#### 2.3 Copy `.so` to your Android project

```
YourApp/
  app/
    src/
      main/
        jniLibs/
          arm64-v8a/
            libmimi_jni.so     ← copy here
          x86_64/
            libmimi_jni.so     ← for emulator
```

Or automate it in `build.gradle.kts`:
```kotlin
android {
    sourceSets {
        getByName("main") {
            jniLibs.srcDirs("src/main/jniLibs")
        }
    }
}
```

---

### Phase 3: Kotlin Side — Android App

#### 3.1 Create the Kotlin wrapper class

```kotlin
// MimiCodec.kt — thin wrapper around native functions
class MimiCodec private constructor(private var handle: Long) : AutoCloseable {

    companion object {
        init { System.loadLibrary("mimi_jni") }

        fun create(modelPath: String, numCodebooks: Int = 8): MimiCodec {
            val handle = nativeCreate(modelPath, numCodebooks)
            if (handle == 0L) throw RuntimeException("Failed to load Mimi model")
            return MimiCodec(handle)
        }

        @JvmStatic private external fun nativeCreate(modelPath: String, numCodebooks: Int): Long
        @JvmStatic private external fun nativeEncodeStep(handle: Long, pcm: FloatArray): IntArray?
        @JvmStatic private external fun nativeDecodeStep(handle: Long, codes: IntArray): FloatArray?
        @JvmStatic private external fun nativeReset(handle: Long)
        @JvmStatic private external fun nativeDestroy(handle: Long)
    }

    /** Encode 80ms of audio (1920 float samples at 24kHz) to token array */
    fun encodeStep(pcmChunk: FloatArray): IntArray? {
        require(pcmChunk.size == 1920) { "Must provide exactly 1920 samples (80ms at 24kHz)" }
        return nativeEncodeStep(handle, pcmChunk)
    }

    /** Decode a token array back to 80ms of audio */
    fun decodeStep(codes: IntArray): FloatArray? {
        return nativeDecodeStep(handle, codes)
    }

    fun resetState() = nativeReset(handle)

    override fun close() {
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0L
        }
    }
}
```

#### 3.2 Audio capture (mic → encoder)

```kotlin
// Use AudioRecord for low-latency mic capture at 24kHz
class AudioEncoder(private val codec: MimiCodec) {

    private val sampleRate = 24000
    private val frameSize = 1920  // 80ms at 24kHz
    private var recording = false

    fun start(onTokens: (IntArray) -> Unit) {
        recording = true
        thread(name = "mimi-encoder") {
            val bufferSize = AudioRecord.getMinBufferSize(
                sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT
            )
            val recorder = AudioRecord(
                MediaRecorder.AudioSource.MIC, sampleRate,
                AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT,
                maxOf(bufferSize, frameSize * 4)  // float = 4 bytes
            )

            val buffer = FloatArray(frameSize)
            recorder.startRecording()

            while (recording) {
                val read = recorder.read(buffer, 0, frameSize, AudioRecord.READ_BLOCKING)
                if (read == frameSize) {
                    val tokens = codec.encodeStep(buffer)
                    if (tokens != null) onTokens(tokens)
                }
            }
            recorder.stop()
            recorder.release()
        }
    }

    fun stop() { recording = false }
}
```

#### 3.3 Audio playback (decoder → speaker)

```kotlin
class AudioDecoder(private val codec: MimiCodec) {

    private val sampleRate = 24000
    private val frameSize = 1920

    fun start(tokenSource: () -> IntArray?) {
        thread(name = "mimi-decoder") {
            val track = AudioTrack.Builder()
                .setAudioAttributes(AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .build())
                .setAudioFormat(AudioFormat.Builder()
                    .setSampleRate(sampleRate)
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build())
                .setBufferSizeInBytes(frameSize * 4 * 4)
                .setTransferMode(AudioTrack.MODE_STREAM)
                .build()

            track.play()
            while (true) {
                val tokens = tokenSource() ?: break
                val pcm = codec.decodeStep(tokens)
                if (pcm != null) {
                    track.write(pcm, 0, pcm.size, AudioTrack.WRITE_BLOCKING)
                }
            }
            track.stop()
            track.release()
        }
    }
}
```

#### 3.4 Streaming over network

The encoded tokens are tiny — 8 integers per 80ms frame. That's 32 bytes per frame,
or **400 bytes/second**. You can send these over:

- **WebSocket** — simplest for real-time bidirectional
- **UDP** — lowest latency, but needs packet ordering
- **WebRTC DataChannel** — if you want NAT traversal

```kotlin
// Example: send tokens over WebSocket
fun onTokensEncoded(tokens: IntArray) {
    val buffer = ByteBuffer.allocate(tokens.size * 4)
    buffer.order(ByteOrder.LITTLE_ENDIAN)
    tokens.forEach { buffer.putInt(it) }
    webSocket.send(ByteString.of(buffer.array()))
}
```

---

### Phase 4: Performance & Optimization

#### 4.1 Expected performance on Android

| Device class | CPU | Expected speed |
|---|---|---|
| Flagship (Snapdragon 8 Gen 3) | Cortex-X4 | ~5-10x realtime |
| Mid-range (Snapdragon 7 Gen 1) | Cortex-A710 | ~3-5x realtime |
| Budget (Snapdragon 4 Gen 1) | Cortex-A78 | ~1.5-3x realtime |

Anything above 1x realtime means the codec can keep up with live audio.
You need headroom for the rest of the app, so **2x+ realtime** is the target.

#### 4.2 If performance is too slow

Options, in order of effort:

1. **Reduce codebooks** — 4 codebooks = half the compute, still intelligible
2. **Use f16 instead of f32** — ARM NEON has f16 support, ~1.5x speedup
3. **Quantize model weights** — `mimi-core` already supports GGML quantized
   weights via `MaybeQuantizedLinear` in `nn.rs`. INT8 quantization cuts
   model size to ~90MB and speeds up inference
4. **Use NNAPI/GPU delegate** — would require major rewrite, not recommended initially

#### 4.3 Threading model

```
Main Thread (UI)
    │
    ├── Encoder Thread
    │   └── AudioRecord → read 1920 samples → JNI encodeStep → send tokens
    │
    ├── Decoder Thread
    │   └── receive tokens → JNI decodeStep → AudioTrack.write
    │
    └── Network Thread
        └── WebSocket send/receive
```

Each JNI call (`encodeStep`/`decodeStep`) takes ~8-15ms on a flagship phone.
The frame interval is 80ms, so there's plenty of headroom.

---

## Full TODO List

### Setup (do once)
- [ ] Install Android NDK via Android Studio SDK Manager
- [ ] Install Rust Android targets (`rustup target add aarch64-linux-android x86_64-linux-android`)
- [ ] Install `cargo-ndk` (`cargo install cargo-ndk`)
- [ ] Set `ANDROID_NDK_HOME` environment variable

### Rust: JNI Crate (`mimi-jni/`)
- [ ] Create `mimi-jni/Cargo.toml` (cdylib, depends on mimi-core with no default features, jni crate)
- [ ] Implement `createCodecFromFile` — load safetensors from path, return handle
- [ ] Implement `encodeStep` — take FloatArray, return IntArray (codebook tokens)
- [ ] Implement `decodeStep` — take IntArray, return FloatArray (PCM samples)
- [ ] Implement `resetState` — clear internal KV cache
- [ ] Implement `destroyCodec` — free the Box'd Mimi
- [ ] Cross-compile: `cargo ndk -t arm64-v8a build --release -p mimi-jni`
- [ ] Cross-compile for emulator: `cargo ndk -t x86_64 build --release -p mimi-jni`
- [ ] Verify `.so` file size is reasonable (~30-50MB)

### Android App (Kotlin)
- [ ] Create new Android project (or add to existing)
- [ ] Copy `libmimi_jni.so` into `app/src/main/jniLibs/arm64-v8a/`
- [ ] Add model download logic (first-launch: download 367MB safetensors to filesDir)
- [ ] Create `MimiCodec.kt` wrapper class with native method declarations
- [ ] Create `AudioEncoder` — AudioRecord at 24kHz, feed 1920-sample chunks to codec
- [ ] Create `AudioDecoder` — decode tokens, write to AudioTrack
- [ ] Add RECORD_AUDIO permission to AndroidManifest.xml
- [ ] Build basic UI: Record button, playback button, status text
- [ ] Test roundtrip: record → encode → decode → playback on-device

### Stretch Goals
- [ ] Add WebSocket streaming (send tokens to a server or another phone)
- [ ] Add configurable codebook count (quality slider in UI)
- [ ] Benchmark on different devices, log encode/decode times
- [ ] Try INT8 quantized model weights for smaller download + faster inference
- [ ] Add Oboe audio library for even lower latency than AudioRecord/AudioTrack

---

## File Structure (Final)

```
~/dev/mimi-codec/                    ← Rust workspace (what we have now)
  mimi-core/                         ← The codec engine (no changes needed)
  mimi-cli/                          ← Mac testing tool (already working)
  mimi-jni/                          ← NEW: JNI bridge for Android
    Cargo.toml
    src/lib.rs
  mimi-pyo3/                         ← Python bindings (optional)

~/AndroidStudioProjects/MimiApp/     ← Android project (Kotlin)
  app/
    src/main/
      java/com/example/mimi/
        MimiCodec.kt                 ← Native method wrapper
        AudioEncoder.kt              ← Mic → encode pipeline
        AudioDecoder.kt              ← Decode → speaker pipeline
        MainActivity.kt              ← UI
      jniLibs/
        arm64-v8a/libmimi_jni.so     ← Compiled from mimi-jni
        x86_64/libmimi_jni.so        ← For emulator
      AndroidManifest.xml            ← RECORD_AUDIO permission
    assets/ or downloaded/
      mimi_model.safetensors         ← 367MB model weights
```

---

## Key Numbers to Remember

| What | Value |
|---|---|
| Audio sample rate | 24,000 Hz |
| Samples per frame | 1,920 (= 80ms) |
| Frame rate | 12.5 Hz (12-13 frames/second) |
| Tokens per frame | 8 (at 8 codebooks) |
| Token range | 0–2047 (fits in 16-bit int, sent as 32-bit) |
| Bitrate | ~1,100 bps at 8 codebooks |
| Network payload | ~32 bytes per frame = ~400 bytes/sec |
| Model size | 367 MB (f32), ~90 MB if quantized to INT8 |
| Latency budget | 80ms per frame + encode time + network |
