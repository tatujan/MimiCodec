// Mimi JNI Bridge — exposes mimi-core (and optionally mimi-onnx) to Android/Kotlin via JNI
//
// Kotlin class: com.example.mimi.MimiCodec
// Functions:
//   nativeCreate(modelPath: String, numCodebooks: Int): Long
//   nativeCreateOnnx(encoderPath: String, decoderPath: String, numCodebooks: Int, useNnapi: Boolean, streaming: Boolean): Long
//   nativeEncode(handle: Long, pcm: FloatArray): IntArray
//   nativeDecode(handle: Long, codes: IntArray, numCodebooks: Int): FloatArray
//   nativeEncodeStep(handle: Long, pcm: FloatArray): IntArray?
//   nativeDecodeStep(handle: Long, codes: IntArray): FloatArray?
//   nativeGetTimings(handle: Long): FloatArray
//   nativeResetTimings(handle: Long)
//   nativeReset(handle: Long)
//   nativeDestroy(handle: Long)

use jni::objects::{JClass, JFloatArray, JIntArray, JObject, JString};
use jni::sys::{jboolean, jlong};
use jni::JNIEnv;

use candle::{DType, Device, Tensor};
use mimi_core::mimi::Mimi;
use mimi_core::streaming::StreamTensor;

// ─── Backend enum ────────────────────────────────────────────────────────────

enum CodecBackend {
    Candle(Mimi),
    #[cfg(feature = "onnx")]
    Onnx(mimi_onnx::OnnxMimiCodec),
}

// ─── Create (Candle) ─────────────────────────────────────────────────────────

/// Load model from a file path on disk (safetensors or GGUF), return an opaque handle.
#[no_mangle]
pub extern "system" fn Java_com_example_mimi_MimiCodec_nativeCreate(
    mut env: JNIEnv,
    _class: JClass,
    model_path: JString,
    num_codebooks: jni::sys::jint,
) -> jlong {
    let result = (|| -> Result<jlong, String> {
        let path: String = env
            .get_string(&model_path)
            .map_err(|e| format!("Failed to read model path string: {e}"))?
            .into();

        let num_threads = candle::utils::get_num_threads();
        let has_neon = cfg!(target_feature = "neon");
        let dotprod_runtime = candle::quantized::has_dotprod();
        eprintln!("[mimi-jni] backend=candle, threads={num_threads}, neon={has_neon}, dotprod_runtime={dotprod_runtime}, model={path}");

        let device = Device::Cpu;
        let mimi = mimi_core::mimi::load(&path, Some(num_codebooks as usize), &device)
            .map_err(|e| format!("Failed to load model from '{path}': {e}"))?;

        let backend = CodecBackend::Candle(mimi);
        let boxed = Box::new(backend);
        Ok(Box::into_raw(boxed) as jlong)
    })();

    match result {
        Ok(handle) => handle,
        Err(msg) => {
            let _ = env.throw_new("java/lang/RuntimeException", &msg);
            0
        }
    }
}

// ─── Create (ONNX) ──────────────────────────────────────────────────────────

/// Load ONNX encoder + decoder models, return an opaque handle.
///
/// Kotlin signature:
///   external fun nativeCreateOnnx(encoderPath: String, decoderPath: String, numCodebooks: Int, useNnapi: Boolean, streaming: Boolean): Long
#[cfg(feature = "onnx")]
#[no_mangle]
pub extern "system" fn Java_com_example_mimi_MimiCodec_nativeCreateOnnx(
    mut env: JNIEnv,
    _class: JClass,
    encoder_path: JString,
    decoder_path: JString,
    num_codebooks: jni::sys::jint,
    use_nnapi: jboolean,
    streaming: jboolean,
) -> jlong {
    let result = (|| -> Result<jlong, String> {
        let enc_path: String = env
            .get_string(&encoder_path)
            .map_err(|e| format!("Failed to read encoder path: {e}"))?
            .into();
        let dec_path: String = env
            .get_string(&decoder_path)
            .map_err(|e| format!("Failed to read decoder path: {e}"))?
            .into();

        let streaming = streaming != 0;
        eprintln!(
            "[mimi-jni] backend=onnx, nnapi={}, streaming={streaming}, encoder={enc_path}, decoder={dec_path}",
            use_nnapi != 0
        );

        let codec = mimi_onnx::OnnxMimiCodec::new(
            &enc_path,
            &dec_path,
            num_codebooks as usize,
            use_nnapi != 0,
            streaming,
        )
        .map_err(|e| format!("Failed to create ONNX codec: {e}"))?;

        let backend = CodecBackend::Onnx(codec);
        let boxed = Box::new(backend);
        Ok(Box::into_raw(boxed) as jlong)
    })();

    match result {
        Ok(handle) => handle,
        Err(msg) => {
            let _ = env.throw_new("java/lang/RuntimeException", &msg);
            0
        }
    }
}

// ─── Batch encode ────────────────────────────────────────────────────────────

/// Batch-encode an entire audio buffer at once.
/// Returns codes as a flat int array: [codebooks * time_steps], time-major layout.
#[no_mangle]
pub extern "system" fn Java_com_example_mimi_MimiCodec_nativeEncode<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    handle: jlong,
    pcm: JFloatArray<'local>,
) -> JObject<'local> {
    let result = (|| -> Result<Vec<i32>, String> {
        let backend = unsafe { &mut *(handle as *mut CodecBackend) };

        let len = env
            .get_array_length(&pcm)
            .map_err(|e| format!("Failed to get pcm array length: {e}"))? as usize;
        let mut pcm_buf = vec![0f32; len];
        env.get_float_array_region(&pcm, 0, &mut pcm_buf)
            .map_err(|e| format!("Failed to read pcm data: {e}"))?;

        let codes = match backend {
            CodecBackend::Candle(mimi) => {
                let device = Device::Cpu;
                let tensor = Tensor::from_slice(&pcm_buf, (1, 1, len), &device)
                    .map_err(|e| format!("Failed to create tensor: {e}"))?;
                let codes = mimi
                    .encode(&tensor)
                    .map_err(|e| format!("encode failed: {e}"))?;
                let codes_3d = codes
                    .to_vec3::<u32>()
                    .map_err(|e| format!("Failed to extract codes: {e}"))?;
                codes_3d[0].clone()
            }
            #[cfg(feature = "onnx")]
            CodecBackend::Onnx(onnx) => onnx
                .encode(&pcm_buf)
                .map_err(|e| format!("ONNX encode failed: {e}"))?,

        };

        let n_codebooks = codes.len();
        let n_time = if n_codebooks > 0 { codes[0].len() } else { 0 };

        // Flatten time-major: [t0_c0, t0_c1, ..., t1_c0, ...]
        let mut flat = Vec::with_capacity(n_codebooks * n_time);
        for t in 0..n_time {
            for cb in 0..n_codebooks {
                flat.push(codes[cb][t] as i32);
            }
        }
        Ok(flat)
    })();

    match result {
        Ok(codes) => match env.new_int_array(codes.len() as i32) {
            Ok(arr) => {
                if env.set_int_array_region(&arr, 0, &codes).is_err() {
                    JObject::null()
                } else {
                    JObject::from(arr)
                }
            }
            Err(_) => JObject::null(),
        },
        Err(msg) => {
            let _ = env.throw_new("java/lang/RuntimeException", &msg);
            JObject::null()
        }
    }
}

// ─── Batch decode ────────────────────────────────────────────────────────────

/// Batch-decode an entire code sequence at once.
/// Input codes: flat int array [codebooks * time_steps], time-major layout.
#[no_mangle]
pub extern "system" fn Java_com_example_mimi_MimiCodec_nativeDecode<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    handle: jlong,
    codes: JIntArray<'local>,
    num_codebooks: jni::sys::jint,
) -> JObject<'local> {
    let result = (|| -> Result<Vec<f32>, String> {
        let backend = unsafe { &mut *(handle as *mut CodecBackend) };
        let num_codebooks = num_codebooks as usize;

        let len = env
            .get_array_length(&codes)
            .map_err(|e| format!("Failed to get codes array length: {e}"))? as usize;
        if len == 0 || len % num_codebooks != 0 {
            return Err(format!(
                "Codes length {len} must be a positive multiple of {num_codebooks} codebooks"
            ));
        }
        let n_time = len / num_codebooks;

        let mut codes_buf = vec![0i32; len];
        env.get_int_array_region(&codes, 0, &mut codes_buf)
            .map_err(|e| format!("Failed to read codes data: {e}"))?;

        // Rebuild [codebooks][time] from time-major flat layout
        let mut codes_2d: Vec<Vec<u32>> = vec![vec![0u32; n_time]; num_codebooks];
        for t in 0..n_time {
            for cb in 0..num_codebooks {
                codes_2d[cb][t] = codes_buf[t * num_codebooks + cb] as u32;
            }
        }

        match backend {
            CodecBackend::Candle(mimi) => {
                let device = Device::Cpu;
                let codes_tensor = Tensor::new(codes_2d, &device)
                    .map_err(|e| format!("Failed to create codes tensor: {e}"))?
                    .unsqueeze(0)
                    .map_err(|e| format!("Failed to unsqueeze: {e}"))?;
                let pcm = mimi
                    .decode(&codes_tensor)
                    .map_err(|e| format!("decode failed: {e}"))?;
                let pcm_3d = pcm
                    .to_dtype(DType::F32)
                    .map_err(|e| format!("dtype conversion: {e}"))?
                    .to_vec3::<f32>()
                    .map_err(|e| format!("Failed to extract pcm: {e}"))?;
                Ok(pcm_3d[0][0].clone())
            }
            #[cfg(feature = "onnx")]
            CodecBackend::Onnx(onnx) => onnx
                .decode(&codes_2d)
                .map_err(|e| format!("ONNX decode failed: {e}")),
        }
    })();

    match result {
        Ok(pcm) => match env.new_float_array(pcm.len() as i32) {
            Ok(arr) => {
                if env.set_float_array_region(&arr, 0, &pcm).is_err() {
                    JObject::null()
                } else {
                    JObject::from(arr)
                }
            }
            Err(_) => JObject::null(),
        },
        Err(msg) => {
            let _ = env.throw_new("java/lang/RuntimeException", &msg);
            JObject::null()
        }
    }
}

// ─── Streaming encode step (Candle only) ─────────────────────────────────────

/// Encode one frame of audio (1920 f32 samples) into codec tokens.
/// Returns null if the codec hasn't produced output yet (buffering).
/// Only supported with Candle backend; throws RuntimeException for ONNX.
#[no_mangle]
pub extern "system" fn Java_com_example_mimi_MimiCodec_nativeEncodeStep<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    handle: jlong,
    pcm: JFloatArray<'local>,
) -> JObject<'local> {
    let result = (|| -> Result<Option<Vec<i32>>, String> {
        let backend = unsafe { &mut *(handle as *mut CodecBackend) };

        let len = env
            .get_array_length(&pcm)
            .map_err(|e| format!("Failed to get pcm array length: {e}"))? as usize;
        if len == 0 {
            return Err("PCM array must not be empty".to_string());
        }

        let mut pcm_buf = vec![0f32; len];
        env.get_float_array_region(&pcm, 0, &mut pcm_buf)
            .map_err(|e| format!("Failed to read pcm data: {e}"))?;

        let codes_2d = match backend {
            CodecBackend::Candle(mimi) => {
                let device = Device::Cpu;
                let tensor = Tensor::from_slice(&pcm_buf, (1, 1, len), &device)
                    .map_err(|e| format!("Failed to create tensor: {e}"))?;
                let codes = mimi
                    .encode_step(&StreamTensor::from_tensor(tensor), &().into())
                    .map_err(|e| format!("encode_step failed: {e}"))?;
                match codes.as_option() {
                    Some(codes_tensor) => {
                        let codes_3d = codes_tensor
                            .to_vec3::<u32>()
                            .map_err(|e| format!("Failed to extract codes: {e}"))?;
                        Some(codes_3d[0].clone())
                    }
                    None => None,
                }
            }
            #[cfg(feature = "onnx")]
            CodecBackend::Onnx(onnx) => onnx
                .encode_step(&pcm_buf)
                .map_err(|e| format!("ONNX encode_step failed: {e}"))?,
        };

        match codes_2d {
            Some(batch) => {
                let n_codebooks = batch.len();
                let n_time = if n_codebooks > 0 { batch[0].len() } else { 0 };
                let mut flat = Vec::with_capacity(n_codebooks * n_time);
                for t in 0..n_time {
                    for cb in 0..n_codebooks {
                        flat.push(batch[cb][t] as i32);
                    }
                }
                Ok(Some(flat))
            }
            None => Ok(None),
        }
    })();

    match result {
        Ok(Some(codes)) => match env.new_int_array(codes.len() as i32) {
            Ok(arr) => {
                if env.set_int_array_region(&arr, 0, &codes).is_err() {
                    JObject::null()
                } else {
                    JObject::from(arr)
                }
            }
            Err(_) => JObject::null(),
        },
        Ok(None) => JObject::null(),
        Err(msg) => {
            let _ = env.throw_new("java/lang/RuntimeException", &msg);
            JObject::null()
        }
    }
}

// ─── Streaming decode step (Candle only) ─────────────────────────────────────

/// Decode codec tokens back into one frame of audio (1920 f32 samples).
/// Returns null if the codec hasn't produced output yet (buffering).
/// Only supported with Candle backend; throws RuntimeException for ONNX.
#[no_mangle]
pub extern "system" fn Java_com_example_mimi_MimiCodec_nativeDecodeStep<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    handle: jlong,
    codes: JIntArray<'local>,
) -> JObject<'local> {
    let result = (|| -> Result<Option<Vec<f32>>, String> {
        let backend = unsafe { &mut *(handle as *mut CodecBackend) };

        let len = env
            .get_array_length(&codes)
            .map_err(|e| format!("Failed to get codes array length: {e}"))? as usize;
        if len == 0 {
            return Err("Codes array must not be empty".to_string());
        }

        let mut codes_buf = vec![0i32; len];
        env.get_int_array_region(&codes, 0, &mut codes_buf)
            .map_err(|e| format!("Failed to read codes data: {e}"))?;

        match backend {
            CodecBackend::Candle(mimi) => {
                let device = Device::Cpu;
                let num_codebooks = mimi.config().quantizer_n_q;
                if len % num_codebooks != 0 {
                    return Err(format!(
                        "Codes length {len} must be a positive multiple of {num_codebooks} codebooks"
                    ));
                }
                let n_time = len / num_codebooks;

                let mut codes_2d: Vec<Vec<u32>> = vec![vec![0u32; n_time]; num_codebooks];
                for t in 0..n_time {
                    for cb in 0..num_codebooks {
                        codes_2d[cb][t] = codes_buf[t * num_codebooks + cb] as u32;
                    }
                }

                let codes_tensor = Tensor::new(codes_2d, &device)
                    .map_err(|e| format!("Failed to create codes tensor: {e}"))?
                    .unsqueeze(0)
                    .map_err(|e| format!("Failed to unsqueeze: {e}"))?;

                let pcm = mimi
                    .decode_step(&StreamTensor::from_tensor(codes_tensor), &().into())
                    .map_err(|e| format!("decode_step failed: {e}"))?;

                match pcm.as_option() {
                    Some(pcm_tensor) => {
                        let pcm_3d = pcm_tensor
                            .to_dtype(DType::F32)
                            .map_err(|e| format!("dtype conversion: {e}"))?
                            .to_vec3::<f32>()
                            .map_err(|e| format!("Failed to extract pcm: {e}"))?;
                        Ok(Some(pcm_3d[0][0].clone()))
                    }
                    None => Ok(None),
                }
            }
            #[cfg(feature = "onnx")]
            CodecBackend::Onnx(onnx) => {
                let num_codebooks = onnx.num_codebooks();
                if len % num_codebooks != 0 {
                    return Err(format!(
                        "Codes length {len} must be a positive multiple of {num_codebooks} codebooks"
                    ));
                }
                let n_time = len / num_codebooks;

                let mut codes_2d: Vec<Vec<u32>> = vec![vec![0u32; n_time]; num_codebooks];
                for t in 0..n_time {
                    for cb in 0..num_codebooks {
                        codes_2d[cb][t] = codes_buf[t * num_codebooks + cb] as u32;
                    }
                }

                onnx.decode_step(&codes_2d)
                    .map_err(|e| format!("ONNX decode_step failed: {e}"))
            }
        }
    })();

    match result {
        Ok(Some(pcm)) => match env.new_float_array(pcm.len() as i32) {
            Ok(arr) => {
                if env.set_float_array_region(&arr, 0, &pcm).is_err() {
                    JObject::null()
                } else {
                    JObject::from(arr)
                }
            }
            Err(_) => JObject::null(),
        },
        Ok(None) => JObject::null(),
        Err(msg) => {
            let _ = env.throw_new("java/lang/RuntimeException", &msg);
            JObject::null()
        }
    }
}

// ─── Timings (Candle only) ───────────────────────────────────────────────────

/// Get per-component timing breakdown as a float array.
/// Returns [seanet_enc, enc_transformer, downsample, quant_enc,
///          quant_dec, upsample, dec_transformer, seanet_dec, steps]
/// Returns null for ONNX backend.
#[no_mangle]
pub extern "system" fn Java_com_example_mimi_MimiCodec_nativeGetTimings<'local>(
    env: JNIEnv<'local>,
    _class: JClass<'local>,
    handle: jlong,
) -> JObject<'local> {
    let backend = unsafe { &*(handle as *const CodecBackend) };
    let mimi = match backend {
        CodecBackend::Candle(m) => m,
        #[cfg(feature = "onnx")]
        CodecBackend::Onnx(_) => return JObject::null(),
    };
    let t = mimi.timings();
    let data: [f32; 9] = [
        t.seanet_encode as f32,
        t.encoder_transformer as f32,
        t.downsample as f32,
        t.quantizer_encode as f32,
        t.quantizer_decode as f32,
        t.upsample as f32,
        t.decoder_transformer as f32,
        t.seanet_decode as f32,
        t.steps as f32,
    ];
    match env.new_float_array(9) {
        Ok(arr) => {
            if env.set_float_array_region(&arr, 0, &data).is_err() {
                JObject::null()
            } else {
                JObject::from(arr)
            }
        }
        Err(_) => JObject::null(),
    }
}

/// Reset accumulated timings (Candle only, no-op for ONNX).
#[no_mangle]
pub extern "system" fn Java_com_example_mimi_MimiCodec_nativeResetTimings(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    let backend = unsafe { &mut *(handle as *mut CodecBackend) };
    if let CodecBackend::Candle(mimi) = backend {
        mimi.reset_timings();
    }
}

// ─── Reset & Destroy ─────────────────────────────────────────────────────────

/// Reset the codec's internal state (KV caches, streaming buffers).
/// Candle only; no-op for ONNX (stateless).
#[no_mangle]
pub extern "system" fn Java_com_example_mimi_MimiCodec_nativeReset(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    let backend = unsafe { &mut *(handle as *mut CodecBackend) };
    match backend {
        CodecBackend::Candle(mimi) => mimi.reset_state(),
        #[cfg(feature = "onnx")]
        CodecBackend::Onnx(onnx) => onnx.reset_state(),
    }
}

/// Free the codec instance.
#[no_mangle]
pub extern "system" fn Java_com_example_mimi_MimiCodec_nativeDestroy(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    if handle != 0 {
        unsafe {
            let _ = Box::from_raw(handle as *mut CodecBackend);
        }
    }
}
