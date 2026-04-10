#!/usr/bin/env python3
"""Compare PyTorch vs ONNX Runtime output for Mimi codec.

Stage 1: Batch mode comparison (no streaming/KV cache).
Stage 2: Streaming mode comparison (frame-by-frame with KV cache).

Usage:
    python scripts/compare_backends.py --input test.wav --num-codebooks 8
    python scripts/compare_backends.py --input test.wav --num-codebooks 8 --streaming
"""

import argparse
import os

import numpy as np
import soundfile as sf
import torch

NUM_LAYERS = 8
NUM_HEADS = 8
HEAD_DIM = 64
FRAME_SIZE = 1920  # 80ms at 24kHz


def load_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 24000:
        raise ValueError(f"Expected 24kHz audio, got {sr}Hz")
    return audio


def compute_metrics(a: np.ndarray, b: np.ndarray, label: str):
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]
    diff = a - b
    max_abs = np.max(np.abs(diff))
    mse = np.mean(diff ** 2)
    rms_signal = np.sqrt(np.mean(a ** 2))
    rms_noise = np.sqrt(mse)
    snr = 20 * np.log10(rms_signal / rms_noise) if rms_noise > 0 else float("inf")
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    print(f"\n--- {label} ---")
    print(f"  Samples:     {min_len}")
    print(f"  Max abs diff: {max_abs:.8f}")
    print(f"  MSE:          {mse:.10f}")
    print(f"  SNR:          {snr:.1f} dB")
    print(f"  Cosine sim:   {cos_sim:.8f}")
    return snr


def compare_batch(audio: np.ndarray, num_codebooks: int, onnx_dir: str, output_dir: str):
    """Stage 1: Compare PyTorch vs ONNX in batch mode."""
    import onnxruntime as ort
    from transformers import AutoConfig, MimiModel

    print("=" * 60)
    print("STAGE 1: Batch Mode Comparison (PyTorch vs ONNX)")
    print("=" * 60)

    # --- PyTorch ---
    print("\nLoading PyTorch model...")
    config = AutoConfig.from_pretrained("kyutai/mimi")
    config.num_quantizers = num_codebooks
    model = MimiModel.from_pretrained("kyutai/mimi", config=config)
    model.eval()

    audio_tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)  # (1, 1, seq)
    print(f"Input shape: {audio_tensor.shape}")

    with torch.no_grad():
        enc_out = model.encode(audio_tensor, num_quantizers=num_codebooks, return_dict=True)
        pt_codes = enc_out.audio_codes.numpy()  # (1, num_cb, T)
        print(f"PyTorch codes: {pt_codes.shape}, range [{pt_codes.min()}, {pt_codes.max()}]")

        dec_out = model.decode(enc_out.audio_codes, return_dict=True)
        pt_audio = dec_out.audio_values.squeeze().numpy()
        print(f"PyTorch decoded audio: {pt_audio.shape}")

    # --- ONNX ---
    enc_path = os.path.join(onnx_dir, "encoder_model.onnx")
    dec_path = os.path.join(onnx_dir, "decoder_model.onnx")
    print(f"\nLoading ONNX models from {onnx_dir}...")

    enc_sess = ort.InferenceSession(enc_path)
    dec_sess = ort.InferenceSession(dec_path)

    # Print ONNX input names for debugging
    print(f"  Encoder inputs: {[i.name for i in enc_sess.get_inputs()]}")
    print(f"  Decoder inputs: {[i.name for i in dec_sess.get_inputs()]}")

    audio_np = audio.reshape(1, 1, -1).astype(np.float32)
    enc_outputs = enc_sess.run(None, {"input_values": audio_np})
    onnx_codes = enc_outputs[0]  # (1, num_cb, T)
    print(f"ONNX codes: {onnx_codes.shape}, range [{onnx_codes.min()}, {onnx_codes.max()}]")

    dec_outputs = dec_sess.run(None, {"audio_codes": onnx_codes})
    onnx_audio = dec_outputs[0].squeeze()
    print(f"ONNX decoded audio: {onnx_audio.shape}")

    # --- Compare codes ---
    pt_codes_flat = pt_codes.flatten()
    onnx_codes_flat = onnx_codes.flatten()
    min_codes = min(len(pt_codes_flat), len(onnx_codes_flat))
    codes_match = np.array_equal(pt_codes_flat[:min_codes], onnx_codes_flat[:min_codes])
    codes_diff_count = np.sum(pt_codes_flat[:min_codes] != onnx_codes_flat[:min_codes])
    print(f"\n--- Codes Comparison ---")
    print(f"  PyTorch codes length: {len(pt_codes_flat)}")
    print(f"  ONNX codes length:    {len(onnx_codes_flat)}")
    print(f"  Exact match:          {codes_match}")
    print(f"  Differing codes:      {codes_diff_count} / {min_codes} ({100*codes_diff_count/max(min_codes,1):.2f}%)")

    # --- Compare audio ---
    compute_metrics(pt_audio, onnx_audio, "Audio: PyTorch vs ONNX (batch)")

    # Cross-decode: PyTorch codes through ONNX decoder (isolates decoder)
    dec_outputs_cross = dec_sess.run(None, {"audio_codes": pt_codes.astype(np.int64)})
    cross_audio = dec_outputs_cross[0].squeeze()
    compute_metrics(pt_audio, cross_audio, "Audio: PyTorch encode+decode vs PyTorch encode + ONNX decode")

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    sf.write(os.path.join(output_dir, "pytorch_batch.wav"), pt_audio, 24000)
    sf.write(os.path.join(output_dir, "onnx_batch.wav"), onnx_audio, 24000)
    print(f"\nSaved WAVs to {output_dir}/")

    return codes_match


def compare_streaming(audio: np.ndarray, num_codebooks: int, onnx_dir: str,
                      batch_onnx_dir: str, output_dir: str, frame_size: int = FRAME_SIZE):
    """Stage 2: Compare ONNX v2 streaming (with conv state) vs batch baseline."""
    import onnxruntime as ort

    print("\n" + "=" * 60)
    print("STAGE 2: Streaming v2 vs Batch Comparison")
    print("=" * 60)

    # --- Load streaming ONNX models ---
    enc_path = os.path.join(onnx_dir, "encoder_model.onnx")
    dec_path = os.path.join(onnx_dir, "decoder_model.onnx")
    print(f"Loading streaming ONNX models from {onnx_dir}...")
    enc_sess = ort.InferenceSession(enc_path)
    dec_sess = ort.InferenceSession(dec_path)

    # Discover conv state inputs and their shapes
    def get_state_info(session, prefix="past_conv_"):
        """Extract conv state input names and zero-init shapes."""
        states = {}
        for inp in session.get_inputs():
            if inp.name.startswith(prefix):
                shape = [d if isinstance(d, int) else 1 for d in inp.shape]
                states[inp.name] = np.zeros(shape, dtype=np.float32)
        return states

    enc_conv_states = get_state_info(enc_sess)
    dec_conv_states = get_state_info(dec_sess)
    print(f"  Encoder conv states: {len(enc_conv_states)}")
    print(f"  Decoder conv states: {len(dec_conv_states)}")

    if not enc_conv_states:
        print("ERROR: No conv state inputs found. These are not v2 streaming models.")
        return

    # KV cache init
    enc_kv = [np.zeros((1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32) for _ in range(NUM_LAYERS * 2)]
    dec_kv = [np.zeros((1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32) for _ in range(NUM_LAYERS * 2)]
    kv_names = []
    for i in range(NUM_LAYERS):
        kv_names.append(f"past_key_{i}")
        kv_names.append(f"past_value_{i}")

    # Pad audio to multiple of frame_size
    n_frames = (len(audio) + frame_size - 1) // frame_size
    padded = np.zeros(n_frames * frame_size, dtype=np.float32)
    padded[: len(audio)] = audio

    # --- ONNX v2 streaming ---
    print(f"\nRunning ONNX v2 streaming ({n_frames} frames)...")
    onnx_all_codes = []
    onnx_all_audio = []

    # Count conv state outputs to know how to split outputs
    enc_output_names = [o.name for o in enc_sess.get_outputs()]
    dec_output_names = [o.name for o in dec_sess.get_outputs()]
    n_enc_conv = len(enc_conv_states)
    n_dec_conv = len(dec_conv_states)
    enc_conv_state_names = sorted(enc_conv_states.keys())
    dec_conv_state_names = sorted(dec_conv_states.keys())

    for i in range(n_frames):
        frame_np = padded[i * frame_size : (i + 1) * frame_size].reshape(1, 1, -1)

        # Build encoder inputs: audio + conv states + KV cache
        enc_inputs = {"input_values": frame_np}
        for name in enc_conv_state_names:
            enc_inputs[name] = enc_conv_states[name]
        for j, name in enumerate(kv_names):
            enc_inputs[name] = enc_kv[j]

        enc_outputs = enc_sess.run(None, enc_inputs)
        # Output layout: [codes, conv_states..., kv_cache...]
        codes_np = enc_outputs[0]
        for j, name in enumerate(enc_conv_state_names):
            present_name = name.replace("past_conv_", "present_conv_")
            idx = enc_output_names.index(present_name)
            enc_conv_states[name] = enc_outputs[idx]
        for j in range(NUM_LAYERS * 2):
            present_name = kv_names[j].replace("past_", "present_")
            idx = enc_output_names.index(present_name)
            enc_kv[j] = enc_outputs[idx]

        if codes_np.shape[2] > 0:
            onnx_all_codes.append(codes_np)

            # Build decoder inputs: codes + conv states + KV cache
            dec_inputs = {"audio_codes": codes_np}
            for name in dec_conv_state_names:
                dec_inputs[name] = dec_conv_states[name]
            for j, name in enumerate(kv_names):
                dec_inputs[name] = dec_kv[j]

            dec_outputs = dec_sess.run(None, dec_inputs)
            audio_chunk = dec_outputs[0].squeeze()
            for j, name in enumerate(dec_conv_state_names):
                present_name = name.replace("past_conv_", "present_conv_")
                idx = dec_output_names.index(present_name)
                dec_conv_states[name] = dec_outputs[idx]
            for j in range(NUM_LAYERS * 2):
                present_name = kv_names[j].replace("past_", "present_")
                idx = dec_output_names.index(present_name)
                dec_kv[j] = dec_outputs[idx]

            if audio_chunk.ndim > 0 and len(audio_chunk) > 0:
                onnx_all_audio.append(audio_chunk)

    onnx_stream_audio = np.concatenate(onnx_all_audio) if onnx_all_audio else np.array([])
    onnx_stream_codes = np.concatenate([c.flatten() for c in onnx_all_codes]) if onnx_all_codes else np.array([])
    print(f"  ONNX v2 streaming: {len(onnx_all_codes)} code frames, {len(onnx_stream_audio)} audio samples")

    # --- Batch baseline ---
    batch_enc_path = os.path.join(batch_onnx_dir, "encoder_model.onnx")
    batch_dec_path = os.path.join(batch_onnx_dir, "decoder_model.onnx")
    print(f"\nRunning batch ONNX baseline from {batch_onnx_dir}...")
    batch_enc = ort.InferenceSession(batch_enc_path)
    batch_dec = ort.InferenceSession(batch_dec_path)

    audio_np = padded.reshape(1, 1, -1).astype(np.float32)
    batch_codes = batch_enc.run(None, {"input_values": audio_np})[0]
    batch_audio = batch_dec.run(None, {"audio_codes": batch_codes})[0].squeeze()
    print(f"  Batch: codes {batch_codes.shape}, audio {batch_audio.shape}")

    # --- Compare codes ---
    batch_codes_flat = batch_codes.flatten()
    min_codes = min(len(onnx_stream_codes), len(batch_codes_flat))
    if min_codes > 0:
        codes_match = np.array_equal(onnx_stream_codes[:min_codes], batch_codes_flat[:min_codes])
        codes_diff = np.sum(onnx_stream_codes[:min_codes] != batch_codes_flat[:min_codes])
        print(f"\n--- Codes: Streaming v2 vs Batch ---")
        print(f"  Stream codes: {len(onnx_stream_codes)}, Batch codes: {len(batch_codes_flat)}")
        print(f"  Exact match:     {codes_match}")
        print(f"  Differing codes: {codes_diff} / {min_codes} ({100*codes_diff/max(min_codes,1):.2f}%)")

    # --- Compare audio ---
    if len(onnx_stream_audio) > 0 and len(batch_audio) > 0:
        compute_metrics(onnx_stream_audio, batch_audio, "Audio: ONNX v2 streaming vs ONNX batch")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    if len(onnx_stream_audio) > 0:
        sf.write(os.path.join(output_dir, "onnx_v2_streaming.wav"), onnx_stream_audio, 24000)
    sf.write(os.path.join(output_dir, "onnx_batch.wav"), batch_audio, 24000)
    print(f"Saved WAVs to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Compare PyTorch vs ONNX Mimi codec output")
    parser.add_argument("--input", required=True, help="Input WAV file (24kHz mono)")
    parser.add_argument("--num-codebooks", type=int, default=8)
    parser.add_argument("--batch-onnx-dir", default="onnx-models/onnx-8cb",
                        help="Directory with batch ONNX encoder/decoder")
    parser.add_argument("--streaming-onnx-dir", default="onnx-models/streaming-8cb",
                        help="Directory with streaming ONNX encoder/decoder")
    parser.add_argument("--output-dir", default="comparison_output")
    parser.add_argument("--streaming", action="store_true",
                        help="Also run Stage 2 streaming comparison")
    parser.add_argument("--chunk-ms", type=int, default=80,
                        help="Streaming chunk size in ms (must match model export)")
    args = parser.parse_args()

    audio = load_audio(args.input)
    print(f"Input: {args.input} ({len(audio)} samples, {len(audio)/24000:.2f}s)")

    compare_batch(audio, args.num_codebooks, args.batch_onnx_dir, args.output_dir)

    if args.streaming:
        stream_frame_size = int(24000 * args.chunk_ms / 1000)
        compare_streaming(audio, args.num_codebooks, args.streaming_onnx_dir,
                          args.batch_onnx_dir, args.output_dir,
                          frame_size=stream_frame_size)


if __name__ == "__main__":
    main()
