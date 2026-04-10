#!/usr/bin/env python3
"""Export streaming Mimi ONNX models with conv state + KV cache I/O.

Implements Candle-compatible streaming: each conv layer carries forward a
small state buffer (the "tail" of the previous input or overlap from the
previous transpose conv output). This matches the StreamableConv1d.step()
and StreamableConvTranspose1d.step() logic in mimi-core/src/conv.rs.

Usage:
    python scripts/export_streaming_onnx.py --num-codebooks 8  --output-dir onnx-models/streaming-8cb --chunk-ms 320
    python scripts/export_streaming_onnx.py --num-codebooks 16 --output-dir onnx-models/streaming-16cb --chunk-ms 320
"""

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, MimiModel
from transformers.cache_utils import DynamicCache

NUM_LAYERS = 8
NUM_HEADS = 8
HEAD_DIM = 64
DEFAULT_FRAME_SIZE = 1920  # minimum frame; actual frame set by --chunk-ms


# ─── Streaming conv primitives ──────────────────────────────────────────────

@dataclass
class ConvStateSpec:
    """Describes a conv state tensor's shape (B, channels, temporal_size)."""
    channels: int
    temporal_size: int  # kernel_eff - stride (0 means no state needed)
    name: str
    is_transpose: bool = False


def streaming_conv1d(conv: nn.Conv1d, xs: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Streaming Conv1d step matching Candle's StreamableConv1d.step().

    Args:
        conv: The Conv1d module
        xs: (B, C_in, T) current frame input
        state: (B, C_in, state_size) previous state. state_size = eff_kernel - stride.
               On first frame, this is zeros (equivalent to causal constant padding).
    Returns:
        (output, new_state)
    """
    stride = conv.stride[0]
    dilation = conv.dilation[0]
    kernel = conv.kernel_size[0]
    eff_kernel = (kernel - 1) * dilation + 1

    # Concatenate state (tail of previous input) with current input
    xs_cat = torch.cat([state, xs], dim=-1)

    seq_len = xs_cat.shape[-1]
    num_frames = (seq_len + stride - eff_kernel) // stride

    # Extract the portion that can be convolved and the leftover state
    offset = num_frames * stride
    new_state = xs_cat[:, :, offset:]
    in_l = (num_frames - 1) * stride + eff_kernel
    xs_for_conv = xs_cat[:, :, :in_l]

    # Apply conv directly (no padding — we handle it via state)
    ys = F.conv1d(xs_for_conv, conv.weight, conv.bias,
                  stride=stride, dilation=dilation, groups=conv.groups)

    return ys, new_state


def streaming_conv_tr1d(conv: nn.ConvTranspose1d, xs: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Streaming ConvTranspose1d step matching Candle's StreamableConvTranspose1d.step().

    Args:
        conv: The ConvTranspose1d module
        xs: (B, C_in, T) current frame input
        state: (B, C_out, overlap_size) overlap from previous frame output.
               overlap_size = kernel - stride. Initially zeros.
    Returns:
        (output, new_state)
    """
    stride = conv.stride[0]
    kernel = conv.kernel_size[0]
    invalid_steps = kernel - stride

    # Run transpose conv (no padding)
    ys = F.conv_transpose1d(xs, conv.weight, conv.bias, stride=stride, groups=conv.groups)

    ot = ys.shape[-1]

    # Overlap-add: the first `invalid_steps` samples overlap with previous state
    pt = state.shape[-1]
    if pt > 0:
        # Remove bias from prev state (it was included when computed, will be double-counted)
        prev_ys = state
        if conv.bias is not None:
            prev_ys = prev_ys - conv.bias.reshape(1, -1, 1)
        # Add overlap
        ys_head = ys[:, :, :pt] + prev_ys
        ys_tail = ys[:, :, pt:]
        ys = torch.cat([ys_head, ys_tail], dim=-1)

    # Split: output everything except the last `invalid_steps` samples
    output_len = ot - invalid_steps
    ys_out = ys[:, :, :output_len]
    new_state = ys[:, :, output_len:]

    return ys_out, new_state


# ─── Streaming encoder wrapper ──────────────────────────────────────────────

class StreamingEncoderV2(nn.Module):
    """Full streaming encoder: SEANet → Transformer → Downsample → Quantizer.

    Manages both conv state buffers and transformer KV cache as explicit I/O.
    """

    def __init__(self, model: MimiModel, num_codebooks: int):
        super().__init__()
        self.model = model
        self.num_codebooks = num_codebooks

        # Enumerate conv layers that need state
        self.conv_specs: List[ConvStateSpec] = []
        self._build_conv_specs()

    def _build_conv_specs(self):
        """Identify all conv layers with non-zero state in the encoder path."""
        enc = self.model.encoder

        for i, layer in enumerate(enc.layers):
            if hasattr(layer, "conv") and isinstance(layer.conv, nn.Conv1d):
                c = layer.conv
                k, s, d = c.kernel_size[0], c.stride[0], c.dilation[0]
                eff_k = (k - 1) * d + 1
                state_size = eff_k - s
                if state_size > 0:
                    self.conv_specs.append(ConvStateSpec(c.in_channels, state_size, f"enc_{i}"))
            if hasattr(layer, "block"):
                for j, sub in enumerate(layer.block):
                    if hasattr(sub, "conv") and isinstance(sub.conv, nn.Conv1d):
                        c = sub.conv
                        k, s, d = c.kernel_size[0], c.stride[0], c.dilation[0]
                        eff_k = (k - 1) * d + 1
                        state_size = eff_k - s
                        if state_size > 0:
                            self.conv_specs.append(ConvStateSpec(c.in_channels, state_size, f"enc_{i}_b{j}"))

        # Downsample conv
        ds = self.model.downsample
        c = ds.conv
        k, s = c.kernel_size[0], c.stride[0]
        state_size = k - s
        if state_size > 0:
            self.conv_specs.append(ConvStateSpec(c.in_channels, state_size, "ds"))

    @torch.no_grad()
    def forward(self, input_values: torch.Tensor, *state_tensors):
        """
        Args:
            input_values: (B, 1, 1920) one frame of audio
            state_tensors: flat list of [conv_state_0, ..., conv_state_N, kv_key_0, kv_val_0, ...]
        Returns:
            (audio_codes, *updated_conv_states, *updated_kv_cache)
        """
        n_conv = len(self.conv_specs)
        conv_states = list(state_tensors[:n_conv])
        kv_flat = state_tensors[n_conv:]

        # Rebuild transformer KV cache
        enc_cache = DynamicCache()
        for layer_idx in range(NUM_LAYERS):
            k = kv_flat[layer_idx * 2]
            v = kv_flat[layer_idx * 2 + 1]
            enc_cache.update(k, v, layer_idx)

        # ─── SEANet encoder ───
        xs = input_values
        conv_idx = 0
        enc = self.model.encoder

        for i, layer in enumerate(enc.layers):
            if hasattr(layer, "conv") and isinstance(layer.conv, nn.Conv1d):
                c = layer.conv
                k, s, d = c.kernel_size[0], c.stride[0], c.dilation[0]
                eff_k = (k - 1) * d + 1
                state_size = eff_k - s
                if state_size > 0:
                    xs, new_state = streaming_conv1d(c, xs, conv_states[conv_idx])
                    conv_states[conv_idx] = new_state
                    conv_idx += 1
                else:
                    # Pointwise conv (k=1, s=1) — no state needed, but still need padding=0
                    xs = F.conv1d(xs, c.weight, c.bias, stride=s, dilation=d, groups=c.groups)
            elif hasattr(layer, "block"):
                # ResnetBlock: shortcut is Identity (true_skip)
                residual = xs
                for j, sub in enumerate(layer.block):
                    if isinstance(sub, nn.ELU):
                        xs = F.elu(xs)
                    elif hasattr(sub, "conv") and isinstance(sub.conv, nn.Conv1d):
                        c = sub.conv
                        k, s, d = c.kernel_size[0], c.stride[0], c.dilation[0]
                        eff_k = (k - 1) * d + 1
                        state_size = eff_k - s
                        if state_size > 0:
                            xs, new_state = streaming_conv1d(c, xs, conv_states[conv_idx])
                            conv_states[conv_idx] = new_state
                            conv_idx += 1
                        else:
                            xs = F.conv1d(xs, c.weight, c.bias, stride=s, dilation=d, groups=c.groups)
                xs = xs + residual
            elif isinstance(layer, nn.ELU):
                xs = F.elu(xs)

        # ─── Encoder Transformer (single call with causal mask) ───
        embeddings = xs.transpose(1, 2)  # (B, T, C)
        n_tokens = embeddings.shape[1]
        past_len = enc_cache.key_cache[0].shape[2]
        total_len = past_len + n_tokens
        row_pos = torch.arange(n_tokens, device=embeddings.device).unsqueeze(1) + past_len
        col_pos = torch.arange(total_len, device=embeddings.device).unsqueeze(0)
        causal_mask = torch.where(col_pos <= row_pos, 0.0, -1e9)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, n_tokens, total_len)

        out = self.model.encoder_transformer(
            embeddings, attention_mask=causal_mask, past_key_values=enc_cache, return_dict=True
        )
        new_enc_cache = out.past_key_values
        embeddings = out.last_hidden_state.transpose(1, 2)  # (B, C, T)

        # ─── Downsample ───
        ds_conv = self.model.downsample.conv
        k, s = ds_conv.kernel_size[0], ds_conv.stride[0]
        state_size = k - s
        if state_size > 0:
            embeddings, new_state = streaming_conv1d(ds_conv, embeddings, conv_states[conv_idx])
            conv_states[conv_idx] = new_state
            conv_idx += 1
        else:
            embeddings = F.conv1d(embeddings, ds_conv.weight, ds_conv.bias, stride=s)

        # ─── Quantizer ───
        # quantizer.encode expects (B, C, T), returns (num_cb, B, T)
        codes = self.model.quantizer.encode(embeddings)
        codes = codes.permute(1, 0, 2)  # → (B, num_cb, T)

        # ─── Flatten outputs ───
        outputs = [codes]
        for s in conv_states:
            outputs.append(s)
        for layer_idx in range(NUM_LAYERS):
            outputs.append(new_enc_cache.key_cache[layer_idx])
            outputs.append(new_enc_cache.value_cache[layer_idx])

        return tuple(outputs)


# ─── Streaming decoder wrapper ──────────────────────────────────────────────

class StreamingDecoderV2(nn.Module):
    """Full streaming decoder: Dequantizer → Upsample → Transformer → SEANet."""

    def __init__(self, model: MimiModel):
        super().__init__()
        self.model = model

        self.conv_specs: List[ConvStateSpec] = []
        self._build_conv_specs()

    def _build_conv_specs(self):
        """Identify all conv layers with non-zero state in the decoder path."""
        # Upsample (ConvTranspose1d)
        us = self.model.upsample
        c = us.conv
        k, s = c.kernel_size[0], c.stride[0]
        state_size = k - s
        if state_size > 0:
            self.conv_specs.append(ConvStateSpec(c.out_channels, state_size, "us", is_transpose=True))

        # Decoder layers
        dec = self.model.decoder
        for i, layer in enumerate(dec.layers):
            if hasattr(layer, "conv"):
                if isinstance(layer.conv, nn.ConvTranspose1d):
                    c = layer.conv
                    k, s = c.kernel_size[0], c.stride[0]
                    state_size = k - s
                    if state_size > 0:
                        self.conv_specs.append(ConvStateSpec(c.out_channels, state_size, f"dec_{i}", is_transpose=True))
                elif isinstance(layer.conv, nn.Conv1d):
                    c = layer.conv
                    k, s, d = c.kernel_size[0], c.stride[0], c.dilation[0]
                    eff_k = (k - 1) * d + 1
                    state_size = eff_k - s
                    if state_size > 0:
                        self.conv_specs.append(ConvStateSpec(c.in_channels, state_size, f"dec_{i}"))
            if hasattr(layer, "block"):
                for j, sub in enumerate(layer.block):
                    if hasattr(sub, "conv") and isinstance(sub.conv, nn.Conv1d):
                        c = sub.conv
                        k, s, d = c.kernel_size[0], c.stride[0], c.dilation[0]
                        eff_k = (k - 1) * d + 1
                        state_size = eff_k - s
                        if state_size > 0:
                            self.conv_specs.append(ConvStateSpec(c.in_channels, state_size, f"dec_{i}_b{j}"))

    @torch.no_grad()
    def forward(self, audio_codes: torch.Tensor, *state_tensors):
        n_conv = len(self.conv_specs)
        conv_states = list(state_tensors[:n_conv])
        kv_flat = state_tensors[n_conv:]

        # Rebuild transformer KV cache
        dec_cache = DynamicCache()
        for layer_idx in range(NUM_LAYERS):
            k = kv_flat[layer_idx * 2]
            v = kv_flat[layer_idx * 2 + 1]
            dec_cache.update(k, v, layer_idx)

        # ─── Dequantizer ───
        # audio_codes: (B, num_cb, T) → quantizer.decode returns (B, C, T)
        embeddings = self.model.quantizer.decode(audio_codes)

        conv_idx = 0

        # ─── Upsample ───
        us_conv = self.model.upsample.conv
        k, s = us_conv.kernel_size[0], us_conv.stride[0]
        state_size = k - s
        if state_size > 0:
            embeddings, new_state = streaming_conv_tr1d(us_conv, embeddings, conv_states[conv_idx])
            conv_states[conv_idx] = new_state
            conv_idx += 1

        # ─── Decoder Transformer (single call with causal mask) ───
        embeddings_t = embeddings.transpose(1, 2)  # (B, T, C)
        n_tokens = embeddings_t.shape[1]
        past_len = dec_cache.key_cache[0].shape[2]
        total_len = past_len + n_tokens
        row_pos = torch.arange(n_tokens, device=embeddings_t.device).unsqueeze(1) + past_len
        col_pos = torch.arange(total_len, device=embeddings_t.device).unsqueeze(0)
        causal_mask = torch.where(col_pos <= row_pos, 0.0, -1e9)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        out = self.model.decoder_transformer(
            embeddings_t, attention_mask=causal_mask, past_key_values=dec_cache, return_dict=True
        )
        new_dec_cache = out.past_key_values
        embeddings = out.last_hidden_state.transpose(1, 2)  # (B, C, T)

        # ─── SEANet decoder ───
        xs = embeddings
        dec = self.model.decoder

        for i, layer in enumerate(dec.layers):
            if hasattr(layer, "conv"):
                if isinstance(layer.conv, nn.ConvTranspose1d):
                    c = layer.conv
                    k, s = c.kernel_size[0], c.stride[0]
                    state_size = k - s
                    if state_size > 0:
                        xs, new_state = streaming_conv_tr1d(c, xs, conv_states[conv_idx])
                        conv_states[conv_idx] = new_state
                        conv_idx += 1
                    else:
                        xs = F.conv_transpose1d(xs, c.weight, c.bias, stride=s, groups=c.groups)
                elif isinstance(layer.conv, nn.Conv1d):
                    c = layer.conv
                    k, s, d = c.kernel_size[0], c.stride[0], c.dilation[0]
                    eff_k = (k - 1) * d + 1
                    state_size = eff_k - s
                    if state_size > 0:
                        xs, new_state = streaming_conv1d(c, xs, conv_states[conv_idx])
                        conv_states[conv_idx] = new_state
                        conv_idx += 1
                    else:
                        xs = F.conv1d(xs, c.weight, c.bias, stride=s, dilation=d, groups=c.groups)
            elif hasattr(layer, "block"):
                residual = xs
                for j, sub in enumerate(layer.block):
                    if isinstance(sub, nn.ELU):
                        xs = F.elu(xs)
                    elif hasattr(sub, "conv") and isinstance(sub.conv, nn.Conv1d):
                        c = sub.conv
                        k, s, d = c.kernel_size[0], c.stride[0], c.dilation[0]
                        eff_k = (k - 1) * d + 1
                        state_size = eff_k - s
                        if state_size > 0:
                            xs, new_state = streaming_conv1d(c, xs, conv_states[conv_idx])
                            conv_states[conv_idx] = new_state
                            conv_idx += 1
                        else:
                            xs = F.conv1d(xs, c.weight, c.bias, stride=s, dilation=d, groups=c.groups)
                xs = xs + residual
            elif isinstance(layer, nn.ELU):
                xs = F.elu(xs)

        audio = xs  # (B, 1, T)

        # ─── Flatten outputs ───
        outputs = [audio]
        for s in conv_states:
            outputs.append(s)
        for layer_idx in range(NUM_LAYERS):
            outputs.append(new_dec_cache.key_cache[layer_idx])
            outputs.append(new_dec_cache.value_cache[layer_idx])

        return tuple(outputs)


# ─── Export logic ────────────────────────────────────────────────────────────

def make_state_names(specs: List[ConvStateSpec], prefix: str, suffix: str) -> List[str]:
    return [f"{prefix}_{spec.name}_{suffix}" for spec in specs]


def make_kv_names(prefix: str) -> List[str]:
    names = []
    for i in range(NUM_LAYERS):
        names.append(f"{prefix}_key_{i}")
        names.append(f"{prefix}_value_{i}")
    return names


def export_streaming(num_codebooks: int, output_dir: str, frame_size: int = 1920):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading Mimi model with {num_codebooks} codebooks, frame_size={frame_size}...")
    config = AutoConfig.from_pretrained("kyutai/mimi")
    config.num_quantizers = num_codebooks
    config.use_cache = True
    model = MimiModel.from_pretrained("kyutai/mimi", config=config)
    model.eval()

    # ─── Export Encoder ───
    print("Building streaming encoder...")
    encoder = StreamingEncoderV2(model, num_codebooks)
    print(f"  Encoder conv states: {len(encoder.conv_specs)}")
    for spec in encoder.conv_specs:
        print(f"    {spec.name}: ({spec.channels}, {spec.temporal_size})")

    # Build dummy inputs
    dummy_audio = torch.randn(1, 1, frame_size)
    dummy_conv_states = []
    for spec in encoder.conv_specs:
        dummy_conv_states.append(torch.zeros(1, spec.channels, spec.temporal_size))
    dummy_kv = []
    for _ in range(NUM_LAYERS):
        dummy_kv.append(torch.zeros(1, NUM_HEADS, 2, HEAD_DIM))  # key
        dummy_kv.append(torch.zeros(1, NUM_HEADS, 2, HEAD_DIM))  # value

    # Verify forward pass works
    print("  Verifying encoder forward pass...")
    with torch.no_grad():
        out = encoder(dummy_audio, *dummy_conv_states, *dummy_kv)
        print(f"  Output codes shape: {out[0].shape}")
        n_conv = len(encoder.conv_specs)
        for i, spec in enumerate(encoder.conv_specs):
            print(f"  Updated state {spec.name}: {out[1 + i].shape}")

    # ONNX input/output names
    enc_input_names = ["input_values"]
    enc_input_names += make_state_names(encoder.conv_specs, "past_conv", "state")
    enc_input_names += make_kv_names("past")

    enc_output_names = ["audio_codes"]
    enc_output_names += make_state_names(encoder.conv_specs, "present_conv", "state")
    enc_output_names += make_kv_names("present")

    # Dynamic axes
    enc_dynamic_axes = {
        "input_values": {0: "batch", 2: "seq_len"},
        "audio_codes": {0: "batch", 2: "codes_len"},
    }
    for name in make_kv_names("past"):
        enc_dynamic_axes[name] = {0: "batch", 2: "past_seq_len"}
    for name in make_kv_names("present"):
        enc_dynamic_axes[name] = {0: "batch", 2: "present_seq_len"}

    encoder_path = os.path.join(output_dir, "encoder_model.onnx")
    print(f"  Exporting encoder to {encoder_path}...")
    with torch.no_grad():
        torch.onnx.export(
            encoder,
            (dummy_audio, *dummy_conv_states, *dummy_kv),
            encoder_path,
            input_names=enc_input_names,
            output_names=enc_output_names,
            dynamic_axes=enc_dynamic_axes,
            opset_version=14,
            dynamo=False,
        )
    enc_size = os.path.getsize(encoder_path) / 1024 / 1024
    print(f"  Saved: {encoder_path} ({enc_size:.1f} MB)")

    # ─── Export Decoder ───
    print("\nBuilding streaming decoder...")
    decoder = StreamingDecoderV2(model)
    print(f"  Decoder conv states: {len(decoder.conv_specs)}")
    for spec in decoder.conv_specs:
        tr = " (transpose)" if spec.is_transpose else ""
        print(f"    {spec.name}: ({spec.channels}, {spec.temporal_size}){tr}")

    # Build dummy inputs — code frames = frame_size / (960 * 2) where 960 is SEANet stride, 2 is downsample stride
    n_code_frames = frame_size // (960 * 2)
    print(f"  Decoder dummy: {n_code_frames} code frames (from {frame_size} samples)")
    dummy_codes = torch.zeros(1, num_codebooks, max(1, n_code_frames), dtype=torch.long)
    dummy_dec_conv_states = []
    for spec in decoder.conv_specs:
        dummy_dec_conv_states.append(torch.zeros(1, spec.channels, spec.temporal_size))
    dummy_dec_kv = []
    for _ in range(NUM_LAYERS):
        dummy_dec_kv.append(torch.zeros(1, NUM_HEADS, 2, HEAD_DIM))
        dummy_dec_kv.append(torch.zeros(1, NUM_HEADS, 2, HEAD_DIM))

    print("  Verifying decoder forward pass...")
    with torch.no_grad():
        dec_out = decoder(dummy_codes, *dummy_dec_conv_states, *dummy_dec_kv)
        print(f"  Output audio shape: {dec_out[0].shape}")

    dec_input_names = ["audio_codes"]
    dec_input_names += make_state_names(decoder.conv_specs, "past_conv", "state")
    dec_input_names += make_kv_names("past")

    dec_output_names = ["audio_values"]
    dec_output_names += make_state_names(decoder.conv_specs, "present_conv", "state")
    dec_output_names += make_kv_names("present")

    dec_dynamic_axes = {
        "audio_codes": {0: "batch", 2: "codes_len"},
        "audio_values": {0: "batch", 2: "seq_len"},
    }
    for name in make_kv_names("past"):
        dec_dynamic_axes[name] = {0: "batch", 2: "past_seq_len"}
    for name in make_kv_names("present"):
        dec_dynamic_axes[name] = {0: "batch", 2: "present_seq_len"}

    decoder_path = os.path.join(output_dir, "decoder_model.onnx")
    print(f"  Exporting decoder to {decoder_path}...")
    with torch.no_grad():
        torch.onnx.export(
            decoder,
            (dummy_codes, *dummy_dec_conv_states, *dummy_dec_kv),
            decoder_path,
            input_names=dec_input_names,
            output_names=dec_output_names,
            dynamic_axes=dec_dynamic_axes,
            opset_version=14,
            dynamo=False,
        )
    dec_size = os.path.getsize(decoder_path) / 1024 / 1024
    print(f"  Saved: {decoder_path} ({dec_size:.1f} MB)")

    # ─── Verify streaming roundtrip ───
    print("\nVerifying ONNX streaming roundtrip...")
    import onnxruntime as ort

    enc_sess = ort.InferenceSession(encoder_path)
    dec_sess = ort.InferenceSession(decoder_path)

    print(f"  Encoder inputs: {[i.name for i in enc_sess.get_inputs()]}")
    print(f"  Decoder inputs: {[i.name for i in dec_sess.get_inputs()]}")

    # Initialize states
    enc_conv_states_np = [np.zeros((1, s.channels, s.temporal_size), dtype=np.float32) for s in encoder.conv_specs]
    enc_kv_np = [np.zeros((1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32) for _ in range(NUM_LAYERS * 2)]

    dec_conv_states_np = [np.zeros((1, s.channels, s.temporal_size), dtype=np.float32) for s in decoder.conv_specs]
    dec_kv_np = [np.zeros((1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32) for _ in range(NUM_LAYERS * 2)]

    n_enc_conv = len(encoder.conv_specs)
    n_dec_conv = len(decoder.conv_specs)

    enc_conv_names = make_state_names(encoder.conv_specs, "past_conv", "state")
    enc_kv_names = make_kv_names("past")
    dec_conv_names = make_state_names(decoder.conv_specs, "past_conv", "state")
    dec_kv_names = make_kv_names("past")

    # Run 3 frames
    for frame_idx in range(3):
        frame = np.random.randn(1, 1, frame_size).astype(np.float32)

        enc_inputs = {"input_values": frame}
        for i, name in enumerate(enc_conv_names):
            enc_inputs[name] = enc_conv_states_np[i]
        for i, name in enumerate(enc_kv_names):
            enc_inputs[name] = enc_kv_np[i]

        enc_outputs = enc_sess.run(None, enc_inputs)
        codes = enc_outputs[0]
        enc_conv_states_np = enc_outputs[1:1 + n_enc_conv]
        enc_kv_np = enc_outputs[1 + n_enc_conv:]

        print(f"  Frame {frame_idx}: codes {codes.shape}, kv[0] {enc_kv_np[0].shape}")

        # Decode
        dec_inputs = {"audio_codes": codes}
        for i, name in enumerate(dec_conv_names):
            dec_inputs[name] = dec_conv_states_np[i]
        for i, name in enumerate(dec_kv_names):
            dec_inputs[name] = dec_kv_np[i]

        dec_outputs = dec_sess.run(None, dec_inputs)
        audio = dec_outputs[0]
        dec_conv_states_np = dec_outputs[1:1 + n_dec_conv]
        dec_kv_np = dec_outputs[1 + n_dec_conv:]

        print(f"          audio {audio.shape}")

    # Write state spec metadata for Rust
    spec_path = os.path.join(output_dir, "state_spec.txt")
    with open(spec_path, "w") as f:
        f.write("# Conv state specs for streaming ONNX model\n")
        f.write(f"# num_codebooks={num_codebooks}\n")
        f.write("# Format: type name channels temporal_size\n")
        f.write("\n[encoder]\n")
        for spec in encoder.conv_specs:
            f.write(f"conv {spec.name} {spec.channels} {spec.temporal_size}\n")
        f.write(f"\n[decoder]\n")
        for spec in decoder.conv_specs:
            kind = "conv_tr" if spec.is_transpose else "conv"
            f.write(f"{kind} {spec.name} {spec.channels} {spec.temporal_size}\n")
    print(f"\nState spec written to {spec_path}")

    print(f"\nDone! Models exported to {output_dir}/")
    print(f"  Bitrate at 12.5 Hz: {num_codebooks * 11 * 12.5:.0f} bps = {num_codebooks * 11 * 12.5 / 1000:.2f} kbps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export streaming Mimi ONNX models with conv state + KV cache")
    parser.add_argument("--num-codebooks", type=int, default=8)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--chunk-ms", type=int, default=320,
                        help="Chunk size in ms (must be multiple of 80). Larger = fewer ORT calls = faster.")
    args = parser.parse_args()

    frame_size = int(24000 * args.chunk_ms / 1000)
    if frame_size % 1920 != 0:
        parser.error(f"--chunk-ms={args.chunk_ms} gives {frame_size} samples, must be multiple of 1920 (80ms)")

    export_streaming(args.num_codebooks, args.output_dir, frame_size=frame_size)
