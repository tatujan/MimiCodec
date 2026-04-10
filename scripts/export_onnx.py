#!/usr/bin/env python3
"""Export Mimi ONNX models with configurable number of codebooks.

Uses optimum's export pipeline with a patched config to control num_quantizers.

Usage:
    python scripts/export_onnx.py --num-codebooks 8  --output-dir onnx-models/onnx-8cb
    python scripts/export_onnx.py --num-codebooks 16 --output-dir onnx-models/onnx-16cb
"""

import argparse
import os
import shutil

import numpy as np


def export_with_optimum(num_codebooks: int, output_dir: str):
    """Try exporting via optimum's ONNX export pipeline."""
    from optimum.exporters.onnx import main_export

    os.makedirs(output_dir, exist_ok=True)

    print(f"Exporting Mimi with {num_codebooks} codebooks via optimum...")
    print(f"Output: {output_dir}")

    # optimum handles the export, but we need to patch num_quantizers
    # Export to a temp dir first, then we'll verify
    main_export(
        "kyutai/mimi",
        output=output_dir,
        task="feature-extraction",
        no_post_process=True,
    )

    print(f"\nExported to {output_dir}")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith(".onnx"):
            size_mb = os.path.getsize(os.path.join(output_dir, f)) / 1024 / 1024
            print(f"  {f}: {size_mb:.1f} MB")


def export_manual(num_codebooks: int, output_dir: str):
    """Manual export using torch.onnx with careful tracing setup."""
    import torch
    from transformers import MimiModel, AutoConfig

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading Mimi model...")
    config = AutoConfig.from_pretrained("kyutai/mimi")
    print(f"Original num_quantizers: {config.num_quantizers}")
    config.num_quantizers = num_codebooks
    print(f"Set to: {config.num_quantizers}")

    model = MimiModel.from_pretrained("kyutai/mimi", config=config)
    model.eval()

    # Verify it works in PyTorch first
    print(f"\nVerifying PyTorch encode/decode with {num_codebooks} codebooks...")
    with torch.no_grad():
        dummy = torch.randn(1, 1, 24000)
        enc_out = model.encode(dummy, num_quantizers=num_codebooks)
        codes = enc_out.audio_codes
        print(f"  Encoder output shape: {codes.shape}")  # [1, num_codebooks, T]
        assert codes.shape[1] == num_codebooks, f"Expected {num_codebooks} codebooks, got {codes.shape[1]}"

        dec_out = model.decode(codes)
        audio = dec_out.audio_values
        print(f"  Decoder output shape: {audio.shape}")  # [1, 1, samples]

    # Export encoder using torch.jit.trace approach
    print(f"\nExporting encoder...")

    class TracableEncoder(torch.nn.Module):
        def __init__(self, model, num_q):
            super().__init__()
            self.model = model
            self.num_q = num_q

        @torch.no_grad()
        def forward(self, input_values):
            # Use the internal encode path directly
            encoded = self.model.encode(input_values, num_quantizers=self.num_q)
            return encoded.audio_codes

    class TracableDecoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        @torch.no_grad()
        def forward(self, audio_codes):
            decoded = self.model.decode(audio_codes)
            return decoded.audio_values

    encoder = TracableEncoder(model, num_codebooks)
    decoder = TracableDecoder(model)

    encoder_path = os.path.join(output_dir, "encoder_model.onnx")
    decoder_path = os.path.join(output_dir, "decoder_model.onnx")

    # Use a longer dummy to avoid edge cases
    dummy_audio = torch.randn(1, 1, 48000)  # 2 seconds

    with torch.no_grad():
        torch.onnx.export(
            encoder,
            (dummy_audio,),
            encoder_path,
            input_names=["input_values"],
            output_names=["audio_codes"],
            dynamic_axes={
                "input_values": {0: "batch_size", 2: "sequence_length"},
                "audio_codes": {0: "batch_size", 2: "codes_length"},
            },
            opset_version=14,
            dynamo=False,
        )
    enc_size = os.path.getsize(encoder_path) / 1024 / 1024
    print(f"  Saved: {encoder_path} ({enc_size:.1f} MB)")

    # Get dummy codes for decoder export
    with torch.no_grad():
        dummy_codes = model.encode(dummy_audio, num_quantizers=num_codebooks).audio_codes
    print(f"  Dummy codes shape for decoder: {dummy_codes.shape}")

    with torch.no_grad():
        torch.onnx.export(
            decoder,
            (dummy_codes,),
            decoder_path,
            input_names=["audio_codes"],
            output_names=["audio_values"],
            dynamic_axes={
                "audio_codes": {0: "batch_size", 2: "codes_length"},
                "audio_values": {0: "batch_size", 2: "sequence_length"},
            },
            opset_version=14,
            dynamo=False,
        )
    dec_size = os.path.getsize(decoder_path) / 1024 / 1024
    print(f"  Saved: {decoder_path} ({dec_size:.1f} MB)")

    # Verify ONNX models
    print(f"\nVerifying ONNX models...")
    import onnxruntime as ort

    enc_sess = ort.InferenceSession(encoder_path)
    dec_sess = ort.InferenceSession(decoder_path)

    test_audio = np.random.randn(1, 1, 24000).astype(np.float32)
    onnx_codes = enc_sess.run(None, {"input_values": test_audio})[0]
    print(f"  ONNX encoder output: {onnx_codes.shape}, range [{onnx_codes.min()}, {onnx_codes.max()}]")
    assert onnx_codes.shape[1] == num_codebooks, f"Expected {num_codebooks} codebooks in ONNX output"

    onnx_audio = dec_sess.run(None, {"audio_codes": onnx_codes})[0]
    print(f"  ONNX decoder output: {onnx_audio.shape}")

    print(f"\nDone! {num_codebooks}-codebook models exported to {output_dir}/")
    print(f"  Bitrate at 12.5 Hz: {num_codebooks * 11 * 12.5:.0f} bps = {num_codebooks * 11 * 12.5 / 1000:.2f} kbps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Mimi to ONNX with custom codebook count")
    parser.add_argument("--num-codebooks", type=int, default=8)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--method", choices=["manual", "optimum"], default="manual")
    args = parser.parse_args()

    if args.method == "optimum":
        export_with_optimum(args.num_codebooks, args.output_dir)
    else:
        export_manual(args.num_codebooks, args.output_dir)
