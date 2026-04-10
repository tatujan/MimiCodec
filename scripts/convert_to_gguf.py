#!/usr/bin/env python3
"""Convert Mimi safetensors model to GGUF with Q8_0 quantized transformer weights."""

import argparse
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
from gguf import GGUFWriter, GGMLQuantizationType

QUANT_TYPES = {
    "q8_0": GGMLQuantizationType.Q8_0,
    "q4_0": GGMLQuantizationType.Q4_0,
    "q4_1": GGMLQuantizationType.Q4_1,
    "q5_0": GGMLQuantizationType.Q5_0,
    "q5_1": GGMLQuantizationType.Q5_1,
}
from gguf import quants as gguf_quants


def should_quantize(name: str, shape: tuple) -> bool:
    """Decide if a tensor should be Q8_0 quantized.

    Only quantize 2D weight matrices in the transformer layers (encoder_transformer
    and decoder_transformer). These are the linear layers that dominate compute.
    Biases, scales, norms, and non-transformer weights stay as F16.
    """
    is_transformer = name.startswith("encoder_transformer.") or name.startswith(
        "decoder_transformer."
    )
    is_2d_weight = len(shape) == 2 and "weight" in name
    # Q8_0 requires inner dimension divisible by 32 (block size)
    is_block_aligned = len(shape) >= 1 and shape[-1] % 32 == 0
    return is_transformer and is_2d_weight and is_block_aligned


def convert(input_path: str, output_path: str, quant_type_name: str = "q8_0"):
    print(f"Loading safetensors from {input_path}")
    f = safe_open(input_path, framework="numpy")
    tensor_names = f.keys()
    print(f"Found {len(list(tensor_names))} tensors")

    quant_type = QUANT_TYPES[quant_type_name]
    print(f"Quantization type: {quant_type_name.upper()}")

    writer = GGUFWriter(output_path, "mimi")
    writer.add_name("mimi-codec")
    writer.add_description(f"Mimi audio codec with {quant_type_name.upper()} quantized transformers")

    n_quantized = 0
    n_f16 = 0
    total_original = 0
    total_quantized = 0

    for name in sorted(f.keys()):
        tensor = f.get_tensor(name).astype(np.float32)
        original_bytes = tensor.nbytes
        total_original += original_bytes

        if should_quantize(name, tensor.shape):
            quantized = gguf_quants.quantize(tensor, quant_type)
            writer.add_tensor(name, quantized, raw_dtype=quant_type)
            est_bytes = quantized.nbytes
            total_quantized += est_bytes
            n_quantized += 1
            print(f"  Q8_0  {name:60s} {str(tensor.shape):20s} {original_bytes/1024/1024:6.1f}MB -> {est_bytes/1024/1024:.1f}MB")
        else:
            # F16 for everything else
            f16_data = tensor.astype(np.float16)
            writer.add_tensor(name, f16_data, raw_dtype=GGMLQuantizationType.F16)
            est_bytes = f16_data.nbytes
            total_quantized += est_bytes
            n_f16 += 1
            print(f"  F16   {name:60s} {str(tensor.shape):20s} {original_bytes/1024/1024:6.1f}MB -> {est_bytes/1024/1024:.1f}MB")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    actual_size = Path(output_path).stat().st_size
    print(f"\nDone!")
    print(f"  Tensors:    {n_quantized} Q8_0, {n_f16} F16")
    print(f"  Original:   {total_original/1024/1024:.1f} MB")
    print(f"  Estimated:  {total_quantized/1024/1024:.1f} MB")
    print(f"  Actual:     {actual_size/1024/1024:.1f} MB")
    print(f"  Ratio:      {total_original/actual_size:.1f}x smaller")
    print(f"  Output:     {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Mimi safetensors to GGUF")
    parser.add_argument("input", help="Input safetensors file")
    parser.add_argument("-o", "--output", help="Output GGUF file (default: mimi_<quant>.gguf)")
    parser.add_argument("-q", "--quant", default="q8_0", choices=list(QUANT_TYPES.keys()),
                        help="Quantization type for transformer weights (default: q8_0)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: {args.input} not found")
        sys.exit(1)

    output = args.output or f"mimi_{args.quant}.gguf"
    convert(args.input, output, args.quant)
