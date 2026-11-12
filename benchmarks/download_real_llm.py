#!/usr/bin/env python3
"""
Downloads real pretrained Transformer weight tensors from HuggingFace (`SmolLM2-135M`)
using HTTP Range Requests (IC2 Tema 1/2: TCP/IP & HTTP byte-range streaming) and packages
them into a standalone `.safetensors` binary file (`data/smollm2_135m_real.safetensors`)
for bare-metal Rust benchmarking (IC2 Tema 6: File Systems & Binary Formats).
"""
import json
import os
import struct
import urllib.request

HF_URL = "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/model.safetensors"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT_FILE = os.path.join(OUT_DIR, "smollm2_135m_real.safetensors")

def fetch_range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "tiny-tensor-engine/0.4 (ULPGC DeepTech Research)",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[1/3] Fetching 8-byte .safetensors header length from {HF_URL}...")
    prefix = fetch_range(HF_URL, 0, 7)
    header_len = struct.unpack("<Q", prefix)[0]
    print(f"      Header JSON size: {header_len:,} bytes")

    raw_header = fetch_range(HF_URL, 8, 8 + header_len - 1)
    header = json.loads(raw_header.decode("utf-8"))

    # Select real pretrained attention & MLP projection tensors (576 x 576 in SmolLM2-135M)
    target_names = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.15.self_attn.q_proj.weight",
        "model.layers.29.self_attn.o_proj.weight",
    ]

    new_header = {
        "__metadata__": {
            "source_model": "HuggingFaceTB/SmolLM2-135M",
            "format": "pt",
        }
    }
    payload_chunks = []
    current_offset = 0

    for name in target_names:
        if name not in header:
            raise KeyError(f"Tensor {name} not found in remote .safetensors header")
        meta = header[name]
        dtype = meta["dtype"]
        shape = meta["shape"]
        start_rel, end_rel = meta["data_offsets"]
        byte_len = end_rel - start_rel
        abs_start = 8 + header_len + start_rel
        abs_end = 8 + header_len + end_rel - 1

        print(f"[2/3] Downloading real tensor '{name}' (dtype={dtype}, shape={shape}, {byte_len:,} bytes)...")
        tensor_bytes = fetch_range(HF_URL, abs_start, abs_end)
        assert len(tensor_bytes) == byte_len, f"Expected {byte_len} bytes, got {len(tensor_bytes)}"

        new_header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [current_offset, current_offset + byte_len],
        }
        payload_chunks.append(tensor_bytes)
        current_offset += byte_len

    new_header_bytes = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    # Pad header to 8-byte alignment per safetensors spec
    pad_len = (8 - (len(new_header_bytes) % 8)) % 8
    new_header_bytes += b" " * pad_len

    with open(OUT_FILE, "wb") as f:
        f.write(struct.pack("<Q", len(new_header_bytes)))
        f.write(new_header_bytes)
        for chunk in payload_chunks:
            f.write(chunk)

    total_kb = os.path.getsize(OUT_FILE) / 1024.0
    print(f"[3/3] Saved standalone real LLM .safetensors to: {OUT_FILE} ({total_kb:.1f} KB)")

if __name__ == "__main__":
    main()
