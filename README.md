# ⚡ `tiny-tensor-engine`

**Zero-dependency, bare-metal tensor & quantized inference runtime written in 100% pure Rust.**

[![Language: Rust](https://img.shields.io/badge/Language-Rust_2021-orange.svg)](https://www.rust-lang.org/)
[![Dependencies: Zero](https://img.shields.io/badge/Dependencies-Zero-brightgreen.svg)](#architecture)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

`tiny-tensor-engine` is a from-scratch systems engineering project exploring the hardware bottlenecks of modern AI inference—specifically **the Memory Wall (L1/L2/L3 cache hierarchy)**, **SIMD vectorization (`AVX2`/`FMA`)**, **lock-free multi-core parallelism**, and **statistical block-wise weight quantization (`Q8_0`)**.

Built without external crates (`ndarray`, `rayon`, or `blas` are intentionally omitted), every memory layout, thread scheduler, PRNG, and matrix kernel is implemented from first principles using only the Rust standard library (`std`).

---

## 🚀 Benchmark Results (`1024 × 1024` GEMM — 2.15 Billion FLOPs)

Measured on a 16-thread x86_64 CPU (`rustc 1.99.1 -C opt-level=3 -C target-cpu=native`):

| Engine / Kernel Implementation | Latency (`1024×1024`) | Throughput (`GFLOPS`) | Speedup vs Rust Naive | Speedup vs Python | Max Error vs FP32 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **0. Pure Python (`i -> j -> k`)** | `62,595.64 ms` | `0.03 GFLOPS` | `0.05x` | `1.0x` | — |
| **1. Rust Naive (`i -> j -> k`, cache thrashing)** | `3,382.31 ms` | `0.63 GFLOPS` | `1.00x` | `18.5x` | `0.00e0` |
| **2. Rust Cache-Tiled (`64×64` L1 blocking)** | `134.90 ms` | `15.92 GFLOPS` | `25.07x` | `464.0x` | `0.00e0` |
| **3. Rust Cache-Linear (`i -> k -> j` + Auto-SIMD)** | `106.25 ms` | `20.21 GFLOPS` | `31.83x` | `589.1x` | `0.00e0` |
| **4. Rust Parallel SIMD (`16 CPU threads`)** | **`27.25 ms`** | **`78.81 GFLOPS`** | **`124.13x`** | **`2,297.1x`** | **`0.00e0`** |

> **Why is `Parallel SIMD` 124x faster than `Naive Rust` in the exact same language?**
> In a `1024 × 1024` matrix (`4.00 MB`), the naive `i -> j -> k` inner loop accesses `B[k * N + j]`, jumping `4,096 bytes` on every step. Because a CPU cache line is `64 bytes` (`16 floats`), **93.75% of every cache line fetched from RAM is wasted** and evicted before reuse. Reordering the loop to `i -> k -> j` and blocking into `64 × 64` tiles achieves 100% spatial locality (stride-1 access), unlocking hardware prefetching and single-cycle `AVX2` Fused Multiply-Add (`vfmadd231ps`) vector instructions.

---

## 🧠 Statistical Block Quantization (`Q8_0`)

Modern LLMs are memory-bandwidth bound rather than compute bound. Inspired by `GGML` / `llama.cpp`, `tiny-tensor-engine` implements block-wise symmetric 8-bit integer quantization (`BlockQ8_0`):

$$\Delta_b = \frac{\max_{i \in b} |X_i|}{127}, \qquad Q_i = \text{clip}\left(\text{round}\left(\frac{X_i}{\Delta_b}\right), -127, 127\right)$$

Each block of `32` weights (`128 bytes` in `FP32`) is packed into `1 × f32` scale factor (`4 bytes`) + `32 × i8` quantized weights (`32 bytes`) = **`36 bytes` per block** (**`3.56x` memory compression**, saving **71.9%** of RAM bandwidth).

### `Q8_0` Telemetry on Gaussian Neural Weights $X \sim \mathcal{N}(0, 0.02^2)$ (`512 × 512`)

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Sample Mean ($\mu$) & Std Dev ($\sigma$)** | $\mu = +0.000006, \quad \sigma = 0.019985$ | Box-Muller transform + Xorshift64* PRNG |
| **$3\sigma$ Tail Outlier Ratio** | `0.291%` | Theoretical Gaussian tail: `0.270%` |
| **FP32 Footprint $\rightarrow$ `Q8_0` Footprint** | `1.00 MB` $\rightarrow$ **`0.28 MB`** | **`3.56x` RAM compression** (`71.9%` saved) |
| **Root Mean Squared Error (`RMSE`)** | `1.0717e-4` | Max absolute weight error: `3.607e-4` |
| **Signal-to-Quantization-Noise Ratio (`SQNR`)** | **`45.41 dB`** | High-fidelity reconstruction (> 35 dB baseline) |
| **Quantized Integer GEMM (`matmul_q8_0`)** | **`9.32 ms`** | Integer `i32` dot-product accumulation |

---

## 🏗️ Architecture

```
tiny-tensor-engine/
├── Cargo.toml                  # Release profile: opt-level=3, fat LTO, codegen-units=1
├── src/
│   ├── lib.rs                  # Library root & numerical equivalence unit test suite
│   ├── main.rs                 # Hardware telemetry & GEMM / Quantization benchmark runner
│   ├── tensor.rs               # Contiguous Row-Major Tensor2D & Box-Muller Gaussian PRNG
│   ├── matmul.rs               # Naive, Cache-Tiled, Stride-1 SIMD, and Scoped Multi-Thread GEMM
│   └── quant.rs                # Statistical profiling, BlockQ8_0 quantization & INT8 GEMM
└── benchmarks/
    └── python_baseline.py      # Pure Python baseline reference for speedup comparison
```

---

## 🛠️ Quick Start

### 1. Run the Unit Test Suite
```bash
RUSTFLAGS="-C target-cpu=native" cargo test --release
```

### 2. Run the Full Hardware Benchmark Suite
```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release
```

---

## 🗺️ Roadmap

- [x] **Sprint 1:** Contiguous `Tensor2D` layout, cache-tiled GEMM, stride-1 auto-vectorized SIMD, and lock-free `std::thread::scope` multi-threading.
- [x] **Sprint 2:** Statistical tensor profiling ($\mu, \sigma^2, 3\sigma$ outliers, SQNR) and `Q8_0` symmetric block quantization with integer dot-product GEMM.
- [ ] **Sprint 3:** Ultra-low-bit `Q4_0` (4-bit nibble packing, 2 weights per byte) and outlier-aware mixed-precision quantization.
- [ ] **Sprint 4:** Zero-copy Python bindings via `PyO3` (`import tiny_tensor_engine`).

---

## 👤 Author

**Yone Suárez** — Data Science & Engineering (ULPGC)  
🌐 [yonesuarez.es](https://www.yonesuarez.es) • 💼 [LinkedIn](https://linkedin.com/in/yone-suarez) • 🔗 [GitHub](https://github.com/yonillo)
