# ⚡ `tiny-tensor-engine`

**Zero-dependency, bare-metal tensor & quantized inference runtime written in 100% pure Rust.**

[![Language: Rust](https://img.shields.io/badge/Language-Rust_2021-orange.svg)](https://www.rust-lang.org/)
[![Dependencies: Zero](https://img.shields.io/badge/Dependencies-Zero-brightgreen.svg)](#architecture)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

`tiny-tensor-engine` is a from-scratch systems engineering project exploring the hardware and statistical bottlenecks of modern AI inference—specifically **the Memory Wall (L1/L2/L3 cache hierarchy)**, **SIMD vectorization (`AVX2`/`FMA`)**, **lock-free multi-core parallelism**, and **outlier-aware 4-bit statistical weight quantization (`Q4_Stats`)**.

Built without external crates (`ndarray`, `rayon`, or `blas` are intentionally omitted), every memory layout, thread scheduler, PRNG, bitwise nibble packer, and statistical estimator is implemented from first principles using only the Rust standard library (`std`).

---

## 🚀 Sprint 1: Silicon GEMM Benchmark (`1024 × 1024` — 2.15 Billion FLOPs)

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

## 🧠 Sprint 2 & 3: Extreme 4-Bit Statistical Quantization (`Q4_0` vs `Q4_Stats`)

### The Problem with Standard 4-Bit Quantization (`Q4_0`)
Standard 4-bit block quantization (`Q4_0`) packs two 4-bit weights into each byte (`u8`) using a block-level scale $\Delta_b = \max_{i \in b} |X_i| / 7$. However, real LLM weights exhibit **heavy-tailed emergent outliers** ($10\times$ magnitude). When a single outlier appears inside a 32-weight block:
1. **Grid Collapse:** The outlier stretches $\Delta_b$ by $10\times$, crushing the other 31 normal weights in the block toward `0`.
2. **Small-Sample Masking ($n = 32$):** Naive block mean/variance estimators are polluted by the outlier itself, failing to isolate it.

### Our Solution: `Q4_Stats` (Trimmed-MAD + Lloyd-Max Gaussian Centroids + OLS Scale Refinement)
`Q4_Stats` combines three statistical methods per 32-weight block:
1. **Trimmed-MAD Dispersion Estimator:** Sorts absolute values in the 32-element block and estimates $\hat{\sigma}_{\text{robust}}$ from the inner 28 elements ($87.5\%$ core), making outlier detection immune to the small-sample masking effect.
2. **Non-Uniform Lloyd-Max Gaussian Centroids:** Maps 4-bit nibbles (`0..15`) to the 16 minimum-MSE centroids of $\mathcal{N}(0, 1)$, concentrating resolution near zero where $68.2\%$ of weights reside, while isolating $>3.2\hat{\sigma}$ outliers into a sparse table.
3. **Ordinary Least Squares (OLS) Scale Calibration:** Refines the block scale $s_b^*$ via closed-form linear regression without intercept:
   $$s_b^* = \frac{\sum_{i \in \text{inliers}} x_i \cdot c_{q_i}}{\sum_{i \in \text{inliers}} c_{q_i}^2}$$

### Head-to-Head Benchmark on Heavy-Tailed LLM Weights (`512 × 512`, `FP32 = 1.00 MB`)

| Quantization Format | Size (`MB`) | RAM Compression | RMSE Error | Max Abs Error | SQNR (`dB`) | Layer Output Drift |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **1. `Q8_0` (8-bit Symmetric Block)** | `0.28 MB` | `3.56x` | `2.0145e-4` | `2.8912e-3` | `41.64 dB` | — |
| **2. `Q4_0` (4-bit Canonical Basis)** | `0.16 MB` | `6.40x` | `3.5120e-3` | `4.9320e-2` | `16.81 dB` | `1.8740e-2` (`1.00x`) |
| **3. `Q4_Hadamard` (4-bit Orthonormal Basis $H_{32}$)** | **`0.16 MB`** | **`6.40x`** | **`2.1220e-3`** | **`3.3063e-2`** | **`21.18 dB`** | **`1.6909e-2` (`1.11x`)** |
| **4. `Q4_Stats` (4-bit Outlier-Aware OLS)** | **`0.17 MB`** | **`5.73x`** | **`1.6692e-3`** | **`9.6453e-3`** | **`23.27 dB`** | **`8.8033e-3` (`2.13x`)** |
| **5. `Q4_GoldenOpt` (IEEE-754 Sign-Bit Stealing + Golden Search)** | **`0.16 MB`** | **`6.40x`** | **`1.5209e-3`** | **`1.9680e-2`** | **`24.08 dB`** | **`8.2011e-3` (`2.29x`)** |

> **Algebraic Breakthrough (`Q4_Hadamard`):** By rotating each 32-weight block into the normalized Walsh-Hadamard orthonormal basis ($\mathbf{w}' = \mathbf{H}_{32}\mathbf{w}$, where $\mathbf{H}_{32}^T \mathbf{H}_{32} = \mathbf{I}$ and $\|\mathbf{H}_{32}\mathbf{w}\|_2 = \|\mathbf{w}\|_2$), coordinate outliers are dissolved uniformly across all 32 dimensions via the Fast Walsh-Hadamard Transform (`O(K log K)` additions/subtractions). This boosts 4-bit SQNR by **`+4.38 dB` (`2.74x` less noise power) with ZERO extra memory (`6.40x` compression)**, and enables direct rotated-basis GEMM: $(\mathbf{A}\mathbf{H}_{32})(\mathbf{B}\mathbf{H}_{32})^T = \mathbf{A}\mathbf{B}^T$.
>
> **Statistical Breakthrough (`Q4_Stats`):** Isolating $>3.2\hat{\sigma}$ outliers via Trimmed-MAD + OLS scale regression improves 4-bit SQNR by **`+6.46 dB`** (**`4.43x` less quantization noise power**) and makes downstream neural layer matrix multiplication **`2.13x` more accurate** at **`5.73x` memory compression**.
>
> **Optimization Breakthrough (`Q4_GoldenOpt`):** In any 20-byte block (`scale_bf16: u16`, `param_bf16: u16`, `nibbles: [u8; 16]`), both scale parameters $s > 0$ and $p > 0$ are strictly positive magnitudes, leaving their **two IEEE-754 sign bits (`0x8000`) 100% unused**. `Q4_GoldenOpt` steals these 2 bits to switch per-block between **4 adaptive codebook topologies** (a 16-level power-law manifold $w_k(\alpha) = \text{sign}(c_k)|c_k|^\alpha$ for Gaussian blocks, and decoupled dual-scale inlier/outlier manifolds for heavy-tailed blocks), solved via **derivative-free Golden Section Search** ($r = \frac{\sqrt{5}-1}{2} \approx 0.618034$) and **closed-form decoupled OLS ($\nabla J = \mathbf{0}$)**. Result: **`24.08 dB` SQNR (`+7.27 dB` over `Q4_0` and `+0.81 dB` over `Q4_Stats`) and `2.29x` lower neural layer drift at full `6.40x` compression (`0.16 MB`, zero external outlier tables)!**

---

## 🏗️ Architecture

```
tiny-tensor-engine/
├── Cargo.toml                  # Release profile: opt-level=3, fat LTO, codegen-units=1
├── src/
│   ├── lib.rs                  # Library root & numerical equivalence unit test suite
│   ├── main.rs                 # Hardware telemetry & GEMM / Quantization benchmark runner
│   ├── tensor.rs               # Contiguous Row-Major Tensor2D & Box-Muller / Heavy-Tailed PRNG
│   ├── matmul.rs               # Naive, Cache-Tiled, Stride-1 SIMD, and Scoped Multi-Thread GEMM
│   └── quant.rs                # Q8_0, Q4_0 Bitwise Nibble Packing, and Q4_Stats (Trimmed-MAD + OLS)
└── benchmarks/
    └── python_baseline.py      # Pure Python baseline reference for speedup comparison
```

---

## 🛠️ Quick Start

```bash
# Run unit tests
RUSTFLAGS="-C target-cpu=native" cargo test --release

# Run full hardware & statistical quantization benchmarks
RUSTFLAGS="-C target-cpu=native" cargo run --release
```

---

## 👤 Author

**Yone Suárez** — Data Science & Engineering (ULPGC)  
🌐 [yonesuarez.es](https://www.yonesuarez.es) • 💼 [LinkedIn](https://linkedin.com/in/yone-suarez) • 🔗 [GitHub](https://github.com/yonillo)
