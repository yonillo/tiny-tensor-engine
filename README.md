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
| **5. `Q4_GoldenOpt` (Sign-Bit + 6-bit Mantissa + Golden Search)** | **`0.16 MB`** | **`6.40x`** | **`1.4554e-3`** | **`1.2857e-2`** | **`24.46 dB`** | **`8.0671e-3` (`2.32x`)** |

> **Algebraic Breakthrough (`Q4_Hadamard`):** By rotating each 32-weight block into the normalized Walsh-Hadamard orthonormal basis ($\mathbf{w}' = \mathbf{H}_{32}\mathbf{w}$, where $\mathbf{H}_{32}^T \mathbf{H}_{32} = \mathbf{I}$ and $\|\mathbf{H}_{32}\mathbf{w}\|_2 = \|\mathbf{w}\|_2$), coordinate outliers are dissolved uniformly across all 32 dimensions via the Fast Walsh-Hadamard Transform (`O(K log K)` additions/subtractions). This boosts 4-bit SQNR by **`+4.38 dB` (`2.74x` less noise power) with ZERO extra memory (`6.40x` compression)**, and enables direct rotated-basis GEMM: $(\mathbf{A}\mathbf{H}_{32})(\mathbf{B}\mathbf{H}_{32})^T = \mathbf{A}\mathbf{B}^T$.
>
> **Statistical Breakthrough (`Q4_Stats`):** Isolating $>3.2\hat{\sigma}$ outliers via Trimmed-MAD + OLS scale regression improves 4-bit SQNR by **`+6.46 dB`** (**`4.43x` less quantization noise power**) and makes downstream neural layer matrix multiplication **`2.13x` more accurate** at **`5.73x` memory compression**.
>
> **Optimization Breakthrough (`Q4_GoldenOpt`):** In any 20-byte block (`scale_bf16: u16`, `param_bf16: u16`, `nibbles: [u8; 16]`), both scale parameters $s > 0$ and $p > 0$ are strictly positive magnitudes, leaving their **two IEEE-754 sign bits (`0x8000`) 100% unused**. `Q4_GoldenOpt` steals these 2 bits to switch per-block between **4 adaptive codebook topologies**, and in multi-outlier blocks (`Modes 10 & 11`) embeds a **6-bit secondary outlier ratio sub-channel ($\rho = T_2 / T_1$)** inside the 4 LSB mantissa bits of `scale_bf16` and 2 LSB mantissa bits of `param_bf16`, solved via **derivative-free Golden Section Search** ($r = \frac{\sqrt{5}-1}{2} \approx 0.618034$) and **closed-form decoupled OLS ($\nabla J = \mathbf{0}$)**. Result: **`24.46 dB` SQNR (`+7.65 dB` over `Q4_0` and `+1.19 dB` over `Q4_Stats`), `2.32x` lower neural layer drift, and `27.66 ms` direct compressed-domain GEMM at full `6.40x` compression (`0.16 MB`, zero external outlier tables)!**

---

## 🔬 Sprint 4: 4-Layer Deep Neural Network Cascade (`RMSNorm + SiLU` + `Top-1 Token %`)

Passing 512 token activation vectors through 4 cascaded non-linear heavy-tailed layers (`4.00 MB` in `FP32`):

| 4-Layer Cascaded Engine (`4.00 MB FP32`) | Total RAM | Logits SQNR | Cosine Similarity | Top-1 Token Agreement |
| :--- | ---: | ---: | ---: | ---: |
| **1. `Q4_0` (Standard 4-bit)** | `0.63 MB` (`6.40x`) | `10.35 dB` | `95.4469%` | `65.04%` |
| **2. `Q4_Hadamard` (Orthonormal $\mathbf{H}_{32}$)** | `0.63 MB` (`6.40x`) | `14.63 dB` | `98.2737%` | `76.95%` |
| **3. `Q4_Stats` (Sparse Outlier + OLS)** | `0.70 MB` (`5.73x`) | `16.82 dB` | `98.9614%` | `84.77%` |
| **4. `Q4_GoldenOpt` (Sign-Bit + 6-bit Mantissa + Golden)** | **`0.63 MB` (`6.40x`)** | **`18.02 dB`** | **`99.2094%`** | **`85.55%`** |

---

## 🌍 Sprint 5: Real Pretrained LLM Benchmark (`SmolLM2-135M` `.safetensors` + `AEDV` Suite)

Using HTTP Range Streaming (`benchmarks/download_real_llm.py`, **IC2 Tema 1/2**) and our zero-dependency Rust `.safetensors` binary parser (`src/safetensors.rs`, **IC2 Tema 6**), we benchmarked **4 real pretrained Transformer layers (`1,327,104` parameters)** from `HuggingFaceTB/SmolLM2-135M`, coupled with **AEDV Exploratory Data Analysis** (Fisher Excess Kurtosis, IQR, Yeo-Johnson $\lambda^*$, and automatic interactive Plotly HTML dashboard generation in `reports/aedv_smollm2_dashboard.html`):

| Real Tensor (`SmolLM2-135M`) | Shape | Excess Kurtosis | Outliers (`>3σ`) | Yeo-Johnson $\lambda^*$ | `Q4_0` (`6.40x`) | `Q4_Hadamard` (`6.40x`) | `Q4_Stats` (`5.73x`) | **`Q4_GoldenOpt` (`6.40x`)** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `layers.0.self_attn.q_proj.weight` | `576×576` | `7.99` | `1.599%` | `1.024` | `18.92 dB` | `21.10 dB` | `22.89 dB` | **`23.04 dB` (`+4.12 dB`)** |
| `layers.0.self_attn.o_proj.weight` | `576×576` | **`25.00`** | `1.131%` | `0.990` | `18.38 dB` | `21.13 dB` | `22.59 dB` | **`23.26 dB` (`+4.88 dB`)** |
| `layers.15.self_attn.q_proj.weight` | `576×576` | `3.14` | `1.106%` | `1.000` | `19.78 dB` | `21.09 dB` | `21.84 dB` | **`23.02 dB` (`+3.24 dB`)** |
| `layers.29.self_attn.o_proj.weight` | `576×576` | `2.58` | `0.555%` | `1.008` | `20.13 dB` | `21.06 dB` | `21.65 dB` | **`23.03 dB` (`+2.90 dB`)** |

### 4-Layer Real `SmolLM2-135M` Cascade Fidelity (`576 Tokens × 576 Channels`)

| Real `SmolLM2-135M` 4-Layer Cascade | Total RAM | Logits SQNR | Cosine Similarity | Top-1 Token Agreement |
| :--- | ---: | ---: | ---: | ---: |
| **1. `Q4_0` (Standard 4-bit)** | `0.79 MB` (`6.40x`) | `12.69 dB` | `97.2995%` | `71.01%` |
| **2. `Q4_Hadamard` (Orthonormal $\mathbf{H}_{32}$)** | `0.79 MB` (`6.40x`) | `14.72 dB` | `98.3140%` | `76.39%` |
| **3. `Q4_Stats` (Sparse Outlier + OLS)** | `0.89 MB` (`5.73x`) | `15.99 dB` | `98.7412%` | `78.12%` |
| **4. `Q4_GoldenOpt` (Sign-Bit + 6-bit Mantissa + Golden)** | **`0.79 MB` (`6.40x`)** | **`16.41 dB`** | **`98.8585%`** | **`78.12%`** |

---

## ⚡ Sprint 6: Bare-Metal Assembly, SIMD Pipeline Unrolling & IC1 Hamming(7,4) Hardware ECC

To push hardware efficiency down to the silicon boundary, Sprint 6 introduces native instruction-set extensions for both **Intel/AMD (`x86_64` AVX2 + FMA)** and **Apple Silicon M1–M4 (`aarch64` NEON)**:

1. **Bare-Metal Cycle Counters via Inline Assembly:**
   - On `x86_64`: Reads high-resolution timestamp counter via `core::arch::x86_64::_rdtsc()` instruction.
   - On `aarch64` (Apple Silicon M-Series): Reads the virtual cycle register via inline assembly `asm!("mrs {}, cntvct_el0", out(reg) val)`.
2. **SIMD Dot-Products & RAW Hazard Pipeline Elimination:**
   - **`Q8_0 Silicon`:** Vectorizes integer dot-products with `_mm256_madd_epi16` (`x86_64`) / `vmlal_s8` (`aarch64 NEON`), achieving **`105.77 GOPS`** (**`3.97x` speedup** over scalar loop, `0.00e0` numerical error).
   - **`Q4_GoldenOpt Silicon`:** Hoists 20-byte block codebook dequantization out of inner loops and unrolls the dot product across **4 independent vector accumulators** (`_mm256_fmadd_ps` / `vfmaq_f32`). This completely eliminates the 4-cycle Read-After-Write (`RAW`) execution stall in the CPU arithmetic pipeline, dropping latency from `25.00 ms` to **`3.77 ms`** (**`6.64x` speedup**, **`71.24 GFLOPS`** directly on compressed weights!).
3. **`Hamming(7,4)` Hardware Self-Healing ECC (ULPGC IC1 Specification):**
   - Implements hardware Single Error Correction (SEC) per 4-bit nibble using parity bit positions $\{p_1, p_2, p_4\}$ and data $\{d_1, d_2, d_3, d_4\}$ spanning parity groups $G_1=\{1,3,5,7\}$, $G_2=\{2,3,6,7\}$, $G_3=\{4,5,6,7\}$.
   - In radiation or row-hammer induced bit flips, calculating the syndrome $S = (s_3, s_2, s_1)_2$ locates and self-heals corrupted DRAM bits in **1 single XOR cycle** (`cw ^= 1 << (syndrome - 1)`). Verified with **100.0% recovery (`112/112` single-bit flips)**.

### Silicon Execution Benchmarks (`512 × 512` Matrix Multiplication)

| Quantized Silicon Kernel (`512×512`) | Latency | CPU Cycles | Throughput | Speedup vs Baseline | Numerical Diff |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **1. `Q8_0` Baseline (Single-Core Scalar)** | `10.07 ms` | `29,382,748` | `26.67 GOPS` | `1.00x` | Baseline |
| **2. `Q8_0` Silicon (`_mm256_madd_epi16` / `NEON`)** | **`2.54 ms`** | **`7,407,122`** | **`105.77 GOPS`** | **`3.97x`** | **`0.00e0`** |
| **3. `Q4_GoldenOpt` Direct (Per-Row Dequant)** | `25.00 ms` | `72,971,974` | `10.74 GFLOPS` | `1.00x` | Baseline |
| **4. `Q4_GoldenOpt` Silicon (Hoisted + 4x FMA)** | **`3.77 ms`** | **`10,997,138`** | **`71.24 GFLOPS`** | **`6.64x`** | **`1.04e-7`** |

---

## 🏗️ Architecture

```
tiny-tensor-engine/
├── Cargo.toml                      # Release profile: opt-level=3, fat LTO, codegen-units=1
├── src/
│   ├── lib.rs                      # Library root & numerical equivalence unit test suite
│   ├── main.rs                     # Sprints 1–6 hardware, quantization & silicon benchmark runner
│   ├── tensor.rs                   # Contiguous Row-Major Tensor2D & Box-Muller / Heavy-Tailed PRNG
│   ├── matmul.rs                   # Naive, Cache-Tiled, Stride-1 SIMD, and Scoped Multi-Thread GEMM
│   ├── quant.rs                    # Q8_0, Q4_0, Q4_Hadamard, Q4_Stats, and Q4_GoldenOpt + Direct GEMM
│   ├── safetensors.rs              # Zero-dependency .safetensors binary container parser (IC2 Tema 6)
│   ├── aedv.rs                     # AEDV Exploratory Suite (Yeo-Johnson, Kurtosis, Interactive Plotly HTML)
│   └── silicon.rs                  # AVX2/NEON intrinsics, rdtsc inline asm, 4x FMA unrolling & Hamming(7,4) ECC
├── data/
│   └── smollm2_135m_real.safetensors # 4 real pretrained 576x576 layers from HuggingFace SmolLM2-135M
├── reports/
│   └── aedv_smollm2_dashboard.html # Interactive AEDV Plotly telemetry dashboard
└── benchmarks/
    ├── download_real_llm.py        # HTTP Range downloader for HuggingFace .safetensors layers
    └── python_baseline.py          # Pure Python baseline reference for speedup comparison
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
