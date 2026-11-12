//! # Sprint 6: Bare-Metal Assembly, Pipeline Hazard Elimination & Apple M-Series / x86_64 Intrinsics
//!
//! Directly connects **Ingeniería de Computadores I (IC1)**:
//! 1. **RISC Register File & "Regla del 5" (MIPS -> Apple Silicon ARM64 `aarch64`):**
//!    Both MIPS (studied in IC1) and Apple M1/M2/M3/M4 (`aarch64`) are 32-register RISC
//!    architectures (5-bit register specifier $2^5 = 32$) with Load/Store semantics and
//!    fixed 32-bit instruction words. On Apple Silicon, `vtbl1q_u8` performs a 16-entry
//!    4-bit nibble lookup in a single clock cycle inside the 128-bit NEON register file,
//!    while `vfmaq_f32` executes 4-wide Fused Multiply-Add.
//! 2. **Pipeline Data Hazards (Riesgos RAW en etapa `EX`):**
//!    An FMA instruction (`vfmadd231ps` on Intel/AMD AVX2, `fmla` on Apple M-series) has a
//!    4-cycle latency but 2 instructions/cycle throughput. Accumulating into a single scalar
//!    register `sum += a * b` stalls the pipeline for 3 out of every 4 cycles (75% bubbles!).
//!    By unrolling across **4 independent SIMD vector accumulators (`acc0..acc3`)**, we keep
//!    32 floats in flight simultaneously and eliminate RAW stalls.
//! 3. **Inline Assembly Hardware Cycle Counter (`rdtsc` / `cntvct_el0`):**
//!    Reads the CPU's internal silicon timestamp counter directly via `core::arch::asm!`.
//! 4. **Hamming(7,4) Hardware ECC Self-Healing (`IC1 Lab 1 Sesión 3`):**
//!    Protects critical 4-bit block mode/exponent nibbles against single-bit flips (SEU /
//!    cosmic rays / voltage droop) using the exact IC1 parity groups:
//!    `G1 = {1,3,5,7}`, `G2 = {2,3,6,7}`, `G3 = {4,5,6,7}`.

use crate::quant::{QuantizedTensorQ4Golden, QuantizedTensorQ8, Q_BLOCK_SIZE};
use crate::tensor::Tensor2D;

/// Reads the hardware CPU cycle counter directly via inline assembly (`rdtsc` on x86_64,
/// `cntvct_el0` on Apple Silicon / ARM64).
#[inline]
pub fn read_cpu_cycles() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let lo: u32;
        let hi: u32;
        core::arch::asm!(
            "rdtsc",
            out("eax") lo,
            out("edx") hi,
            options(nomem, nostack, preserves_flags)
        );
        ((hi as u64) << 32) | (lo as u64)
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        let cnt: u64;
        core::arch::asm!(
            "mrs {}, cntvct_el0",
            out(reg) cnt,
            options(nomem, nostack, preserves_flags)
        );
        cnt
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        0
    }
}

/// Hardware-accelerated 32-element dot product using explicit CPU SIMD intrinsics
/// (`AVX2 + FMA` on x86_64, `NEON` on Apple Silicon M1/M2/M3/M4) with 4 independent
/// vector accumulators to eliminate Read-After-Write (RAW) pipeline stalls.
#[inline]
pub fn dot32_hardware_simd(a: &[f32; 32], b: &[f32; 32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        use core::arch::x86_64::*;
        let a0 = _mm256_loadu_ps(a.as_ptr());
        let b0 = _mm256_loadu_ps(b.as_ptr());
        let mut acc0 = _mm256_mul_ps(a0, b0);

        let a1 = _mm256_loadu_ps(a.as_ptr().add(8));
        let b1 = _mm256_loadu_ps(b.as_ptr().add(8));
        let mut acc1 = _mm256_mul_ps(a1, b1);

        let a2 = _mm256_loadu_ps(a.as_ptr().add(16));
        let b2 = _mm256_loadu_ps(b.as_ptr().add(16));
        acc0 = _mm256_fmadd_ps(a2, b2, acc0);

        let a3 = _mm256_loadu_ps(a.as_ptr().add(24));
        let b3 = _mm256_loadu_ps(b.as_ptr().add(24));
        acc1 = _mm256_fmadd_ps(a3, b3, acc1);

        // Tree reduction of acc0 + acc1 (8 floats -> 4 floats -> 1 float)
        let sum256 = _mm256_add_ps(acc0, acc1);
        let hi128 = _mm256_extractf128_ps(sum256, 1);
        let lo128 = _mm256_castps256_ps128(sum256);
        let sum128 = _mm_add_ps(lo128, hi128);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let final_sum = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(final_sum)
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        // Apple Silicon (M1/M2/M3/M4) ARM64 NEON 4-Accumulator Pipeline-Unrolled FMA
        use core::arch::aarch64::*;
        let a0 = vld1q_f32(a.as_ptr());
        let b0 = vld1q_f32(b.as_ptr());
        let mut acc0 = vmulq_f32(a0, b0);

        let a1 = vld1q_f32(a.as_ptr().add(4));
        let b1 = vld1q_f32(b.as_ptr().add(4));
        let mut acc1 = vmulq_f32(a1, b1);

        let a2 = vld1q_f32(a.as_ptr().add(8));
        let b2 = vld1q_f32(b.as_ptr().add(8));
        let mut acc2 = vmulq_f32(a2, b2);

        let a3 = vld1q_f32(a.as_ptr().add(12));
        let b3 = vld1q_f32(b.as_ptr().add(12));
        let mut acc3 = vmulq_f32(a3, b3);

        acc0 = vfmaq_f32(acc0, vld1q_f32(a.as_ptr().add(16)), vld1q_f32(b.as_ptr().add(16)));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a.as_ptr().add(20)), vld1q_f32(b.as_ptr().add(20)));
        acc2 = vfmaq_f32(acc2, vld1q_f32(a.as_ptr().add(24)), vld1q_f32(b.as_ptr().add(24)));
        acc3 = vfmaq_f32(acc3, vld1q_f32(a.as_ptr().add(28)), vld1q_f32(b.as_ptr().add(28)));

        let s01 = vaddq_f32(acc0, acc1);
        let s23 = vaddq_f32(acc2, acc3);
        vaddvq_f32(vaddq_f32(s01, s23))
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"),
        target_arch = "aarch64"
    )))]
    {
        // 4-way unrolled scalar fallback to break RAW dependency chain
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;
        for i in (0..32).step_by(4) {
            s0 += a[i] * b[i];
            s1 += a[i + 1] * b[i + 1];
            s2 += a[i + 2] * b[i + 2];
            s3 += a[i + 3] * b[i + 3];
        }
        (s0 + s1) + (s2 + s3)
    }
}

/// Silicon-Optimized Direct `Q4_GoldenOpt` GEMM Kernel:
/// 1. Hoists codebook construction (`powf(alpha)`) OUT of the `M x N` loop: builds the
///    16-entry LUT once per weight block instead of `M` times!
/// 2. Uses 4-way unrolled SIMD FMA (`AVX2` on x86_64 / `NEON` on Apple Silicon M-series)
///    to eliminate Read-After-Write (RAW) pipeline stalls in the `EX` stage.
pub fn matmul_q4_golden_silicon(a: &Tensor2D, b_t_q4: &QuantizedTensorQ4Golden) -> Tensor2D {
    assert_eq!(a.cols, b_t_q4.cols, "Inner dimensions must match");
    let m = a.rows;
    let k = a.cols;
    let n = b_t_q4.rows;
    let blocks_per_row = k / Q_BLOCK_SIZE;

    // Pre-decode each 20-byte block into a cache-aligned 32-float micro-tile once per weight block
    // (or keep 16-entry LUTs), avoiding 512x redundant `powf` calls per block!
    let mut decoded_blocks = vec![[0.0f32; Q_BLOCK_SIZE]; b_t_q4.blocks.len()];
    for (idx, block) in b_t_q4.blocks.iter().enumerate() {
        let m0 = (block.scale_bf16 >> 15) as u8;
        let m1 = (block.alpha_bf16 >> 15) as u8;
        let mode = (m1 << 1) | m0;
        let (scale, param, rho) = if mode < 2 {
            let s = f32::from_bits(((block.scale_bf16 & 0x7FFF) as u32) << 16);
            let p = f32::from_bits(((block.alpha_bf16 & 0x7FFF) as u32) << 16);
            (s, p, 1.0)
        } else {
            let r_idx = (((block.alpha_bf16 & 0x0003) << 4) | (block.scale_bf16 & 0x000F)) as u8;
            let r = 0.06 + (r_idx.min(63) as f32) * (0.94 / 63.0);
            let s = f32::from_bits(((block.scale_bf16 & 0x7FF0) as u32) << 16);
            let p = f32::from_bits(((block.alpha_bf16 & 0x7FFC) as u32) << 16);
            (s, p, r)
        };
        let cb = QuantizedTensorQ4Golden::build_codebook(mode, scale, param, rho);
        let dst = &mut decoded_blocks[idx];
        for byte_idx in 0..(Q_BLOCK_SIZE / 2) {
            let byte = block.nibbles[byte_idx];
            dst[2 * byte_idx] = cb[(byte & 0x0F) as usize];
            dst[2 * byte_idx + 1] = cb[((byte >> 4) & 0x0F) as usize];
        }
    }

    let mut c = Tensor2D::zeros(m, n);
    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(4)
        .min(m.max(1));
    let rows_per_thread = (m + threads - 1) / threads;

    std::thread::scope(|s| {
        for (t_idx, c_chunk) in c.data.chunks_mut(rows_per_thread * n).enumerate() {
            let row_start = t_idx * rows_per_thread;
            let row_count = c_chunk.len() / n;
            let a_slice = &a.data[row_start * k..(row_start + row_count) * k];
            let b_tiles = &decoded_blocks;

            s.spawn(move || {
                let mut a_blocks = vec![[0.0f32; Q_BLOCK_SIZE]; blocks_per_row];
                for i in 0..row_count {
                    let a_row = &a_slice[i * k..(i + 1) * k];
                    for b_idx in 0..blocks_per_row {
                        a_blocks[b_idx].copy_from_slice(
                            &a_row[b_idx * Q_BLOCK_SIZE..(b_idx + 1) * Q_BLOCK_SIZE],
                        );
                    }
                    let c_row = &mut c_chunk[i * n..(i + 1) * n];

                    for j in 0..n {
                        let b_row_tiles = &b_tiles[j * blocks_per_row..(j + 1) * blocks_per_row];
                        let mut sum0 = 0.0f32;
                        let mut sum1 = 0.0f32;
                        let mut b_idx = 0;
                        while b_idx + 1 < blocks_per_row {
                            sum0 += dot32_hardware_simd(&a_blocks[b_idx], &b_row_tiles[b_idx]);
                            sum1 +=
                                dot32_hardware_simd(&a_blocks[b_idx + 1], &b_row_tiles[b_idx + 1]);
                            b_idx += 2;
                        }
                        if b_idx < blocks_per_row {
                            sum0 += dot32_hardware_simd(&a_blocks[b_idx], &b_row_tiles[b_idx]);
                        }
                        c_row[j] = sum0 + sum1;
                    }
                }
            });
        }
    });

    c
}

/// Multi-Threaded Hardware SIMD `INT8` (`Q8_0`) GEMM Kernel using `AVX2` 16-bit/32-bit integer
/// multiply-add (`_mm256_madd_epi16`) on x86_64 or `NEON` on Apple Silicon.
pub fn matmul_q8_0_silicon(a_q: &QuantizedTensorQ8, b_t_q: &QuantizedTensorQ8) -> Tensor2D {
    assert_eq!(a_q.cols, b_t_q.cols);
    let m = a_q.rows;
    let n = b_t_q.rows;
    let blocks_per_row = a_q.cols / Q_BLOCK_SIZE;
    let mut c = Tensor2D::zeros(m, n);

    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(4)
        .min(m.max(1));
    let rows_per_thread = (m + threads - 1) / threads;

    std::thread::scope(|s| {
        for (t_idx, c_chunk) in c.data.chunks_mut(rows_per_thread * n).enumerate() {
            let row_start = t_idx * rows_per_thread;
            let row_count = c_chunk.len() / n;
            let a_blocks = &a_q.blocks;
            let b_blocks = &b_t_q.blocks;

            s.spawn(move || {
                for i in 0..row_count {
                    let global_i = row_start + i;
                    let a_row = &a_blocks[global_i * blocks_per_row..(global_i + 1) * blocks_per_row];
                    let c_row = &mut c_chunk[i * n..(i + 1) * n];

                    for j in 0..n {
                        let b_row = &b_blocks[j * blocks_per_row..(j + 1) * blocks_per_row];
                        let mut total = 0.0f32;
                        for (blk_a, blk_b) in a_row.iter().zip(b_row.iter()) {
                            let int_dot = dot32_i8_simd(&blk_a.qs, &blk_b.qs);
                            total += (int_dot as f32) * (blk_a.scale * blk_b.scale);
                        }
                        c_row[j] = total;
                    }
                }
            });
        }
    });

    c
}

#[inline]
fn dot32_i8_simd(a: &[i8; 32], b: &[i8; 32]) -> i32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let va_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128(a.as_ptr() as *const __m128i));
        let vb_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128(b.as_ptr() as *const __m128i));
        let prod_lo = _mm256_madd_epi16(va_lo, vb_lo);

        let va_hi = _mm256_cvtepi8_epi16(_mm_loadu_si128(a.as_ptr().add(16) as *const __m128i));
        let vb_hi = _mm256_cvtepi8_epi16(_mm_loadu_si128(b.as_ptr().add(16) as *const __m128i));
        let prod_hi = _mm256_madd_epi16(va_hi, vb_hi);

        let sum256 = _mm256_add_epi32(prod_lo, prod_hi);
        let hi128 = _mm256_extracti128_si256(sum256, 1);
        let lo128 = _mm256_castsi256_si128(sum256);
        let sum128 = _mm_add_epi32(lo128, hi128);
        let sum64 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
        let sum32 = _mm_add_epi32(sum64, _mm_srli_si128(sum64, 4));
        _mm_cvtsi128_si32(sum32)
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut s0 = 0i32;
        let mut s1 = 0i32;
        for k in (0..32).step_by(2) {
            s0 += (a[k] as i32) * (b[k] as i32);
            s1 += (a[k + 1] as i32) * (b[k + 1] as i32);
        }
        s0 + s1
    }
}

// =========================================================================================
// Hamming(7,4) Hardware ECC Self-Healing (IC1 Lab 1 Sesión 3 - Chuleta Maestra IC1)
//
// Parity Groups (1-indexed bit positions b1..b7):
//   Data bits (4 bits): d3=b7, d2=b6, d1=b5, d0=b3
//   Parity bits (3 bits):
//     p1 (b1) covers G1 = {1, 3, 5, 7} -> p1 = d0 ^ d1 ^ d3
//     p2 (b2) covers G2 = {2, 3, 6, 7} -> p2 = d0 ^ d2 ^ d3
//     p3 (b4) covers G3 = {4, 5, 6, 7} -> p3 = d1 ^ d2 ^ d3
//   Syndrome s = (s3 << 2) | (s2 << 1) | s1 gives the exact 1-based position (1..7)
//   of the corrupted bit! Flipping `codeword ^ (1 << (s - 1))` heals the bit in 1 cycle!
// =========================================================================================

/// Encodes a 4-bit data nibble (`0..15`) into a 7-bit Hamming(7,4) codeword (`bits 0..6`).
#[inline]
pub fn hamming74_encode(nibble: u8) -> u8 {
    let d0 = (nibble) & 1;
    let d1 = (nibble >> 1) & 1;
    let d2 = (nibble >> 2) & 1;
    let d3 = (nibble >> 3) & 1;

    let p1 = d0 ^ d1 ^ d3;
    let p2 = d0 ^ d2 ^ d3;
    let p3 = d1 ^ d2 ^ d3;

    // Bit positions 1..7 mapped to bits 0..6:
    // b1=p1, b2=p2, b3=d0, b4=p3, b5=d1, b6=d2, b7=d3
    p1 | (p2 << 1) | (d0 << 2) | (p3 << 3) | (d1 << 4) | (d2 << 5) | (d3 << 6)
}

/// Decodes a potentially corrupted 7-bit Hamming(7,4) codeword, returning `(healed_nibble, syndrome)`.
/// If `syndrome != 0`, `syndrome` (`1..=7`) is the exact 1-based bit position that flipped and was healed!
#[inline]
pub fn hamming74_decode_heal(mut codeword: u8) -> (u8, u8) {
    let b1 = (codeword) & 1;
    let b2 = (codeword >> 1) & 1;
    let b3 = (codeword >> 2) & 1;
    let b4 = (codeword >> 3) & 1;
    let b5 = (codeword >> 4) & 1;
    let b6 = (codeword >> 5) & 1;
    let b7 = (codeword >> 6) & 1;

    // Syndrome groups from IC1 Chuleta: G1={1,3,5,7}, G2={2,3,6,7}, G3={4,5,6,7}
    let s1 = b1 ^ b3 ^ b5 ^ b7;
    let s2 = b2 ^ b3 ^ b6 ^ b7;
    let s3 = b4 ^ b5 ^ b6 ^ b7;
    let syndrome = (s3 << 2) | (s2 << 1) | s1;

    if syndrome >= 1 && syndrome <= 7 {
        // Self-heal the exact corrupted bit in 1 XOR instruction!
        codeword ^= 1u8 << (syndrome - 1);
    }

    let d0 = (codeword >> 2) & 1;
    let d1 = (codeword >> 4) & 1;
    let d2 = (codeword >> 5) & 1;
    let d3 = (codeword >> 6) & 1;
    let healed_nibble = d0 | (d1 << 1) | (d2 << 2) | (d3 << 3);

    (healed_nibble, syndrome)
}
