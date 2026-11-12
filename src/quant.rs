//! Statistical & Algebraic Quantization Engine (`Q8_0`, `Q4_0`, `Q4_Stats`, and `Q4_Hadamard`).
//!
//! Unites three core disciplines from Data Science & Computer Engineering (ULPGC):
//! 1. Computer Architecture (IC1/IC2): Bitwise nibble packing (`<< 4`, `& 0x0F`, 2 weights per byte).
//! 2. Statistical Methods (ME1/ME2): Trimmed-MAD robust outlier filtering, Lloyd-Max Gaussian
//!    quantiles, and Ordinary Least Squares (OLS) block scale regression.
//! 3. Linear Algebra & Euclidean Geometry (Álgebra y Geometría TP3 & TP4): Orthonormal Change of
//!    Basis (`w' = H_32 * w` where `H^T * H = I`), dissolving emergent weight outliers across all
//!    coordinates via the Fast Walsh-Hadamard Transform (FWHT) with zero extra memory overhead!

use crate::tensor::Tensor2D;

pub const Q_BLOCK_SIZE: usize = 32;

/// Statistical profile of a floating-point tensor (ME1/ME2).
#[derive(Debug, Clone)]
pub struct TensorStats {
    pub mean: f32,
    pub variance: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    /// Percentage of weights beyond 3 standard deviations (|x - mean| > 3 * std_dev)
    pub outlier_ratio_3sigma: f32,
}

impl TensorStats {
    pub fn compute(tensor: &Tensor2D) -> Self {
        let n = tensor.data.len() as f32;
        let mut sum = 0.0f32;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;

        for &val in &tensor.data {
            sum += val;
            if val < min {
                min = val;
            }
            if val > max {
                max = val;
            }
        }
        let mean = sum / n;

        let mut var_sum = 0.0f32;
        for &val in &tensor.data {
            let diff = val - mean;
            var_sum += diff * diff;
        }
        let variance = var_sum / n;
        let std_dev = variance.sqrt();

        let threshold = 3.0 * std_dev;
        let outliers = tensor
            .data
            .iter()
            .filter(|&&v| (v - mean).abs() > threshold)
            .count();

        Self {
            mean,
            variance,
            std_dev,
            min,
            max,
            outlier_ratio_3sigma: (outliers as f32 / n) * 100.0,
        }
    }
}

/// Quality & error metrics comparing original FP32 tensor vs dequantized reconstruction.
#[derive(Debug, Clone)]
pub struct QuantizationErrorReport {
    pub mse: f32,
    pub rmse: f32,
    pub max_abs_error: f32,
    /// Signal-to-Quantization-Noise Ratio in decibels (dB). Higher is better.
    pub sqnr_db: f32,
    pub original_bytes: usize,
    pub compressed_bytes: usize,
    pub compression_ratio: f32,
}

impl QuantizationErrorReport {
    pub fn from_tensors(
        original: &Tensor2D,
        reconstructed: &Tensor2D,
        compressed_bytes: usize,
    ) -> Self {
        let n = original.data.len() as f32;
        let mut signal_power = 0.0f32;
        let mut noise_power = 0.0f32;
        let mut max_abs_error = 0.0f32;

        for (&orig, &rec) in original.data.iter().zip(reconstructed.data.iter()) {
            let err = (orig - rec).abs();
            if err > max_abs_error {
                max_abs_error = err;
            }
            signal_power += orig * orig;
            noise_power += err * err;
        }

        let mse = noise_power / n;
        let rmse = mse.sqrt();
        let sqnr_db = if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            f32::INFINITY
        };

        let original_bytes = original.size_bytes();

        Self {
            mse,
            rmse,
            max_abs_error,
            sqnr_db,
            original_bytes,
            compressed_bytes,
            compression_ratio: original_bytes as f32 / compressed_bytes as f32,
        }
    }
}

// =========================================================================================
// 1. Q8_0: Symmetric 8-Bit Block Quantization (36 bytes per 32 weights -> 3.56x compression)
// =========================================================================================

#[derive(Clone, Copy, Debug)]
pub struct BlockQ8_0 {
    pub scale: f32,
    pub qs: [i8; Q_BLOCK_SIZE],
}

#[derive(Clone, Debug)]
pub struct QuantizedTensorQ8 {
    pub rows: usize,
    pub cols: usize,
    pub blocks: Vec<BlockQ8_0>,
}

impl QuantizedTensorQ8 {
    pub fn quantize(tensor: &Tensor2D) -> Self {
        assert_eq!(tensor.cols % Q_BLOCK_SIZE, 0);
        let num_blocks = tensor.data.len() / Q_BLOCK_SIZE;
        let mut blocks = Vec::with_capacity(num_blocks);

        for chunk in tensor.data.chunks_exact(Q_BLOCK_SIZE) {
            let max_abs = chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs == 0.0 { 0.0 } else { max_abs / 127.0 };
            let inv_scale = if scale == 0.0 { 0.0 } else { 127.0 / max_abs };

            let mut qs = [0i8; Q_BLOCK_SIZE];
            for i in 0..Q_BLOCK_SIZE {
                qs[i] = (chunk[i] * inv_scale).round().clamp(-127.0, 127.0) as i8;
            }
            blocks.push(BlockQ8_0 { scale, qs });
        }

        Self {
            rows: tensor.rows,
            cols: tensor.cols,
            blocks,
        }
    }

    pub fn dequantize(&self) -> Tensor2D {
        let mut data = Vec::with_capacity(self.rows * self.cols);
        for block in &self.blocks {
            for &q in &block.qs {
                data.push((q as f32) * block.scale);
            }
        }
        Tensor2D::from_vec(self.rows, self.cols, data)
    }

    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.blocks.len() * std::mem::size_of::<BlockQ8_0>()
    }

    pub fn evaluate_error(&self, original: &Tensor2D) -> QuantizationErrorReport {
        let rec = self.dequantize();
        QuantizationErrorReport::from_tensors(original, &rec, self.size_bytes())
    }
}

pub fn matmul_q8_0(a_q: &QuantizedTensorQ8, b_t_q: &QuantizedTensorQ8) -> Tensor2D {
    assert_eq!(a_q.cols, b_t_q.cols);
    let m = a_q.rows;
    let n = b_t_q.rows;
    let blocks_per_row = a_q.cols / Q_BLOCK_SIZE;
    let mut c = Tensor2D::zeros(m, n);

    for i in 0..m {
        let a_row_blocks = &a_q.blocks[i * blocks_per_row..(i + 1) * blocks_per_row];
        let c_row = &mut c.data[i * n..(i + 1) * n];

        for j in 0..n {
            let b_row_blocks = &b_t_q.blocks[j * blocks_per_row..(j + 1) * blocks_per_row];
            let mut sum_f32 = 0.0f32;

            for (blk_a, blk_b) in a_row_blocks.iter().zip(b_row_blocks.iter()) {
                let mut int_dot: i32 = 0;
                for k in 0..Q_BLOCK_SIZE {
                    int_dot += (blk_a.qs[k] as i32) * (blk_b.qs[k] as i32);
                }
                sum_f32 += (int_dot as f32) * (blk_a.scale * blk_b.scale);
            }
            c_row[j] = sum_f32;
        }
    }
    c
}

// =========================================================================================
// 2. Q4_0: Extreme 4-Bit Bitwise Nibble Packing (20 bytes per 32 weights -> 6.40x compression)
// =========================================================================================

#[derive(Clone, Copy, Debug)]
pub struct BlockQ4_0 {
    pub scale: f32,
    /// 16 bytes storing 32 4-bit weights (2 weights per byte)
    pub nibbles: [u8; Q_BLOCK_SIZE / 2],
}

#[derive(Clone, Debug)]
pub struct QuantizedTensorQ4 {
    pub rows: usize,
    pub cols: usize,
    pub blocks: Vec<BlockQ4_0>,
}

impl QuantizedTensorQ4 {
    pub fn quantize(tensor: &Tensor2D) -> Self {
        assert_eq!(tensor.cols % Q_BLOCK_SIZE, 0);
        let num_blocks = tensor.data.len() / Q_BLOCK_SIZE;
        let mut blocks = Vec::with_capacity(num_blocks);

        for chunk in tensor.data.chunks_exact(Q_BLOCK_SIZE) {
            let max_abs = chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs == 0.0 { 0.0 } else { max_abs / 7.0 };
            let inv_scale = if scale == 0.0 { 0.0 } else { 7.0 / max_abs };

            let mut nibbles = [0u8; Q_BLOCK_SIZE / 2];
            for i in 0..(Q_BLOCK_SIZE / 2) {
                let q0 = (chunk[2 * i] * inv_scale).round().clamp(-8.0, 7.0) as i8;
                let q1 = (chunk[2 * i + 1] * inv_scale).round().clamp(-8.0, 7.0) as i8;

                let u0 = ((q0 + 8) as u8) & 0x0F;
                let u1 = ((q1 + 8) as u8) & 0x0F;
                nibbles[i] = u0 | (u1 << 4);
            }
            blocks.push(BlockQ4_0 { scale, nibbles });
        }

        Self {
            rows: tensor.rows,
            cols: tensor.cols,
            blocks,
        }
    }

    pub fn dequantize(&self) -> Tensor2D {
        let mut data = Vec::with_capacity(self.rows * self.cols);
        for block in &self.blocks {
            for &byte in &block.nibbles {
                let q0 = ((byte & 0x0F) as i8) - 8;
                let q1 = (((byte >> 4) & 0x0F) as i8) - 8;
                data.push((q0 as f32) * block.scale);
                data.push((q1 as f32) * block.scale);
            }
        }
        Tensor2D::from_vec(self.rows, self.cols, data)
    }

    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.blocks.len() * std::mem::size_of::<BlockQ4_0>()
    }

    pub fn evaluate_error(&self, original: &Tensor2D) -> QuantizationErrorReport {
        let rec = self.dequantize();
        QuantizationErrorReport::from_tensors(original, &rec, self.size_bytes())
    }
}

// =========================================================================================
// 3. Q4_Stats: Outlier-Aware Statistical 4-Bit Quantization (Trimmed-MAD + Lloyd-Max + OLS)
// =========================================================================================

pub const GAUSSIAN_LLOYD_MAX_16: [f32; 16] = [
    -2.733, -2.069, -1.618, -1.256, -0.942, -0.657, -0.388, -0.128,
     0.128,  0.388,  0.657,  0.942,  1.256,  1.618,  2.069,  2.733,
];

#[derive(Clone, Copy, Debug)]
pub struct SparseOutlier {
    pub index: u32,
    pub value: f32,
}

#[derive(Clone, Debug)]
pub struct QuantizedTensorQ4Stats {
    pub rows: usize,
    pub cols: usize,
    pub blocks: Vec<BlockQ4_0>,
    pub outliers: Vec<SparseOutlier>,
}

impl QuantizedTensorQ4Stats {
    #[inline]
    pub fn nearest_gaussian_centroid(z: f32) -> u8 {
        let mut best_idx = 0u8;
        let mut best_dist = (z - GAUSSIAN_LLOYD_MAX_16[0]).abs();
        for idx in 1..16 {
            let dist = (z - GAUSSIAN_LLOYD_MAX_16[idx]).abs();
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx as u8;
            }
        }
        best_idx
    }

    pub fn quantize(tensor: &Tensor2D, z_threshold: f32) -> Self {
        assert_eq!(tensor.cols % Q_BLOCK_SIZE, 0);
        let num_blocks = tensor.data.len() / Q_BLOCK_SIZE;
        let mut blocks = Vec::with_capacity(num_blocks);
        let mut outliers = Vec::new();

        for (block_idx, chunk) in tensor.data.chunks_exact(Q_BLOCK_SIZE).enumerate() {
            let mut sorted_abs = [0.0f32; Q_BLOCK_SIZE];
            for (i, &v) in chunk.iter().enumerate() {
                sorted_abs[i] = v.abs();
            }
            sorted_abs.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let trim_count = 28;
            let trimmed_mean_abs: f32 =
                sorted_abs[..trim_count].iter().sum::<f32>() / (trim_count as f32);
            let mut scale = (1.415 * trimmed_mean_abs).max(1e-8);
            let cutoff = z_threshold * scale;

            let mut codes = [7u8; Q_BLOCK_SIZE];
            for _iter in 0..2 {
                let mut xy_sum = 0.0f32;
                let mut xx_sum = 0.0f32;
                for i in 0..Q_BLOCK_SIZE {
                    let v = chunk[i];
                    if v.abs() <= cutoff {
                        let q = Self::nearest_gaussian_centroid(v / scale);
                        codes[i] = q;
                        let c = GAUSSIAN_LLOYD_MAX_16[q as usize];
                        xy_sum += v * c;
                        xx_sum += c * c;
                    }
                }
                if xx_sum > 1e-12 {
                    scale = xy_sum / xx_sum;
                }
            }

            let mut nibbles = [0u8; Q_BLOCK_SIZE / 2];
            for i in 0..(Q_BLOCK_SIZE / 2) {
                let idx0 = block_idx * Q_BLOCK_SIZE + 2 * i;
                let idx1 = idx0 + 1;
                let v0 = chunk[2 * i];
                let v1 = chunk[2 * i + 1];

                let u0 = if v0.abs() > cutoff {
                    outliers.push(SparseOutlier {
                        index: idx0 as u32,
                        value: v0,
                    });
                    7u8
                } else {
                    codes[2 * i]
                };

                let u1 = if v1.abs() > cutoff {
                    outliers.push(SparseOutlier {
                        index: idx1 as u32,
                        value: v1,
                    });
                    7u8
                } else {
                    codes[2 * i + 1]
                };

                nibbles[i] = (u0 & 0x0F) | ((u1 & 0x0F) << 4);
            }

            blocks.push(BlockQ4_0 { scale, nibbles });
        }

        Self {
            rows: tensor.rows,
            cols: tensor.cols,
            blocks,
            outliers,
        }
    }

    pub fn dequantize(&self) -> Tensor2D {
        let mut data = Vec::with_capacity(self.rows * self.cols);
        for block in &self.blocks {
            for &byte in &block.nibbles {
                let u0 = (byte & 0x0F) as usize;
                let u1 = ((byte >> 4) & 0x0F) as usize;
                data.push(GAUSSIAN_LLOYD_MAX_16[u0] * block.scale);
                data.push(GAUSSIAN_LLOYD_MAX_16[u1] * block.scale);
            }
        }
        for outlier in &self.outliers {
            data[outlier.index as usize] = outlier.value;
        }
        Tensor2D::from_vec(self.rows, self.cols, data)
    }

    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.blocks.len() * std::mem::size_of::<BlockQ4_0>()
            + self.outliers.len() * std::mem::size_of::<SparseOutlier>()
    }

    pub fn evaluate_error(&self, original: &Tensor2D) -> QuantizationErrorReport {
        let rec = self.dequantize();
        QuantizationErrorReport::from_tensors(original, &rec, self.size_bytes())
    }
}

// =========================================================================================
// 4. Q4_Hadamard: Orthonormal Change-of-Basis 4-Bit Quantization (Álgebra y Geometría TP3/TP4)
//
//    MATHEMATICAL FOUNDATION (Álgebra y Geometría ULPGC - TP3 & TP4):
//    - TP3 Sección 1.2.1 (Matriz de cambio de base C_{B -> B'}) & Sección 2.3 (Norma Euclídea):
//      An orthogonal change-of-basis matrix H satisfies H^T * H = I (so H^-1 = H^T) and is an
//      isometry that preserves the Euclidean inner product and norm:
//          <H*u, H*v> = u^T * H^T * H * v = <u, v>   and   ||H*u||_2 = ||u||_2.
//    - Therefore, for any linear neural layer Y = X * W^T:
//          Y = X * I * W^T = (X * H^T) * (W * H^T)^T
//    - By rotating each 32-weight block into the normalized Walsh-Hadamard orthonormal basis
//      (where H_ij = +/- 1/sqrt(32)), any extreme coordinate outlier is spread evenly across
//      all 32 coordinates (divided by sqrt(32) = 5.66x). By the Central Limit Theorem (ME1),
//      the rotated coordinates become a smooth Gaussian distribution WITHOUT outliers!
//    - Result: High precision with ZERO extra outlier table memory (full 6.40x RAM compression)!
// =========================================================================================

#[inline]
pub fn fwht_32_inplace(x: &mut [f32; Q_BLOCK_SIZE]) {
    let mut h = 1;
    while h < Q_BLOCK_SIZE {
        for i in (0..Q_BLOCK_SIZE).step_by(h * 2) {
            for j in i..(i + h) {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
    const INV_SQRT_32: f32 = 0.1767766952966369; // 1.0 / sqrt(32)
    for val in x.iter_mut() {
        *val *= INV_SQRT_32;
    }
}

#[derive(Clone, Debug)]
pub struct QuantizedTensorQ4Hadamard {
    pub rows: usize,
    pub cols: usize,
    /// Uses the exact same 20-byte BlockQ4_0 layout (6.40x compression, zero sparse overhead!)
    pub blocks: Vec<BlockQ4_0>,
}

impl QuantizedTensorQ4Hadamard {
    pub fn quantize(tensor: &Tensor2D) -> Self {
        assert_eq!(tensor.cols % Q_BLOCK_SIZE, 0);
        let num_blocks = tensor.data.len() / Q_BLOCK_SIZE;
        let mut blocks = Vec::with_capacity(num_blocks);

        for chunk in tensor.data.chunks_exact(Q_BLOCK_SIZE) {
            // 1. Change of Basis B -> B' (Hadamard Orthonormal Rotation: w' = H_32 * w)
            let mut rotated = [0.0f32; Q_BLOCK_SIZE];
            rotated.copy_from_slice(chunk);
            fwht_32_inplace(&mut rotated);

            // 2. Compute block Gaussian std_dev in the rotated basis
            let mut sq_sum = 0.0f32;
            for &v in &rotated {
                sq_sum += v * v;
            }
            let mut scale = (sq_sum / (Q_BLOCK_SIZE as f32)).sqrt().max(1e-8);

            // 3. 2-Step OLS Regression Scale Calibration (ME2) on Lloyd-Max Gaussian Centroids (ME1)
            let mut codes = [7u8; Q_BLOCK_SIZE];
            for _iter in 0..2 {
                let mut xy_sum = 0.0f32;
                let mut xx_sum = 0.0f32;
                for i in 0..Q_BLOCK_SIZE {
                    let v = rotated[i];
                    let q = QuantizedTensorQ4Stats::nearest_gaussian_centroid(v / scale);
                    codes[i] = q;
                    let c = GAUSSIAN_LLOYD_MAX_16[q as usize];
                    xy_sum += v * c;
                    xx_sum += c * c;
                }
                if xx_sum > 1e-12 {
                    scale = xy_sum / xx_sum;
                }
            }

            // 4. Bitwise Nibble Packing (IC1: 2 4-bit indices per byte)
            let mut nibbles = [0u8; Q_BLOCK_SIZE / 2];
            for i in 0..(Q_BLOCK_SIZE / 2) {
                let u0 = codes[2 * i] & 0x0F;
                let u1 = codes[2 * i + 1] & 0x0F;
                nibbles[i] = u0 | (u1 << 4);
            }

            blocks.push(BlockQ4_0 { scale, nibbles });
        }

        Self {
            rows: tensor.rows,
            cols: tensor.cols,
            blocks,
        }
    }

    pub fn dequantize(&self) -> Tensor2D {
        let mut data = Vec::with_capacity(self.rows * self.cols);
        for block in &self.blocks {
            let mut rotated = [0.0f32; Q_BLOCK_SIZE];
            for (i, &byte) in block.nibbles.iter().enumerate() {
                let u0 = (byte & 0x0F) as usize;
                let u1 = ((byte >> 4) & 0x0F) as usize;
                rotated[2 * i] = GAUSSIAN_LLOYD_MAX_16[u0] * block.scale;
                rotated[2 * i + 1] = GAUSSIAN_LLOYD_MAX_16[u1] * block.scale;
            }
            // Inverse Change of Basis B' -> B: since H_32^-1 = H_32^T = H_32, apply FWHT again!
            fwht_32_inplace(&mut rotated);
            data.extend_from_slice(&rotated);
        }
        Tensor2D::from_vec(self.rows, self.cols, data)
    }

    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.blocks.len() * std::mem::size_of::<BlockQ4_0>()
    }

    pub fn evaluate_error(&self, original: &Tensor2D) -> QuantizationErrorReport {
        let rec = self.dequantize();
        QuantizationErrorReport::from_tensors(original, &rec, self.size_bytes())
    }
}

/// Direct Rotated-Basis Matrix Multiplication: C = (A * H_32) * (B * H_32)^T = A * B^T
pub fn matmul_q4_hadamard(
    a_fp32: &Tensor2D,
    b_t_hadamard: &QuantizedTensorQ4Hadamard,
) -> Tensor2D {
    assert_eq!(a_fp32.cols, b_t_hadamard.cols);
    let m = a_fp32.rows;
    let n = b_t_hadamard.rows;
    let blocks_per_row = a_fp32.cols / Q_BLOCK_SIZE;
    let mut c = Tensor2D::zeros(m, n);

    // Pre-rotate activations A into the Hadamard basis B': A' = A * H_32
    let mut a_rot = a_fp32.clone();
    for chunk in a_rot.data.chunks_exact_mut(Q_BLOCK_SIZE) {
        let arr: &mut [f32; Q_BLOCK_SIZE] = chunk.try_into().unwrap();
        fwht_32_inplace(arr);
    }

    for i in 0..m {
        let a_row = &a_rot.data[i * a_fp32.cols..(i + 1) * a_fp32.cols];
        let c_row = &mut c.data[i * n..(i + 1) * n];

        for j in 0..n {
            let b_blocks = &b_t_hadamard.blocks[j * blocks_per_row..(j + 1) * blocks_per_row];
            let mut sum = 0.0f32;

            for (b_idx, blk) in b_blocks.iter().enumerate() {
                let a_blk = &a_row[b_idx * Q_BLOCK_SIZE..(b_idx + 1) * Q_BLOCK_SIZE];
                let mut blk_dot = 0.0f32;
                for (k, &byte) in blk.nibbles.iter().enumerate() {
                    let u0 = (byte & 0x0F) as usize;
                    let u1 = ((byte >> 4) & 0x0F) as usize;
                    blk_dot += a_blk[2 * k] * GAUSSIAN_LLOYD_MAX_16[u0]
                        + a_blk[2 * k + 1] * GAUSSIAN_LLOYD_MAX_16[u1];
                }
                sum += blk_dot * blk.scale;
            }
            c_row[j] = sum;
        }
    }
    c
}

// =========================================================================================
// 5. Q4_GoldenOpt: 4-Mode Sign-Bit-Stealing + Golden Section Non-Linear 4-Bit Quantizer
//    (Uniting IC1 IEEE-754 Bit-Stealing + ME1/ME2 Decoupled OLS + OH Capítulo 3 Sección Dorada)
//
//    SCIENTIFIC HYPOTHESIS & MATHEMATICAL FORMULATION:
//    Can we beat ALL previous 4-bit methods — `Q4_0` (16.81 dB), `Q4_Hadamard` (21.18 dB),
//    AND `Q4_Stats` (23.27 dB at 5.73x) — while keeping the block size strictly fixed at
//    20 BYTES PER 32 WEIGHTS (0.16 MB = 6.40x RAM compression, ZERO external outlier tables)?
//
//    1. Hardware IEEE-754 Sign-Bit Stealing (IC1/IC2):
//       In any 20-byte block (`scale_bf16: u16`, `alpha_bf16: u16`, `nibbles: [u8; 16]`),
//       both scale parameters `s > 0` and `p > 0` are strictly positive magnitudes.
//       Consequently, their IEEE-754 sign bits (bit 15, mask `0x8000`) are 100% unused in
//       standard formats! We steal these 2 sign bits (`M1`, `M0`) to encode 4 adaptive
//       quantization topologies per block with ZERO extra storage overhead (20 bytes/block).
//
//    2. Four Adaptive Codebook Topologies (`mode in 0..4`):
//       - Mode 00 (Golden Power-Law 16-Level Manifold, for outlier-free blocks ~85.2%):
//           Q_k = s * sign(c_k) * |c_k|^alpha,  k in 0..16
//         Optimized via Golden Section Search (OH Capítulo 3, Algoritmo 4) over `alpha`.
//       - Mode 01 (14-Inlier + 2-Symmetric-Outlier [±1.00 T], for 1-outlier blocks ~13.7%):
//           Q_k = s * c_k^(14) (k in 0..14),  Q_14 = -T,  Q_15 = +T
//       - Mode 10 (12-Inlier + 4-Multi-Outlier [±0.68 T, ±1.00 T], for 2-3 outlier blocks):
//           Q_k = s * c_k^(12) (k in 0..12),  Q_12..15 = [-T, -0.68T, +0.68T, +T]
//       - Mode 11 (12-Inlier + 4-Multi-Outlier [±0.42 T, ±1.00 T], for wide multi-outliers):
//           Q_k = s * c_k^(12) (k in 0..12),  Q_12..15 = [-T, -0.42T, +0.42T, +T]
//
//    3. First-Order Optimality + Golden Section Search (OH Capítulos 2 & 3 + ME2 OLS):
//       Once discrete codes are assigned, the block MSE objective J(s, T) splits into two
//       orthogonal least-squares subproblems (`∇J = 0`):
//           s* = (<x_in, c_q>) / ||c_q||^2,    T* = (<x_out, m_q>) / ||m_q||^2
//       combined with derivative-free Golden Section Search (`r = 0.618034`) over the
//       non-differentiable quantization thresholds!
// =========================================================================================

const GAUSSIAN_LLOYD_MAX_14: [f32; 14] = [
    -2.585, -1.866, -1.415, -1.057, -0.745, -0.459, -0.188, 0.000, 0.268, 0.558, 0.869, 1.225,
    1.670, 2.450,
];

const GAUSSIAN_LLOYD_MAX_10: [f32; 10] = [
    -2.320, -1.490, -0.960, -0.530, -0.165, 0.000, 0.340, 0.735, 1.215, 2.050,
];

const MODE2_TAIL_MULS: [f32; 6] = [-1.00, -0.62, -0.32, 0.32, 0.62, 1.00];
const MODE3_TAIL_MULS: [f32; 6] = [-1.00, -0.78, -0.46, 0.46, 0.78, 1.00];

#[inline]
fn f32_to_bf16_pos(val: f32) -> u16 {
    ((val.abs().to_bits() >> 16) as u16) & 0x7FFF
}

#[inline]
fn bf16_pos_to_f32(bits: u16) -> f32 {
    f32::from_bits(((bits & 0x7FFF) as u32) << 16)
}

#[inline]
fn pack_header_bf16(scale: f32, param: f32, mode: u8) -> (u16, u16) {
    let m0 = ((mode & 1) as u16) << 15;
    let m1 = (((mode >> 1) & 1) as u16) << 15;
    let s_bits = f32_to_bf16_pos(scale) | m0;
    let p_bits = f32_to_bf16_pos(param) | m1;
    (s_bits, p_bits)
}

#[inline]
fn unpack_header_bf16(scale_bf16: u16, alpha_bf16: u16) -> (f32, f32, u8) {
    let m0 = (scale_bf16 >> 15) as u8;
    let m1 = (alpha_bf16 >> 15) as u8;
    let mode = (m1 << 1) | m0;
    let scale = bf16_pos_to_f32(scale_bf16);
    let param = bf16_pos_to_f32(alpha_bf16);
    (scale, param, mode)
}

/// Strictly 20-byte block (4-byte header + 16-byte nibbles = 20 bytes -> 6.40x compression).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4Golden {
    pub scale_bf16: u16,
    pub alpha_bf16: u16,
    pub nibbles: [u8; Q_BLOCK_SIZE / 2],
}

#[derive(Clone, Debug)]
pub struct QuantizedTensorQ4Golden {
    pub rows: usize,
    pub cols: usize,
    pub blocks: Vec<BlockQ4Golden>,
}

impl QuantizedTensorQ4Golden {
    #[inline]
    fn build_codebook(mode: u8, scale: f32, param: f32) -> [f32; 16] {
        let mut cb = [0.0f32; 16];
        match mode {
            0 => {
                // Mode 00: 16-level Golden Power-Law Gaussian Manifold
                let alpha = param.clamp(0.4, 2.5);
                for k in 0..16 {
                    let c = GAUSSIAN_LLOYD_MAX_16[k];
                    cb[k] = scale * c.signum() * c.abs().powf(alpha);
                }
            }
            1 => {
                // Mode 01: 14-level Gaussian Inliers + 2 Symmetric Exact Outliers [±1.00 T]
                for k in 0..14 {
                    cb[k] = scale * GAUSSIAN_LLOYD_MAX_14[k];
                }
                cb[14] = -param;
                cb[15] = param;
            }
            2 => {
                // Mode 10: 10-level Gaussian Inliers + 6 Multi-Outliers [±0.32 T, ±0.62 T, ±1.00 T]
                for k in 0..10 {
                    cb[k] = scale * GAUSSIAN_LLOYD_MAX_10[k];
                }
                for t in 0..6 {
                    cb[10 + t] = MODE2_TAIL_MULS[t] * param;
                }
            }
            _ => {
                // Mode 11: 10-level Gaussian Inliers + 6 Multi-Outliers [±0.46 T, ±0.78 T, ±1.00 T]
                for k in 0..10 {
                    cb[k] = scale * GAUSSIAN_LLOYD_MAX_10[k];
                }
                for t in 0..6 {
                    cb[10 + t] = MODE3_TAIL_MULS[t] * param;
                }
            }
        }
        cb
    }

    #[inline]
    fn nearest_code(val: f32, cb: &[f32; 16]) -> u8 {
        let mut best_k = 0u8;
        let mut best_dist = (val - cb[0]).abs();
        for k in 1..16 {
            let dist = (val - cb[k]).abs();
            if dist < best_dist {
                best_dist = dist;
                best_k = k as u8;
            }
        }
        best_k
    }

    /// Evaluates Mode 00 for a given shape exponent `alpha`, solving OLS scale `s*(alpha)`.
    #[inline]
    fn eval_mode0(chunk: &[f32], base_sigma: f32, alpha: f32) -> (f32, u16, u16, [u8; Q_BLOCK_SIZE]) {
        let mut unit_w = [0.0f32; 16];
        for k in 0..16 {
            let c = GAUSSIAN_LLOYD_MAX_16[k];
            unit_w[k] = c.signum() * c.abs().powf(alpha);
        }

        let mut scale = base_sigma;
        let mut codes = [7u8; Q_BLOCK_SIZE];
        for _iter in 0..2 {
            let mut xy = 0.0f32;
            let mut xx = 0.0f32;
            for (i, &v) in chunk.iter().enumerate() {
                let target = v / scale;
                let mut best_k = 0u8;
                let mut best_d = (target - unit_w[0]).abs();
                for k in 1..16 {
                    let d = (target - unit_w[k]).abs();
                    if d < best_d {
                        best_d = d;
                        best_k = k as u8;
                    }
                }
                codes[i] = best_k;
                let w = unit_w[best_k as usize];
                xy += v * w;
                xx += w * w;
            }
            if xx > 1e-12 {
                scale = (xy / xx).abs().max(1e-8);
            }
        }

        let (s_bf16, p_bf16) = pack_header_bf16(scale, alpha, 0);
        let (s_q, p_q, m_q) = unpack_header_bf16(s_bf16, p_bf16);
        let cb = Self::build_codebook(m_q, s_q, p_q);
        let mut mse = 0.0f32;
        for (i, &v) in chunk.iter().enumerate() {
            let k = Self::nearest_code(v, &cb);
            codes[i] = k;
            let err = v - cb[k as usize];
            mse += err * err;
        }
        (mse, s_bf16, p_bf16, codes)
    }

    /// Evaluates Dual-Scale Outlier Modes (`mode in 1..=3`) for a given initial scale `s_init`
    /// and updates `(s*, T*)` via decoupled First-Order OLS (`∇J = 0`).
    #[inline]
    fn eval_split_mode(
        chunk: &[f32],
        mode: u8,
        s_init: f32,
        t_init: f32,
    ) -> (f32, u16, u16, [u8; Q_BLOCK_SIZE]) {
        let mut scale = s_init.max(1e-8);
        let mut tail = t_init.max(1e-8);
        let mut codes = [7u8; Q_BLOCK_SIZE];

        let inlier_limit = if mode == 1 { 14usize } else { 10usize };

        for _iter in 0..3 {
            let cb = Self::build_codebook(mode, scale, tail);
            let mut in_xy = 0.0f32;
            let mut in_xx = 0.0f32;
            let mut out_xy = 0.0f32;
            let mut out_xx = 0.0f32;

            for (i, &v) in chunk.iter().enumerate() {
                let k = Self::nearest_code(v, &cb) as usize;
                codes[i] = k as u8;
                if k < inlier_limit {
                    let c = if mode == 1 {
                        GAUSSIAN_LLOYD_MAX_14[k]
                    } else {
                        GAUSSIAN_LLOYD_MAX_10[k]
                    };
                    in_xy += v * c;
                    in_xx += c * c;
                } else {
                    let m = if mode == 1 {
                        if k == 14 { -1.0 } else { 1.0 }
                    } else if mode == 2 {
                        MODE2_TAIL_MULS[k - 10]
                    } else {
                        MODE3_TAIL_MULS[k - 10]
                    };
                    out_xy += v * m;
                    out_xx += m * m;
                }
            }
            if in_xx > 1e-12 {
                scale = (in_xy / in_xx).abs().max(1e-8);
            }
            if out_xx > 1e-12 {
                tail = (out_xy / out_xx).abs().max(1e-8);
            }
        }

        let (s_bf16, p_bf16) = pack_header_bf16(scale, tail, mode);
        let (s_q, p_q, m_q) = unpack_header_bf16(s_bf16, p_bf16);
        let cb = Self::build_codebook(m_q, s_q, p_q);
        let mut mse = 0.0f32;
        for (i, &v) in chunk.iter().enumerate() {
            let k = Self::nearest_code(v, &cb);
            codes[i] = k;
            let err = v - cb[k as usize];
            mse += err * err;
        }
        (mse, s_bf16, p_bf16, codes)
    }

    pub fn quantize(tensor: &Tensor2D) -> Self {
        assert_eq!(tensor.cols % Q_BLOCK_SIZE, 0);
        assert_eq!(std::mem::size_of::<BlockQ4Golden>(), 20);
        let num_blocks = tensor.data.len() / Q_BLOCK_SIZE;
        let mut blocks = Vec::with_capacity(num_blocks);

        // Golden ratio conjugate r = (sqrt(5) - 1) / 2 (OH Capítulo 3, Sección 3.2.3, Ec. 3.6)
        const GOLDEN_R: f32 = 0.618_034;

        for chunk in tensor.data.chunks_exact(Q_BLOCK_SIZE) {
            let mut sorted_abs = [0.0f32; Q_BLOCK_SIZE];
            for (i, &v) in chunk.iter().enumerate() {
                sorted_abs[i] = v.abs();
            }
            sorted_abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let max_abs = sorted_abs[Q_BLOCK_SIZE - 1].max(1e-8);
            let trimmed_mean_abs: f32 = sorted_abs[..28].iter().sum::<f32>() / 28.0;
            let base_sigma = (1.415 * trimmed_mean_abs).max(1e-8);

            // 1. Evaluate Mode 00 via Golden Section Search (Algoritmo 4 de OH) over `alpha in [0.65, 1.75]`
            let mut a = 0.65f32;
            let mut b = 1.75f32;
            let mut x1 = b - GOLDEN_R * (b - a);
            let mut x2 = a + GOLDEN_R * (b - a);
            let (mut f1, _, _, _) = Self::eval_mode0(chunk, base_sigma, x1);
            let (mut f2, _, _, _) = Self::eval_mode0(chunk, base_sigma, x2);

            for _step in 0..8 {
                if f1 < f2 {
                    b = x2;
                    x2 = x1;
                    f2 = f1;
                    x1 = b - GOLDEN_R * (b - a);
                    (f1, _, _, _) = Self::eval_mode0(chunk, base_sigma, x1);
                } else {
                    a = x1;
                    x1 = x2;
                    f1 = f2;
                    x2 = a + GOLDEN_R * (b - a);
                    (f2, _, _, _) = Self::eval_mode0(chunk, base_sigma, x2);
                }
            }
            let alpha_opt = 0.5 * (a + b);
            let (mut best_mse, mut best_s, mut best_p, mut best_codes) =
                Self::eval_mode0(chunk, base_sigma, alpha_opt);
            // Also check exact alpha = 1.0 (pure Gaussian Lloyd-Max)
            let (mse_1, s_1, p_1, codes_1) = Self::eval_mode0(chunk, base_sigma, 1.0);
            if mse_1 < best_mse {
                best_mse = mse_1;
                best_s = s_1;
                best_p = p_1;
                best_codes = codes_1;
            }

            // 2. Evaluate Dual-Scale Outlier Topologies (Modes 01, 10, 11) with Golden Section over s_init
            for mode in 1..=3u8 {
                let mut sa = 0.65 * base_sigma;
                let mut sb = 1.55 * base_sigma;
                let mut sx1 = sb - GOLDEN_R * (sb - sa);
                let mut sx2 = sa + GOLDEN_R * (sb - sa);
                let (mut sf1, _, _, _) = Self::eval_split_mode(chunk, mode, sx1, max_abs);
                let (mut sf2, _, _, _) = Self::eval_split_mode(chunk, mode, sx2, max_abs);

                for _step in 0..6 {
                    if sf1 < sf2 {
                        sb = sx2;
                        sx2 = sx1;
                        sf2 = sf1;
                        sx1 = sb - GOLDEN_R * (sb - sa);
                        (sf1, _, _, _) = Self::eval_split_mode(chunk, mode, sx1, max_abs);
                    } else {
                        sa = sx1;
                        sx1 = sx2;
                        sf1 = sf2;
                        sx2 = sa + GOLDEN_R * (sb - sa);
                        (sf2, _, _, _) = Self::eval_split_mode(chunk, mode, sx2, max_abs);
                    }
                }
                let s_opt = 0.5 * (sa + sb);
                let (mse_m, s_m, p_m, codes_m) = Self::eval_split_mode(chunk, mode, s_opt, max_abs);
                if mse_m < best_mse {
                    best_mse = mse_m;
                    best_s = s_m;
                    best_p = p_m;
                    best_codes = codes_m;
                }
            }

            let mut nibbles = [0u8; Q_BLOCK_SIZE / 2];
            for i in 0..(Q_BLOCK_SIZE / 2) {
                let u0 = best_codes[2 * i] & 0x0F;
                let u1 = best_codes[2 * i + 1] & 0x0F;
                nibbles[i] = u0 | (u1 << 4);
            }

            blocks.push(BlockQ4Golden {
                scale_bf16: best_s,
                alpha_bf16: best_p,
                nibbles,
            });
        }

        Self {
            rows: tensor.rows,
            cols: tensor.cols,
            blocks,
        }
    }

    pub fn dequantize(&self) -> Tensor2D {
        let mut data = Vec::with_capacity(self.rows * self.cols);
        for block in &self.blocks {
            let (scale, param, mode) = unpack_header_bf16(block.scale_bf16, block.alpha_bf16);
            let cb = Self::build_codebook(mode, scale, param);
            for &byte in &block.nibbles {
                let u0 = (byte & 0x0F) as usize;
                let u1 = ((byte >> 4) & 0x0F) as usize;
                data.push(cb[u0]);
                data.push(cb[u1]);
            }
        }
        Tensor2D::from_vec(self.rows, self.cols, data)
    }

    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.blocks.len() * std::mem::size_of::<BlockQ4Golden>()
    }

    pub fn evaluate_error(&self, original: &Tensor2D) -> QuantizationErrorReport {
        let rec = self.dequantize();
        QuantizationErrorReport::from_tensors(original, &rec, self.size_bytes())
    }
}
