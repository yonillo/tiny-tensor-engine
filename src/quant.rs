//! Statistical Quantization Engine (`Q8_0`, `Q4_0` Bitwise Nibble Packing, and `Q4_Stats` Outlier-Aware).
//!
//! Connects Statistical Methods (ME1 & ME2: Descriptive Statistics, Normal Distributions,
//! Z-Score Outlier Detection, Mean Squared Error, and Signal-to-Noise Ratio) with Low-Level
//! Bitwise Manipulation (IC1: shifts `<<`, `>>`, masks `& 0x0F`) used in frontier LLM engines.

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
    pub fn from_tensors(original: &Tensor2D, reconstructed: &Tensor2D, compressed_bytes: usize) -> Self {
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
//    Packs TWO 4-bit weights [-8, +7] into a SINGLE byte (u8) using bitwise shifts & masks:
//    Low nibble (bits 0..3): weight_even + 8
//    High nibble (bits 4..7): (weight_odd + 8) << 4
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
            // 4-bit signed range [-7, +7] symmetric around 0 (offset by +8 -> [1, 15])
            let scale = if max_abs == 0.0 { 0.0 } else { max_abs / 7.0 };
            let inv_scale = if scale == 0.0 { 0.0 } else { 7.0 / max_abs };

            let mut nibbles = [0u8; Q_BLOCK_SIZE / 2];
            for i in 0..(Q_BLOCK_SIZE / 2) {
                let q0 = (chunk[2 * i] * inv_scale).round().clamp(-8.0, 7.0) as i8;
                let q1 = (chunk[2 * i + 1] * inv_scale).round().clamp(-8.0, 7.0) as i8;

                let u0 = ((q0 + 8) as u8) & 0x0F;
                let u1 = ((q1 + 8) as u8) & 0x0F;
                // Bitwise pack two 4-bit numbers into 1 byte (IC1 binary representation)
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
// 3. Q4_Stats: Outlier-Aware Statistical 4-Bit Quantization (Our Algorithmic Improvement!)
//
//    PROBLEM IN STANDARD Q4_0:
//    In a 4-bit grid (only 16 steps), a single 10x outlier weight inside a 32-element block
//    stretches `scale = max_abs / 7.0` by 10x, crushing the other 31 normal weights to 0!
//
//    STATISTICAL SOLUTION (ME1/ME2 Z-Score Filtering + Sparse Outlier Extraction):
//    1. Estimate the block's inlier dispersion using the Mean Absolute Deviation (MAD):
//       robust_sigma = 1.2533 * mean(|x|).
//    2. Any weight exceeding `z_threshold * robust_sigma` (e.g., 3.5 sigma) is extracted into
//       a compact sparse outlier table `(u32 index, f32 exact_value)`.
//    3. The remaining 99.5% of inlier weights are quantized to 4-bit nibbles using the tight
//       inlier scale! This preserves full 4-bit resolution for normal weights AND 100% exact
//       precision for emergent outliers.
// =========================================================================================

/// 16-level Lloyd-Max optimal centroids for a Standard Normal distribution N(0, 1)
/// (Statistical Methods ME1: Quantiles & minimum MSE reconstruction points for Gaussian density).
/// Unlike uniform grids [-7..+7] that waste resolution in the sparse tails, these centroids
/// concentrate resolution near zero where 68.2% of neural weights reside.
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
    /// Each block stores `robust_sigma` in `scale` and 32 4-bit indices (0..15) into `GAUSSIAN_LLOYD_MAX_16`
    pub blocks: Vec<BlockQ4_0>,
    pub outliers: Vec<SparseOutlier>,
}

impl QuantizedTensorQ4Stats {
    #[inline]
    fn nearest_gaussian_centroid(z: f32) -> u8 {
        // Binary search / branchless-friendly lookup in the 16-element sorted centroid table
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
            // 1. Trimmed-MAD Estimator (ME2): sort 32 absolute values and use inner 28 to prevent
            //    the Small-Sample Masking Effect from emergent outliers.
            let mut sorted_abs = [0.0f32; Q_BLOCK_SIZE];
            for (i, &v) in chunk.iter().enumerate() {
                sorted_abs[i] = v.abs();
            }
            sorted_abs.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let trim_count = 28;
            let trimmed_mean_abs: f32 =
                sorted_abs[..trim_count].iter().sum::<f32>() / (trim_count as f32);
            // Unbiased conversion from 87.5% truncated E[|X|] to Gaussian sigma
            let mut scale = (1.415 * trimmed_mean_abs).max(1e-8);
            let cutoff = z_threshold * scale;

            // 2. Ordinary Least Squares (OLS) Alternating Scale Optimization (Statistical Methods ME2 - Regresión Lineal):
            //    Given quantized centroid assignments c_{q_i}, the scale `s` that strictly minimizes
            //    block MSE = sum(x_i - s * c_{q_i})^2 is the OLS estimator:
            //    s* = sum(x_i * c_{q_i}) / sum(c_{q_i}^2)
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
        // Restore exact sparse outliers
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
