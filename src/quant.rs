//! Statistical Quantization Engine (`Q8_0` & `Q4_0` Block Quantization).
//!
//! Connects Statistical Methods (ME1 & ME2: Descriptive Statistics, Normal Distributions,
//! Outlier Sensitivity, Mean Squared Error, and Signal-to-Noise Ratio) with Low-Level Memory
//! Compression used in modern LLM engines (`llama.cpp` / `GGML`).
//!
//! Instead of storing every neural network weight as a 32-bit float (4 bytes), we group weights
//! into blocks of `BLOCK_SIZE = 32` and map them to 8-bit signed integers `[-127, 127]` or 4-bit
//! nibbles `[-8, 7]` using a block-level statistical scale factor `delta = max(|x|) / 127.0`.

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

/// A single `Q8_0` block: 1 `f32` scale factor + 32 `i8` quantized values.
/// Footprint: 4 + 32 = 36 bytes per 32 weights (vs 128 bytes in FP32 -> 3.56x compression).
#[derive(Clone, Copy, Debug)]
pub struct BlockQ8_0 {
    pub scale: f32,
    pub qs: [i8; Q_BLOCK_SIZE],
}

/// Quantized 2D Tensor in `Q8_0` format.
#[derive(Clone, Debug)]
pub struct QuantizedTensorQ8 {
    pub rows: usize,
    pub cols: usize,
    pub blocks: Vec<BlockQ8_0>,
}

impl QuantizedTensorQ8 {
    /// Quantizes a `Tensor2D` (whose `cols` is a multiple of 32) into `Q8_0` blocks.
    pub fn quantize(tensor: &Tensor2D) -> Self {
        assert_eq!(
            tensor.cols % Q_BLOCK_SIZE,
            0,
            "Tensor cols ({}) must be a multiple of Q_BLOCK_SIZE ({})",
            tensor.cols,
            Q_BLOCK_SIZE
        );

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

    /// Reconstructs an approximate `Tensor2D` (`f32`) from quantized `Q8_0` blocks.
    pub fn dequantize(&self) -> Tensor2D {
        let mut data = Vec::with_capacity(self.rows * self.cols);
        for block in &self.blocks {
            for &q in &block.qs {
                data.push((q as f32) * block.scale);
            }
        }
        Tensor2D::from_vec(self.rows, self.cols, data)
    }

    /// Memory size in bytes of the compressed `Q8_0` representation.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.blocks.len() * std::mem::size_of::<BlockQ8_0>()
    }

    /// Evaluates statistical error metrics (MSE, RMSE, Max Error, SQNR in dB) against original FP32 tensor.
    pub fn evaluate_error(&self, original: &Tensor2D) -> QuantizationErrorReport {
        let reconstructed = self.dequantize();
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
        let compressed_bytes = self.size_bytes();

        QuantizationErrorReport {
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

/// Quantized Matrix Multiplication (`C = A * B^T`) operating directly on `Q8_0` blocks!
/// Both `a` and `b_transposed` are quantized to `Q8_0`.
/// Inner dot products are computed using fast integer arithmetic (`i32` accumulators)
/// and scaled once per 32-element block.
pub fn matmul_q8_0(a_q: &QuantizedTensorQ8, b_t_q: &QuantizedTensorQ8) -> Tensor2D {
    assert_eq!(a_q.cols, b_t_q.cols, "Incompatible quantized dimensions");
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
                // Integer dot product over 32 elements (auto-vectorized to SIMD integer instructions)
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
