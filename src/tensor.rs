//! Contiguous 2D Tensor representation in Row-Major layout.
//!
//! Connects directly to Computer Architecture (IC1/IC2 - Tema 5: Gestión de Memoria Contigua):
//! Instead of `Vec<Vec<f32>>` (which allocates separate heap buffers per row and causes pointer-chasing
//! L1/L2 cache misses), `Tensor2D` stores all elements in a single contiguous memory buffer `Vec<f32>`.
//! Element `(row, col)` is located at linear offset `row * cols + col`.

/// Contiguous Row-Major 2D Tensor (`f32`).
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor2D {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Tensor2D {
    /// Allocates a zero-initialized tensor of shape `(rows, cols)` in contiguous memory.
    #[inline]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Creates a tensor from an existing flat vector.
    #[inline]
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(
            rows * cols,
            data.len(),
            "Shape ({}, {}) does not match data length {}",
            rows,
            cols,
            data.len()
        );
        Self { rows, cols, data }
    }

    /// Generates a tensor initialized with a Normal/Gaussian distribution N(0, std_dev^2)
    /// using the Box-Muller transform (Statistical Methods ME1) and a deterministic Xorshift64* PRNG.
    /// Models real neural network weight distributions (Xavier / He initialization).
    pub fn random_normal(rows: usize, cols: usize, std_dev: f32, seed: u64) -> Self {
        let total = rows * cols;
        let mut data = Vec::with_capacity(total);
        let mut state = if seed == 0 { 0xDEAD_BEEF_CAFE_BABEu64 } else { seed };

        let mut next_u64 = || -> u64 {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_F491_4F6C_DD1Du64)
        };

        let mut next_uniform = || -> f32 {
            // Uniform in (0, 1) open interval
            let u = (next_u64() >> 40) as f32 / ((1u64 << 24) as f32);
            u.clamp(1e-7, 1.0 - 1e-7)
        };

        let mut i = 0;
        while i < total {
            let u1 = next_uniform();
            let u2 = next_uniform();
            let radius = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;

            data.push(radius * theta.cos() * std_dev);
            i += 1;
            if i < total {
                data.push(radius * theta.sin() * std_dev);
                i += 1;
            }
        }

        Self { rows, cols, data }
    }

    /// Generates realistic LLM neural network weights with a Gaussian core N(0, std_dev^2)
    /// plus 0.5% emergent heavy-tailed outlier channels (8x-12x magnitude), as observed in
    /// real-world Transformer models (Dettmers et al., LLM.int8() / SpQR / AWQ).
    pub fn random_llm_weights(rows: usize, cols: usize, std_dev: f32, seed: u64) -> Self {
        let mut tensor = Self::random_normal(rows, cols, std_dev, seed);
        let mut state = seed ^ 0xA5A5_5A5A_F0F0_0F0Fu64;
        for val in tensor.data.iter_mut() {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let r = state.wrapping_mul(0x2545_F491_4F6C_DD1Du64) % 1000;
            // 0.5% of weights are emergent high-magnitude outliers
            if r < 5 {
                *val *= 10.0;
            }
        }
        tensor
    }

    /// Returns the memory footprint of the tensor buffer in bytes.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    /// Returns a transposed copy `B^T` of shape `(cols, rows)`.
    /// Pre-transposing the right-hand matrix converts column-wise stride-N access into
    /// sequential stride-1 access for hardware SIMD dot products.
    pub fn transpose(&self) -> Self {
        let mut out = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            let row_offset = i * self.cols;
            for j in 0..self.cols {
                out.data[j * self.rows + i] = self.data[row_offset + j];
            }
        }
        out
    }

    /// Computes maximum absolute difference between two tensors (for numerical verification).
    pub fn max_abs_diff(&self, other: &Self) -> f32 {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }
}
