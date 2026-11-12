pub mod matmul;
pub mod quant;
pub mod tensor;

#[cfg(test)]
mod tests {
    use crate::matmul::{matmul_ikj_simd, matmul_naive, matmul_parallel, matmul_tiled};
    use crate::quant::{matmul_q8_0, QuantizedTensorQ8, TensorStats};
    use crate::tensor::Tensor2D;

    #[test]
    fn test_gemm_kernels_equivalence() {
        let a = Tensor2D::random_normal(64, 64, 1.0, 42);
        let b = Tensor2D::random_normal(64, 64, 1.0, 1337);

        let c_naive = matmul_naive(&a, &b);
        let c_ikj = matmul_ikj_simd(&a, &b);
        let c_tiled = matmul_tiled(&a, &b, 16);
        let c_par = matmul_parallel(&a, &b, 4);

        assert!(c_naive.max_abs_diff(&c_ikj) < 1e-4);
        assert!(c_naive.max_abs_diff(&c_tiled) < 1e-4);
        assert!(c_naive.max_abs_diff(&c_par) < 1e-4);
    }

    #[test]
    fn test_q8_0_quantization_and_matmul() {
        let a = Tensor2D::random_normal(64, 64, 0.5, 101);
        let b = Tensor2D::random_normal(64, 64, 0.5, 202);
        let b_t = b.transpose();

        let stats = TensorStats::compute(&a);
        assert!(stats.mean.abs() < 0.1);
        assert!((stats.std_dev - 0.5).abs() < 0.08);

        let a_q = QuantizedTensorQ8::quantize(&a);
        let b_t_q = QuantizedTensorQ8::quantize(&b_t);

        let report = a_q.evaluate_error(&a);
        // Q8_0 should achieve > 35 dB SQNR on Gaussian weights
        assert!(report.sqnr_db > 35.0, "SQNR too low: {}", report.sqnr_db);
        assert!(report.compression_ratio > 3.5);

        let c_fp32 = matmul_ikj_simd(&a, &b);
        let c_q8 = matmul_q8_0(&a_q, &b_t_q);

        // Quantized INT8 matmul should closely approximate FP32 matmul
        let max_diff = c_fp32.max_abs_diff(&c_q8);
        assert!(max_diff < 0.25, "Quantized matmul max_diff too high: {}", max_diff);
    }
}
