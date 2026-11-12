use std::time::Instant;
use tiny_tensor_engine::matmul::{matmul_ikj_simd, matmul_naive, matmul_parallel, matmul_tiled};
use tiny_tensor_engine::quant::{
    matmul_q8_0, QuantizedTensorQ4, QuantizedTensorQ4Stats, QuantizedTensorQ8, TensorStats,
};
use tiny_tensor_engine::tensor::Tensor2D;

fn compute_gflops(n: usize, elapsed_secs: f64) -> f64 {
    let flops = 2.0 * (n as f64).powi(3);
    (flops / elapsed_secs) / 1e9
}

fn run_gemm_benchmark(n: usize, threads: usize) {
    println!("================================================================================");
    println!(
        " [SPRINT 1] SILICON GEMM BENCHMARK — Matrix Size: {} x {} ({:.2} MB per matrix)",
        n,
        n,
        (n * n * 4) as f64 / (1024.0 * 1024.0)
    );
    println!("================================================================================");

    let a = Tensor2D::random_normal(n, n, 0.02, 42);
    let b = Tensor2D::random_normal(n, n, 0.02, 1337);

    let t0 = Instant::now();
    let c_naive = matmul_naive(&a, &b);
    let dt_naive = t0.elapsed().as_secs_f64();
    let gflops_naive = compute_gflops(n, dt_naive);

    let t1 = Instant::now();
    let c_tiled = matmul_tiled(&a, &b, 64);
    let dt_tiled = t1.elapsed().as_secs_f64();
    let gflops_tiled = compute_gflops(n, dt_tiled);

    let t2 = Instant::now();
    let c_ikj = matmul_ikj_simd(&a, &b);
    let dt_ikj = t2.elapsed().as_secs_f64();
    let gflops_ikj = compute_gflops(n, dt_ikj);

    let t3 = Instant::now();
    let c_par = matmul_parallel(&a, &b, threads);
    let dt_par = t3.elapsed().as_secs_f64();
    let gflops_par = compute_gflops(n, dt_par);

    let diff_tiled = c_naive.max_abs_diff(&c_tiled);
    let diff_ikj = c_naive.max_abs_diff(&c_ikj);
    let diff_par = c_naive.max_abs_diff(&c_par);

    println!(
        "{:<34} | {:>10} | {:>10} | {:>10} | {:>10}",
        "Kernel Implementation", "Time (ms)", "GFLOPS", "Speedup", "Max Error"
    );
    println!("--------------------------------------------------------------------------------");
    println!(
        "{:<34} | {:>10.2} | {:>10.2} | {:>9.2}x | {:>10.2e}",
        "1. Naive (i->j->k, cache miss)",
        dt_naive * 1000.0,
        gflops_naive,
        1.0,
        0.0
    );
    println!(
        "{:<34} | {:>10.2} | {:>10.2} | {:>9.2}x | {:>10.2e}",
        "2. Cache-Tiled (64x64 L1 block)",
        dt_tiled * 1000.0,
        gflops_tiled,
        dt_naive / dt_tiled,
        diff_tiled
    );
    println!(
        "{:<34} | {:>10.2} | {:>10.2} | {:>9.2}x | {:>10.2e}",
        "3. Cache-Linear (i->k->j + SIMD)",
        dt_ikj * 1000.0,
        gflops_ikj,
        dt_naive / dt_ikj,
        diff_ikj
    );
    println!(
        "{:<34} | {:>10.2} | {:>10.2} | {:>9.2}x | {:>10.2e}",
        format!("4. Parallel SIMD ({} CPU threads)", threads),
        dt_par * 1000.0,
        gflops_par,
        dt_naive / dt_par,
        diff_par
    );
    println!();
}

fn run_quantization_benchmark(n: usize) {
    println!("================================================================================");
    println!(
        " [SPRINT 2 & 3] EXTREME 4-BIT & 8-BIT QUANTIZATION ON LLM WEIGHTS ({} x {})",
        n, n
    );
    println!("================================================================================");

    // Simulate realistic LLM Transformer weights: Gaussian core N(0, 0.02^2) + 0.5% emergent 10x outliers
    let llm_weights = Tensor2D::random_llm_weights(n, n, 0.02, 2026);
    let activations = Tensor2D::random_normal(n, n, 0.05, 7777);

    let stats = TensorStats::compute(&llm_weights);
    println!(" [A] Heavy-Tailed LLM Weight Profile (Gaussian Core + 0.5% Emergent Outliers):");
    println!("     • Mean (μ):               {:+.6}", stats.mean);
    println!(
        "     • Std Deviation (σ):      {:.6} (Variance σ² = {:.6})",
        stats.std_dev, stats.variance
    );
    println!("     • Range [Min, Max]:       [{:+.5}, {:+.5}]", stats.min, stats.max);
    println!(
        "     • Outliers (|x - μ| > 3σ): {:.3}% (Heavy-tailed LLM distribution)",
        stats.outlier_ratio_3sigma
    );
    println!();

    // 1. Q8_0 (8-bit symmetric block quantization)
    let w_q8 = QuantizedTensorQ8::quantize(&llm_weights);
    let rep_q8 = w_q8.evaluate_error(&llm_weights);

    // 2. Standard Q4_0 (4-bit bitwise nibble packing - suffers from outlier grid collapse)
    let w_q4_std = QuantizedTensorQ4::quantize(&llm_weights);
    let rep_q4_std = w_q4_std.evaluate_error(&llm_weights);

    // 3. Q4_Stats (Our Statistical MAD Z-Score Outlier-Aware 4-bit Quantization!)
    let w_q4_stats = QuantizedTensorQ4Stats::quantize(&llm_weights, 3.2);
    let rep_q4_stats = w_q4_stats.evaluate_error(&llm_weights);

    println!(" [B] Head-to-Head Compression & Precision Comparison (FP32 = {:.2} MB):", rep_q8.original_bytes as f64 / 1048576.0);
    println!(
        "{:<34} | {:>9} | {:>11} | {:>11} | {:>11} | {:>9}",
        "Quantization Format", "Size (MB)", "Compression", "RMSE Error", "Max Error", "SQNR (dB)"
    );
    println!("------------------------------------------------------------------------------------------------");
    println!(
        "{:<34} | {:>9.2} | {:>10.2}x | {:>11.4e} | {:>11.4e} | {:>9.2}",
        "1. Q8_0 (8-bit Standard)",
        rep_q8.compressed_bytes as f64 / 1048576.0,
        rep_q8.compression_ratio,
        rep_q8.rmse,
        rep_q8.max_abs_error,
        rep_q8.sqnr_db
    );
    println!(
        "{:<34} | {:>9.2} | {:>10.2}x | {:>11.4e} | {:>11.4e} | {:>9.2}",
        "2. Q4_0 (4-bit Standard Nibble)",
        rep_q4_std.compressed_bytes as f64 / 1048576.0,
        rep_q4_std.compression_ratio,
        rep_q4_std.rmse,
        rep_q4_std.max_abs_error,
        rep_q4_std.sqnr_db
    );
    println!(
        "{:<34} | {:>9.2} | {:>10.2}x | {:>11.4e} | {:>11.4e} | {:>9.2}",
        "3. Q4_Stats (4-bit Outlier-Aware)",
        rep_q4_stats.compressed_bytes as f64 / 1048576.0,
        rep_q4_stats.compression_ratio,
        rep_q4_stats.rmse,
        rep_q4_stats.max_abs_error,
        rep_q4_stats.sqnr_db
    );
    println!();
    println!(
        " >>> ALGORITHMIC BREAKTHROUGH: Q4_Stats improves 4-bit SQNR by +{:.2} dB and reduces RMSE by {:.2}x",
        rep_q4_stats.sqnr_db - rep_q4_std.sqnr_db,
        rep_q4_std.rmse / rep_q4_stats.rmse
    );
    println!(
        "     while maintaining {:.2}x RAM compression (isolated {} outliers out of {} weights = {:.3}%).",
        rep_q4_stats.compression_ratio,
        w_q4_stats.outliers.len(),
        n * n,
        (w_q4_stats.outliers.len() as f64 / (n * n) as f64) * 100.0
    );
    println!();

    // Compare downstream Matrix Multiplication impact: C = Activations * Weights
    let c_fp32 = matmul_ikj_simd(&activations, &llm_weights);
    let c_from_q4_std = matmul_ikj_simd(&activations, &w_q4_std.dequantize());
    let c_from_q4_stats = matmul_ikj_simd(&activations, &w_q4_stats.dequantize());

    let gemm_err_std = c_fp32.max_abs_diff(&c_from_q4_std);
    let gemm_err_stats = c_fp32.max_abs_diff(&c_from_q4_stats);

    let weights_t_q8 = QuantizedTensorQ8::quantize(&llm_weights.transpose());
    let act_q8 = QuantizedTensorQ8::quantize(&activations);
    let t_qgemm = Instant::now();
    let _c_q8 = matmul_q8_0(&act_q8, &weights_t_q8);
    let dt_qgemm = t_qgemm.elapsed().as_secs_f64();

    println!(" [C] Downstream Neural Layer Output Drift (Activations x Quantized Weights):");
    println!("     • Standard Q4_0 Layer Max Drift:      {:.6e}", gemm_err_std);
    println!("     • Outlier-Aware Q4_Stats Max Drift:   {:.6e} ({:.2}x more accurate!)", gemm_err_stats, gemm_err_std / gemm_err_stats);
    println!("     • Direct INT8 Q8_0 GEMM Kernel Time:  {:.2} ms", dt_qgemm * 1000.0);
    println!("================================================================================");
}

fn main() {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║               TINY-TENSOR-ENGINE v0.2.0 — LOW-LEVEL AI RUNTIME              ║");
    println!("║     Pure Rust • Bitwise Nibble Packing (Q4_0) & Outlier-Aware (Q4_Stats)    ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!(" Detected Hardware Concurrency: {} logical CPU cores\n", threads);

    run_gemm_benchmark(512, threads);
    run_gemm_benchmark(1024, threads);
    run_quantization_benchmark(512);
}
