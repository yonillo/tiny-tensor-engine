use std::time::Instant;
use tiny_tensor_engine::matmul::{matmul_ikj_simd, matmul_naive, matmul_parallel, matmul_tiled};
use tiny_tensor_engine::quant::{matmul_q8_0, QuantizedTensorQ8, TensorStats};
use tiny_tensor_engine::tensor::Tensor2D;

fn compute_gflops(n: usize, elapsed_secs: f64) -> f64 {
    // GEMM performs 2 * N^3 floating-point operations (1 multiply + 1 add per inner step)
    let flops = 2.0 * (n as f64).powi(3);
    (flops / elapsed_secs) / 1e9
}

fn run_gemm_benchmark(n: usize, threads: usize) {
    println!("================================================================================");
    println!(" [SPRINT 1] SILICON GEMM BENCHMARK — Matrix Size: {} x {} ({:.2} MB per matrix)",
        n, n, (n * n * 4) as f64 / (1024.0 * 1024.0));
    println!("================================================================================");

    let a = Tensor2D::random_normal(n, n, 0.02, 42);
    let b = Tensor2D::random_normal(n, n, 0.02, 1337);

    // 1. Naive i-j-k (Cache-hostile on Matrix B)
    let t0 = Instant::now();
    let c_naive = matmul_naive(&a, &b);
    let dt_naive = t0.elapsed().as_secs_f64();
    let gflops_naive = compute_gflops(n, dt_naive);

    // 2. Cache-Tiled (Blocked L1 cache residency)
    let t1 = Instant::now();
    let c_tiled = matmul_tiled(&a, &b, 64);
    let dt_tiled = t1.elapsed().as_secs_f64();
    let gflops_tiled = compute_gflops(n, dt_tiled);

    // 3. Cache-Reordered (i-k-j) + Auto-SIMD Vectorization (Single Thread)
    let t2 = Instant::now();
    let c_ikj = matmul_ikj_simd(&a, &b);
    let dt_ikj = t2.elapsed().as_secs_f64();
    let gflops_ikj = compute_gflops(n, dt_ikj);

    // 4. Multi-Threaded Parallel + SIMD (All CPU cores)
    let t3 = Instant::now();
    let c_par = matmul_parallel(&a, &b, threads);
    let dt_par = t3.elapsed().as_secs_f64();
    let gflops_par = compute_gflops(n, dt_par);

    // Verify mathematical precision across all kernels
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
    println!(" [SPRINT 2] STATISTICAL QUANTIZATION (Q8_0) — Neural Weights {} x {}", n, n);
    println!("================================================================================");

    // Simulate realistic Transformer linear layer weights ~ N(0, 0.02^2)
    let weights = Tensor2D::random_normal(n, n, 0.02, 2026);
    let activations = Tensor2D::random_normal(n, n, 0.05, 7777);

    let stats = TensorStats::compute(&weights);
    println!(" [A] Statistical Weight Distribution Profile (Box-Muller Gaussian):");
    println!("     • Mean (μ):               {:+.6}", stats.mean);
    println!("     • Std Deviation (σ):      {:.6} (Variance σ² = {:.6})", stats.std_dev, stats.variance);
    println!("     • Range [Min, Max]:       [{:+.5}, {:+.5}]", stats.min, stats.max);
    println!("     • Outliers (|x - μ| > 3σ): {:.3}% (Theoretical Normal: ~0.270%)", stats.outlier_ratio_3sigma);
    println!();

    let t_q = Instant::now();
    let weights_q8 = QuantizedTensorQ8::quantize(&weights);
    let dt_q = t_q.elapsed().as_secs_f64();

    let report = weights_q8.evaluate_error(&weights);
    println!(" [B] Block-Wise Symmetric INT8 (Q8_0) Compression Telemetry:");
    println!("     • FP32 Memory Footprint:  {:.2} MB ({} bytes)", report.original_bytes as f64 / 1048576.0, report.original_bytes);
    println!("     • Q8_0 Memory Footprint:  {:.2} MB ({} bytes)", report.compressed_bytes as f64 / 1048576.0, report.compressed_bytes);
    println!("     • RAM Compression Ratio:  {:.2}x (71.9% memory bandwidth saved)", report.compression_ratio);
    println!("     • Quantization Speed:     {:.2} ms", dt_q * 1000.0);
    println!("     • RMSE (Root Mean Sq Err): {:.6e}", report.rmse);
    println!("     • Max Absolute Error:     {:.6e}", report.max_abs_error);
    println!("     • Signal-to-Noise (SQNR): {:.2} dB", report.sqnr_db);
    println!();

    // Compare FP32 vs Q8_0 Matrix Multiplication accuracy
    let weights_t = weights.transpose();
    let weights_t_q8 = QuantizedTensorQ8::quantize(&weights_t);
    let act_q8 = QuantizedTensorQ8::quantize(&activations);

    let c_fp32 = matmul_ikj_simd(&activations, &weights);
    let t_qgemm = Instant::now();
    let c_q8 = matmul_q8_0(&act_q8, &weights_t_q8);
    let dt_qgemm = t_qgemm.elapsed().as_secs_f64();

    let matmul_err = c_fp32.max_abs_diff(&c_q8);
    println!(" [C] Quantized INT8 Matrix Multiplication Verification:");
    println!("     • Q8_0 Dot-Product Time:  {:.2} ms", dt_qgemm * 1000.0);
    println!("     • Max Drift vs FP32 GEMM: {:.6e}", matmul_err);
    println!("================================================================================");
}

fn main() {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║               TINY-TENSOR-ENGINE v0.1.0 — LOW-LEVEL AI RUNTIME              ║");
    println!("║        Pure Rust • Zero External Dependencies • Cache & SIMD Optimized       ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!(" Detected Hardware Concurrency: {} logical CPU cores\n", threads);

    run_gemm_benchmark(512, threads);
    run_gemm_benchmark(1024, threads);
    run_quantization_benchmark(512);
}
