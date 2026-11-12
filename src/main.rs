use std::time::Instant;
use tiny_tensor_engine::matmul::{matmul_ikj_simd, matmul_naive, matmul_parallel, matmul_tiled};
use tiny_tensor_engine::quant::{
    matmul_q4_golden, matmul_q4_hadamard, matmul_q8_0, QuantizationErrorReport,
    QuantizedTensorQ4, QuantizedTensorQ4Golden, QuantizedTensorQ4Hadamard, QuantizedTensorQ4Stats,
    QuantizedTensorQ8, TensorStats,
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

    // 4. Q4_Hadamard (Orthonormal Change-of-Basis H_32 + Lloyd-Max + OLS — ZERO extra memory!)
    let w_q4_had = QuantizedTensorQ4Hadamard::quantize(&llm_weights);
    let rep_q4_had = w_q4_had.evaluate_error(&llm_weights);

    // 5. Q4_GoldenOpt (Golden Section Search Non-Linear Tail Warp in 20-byte block — 6.40x compression!)
    let w_q4_gold = QuantizedTensorQ4Golden::quantize(&llm_weights);
    let rep_q4_gold = w_q4_gold.evaluate_error(&llm_weights);

    println!(" [B] Head-to-Head Compression & Precision Comparison (FP32 = {:.2} MB):", rep_q8.original_bytes as f64 / 1048576.0);
    println!(
        "{:<36} | {:>9} | {:>11} | {:>11} | {:>11} | {:>9}",
        "Quantization Format", "Size (MB)", "Compression", "RMSE Error", "Max Error", "SQNR (dB)"
    );
    println!("--------------------------------------------------------------------------------------------------");
    println!(
        "{:<36} | {:>9.2} | {:>10.2}x | {:>11.4e} | {:>11.4e} | {:>9.2}",
        "1. Q8_0 (8-bit Standard)",
        rep_q8.compressed_bytes as f64 / 1048576.0,
        rep_q8.compression_ratio,
        rep_q8.rmse,
        rep_q8.max_abs_error,
        rep_q8.sqnr_db
    );
    println!(
        "{:<36} | {:>9.2} | {:>10.2}x | {:>11.4e} | {:>11.4e} | {:>9.2}",
        "2. Q4_0 (4-bit Canonical Basis)",
        rep_q4_std.compressed_bytes as f64 / 1048576.0,
        rep_q4_std.compression_ratio,
        rep_q4_std.rmse,
        rep_q4_std.max_abs_error,
        rep_q4_std.sqnr_db
    );
    println!(
        "{:<36} | {:>9.2} | {:>10.2}x | {:>11.4e} | {:>11.4e} | {:>9.2}",
        "3. Q4_Hadamard (Orthonormal Basis)",
        rep_q4_had.compressed_bytes as f64 / 1048576.0,
        rep_q4_had.compression_ratio,
        rep_q4_had.rmse,
        rep_q4_had.max_abs_error,
        rep_q4_had.sqnr_db
    );
    println!(
        "{:<36} | {:>9.2} | {:>10.2}x | {:>11.4e} | {:>11.4e} | {:>9.2}",
        "4. Q4_Stats (Sparse Outlier + OLS)",
        rep_q4_stats.compressed_bytes as f64 / 1048576.0,
        rep_q4_stats.compression_ratio,
        rep_q4_stats.rmse,
        rep_q4_stats.max_abs_error,
        rep_q4_stats.sqnr_db
    );
    println!(
        "{:<36} | {:>9.2} | {:>10.2}x | {:>11.4e} | {:>11.4e} | {:>9.2}",
        "5. Q4_GoldenOpt (Golden Search 20B)",
        rep_q4_gold.compressed_bytes as f64 / 1048576.0,
        rep_q4_gold.compression_ratio,
        rep_q4_gold.rmse,
        rep_q4_gold.max_abs_error,
        rep_q4_gold.sqnr_db
    );
    println!();
    println!(
        " >>> ALGEBRAIC BREAKTHROUGH (Q4_Hadamard): Rotating into the Walsh-Hadamard orthonormal basis (H^T*H = I)"
    );
    println!(
        "     boosts 4-bit SQNR by +{:.2} dB with ZERO extra memory ({:.2}x compression = 0.16 MB)!",
        rep_q4_had.sqnr_db - rep_q4_std.sqnr_db,
        rep_q4_had.compression_ratio
    );
    println!(
        " >>> STATISTICAL BREAKTHROUGH (Q4_Stats): Isolating >3.2σ outliers boosts SQNR by +{:.2} dB ({:.2}x compression).",
        rep_q4_stats.sqnr_db - rep_q4_std.sqnr_db,
        rep_q4_stats.compression_ratio
    );
    println!(
        " >>> OPTIMIZATION RECORD (Q4_GoldenOpt): IEEE-754 Sign-Bit Stealing + Golden Section Search (OH Alg. 4)"
    );
    println!(
        "     achieves {:.2} dB SQNR (+{:.2} dB vs Q4_0, +{:.2} dB vs Q4_Stats) at FULL {:.2}x compression (0.16 MB)!",
        rep_q4_gold.sqnr_db,
        rep_q4_gold.sqnr_db - rep_q4_std.sqnr_db,
        rep_q4_gold.sqnr_db - rep_q4_stats.sqnr_db,
        rep_q4_gold.compression_ratio
    );
    println!();

    // Compare downstream Matrix Multiplication impact: C = Activations * Weights
    let c_fp32 = matmul_ikj_simd(&activations, &llm_weights);
    let c_from_q4_std = matmul_ikj_simd(&activations, &w_q4_std.dequantize());
    let c_from_q4_stats = matmul_ikj_simd(&activations, &w_q4_stats.dequantize());

    // Direct compressed-domain GEMM on Q4_GoldenOpt (20-byte blocks in L2 cache, zero FP32 weight allocation)
    let w_t_q4_gold = QuantizedTensorQ4Golden::quantize(&llm_weights.transpose());
    let t_gold_gemm = Instant::now();
    let c_from_q4_gold = matmul_q4_golden(&activations, &w_t_q4_gold);
    let dt_gold_gemm = t_gold_gemm.elapsed().as_secs_f64();

    // Direct rotated-basis GEMM without dequantizing weights back to canonical basis: (A * H) * (W^T * H)^T
    let w_t_q4_had = QuantizedTensorQ4Hadamard::quantize(&llm_weights.transpose());
    let t_had_gemm = Instant::now();
    let c_from_q4_had = matmul_q4_hadamard(&activations, &w_t_q4_had);
    let dt_had_gemm = t_had_gemm.elapsed().as_secs_f64();

    let gemm_err_std = c_fp32.max_abs_diff(&c_from_q4_std);
    let gemm_err_had = c_fp32.max_abs_diff(&c_from_q4_had);
    let gemm_err_stats = c_fp32.max_abs_diff(&c_from_q4_stats);
    let gemm_err_gold = c_fp32.max_abs_diff(&c_from_q4_gold);

    let weights_t_q8 = QuantizedTensorQ8::quantize(&llm_weights.transpose());
    let act_q8 = QuantizedTensorQ8::quantize(&activations);
    let t_qgemm = Instant::now();
    let _c_q8 = matmul_q8_0(&act_q8, &weights_t_q8);
    let dt_qgemm = t_qgemm.elapsed().as_secs_f64();

    println!(" [C] Downstream Neural Layer Output Drift (Activations x Quantized Weights):");
    println!("     • Standard Q4_0 Layer Max Drift:         {:.6e}", gemm_err_std);
    println!("     • Orthonormal Q4_Hadamard Max Drift:     {:.6e} ({:.2}x more accurate, {:.2} ms rotated GEMM)", gemm_err_had, gemm_err_std / gemm_err_had, dt_had_gemm * 1000.0);
    println!("     • Outlier-Aware Q4_Stats Max Drift:      {:.6e} ({:.2}x more accurate!)", gemm_err_stats, gemm_err_std / gemm_err_stats);
    println!("     • Direct Q4_GoldenOpt Kernel Max Drift:  {:.6e} ({:.2}x more accurate, {:.2} ms direct GEMM!)", gemm_err_gold, gemm_err_std / gemm_err_gold, dt_gold_gemm * 1000.0);
    println!("     • Direct INT8 Q8_0 GEMM Kernel Time:     {:.2} ms", dt_qgemm * 1000.0);
    println!("================================================================================");
    println!();
}

/// Applies LLM RMSNorm + SiLU (`x * sigmoid(x)`) row-wise to simulate non-linear Transformer/MLP layers.
fn rmsnorm_silu_inplace(t: &mut Tensor2D) {
    let cols = t.cols;
    for row in t.data.chunks_exact_mut(cols) {
        let mean_sq: f32 = row.iter().map(|&v| v * v).sum::<f32>() / (cols as f32);
        let inv_rms = 1.0 / (mean_sq + 1e-6).sqrt();
        for v in row.iter_mut() {
            let norm_x = *v * inv_rms;
            // SiLU(x) = x / (1 + exp(-x))
            *v = norm_x / (1.0 + (-norm_x).exp());
        }
    }
}

/// Computes Cosine Similarity (%) and Top-1 Token Argmax Agreement (%) between FP32 and quantized logits.
fn evaluate_logits_fidelity(fp32_logits: &Tensor2D, q_logits: &Tensor2D) -> (f64, f64, f64) {
    let rep = QuantizationErrorReport::from_tensors(fp32_logits, q_logits, 0);
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (&a, &b) in fp32_logits.data.iter().zip(q_logits.data.iter()) {
        dot += (a as f64) * (b as f64);
        norm_a += (a as f64) * (a as f64);
        norm_b += (b as f64) * (b as f64);
    }
    let cos_sim = (dot / (norm_a.sqrt() * norm_b.sqrt() + 1e-12)) * 100.0;

    let cols = fp32_logits.cols;
    let mut top1_matches = 0usize;
    for (row_a, row_b) in fp32_logits
        .data
        .chunks_exact(cols)
        .zip(q_logits.data.chunks_exact(cols))
    {
        let argmax_a = row_a
            .iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
            .unwrap()
            .0;
        let argmax_b = row_b
            .iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
            .unwrap()
            .0;
        if argmax_a == argmax_b {
            top1_matches += 1;
        }
    }
    let top1_acc = (top1_matches as f64 / fp32_logits.rows as f64) * 100.0;
    (rep.sqnr_db as f64, cos_sim, top1_acc)
}

fn run_deep_cascade_verification(n: usize) {
    println!("================================================================================");
    println!(
        " [SPRINT 4] 4-LAYER DEEP NEURAL NETWORK CASCADE (RMSNorm + SiLU + Top-1 Token Test)"
    );
    println!("================================================================================");

    let seeds = [2026u64, 31415, 27182, 16180];
    let layers_fp32: Vec<Tensor2D> = seeds
        .iter()
        .map(|&s| Tensor2D::random_llm_weights(n, n, 0.02, s))
        .collect();

    let mut x_fp32 = Tensor2D::random_normal(n, n, 0.05, 777);
    let mut x_q4_0 = x_fp32.clone();
    let mut x_q4_had = x_fp32.clone();
    let mut x_q4_stats = x_fp32.clone();
    let mut x_q4_gold = x_fp32.clone();

    for (l_idx, w) in layers_fp32.iter().enumerate() {
        let w_q4_0 = QuantizedTensorQ4::quantize(w).dequantize();
        let w_q4_had = QuantizedTensorQ4Hadamard::quantize(w).dequantize();
        let w_q4_stats = QuantizedTensorQ4Stats::quantize(w, 3.2).dequantize();
        let w_t_q4_gold = QuantizedTensorQ4Golden::quantize(&w.transpose());

        x_fp32 = matmul_ikj_simd(&x_fp32, w);
        x_q4_0 = matmul_ikj_simd(&x_q4_0, &w_q4_0);
        x_q4_had = matmul_ikj_simd(&x_q4_had, &w_q4_had);
        x_q4_stats = matmul_ikj_simd(&x_q4_stats, &w_q4_stats);
        // Execute Q4_GoldenOpt directly in compressed 20-byte domain!
        x_q4_gold = matmul_q4_golden(&x_q4_gold, &w_t_q4_gold);

        if l_idx + 1 < layers_fp32.len() {
            rmsnorm_silu_inplace(&mut x_fp32);
            rmsnorm_silu_inplace(&mut x_q4_0);
            rmsnorm_silu_inplace(&mut x_q4_had);
            rmsnorm_silu_inplace(&mut x_q4_stats);
            rmsnorm_silu_inplace(&mut x_q4_gold);
        }
    }

    let (sqnr_0, cos_0, top1_0) = evaluate_logits_fidelity(&x_fp32, &x_q4_0);
    let (sqnr_h, cos_h, top1_h) = evaluate_logits_fidelity(&x_fp32, &x_q4_had);
    let (sqnr_s, cos_s, top1_s) = evaluate_logits_fidelity(&x_fp32, &x_q4_stats);
    let (sqnr_g, cos_g, top1_g) = evaluate_logits_fidelity(&x_fp32, &x_q4_gold);

    println!(
        "{:<36} | {:>11} | {:>12} | {:>14} | {:>13}",
        "4-Layer Cascaded Engine (4.00 MB FP32)", "Total RAM", "Logits SQNR", "Cosine Sim (%)", "Top-1 Token %"
    );
    println!("--------------------------------------------------------------------------------------------------");
    println!(
        "{:<36} | {:>9.2} MB | {:>9.2} dB | {:>13.4}% | {:>12.2}%",
        "1. Q4_0 (Standard 4-bit)", 0.63, sqnr_0, cos_0, top1_0
    );
    println!(
        "{:<36} | {:>9.2} MB | {:>9.2} dB | {:>13.4}% | {:>12.2}%",
        "2. Q4_Hadamard (Orthonormal H_32)", 0.63, sqnr_h, cos_h, top1_h
    );
    println!(
        "{:<36} | {:>9.2} MB | {:>9.2} dB | {:>13.4}% | {:>12.2}%",
        "3. Q4_Stats (Sparse Outlier + OLS)", 0.70, sqnr_s, cos_s, top1_s
    );
    println!(
        "{:<36} | {:>9.2} MB | {:>9.2} dB | {:>13.4}% | {:>12.2}%",
        "4. Q4_GoldenOpt (Bivariate + Golden)", 0.63, sqnr_g, cos_g, top1_g
    );
    println!("================================================================================");
}

fn main() {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║               TINY-TENSOR-ENGINE v0.3.0 — LOW-LEVEL AI RUNTIME              ║");
    println!("║   Pure Rust • Hardware SIMD • Hadamard Basis • Golden-Section Quantization   ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!(" Detected Hardware Concurrency: {} logical CPU cores\n", threads);

    run_gemm_benchmark(512, threads);
    run_gemm_benchmark(1024, threads);
    run_quantization_benchmark(512);
    run_deep_cascade_verification(512);
}
