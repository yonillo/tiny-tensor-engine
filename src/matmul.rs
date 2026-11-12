//! High-Performance General Matrix Multiplication (GEMM) Kernels.
//!
//! Demonstrates how Computer Architecture (Cache Hierarchy, Instruction Pipelining, SIMD registers,
//! and OS Threads from IC1 & IC2) accelerates AI compute by orders of magnitude:
//!
//! 1. `matmul_naive`: Classic `i -> j -> k` loop order. Suffers from L1/L2 cache thrashing because
//!    `B[k, j]` jumps `cols * 4` bytes on every inner-loop step.
//! 2. `matmul_ikj_simd`: Reorders loops to `i -> k -> j`. Both `B[k, j]` and `C[i, j]` are accessed
//!    sequentially with stride 1, unlocking hardware cache prefetching and LLVM AVX2/FMA vectorization.
//! 3. `matmul_tiled`: L1 Cache Blocking (`TILE = 64`), keeping working sub-matrices inside the 32KB L1 cache.
//! 4. `matmul_parallel`: Multi-threaded execution using `std::thread::scope` across all CPU cores
//!    with compile-time memory safety (zero locks, zero race conditions).

use crate::tensor::Tensor2D;

/// 1. Naive `i -> j -> k` Matrix Multiplication.
/// Complexity: O(M * N * K), but high L1 data cache miss rate on matrix `b`.
pub fn matmul_naive(a: &Tensor2D, b: &Tensor2D) -> Tensor2D {
    assert_eq!(a.cols, b.rows, "Incompatible dimensions for GEMM");
    let (m, k_dim, n) = (a.rows, a.cols, b.cols);
    let mut c = Tensor2D::zeros(m, n);

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k in 0..k_dim {
                sum += a.data[i * k_dim + k] * b.data[k * n + j];
            }
            c.data[i * n + j] = sum;
        }
    }
    c
}

/// 2. Cache-Friendly `i -> k -> j` GEMM with automatic SIMD vectorization.
/// By broadcasting `a[i, k]` into a register and sweeping sequentially across row `k` of `b`
/// and row `i` of `c`, every 64-byte cache line loaded from RAM is 100% utilized.
pub fn matmul_ikj_simd(a: &Tensor2D, b: &Tensor2D) -> Tensor2D {
    assert_eq!(a.cols, b.rows, "Incompatible dimensions for GEMM");
    let (m, k_dim, n) = (a.rows, a.cols, b.cols);
    let mut c = Tensor2D::zeros(m, n);

    for i in 0..m {
        let c_row = &mut c.data[i * n..(i + 1) * n];
        let a_row = &a.data[i * k_dim..(i + 1) * k_dim];

        for (k, &a_val) in a_row.iter().enumerate() {
            let b_row = &b.data[k * n..(k + 1) * n];
            // Sequential stride-1 Fused Multiply-Add (FMA) across contiguous slices
            for (c_val, &b_val) in c_row.iter_mut().zip(b_row.iter()) {
                *c_val += a_val * b_val;
            }
        }
    }
    c
}

/// 3. Cache-Tiled (Blocked) GEMM.
/// Partitions matrices into `TILE x TILE` blocks so hot sub-matrices remain resident in L1/L2 cache.
pub fn matmul_tiled(a: &Tensor2D, b: &Tensor2D, tile_size: usize) -> Tensor2D {
    assert_eq!(a.cols, b.rows, "Incompatible dimensions for GEMM");
    let (m, k_dim, n) = (a.rows, a.cols, b.cols);
    let mut c = Tensor2D::zeros(m, n);

    for i0 in (0..m).step_by(tile_size) {
        let i_max = (i0 + tile_size).min(m);
        for k0 in (0..k_dim).step_by(tile_size) {
            let k_max = (k0 + tile_size).min(k_dim);
            for j0 in (0..n).step_by(tile_size) {
                let j_max = (j0 + tile_size).min(n);

                for i in i0..i_max {
                    let c_slice = &mut c.data[i * n + j0..i * n + j_max];
                    for k in k0..k_max {
                        let a_val = a.data[i * k_dim + k];
                        let b_slice = &b.data[k * n + j0..k * n + j_max];
                        for (c_val, &b_val) in c_slice.iter_mut().zip(b_slice.iter()) {
                            *c_val += a_val * b_val;
                        }
                    }
                }
            }
        }
    }
    c
}

/// 4. Multi-Threaded Parallel SIMD GEMM (IC2 Tema 4: Procesos, Hilos y Concurrencia Segura).
/// Splits output rows across physical/logical CPU threads using `std::thread::scope`.
/// Rust's borrow checker proves at compile time that disjoint mutable slices `chunks_mut`
/// cannot cause data races, requiring zero mutexes or atomic locks during computation.
pub fn matmul_parallel(a: &Tensor2D, b: &Tensor2D, num_threads: usize) -> Tensor2D {
    assert_eq!(a.cols, b.rows, "Incompatible dimensions for GEMM");
    let (m, k_dim, n) = (a.rows, a.cols, b.cols);
    let mut c = Tensor2D::zeros(m, n);

    let threads = num_threads.max(1).min(m);
    let rows_per_thread = m.div_ceil(threads);
    let elements_per_chunk = rows_per_thread * n;

    std::thread::scope(|s| {
        for (chunk_idx, c_chunk) in c.data.chunks_mut(elements_per_chunk).enumerate() {
            let row_start = chunk_idx * rows_per_thread;
            let chunk_rows = c_chunk.len() / n;
            let a_slice = &a.data[row_start * k_dim..(row_start + chunk_rows) * k_dim];
            let b_ref = &b.data;

            s.spawn(move || {
                for i in 0..chunk_rows {
                    let c_row = &mut c_chunk[i * n..(i + 1) * n];
                    let a_row = &a_slice[i * k_dim..(i + 1) * k_dim];

                    for (k, &a_val) in a_row.iter().enumerate() {
                        let b_row = &b_ref[k * n..(k + 1) * n];
                        for (c_val, &b_val) in c_row.iter_mut().zip(b_row.iter()) {
                            *c_val += a_val * b_val;
                        }
                    }
                }
            });
        }
    });

    c
}
