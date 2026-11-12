"""
Pure Python Baseline for Matrix Multiplication (GEMM).
Demonstrates why interpreted high-level loops are ~1,000x-5,000x slower than
cache-aligned, SIMD-vectorized Rust code on the exact same CPU hardware.
"""

import random
import time


def matmul_python_naive(n: int) -> float:
    random.seed(42)
    a = [random.gauss(0.0, 0.02) for _ in range(n * n)]
    b = [random.gauss(0.0, 0.02) for _ in range(n * n)]
    c = [0.0] * (n * n)

    t0 = time.perf_counter()
    for i in range(n):
        i_n = i * n
        for j in range(n):
            acc = 0.0
            for k in range(n):
                acc += a[i_n + k] * b[k * n + j]
            c[i_n + j] = acc
    elapsed = time.perf_counter() - t0
    return elapsed


if __name__ == "__main__":
    n = 256
    print(f"[Python Baseline] Running {n}x{n} naive GEMM in pure Python...")
    dt = matmul_python_naive(n)
    flops = 2.0 * (n ** 3)
    gflops = (flops / dt) / 1e9
    # Extrapolate 512x512 and 1024x1024 using O(N^3) scaling (8x and 64x)
    est_512_ms = dt * 8.0 * 1000.0
    est_1024_ms = dt * 64.0 * 1000.0
    print(f" -> {n}x{n} Time:             {dt * 1000.0:.2f} ms ({gflops:.4f} GFLOPS)")
    print(f" -> Extrapolated 512x512:   {est_512_ms:.2f} ms ({est_512_ms / 1000.0:.2f} s)")
    print(f" -> Extrapolated 1024x1024: {est_1024_ms:.2f} ms ({est_1024_ms / 1000.0:.2f} s)")
