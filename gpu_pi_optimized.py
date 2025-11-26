# gpu_pi_optimized.py

import time

try:
    import cupy as cp
    cuda_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void monte_carlo_pi(const float* x, const float* y, unsigned long long* counter, int n) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < n) {
            if (x[tid] * x[tid] + y[tid] * y[tid] <= 1.0f) {
                atomicAdd(counter, 1);
            }
        }
    }
    ''', 'monte_carlo_pi')
except (ImportError, NameError):
    cuda_kernel = None

def estimate_pi_gpu_optimized(n_points):
    """
    Estime Pi sur GPU avec un kernel CUDA optimisé.
    Retourne l'estimation et le temps de calcul.
    """
    if cuda_kernel is None:
        return None, float('inf')

    cp.cuda.runtime.deviceSynchronize()
    start_time = time.perf_counter()
    
    x = cp.random.rand(n_points, dtype=cp.float32)
    y = cp.random.rand(n_points, dtype=cp.float32)
    counter = cp.zeros(1, dtype=cp.uint64)
    
    threads_per_block = 256
    blocks_per_grid = (n_points + threads_per_block - 1) // threads_per_block
    
    cuda_kernel((blocks_per_grid,), (threads_per_block,), (x, y, counter, n_points))
    
    pi_estimate = 4 * counter[0] / n_points
    
    cp.cuda.runtime.deviceSynchronize()
    end_time = time.perf_counter()
    gpu_opt_time = end_time - start_time
    
    return pi_estimate, gpu_opt_time
