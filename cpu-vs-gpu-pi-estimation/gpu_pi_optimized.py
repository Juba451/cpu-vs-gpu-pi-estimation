# gpu_pi_optimized.py

import time

try:
    import cupy as cp
except ImportError:
    pass

# On définit le kernel en dehors de la fonction pour ne le compiler qu'une fois
try:
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
except NameError:
    cuda_kernel = None

def estimate_pi_gpu_optimized(n_points):
    """
    Estime Pi en utilisant un kernel CUDA optimisé pour le GPU.
    """
    print(f"\n--- 3. Calcul sur GPU Optimisé avec un Kernel CUDA sur {n_points:,} points ---")
    
    if cuda_kernel is None:
        print("Erreur : CuPy n'est pas installé ou un GPU n'est pas disponible.")
        return float('inf')

    cp.cuda.runtime.deviceSynchronize()
    start_time = time.time()
    
    x = cp.random.rand(n_points, dtype=cp.float32)
    y = cp.random.rand(n_points, dtype=cp.float32)
    counter = cp.zeros(1, dtype=cp.uint64)
    
    threads_per_block = 256
    blocks_per_grid = (n_points + threads_per_block - 1) // threads_per_block
    
    cuda_kernel((blocks_per_grid,), (threads_per_block,), (x, y, counter, n_points))
    
    pi_estimate = 4 * counter[0] / n_points
    
    cp.cuda.runtime.deviceSynchronize()
    end_time = time.time()
    
    gpu_opt_time = end_time - start_time
    
    print(f"Estimation de π (GPU Optimisé) ≈ {float(pi_estimate):.6f}")
    print(f"Temps de calcul (GPU Optimisé) : {gpu_opt_time:.4f} secondes")
    return gpu_opt_time