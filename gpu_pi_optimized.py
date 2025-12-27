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

def estimater_pi_gpu_optimise(nombre_simulations):
    """
    Estime Pi sur GPU avec un kernel CUDA optimisé.
    Retourne l'estimation et le temps de calcul.
    """
    if cuda_kernel is None:
        return None, float('inf')

    cp.cuda.runtime.deviceSynchronize()
    start_time = time.perf_counter()
    
    x = cp.random.rand(nombre_simulations, dtype=cp.float32)
    y = cp.random.rand(nombre_simulations, dtype=cp.float32)
    compteur = cp.zeros(1, dtype=cp.uint64)
    
    threads_par_bloc = 256
    blocs_par_grille = (nombre_simulations + threads_par_bloc - 1) // threads_par_bloc
    
     kernel_cuda((blocs_par_grille,), (threads_par_bloc,), (x, y, compteur, nombre_simulations))
    
    estimation_pi = 4 * compteur[0] / nombre_simulations
    
    cp.cuda.runtime.deviceSynchronize()
    fin_chrono = time.perf_counter()
    temps_gpu_opt = fin_chrono - debut_chrono
    
    return estimation_pi, temps_gpu_opt
