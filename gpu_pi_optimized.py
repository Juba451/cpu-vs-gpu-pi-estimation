# gpu_pi_optimized.py

import time

try:
    import cupy as cp
    CUPY_DISPONIBLE = True

    kernel_cuda = cp.RawKernel(r'''
    extern "C" __global__
    void monte_carlo_pi(const float* x, const float* y, unsigned long long* counter, int n) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < n) {
            if (x[tid] * x[tid] + y[tid] * y[tid] <= 1.0f) {
                // On utilise "counter" ici, qui a été défini dans les arguments juste au-dessus.
                atomicAdd(counter, 1);
            }
        }
    }
    ''', 'monte_carlo_pi')
except (ImportError, NameError):
    CUPY_DISPONIBLE = False
    kernel_cuda = None

def estimer_pi_gpu_optimise(nombre_simulations):
    """
    Estime Pi sur GPU avec un kernel CUDA optimisé.
    Retourne l'estimation de Pi et le temps de calcul.
    """
    if not CUPY_DISPONIBLE or kernel_cuda is None:
        print("Avertissement : CuPy n'est pas disponible. Le calcul GPU optimisé est ignoré.")
        return None, float('inf')

    print(f"\n--- 3. Calcul sur GPU Optimisé (Kernel) sur {nombre_simulations:,} points ---")
    
    cp.cuda.runtime.deviceSynchronize()
    debut_chrono = time.perf_counter()
    
    x = cp.random.rand(nombre_simulations, dtype=cp.float32)
    y = cp.random.rand(nombre_simulations, dtype=cp.float32)
    # La variable Python s'appelle "compteur"
    compteur = cp.zeros(1, dtype=cp.uint64)
    
    threads_par_bloc = 256
    blocs_par_grille = (nombre_simulations + threads_par_bloc - 1) // threads_par_bloc
    
    # --- CORRECTION ICI ---
    # Quand on appelle le kernel, l'ordre des arguments compte.
    # On envoie bien (x, y, compteur, nombre_simulations).
    # Le kernel va les recevoir dans le même ordre.
    kernel_cuda((blocs_par_grille,), (threads_par_bloc,), (x, y, compteur, nombre_simulations))
    
    # On utilise la variable Python "compteur" pour lire le résultat
    estimation_pi = 4 * compteur[0] / nombre_simulations
    
    cp.cuda.runtime.deviceSynchronize()
    fin_chrono = time.perf_counter()
    temps_gpu_opt = fin_chrono - debut_chrono
    
    return estimation_pi, temps_gpu_opt
