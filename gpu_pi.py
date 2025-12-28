# gpu_pi.py

import time

try:
    import cupy as cp
    CUPY_DISPONIBLE = True
except ImportError:
    CUPY_DISPONIBLE = False

def estimer_pi_gpu(total_simulations):
    """
    Estime Pi sur GPU en utilisant CuPy pour un calcul parallèle "brute force".
    """
    if not CUPY_DISPONIBLE:
        print("Avertissement : CuPy n'est pas disponible. Le calcul GPU est ignoré.")
        return None, float('inf')

    debut_chrono = time.perf_counter()
    
    # On génère tous les points d'un seul coup sur le GPU
    points = cp.random.rand(total_simulations, 2)
    
    # On fait tous les calculs en parallèle
    distances_carre = points[:, 0]**2 + points[:, 1]**2
    points_dans_cercle = cp.sum(distances_carre <= 1)
    
    estimation_pi_gpu = 4 * points_dans_cercle / total_simulations
    
    cp.cuda.runtime.deviceSynchronize()
    fin_chrono = time.perf_counter()
    temps_gpu = fin_chrono - debut_chrono
    
    return estimation_pi_gpu, temps_gpu
