# gpu_pi_batch.py

import time

# On essaie d'importer CuPy et on définit une variable pour savoir si ça a marché.
try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
    CUPY_ET_GPU_DISPONIBLE = True
except (ImportError, cp.cuda.runtime.CudaAPIError):
    CUPY_ET_GPU_DISPONIBLE = False

def estimer_pi_gpu_par_lots(total_simulations, taille_lot):
    """
    Estime Pi sur GPU avec une méthode par lots.
    Retourne l'estimation de Pi et le temps de calcul.
    """
    # On vérifie si CuPy est disponible au début de la fonction.
    if not CUPY_DISPONIBLE:
        # Si non, on s'arrête tout de suite.
        print("Avertissement : CuPy ou un GPU/Driver compatible n'est pas disponible. Le calcul GPU par lots est ignoré.")
        return None, float('inf')

   
    debut_chrono = time.perf_counter()
    
    total_points_dedans = 0
    nombre_de_lots = total_simulations // taille_lot
    
    for _ in range(nombre_de_lots):
        points = cp.random.rand(taille_lot, 2)
        distances_carre = points[:, 0]**2 + points[:, 1]**2
        dedans_ce_lot = cp.sum(distances_carre <= 1)
        total_points_dedans += dedans_ce_lot
        
    estimation_pi_gpu = 4 * total_points_dedans / total_simulations
    
    cp.cuda.runtime.deviceSynchronize()
    fin_chrono = time.perf_counter()
    temps_gpu = fin_chrono - debut_chrono
    
    return estimation_pi_gpu, temps_gpu
