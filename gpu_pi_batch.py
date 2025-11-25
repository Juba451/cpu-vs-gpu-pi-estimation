# gpu_pi_batch.py

import time

try:
    import cupy as cp
except ImportError:
    pass

def estimate_pi_gpu_batch(total_points, batch_size):
    """
    Estime Pi en utilisant CuPy pour un calcul en parallèle sur le GPU,
    optimisé pour la mémoire en travaillant "par lots" (batches).
    """
    print(f"\n--- 2. Calcul sur GPU 'par lots' sur {total_points:,} points ---")
    
    try:
        cp.cuda.runtime.deviceSynchronize()
        start_time = time.time()
        
        total_inside_circle = 0
        num_batches = total_points // batch_size
        
        for _ in range(num_batches):
            x_gpu = cp.random.rand(batch_size)
            y_gpu = cp.random.rand(batch_size)
            distances = x_gpu**2 + y_gpu**2
            inside_this_batch = cp.sum(distances <= 1)
            total_inside_circle += inside_this_batch
            
        pi_estimate_gpu = 4 * total_inside_circle / total_points
        
        cp.cuda.runtime.deviceSynchronize()
        end_time = time.time()
        
        gpu_time = end_time - start_time
        
        print(f"Estimation de π (GPU par lots) ≈ {float(pi_estimate_gpu):.6f}")
        print(f"Temps de calcul (GPU par lots) : {gpu_time:.4f} secondes")
        return gpu_time
    except NameError:
        print("Erreur : CuPy n'est pas installé ou un GPU n'est pas disponible.")
        return float('inf')
