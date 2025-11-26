# gpu_pi_batch.py

import time

try:
    import cupy as cp
except ImportError:
    pass

def estimate_pi_gpu_batch(total_points, batch_size):
    """
    Estime Pi sur GPU avec une méthode par lots.
    Retourne l'estimation et le temps de calcul.
    """
    try:
        cp.cuda.runtime.deviceSynchronize()
        start_time = time.perf_counter()
        
        total_inside_circle = 0
        num_batches = total_points // batch_size
        
        for _ in range(num_batches):
            points = cp.random.rand(batch_size, 2)
            distances_sq = points[:, 0]**2 + points[:, 1]**2
            inside_this_batch = cp.sum(distances_sq <= 1)
            total_inside_circle += inside_this_batch
            
        pi_estimate_gpu = 4 * total_inside_circle / total_points
        
        cp.cuda.runtime.deviceSynchronize()
        end_time = time.perf_counter()
        gpu_time = end_time - start_time
        
        return pi_estimate_gpu, gpu_time
        
    except NameError:
        return None, float('inf')
