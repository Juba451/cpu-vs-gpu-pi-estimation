# gpu_pi_bruteforce.py
import time
try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
    GPU_DISPONIBLE = True
except:
    GPU_DISPONIBLE = False

def estimer_pi_gpu_bruteforce(nombre_points):
    if not GPU_DISPONIBLE: return None, float('inf')
    
    debut_chrono = time.perf_counter()
    points = cp.random.rand(nombre_points, 2)
    dist_sq = points[:, 0]**2 + points[:, 1]**2
    points_dans_cercle = cp.sum(dist_sq <= 1)
    estimation_pi = 4 * points_dans_cercle / nombre_points
    cp.cuda.runtime.deviceSynchronize()
    fin_chrono = time.perf_counter()
    return estimation_pi, (fin_chrono - debut_chrono)