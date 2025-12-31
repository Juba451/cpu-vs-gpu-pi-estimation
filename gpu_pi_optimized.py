# gpu_pi_optimized.py
import time
try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
    GPU_DISPONIBLE = True
except:
    GPU_DISPONIBLE = False

def estimer_pi_gpu_optimise(nombre_points, taille_lot):
    if not GPU_DISPONIBLE: return None, float('inf')
    
    debut_chrono = time.perf_counter()
    total_points_dedans = 0
    nombre_de_lots = nombre_points // taille_lot
    
    for _ in range(nombre_de_lots):
        points = cp.random.rand(taille_lot, 2)
        dist_sq = points[:, 0]**2 + points[:, 1]**2
        dedans_ce_lot = cp.sum(dist_sq <= 1)
        total_points_dedans += dedans_ce_lot
        
    estimation_pi = 4 * total_points_dedans / nombre_points
    cp.cuda.runtime.deviceSynchronize()
    fin_chrono = time.perf_counter()
    return estimation_pi, (fin_chrono - debut_chrono)
