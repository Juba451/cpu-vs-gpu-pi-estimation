# gpu_pi_optimise.py
import time

def estimer_pi_gpu_optimise(cp, nombre_points, taille_lot):
    
    debut_chrono = time.perf_counter()
    total_points_dedans = 0
    nombre_de_lots = nombre_points // taille_lot
    
    for _ in range(nombre_de_lots):
        points = cp.random.rand(taille_lot, 2)
        distances_carrees = points[:, 0]**2 + points[:, 1]**2
        dedans_ce_lot = cp.sum(distances_carrees <= 1)
        total_points_dedans += dedans_ce_lot
        
    estimation_pi = 4 * total_points_dedans / nombre_points
    cp.cuda.runtime.deviceSynchronize()
    fin_chrono = time.perf_counter()
    return estimation_pi, (fin_chrono - debut_chrono)
