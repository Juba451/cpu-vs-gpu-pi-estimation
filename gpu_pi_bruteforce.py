# gpu_pi_bruteforce.py
import time

def estimer_pi_gpu_bruteforce(cp, nombre_points):
    """
    Version GPU naive : génère tous les points d'un seul coup en mémoire GPU.
    Rapide, mais peut poser des problèmes si le nombre de points est trop grand.
    """
    debut_chrono = time.perf_counter()
    points = cp.random.rand(nombre_points, 2)
    distances_carrees = points[:, 0]**2 + points[:, 1]**2
    points_dans_cercle = cp.sum(distances_carrees <= 1)
    estimation_pi = 4 * points_dans_cercle / nombre_points
    cp.cuda.runtime.deviceSynchronize()
    fin_chrono = time.perf_counter()
    return estimation_pi, (fin_chrono - debut_chrono)