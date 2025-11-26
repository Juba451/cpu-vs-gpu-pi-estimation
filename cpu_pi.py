# cpu_pi.py

import numpy as np
import time

def estimate_pi_cpu(n_points):
    """
    Estime Pi en utilisant NumPy pour un calcul vectorisé sur le CPU.
    """
    start_time = time.perf_counter()
    
    # 1. Utiliser NumPy pour générer TOUS les points d'un coup (plus rapide)
    points = np.random.rand(n_points, 2)
    
    # 2. Calcul vectorisé
    distances_sq = points[:, 0]**2 + points[:, 1]**2
    points_inside_circle = np.sum(distances_sq <= 1)
    
    # 3. Estimation de Pi
    pi_estimate = 4 * points_inside_circle / n_points
    
    end_time = time.perf_counter()
    cpu_time = end_time - start_time
    
    # 4. La fonction retourne les résultats, elle n'affiche rien.
    return pi_estimate, cpu_time
