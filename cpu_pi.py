# cpu_pi.py

import random
import time

def estimate_pi_cpu_loop(n_points):
    """
    Estime Pi sur le CPU.
    C'est la méthode la plus lente (séquentielle).
    """
    print(f"--- 1. Calcul sur CPU avec une boucle for sur {n_points:,} points ---")
    
    start_time = time.time()
    points_inside_circle = 0
    
    for _ in range(n_points):
        x = random.random()
        y = random.random()
        if x**2 + y**2 <= 1.0:
            points_inside_circle += 1
            
    pi_estimate = 4 * points_inside_circle / n_points
    end_time = time.time()
    
    cpu_time = end_time - start_time
    
    print(f"Estimation de π (CPU) ≈ {pi_estimate}")
    print(f"Temps de calcul (CPU) : {cpu_time:.4f} secondes")
    return cpu_time