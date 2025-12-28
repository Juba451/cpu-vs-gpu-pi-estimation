# cpu_pi.py

import numpy as np
import time

def estimer_pi_cpu(nombre_simulations):
    """
    Estime Pi en utilisant NumPy pour un calcul vectorisé sur le CPU.
    Retourne l'estimation de Pi et le temps de calcul.
    """
    debut_chrono = time.perf_counter()
    
    # 1. Utiliser NumPy pour générer tous les points d'un coup (plus rapide)
    points = np.random.rand(nombre_simulations, 2)
    
    # 2. Calcul vectorisé
    distances_carre = points[:, 0]**2 + points[:, 1]**2
    points_dans_cercle = np.sum(distances_carre <= 1)
    
    # 3. Estimation de Pi
    estimation_pi = 4 * points_dans_cercle / nombre_simulations
    
    fin_chrono = time.perf_counter()
    temps_cpu = fin_chrono - debut_chrono
    
    # 4. La fonction retourne les résultats, elle n'affiche rien.
    return estimation_pi, temps_cpu
