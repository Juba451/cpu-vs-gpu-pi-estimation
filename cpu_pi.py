# cpu_pi.py

import numpy as np
import time

def estimer_pi_cpu(nombre_points):
    """
    Estime Pi en utilisant NumPy pour un calcul vectorisé sur le CPU.
    Retourne l'estimation de Pi et le temps de calcul.
    
    Note: chaque point (ou simulation) représente les coordonnées (x, y) d'une fléchette virtuelle
    """
    debut_chrono = time.perf_counter()
    
    # 1. Utiliser NumPy pour générer tous les points d'un coup (plus rapide)
    points = np.random.rand(nombre_points, 2)
    
    # 2. Calcul vectorisé + compter les points à l'intérieur du cercle
    distances_carres = points[:, 0]**2 + points[:, 1]**2
    points_dans_cercle = np.sum(distances_carre <= 1)
    
    # 3. Estimation de Pi
    estimation_pi = 4 * points_dans_cercle / nombre_points
    
    fin_chrono = time.perf_counter()
    temps_de_calcul_cpu = fin_chrono - debut_chrono
    
    # 4. La fonction retourne les résultats, elle n'affiche rien.
    return estimation_pi, temps_de_calcul_cpu
