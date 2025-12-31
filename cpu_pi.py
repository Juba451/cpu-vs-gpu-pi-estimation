

import numpy as np
import time

def estimer_pi_cpu(nombre_points):
    """
    Estime Pi en utilisant NumPy pour un calcul vectorisé sur le CPU.
    Retourne l'estimation de Pi et le temps de calcul.
    
    Note: chaque point (ou simulation) représente les coordonnées (x, y) d'une fléchette virtuelle
    """
    debut_chrono = time.perf_counter()
    points_dans_cercle = 0

    for _ in range(nombre_points):
        x = random.random()
        y = random.random()
        if x**2 + y**2 <= 1.0:
            points_dans_cercle += 1
    estimation_pi = 4 * points_dans_cercle / nombre_points
    fin_chrono = time.perf_counter()
    return estimation_pi, (fin_chrono - debut_chrono)
