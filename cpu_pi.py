

import numpy as np
import time

def estimer_pi_cpu(nombre_points):
    """
    Estime Pi en utilisant NumPy pour un calcul vectorisé sur le CPU.
    Retourne l'estimation de Pi et le temps de calcul.
    
    Note: chaque point (ou simulation) représente les coordonnées (x, y) d'une fléchette virtuelle
    """
    debut_chrono = time.perf_counter()
<<<<<<< HEAD
   points_dans_cercle = 0

    for _ in range(nombre_points):
        x = random.random()
        y = random.random()
        if x**2 + y**2 <= 1.0:
            points_dans_cercle += 1
=======
    
    # 1. Utiliser NumPy pour générer tous les points d'un coup (plus rapide)
    points = np.random.rand(nombre_points, 2)
    
    # 2. Calcul vectorisé + compter les points à l'intérieur du cercle
    distances_carres = points[:, 0]**2 + points[:, 1]**2
    points_dans_cercle = np.sum(distances_carre <= 1)
    
    # 3. Estimation de Pi
>>>>>>> 016c4d18d7d1c3cef2fe31f548681a72213d42fe
    estimation_pi = 4 * points_dans_cercle / nombre_points
    fin_chrono = time.perf_counter()
    return estimation_pi, (fin_chrono - debut_chrono)