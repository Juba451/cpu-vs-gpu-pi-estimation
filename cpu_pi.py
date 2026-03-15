

import random
import time
import random

def estimer_pi_cpu(nombre_points):
    """
    Méthode de référence (lente) : boucle Python classique, un point à la fois.
    Chaque point est une fléchette virtuelle tirée aléatoirement dans le carré [0,1]².
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
