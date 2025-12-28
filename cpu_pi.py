# cpu_pi.py

import random
import time

def estimer_pi_cpu(nombre_simulations):
    """
    Estime Pi en utilisant une boucle for simple sur le CPU.
    C'est la méthode la plus lente et la plus facile à comprendre.
    """
    debut_chrono = time.perf_counter()
    
    points_dans_cercle = 0
    
    for _ in range(nombre_simulations):
        x = random.random()
        y = random.random()
        if x**2 + y**2 <= 1.0:
            points_dans_cercle += 1
            
    estimation_pi = 4 * points_dans_cercle / nombre_simulations
    
    fin_chrono = time.perf_counter()
    temps_cpu = fin_chrono - debut_chrono
    
    return estimation_pi, temps_cpu
