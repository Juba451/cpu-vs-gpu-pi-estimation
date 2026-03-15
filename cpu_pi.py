import random
import time

def estimer_pi_cpu(nombre_points):
    # on compte combien de points tombent dans le cercle
    compteur = 0
    debut = time.perf_counter()

    for i in range(nombre_points):
        x = random.random()
        y = random.random()
        # si le point est dans le cercle de rayon 1
        if x*x + y*y <= 1:
            compteur += 1

    resultat = 4 * compteur / nombre_points
    fin = time.perf_counter()
    temps = fin - debut

    return resultat, temps
