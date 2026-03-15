import time

def estimer_pi_gpu_bruteforce(cp, nombre_points):
    # on génère tous les points d'un seul coup sur le GPU (version simple)
    debut = time.perf_counter()

    # deux colonnes : x et y pour chaque point
    points = cp.random.rand(nombre_points, 2)

    x = points[:, 0]
    y = points[:, 1]
    dist = x**2 + y**2

    # on compte les points qui sont dans le cercle unité
    dans_cercle = cp.sum(dist <= 1)

    pi = 4 * dans_cercle / nombre_points

    # on attend que le GPU finisse avant de mesurer le temps
    cp.cuda.runtime.deviceSynchronize()
    fin = time.perf_counter()

    return pi, fin - debut