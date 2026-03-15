import time

def estimer_pi_gpu_optimise(cp, nombre_points, taille_lot):
    # même chose que la brute force mais on découpe en petits lots
    # pour pas exploser la mémoire du GPU

    debut = time.perf_counter()
    total_dans_cercle = 0
    nb_lots = nombre_points // taille_lot

    for i in range(nb_lots):
        points = cp.random.rand(taille_lot, 2)
        x = points[:, 0]
        y = points[:, 1]
        dist = x**2 + y**2
        total_dans_cercle += cp.sum(dist <= 1)

    pi = 4 * total_dans_cercle / nombre_points

    cp.cuda.runtime.deviceSynchronize()
    fin = time.perf_counter()

    return pi, fin - debut
