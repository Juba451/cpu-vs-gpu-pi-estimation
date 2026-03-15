import time
from cpu_pi import estimer_pi_cpu
from gpu_pi_bruteforce import estimer_pi_gpu_bruteforce
from gpu_pi_optimise import estimer_pi_gpu_optimise

# on vérifie si un GPU est disponible avant de lancer les calculs
gpu_dispo = False
cp = None

try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
    gpu_dispo = True
except:
    print("Pas de GPU détecté, on va juste faire le calcul CPU.\n")

# --- paramètres ---
N = 100_000_000    # nombre de points total
taille_lot = 10_000_000  # pour la version par lots

print("Début de la comparaison...\n")

# calcul CPU
pi_cpu, t_cpu = estimer_pi_cpu(N)
print(f"CPU  -> pi ≈ {float(pi_cpu):.6f}  ({t_cpu:.4f}s)")

# calculs GPU si disponible
if gpu_dispo:
    pi_brute, t_brute = estimer_pi_gpu_bruteforce(cp, N)
    print(f"GPU brute force -> pi ≈ {float(pi_brute):.6f}  ({t_brute:.4f}s)")

    pi_opt, t_opt = estimer_pi_gpu_optimise(cp, N, taille_lot)
    print(f"GPU optimisé    -> pi ≈ {float(pi_opt):.6f}  ({t_opt:.4f}s)")

# affichage des résultats
print("\n" + "="*45)
print("            RÉSULTATS")
print("="*45)
print(f"  CPU          : {t_cpu:.4f}s")

if gpu_dispo:
    print(f"  GPU brute    : {t_brute:.4f}s")
    print(f"  GPU optimisé : {t_opt:.4f}s")

print("="*45)

if gpu_dispo:
    # combien de fois le GPU est plus rapide ?
    ratio_brute = t_cpu / t_brute
    ratio_opt   = t_cpu / t_opt
    print(f"Le GPU brute force est {ratio_brute:.0f}x plus rapide que le CPU")
    print(f"Le GPU optimisé est {ratio_opt:.0f}x plus rapide que le CPU")