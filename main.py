# main.py
from cpu_pi import estimer_pi_cpu
from gpu_pi_bruteforce import estimer_pi_gpu_bruteforce
from gpu_pi_optimise import estimer_pi_gpu_optimise

# --- Détection GPU ---
try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
    GPU_DISPONIBLE = True
except Exception:
    cp = None
    GPU_DISPONIBLE = False
    print("  Aucun GPU détecté. Les méthodes GPU seront ignorées.\n")

def main():
    nombre_points = 10_000_000 # Nombre de fléchettes à simuler
    taille_lot = 1_000_000   # Taille des lots pour la version optimisée

    print("Lancement de la comparaison...\n")

    # --- Calculs ---
    pi_cpu, temps_cpu = estimer_pi_cpu(nombre_points)

    if GPU_DISPONIBLE:
        pi_gpu_brute, temps_gpu_brute = estimer_pi_gpu_bruteforce(cp, nombre_points)
        pi_gpu_opt, temps_gpu_opt = estimer_pi_gpu_optimise(cp, nombre_points, taille_lot)
    else:
        pi_gpu_brute, temps_gpu_brute = None, float('inf')
        pi_gpu_opt, temps_gpu_opt = None, float('inf')

    # --- Affichage ---
    print("\n" + "="*50)
    print("           TABLEAU FINAL DES RÉSULTATS")
    print("="*50)
    print(f"Nombre de points simulés : {nombre_points:,}")
    print("-" * 50)
    print(f"Méthode CPU (lent)        : Temps: {temps_cpu:.4f}s")
    if temps_gpu_brute != float('inf'):
        print(f"Méthode GPU (brute force) : Temps: {temps_gpu_brute:.4f}s")
    if temps_gpu_opt != float('inf'):
        print(f"Méthode GPU (optimisé)    : Temps: {temps_gpu_opt:.4f}s")
    print("="*50)

    if temps_gpu_brute != float('inf') and temps_gpu_brute > 0:
        print(f"✅ Le GPU (brute force) est {temps_cpu / temps_gpu_brute:.0f} fois plus rapide que le CPU.")
    if temps_gpu_opt != float('inf') and temps_gpu_opt > 0:
        print(f"✅ Le GPU (optimisé) est {temps_cpu / temps_gpu_opt:.0f} fois plus rapide que le CPU.")
    print("="*50)

if __name__ == "__main__":
    main()