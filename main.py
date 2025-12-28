# main.py

from cpu_pi import estimer_pi_cpu
from gpu_pi import estimer_pi_gpu

def main():
    """
    Programme principal qui compare la performance du CPU vs GPU.
    """
    # Paramètre unique pour le nombre de simulations
    nombre_de_simulations = 10_000_000
    
    print("Lancement de la comparaison CPU vs GPU...")
    print(f"Nombre de simulations : {nombre_de_simulations:,}")
    
    # --- Calcul CPU ---
    pi_cpu, temps_cpu = estimer_pi_cpu(nombre_de_simulations)
    
    # --- Calcul GPU ---
    pi_gpu, temps_gpu = estimer_pi_gpu(nombre_de_simulations)

    # --- Conclusion ---
    print("\n" + "="*40)
    print("           TABLEAU DES RÉSULTATS")
    print("="*40)
    print(f"Méthode CPU : π ≈ {pi_cpu:.6f}  | Temps: {temps_cpu:.4f}s")
    
    if pi_gpu is not None:
        print(f"Méthode GPU : π ≈ {float(pi_gpu):.6f}  | Temps: {temps_gpu:.4f}s")
    else:
        print("Méthode GPU : Non exécutée (CuPy non disponible)")
    
    print("="*40)
    
    if temps_gpu > 0:
        acceleration = temps_cpu / temps_gpu
        print(f"✅ Le GPU est {acceleration:.0f} fois plus rapide que le CPU.")
        print("="*40)

if __name__ == "__main__":
    main()
