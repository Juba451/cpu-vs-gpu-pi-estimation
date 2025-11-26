# main.py

# On importe les fonctions spécifiques de nos autres fichiers
from cpu_pi import estimate_pi_cpu
from gpu_pi_batch import estimate_pi_gpu_batch
from gpu_pi_optimized import estimate_pi_gpu_optimized

def main():
    """
    Programme principal qui exécute et compare les différentes méthodes
    de calcul de Pi, et qui gère l'affichage des résultats.
    """
    # Paramètres
    n_points_cpu = 100_000_000 # On peut augmenter car NumPy est rapide
    n_points_gpu = 100_000_000
    batch_size_gpu = 10_000_000
    
    # --- Exécution ---
    print(f"--- 1. Calcul sur CPU (NumPy) sur {n_points_cpu:,} points ---")
    pi_cpu, cpu_time = estimate_pi_cpu(n_points_cpu)

    print(f"\n--- 2. Calcul sur GPU 'par lots' sur {n_points_gpu:,} points ---")
    pi_gpu_batch, gpu_batch_time = estimate_pi_gpu_batch(n_points_gpu, batch_size_gpu) 
    
    print(f"\n--- 3. Calcul sur GPU Optimisé (Kernel) sur {n_points_gpu:,} points ---")
    pi_gpu_opt, gpu_opt_time = estimate_pi_gpu_optimized(n_points_gpu)

    # --- Conclusion ---
    print("\n" + "="*40)
    print("           TABLEAU DES RÉSULTATS")
    print("="*40)
    print(f"Méthode CPU (NumPy)    : π ≈ {pi_cpu:.6f}  | Temps: {cpu_time:.4f}s")
    
    if pi_gpu_batch is not None:
        print(f"Méthode GPU (par lots)   : π ≈ {float(pi_gpu_batch):.6f}  | Temps: {gpu_batch_time:.4f}s")
    else:
        print("Méthode GPU (par lots)   : Non exécutée (CuPy non disponible)")
        
    if pi_gpu_opt is not None:
        print(f"Méthode GPU (optimisé)   : π ≈ {float(pi_gpu_opt):.6f}  | Temps: {gpu_opt_time:.4f}s")
    else:
        print("Méthode GPU (optimisé)   : Non exécutée (CuPy non disponible)")
    
    print("="*40)
    
    if gpu_batch_time != float('inf'):
        speedup_batch = cpu_time / gpu_batch_time
        print(f"✅ Le GPU (par lots) est {speedup_batch:.0f} fois plus rapide que le CPU.")
    
    if gpu_opt_time != float('inf'):
        speedup_opt = cpu_time / gpu_opt_time
        print(f"✅ Le GPU (Optimisé) est {speedup_opt:.0f} fois plus rapide que le CPU.")
    print("="*40)

if __name__ == "__main__":
    main()
