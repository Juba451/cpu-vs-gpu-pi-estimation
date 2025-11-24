# main.py

from cpu_pi import estimate_pi_cpu_loop
from gpu_pi_batch import estimate_pi_gpu_batch
from gpu_pi_optimized import estimate_pi_gpu_optimized

def main():
    """
    Programme principal qui exécute et compare les différentes méthodes
    de calcul de Pi.
    """
    # Paramètres
    n_points_cpu = 100_000_000
    n_points_gpu = 100_000_000
    batch_size_gpu = 10_000_000 # On définit la taille du lot ici
    
    # Exécution
    cpu_time = estimate_pi_cpu_loop(n_points_cpu)
    # On appelle la nouvelle fonction avec les bons arguments
    gpu_batch_time = estimate_pi_gpu_batch(n_points_gpu, batch_size_gpu) 
    gpu_opt_time = estimate_pi_gpu_optimized(n_points_gpu)

    # Conclusion
    print("\n" + "="*30)
    print("     TABLEAU DES RÉSULTATS")
    print("="*30)
    print(f"Temps CPU (boucle)           : {cpu_time:.4f}s")
    print(f"Temps GPU (par lots)         : {gpu_batch_time:.4f}s") 
    print(f"Temps GPU (optimisé)         : {gpu_opt_time:.4f}s")
    print("="*30)
    
    if gpu_batch_time != float('inf') and gpu_opt_time != float('inf'):
        # On compare toujours par rapport au CPU
        speedup_batch = cpu_time / gpu_batch_time
        speedup_opt = cpu_time / gpu_opt_time
        
        print(f"✅ Le GPU (par lots) est {speedup_batch:.0f} fois plus rapide que le CPU.")
        print(f"✅ Le GPU (Optimisé) est {speedup_opt:.0f} fois plus rapide que le CPU.")
        print("="*30)

if __name__ == "__main__":
    main()