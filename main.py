# On importe les fonctions spécifiques de nos autres fichiers
from cpu_pi import estimer_pi_cpu
from gpu_pi_batch import estimer_pi_gpu_par_lots
from gpu_pi_optimized import estimer_pi_gpu_optimise

def main():
    """
    Programme principal qui exécute et compare les différentes méthodes
    de calcul de Pi, et qui gère l'affichage des résultats.
    """
    # Paramètres
    nombre_points = 100_000_000 # représente le nombre total de "fléchettes" virtuelles que nous allons lancer pour estimer Pi.
    taille_du_lot = 10_000_000 # un paramètre technique pour la méthode GPU par lots. Il définit combien de points sont traités en parallèle à chaque étape.

    
    #Exécution
    print("Lancement des calculs...")
    
    pi_cpu, temps_cpu = estimer_pi_cpu(nombre_points)
    pi_gpu_batch, temps_gpu_batch = estimer_pi_gpu_par_lots(nombre_points, taille_du_lot) 
    pi_gpu_opt, temps_gpu_opt = estimer_pi_gpu_optimise(nombre_points)

    # ====================================================================
    # AFFICHAGE DES RÉSULTATS
    # ====================================================================
    print("\n" + "="*50)
    print("           TABLEAU FINAL DES RÉSULTATS")
    print("="*50)
    print(f"Nombre de simulations : {nombre_points:,}")
    print("-" * 50)
    
    #résultats du CPU
    print(f"Méthode CPU       : π ≈ {pi_cpu:.6f}  | Temps: {temps_cpu:.4f}s")
    
    #résultats du GPU (par lots)
    if pi_gpu_batch is not None:
        print(f"Méthode GPU (lots)  : π ≈ {float(pi_gpu_batch):.6f}  | Temps: {temps_gpu_batch:.4f}s")
    else:
        print("Méthode GPU (lots)  : Non exécutée (CuPy non disponible)")
        
    #résultats du GPU (optimisé)
    if pi_gpu_opt is not None:
        print(f"Méthode GPU (optim) : π ≈ {float(pi_gpu_opt):.6f}  | Temps: {temps_gpu_opt:.4f}s")
    else:
        print("Méthode GPU (optim) : Non exécutée (CuPy non disponible)")
    
    print("="*50)
    
    # ====================================================================
    # CALCUL DES ACCÉLÉRATIONS
    # ====================================================================
    
    if temps_gpu_batch > 0:
        acceleration_batch = temps_cpu / temps_gpu_batch
        print(f"✅ Le GPU (par lots) est {acceleration_batch:.0f} fois plus rapide que le CPU.")
    
    if temps_gpu_opt > 0:
        acceleration_opt = temps_cpu / temps_gpu_opt
        print(f"✅ Le GPU (Optimisé) est {acceleration_opt:.0f} fois plus rapide que le CPU.")
    
    print("="*50)

if __name__ == "__main__":
    main()
