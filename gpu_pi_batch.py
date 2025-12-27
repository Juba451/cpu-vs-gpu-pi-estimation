# gpu_pi_batch.py

import time

try:
    import cupy as cp
except ImportError:
    pass

def estimer_pi_gpu_par_lots(total_simulations, taille_lot):
    """
    Estime Pi sur GPU avec une méthode par lots.
    Retourne l'estimation et le temps de calcul.
    """
   if not CUPY_DISPONIBLE:
        print("Avertissement : CuPy n'est pas disponible. Le calcul GPU par lots est ignoré.")
        return None, float('inf')

    # Si CuPy est disponible, on exécute le code normalement.
    print(f"\n--- 2. Calcul sur GPU 'par lots' sur {total_simulations:,} points ---")
    
        cp.cuda.runtime.deviceSynchronize()
        start_time = time.perf_counter()
        
        total_points_dedans = 0
        nombre_de_lots = total_simulations // taille_lot
        
        for _ in range(nombre_de_lots):
            points = cp.random.rand(taille_lot, 2)
            distances_carre = points[:, 0]**2 + points[:, 1]**2
            dedans_ce_lot = cp.sum(distances_carre <= 1)
            total_points_dedans += dedans_ce_lot
            
         estimation_pi_gpu = 4 * total_points_dedans / total_simulations
        
        cp.cuda.runtime.deviceSynchronize()
        end_time = time.perf_counter()
        gpu_time = end_time - start_time
        
        return estimation_pi_gpu, temps_gpu
        
    except NameError:
        return None, float('inf')
