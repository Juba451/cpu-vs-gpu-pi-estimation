# Mini-Projet : Comparaison de Performance CPU vs GPU

Ce projet est une démonstration simple mais puissante de l'accélération fournie par un GPU (Graphics Processing Unit) par rapport à un CPU (Central Processing Unit) pour des tâches de calcul parallèle.

## Le Problème : Estimer Pi (π) avec la Méthode de Monte-Carlo

L'objectif est d'estimer la valeur de π sans la calculer directement, en utilisant une méthode probabiliste qui simule un jeu de lancer de fléchettes.

### La Théorie

On se base sur le rapport entre l'aire d'un quart de cercle de rayon 1 et l'aire d'un carré de côté 1. En lançant des milliers de "fléchettes" (points aléatoires) sur cette cible, on peut estimer π avec la formule : `π ≈ 4 * (nombre de points dans le cercle / nombre total de points)`.


Chaque lancer étant un calcul **indépendant**, ce problème est parfaitement adapté au **calcul parallèle** sur GPU.

## Structure du Projet

Le projet est organisé en plusieurs modules Python pour plus de clarté :

-   `cpu_pi.py`: Contient la fonction pour le calcul **séquentiel sur CPU**, utilisant une simple boucle `for`. C'est notre référence lente.
-   `gpu_pi_batch.py`: Contient la fonction pour le calcul **parallèle sur GPU** en utilisant la bibliothèque CuPy, avec une **optimisation par lots (batching)**.
-   `gpu_pi_optimized.py`: Contient la fonction **optimisée pour GPU** qui utilise un kernel CUDA personnalisé pour une performance maximale.
-   `main.py`: Le **programme principal** qui importe les fonctions des autres modules, les exécute dans l'ordre et affiche la comparaison finale des performances.
-   `requirements.txt`: Liste les bibliothèques nécessaires pour le projet.

## Les Trois Méthodes Comparées

1.  **Version CPU (Séquentielle) :** Un code simple qui simule le lancer de chaque fléchette **une par une**. C'est la méthode la plus lente.
2.  **Version GPU "par Lots" (Parallèle) :** Cette version utilise CuPy pour traiter les points sur le GPU. Pour optimiser l'utilisation de la mémoire, elle fonctionne **par "lots"** : elle traite des millions de points en parallèle, répète l'opération plusieurs fois et additionne les résultats.
3.  **Version GPU Optimisée (Kernel CUDA) :** Une version avancée où un **mini-programme (kernel)** est envoyé directement au GPU pour organiser le travail de la manière la plus efficace possible.

## Résultats Obtenus

--- 1. Calcul sur CPU avec une boucle for sur 100,000,000 points ---
Estimation de π (CPU) ≈ 3.141406
Temps de calcul (CPU) : 29.6060 secondes

--- 2. Calcul sur GPU 'par lots' sur 100,000,000 points ---
Estimation de π (GPU par lots) ≈ 3.141609
Temps de calcul (GPU par lots) : 0.4219 secondes

--- 3. Calcul sur GPU Optimisé avec un Kernel CUDA sur 100,000,000 points ---
Estimation de π (GPU Optimisé) ≈ 3.141913
Temps de calcul (GPU Optimisé) : 0.0166 secondes

==============================
     TABLEAU DES RÉSULTATS
==============================
Temps CPU (boucle)           : 29.6060s
Temps GPU (par lots)         : 0.4219s
Temps GPU (optimisé)         : 0.0166s
==============================
 Le GPU (par lots) est 70 fois plus rapide que le CPU.
 Le GPU (Optimisé) est 1783 fois plus rapide que le CPU.
==============================




## Comment l'exécuter

Ce projet est conçu pour être exécuté dans un environnement disposant d'un GPU Nvidia et de CUDA.

### Méthode 1 : En local (si vous avez un GPU Nvidia)

1.  Clonez ce dépôt sur votre machine :
    ```bash
    git clone https://github.com/Juba451/cpu-vs-gpu-pi-estimation.git
    cd cpu-vs-gpu-pi-estimation
    ```
2.  Installez les dépendances nécessaires :
    ```bash
    pip install -r requirements.txt
    ```
3.  Lancez le script principal :
    ```bash
    python main.py
    ```

### Méthode 2 : Sur Google Colab (recommandé et gratuit)

Cette méthode vous permet de faire tourner le projet directement depuis GitHub.

1.  Ouvrez un [nouveau notebook sur Google Colab](https://colab.research.google.com).
2.  **Activez le GPU :** Allez dans `Runtime` -> `Change runtime type` et sélectionnez `GPU` comme "Hardware accelerator".
3.  Dans une seule cellule de code, copiez-collez les commandes suivantes pour cloner le projet et l'exécuter :

    ```python
    # 1. Cloner le dépôt GitHub
    !git clone https://github.com/Juba451/cpu-vs-gpu-pi-estimation.git

    # 2. Se déplacer dans le dossier du projet
    %cd cpu-vs-gpu-pi-estimation

    # 3. Installer les dépendances et lancer le script principal
    !pip install -r requirements.txt && python main.py
    ```
4.  Exécutez la cellule. Les résultats s'afficheront directement dans la sortie.
