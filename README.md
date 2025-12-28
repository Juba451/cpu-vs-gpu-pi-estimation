# Mini-Projet : Comparaison de Performance CPU vs GPU (Version Simplifiée)

Ce projet est une démonstration directe de l'accélération fournie par un GPU (Graphics Processing Unit) par rapport à un CPU (Central Processing Unit) pour des tâches de calcul parallèle.

## Le Problème : Estimer Pi (π) avec la Méthode de Monte-Carlo

L'objectif est d'estimer la valeur de π en utilisant une méthode probabiliste qui simule un jeu de lancer de fléchettes virtuel. En lançant des millions de points aléatoires sur une cible, on peut estimer π avec la formule : `π ≈ 4 * (nombre de points dans le cercle / nombre total de points)`.

Chaque lancer étant un calcul **indépendant**, ce problème est parfaitement adapté au **calcul parallèle**.

## Structure du Projet

Le projet est organisé en trois fichiers simples :

-   `cpu_pi.py`: Contient la fonction pour le calcul **séquentiel sur CPU**, utilisant une simple boucle `for`. C'est notre référence lente.
-   `gpu_pi.py`: Contient la fonction pour le calcul **parallèle sur GPU** en utilisant la bibliothèque CuPy.
-   `main.py`: Le **script principal** qui exécute les deux méthodes et affiche la comparaison des performances.

## Les Deux Méthodes Comparées

1.  **Version CPU (Séquentielle) :** Un code simple qui simule le lancer de chaque fléchette **une par une**. Cette méthode est très lente.
2.  **Version GPU (Parallèle) :** La même logique, mais tous les calculs sont exécutés **en même temps** par les milliers de cœurs du GPU, offrant une accélération spectaculaire.

## Résultats Attendus

Le script affichera le temps de calcul pour chaque méthode et calculera le facteur d'accélération (`Speedup`), démontrant à quel point le GPU est plus rapide pour cette tâche.

## Comment l'exécuter sur Google Colab

1.  Ouvrez un [nouveau notebook sur Google Colab](https://colab.research.google.com).
2.  **Activez le GPU :** Allez dans `Runtime` -> `Change runtime type` et sélectionnez `GPU`.
3.  Dans une cellule de code, copiez-collez les commandes suivantes pour cloner le projet (en spécifiant votre nouvelle branche) et l'exécuter :

    ```python
    # Remplacez "VOTRE_NOM_UTILISATEUR" et "NOM_DE_LA_BRANCHE"
    GITHUB_USERNAME = "Juba451"
    REPO_NAME = "cpu-vs-gpu-pi-estimation"
    BRANCH_NAME = "testing-other-approaches"

    # 1. Cloner la branche spécifique du dépôt
    !git clone -b {BRANCH_NAME} https://github.com/{GITHUB_USERNAME}/{REPO_NAME}.git

    # 2. Se déplacer dans le dossier du projet
    %cd {REPO_NAME}

    # 3. Installer les dépendances et lancer le script principal
    !pip install -r requirements.txt && python main.py
    ```
4.  Exécutez la cellule. Les résultats s'afficheront directement dans la sortie.
