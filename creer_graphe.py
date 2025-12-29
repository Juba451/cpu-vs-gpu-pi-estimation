import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 100)

y = np.sqrt(1 - x**2)

# Créer le graphe
plt.figure(figsize=(6, 6))

# dessiner la courbe du quart de cercle
plt.plot(x, y, color='blue', linewidth=2)


plt.fill_between(x, y, color='skyblue', alpha=0.4)


plt.title("Aire sous la courbe y = sqrt(1 - x²)")
plt.xlabel("Axe des x")
plt.ylabel("Axe des y")


plt.axis('equal')
plt.xlim(0, 1.1)
plt.ylim(0, 1.1)

# Ajouter une grille
plt.grid(True, linestyle='--', alpha=0.6)

# Ajouter le label "A" pour l'aire
plt.text(0.5, 0.3, 'A', fontsize=20, horizontalalignment='center')

# 5. Sauvegarder le graphe dans un fichier image
plt.savefig('images/aire_integrale.png')

print("Graphe 'aire_integrale.png' créé avec succès dans le dossier 'images' !")