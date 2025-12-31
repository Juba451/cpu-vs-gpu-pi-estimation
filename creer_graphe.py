import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 100)
y = np.sqrt(1 - x**2)

plt.figure(figsize=(8, 8))
plt.plot(x, y, color='blue', linewidth=2.5)
plt.fill_between(x, y, color='skyblue', alpha=0.4)

plt.title("Aire sous la courbe y = √(1 - x²)", fontsize=16) 


plt.axis('equal')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.6)

plt.text(0.5, 0.3, 'A', fontsize=24, horizontalalignment='center')


plt.xticks(np.arange(0, 1.1, 0.2))
plt.yticks(np.arange(0, 1.1, 0.2))

plt.savefig('images/aire_integrale.png', bbox_inches='tight')
print("Graphe 'aire_integrale.png' créé avec succès.")