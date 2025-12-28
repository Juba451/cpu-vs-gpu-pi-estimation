### La Démonstration Mathématique (Solution Analytique)

Pour prouver que l'aire du quart de cercle est bien `π/4`, on peut la calculer de manière analytique en utilisant une intégrale. C'est la méthode "exacte", par opposition à la méthode "estimée" de Monte-Carlo.

**1. Définition du Domaine**

Le domaine `D` est le quart de cercle unité dans le premier quadrant, défini par :
$$
D = \{ (x, y) \in \mathbb{R}^2 \mid 0 \le x \le 1, \, 0 \le y \le \sqrt{1-x^2} \}
$$
L'arc de cercle suit l'équation :
$$
x^2+y^2=1 \implies y = \sqrt{1-x^2} \quad (\text{pour } y \ge 0)
$$

**2. Mise en place de l'Intégrale**

L'aire `A` de ce domaine peut être calculée avec une intégrale double :
$$
\begin{aligned}
A &= \iint_D 1 \, dA \\
&= \int_{0}^{1} \int_{0}^{\sqrt{1 - x^2}} 1 \, dy \, dx \\
&= \int_{0}^{1} \sqrt{1 - x^2} \, dx
\end{aligned}
$$

**3. Substitution Trigonométrique**

Cette intégrale est difficile à calculer directement. On effectue donc une substitution trigonométrique :
$$
x = \sin(\theta) \implies dx = \cos(\theta) \, d\theta
$$
Il faut aussi changer les bornes de l'intégration :
- Si `x = 0`, alors `sin(θ) = 0 \implies θ = 0`.
- Si `x = 1`, alors `sin(θ) = 1 \implies θ = π/2`.

En remplaçant `x` et `dx` dans l'intégrale, on obtient :
$$
A = \int_{0}^{\pi/2} \sqrt{1 - \sin^2(\theta)} \cdot \cos(\theta) \, d\theta
$$
Puisque `1 - sin²(θ) = cos²(θ)` et que `cos(θ) ≥ 0` sur l'intervalle `[0, π/2]`, l'intégrale se simplifie en :
$$
A = \int_{0}^{\pi/2} \cos^2(\theta) \, d\theta
$$

**4. Résolution de l'Intégrale**

Pour résoudre cette intégrale, on utilise l'identité de l'angle double :
$$
\cos^2(\theta) = \frac{1 + \cos(2\theta)}{2}
$$
Le calcul final devient :
$$
\begin{aligned}
A &= \int_{0}^{\pi/2} \frac{1 + \cos(2\theta)}{2} \, d\theta \\
&= \frac{1}{2} \int_{0}^{\pi/2} (1 + \cos(2\theta)) \, d\theta \\
&= \frac{1}{2} \left[ \theta + \frac{1}{2}\sin(2\theta) \right]_{0}^{\pi/2} \\
&= \frac{1}{2} \left( \left( \frac{\pi}{2} + \frac{1}{2}\sin(\pi) \right) - \left( 0 + \frac{1}{2}\sin(0) \right) \right) \\
&= \frac{1}{2} \left( \left( \frac{\pi}{2} + 0 \right) - ( 0 + 0 ) \right) \\
&= \frac{\pi}{4}
\end{aligned}
$$

Cette démonstration confirme la base théorique de notre projet : l'aire du quart de cercle unité est bien **π/4**.
