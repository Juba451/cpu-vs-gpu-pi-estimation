$$
D ={{ \{ (x, y) \in \mathbb{R}^2 \mid 0 \le x \le 1, 0 \le y \le \sqrt{1-x^2} \}}}
$$

$$
x^2+y^2=1 \Rightarrow y = \sqrt{1-x^2} \quad (y \ge 0)
$$

Calculer l'aire A:

$$
\begin{aligned}
& A=\iint_D d y d x=\int_0^1 \int_0^{\sqrt{1-x^2}} d y d x \\
& A=\int_0^1 \sqrt{1-x^2} d x
\end{aligned}
$$

on fait une substitution trigonometrique

$$
\begin{aligned}
& \Rightarrow \sin \theta=\frac{x}{1} \\
& \frac{d \sin  \theta}{d \theta}=\frac{d x}{d \theta} \\
& \{d x}=\cos \theta d \theta
\end{aligned}
$$

$$
\begin{aligned}
& x \in[0 ; 1] \Leftrightarrow \theta \in\left[0 ; \frac{\pi}{2}\right] \\
& \int_0^1 \sqrt{1-x^2} d x=\int_0^{\frac{\pi}{2}} \underbrace{(\sqrt{1-\sin ^2 \theta}=\cos \theta)} d \theta \\
& {car}  {} {}    \cos \theta >= 0 {}{}{sur}\left[0 ; \frac{\pi}{2}\right] {donc}{} A=\int_0^{\frac{\pi}{2}}\cos ^2 \theta d \theta
\end{aligned}
$$

On utilise l'identité d'angle double:

$$
\begin{gathered}
\cos (2 \theta)=2 \cos (\theta)-1 \\
\cos ^2(\theta)=\frac{\cos (2 \theta)+1}{2} \\
\int_0^{\frac{\pi}{2}} \cos ^2 \theta d \theta=\int_0^{\frac{\pi}{2}} \frac{\cos (2 \theta)+1}{2} d \theta \\
A=\frac{1}{2} \int_0^{\frac{\pi}{2}} \cos (2 \theta) d \theta+\frac{1}{2} \int_0^{\frac{\pi}{2}} 1 d \theta \\
A=\frac{1}{2}[\sin (2 \theta)]_0^{\frac{\pi}{2}}+\frac{1}{2}[\theta]_0^{\frac{\pi}{2}} \\
A=\frac{1}{2}[\sin (\pi)-\sin (0)]+\frac{1}{2}\left[\frac{\pi}{2}\right] \\
A=\frac{1}{2} \times \frac{\pi}{2}=\frac{\pi}{4}
\end{gathered}
$$

