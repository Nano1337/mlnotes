Also called Evasion Attacks, is the act of generating adversarial examples

### General Methods: 
- White-Box: architecture and weights are known
- Black-Box: only access to input and output (e.g. API)
- Gray-Box: architecture known but not weights

We can attain an adversarial generally by adding some perturbation that maximizes the loss: 
$$max_{\delta} \text{ } loss(\theta, x + \delta, y) $$
We want $\delta$ that's small w.r.t.: 
- $l_p$ norm
- Rotation/Translation
- VGG feature perturbation or any other perturbation

### White Box Attack Methods

**Fast Gradient Sign Method (FGSM) Attack**: 
- Use pretrained classifier like ResNet50: $\tilde y = f(\theta, x)$
- Find adversarial example that maximizes the loss: $\mathcal{L} (x', y) = \mathcal{L}(f(\theta, x'), y)$ 
- Bounded perturbation s.t.: $||x'-x||_{\infty}\leq \epsilon$ , where $\epsilon$ is attack strength
Optimal adversarial image: $x' = x + \epsilon sign(\nabla_{x}\mathcal{L}(x, y))$ 

**Iterative FGSM Attack**: 
- Let m be number of iterations 
- $x^{(m)} = x^{(m-1)} + \epsilon sign(\nabla_{x}\mathcal{L}(x^{(m-1)}, y))$ 
Both (I)FGSM are fix-perturbation attacks

**(Iterative) Least Likely Attack**: 
- Similar to FSGM but $y_{LL}$ is the least likely (LL) class predicted by the network on clean image x
- $x' = x - \epsilon sign(\nabla_{x}\mathcal{L}(x, y_{LL}))$
- Strong attack as it emphasizes the least likely class

**Projected Gradient Descent Attack** : 
- We can take a gradient step and then project it back to the feasible set $\Delta$ since the perturbed input may not lie on the data manifold (similar to denoising AutoEncoder logic)
- $\delta := \mathcal{P}_{\Delta}[\delta + \nabla_{\delta}Loss(x + \delta), y; \theta]$ 
- This can be shown geometrically as: 
![[img/Pasted image 20240408084101.png]]
- For example, the projected gradient descent applied to $l_\infty$ ball, repeat: 
	- $\delta := Clip_{\epsilon}[\delta + \alpha \nabla_{\delta}J(\delta)]$ 
- Slower than FGSM but typically able to find better optima 

**Carlini and Wagner (CW) L2 Attack**: 
- zero-confidence attack 
- for all $t \neq y$ , find the adversarial image that will be classified by $t$ as solving the problem: $min_{\delta}||\delta||_{2}^{2}$ subject to $f(x + \delta) = y, x + \delta \in [0, 1]^{n}$
- Finding the exact solution is difficult so we use relaxed version
- $min_{\delta}||\delta||_{2}^{2} + c \cdot g(x + \delta)$  subject to $x + \delta \in [0, 1]^{n}, c \geq 0$ 
- Let $Z(x)$ be the NN activations before the logit output layer, also called the embeddings
	- $g(x) = max\left(max_{i\neq t} (Z(x)_{i} - Z(x)_{t}), 0\right)$
- Let $\delta = \frac{1}{2}(tanh(w) + 1) - x$
- With the following constrained optimization problem: 
$$
\begin{align}
min_w||\frac{1}{2}(tanh(w)+1)-x||_{2}^{2} + c \\
ReLU\{max_{i\neq t}Z\left(\left(\frac{1}{2}tanh(w) + 1\right)_{i}\right)- Z\left(\frac{1}{2}\left(tanh(w) + 1\right)\right)_{t}\}
\end{align}
$$
- Powerful attack method that resists many defenses

**Universal Adversarial Perturbation Attacks**: 
- A single perturbation on any image with high probability (e.g. 0.8+)
- Generalize well across different models

**Single Pixel Attack**:
- self-explanatory. Only modify one pixel

### Poisoning Attacks
- manipulating the training data itself rather than during inference
- Maintain accuracy but hamper generalization due to outliers through poisoning