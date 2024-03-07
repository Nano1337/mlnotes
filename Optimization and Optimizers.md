
### Gradient Descent Update Formula: 

Assume we define $MSE = \frac{1}{2}||\Phi^{T}w - y||^2$, then the gradient can be derived to be: 
$$g^{(t)} = \Phi^T\Phi w^{(t)} - \Phi^Ty$$
Then the gradient update step would be: 
$$w^{(t+1)} = w^{(t)} - \epsilon \nabla_{w}(\Phi^{T}\Phi w^{(t)} -\Phi^{T}y)$$

### Theoretical Upper Bound of $\epsilon$ 

The Hessian is defined to be: 
$$\left[H^{(t)}\right]_{ij} = \frac{\partial \mathcal{L}}{\partial w_{i}\partial w_{j}}(w^{(t)})$$
The Hessian is not a function of the weights since the second derivative gets rid of all weight terms (and thus time t). It also measures the curvature of the loss landscape, which makes it a function of the data: 
$$H = \Phi^{T}(x)\Phi(x)$$

Using the 2nd order Taylor approximation: 
$$f(x) \approx f(x_0) + f'(x_0)(x-x_0) + \frac{1}{2}f''(x_0)(x-x_0)^2$$
We can apply this to approximate $\mathcal{L}(w)$: 
$$    \mathcal{L}(w) \approx \mathcal{L}(w^{t}) + (w-w^{t})^T g^{(t)} + \frac{1}{2} (w-w^{t})^T H^{(t)} (w-w^{t})$$

Substituting in $w^{(t+1)} =  (w^{(t)} - \epsilon g^{(t)})$: 
$${L}(w^{t+1}) = {L}(w^{t}) - \epsilon (g^{(t)})^T g^{(t)} + \frac{1}{2}\epsilon^2 (g^{(t)})^T H^{(t)} g^{(t)} $$
Solving for $\epsilon$ we get: 
$$    \epsilon \leq \frac{2 (g^{(t)})^T g^{(t)} }{(g^{(t)})^T H^{(t)} g^{(t)} }$$
**Beta-Smoothness:**
Now assume $v^T H v \leq \beta ||v||^2$, $\forall v$. This can be re-arranged as follows: 
$$    \frac{v^T H v}{||v||^2} \leq \beta$$
$$    \frac{v^T H v}{v^T v} \leq \beta $$
$$    \frac{1}{\beta} \leq \frac{v^T v}{v^T H v} $$
$$    \frac{2}{\beta} \leq \frac{2v^T v}{v^T H v}$$
$$    \frac{2}{\beta} \leq \frac{2 (g^{(t)})^T g^{(t)}}{(g^{(t)})^T H^{(t)} g^{(t)}}$$

Since $\epsilon \leq \frac{2 (g^{(t)})^T g^{(t)} }{(g^{(t)})^T H^{(t)} g^{(t)} }$, then $\epsilon \leq \frac{2}{\beta}$ where $\beta = \lambda_{max}(H)$

### Theoretical Bounds of Convergence Rate: 
- Gradient norm goes to 0 at a rate of 1/t
- Alternatively the number of iterations we need (t) is given by the formula: 
$$t \geq \frac{2\beta \mathcal{L} (w^{(0)} - \mathcal{L}(w^{*})) }{\delta}$$

### Basin Sizes: 

From *Jastrzebski et al (2017)*: SGD will converge to minimal with a basin width:
$$w = \frac{\epsilon}{B}$$ where $\epsilon$ is the learning rate and B is the batch size. This means that full Batch Descent prefers small width (sharp basins) while larger learning rate and smaller batch size (more stochastic) will prefer wider basins. From other papers, it has be theorized that wider basins allow for better generalization.

**What are other ways to find wider basins?**
- We want larger Hessians for curvature but minimize curvature for flat and wide basins. 
- We can find the largest spectral radius (largest eigenvalue) and add that to the loss to optimize, which gives us the SAM optimizer 

### Momentum 
