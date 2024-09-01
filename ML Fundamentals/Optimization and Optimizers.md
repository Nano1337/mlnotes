
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
 **Newton's method solver:**
 $\delta = -H(x)^{-1}\nabla_{x}f(x)$
- But this isn't practical bc 1) we'd have to calculate inverse hessian at every step, 2) Hessian isn't always invertible and not full rank bc data lies on lower dimensional manifold

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

## First Order Methods: 
- Improve convergence by normalizing the mean of the derivatives
- Note for diagrams: 
	- we see oscillations bc gradient points in direction orthogonal to loss trace
### Momentum 
maintains a running average of all gradients until current step. The running average: 
- Gets longer in directions where the gradient retains the same sign 
- Becomes shorter in directions where the sign keeps flipping 
The momentum method: 
$$\Delta W^{(k)} = \beta \Delta W^{(k-1)} - \eta \nabla_{w}Loss(W^{(k-1)})^{T}$$
Where $\beta$ is the momentum factor, determine how much of the previous update is kept
![[img/Pasted image 20240306204607.png]]
Steps: 
- First compute gradient w.r.t. current location 
- Add in the scaled previous step (moving average)
	- This is why if the previous step delta was positive and the sign doesn't change, then the next iteration's delta will be even more positive and so on 
	- the other side applies where if the sign flips often then there is much less momentum 
Benefits: 
- Incremental SGD and mini-batch gradients tend to have high variance, so using momentum smooths out variations and allows faster convergence

### Nesterov's Accelerated Gradient
![[img/Pasted image 20240306210648.png]]
- Change the order of operations from momentum: 
	- Extend the previous step
	- Calculate gradient
- Converges much faster

## Second Order Methods: 
- Steps in large oscillatory (variance) motions result in large unnecessary distance being traveled. 
- We can dampen the step in directions with high motion, which is regularizing the second order term
- Scale updates in inverse proportion to total movement of that component in the past (according to variation). For example:
![[img/Pasted image 20240306211419.png]]
- Y component is scaled down while X component is scaled up

### RMS Prop
- Updates are by **parameter**
- Let $E\left[\partial^{2}_{w}D\right]$ be the mean squared derivative (MSD). Note that is **NOT** the second partial derivative, it is simply the partial derivative of the loss (D) w.r.t. the weights then squared. 
- We want to scale down updates with large MSD & scale up updates with small MSD. 
$$
\begin{align}
\mathbb{E}\left[\partial^{2}_{w}D\right]_{k} = \gamma \mathbb{E}\left[\partial^{2}_{w}D\right]_{k-1} + (1-\gamma) \left(\partial^{2}_{w}D\right)_{k} \\
w_{k+1}= w_{k}- \frac{1}{\sqrt{\mathbb{E}\left[\partial^{2}_{w}D\right]_{k}+ \epsilon}}\partial_{w}^{2}D
\end{align}$$
Steps: 
- Maintain running estimate MSD for each param 
- Scale update of each parameter by inverse of Root MSD (RMSD)

### ADAM 
- Considers both the first and second moments
$$
\begin{align}
m_{k}= \delta m_{k-1} + (1-\delta)(\partial_{w}D)_{k} \\
v_{k}= \delta v_{k-1}+ (1-\gamma )(\partial_{w}^{2}D)_{k} \\
\hat m_{k}= \frac{m_{k}}{1-\delta^{k}}, \hat v = \frac{v_{k}}{1-\gamma^{k}} \\ 
w_{k+1}= w_{k} - \frac{\eta}{\sqrt{\hat v_{k}+ \epsilon }}\hat m_{k}
\end{align}
$$
Steps: 
- Maintain running mean derivative for each parameter
- maintain running MSD for each parameter
- scale update of the param by the inverse of the RMSD 


### Non-Convex Optimization 
- There contain many local optima 

### Role of Overparameterization
![[img/Pasted image 20240306213656.png]]
- Mikhail Belkin says overparameterization allows for basins of global minima even if not convex locally

### Lipschitz Continuity for Step size
- Assume that the gradient of our loss function is Lipschitz continuous
	- TODO: add formula here
- Assume constant step size for now. 
	- TODO: include derivation here
- Eventually we get to $\alpha=1/L$, where L is our Lipshitz constant
	- TODO: add inequality 
- Equality is achieved when the norm of the gradient is 0, indicating convergence

**Optimality Gap**
- At some iteration $t$, what is the difference from the optimum functoin value? 
$$

gap_{t}= f(w_{t}) - F_{*}
$$
- Start by invoking convexity of loss, to get bound: 
	$$
	F(w_{t}) \leq F_{*} + g_{t}^T (w_{t}- w^{*})
$$
- then invoke Cauchy Schwarz inequality to get upper bound: 
	- TODO: add here

**Effects from Initialization**
- Gap depends on 1) norm of gradient. 2) how far away we are from global minimum (euclidean distance between current param vector and optimal one)
	- but we don't know how this varies as a function of init
	- intuitively, if our init is far away from optimal, then we need to take many more steps
- The bound of terms of losses gives that the euclidean distance is farthest away from optimum at init (which is obvious???)
	- TODO: add formula here
- We eventually find that: TODO: add it here
$$

gap_{t} \leq \frac{2L||w_{0}-w^{*}||}{t+1}
$$
- Practically speaking, should you set the step size to the inverse of the Lipshitz constant? 
	- NO, bc this is a pessimistic bound and must satisfy over all possible inputs (Supremum). This is same as finding the supremum of norm of Hessian, but this is intractable to compute for all inputs. 
		- TODO: add hessian formula here


#### Polyak-Łojasiewicz (PL) Condition
$$||\nabla \mathcal{L}(w) ||^{2} \geq \mu(\mathcal L(w) - \mathcal L(w^{*}))$$

- the $\mu$-$PL^{*}$ condition simplifies by assuming that $\mathcal L(w^{*})= 0$ . 
- PL* condition implies the existence of a solution and convergence of (S)GD to it

Assuming MSE: 
$$\mathcal L(w) = \frac{1}{2} ||F(w)-y||^{2}$$ where F(w) is the weight times the input data. Therefore, we have (D is derivative/Jacobian): 
$$\nabla_{w}\mathcal L(w) = (F(w)-y)^{T}DF(w)$$
and taking the squared norm and expanding we get: 
$$||\nabla \mathcal L(w)||^{2}=(F(w) -y)^{T}DF(w)DF^{T}(w)(F(w) -y)$$
From this, we can identify the presence of the tangent kernel:
$$K(w) = DF(w)DF^{T}(w)$$The tangent kernel captures how the output of the network changes to small changes in the weights. We can then lower bound the gradient norm with: 
$$||\nabla \mathcal L(w)||^{2} \geq \left[\lambda_{min}(K(w))\right]||F(w)-y||^{2}$$
- In this lower bound, we see that the second half of the term is simply our original $\mathcal L(w)$ and the first half is our $\mu = \lambda_{min}(K(w))$. 

**In Summary:**
If $\mathcal L(w)$: 
1. is $\beta$-smooth
$$\lambda_{max}(H(w)) \leq \beta$$
3. Satisfies the $\mu$- PL* condition in a Ball($w_{0}$, R) with: 
$$R = \frac{2\sqrt{2\beta \mathcal L(w_0)}}{\mu}$$
then (S)GD, initialized at $w_0$ with learning rate $\eta \leq \frac{2}{\beta}$ will recover $w^{*}$ such that F(w*) = y, which means there's a solution on the solution manifold. 
- Rate of convergence: O($\beta$ / t)
- Norm of gradient below some $\delta$ threshold then we need O($\beta$ / $\delta$) iterations. 





