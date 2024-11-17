
- Posterior is not always exactly tractable (bc normalization constants) so need approximation to th eposterior
- Addresses problems of: 
	- lack of conjugacy
	- scalability (e.g. GP don't scale in num samples)
- Summary: 
	- Laplace: simple but poor approx
	- Variational: complex but scalable, underlies generative models
	- MCMC: simple, poor scalability but good approx

### Laplace Approx

Consider posterior: 
$$
p(w|D)=\frac{1}{Z}e^{-E(w)}
$$
where Z = p(D) and E(w) = -log p(w,D)

Performing the multivariate Taylor series expansion in E (around the MAP soln): 
$$
E(w) \approx E(w_{MAP}) + \nabla_{w}PNLL(w_{MAP})(w-w_{MAP}) + 0.5(w-w_{MAP})^{T H(w_{MAP})}(w-w_{MAP})
$$
by definition the grad of the PNLL is zero at MAP and first term is constant so the negative log-joint is approx by quadratic function. Plugging back into posterior, we attain Gaussian approx: 
$$
p(w|D) \approx N(w|w_{MAP}, H^{-1}(w_{MAP}))
$$
- Not good, not bc of Gaussian assumption (bc VI uses this too) but rather assumption that mean vector is by definition the MAP estimate


How do we get the posterior predictive distribution? 
- First marginalize out the weight vectors:
  $$
  p(y_{*}|x_{*},D) = \int_{w}p(y_{*}, w|x_{*}, D)dw = \int_{w}p(y_{*}|x_{*}, w)p(w|D)dw
  $$
- Since we don't know the posterior $p(w|D)$ exactly, we can use the Laplace approx of the posterior to get $\int_{w}p(y_{*}|x_{*}, w)q(w|D)$. To approximate this integral, we can use MC integration

### Monte Carlo (MC) Integration
1. Draw samples from the posterior: 
   $$
   w_{s} \sim N(w_{MAP}, H^{-1}(w_{MAP}))
   $$
2. Average sigmoid transformed values: 
   $$
   p(y_{*}=1|x_{*}, D) \approx \sum\limits_{s=1}^{S}\sigma(w_{s}^{T}x)
   $$
   Intuition: 
   - Don't uniformly random sample bc that's lots of wasted samples of low density and doesn't contribute much to integral
   - Instead, draw samples in proportion to posterior density

More analysis: 
- MC is an unbiased estimator
- By the CLT, the more samples you draw, the smaller variance

Problem: 
- as you increase feature dim D, sampling from infinite dim Gaussian (Normal dist) has most density at shell, so approx won't be close to true val anymore 

### Variational Inference

KL-Divergence: 
- distributional distance measure $D(q(z), p(z|x))$ where $q(z)$ is the actual posterior 
- $$D_{KL}(q||p) = \sum\limits q(z_{n})\log \frac{q(z_{n})}{p(z_{n})}$$
- Property 1: assuming q and p share same support, then q = p iif KLD = 0
	- ratio in log is always 1 so log always 0
	- in other words: if we find KLD=0 then q=p

Jensen's Inequality: 
- Given convex function f and probability function p, then inequality states: 
  $$
  f(\mathbb{E}_{p}[X]) \leq \mathbb{E}_{p}[f(X)]
  $$
- Can be understood as generalization of convexity
- used to construct lower bounds on proba distributions - used to maximize lower bound
- ![[Pasted image 20240930220856.png]]
  ![[Pasted image 20240930220904.png]]
  - In this case, log is the function and the inside summation is the E(X)

More KLD properties: 
- KLD is nonnegative (per Jensen's inequality)
- KLD is asymmetric (fixed by JSD by just doing KLD twice)

Mode dropping/collapse: 
- If q is underparameterized, mode dropping/collapse can occur

### ELBO: 
- variational parameters $\psi$ to parameterize variational distirbution $q_\psi$ 
- Goal: 
	- find variational parameters that min KLD between variational and posterior distribution: 
	  $$
	  \min_{\psi} D_{KL}(q_{\psi}(z) || p(z|x))
	  $$
	- where z corresponds to model parameters (note: different from variational params) 
- Bc we don't know the posterior p(z|x) due to the intractable marginal likelihood (normalizing constant), we'll need to alter this objective to be tractable:
  ![[Pasted image 20240930223925.png]]
  ![[Pasted image 20240930230322.png]]
  ![[Pasted image 20240930230336.png]]
  ![[Pasted image 20240930230347.png]]
  ![[Pasted image 20240930230717.png]]
  ![[Pasted image 20240930230707.png]]
  

### Flavors of VI: 
![[Pasted image 20240930230911.png]]
- Mean-field: specify how distribution factorizes across variables, optimal variational distribution derived from porbability specification (likelihood + prior)
- Gradient-based VI: choose form of variational dist then optimize ELBO via gradient-based methods
- Autodiff VI: similar to above but variational dist fixed to Gaussian (regardless of proba specification)
  
Reparameterization Trick: 
- Problem: to find the gradient of the ELBO, we have to compute gradients on the variational distribution, which is intractable (bc is not an expectation)
- trick: 
	- draw RV from simple (e.g. uniform) distribution that doesn't depend on variational params
	- Apply transformation that's differentiable and invertible and depends on variational params to give RV in target distribution

### Automatic Differentiation VI (ADVI)
- Sample from unconstrained variational distribution
- Transform to constrained space when evaluating likelihood and prior (log-joint)
- Entropy calculated on unconstrained variational distribution - closed form expression for multivariate Gaussian
  ![[Pasted image 20241001003124.png]]
