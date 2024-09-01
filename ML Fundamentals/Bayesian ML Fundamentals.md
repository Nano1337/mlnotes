
### Inference (In Bayesian ML)
- representing and computing the probability distribution of a model given data 
- emphasis on finding a good set of models (rather than just one)
	- This allows for us to quantify uncertainty in the model upon making predictions

**Exact Inference**
- Can compute posterior in closed form
	- bayesian regression 
	- bayesian model selection

**Approximate Inference**
- a neat analogy is "optimization : differentiation :: inference : integration"
- What follows is that for more complex problems (e.g. latent variable models) exact inference not tractable
	- Laplace approximation good starting point. Simple but poor approx
	- Variational inference with an emphasis on gradient-based methods. Monte Carlo gradient estimation more broadly 
	- Markov Chain Monte Carlo (MCMC): Metropolis Hastings, Gibbs sampling. Also includes Hamiltonian Monte Carlo 