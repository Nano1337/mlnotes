
### Gradient Descent: 
- Batch/Stochastic GD
- How SGD is impacted by: 
	- Initialization 
	- Step Size 
	- Loss: convex, nonconvex, Lipshitz continuity
- How to measure performance of GD: (expected optimality gap)
- no questions on momentum NTK

### Variance Reduction in Gradient Descent
- What problem is variance reduction solving
- Impact of batching
- Know variance reduction methods: 
	- SVRG
	- SAG
	- SAGA

### Fundamentals of Bayesian Inference
- Steps of inference: specify (likelihood, prior) -> derive posterior, marginal likelihood, posterior predictive distribution
- Point Estimates (MLE, MAP): how does a prediction made with point estimate compare with the posterior predictive distribution?
- Conjugate Priors
- Beta-binomial model: how do we understand posterior as function of likelihood/prior

### Bayesian Linear Regression
- Formation of Gaussian-Gaussian model: 
	- Gaussian Likelihood interpretation 
	- Gaussian Prior interpretation
- Gaussian posterior: recall the 1D and what happens when we add more data points
- Posterior predictive distribution: what does this provide as we make more predictions? 
- Not expected to derive posterior or posterior predictive distribution

### Gaussian Process Regression
- Know the steps: 
	- Gaussian process prior on functions
	- Gaussian likelihood: noise model on function values 
	- Posterior distribution: condition on training function values, infer test function values
- Interpolating GP vs Noisy GP 
- Know squared exp cov func (no need to memorize others)
### Bayesian Model Selection
- Role of marginal likelihood in model selection 
	- maximizing marginal likelihood gives tradeoff
	- **Model Fit**: The marginal likelihood takes into account how well the model fits the observed data. A model that fits the data well will have high likelihood for some values of the parameters.
	- **Model Complexity**: It also penalizes models that are too complex by integrating over the entire parameter space. More complex models, which can explain many different datasets, often have lower marginal likelihood because their probability mass is spread over a large parameter space (Occam’s Razor).
- Occam's razor
	- balance data fit vs model complexity (bias, variance lol)
- Empirical Bayes for GP: 
	- Data fit and model complexity
- Undestand empirical Bayes in context of sq. exp. cov (assignment 2)
	- small length scale, good data fit
	- large length scale, low model complexity (i.e. smoothing)

### Approximate Inference
- When do we need approx inference? 
- Laplace approx: 
	- what is it (need to know multivariate Taylor series expansion)
	- What are limitations
		- assuming mean vector is MAP estimate
- MC integration: 
	- Intuition (sample in high-density regions)
		- draw samples in proportion to posterior density
	- Basic properties (unbiased, what impacts variance?)
		- more samples the lower variance 

### Variational Inference
- Know properties of KLD
	- if KLD is 0 then q = p
	- KLD always non-negative
	- asymmetric
- Know ELBO: 
	- expected log-joint + entropy (MSE + KLD to Gaussian prior)
- Gradient-Based VI: 
	- why is it difficult to take gradients of ELBO 
		- gradient on variational distribution is intractable bc not an expectation
	- Reparam + change of variable
- ADVI (Automatic Differentiation VI)
	- assuming variational distribution always multivariate Gaussian
	- Use constrained params for log-joint and unconstrained for KLD

### MCMC: 
- Different flavors of MC integration (proposal distributions, rejection sampling, importance sampling)
- Fundamentals of Markov Chains
- Metropolis-Hastings
	- know algorithm
		- what are challenges in deciding on proposal distributions and how do they impact Metropolis-Hastings`
			- we don't want long mixing times, use random walk MH