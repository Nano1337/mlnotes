
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

### MLE
- Given Bernoulli distribution: 
  $$
  p(y|\theta) = \theta^{y}\cdot (1-\theta)^{1-y}
  $$
  - The likelihood is just the probability over all outcomes as product of individual outcome probabilities
    $$
    p(D|\theta) = p(y_{1},y_{2}, \dots, y_{n}|\theta) = \Pi_{n=1}^{N}p(y_{n}|\theta)
    $$
  - The parameter $\theta$ here can be solved via the maximum likelihood estimator and formulate as minimization problem to get NLL: 
    $$
    -\log p(D|\theta) = -\sum\limits_{n=1}^{N}y_{n}\log\theta+(1-y_{n})\log(1-\theta)
    $$
    Then differentiate and solve for $\theta_{MLE} = X/N$
    

### Beta Distribution and More: 
- to account for uncertainty in estimate, use prior probability on parameter. For $\theta \in [0,1]$ , it's common to use beta distribution as prior: 
  $$
  p(\theta;a,b) = Beta(\theta;a,b) = Z^{-1}_{beta}(\theta^{a-1}\cdot (1-\theta)^{b-1})
  $$
  where normalizing constant $Z_{beta}=\int_{x} x^{a-1}\cdot (1-x)^{b-1}dx$ ensures pdf integrates to 1
- Going back to data likelihood of bernoulli, we find that this is equivalent to: 
  $$
  \theta^{N_{h}} \cdot (1-\theta)^{N_{t}}
  $$
  where $N_{h}$ is number of heads and the other is number of tails and added together is overall N
- The product of individual bernoulli distributions (likelihood) becomes the binomial: 
  $$
  p(D|\theta) = C(N, N_{h}) \theta^{N_{h}}\cdot (1-\theta)^{N_{t}}
  $$
  and the corresponding normalizing constant is simply the inverted $Z_{bin} = C(N, N_{h})^{-1}$ 
- Given that we have the likelihood and prior, this is the numerator for the posterior and by maximizing this product (and treating the marginal likelihood denominator as a constant), we will strike a balance between 1) data fit and 2) adherence to the prior
	- This is known as maximum a posteriori (MAP): $\max p(D|\theta) p(\theta)$ 
	- You can solve this via MLE as before and get proportional: 
	  $-\log p(D|\theta)p(\theta) \propto -(N_{h}+a-1)\log \theta -(n_{t}+b-1)\cdot \log(1-\theta)$
	- Differentiating and setting to zero: 
	  $$
	  \theta_{MAP}= \frac{N_{h}+a-1}{N+a+b-2}
	  $$
	  Where did the normalizing constants go? Well, adding those back in, they get zeroed out during differentiation since they're log constants and this fact makes MLE/MAP tractable 
- Posterior distribution: 
	- challenge: normalizing constants required for computing posterior
		- soln: use approximate inference algos
	- Consider unnormalized prior: 
	  $$ 
	  p(D|\theta)p(\theta) = \hat{Z}^{-1}[\theta^{N_{h}+a-1}\cdot (1-\theta)^{N_{t}+b-1}]
	  $$
	  - Although unnormalized, this is the exact form of the beta distribution! If prior and posterior are in same proba distribution family, then we say that the prior is a **conjugate prior** for the likelihood
- Verifying conjugacy: 
	- can be verified by taking a look at the form taken by the unnormalized posterior: 
	  $$
	  \begin{align}
	  p(\theta|D) = Z^{-1}_{beta}Z^{-1}_{bin}\frac{\theta^{N_{h}+a-1}\cdot (1-\theta)^{N_{t}+b-1}}{p(D)} \\
	  p(D) = \int_{x}p(D|x)p(x)dx=Z^{-1}_{beta}Z^{-1}_{bin}\int_{x}x^{N_{h}+a-1}\cdot (1-x)^{N_{t}+b-1} = Z^{-1}_{beta}Z^{-1}_{bin}\hat{Z}_{beta}
	  \end{align}
	  $$
	  where $\hat{Z}_{beta}$ is the normalizing constant for the new beta distribution. The posterior then parameterized as: 
	  $$
	  p(\theta|D) = Beta(\theta|N_{h}+a, N_{t}+b)
	  $$
- Interpreting the posterior: 
	- even if our observations based on N are unlucky and seem to be very biased, the prior restricts the posterior to stay as a Beta distribution
- Posterior predictive distribution: 
	- we can marginalize out the model parameters
	- say we wanted to find $p(y=Heads|D) = \int_{\theta}p(y=Heads,\theta|D)d\theta$ by the chaining rule this is equivalent to $\int_{\theta}p(y=Heads|\theta,D)p(\theta|D)d\theta$ 
	- Through the conditional independence assumption, probability of heads is independent of the data and thus we can take that out: $\int_{\theta}p(y=Heads|\theta)p(\theta|D)d\theta$ 
	- And we find that this is simply just the marginal likelihood: 
	  $$
	  p(y=Heads|D) = \frac{Z_{beta}(N_{h}+a+1,N_{t}+b)}{Z_{beta}(N_{h}+a,N_{t}+b)}
	  $$
- Beta-binomial: 
	- if we want to know the probability of flipping some $M_{h}$ number of heads out of total M coin flips, then need to also consider the binomial normalizing constant (coeff): 
	  $$
	  p(y=M_{h}) = Z^{-1}_{bin}(M_{h},M)\frac{Z_{beta}(N_{h}+a+M_{h}, N_{t}+b+M-M_{h})}{Z_{beta}(N_{h}+a, N_{t}+b)}
	  $$
	- thus, the posterior predictive distribution is also known as the Bayesian model average. It's the weighted average of predictions, weighted by the posterior: 
	  ![[Pasted image 20240923201014.png]]
- Point estimates: 
	- $\mathbb{E}[\theta] = \int_{x}xp(x|D)dx$ , the plug-in estimator 

### Bayesian Regression: 
- Reviewing Gaussians: 
  ![[Pasted image 20240923201226.png]]
  ![[Pasted image 20240923201234.png]]
  ![[Pasted image 20240923201313.png]]
  - we can also invert this covariance matrix to get the precision matrix: 
    $\Lambda = \Sigma^{-1} = U\Phi^{-1}U^T$ where $\Phi^{-1}$ is simply the reciprocal of the diagonal matrix by definition
    ![[Pasted image 20240923201524.png]]
    ![[Pasted image 20240923201903.png]]
    -  From the last line, we can see how covariance is a fixed transformation of the squared euclidean distance. This is well defined (and invertible) bc matrix is PD 
      ![[Pasted image 20240923202023.png]]


### Bayesian Regression: 
- allows us to model regression with uncertainty in predictions via the posterior predictive distribution: 
  $$
  p(y|D,x) = \int_{x}p(y|w,x)p(w|D)dw
  $$
- Gaussian Noise Model: 
	- $y_{n} = w^{T}x_{n}+\epsilon, \epsilon \sim N(0, \sigma^{2}_{n})$
		- this implies that the label can be decomposed as noise free prediction and zero mean Gaussian noise
- Other likelihood and priors: 
	- Use student's t-distribution for likelihood and Laplace for prior
		- Allows for robustness to outliers
		- sparsity: only a few components of vector should be nonzero. We can use Laplace distribution and this for multivariate data this gives the L1 norm. 
		- Downside is that this doesn't allow for conjugate priors
	- Then why do we use Gaussians for both likelihood and prior? 
		- Gaussian prior is a conjugate prior for the Gaussian likelihood

### Gaussian Process
- instance of Bayesian nonparametrics
- this is kernel learning but in a Bayesian context
- If we use nonlinear functions, why not just make the kernels themselves learnable -> MLPs? The reasons: 
	- well-calibrated predictive uncertainty
	- Bayesian model selection "training the model"
  - Collection of RV, finite number of which have joint Gaussian distribution
    ![[Pasted image 20240923205143.png]]
    ![[Pasted image 20240923205422.png]]
    ![[Pasted image 20240923205438.png]]
    ![[Pasted image 20240923205540.png]]
    ![[Pasted image 20240923205559.png]]
    ![[Pasted image 20240923205616.png]]



### Sampling from GP
- From a single GP: 
	- specify (finite) set of points in domain
	- evaluate mean function at these points $\mu_{x}$
	- Eval covariance function between all pairs of points $K_{X,X}$
	- Sample from multivariate Gaussian: $f \sim N(\mu_{X}, K_{X,X})$
- From a multivariate Gaussian: 
	- easy to sample from distribution $X \sim N(\mu, \Sigma)$
		- Compute the Cholesky decomp of covariance: $\Sigma = LL^T$ 
		- Draw IID samples from standard normal distribution: $z \sim N(0, I)$
		- Transform the vector: $x = \mu + Lz$ ![[Pasted image 20240923205931.png]]
- Noise Free Predictions: 
	- ![[Pasted image 20240924112400.png]]
	  I have opened all the images you uploaded. Let me provide concise and organized notes based on their content:

### **Gaussian Process Regression (GP) Overview**

#### **Noise-free Predictions**
- **Purpose**: To interpolate given data using the posterior distribution in a noise-free scenario.
- **Joint Distribution**: Defined by the input matrix `X` and test inputs `X*`, and the covariance between training and test inputs (`K_X,X*` and `K_**,X*`).
- **Prediction Formula**: 
  $$
\hat{f}^* = \mu_* + K_{X*,X} K_{X,X}^{-1} (f_X - \mu_X)$$
  
  - In this formula, `K_X,X` is the covariance matrix for the training inputs, and `K_X*,X` is the covariance matrix between the training and test inputs.

#### **Posterior Distribution**:
- Posterior: Given by:$$
    P(f*|f_X) = \mathcal{N}(\hat{\mu}, \hat{\Sigma})
  $$
  where: $$
    \hat{\mu} = \mu_* + K_{X*,X} K_{X,X}^{-1} (f_X - \mu_X), \quad \hat{\Sigma} = K_{*,*} - K_{X*,X} K_{X,X}^{-1} K_{X,X*}
  $$
  
#### **Noise Model**
- **Data with Noise**: If the data is noisy, the likelihood of a Gaussian process can model the noise with the following assumptions:
  - **Covariance Matrix with Noise**:$$
    Cov[y|X] = K_{X,X} + \sigma_n^2 I
    $$
- **Posterior Prediction**:$$
  p(f*|y) = \mathcal{N}(\hat{\mu}, \hat{\Sigma})
  $$
  where 
  $$
\begin{align}
    \hat{\mu} &= \mu_* + K_{X*,X}(K_{X,X} + \sigma_n^2 I)^{-1}(y - \mu_X) \\
    \hat{\Sigma} &= K_{*,*} - K_{X*,X}(K_{X,X} + \sigma_n^2 I)^{-1} K_{X,X*}
\end{align}
  $$
#### **GP Inference and Computation**
- **Cholesky Decomposition**: Used to compute matrix inversions efficiently:
  $$
  (K_{X,X} + \sigma_n^2 I) = L L^T
  $$
  The lower triangular matrix \(L\) is used to compute the inverse.

#### **Predictions**:
- Prediction is made by solving a linear system:
  $$
  f(x_*) = \sum_{i=1}^{n} \alpha_i k(x_*, x_i), \quad \alpha = (K_{X,X} + \sigma_n^2 I)^{-1} y
  $$
#### **Predictive Variance and Defining Kernels**
- **Predictive Variance**: Defined by the diagonal entries of the covariance matrix.
  $$
  \text{Predictive Variance} = K_{*,*} - K_{X*,X}(K_{X,X} + \sigma_n^2 I)^{-1}K_{X,X*}
  $$
- **Kernel Functions**: Kernels define the covariance function and should be symmetric and positive definite. A common example of a kernel function is the Radial Basis Function (RBF).
### Summary:
- **Noise-free Scenario**: Posterior is an interpolator of the data.
- **Noisy Scenario**: Gaussian process likelihood can model noisy data, adjusting the covariance matrix.
- **Prediction**: Made using a combination of kernel functions and solving a linear system.
- **Kernels**: Define similarity between points and are the backbone of Gaussian process regression.


### Types of Kernels: 
- Stationary: translation invariant
- Isotropic: invariant to rigid transformations (e.g. rotation + translation)
- Types: 
	- dot product Kernel has properties: 
		  - symmetric, PD (only if we use noise variance) but if n>d then singular 
		  - is stationary and isotropic![[Pasted image 20240924154414.png]]
	- Squared-exp kernel: 
		- most common kernel for GP: 
		  $$
		  k_{SE}(x,x') = exp[-\frac{1}{2l^{2}}||x-x'||^{2}]
		  $$
			- where $l$ is length scale and determines the distance at which we consider points similar. As $l \rightarrow 0$, only assign high similarity to points near training data and converges towards identity matrix. The function values between any pair of points are seen as unrelated and leads to overfitting 
			- Large length scale will result in kernel matrix approaching all ones. Thus any pair of points function values will be related and encourages smoothness
	- Non-squared exponential kernel: less nice
	- Rational quadratic covariance function and many other kernels