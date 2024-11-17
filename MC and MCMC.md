
Problem Setting: 
- If we only care about drawing samples, we don't require approx of posterior

### Markov Chain Monte Carlo (MCMC)
- especially for posteriors with intractable normalizing constants (bc of marginal likelihood) 
- Two concepts: 
	- Monte Carlo: integral approx via random sampling
	- Markov Chain: stochastic model that governs conditional probability distribution amongst series of random variables

### Monte Carlo Sampling
Three scenarios: 
- Inverse CDF known
- PDF known
- unnormalized PDF known 

### Rejection Sampling
- For when we have inverse CDF but maybe an unnormalized distribution
- Rejection sampling allows us to draw samples from unnormalized distribution as long as we supply proposal distribution (upper bound for unnormalized dist)
- Steps: 
	- Propose: Draw random sample from proposal distribution $x_{0} \sim q(x)$
	- Score: Sample from uniform dist $u \sim Uniform(0, Cq(x_{0}))$ which is some point under the upper bound
	- Verify: Evaluate sample $x_{0}$ in target distribution, if $u \leq \hat{p}(x_{0})$ then accept else reject
- Implication: allows us to draw samples from distribution without evaluating normalizing constant
- limitations: 
	- can spend much time rejecting samples if loose proposal dist
		- mainly a problem for high-dim spaces - curse of dimensionality

### Importance Sampling
- Proposal dist is now a good approximation. From MC integration, we can draw from the proposal distribution and compute importance weights from each drawn sample to form a weighted sum (weights differ depending whether target dist is normalized)
- Direct Importance Sampling: 
	- assume we can eval normalization constant
	  ![[Pasted image 20241001010937.png]]
- Self-normalized importance sampling: 
	- if normalizing constant intractable then: 
	  ![[Pasted image 20241001011008.png]]
How do choose proposals? 
- Trade-off: want support of proposal to cover target but also shouldn't be too loose else importance weights are too small
- Learn proposal: VI and use approx as proposal 

MCMC vs MC: 
- Rejection sampling suffers from Curse of dimensionality and importance sampling requires good proposal dist
- MCMC allows: 
	- drawing samples from proba dist that live in high dim spaces
	- weaker reliance on proposal dist
- main diff is that we take correlated samples rather than independent bc samples are from Markov chain

### Markov Chain
- Definitions:
  ![[Pasted image 20241001012126.png]]
  - Given any pair of states we have a conditional proba known as the transition/Markov kernel
  - Given a series of RV, if the transition kernel is independent (static) of order within series then it's stationary - can be represented in matrix form
    ![[Pasted image 20241001012331.png]]
    ![[Pasted image 20241001012423.png]]
    - diagonals are self loops
    - (r, c) is read as probability of transition from state r -> c: $p(c | r)$ 
      ![[Pasted image 20241001012655.png]]
  - Stationary distribution: when transitions given a stationary transition kernel are taken to the limit: $\pi = \pi A$
    - needs to satisfy property $A1 =1$ which means that each row of matrix should sum to 1. After norm then constant vector is eigenvector with eigenvalue 1. 
    - We also have that $A^{T}\pi = \pi$ 
    - Eigenvectors can also be computed via power method: 
      $$
      \pi_{t+1}= \frac{A^{T}\pi_{t}}{||A^{T}\pi_{t}||}
      $$
      - from this, we can see that iterating through Markov chain is like performing the power method => initial dist has no bearing on converged eigenvector. So for a carefully chosen transition kernel, the corresponding stationary distribution approx target dist
  - Ensuring stationary dist: 
    - for finite-state Markov chain, if it's irreducible (all states reachable) and aperiodic (has self-loops) then has unique stationary dist
    - For continuous spaces, require third condition: non-null recurrent - a state is non-null recurrent if expected time to return to the state is finite. Example: 
      ![[Pasted image 20241001014927.png]]
    - Ergodic if aperiodic and non-null recurrent 

### Markov Chain Monte Carlo
- Problem setting is reversed from Markov Chains, we have stationary distribution as posterior but we need to find transition kernel
	- MCMC is all about designing transition kernels so that their unique stationary distribution is target distribution we care about
	- MCMC algos work by sequentially drawing samples from Markov Chain
- Metropolis-Hastings Algorithm: 
	- simplest MCMC algo
	- begin with proposal dist that's conditional on current point $q(x_{s}|x_{s-1})$ 
	- Algo: 
		- Propose: at some iteration s, draw sample from proposal: $$
		  x_{s} \sim q(x_{s}|x_{s-1})
		  $$
		- Score 1: Assuming q is symmetric in args, compute acceptance proba from target dist: $$
		  A = min(1, p(x_{s})/p(x_{s-1}))
		  $$
		- Score 2: Like rejection sampling, draw from uniform dist $u \sim U(0,1)$
		- Verify: If $u \leq A$ then accept $x_{s}$ else reject and keep previous $x_{s} \leftarrow x_{s-1}$ 
	- Analysis: 
		- Acceptance probability intuition: if draw from our proposal leads us to higher density point (i.e. $p(x_{s})>p(x_{s-1}$) then definitely accept it (with probability of 1) but if less than, we'd rather not want to move to less-probable state so make probability of acceptance equal to the ratio less than 1 (this is like a reverse ReLU lmao)
		- Hastings Correction: made assumption earlier that proposal is symmetric in args, but if it isn't then need to modify acceptance: ![[Pasted image 20241001110435.png]]
		- Transition Kernel:![[Pasted image 20241001112543.png]]
		  The transition kernel here is a combo of proposal dist and acceptance proba
		- Detailed Balance: 
			- require that $p(x)p(x'|x) = p(x')p(x|x')$ 
			- if it holds -> stationary dist of Markov Chain is target dist![[Pasted image 20241001112806.png]]
			  
		- What makes a good proposal? 
			- we want to minimize mixing time: number of iters required to converge to stationarity (first few iters don't resemble eigenvector)
			- independence sampler likely to lead to long mixing times since not taking advantage of data (i.e. experience) and where we currently are in domain
			- Popular method is random walk Metropolis:$$
			  q(x'|x) = N(x'|x, \tau^{2}I)
			  $$ ![[Pasted image 20241001113251.png]]
			  ![[Pasted image 20241001114442.png]]![[Pasted image 20241001114549.png]]
			  
	- Limitations: 
		- need to specify proposal distribution, which is usually done independent of data -> leads to many rejections -> long mixing times
		- More complex MCMC methods rely on data to create better proposal dist
		  