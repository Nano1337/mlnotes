
### Basics
- Common BNN prior choice is Gaussian (i.e. weights in single layer share Gaussian prior)
	- Not considering data, we can sample from this chain of priors using MCMC
	- To approximate the posterior, usually use HMC and loss function becomes potential energy but HMC assumes full-batch loss which is not practical in modern practice

### Tractable Posterior Approx

Dropout: 
- at some layer l, draw Bernoulli distributed RV for each neuron and zero out those neurons (equivalent to zero out col vectors in weight matrices)
- MC Dropout: 
	- at test time, weights of NN scaled to compensate for dropout proba at training but can also keep dropout at test time to get a collection of different NN
	- This gives us a a way to approx posterior predictive distribution (i.e. each NN is a draw from some posterior distribution) and allows us to measure uncertainty

Deep Ensembles: 
- More straightforward way than MC dropout is to just get a collection of NN (independently and separately trained) and avg their predictions
	- Stochasticity comes from: weight init, data subsets subsampled during batching, hparam selection
	- The expectation is that each member of the ensemble converges to a different mode of the weight space for diversified predictions. 
		- comparing weight space isn't useful bc permutation invariance of weights but looking at function space to enforce repulsive force to enforce diverse outputs is much more effective
VI: 
- ADVI as we studied before
- Diff: VI methods capture local uncertainty around mode, ensembles identify different modes but ignore local uncertainty and might not pick the best point from each mode

### SWA
- Stochastic Weight Averaging
- After a certain point after some sort of convergence measure, collect model weights at different points of time along SGD trajectory. You then average these checkpoints bc you assume mode connectivity and flatness of the optimization landscape and this is usually related with generalization

### SWAG
- SWA but with a Gaussian fit over the collected model weights from different time points so we get the mean (which is just SWA) and a covariance. We can then just sample from this Gaussian to get a distribution of relevant model weights. This allows us to also model uncertainty. 
- To practically calculate covariance and not have to save all models on disk, we can calculate a diagonal and low-rank approx of covariance matrix

### Multi-SWAG
- SWAG is limited to single model, we can ensemble SWAG across multiple proposals (SWA run under different initializations)
- So to compute predictive distribution: Sample a SWAG mode and then sample network from this mode 
- particularly helpful for prediction data that's been corrupted where single mode might bias certain types of data distributions