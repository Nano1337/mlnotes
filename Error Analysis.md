We want to better understand the different types of errors that may arise before we go about trying to reduce it. 

From this figure below, we see: 
- $F_{\delta}$ is feasible function space
- $F$ is all possible function space
- $\hat f \in F_\delta$ is one function from feasible function space
![[img/Pasted image 20240304215940.png]]


**Population Error**: 
- Considers all of function space. Formulated as: 
$$R(\hat f) - \inf_{f \in F}R(f)$$
- first term: expected/true risk, represents average error of $\hat f$ on ALL possible inputs
	- This is not possible, is a theoretical measure
- $\inf_{f \in F} R(f) = f^{*}$ is the lowest possible true risk over all of F (tight lower bound). This is the optimal function that can ever be learned

Population Error can be decomposed into: 
- **Statistical Error**: 
	- Defined as $\sup_{f \in F_{\delta}} |R(f) - \hat R(f)| \leq constant*\frac{complexity}{n^\frac{1}{d}}$ 
	- Only involves $F_\delta$ space and the same function in both risk terms
	- tight upper bound difference between true and empirical risk
	- Basically, how well does this feasible function operate across all theoretical inputs vs the train set inputs
	- Empirical risk depends on n, so increasing n would decrease statistical error
	- Complexity results from increasing data dimensionality, which would increase statistical error
	- Analogous to **Variance**
- **Optimization Error**: 
	- Defined as $\hat R (\hat f) - \inf_{f \in F_{\delta}} \hat R(f)$
	- Only deals with empirical risk since optimization is only possible with existing data
	- Measures how close a feasible function is to the optimal feasible function
	- Error can be decreased with better optimizer
- **Approximation Error**: 
	- Defined as  $\inf_{f \in F_{\delta}} R(f) - \inf_{f \in F_{\delta}} R(f)$
	- Only deals with true error 
	- Measures how close optimal feasible function is with optimal possible function
	- Error can be decreased by increasing complexity since that allows $F_{\delta}$ to have a greater chance of containing the optimal possible function
	- Analogous to **Bias**

This drawing below shows what each of the errors does. The unlabeled error is approximation
![[img/Pasted image 20240304225223.png]]

While it seems impossible to decrease statistical and approximation error by increasing model complexity through overparameterization, this is possible because increasing model complexity decreases approximation error and statistical error is hypothesized to decrease because either: 
1. Overparameterized models find simpler solutions, thus preventing overfitting and decrease variance
2. Inductive bias of certain architectures like CNN or Transformers reduce total function space $\mathcal{F}$ so approximation error goes down? 