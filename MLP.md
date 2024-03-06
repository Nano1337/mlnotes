
### Motivation
Nonlinear feature maps suffer from: 
- Curse of Dimensionality: 
	- $O(d^K)$ terms required for polynomial of degree K in d dimensional space
	- RBFs and Sigmoids require $K^d$ number of bins
- Sub-optimal: 
	- Basis mismatch. For example, a fixed RBF kernel we bin up into nxn bins is not the optimal basis for a mixture of 2 Gaussians that can be fit with two learned Gaussian basis functions. 

Instead of only making the weight of the kernel functions learnable, why don't we make the basis functions/kernels also learnable? This is exactly an MLP! 
$$f(x) = \sum\limits_{m=1}^{M}w_{m}^{1}\left(\sum\limits_{j=0}^{d}W_{jm}\hat x_{j}\right)$$
### Universal Approximation Theorem
Neural Network with at least one hidden layer and nonlinear activation can approximate any continuous function to any arbitrary level of accuracy

![[img/Pasted image 20240306170714.png]]
From the figure above, we can outline how MLP can function as a universal approximator. Two hidden neurons put together with can activation can create a bin that can be scaled/learned. As we increase the width of the MLP, we attain more bins, which is essentially reducing $\Delta x \rightarrow 0$ and we get a better approximator.     
### Boolean Logic 
Famously, you need an MLP with at least one hidden layer to model an XOR function. 
This is because the XOR is only 1 when only one of either inputs X or Y is on. This is represented as: 
$$X \oplus Y = (X \lor Y) \land (\bar X \lor \bar Y)$$
This can be represented intuitively as 6 weights and 3 thresholds or even smaller using skip connections with 5 weights and 2 thresholds: 
![[img/Pasted image 20240304234812.png]]

#### Width vs Depth: 
- Let N be the number of boolean variables, we will be concerned with worst case, which is XOR of all N boolean variables
- Max Width: single hidden layer would require $2^{N+1}+1$ perceptrons, which is exponential in N
- Max Depth: Will require 3(N-1) perceptrons (2(N-1) with skip cxns), which is linear in N but is very computationally expensive bc of such depth, adding 
- Better Depth: pairwise N, so we get $2\log_{2}N$ 
![[img/Pasted image 20240306173626.png]]

