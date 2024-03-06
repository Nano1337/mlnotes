
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

Intuition on why it's a 

### Boolean Logic 
Famously, you need an MLP with at least one hidden layer to model an XOR function. 
This is because the XOR is only 1 when only one of either inputs X or Y is on. This is represented as: 
$$X \oplus Y = (X \lor Y) \land (\bar X \lor \bar Y)$$
This can be represented intuitively as 6 weights and 3 thresholds or even smaller using skip connections with 5 weights and 2 thresholds: 
![[img/Pasted image 20240304234812.png]]


