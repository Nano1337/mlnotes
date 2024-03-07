
### Batch Normalization
- Training assumes data comes from same distribution but each mini-batch may have a different distribution that may occur in each layer of the network -> covariate shift 
- Move all batches to the same place with 0 mean and unit standard deviation 
- Batch norm is a shift-adjustment that happens after the weighted addition of inputs but before the application of activation

Let $z_{i}$ be the pre-activation input to the BN layer. Then we perform scaling via: 
$$u_{i}= \frac{z_{i}-\mu_{B}}{\sqrt {\sigma_{B}^{2}+\epsilon}}$$
And allow for learnable parameters $\gamma$ and $\beta$: 
$$\hat z_{i}= \gamma u_{i} + \beta $$
Backprop through a BN layer. 
$$\frac{\partial L}{\partial u_{i}} = \gamma \frac{\partial L}{\partial y}f'(\hat z)$$
![[img/Pasted image 20240306235001.png]]

