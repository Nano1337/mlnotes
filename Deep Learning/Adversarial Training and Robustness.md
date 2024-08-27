
Do training with adversarial samples

### Outer Minimization
$$min_{\theta}\sum\limits_{x, y \in S} max_{\delta \in \Delta} Loss(x + \delta, y; \theta)$$
We can do gradient descent, but how do we find the gradient of the inner maximization term? 

**Danskin's Theorem**: 
A fundamental result from optimization is: 
$$\nabla_{\theta}max_{\delta \in \Delta} Loss(x + \delta, y; \theta) = \nabla_{\theta}Loss(x = \delta^{*}, y; \theta)$$
where
$$\delta^{*}=max_{\delta \in \Delta} Loss(x + \delta, y; \theta)$$
which means we can optimize through the max by finding its maximum value, but this only applies when max is found exactly. 

### Steps: 
1. Select mini-batch B
2. For each sample, calculate the corresponding adversarial sample 
3. Update parameters where learning rate is divided by batch size
It's also common to mix adversarial samples and normal samples for stability reasons

## Other Defense Mechanisms 

#### DefenseGANs
- train GAN that generates unperturbed images
- instead of classifying an input image, use the closest image generated by the GAN 
- Pros: effective against white box and black box attacks and theoretically no accuracy drop
- Cons: complex method and I hate training GANs