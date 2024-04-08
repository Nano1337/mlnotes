

**Table of Contents: **
- General Info 
- GAN
- VAE
- Denoising AE
- 
#### Comparison to PCA:
- Autoencoders have an encoder and decoder scheme where the encoder tries to compress the raw input to a lower dimensional latent and then the decoder tries to reconstruct the original input
- This is similar to PCA where we get the lower dimensional projection (principal coordinates) by using the eigenvectors to capture the directions of most variance from the raw input and reconstruct the original input through a linear combination of the principal coordinates and their respective principal coordinates and then calculate the reconstruction error
- In this sense, deep Autoencoders could provide pseudo-invertible nonlinear dimensionality reduction 

### What distribution should the latent follow? 
- We can enforce a prior distribution by adding a probabilistic distance error (e.g. KLD or JSD). 
- Kullback-Leibler (KL) Divergence: 
$$D_{KL}(p||q) = \mathbb{E}_{x}\left[\log \frac{p(x)}{q(x)}\right]$$
	- But this is not defined in non-overlapping support, i.e. when q(x) = 0
	- Not symmetric
- The Jensen-Shannon (JSD) Divergence fixes this by essentially repeating the KL divergence twice to make it symmetric
- How do we calculate a Divergence metric when we don't have access to the data distribution? Recall, we are only given access to a batch of samples at training time. 

### Adversarial Networks
One way we can enforce the latent distribution to follow the prior distribution is through adversarial networks by using a discriminator (adversary) to judge whether a generated output is real or fake. This training method actually allows us to provably measure JSD at an individual sample level. The proof is shown below. Note that $\eta$ is the Discriminator D here. 
![[img/Pasted image 20240407200824.png]]
We can see that there is actually an extra term $-2\log(2)$, which acts as a lower bound on the loss if p(z) = q(z). In other words, if discriminator gives p = 0.5, meaning that $\eta(z) = 0.5$ from the initial distance formula, then we get $-\log(2) + -\log(2) = -2\log(2)$. 

### Mode Collapse
However, we may get mode collapse when the generator starts producing a limited variety of outputs, failing to capture the full diversity of the target distribution. 

### Wasserstein GAN
- Also known as Earth Mover's distance. Quantifies the amount of "work" moving a probability distribution to another. This better quantifies probabilistic distance than other metric, but I won't get into the details here. 


## Variational Autoencoders
The prior $p(z)$ here is a simple Gaussian, which is mathematically convenient to work with, while the conditional output $p(x|z)$ is complex (image generated). Thus, we would ideally use MLE to estimate the parameters: 
$$p_{\theta}(x) = \int p_{\theta}(x)p_{\theta}(x|z)dz$$
However, marginalizing across all $z$ is simply not practically possible, which makes this MLE intractable. Thus, the posterior density $p_{\theta}(z|x)$ is also intractable. 

Solution, in addition to decoder network $p_{\theta}(x|z)$, we also need to estimate an encoder $q_{\phi}(z|x)$ model. We can derive the loss terms based on the Evidence-based Lower Bound (ELBO) that we want to maximize: 
![[img/Pasted image 20240407202828.png]]
The first term is the reconstruction loss and the second term is to make the approximate posterior close to the Gaussian prior. 

#### Reparameterization Trick: 
To take a forward pass through the VAE, we have to sample from the latent distribution (Gaussian in this case). However, we can't backprop through such a stochastic operation, so the reparameterization trick is used. Since we estimate $\mu$ and $\sigma$ through the encoder model, instead of drawing z from $N(\mu, \sigma^{2})$ , we can instead draw $\epsilon$ from N(0, 1) and then get z = $\mu$ + $\sigma * \epsilon$, and thus gradient calculation isn't affected. 

### Generating Data
Once the VAE is trained, we can simply sample data from a Gaussian as the latent z and use the decoder model to attain our output image. Since z is a multivariate Gaussian, the diagonal prior on z are independent latent variables that cause different factors of variation (think principal components in PCA). 

## $\beta$-VAE
Simply just add a $\beta$ coefficient to the KLD loss term to weigh how strong the VAE should weigh having the latent match the Gaussian prior. 
## Denoising Autoencoders
![[img/Pasted image 20240407203829.png]]

It seems that $\psi(\phi(x)) - x$ would give us a vector that projects data from outside the manifold onto the manifold: 
![[img/Pasted image 20240407204203.png]]

Through the Bengio paper "What Regularized Auto-Encoders Learn from the Data-Generating Distribution", they find that: 
$$\nabla_{x}\log(p(x)) = -\nabla E(x)$$ In other words, the score function is the gradient of the log probability of the data! If we have the score function, we can sample from the distribution by setting an initial x drawn from an arbitrary prior distribution and then following the gradient of the data to eventually converge at a sample from p(x). This is called Langevin Dynamics: 
$$x_{i+1} \leftarrow x_{i}+ \epsilon \nabla \log p(x) + \sqrt{2\epsilon} \text{ }z_{i}, \text{ } i \in 0 \dots K $$ where the "learning rate" $\epsilon$ is sufficiently small and number of steps K is sufficiently large. 