
### Forward Process
Gradually introduce noise into input sample. This is described by a Markov chain: 
$$q(x_{0}, \dots, x_{N}) = q(x_{0})q(x_{1}|x_{0})\dots q(x_{N}|x_{N-1})$$
This noise is drawn from a Gaussian: 
$$q(x_{t}|x_{t-1}) =  N(x_{t}|\sqrt{1-\beta_{t}}x_{t-1}, \beta_{t}I)$$

which can be reparameterized as 
$$x_{t}= \sqrt{1-\beta_{t}}x_{t-1} + \sqrt{\beta_{t}} \epsilon \text{ , where } \epsilon\sim N(0, I)$$
### Reverse Process
Denoising from noise is modeled as a Markov chain in the reverse direction 
$$q(x_{0}, \dots, x_{N}) = q(x_{0}|x_{1})q(x_{1}|x_{0})\dots q(x_{N}|x_{N-1})q(x_{N})$$


### Optimizations 
We can speed up the forward process of adding noise by computing $x_{t}$ in one step. 
$$
\begin{align}
x_{t} = \sqrt{\bar a_{t}}x_{0} + \sqrt{1-\bar a_{t}} \epsilon \\ 
\text {where } \epsilon \sim N(0, 1) \\
\bar a_{t}= \Pi_{i=1}^{t}a_{i} \text{ and } a_{i}=1-\beta_{i}
\end{align}
$$

Then we find that the reverse denoising process is also Gaussian: 
![[img/Pasted image 20240407215009.png]]

The reverse conditional doesn't have a closed form, but another paper derived an approximation: 
$$\tilde \mu_{t} = \frac{1}{\sqrt {a_{t}}}\left(x_{t}- \frac{1-a_{t}}{\sqrt{1-\bar a_{t}}} \epsilon_{t}\right) $$
where $\epsilon_t$  is the noise introduced at step t. We don't know this value, but we can train a network $\epsilon_{\theta}(x_{t}, t)$ to approximate it: 
$$L(\theta) = ||\epsilon_{t}-\epsilon_{\theta}(x_{t}, t)||_{2}^{2}$$

### Training Algorithm: 
1. Randomly select timestep t and encode it (time step embedding)
2. Add noise to the image (forward process)
3. Train the denoising U-net to predict noise at timestep t (we have noise ground truth at timestep t from forward process)

### Testing Algorithm: 
1. Sample Gaussian Noise 
2. Iteratively denoise image with trained UNet model and subtract predicted noise (reverse process)

### Latent/Stable Diffusion
Speed Up: perform diffusion on low dimensional latent space 
Conditional Generation: condition denoising on text, images, other embeddings (e.g. fMRI)


### In Summary: 
For the training algorithm:

$\text{Repeat} \quad x_0 \sim q(x_0) \quad t \sim \text{uniform}\{1, \ldots, T\} \quad \epsilon \sim \mathcal{N}(0, I) \quad x_t = \sqrt{\alpha_t} x_0 + \sqrt{1 - \alpha_t} \epsilon \quad \theta \leftarrow \theta + \eta \nabla_\theta \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \quad \text{Until convergence}$

For the testing algorithm:

$x_T \sim \mathcal{N}(x_0, I) \quad \text{For } t = T, \ldots, 1 \quad \text{do} \quad \epsilon \sim \mathcal{N}(0, I) \text{ if } t > 1, \text{ else } \epsilon = 0 \quad x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\alpha_t}}\epsilon_\theta(x_t, t) \right) + \sqrt{\alpha_t}\epsilon \quad \text{Return } x_0$

