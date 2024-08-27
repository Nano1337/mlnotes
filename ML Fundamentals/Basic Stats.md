TODO: 
- Add stuff from previous stats cheat sheets when I have the time
- Add all relevant probability distributions, expected values, variances, and examples

Discrete RV: 
- have Probability Mass Functions (PMFs) and Cumulative Mass Functions (CMFs)

Continuous RV: 
- have Probability Density Functions (PDFs) and Cumulative Density Functions (CDFs)

## Random Variables: 

#### Bernoulli: 
PMF: $f(y) = p^{y}(1-p)^{1-y}$

#### Normal: 

Univariate Gaussian PDF: $f(x; \mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}}exp(-\frac{(x-\mu)^2}{2\sigma^2})$ 
Multivariate Gaussian PDF: $f(x; \mu, \Sigma)=\frac{1}{(2\pi)^{\frac{n}{2}}|\Sigma|^{\frac{1}{2}}}exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))$ 

## Calculating MLE: 

Maximum Likelihood Estimation (MLE) is from Bayesian statistics where we want to maximize the likelihood function p(data|model) where we want to estimate the parameters that maximizes likelihood - essentially finding a probability distribution that best models/fits the underlying data distribution. This is generally done through the following: 
$$\frac{\partial}{\partial param} \log \Sigma p(data|param) = 0$$
and then we solve for param. This is because the likelihood function is a joint function that can be broken down as the product of its conditional probabilities, so applying a log turns that into a summation of conditional probabilities. Taking the partial with respect to the param and setting to zero gives the maximal point. 