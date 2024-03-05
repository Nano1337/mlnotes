
## Linear Regression
Want to find line of best fit given our data, assuming that our data (x) and label (y) are related through a linear transformation. This can be achieved by minimizing the loss function Mean Squared Error (MSE): $$\mathcal{L}(\theta) = \frac{1}{m}\sum\limits_{m=1}^{m} (\theta^{T}x - y)^{2}$$ The solution can be found either through Gradient Descent (GD) or a Closed Form solution since linear regression always has a global optimum. 

**Gradient Descent:** $$\theta_{t+1} = \theta_{t} - \alpha\nabla_{\theta}\mathcal{L}$$
where $\nabla_{\theta}\mathcal{L}$ is the gradient, $\alpha$ is the learning rate. 

**Closed Form Derivation**: 

$$
\begin{align}
\mathcal{L}(\theta) = ||\theta^{T}X - y||^{2} \\ 
(\theta^{T}X - y)^{T}(\theta^{T}X - y) \\
(X^{T}\theta - y)(\theta^{T}X - y) \\
X^{T}\theta^{T}\theta X - 2X^{T}\theta^{T}y + y^{T}y \\
\text{apply } \nabla_{\theta}\mathcal{L} \\
2X^{T}X\theta - 2X^{T}y = 0 \\
\theta = (X^{T}T)^{-1}X^{T}y
\end{align}
$$

**Probabilistic View of Linear Regression**
Assume every data point is modeled by $y = X\beta + \epsilon$ where $\epsilon \in N(0, \sigma^{2})$. By MLE, we get that the line of best fit can also be derived by minimizing MSE. 
![[Pasted image 20240304191732.png]]
Using MLE, we derive: 
$$\mathcal{L}(\beta) = \prod\limits_{i=1}^{n}\frac{1}{\sqrt{2\pi\sigma^{2}}}exp(-\frac{y_{i}-X_{i}\beta}{2\sigma^{2}})$$ After taking the log, we get: 
$$= -\frac{1}{2\sigma^{2}}\sum\limits_{i=1}^{n} (y_{i}-X_{i}\beta)^{2}$$
which is MSE off by a constant factor. 

## Polynomial Regression 

Allows us to apply kernel functions to the original X vector to create a design matrix and then learn beta coefficients to model nonlinear relationships. The design matrix for polynomial regression has the first column as a ones vector to act as a bias and each column vector after is the original x data vector taken to the d-th power. Model params $\theta \in \mathbb{R}^{d+1}$ . 

## Locally Weighted Linear Regression

Loss function: 
$$
\begin{align} 
\mathcal{L}(\theta) = \sum\limits w^{(i)}(\theta^{T}x -y)^{2} \\
w^{(i)}=exp({-\frac{(x^{(i)} - x)^2}{2\tau^{2}}})
\end{align}

$$
Where higher w is a greater influence on the loss.

At a given $x^{(i)}$, the weights of a neighbor are a higher. LWR is considered "non-parametric" bc the number of parameters grows with respect to train set size (need to calculate weights w.r.t entire train set during prediction with a new query x). 

**Pros:**
- Don't need a single function to fit all the data since it's smooth and locally weighted.
**Cons:**
- Less efficient use of data
- Sparse data not tolerated since it requires dense neighbors for a good local approximation
- Computationally expensive since each query requires creating a weight matrix w.r.t. all training samples

**Code:**
```python 
# assuming self.X is all train samples
# x is the query point
def calc_weights(self, x): # x is shape (1,m)
	diff_mn = self.X - x
	diffsq_m = np.sum(diff_mn**2, dim=1)
	w_n = np.exp(diffsq_m / (2*self.tau**2))
	W_nn = np.diag(w_n)
	return W_nn

def calc_theta(self, x): 
	W = calc_weights(x)
	theta = np.linalg.inv(self.X.T @ W @ self.X) @ self.X.T @ W @ self.y
	return theta 

def predict(x): 
	x_with_int = np.hstack(([1], x))
	theta = calc_theta(x_with_int)
	return theta.T @ x_with_int
```

## Logistic Regression 

Prediction function: 
$$h_{\theta}(x) = \frac{1}{1 + e^{-\theta^{T}x}} = \sigma(\theta^{T}x)$$
Likelihood is Bernoulli where p = $h_{\theta}(x)$ 
Taking the log of the likelihood function, we get the loss function: 
$$\mathcal{L}(\theta) = \sum\limits y\log h_{\theta}(x) + (1-y)(1-h_{\theta}(x))$$









## Regularization
- Ridge (L2) weight norm, causes irrelevant features to converge towards 0
- Lasso (L1) weight norm, causes sparsity by causing irrelevant features to eventually become 0. 

## Feature Selection

Pretty intuitive concepts: 
- Filter: use heuristic to score and select features with top-k scores. Example score function using mutual information $S(x) = I(x;y)$. 
- Wrapper: wraps around learning algorithm to evaluate how well it does using different feature subsets
	- Forward Search algorithm: Start with an empty set F. For $i \in \{i, \dots, n\}$, add new feat to F and train/test model. $F = F \cup i$ if best performing subset. Repeat and terminate when all features have gone through or you reach some threshold of n features. 
	- Backward Search: exact opposite, start with full feature set and remove until you hit empty set. 