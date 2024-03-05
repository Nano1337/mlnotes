
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
![[img/Pasted image 20240304191732.png]]
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


## Gaussian Discriminant Analysis (GDA)

Despite the name, this is the first "generative" model we cover. In other words, it models the joint distribution $p(x, y) = p(x|y)p(y)$ rather than just $p(y|x)$ in discriminative models. 

In this case, $p_{m}(x|y) = N(\mu_{m}, \Sigma)$ where m is the number of classes (usually 2). $p(y) \sim Bernoulli(y)$. This model assumes each label's data is Gaussian distributed, parameterized by mean and covariance. We can derive the MLE for each of the parameters $\phi_{y}, \mu_{0}, \mu_{1}, \Sigma$, where $\phi_{y}$ is the proportion of samples of y=1, $\mu_{0}$ is the mean of all samples in class y=0, same with other mean, and $\Sigma = \frac{1}{m} \sum\limits_{i=1}^{m}(x^{(i)}- \mu_{y(i)}))(x^{(i)}- \mu_{y(i)}))^{T}$ 

#### Formula for $\phi_y$ after MLE:  
$\phi_y$ = $\frac{1}{m}\sum\limits_{i=1}^{m}\mathbb{1}\{y^{(i)}=1\}$ 

#### MLE for $\mu_0$: 
Substituting in L and ignoring unrelated terms, we get:
$\frac{\partial}{\partial \mu_0} \Sigma^m_{i=1, y^{(i)} = 0} \log p(x^{(i)} | y^{(i)}; \mu_0, \Sigma)$
Substituting in the definition of a multivariate Gaussian: 
$\frac{\partial}{\partial \mu_0} \Sigma^m_{i=1, y^{(i)} = 0} \log \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} exp(-\frac{1}{2}(x^{(i)} - \mu_0)^T \Sigma^{-1} (x^{(i)} - \mu_0))$
After applying the log, we can get rid of the first constant since it doesn't contain $\mu_0$: 
$\frac{\partial}{\partial \mu_0} \Sigma^m_{i=1, y^{(i)} = 0} -\frac{1}{2}(x^{(i)} - \mu_0)^T \Sigma^{-1} (x^{(i)} - \mu_0)$
Now we apply the partial derivative with respect to $\mu_0$.
$\Sigma^m_{i=1, y^{(i)} = 0} -\frac{1}{2}(-2)\Sigma^{-1}(x^{(i)} - \mu_0)$
Simplifying and setting equal to zero: 
$\Sigma^m_{i=1, y^{(i)} = 0} \Sigma^{-1}(x^{(i)} - \mu_0) = 0$
Multiply both sides by $\Sigma$:  
$\Sigma^m_{i=1, y^{(i)} = 0} (x^{(i)} - \mu_0) = 0$
Let $m_0 = \Sigma^m_{i=1} \mathbb{1}\{y^{(i)}=0\}$, which gives us:
$m_0 \Sigma^m_{i=1}x^{(i)} = m_0\mu_0$
Rearranging and expanding $m_0$: 
$\mu_0 = \frac{\Sigma^m_{i=1} \mathbb{1}\{y^{(i)}=0\} x^{(i)}}{\Sigma^m_{i=1} \mathbb{1}\{y^{(i)}=0\}} \blacksquare$

#### MLE for $\Sigma$: 
To find $\Sigma$, we use MLE again with respect to $\Sigma$. For ease of notation we use $\mu_k$. In other words, we start with $\frac{\partial}{\partial \Sigma} L(\phi , \mu_k, \Sigma)$.
Substituting in L, we get: 
$\frac{\partial}{\partial \mu_0} \Sigma^m_{i=1, y^{(i)} = 0} \log \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} exp(-\frac{1}{2}(x^{(i)} - \mu_k)^T \Sigma^{-1} (x^{(i)} - \mu_k))$
After applying the log, we attain: 
$\frac{\partial}{\partial \mu_0} \Sigma^m_{i=1} -\frac{n}{2} \log (2\pi) - \frac{1}{2} \log | \Sigma| - \frac{1}{2}(x^{(i)} - \mu_k)^T \Sigma^{-1} (x^{(i)} - \mu_k)$
As an aside, we take the partial derivative with respect to each of these terms. The first term is a constant so it is dropped. 
The second term:
$\frac{\partial \log |\Sigma|}{\partial \Sigma}$ by chain rule $= \frac{\partial \log |\Sigma|}{\partial |\Sigma|}  \frac{\partial |\Sigma|}{\partial \Sigma} = \frac{1}{|\Sigma|} |\Sigma|\Sigma^{-T} = \Sigma^{-T} = \Sigma^{-1}$
The third term: $\frac{\partial}{\partial \Sigma}[\frac{1}{2}(x^{(i)} - \mu_k)^T \Sigma^{-1} (x^{(i)} - \mu_k)]$ becomes $-\frac{1}{2}\Sigma^{-1}(x^{(i)} - \mu_k)^T (x^{(i)} - \mu_k)\Sigma^{-1}$
Subsituting back in, factoring out a $-\frac{1}{2} \Sigma^{-1}$ and setting to 0: \\
$\Sigma^m_{i=1} -\frac{1}{2} \Sigma^{-1} (1 - (x^{(i)} - \mu_k) (x^{(i)} - \mu_k)^T\Sigma^{-1}) = 0$
Divide both sides by $-\frac{1}{2} \Sigma^{-1}$: 
$\Sigma^m_{i=1} [1 - (x^{(i)} - \mu_k) (x^{(i)} - \mu_k)^T \Sigma^{-1} ]= 0$
Rearranging and solving for $\Sigma$, we attain: 
$\hat \Sigma = \frac{1}{m} \Sigma^m_{i=1} (x^{(i)} - \mu_k) (x^{(i)} - \mu_k)^T \blacksquare$

#### Relation to Logistic Regression and Decision Boundary

If $\Sigma_{0} = \Sigma_{1} = \Sigma$, then there exists a linear decision boundary. This can be used during inference by plugging in x to $f(x) = w^{T}x + b$ where w and b are previously derived terms. If f(x) > 0, then y = 1 else y = 0. 

Assumptions: GDA makes a stronger assumption than logistic regression because it assumes that the underlying data distribution is Gaussian, so when this assumption holds then it learns much faster. However, this is usually not the case, and logistic regression is much more robust to incorrect assumptions. 

## Naive Bayes

Another generative model. Assumption: data is IID, meaning that likelihood is product of individual likelihoods. If we don't make this assumption, then there would n-1 params where n is the number of features (think bi-gram vs N-gram). This assumption is "naive" because for example, the presence of a word is assumed to be independent of every other word, which is obviously not true because words have context. 

Parameterized by: 
$\phi_{j|y=1}= p(x_{j}=1|y=1) =\frac{\sum\limits \mathbb{1}\{x_{j}=1 \land y=1\}}{\sum\limits \mathbb{1}\{y=1\}}$ and $\phi_{j|y=0}$. 

Prediction: 
![[img/Pasted image 20240304205511.png]]


## Support Vector Machine

Notation: 
- Classifier $g(z)$
- Hyperplane $h(x)$
- labels $y \in [-1,1]$
- $g(z) = 1$ if $z \geq 0$ else $g(z)=-1$ 
- $r$ is euclidean distance from data point x to x projected onto hyperplane 
- $\frac{w}{||w||}$ is unit vector orthogonal to hyperplane

Any data point x can be decomposed into: $$x = x_{p} + r\frac{w}{||w||}$$ This can be rearranged and rewritten as: $$r = \frac{h(x)}{||w||}$$
Classes are separated by two hyperplanes of margin $\rho$ where $w^{T}x+b \leq \rho, y=-1$ and $w^{T}x+b \geq \rho, y=1$. By definition, support vectors are data points that lie on the margin and thus satisfy equality. 

The distance from x to margin $\rho$ is $r(x) = \frac{y(w^{T}x+b)}{||w||}$ and margin $\rho = \frac{2}{||w||}$ 

#### Constraints: 
Since we want to maximize margin, then we want to minimize $||w||$. We also want to ensure correctness by placing the hyperplane between the classes. Given these two constraints, we attain the optimization problem: 
$$
\begin{align}
\min_{w,b}w^{T}w \\
(w^{T}x+b)y^{(i)}\geq 1, \forall i
\end{align}
$$
which can be solved with quadratic programming. The above is referred to as the primal form (cue monkey sounds). Through some fancy optimization math, we can derive the dual form: 
$$\max_{a^{(i)} \geq 0} \min_{w,b} \frac{1}{2} + \sum\limits_{i}\alpha^{(i)}\left[1-(w^{T}x^{(i)}+b )y^{(i)}\right]$$
We can leverage this formulation for ease of inference with simply an inner product of a query x and a matrix of support vectors. Given that: 
$$
\begin{align*}
\text{primal: } f(x) = g(w^{T}x + b) \\
\text{dual: } w = \sum\limits_{i\in SV} alpha^{(i)}y^{(i)}x^{(i)} \\
\text{substituting in } w \text{: } \\
f(x) &= g\left(\sum\limits_{i\in SV} alpha^{(i)}y^{(i)}\left<x^{(i)}, x\right>+b\right)
\end{align*}
$$

In code, this can be seen as: 
```python 
f(x) = alpha * y_SV @ X_SV @ x_new + b
```

#### Kernel Functions
Because of we can do prediction with an inner product, we can apply a kernel to lift data into higher dimensions that are more linearly separable. Some example kernel functions: 
- Polynomial: $K(x, x') = (\gamma x^{T}x + r)^{d}$
- Radial Basis Function (RBF): $K(x, x') = exp(-\gamma||x-x'||^{2})$
- Sigmoid: $K(x, x') = tanh(\gamma x^{T}x + r)$

The RBF $\gamma$ parameter when increased can cause overfitting since it allows each data point to have more "influence" and thus a tighter decision boundary that wraps around the classes' data points. 

#### Soft-Margin SVM
Since there may be outliers, we want to also quantify how much misclassification we're willing to tolerate. The optimization problem is then: 
$$
\begin{align}
\min_{w,b}w^{T}w + C\sum\limits_{i}\zeta^{(i)} \\
(w^{T}x+b)y^{(i)}\geq 1-\zeta^{(i)}, \forall i \\
\zeta^{(i)}\geq 0, \forall i
\end{align}
$$
Where: 
- $\zeta = 0$  when x is on correct side of hyperplane (y is correct)
- $0 < \zeta < 1$ when x is on correct side but in margins of hyperplane (y is correct)
- $\zeta > 1$ when x is on wrong side of hyperplane (y is incorrect)

**Influence of C:** 
C changes the tolerance of misclassifications. Increasing C thus increases the loss, which discourages misclassification more. Theoretically, setting C to infinity recovers hard-margin SVM since there is no tolerance for misclassification. 


## Regularization
- Ridge (L2) weight norm, causes irrelevant features to converge towards 0
- Lasso (L1) weight norm, causes sparsity by causing irrelevant features to eventually become 0. 

## Feature Selection

Pretty intuitive concepts: 
- Filter: use heuristic to score and select features with top-k scores. Example score function using mutual information $S(x) = I(x;y)$. 
- Wrapper: wraps around learning algorithm to evaluate how well it does using different feature subsets
	- Forward Search algorithm: Start with an empty set F. For $i \in \{i, \dots, n\}$, add new feat to F and train/test model. $F = F \cup i$ if best performing subset. Repeat and terminate when all features have gone through or you reach some threshold of n features. 
	- Backward Search: exact opposite, start with full feature set and remove until you hit empty set. 