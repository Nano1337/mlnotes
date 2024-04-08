
### PCA: Principal Component Analysis

Given $x^{(i)}, \dots x^{(m)} \in \mathbb{R^{n}}$ , we want to reduce to $z^{(i)}, \dots z^{(m)} \in \mathbb{R^{k}}$ where $k < n$ . 
For example, if we have a strongly positively correlated points on a 2D scatterplot, we can project those points onto a 1D line of best fit to capture most of information (good approximation)

**General Steps**: 
1. Preprocessing by normalizing **features** to mean 0 and std 1. 
2. Find direction of maximum variance 

#### Step 1: Preprocessing
We want to z-score normalize across the feature dimension to achieve mean 0 and std 1 for each feature across all samples since we want to move all the data to sit on the unit hypersphere around the origin. This makes our life easier in future calculations

#### Step 2: Find Direction of Maximum Variance

Consider the projection of a point onto a unit vector $u$. This is notated as $proj_{u}x^{(i)} = x^{(i)T} u$ where $||u||=1$

Now the question is: how do we find the unit vector $u$ such that we optimally rotate it to best fit the data? The usual solution is to simply do least squares and minimize distance of a point to its projection on the line, but the better solution for optimization is actually maximizing distance from the projected point to the origin. See below for the intuition: 
![[img/Pasted image 20240407004648.png]]
From the picture, minimizing b would be equivalent to maximizing c. 

We would like to choose a unit vector $u$ to maximize: 
$$\frac{1}{m}\sum\limits_{i=1}^{m}(x^{(i)T}u) = \frac{1}{m}\sum\limits_{i=1}^{m}u^{T}x^{i}x^{iT}u = u^{T}(\frac{1}{m}\sum\limits_{i=1}^{m}x^{i}x^{i})u = u^{T}\Sigma u$$
where $\Sigma$ is the covariance of the data (assuming 0 mean). This form can be achieved through singular value decomposition (SVD), which has the form: $U\Sigma V$ where $\Sigma$ is the diagonal of eigenvalues and $V$ are the left singular vectors, or feature eigenvectors in our case. Note that we don't use the left singular vectors $U$ in this case since we aren't trying to capture the maximum degrees of variance between samples but rather between features within a sample. 

#### Optimization Problem

$$
\begin{align}
\text{max } u^{T}\Sigma u \\
\text{constraint } u^{T}u = ||u|| = 1 \\
\text{Lagrange multiplier }\mathcal{L}(u, \lambda) = u^{T}\Sigma u - \lambda (u^{T}u-1) \\
\nabla_{u}\mathcal{L}(u, \lambda) = \Sigma u - \lambda u = 0 \\
\Sigma u = \lambda u
\end{align}
$$
Where $\lambda$ is the eigenvalue and $u$ is the eigenvector!. Thus, $u$ is the first eigenvector of $\Sigma$ associated with the largest eigenvalue $\lambda$. 

#### Calculating Principal Coordinate
$z^{(i)} = \left[x^{(i)T}u_{1} \text{ } x^{(i)T}u_{2} \dots x^{(i)T}u_{k}\right]^{T}$ for the top-k principal components where $x^{(i)}$ is a $nx1$ vector and each eigenvector $u_{k}$ is also an $nx1$ vector. Using the top-k eigenvectors would give us a scalar value for each dot product and thus $kx1$ "principal coordinate" result. 

#### Reconstruction and Total Variance

The variance explained by a single principal component is the eigenvalue $\lambda$ divided by the sum of all eigenvalues. Extending that, the variance explained by the top-k principal components is summing the top-k eigenvalues /  sum all eigenvalues. 

We can use a **Scree plot** (bar chart) to visualize the sorted eigenvalues in decreasing order. 

We can then reconstruction/approximate the original data point $x^{(i)}$ using k principal components since the top-k principal components should capture most of the variance. 
$$x^{(i)}_{approx} = (x^{(i)T}u_{1})u_{1}+(x^{(i)T}u_{1})u_{1}+\dots+(x^{(i)T}u_{k})u_{k}$$
The approximation/reconstruction error would then be: 
$$\frac{\sum\limits_{i=1}^{m}||x^{i} - x^{i}_{approx} ||^{2} }{\sum\limits_{i=1}^{m} ||x^{i}||^{2}}$$
