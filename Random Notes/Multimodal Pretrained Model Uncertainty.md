
- Let $k$ represent the $k$-th modality
- Let $G_k(\cdot)$ represent the $k$-th unimodal embedding

Class-Conditional Embeddings: 
$$
\begin{align}
P(G_{k(x)} | y = y_{i}) = \mathcal{N}(G_{k}(x) | \mu_{k, y_{i}}, \Sigma_{k}) \\
\mu_{k, y_{i}} = \frac{1}{N_{y_{i}}} \sum\limits_{i}^{N_{y_{i}}}G_{k}(x_{i, k}) \\ 
\Sigma_{k} = \frac{1}{N}\sum\limits_{y_{i}}\sum\limits_{i: y=y_{i}}(G_{k}(x_{i, k}) - \mu_{k})(G_{k}(x_{i, k}) - \mu_{k})^{\top}
\end{align}
$$

Class-Agnostic Embeddings: 
$$
\begin{align}
P(G_{k(x)}) = \mathcal{N}(G_{k}(x) | \mu_{k, agn}, \Sigma_{k, agn}) \\
\mu_{k, agn} = \frac{1}{N} \sum\limits_{i}^{N}G_{k}(x_{i, k}) \\ 
\Sigma_{k,agn} = \frac{1}{N}\sum\limits_{i}^{N}(G_{k}(x_{i, k}) - \mu_{k,agn})(G_{k}(x_{i, k}) - \mu_{k,agn})^{\top}
\end{align}
$$

Relative Mahalanobis Distance (RMD): 
$$
\begin{align}
RM_{k}(x_{i, k}, y_{i}) = M_{k}(x_{i, k}, y_{i}) - M_{k, agn}(x_{i, k}) \\ 
M_{k}(x_{i,k}, y_{i}) = -(G(x_{i, k}) - \mu_{y_{i}})^{\top}\Sigma^{-1}(G(x_{i, k}) - \mu_{y_{i}}) \\
M_{k, agn}(x_{i, k}) = -(G(x_{i, k}) - \mu_{k,agn})^{\top} \Sigma^{-1}(G(x_{i, k}) - \mu_{k,agn})
\end{align}
$$

For the per-sample, per-modality confidence scores: 
$$
s_{k}(x_{i,k}, y_{i}) = \frac{exp(RM_{k}(x_{i,k}, y_{i})/T)}{\max_{i}\{exp(RM_{k}(x_{i,k}, y_{i})/T\}+ \epsilon}
$$

Then, calibrate with: 
$$
s_{k,cal}(x_{i,k}, y_{i}) = \frac{s_{k}(x_{i,k}, y_{i})}{\max_{k}\{s_{k}(x_{i,k}, y_{i})\}}
$$

Then, the per-sample regularization term would be: 
$$
\mathcal{L}_{reg} = \mathbb{E}[-\alpha\sum\limits_{k}s_{k,cal}(x_{i,k}, y_{i})\mathcal{H}[f_{\theta,k}(x_{i,k})]] 
$$
where $\mathcal{H}[\cdot]$ is the entropy of the of $k$-th unimodal output embedding

Method to use instead to calculate $s_{k,cal}(x_{i,k}, y_{i})$ since the method above using RMD is so numerically unstable: 

Using K-means, we can use n_classes of clusters (because of classification task)

After finding clusters, we can calculate distance to nearest centroid for each data point. 
