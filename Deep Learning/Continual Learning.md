
Data is not assumed to be IID and distribution will shift over time so need continual training. 

Terminology: 
- Transfer: from task A to B -> improved perf in B after learning A (gradient aligned, cosim >0)
- Interference: negative transfer (gradients cancel out, cosim < 0)

Rehearsal based solutions: 
- make data close to IID by replaying from old task while learning new task, ideally we'd like to sub losses from each task together. Ideas: 1) learn coreset of data from task A and use that for rehearsal, 2) learn generative model for task A and generate samples for rehearsal 

Overcoming Forgetting by Preserving Latent Distribution: 
- Elastic Weight Consolidation (Metaplastic Networks): $\hat L_{B}(\theta) = L_{B}(\theta) + \lambda \mathbb{E}[d(f(x; \theta), f(x; \theta_{A}^{*}))]$. Assume that $\Delta \theta = \theta - \theta_{A}^{*}$ is small; we can write second order Taylor expansion of regularization term: $d(f(x; \theta), f(x; \theta_{A}^{*})) = d(f_{\theta_{A}^{*}}, f_{\theta_{A}^{*}}) + \Delta \theta^{T}\nabla d(f_{\theta_{A}^{*}}, f_{\theta_{A}^{*}}) + \frac{1}{2} \Delta \theta^{T}Hd(f_{\theta_{A}^{*}}, f_{\theta_{A}^{*}})\Delta \theta$. First two terms become 0 so we have $\hat L_{B}(\theta) = L_{B}(\theta) + \lambda \mathbb{E}[Hd(f_{\theta_{A}^{*}}, f_{\theta_{A}^{*}})\Delta \theta]$. EWC uses KLD for d() when f is a pmf (otherwise use Wasserstein, Sliced-Cramer, or MMD) and assumes that H is diagonal so $\hat L_{B}(\theta) = L_{B}(\theta) + \lambda \sum\limits_{i=1}^{nparam}H_{ii}([\theta]_{i} - [\theta_{A}^{*}]_{i})^{2}$, where $H_{ii}$ is importance weight of each parameter and summation is independent of data from task A.  The second term can be seen as regularization, penalizing the change in "important" parameters or you could project the gradient into a subspace to ensure no interference with prior tasks. 
- Gradient Projection: assumptions: activation subspace is low-rank; otherwise, projection into null space wouldn't lead to informative updates (can't learn new tasks efficiently). There isn't backward transfer. Solution: sparsity leads to low-rank activation subspace. Instead of training dense network with ReLU, train sparse with k-Winner take all activation (set not top-k activations to 0). 
- Non-overlapping subnetworks are highly effective in overcoming forgetting. Keep track of activation freq for each neuron and down-modulate frequently activation neurons. Mechanism: heterogenous dropout. 

Side note: isn't the non-overlapping subnetworks just an ensemble lol? 
