
##### Deadly Triad: 
- Function Approximation: significantly generalizing from large number of examples
- Bootstrapping: 
	- learning valu estimates from other value estimates (e.g. DP, TD)
	- Faster convergence
- Off-policy: learning about policy from data not from that policy (e.g. Q learning) where we learn about greedy policy from data with more exploratory policy (i.e. behavioral policy)

Having 2 of 3 is fine but all 3 can lead to collapse

#### Types of RL: 
- Value-Based: 
	- learned Value fxn
	- implicit policy (e.g. greedy)
- Policy-Based: 
	- no value fxn 
	- learned policy
- Actor-Critic: 
	- Learned value fxn
	- learned policy

### Policy Gradient
Basics: 
- Effective in continuous actions spaces + learning stochastic policies
- approx change to policy that results in highest increase in expected return
- stochastic estimate = approx gradient
Cons: 
- Nonlinear optimization, my fall into local optima
- Overall computationally more expensive
Policy is a parameterized function: 
- needs continuous, differentiable (soft) policy, and softmax is useful
- e.g. DNN with weights and biases
Training: 
- Have Policy objective function
- gradient of policy objective = policy gradient
- Take current game state and calc probability of taking any allowed actions, random init, collect episode, improve policy with backprop (favor actions with positive value estimate reward), introduce noise (sample from agent's action dist) for exploration
- Alternative is evolutionary strat: guess and check process. Take set of weights and jitter with Gaussian noise -> eval each set of weights -> reward -> update param vector as weight sum of all weight vectors = estimating gradient of expected reward
	- no need for backprop, highly parallelizable, high robustness, credit assignment over long timescales buuuuut is not sample efficient
Policy Gradient Thm: 
- The policy is $\pi_{\theta}(a|s)$ 
- The objective function $J(\theta) = \mathbb{E}_{\pi\theta}[\sum\limits_{t=0}^{T}r_{t}]$ 
- Gradient of objective expressed as: $$
  \nabla_{\theta}J(\theta) = \mathbb{E}_{\pi\theta}[\nabla_\theta\log\pi_{\theta}(a|s)G_{t}]
  $$ where $G_{t}$ is the total reward from time t. Use the policy gradient for weight updates (gradient descent)

#### Reinforce Algorithm: 
- Straightforward impl of policy gradient where $G_{t}= \sum\limits_{t'=t}^{T}r_{t'}$ but suffers from high variance
- Baseline Subtraction: 
	- $$ \nabla_\theta J(\theta) = \mathbb{E}{\pi\theta} [ \nabla_\theta \log \pi_\theta(a|s) (G_t - b(s))]$$
		- reduces variance by subtracting baseline b(s), often the value fxn, from rewards
		- b(s) can also be learned alongside target policy
	- Pros: approx optimal policy directly is more accurate and faster to converge compared to value-based approaches
	- Cons: using estimator is noisy (high variance) although unbiased
- Solution: add a critic as a learned baseline to estimate value function

#### Purpose of Critic: 
- approx state-action/advantage values
- trained via bootstrapping and criticizes action chosen by actor
- Note: REINFORCE (+baseline) isn't actor-critic bc state-value fxn isn't learned and doesn't use bootstrapping (instead uses high-variance MC updates)
- Pro: reinforce uses mc sampling, which requires entire episode but policy gradient can learn with TD methods so advantageous
- Think: actor-critic is derivative of policy iteration, where critic is evaluation and actor is improvement. Critic = value estimation, Actor = policy

#### More on Actor-Critic methods: 
- One-step AC: fully online, incremental algo with S,A,R processed as they occur and never revisited
- Challenges: 
	- Stability -> divergence problem: 
		- (A2C, A3C) -> efficient exploitation of hardware resources
		- A2C: Advantage Actor Critic
		- A3C: Async Advantage Actor Critic
		- Deterministic PG: extension of PG thm
		- Deep Q-Networks
	- Scalability to high dim continuous action/state spaces: 
		- DDPG = DQN + DPG

#### A3C: 
- Main Idea: exploit parallel computing resources by executing set of agents interacting with different versions of environment
- may use different exploration strategies in each actor-learner to max diversity of data
- value fxn is learned by critics based on multiple actors trained in parallel
- actors sync with global parameters periodically
- gradient accum from each thread results in correcting global values by a little bit in direction of each training thread independently 

#### A2C
- Synchronous, deterministic version fo A3C
- problem of a3c: all reduce not optimal bc change that some thread specific agents end up using very different policies 
- solution: coordinator is used, waits for all parallel actors to finish before updating global params then parallel actors start from same policy in next step (which also allows to use GPU more efficiently)
