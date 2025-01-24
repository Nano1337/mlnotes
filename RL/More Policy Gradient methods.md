
#### Deterministic Policy Gradient: 
- simpler to compute -> more efficient
- extend PG thm to deterministic policies -> update policy in direction of action-value fxn's gradient
- Use deterministic PG to outperform stochastic policy in high dim cont action spaces
- Critic: learned using bellman like in Q-learning
- Actor: updated by applying chain rule to expected return from start dist w.r.t. actor params 
- Implementation: 
	- problem: large discrete action space or cont action space, can't plug in every possible action to find optimal action
	- soln: learn fxn approx for argmax, via gradietn descent
	- guarantee exploration: on-policy = add noise to policy actions, off-policy = follow different stochastic behavior policy to collect samples

### Deep Q Networks: 
- state representation: feature space
- policy represented as DNN
- experience replay: store agent's experiences of S,A,R and next state tuples in replay memory buffer
- Q-learning update: sample mini-batches of experiences from replay memory to update NN weights
- use greedy method to encourage exploration
- Target network: separate target network with same arch as main network to stabilize learning process, periodically update target network by copying weights from main network
- Challenge: IID sample assumption for gradient descent -> use experience replay buffer

### Deep Deterministic Policy Gradient: 
- Extend DQN from discrete action space to continuous action space
- model-free off-policy RL for learning continuous actions
- actor-critic approach based on DPG
- Critic: learned as DQN (has replay buffer and target Q-network lagging main Q-network)
- Actor: for policy gradient

### Trust Region Policy Optimization (TRPO) algo
- Error defines trust region = stay within error bounds and convergence is guaranteed
- purpose: optimize policy to maximize rewards without falling off cliff of instability 
- Minorize-Maximization Algo: 
	- idea: iterative max lower bound function M, approx expected reward locally 
	- make init policy guess, find lower bound M for using policy 
	- optimize M and use optimal policy for M as next guess + repeat
	- Condition that M must be easier to optimize (i.e. convex function)
- Cons: requires 2nd order derivative + inverse = computationally infeasible (Fisher Matrix)


### Proximal Policy Optimization (PPO)
- Formulated as constrained (trust-region) optimization where Fisher matrix from TRPO is replaced with KL term
  $$
    

\mathcal{L}{\text{PPO}}(\theta) = \mathbb{E}t \left[ r_t(\theta) A_t \right] - \beta \cdot \text{KL} \left[ \pi\theta(\cdot | s_t) \, \| \, \pi{\theta_{\text{old}}}(\cdot | s_t) \right]
  $$
	 Where: 
	 - $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$  is policy ratio
	 - $A_{t}$ is advantage estimate
	 - KL between new and old policies (constrained)
	 - $\beta$ is weight for KL term
- can also add Clipped objective
	- clipped version avoids excessive updates by limiting policy ratio to range $[1-\epsilon, 1+\epsilon]$

