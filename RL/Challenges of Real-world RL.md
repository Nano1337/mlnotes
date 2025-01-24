
### Sample Efficient Learning: 
- MAML meta learning
- Model-based RL: learn ensembles of transition models and use various sampling strategies from those models to drive exploration
- Transfer RL 

### No interactions allowed
- offline RL has problem of distribution shift
- approach: penalize OOD actions but offline evaluation can lead to high variance

### High dim continuous state/action spaces
- e.g. recsys, sensors
- Action-space: intermediate step to generate candidate actions and discard irrelevant action
- Space-space: feature selection and feature learning, lower dim projection

### Reasoning about system constraints that can't be violated
- Examples: 
	- system: limiting system temp, contact forces, min battery levels
	- env: avoid dynamic obstacles, limiting end effector velocity
- Formalized through constrained MDP

### Safety constraints: 
- put constraints in simulators
- use predictive models for safety eval
- risk-aversion RL to decide whether risky move is worth it

### Partial Observability
- Almost all real systems are partially observable: 
	- physical sys don't have observation of how params change across time, i.e. wear and tear on motors or joints of robots
	- recsys no observations of mental state of users
- formalized through POMDP

### Non-Stationarity: 
- system dynamics change with time
- formalized through non-stationary MDP or robust MDP

### Multi-objective, poorly specified rewards
- reward fxn engineering is hard
- multiple objective balances multiple sub goals

### Explainability: 
- enough said

### Real-time RL: 
- ms for recsys or control of robot
- computation time vs slow learning trade-off


### System Delays: 
- sparse/delayed rewards bc real systems have delays in sensation of state, actuators, or reward feedback