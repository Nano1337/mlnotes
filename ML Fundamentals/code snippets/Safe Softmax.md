
We use the LogSumExp trick to ensure numerical stability when performing exponentials for softmax. 

Softmax is used for normalizing logits so they all sum to 1 (addition property of probability)

Three steps: 
1. Find the maximum
2. Exp (elem - max) and Sum
3. Divide each exp(elem-max) by sumexp

Let $a = [1,1,1,1]$

Ground truth: torch.nn.functional.softmax
```python
import torch
import torch.nn.functional as F

a = [1.0,1.0,1.0,1.0]
a = torch.tensor(a)

print(F.softmax(a, dim=0))
```

3 For loop version: 
```python 
from math import exp

a = [1, 1, 1, 1]

maximum = float("-inf")
# find max element
for elem in a: 
	maximum = max(elem, maximum)

# subtract and exp
l = 0
for i in range(len(a)): 
	a[i] = exp(a[i]-maximum)
	l += a[i]

for i in range(len(a)): 
	a[i] /= l

print(a)
```

Fused max and exp-sum (first two loops -> 1 loop)
```python
from math import exp

a = [1.0, 1.0, 1.0, 1.0]
maximum = float("-inf")
l = 0

for i, elem in enumerate(a):
	
	# find local max
	prev_max = maximum
	maximum = max(elem, maximum)
	
	if maximum > prev_max:
		# If we found a new maximum, rescale previous terms
		scale = exp(prev_max - maximum)
		l *= scale
		for j in range(i):
			a[j] *= scale
	
	# exp-sum current element
	exp_term = exp(elem - maximum)
	a[i] = exp_term
	l += exp_term

# divide by exp-sum
for i in range(len(a)):
	a[i] /= l

print(a)
```