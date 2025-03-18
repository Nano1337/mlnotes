
### ML Framework Debate
Eager mode: 
- preferred by users
- easier to use programming model
- easy to debug
- pytorch mainly eager mode
Graph mode: 
- preferred by backends and framework builders
- easier to optimize with compiler
- easier to do automated transformations

Pytorch's attempts at graph modes: 
- Torch.jit.trace: 
	- record + replay
	- unsound
	- can give incorrect results bc it ignores python part of program
- torch.jit.script: 
	- aot parses python into graph format
	- only works on ~45% of real world models
	- high effort to "TorchScript" models
- Lazy Tensors (Torch XLA)
	- graph captures through deferred execution 
	- high ovehead
	- perf cliffs

Thus, pytorch models aren't static graphs: 
- convert tensors native python types (item, tolist, int)
- use other frameworks (numpy, xarray) for part of model
- data dependent python control flow or other dynamism
- exceptions, closures, generators, classes

