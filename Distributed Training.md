
Training State: 
- Model Parameters
- Gradients
- Activations: intermediate results from forward phase required for backpropping gradient
- Optimizer State: maintains info about how params change over time (e.g. momentum), possibly the most expensive

Guiding Question: How to parallelize ML training? 

Goal: Scale data and model size

### Data Parallelism
1. Partition train data into batches
2. Compute gradients of each batch on GPU
3. Aggregate gradients across GPUs (all reduce)
![[Pasted image 20250125162242.png]]
n copies of model params, one on each GPU

### Model Parallelism
![[Pasted image 20250125162533.png]]

**Tensor MP**: ![[Pasted image 20250125162734.png]]
but is slow across inter-server communicate links (but fast within a server node)
Implementation: 
- Core building blocks are RowParallelLinear and ColumnParallelLinear

**Pipeline MP**
![[Pasted image 20250125162958.png]]
- Layers/operators in model sharded over GPUs
- Each batch split into small microbatches and execution pipelined across these microbatches
- P2P communication between consecutive pipeline stages
- Pipeline bubble at start and end of every batch is equal to (p-1) microbatches' forward and backward passes

Interleaved PP
![[Pasted image 20250125163308.png]]
We don't have 0 bubbles if we fully optimized bc we're bottlenecked by communication speeds
### Distributed optimizer

![[Pasted image 20250125162642.png]]
6120GB doesn't fit on one device so can distribute this

### Tradeoffs
- Each parallelism dimension has different limiting factors 
	- e.g. PP only scales up to number of model layers (need to combine parallelisms if scaling to 1000s of GPUs)
	- Naively combining parallelisms lead to poor throughput
		- TP communication can dominate (especially inter-node)
	- Using 3D parallelism efficiently is hard![[Pasted image 20250125163923.png]]
![[Pasted image 20250125164029.png]]

### General Rules of Thumb
- Keep it simple: just use DP + distributed optimizer
- if that OOMs -> TP + SP
- if that OOMs -> +PP

### Other Forms
- Long-context training: Context Parallelism. Intermediate activations of all layers sharded across multiple GPUs along sequence dim
- MoE: Expert parallelism
- FSDP: JIT gathering of model params when needed for forward and backward comptuations (recompute activations, don't store bc not compute blocked)