
### Motivation:
Batching multiple requests together is the solution for better throughput but is limited by inefficient memory management for KV Cache

### Problems: 
- KV cache grows and shrinks. 
- It's HUGE: each token ~1MB, one full request ~several GBs

Previous solutions: 
- Pre-allocate contiguous memory to request's max length
	- useful in traditional DL workloads where I/O shapes are static (e.g. fast pointer arithmetic, efficient memory access)
- Internal Fragmentation: underutilization of preallocated memory
- External Fragmentation: non-uniform per-request max lengths
- Leads to significant memory waste in KV Cache space so only ~20-40% KV cache utilized to store actual token states

### PagedAttention (vLLM)
![[Pasted image 20250125153451.png]]
- KV block is fixed-size contiguous chunk of memory that can store KV token states from left to right
	- e.g. block size=4 (one block contains 4 tokens)
![[Pasted image 20250125154825.png]]
- Logical KV block is contiguous but use virtual table to map to physical memory
- Memory efficiency: 
	- minimal internal fragmentation
	- only happens at last block of sequence
	- num wasted tokens / seq < block size
	- no external fragmentation![[Pasted image 20250125155425.png]]
- How do you choose block size?![[Pasted image 20250125155455.png]]
- Another advantage: shared physical memory in parallel sampling: ![[Pasted image 20250125155733.png]]
	- Can use this for beam search as well