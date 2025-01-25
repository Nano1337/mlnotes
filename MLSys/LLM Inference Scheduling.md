
### Basic SLOs: 
- Time To First Token (TTFT): initial response time, prefill stage
- Time Per Output Token (TPOT): decode stage

#### Continuous Batching: 
- Integrate new requests as old requests rotate (normal batching strategy bottlenecked by longest output)
- Need to perform prefill for new request, which results in stall for existing requests
- Two queues in system: 
	- prefill requests waiting to be integrated
	- ongoing decode requests
- Scheduling types: ![[Pasted image 20250124100120.png]]
	- Prefill prioritizing (vLLM): make existing requests wait so batch decode can happen
		- Paged Attention
	- Decode prioritizing (FasterTransformer): finish existing prefilled requests and decode as faster as possible but this results in low batch size and poor throughput. I'd assume this would lead to low GPU memory utilization bc less batch processing
		- Iteration level batching
- ![[Pasted image 20250124100332.png]]
- Solution: Mixed Batching
	- Fused computation of prefill and decodes but challenge is naively combining prefill/decode ops leads to latency increase
	  ![[Pasted image 20250124103914.png]]
- Stall-free batching: 
	- split large prefill into to smaller chunks, just enough to consume the leftover compute budget in decode batches![[Pasted image 20250124104140.png]]
- 3D parallelism (PP, TP, DP) affects the 2 stages: ![[Pasted image 20250124112613.png]]

## DistServe: 
- Disaggregating Prefill and Decode (decouple and put on separate workers)
- Additional time added in networking bandwidth migrating from prefill worker to decode worker![[Pasted image 20250124112732.png]]
- Potential issues of disaggregation: 
	- overhead of comms
	- machine failure

