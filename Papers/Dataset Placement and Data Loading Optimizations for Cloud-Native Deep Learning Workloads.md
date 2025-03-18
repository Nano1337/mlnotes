
## Motivation
- Training is memory (I/O) bound, not so much compute bound given the accelerators we have available today
- However, models must be trained with large datasets that can't fit in RAM, sometimes disk like network-based file system is common solution to eliminate centralized disk space and reduce storage costs but these distributed file systems aren't design to handle DL datasets
- User needs to define ad hoc dataset placement and eviction decisions (streaming) based on DL workload profile
- Cloud bucket storage cost is based on usage of storage space, API requests, and data transfer - DL workloads calls API every time to access data element which can get very expensive. 

### Solution 
DLCache: 
- cloud buckets used for persistent storage and cluster for training
- 3-tier storage system and orchestrates data services like dataset prep, placement, and cost-aware eviction of least frequently used data in cache
- designed K8s Custom Resource Definition controller and operator to streamline cloud services and execute in special K8s pod called DLTpod
- dynamically loads data from NFS -> in-memory TMPFS
- dynamically adjusts pytorch dataloader num_workers based on real-time train perf

### Background
DLT: Deep Learning Training

**PyTorch Dataloader**: 
- Main Process (MP) starts and creates `index_queue` with a list of indices for each batch
- Each batch of indices is sent to a worker process $i$ that has a copy of the PyTorch Dataset object. These indices are fetched from `__getitem__` and batched together through the `collate_fn`. In other words, $WP_{i}$ pops from the `index_queue` and asks data fetcher to attain batch via iterating its copy of the Dataset object
- This data is then put into the MP's `data_queue`. If `pin_memory` is enabled (which allows faster loading onto GPU), then the `pin_thread` pops data from `data_queue` and copies tensor to pin memory. 
	- Otherwise, MP pops data from `data_queue` and sends to trainer
- `prefetch_factor` determines how many mini-batches can be preloaded into memory before the start of the epoch

### K8s Operators
- Custom Resource Definition is an API template to store a collection of objects of some kind. In this case, it stores the template for the state (that needs to be instantiated) of the train job 
- The Operator Controller is the orchestrator routine that watches for events like create/update/delete on the train job custom resource and updates the state
- the purpose is so user can focus on application without having to worry about underlying support functions

## DL Cache System Design
![[Pasted image 20250209235633.png]]
Four layers: 
- 