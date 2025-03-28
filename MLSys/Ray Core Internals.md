Overall system internals:
![[Pasted image 20250319110416.png]]

### Worker Node
- Identified by unique ID, IP address, and port
- Execute tasks 
- Raylets: brain of a worker - contains a task scheduler and distributed object store
- The distributed object store cumulatively makes up shared memory but since it's distributed, not all objects are available locally for a worker -> data transfer may need to happen - this logic is written in Plasma. 
- **Scheduling**: 
	- scheduler is aware of worker's available resources - memory, CPU, GPU 
	- if there aren't sufficient resources, task won't be scheduled until there are and also limits number of concurrent tasks to prevent resource overuse
	- Requirements for scheduling: 
	  1. resource demands are satisfied
	  2. object dependencies are resolved
	  3. worker identified for task

**Fault Tolerance and Ownership**
How does Ray take care of task failures and redo logic? 
- Workers keep track of all metadata and object references for task dependencies -> maintain start state of task so that if task fails it can be instantly restarted since all information is available. 
- Each worker maintains an ownership table where they hold this task metadata necessary to recompute a task

Example of ownership table given code: 
```python
@ray.remote
def task_owned():
    return

@ray.remote
def task(dependency):
    res_owned = task_owned.remote()
    return

val = ray.put("value")
res = task.remote(dependency=val)
```
- Main program owns val, task, res. 
- Task owns res_owned and task_owned
- res_owned depends on task_owned
- task depends on val

### Head Node
- is a worker node + Driver + Global Control Store (GCS)
- the worker allows for single node clusters (like in local execution)
- Driver: 
	- contains autoscaler to manage cluster
	- can submit tasks but doesn't execute them
- GCS: 
	- KV store that contains system-level metadata and locations of ray actors
	- sends and receives heartbeats from raylets to check that they're alive
	- Ownership model ensures that object information is stored at worker node level, preventing GCS from becoming a comms bottleneck at scale

### Distributed Scheduling and Execution
TODO