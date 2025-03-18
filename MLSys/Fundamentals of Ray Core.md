
### Tasks
Ray tasks are run in the Single Instruction Multiple Data (SIMD) paradigm where a single function (the instruction/task) is provided that's run in parallel across different data partitions on different workers. You specify that a function is a ray task by using the `@ray.remote` decorator. Using an example: 


Assuming we have a dataset and retrieval of an element is delayed by the index position

```python
import ray 
import time

database = ["some", "words", "as", "an", "example"]

def retrieve(idx): 
	time.sleep(idx / 10)
	return idx, database[idx]
```

Baseline (running sequentially): (0+1+2+3+4+5+6+7)/10 = 2.8 seconds ideal
```python
data = [retrieve(idx) for idx in range(8)]
```

Using Ray Tasks: should be bottlenecked by longest task = 7/10 = 0.7 seconds ideal
```python
@ray.remote
def retrieve_task(idx): 
	return retrieve(idx)

obj_refs = [retrieve_tasks.remote(idx) for idx in range(8)]
data = ray.get(obj_refs)
```

Assuming a separate actor for each task (8 total), we are bottlenecked by the longest running task, which is 0.7 seconds. 

To turn a function into a ray task and execute it, there are 3 steps: 
1. Turn the function into a ray task using the ray.remote decorator
2. Call the task.remote() which returns a list of object references as future promises for execution
3. Execute using ray.get() to run distributed process and materialize results

### Ray Object Store
In the previous example, we used a local in-memory database on the driver but how do we scale this to a distributed system? Ray natively takes care of this using a distributed database called the "Object Store". We change the code as follows: 
```python 

# modify task header to include object store db
@ray.remote
def retrieve_task(idx, db): 
	time.sleep(idx / 10)
	return idx, db[idx]

# put the local database into Ray's object store
db_obj_ref = ray.put(database)
obj_refs = [retrieve_task.remote(idx, db_obj_ref) for idx in range(8)]
data = ray.get(obj_refs)
```
Note that this code ran in ~7 seconds due to overhead with creating the object store but this cost amortizes (and is worth it) at scale

### Non-Blocking Calls
So far, we have been bottlenecked by the slowest running task. It would be nice if we could access the outputs of shorter tasks that have already finished. We can do this using `ray.wait()`

Instead of using `ray.get(obj_refs)`, we can run the following instead: 
```python
# using non-blocking calls
start = time.time()
object_refs = [retrieve_task.remote(idx, db_obj_ref) for idx in range(8)]
all_data = []

# run until there's no more object_refs left
while len(object_refs) > 0:
	finished, object_refs = ray.wait(
		object_refs, num_returns=2, timeout=7.0
	)
	data = ray.get(finished) # TODO (haoli): figure out ray internals schedules this bc it's not directly intuitive how this code works
	print_runtime(data, start)
	all_data.extend(data)

print_runtime(all_data, start)
```
The result of early returns is that we can start further processing in the next step as results are streamed out. If we wanted to do this practically though, ray will automatically schedule a DAG with multiple tasks internally as shown below.

#