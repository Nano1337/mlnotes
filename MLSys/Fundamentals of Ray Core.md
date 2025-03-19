
### Ray Core API - TL;DR
- ray.init(): Initializes your Ray cluster. Pass in an address to connect to an existing cluster.
- @ray.remote: Turns functions into tasks and classes into actors.
- ray.put(): Puts values into Ray’s object store.
- ray.get(): Gets valuyes from the object store. Returns the values you’ve put there or that were computed by a task or actor.
- .remote(): Runs actor methods or tasks on your Ray cluster and is used to instantiate actors.
- ray.wait(): Returns two lists of object references, one with finished tasks we’re waiting for and one with unfinished tasks. 
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

### Task Dependencies
what if we had follow-up tasks? It's easier to understand via example: 
```python
@ray.remote
def followup_task(retrieve_result, db):
    """
    ray.get() is internally called before this task is executed so retrieve_result is     materialized
    in other words, retrieve_result passed in is a tuple here, not an object reference
    """
    idx, _ = retrieve_result
    followup_result = db[idx+1]
    return retrieve_result, followup_result

start = time.time()
retrieve_refs = [retrieve_task.remote(idx, db_obj_ref) for idx in [0, 2, 4, 6]]
followup_refs = [followup_task.remote(ref, db_obj_ref) for ref in retrieve_refs]
result = ray.get(followup_refs)
print_runtime(result, start)
```

User-facing code: object references are passed between tasks in a friendly and familiar interface. `ray.get()` is called in the same way at the end

Internally: execute the object ref of the first task by calling `ray.get()`internally and then passing the reference to that output to the second task in materialized form to do further processing. This way, there is minimal overhead so that we don't have to pass the materialized result back to the driver and to the worker again. This is what it looks like inside: 
![[Pasted image 20250318211143.png]]

### Ray Actors
How do we make tasks stateful? Ray uses a pythonic interface to make python classes into actors using the same `ray.remote` decorator

```python 

# define Actor
@ray.remote
class TaskCounter: 
    def __init__(self): 
        self._count = 0

    def increment(self): 
        self._count += 1

    def get_count(self): 
        return self._count

# create ray task to increment when data is retrieved
@ray.remote
def retrieve_and_increment_task(idx, taskcounter, db): 
    time.sleep(idx / 10)
    output = db[idx]
    taskcounter.increment.remote()
    return idx, output

start = time.time()
taskcounter = TaskCounter.remote() # init Actor
retrieve_refs = [retrieve_and_increment_task.remote(idx, taskcounter, db_obj_ref) for idx in range(8)]
results = ray.get(retrieve_refs)
print_runtime(results, start)
print(ray.get(taskcounter.get_count.remote())) # actor function call --> remote
```
Notes: 
- We initialize the actor by calling ActorClass.remote()
- Anytime an actor or task remote function is called, we can't use a direct call but have to call `function.remote()` 
