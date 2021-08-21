# Scheduler

Scheduler is the component that makes the whole pipeline run. In the introduction we showed how to use a scheduler to run a pipeline. This section will go over the same concepts in more detail.

To recap, this is the code to run a pipeline:

```cpp
// create a scheduler
noarr::pipelines::DebuggingScheduler scheduler;

// register all pipeline nodes
scheduler.add(my_hub);
scheduler.add(my_node);
scheduler.add(my_other_hub);
scheduler.add(my_other_node);

// run the pipeline to completion
scheduler.run();
```


## External API

The external API of a scheduler is very simple:

- the constructor
- the `add` method
- the `run` method

When writing your own scheduler, you can use the constructor to pass in any parameters you need. This is nothing unusal, the `DebuggingScheduler`, for example, optionally accepts an output stream to print a log to.

The `add` method accepts a `Node&` reference and remembers it internally to use it during scheduling. The order in which a user registers nodes has only impact on the order in which `initialize` and `terminate` event methods are called on nodes. The first-registered nodes have their event methods executed first. It may or may not have impact on the scheduling algorithm and the user should not rely on it. The `DebuggingScheduler` does use the order for ordering nodes in one scheduling generation, but other schedulers will parallelize the execution and the order will become non-deterministic.

The `run` method blocks, and executes the pipeline to completion. The stopping condition is that all nodes are *idle* and all return `false` from their `can_advance` method. If you incorrectly setup your `can_advance` conditions (or you do not provide the correct number of envelopes in hubs), the pipeline usually enters the stopping condition without having computed anything interesting. The scheduler has no way of detecting this issue. It may also happen that an incorrectly setup pipeline will run indefinitely. Again, the scheduler cannot have an upper limit on the invocation count and so cannot detect this situation.

It is advisable that you run your pipeline only once. If you want to run it multiple times, it is better to destroy the pipeline and create a new one. The reason being that a terminated pipeline might be in a different state than the initial state before the exection. Therefore running it a second time might cause unexpected issues. That being said, there is nothing preventing you from calling `run` multiple times. If you make sure your hubs are in the right state after termination and your own added logic is also consistent, you can easily reuse the pipeline. You would save time on envelope allocation.

The `add` and `run` methods are not part of any interface. Should the framework be extended by adding new schedulers, it is advisable to extract a `Scheduler` interface and make all other schedulers inherit from it. We did not do that yet, because the `DebuggingScheduler` is so far the only scheduler available.


## Internal API

By internal API we mean the methods of a `Node` that are called by the scheduler, and the scheduling behaviour.

The `Node` class exposes four methods that are called by a scheduler:

- `scheduler_initialize()`: Called before the pipeline starts running and it directly calls the `initialize` event method.
- `scheduler_update(void(bool) callback)`: Called periodically by the scheduler to advance data. It acts as the `can_advance` and `advance` method fused together. It may run asynchronously or it may call the callback even before it finishes (may run synchronously). Either way, it is guaranteed to call the callback when it finishes. The callback has one boolean argument that states, whether data was advanced or not. The method should be understood as *an attempt to advance the data through the node asynchronously* and instead of returning a boolean, it calls back with a boolean, stating whether the advancement was successful or not.
- `scheduler_post_update(bool data_was_advanced)`: This method should be always called by the scheduler after the `scheduler_update` method finishes. It receives the result of the previous method as an argument.
- `scheduler_terminate()`: Called after the pipeline stops running and it directly calls the `terminate` event method.

The scheduling logic could be summarized in pseudocode like this:

```txt
1) call scheduler_initialize on each node
    in the order they were registered

2) Handle each node independently in parallel:
2.1)    Call scheduler_update
2.2)    Wait for the callback
2.3)    Call scheduler_post_update

3) Observe all nodes and look for the stopping condition
3.1)    When found, stop the task 2)

4) call scheduler_terminate on each node
    in the order they were registered
```

As you can see, the `scheduler_update` method is called regardless of its output, so it may happen that the `can_advance` of a node is called many times even though it returns `false`. The scheduler does not know, how are all the nodes related, so it has to check.

The stopping condition could be detected by having an array of boolean flags, one for each node and having them all set to `false`. When a node update finishes without advancing data, we set its corresponding flag to `true`, interpreted as the possibility that this node has finished. This way many `true` flags may accumulate. If any node finishes update by advancing data, we know the whole pipeline state might have changed and we reset all the flags back to `false`. We watch this flag array and the moment it has all the flags set to `true`, we know that all nodes cannot advance data and so the stopping condition has been reached.


## Debugging scheduler

The `DebuggingScheduler` is the only scheduler currently available in noarr pipelines. The reason is that we did not want to spend time writing an optimal scheduler, while the whole concept of a pipeline composed of nodes was still not validated and it might turn our optimized scheduler code obsolete. So we only developed the debugging scheduler which will remain useful (for debugging) even if a parallel scheduler is added later.

The debugging scheduler is designed to run all nodes synchronously, to make the scheduling algorithm fully deterministic. Since it runs nodes synchronously, we may define the concept of a generation. One generation is one iteration over all registered nodes. This lets us simplify the detection of the stopping condition - just keep track of data advancements within a generation, and if all the nodes do not advance data, we know we can stop the pipeline. The algorithm for the debugging scheduler could be describe by this code:

```cpp
void noarr::pipelines::DebuggingScheduler::run() {
    
    // pipeline initialization
    for (auto& node : pipeline_nodes)
        node.scheduler_initialize();

    // the main loop, one iteration of which is called a generation
    bool generation_advanced_data;
    do {
        generation_advanced_data = false;

        // update each node
        for (auto& node : pipeline_nodes) {
            node.scheduler_update(this->callback_handler);
            bool advanced_data = this->wait_for_callback();
            node.scheduler_post_update(advanced_data);

            if (advanced_data) {
                generation_advanced_data = true;
            }
        }
    } while (generation_advanced_data);

    // pipeline termination
    for (auto& node : pipeline_nodes)
        node.scheduler_terminate();
}
```

You can see that the example uses internal API methods. The logic regarding callback handling is in reality more complicated than shown in the example. It requires locks, since the callback may be called by any thread. But the basic idea is to synchronize the asynchronous operation.

The primary feature of the debugging scheduler is the ability to print a log. You can construct the scheduler with an output stream:

```cpp
auto scheduler = DebuggingScheduler(std::cout);
```

The scheduler will produce a log like this one:

```txt
[scheduler]: Node has been added: N5noarr9pipelines3HubImiEE
[scheduler]: Node has been added: N5noarr9pipelines3HubImiEE
[scheduler]: Node has been added: producer
[scheduler]: Node has been added: filter
[scheduler]: Node has been added: consumer
[scheduler]: Starting pipeline initialization...
[scheduler]: Initializing node N5noarr9pipelines3HubImiEE ...
[scheduler]: Initializing node N5noarr9pipelines3HubImiEE ...
[scheduler]: Initializing node producer ...
[scheduler]: Initializing node filter ...
[scheduler]: Initializing node consumer ...
[scheduler]: Pipeline initialized.
[scheduler]: Updating node N5noarr9pipelines3HubImiEE ...
[scheduler]: Node did advance data.
[scheduler]: Updating node N5noarr9pipelines3HubImiEE ...
[scheduler]: Node did advance data.
[scheduler]: Updating node producer ...
[scheduler]: Node did advance data.
[scheduler]: Updating node filter ...
[scheduler]: Updating node consumer ...
[scheduler]: Generation 0 has ended.
[scheduler]: Updating node N5noarr9pipelines3HubImiEE ...
[scheduler]: Node did advance data.
[scheduler]: Updating node N5noarr9pipelines3HubImiEE ...
[scheduler]: Updating node producer ...
[scheduler]: Node did advance data.
[scheduler]: Updating node filter ...
[scheduler]: Updating node consumer ...
[scheduler]: Generation 1 has ended.
...
...
...
[scheduler]: Termination condition met.
[scheduler]: Starting pipeline termination...
[scheduler]: Terminating node N5noarr9pipelines3HubImiEE ...
[scheduler]: Terminating node N5noarr9pipelines3HubImiEE ...
[scheduler]: Terminating node producer ...
[scheduler]: Terminating node filter ...
[scheduler]: Terminating node consumer ...
[scheduler]: Pipeline terminated.
```

Each pipeline node has a label that is used in the log. You can set a label for a compute node during construction:

```cpp
auto my_node = LambdaComputeNode("my_node");
```

Internally, the `DebuggingScheduler` uses an instance of the `SchedulerLogger` class to encapsulate the logging logic.
