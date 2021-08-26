# Compute node

In the previous section on basic concepts, we already talked about *nodes*. We said that *nodes* are advanced by the scheduler. We then defined *compute nodes* as *nodes*, whose advancement is conditioned by having all of their *links* in the *ready* state. This section will mainly discuss details regarding *compute nodes*, but many of the concepts are directly applicable to generic *nodes* since they do not differ by much.


## Construction

There are two options of how to construct a compute node for a pipeline:

- via inheritance
- by lambda functions

Most of the documentation and examples show the lambda-expression way, as it allows us to define everything in one place. But if we should build more complicated nodes and pipelines, we might want to choose the inheritance approach instead.

Here is how to define a custom compute node via inheritance:

```cpp
class MyCustomComputeNode : private ComputeNode {
protected:
    void can_advance() override {
        return true; // implement the condition
    }

    void advance() override {
        // implement the advancement

        // call back
        this->callback();
    }
};

void use_the_node_in_a_pipeline() {
    // make an instance
    auto my_node = MyCustomComputeNode();

    // add it to the scheduler and run the pipeline
    SimpleScheduler scheduler;
    scheduler.add(my_node);
    scheduler.run();
}
```

The disadvantage of the inheritance approach is that if we want to access any shared state (hubs or variables), we need to pass references via the constructor which makes the approach require more code and boilerplate.

The other option is to define the custom node via lambda expressions:

```cpp
// make an instance
auto my_node = LambdaComputeNode();

// define methods via lambda expressions
my_node.can_advance([&](){
    return true; //implement the condition
});

my_node.advance([&](){
    // implement the advancement

    // call back
    my_node.callback();
});

// add it to the scheduler and run the pipeline
SimpleScheduler scheduler;
scheduler.add(my_node);
scheduler.run();
```

The `LambdaComputeNode` is a special compute node that acts as a builder for a compute node, accepting lambda expressions. It is a builder and a compute node at once, it does inherit from the `ComputeNode` class. It is the case that for other types of compute nodes there are their `Lambda-` prefix variants for building via lambda expressions.


## Event methods

A `ComputeNode` has the following event methods. These methods are automatically called by the scheduler (and so run in the scheduler thread) and we can override them to provide custom behavior.

```cpp
my_node.initialize([&](){
    // ...
});
```

The `initialize` event method is called before the pipeline starts on each node. The order in which it is called on the nodes is the same in which these nodes were registered into the scheduler. It can be used to put empty chunks into hubs, open files and perform other kinds of initialization.

```cpp
my_node.can_advance([&](){
    return true; // a condition expression
});
```

The `can_advance` event method is called when the node is *idle* and the scheduler tries to advance it. If it returns `true` the scheduler will immediately call the `advance` method. For *compute nodes*, this method is not called, unless all links are already ready.

```cpp
my_node.advance([&](){
    // ...
});
```

The `advance` event method should start the actual computation of the compute node. This computation may be asynchronous and run in another thread. It is advised for expensive computations for them not to block the scheduler thread. Since this is a common requirement, noarr pipelines provide an `AsyncComputeNode` that already asynchronizes the `advance` method. This extended node is explained in a [following section](#async-compute-node). When the asynchronous operation finishes, it must call the `my_node.callback()` method to signal its completion. Otherwise, the scheduler will remain thinking the operation is still running and will not advance the node anymore. We do not need to start an asynchronous operation if the logic we want to perform is relatively simple and/or it accesses a lot of shared variables, so we need to run in the scheduler thread anyways. But even in this case, we need to call `callback` when we are done.

After the `advance` method is invoked are all the links for this compute node ready, have envelopes attached and may be modified by the node from any thread. This remains true until the `post_advance` method finishes.

```cpp
my_node.post_advance([&](){
    // ...
});
```

The `post_advance` event method is executed by the scheduler thread right after the `callback` is called. It can be used to update any shared variables. Links for the node are still in the same state they were in the `advance` method and can be used in the usual way.

```cpp
my_node.terminate([&](){
    // ...
});
```

The `terminate` event method is an analog of the `initialize` method. It is called after the pipeline finishes all computation. It is invoked on all nodes in the order they were registered into the scheduler.


## Async compute node

The `AsyncComputeNode` and `LambdaAsyncComputeNode` provide an extension to the plain compute node by making the `advance` method asynchronous. In fact, they add an additional event method `advance_async`:

```cpp
auto my_async_node = LambdaAsyncComputeNode();

my_async_node.advance_async([&](){
    // ...
});
```

This method can be used instead of the plain `advance` method and it executes in a background thread, so it does not block the scheduler thread. In most cases, when we do not need to access shared variables, we should use this compute node and this method instead of the plain `advance` method as it increases the throughput of our pipeline.

Because the async compute node manages the asynchronous operation by itself, it knows when it finishes. Therefore it knows when to call the `callback` method, so we do not have to.

We can still use the `advance` method in addition to the `advance_async` when we want to access shared variables from the scheduler thread before the asynchronous operation starts:

```cpp
my_async_node.advance([&](){
    // runs first, in the scheduler thread
});

my_async_node.advance_async([&](){
    // runs second, in a background thread
});

my_async_node.post_advance([&](){
    // runs last, in the scheduler thread
});
```

<!-- TeX: \pagebreak -->

## Other nodes and compute nodes

This diagram shows the full inheritance hierarchy for nodes provided by noarr pipelines and the CUDA extension:

```txt
              +------+
              | Node |
              +------+
                |  |
          ------    ------- 
         |                 |
  +-------------+       +-----+
  | ComputeNode |       | Hub |
  +-------------+       +-----+
         |
+------------------+
| AsyncComputeNode |
+------------------+
         |
+-----------------+
| CudaComputeNode |
+-----------------+
```

The diagram excludes the `Lambda-` prefixed variants since they simply inherit from their non-prefixed variants. There are these lambda variants:

- `LambdaComputeNode`
- `LambdaAsyncComputeNode`
- `LambdaCudaComputeNode`


## Custom compute node extension

This section describes how we could make our own extended compute nodes, like the `AsyncComputeNode` or the `CudaComputeNode`.

First, we need to differentiate between two users of a compute node:

- **End-user**: This is what we have been doing so far. We defined custom compute nodes via lambda expressions or inheritance, but we have overridden only the event methods that were designed to be overridden in this way.
- **Extending-user**: This is what we want now. We want to add custom event methods, modify the way existing event methods are called, or add additional fields and utility methods. We want our product to be used by the end-user.

Extending the `Node` class is mostly the same as extending any other class, but there are a few design decisions that make it non-obvious how to change the behavior of event methods.

Say we are extending the `ComputeNode` and we want to have a piece of code be executed before the `advance` method is called. We would typically do something like this:

```cpp
class MyExtendedComputeNode : public ComputeNode {
protected:
    void advance() override {
        // run our logic
        run_our_custom_logic();

        // call the parent implementation
        ComputeNode::advance();
    }
}
```

There are two problems with this approach:

1. We call the parent implementation, but the parent implementation does nothing. It makes no sense for an abstract compute node to be advanced. What makes sense is to call the child implementation, but that cannot be done.
2. We expect the end-user to call our implementation in order for our code to work. But they will forget. And also if they use lambda expressions to define their compute nodes, they cannot even call the parent implementation.

The problem is that end-users will not call their parent implementations and the extending users need to have their implementation be called. To get around this problem we have split each event method into an external and an internal variant. Their definitions for the `Node` class look like this:

```cpp
class Node {
public:
    // this is what the scheduler actually calls
    void scheduler_update() {
        this->__internal__advance();
    }

protected:
    virtual void __internal__advance() {
        this->advance();
    }

    virtual void advance() {
        // empty
    }
}
```

We can see that the scheduler calls the internal variant. Since this variant is virtual, it will be called by the most inherited child first, and propagate the calls to its parents. Ultimately, the calls will hit the `Node::__internal__advance` method, which finally calls the `advance` method. The external `advance` method is only overridden by the last child and does not need to care about calling parent implementations.

So to correct the example, we would write this code:

```cpp
class MyComputeNode : public ComputeNode {
protected:
    void __internal__advance() override {
        // run our logic
        run_our_custom_logic();

        // call the parent implementation
        // (ultimately calls the external advance() method)
        ComputeNode::__internal__advance();
    }
}
```

One way to look at this is that the internal variants of event methods are designed to be nested and called from the child to the parent by the scheduler. The external variants are not nested at all, they have only the top-most child implementation-defined, provided by the end-user. External variants are for the end-user, internal variants are for the extending user.
