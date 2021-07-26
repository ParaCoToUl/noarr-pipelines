# Compute node

This section talks more about how comptue nodes work and what types of compute nodes there are. The previous section already talked about the pipeline structure and that a compute node is a node that can advance only when all of its links are ready and we will build on that.

The core of noarr pipelines defines four different compute nodes: `ComputeNode`, `AsyncComputeNode`, `LambdaComputeNode`, `LambdaAsyncComputeNode`. There are also `CudaComputeNode` and `LambdaCudaComputeNode` but they are described in more detail in a following section on CUDA integration.


## Construction

You can see that each compute node is provided in two variants: one as is, and one with a `Lambda` prefix. This refers to the two ways in which a compute node can be constructed. The first one uses inheritance and the other one uses lambda expressions.

To construct a compute node via inheritance, do the following:

```cpp
class MyCustomNode : private AsyncComputeNode {
protected:
    void advance() override {
        // implement the advancement
    }
};

// make an instance
auto node = MyCustomNode();
```

To construct a compute node via lambda expressions, do the following:

```cpp
auto node = LambdaAsyncComputeNode();

node.advance([&](){
    // implement the advancement
});
```


## Event methods

A compute node has a set of event methods, that are called by the scheduler, and you can use them to process the data. These methods are called in the order they are listed:

```cpp
auto node = LambdaAsyncComputeNode();

node.initialize([&](){
    // called before the pipeline starts,
    // runs on the scheduler thread,
    
    // useful to put empty chunks into hubs or to initialize global state
    // or to pre-compute some datastructure and put it into a hub

    // do not access links here
});

node.can_advance([&](){
    // the condition to advance the node,
    // only invoked when all links are ready,
    // runs on the scheduler thread
    return some_condition;
});

node.advance([&](){
    // access links and advance the data from here,
    // runs on the scheduler thread,
    // can be omitted if "advance_async" is defined
});

node.advance_async([&](){
    // access links and advance the data from here,
    // runs on a dedicated thread
    // called after "advance" finishes
});

node.post_advance([&](){
    // called after advance and post_advance,
    // runs on the scheduler thread,
    
    // useful to update the global state
    // (increment iteration count, manually modify hub state, etc...)
});

node.terminate([&](){
    // called after the entire pipeline finishes,
    // runs on the scheduler thread,

    // useful to read out global state or manually read data left in some hubs

    // do not access links here
});
```


## Advance with callback

Most examples shown so far used the `AsyncComputeNode`. That is because this compute node has features that make it easier to demonstrate the core concepts. It has the `advance` method that runs on the scheduler thread and it has the `advance_async` method called afterwards that runs in a background thread. In each of these methods, if you want to end the execution, you simply `return` from these methods.

This behaviour is built on top of a plain `ComputeNode` class, that only provides the `advance` method. The difference is, that this simple compute node treats the `advance` method as something that lanuches an asynchronous operation and this operation finishes only when a `callback()` method is called. Here is an example of a compute node that performs some computation on the scheduler thread:

```cpp
auto node = LambdaComputeNode();

node.advance([&](){
    // perform computation here on the scheduler thread

    node.callback(); // the operation has finished
});
```

If you forget to call the `callback()`, the node assumes there's some asynchronous operation still running and it blocks forever (when you for example do a `return` midway through the method).

But it also lets you start an asynchronous operation that won't block the scheduler thread, as long as you give it the callback to call when it's done:

```cpp
auto node = LambdaComputeNode();

node.advance([&](){
    start_some_asynchronous_operation([&](){
        node.callback();
    });
});
```

The `callback` method may be called from any thread (it is thread-safe).

The `post_advance` event method is called on the scheduler thread right after you call the `callback()`.


## Extending a compute node

If you wish to write a connector for a different GPGPU framework and you want to extend the compute node, a good place to start is to read the `AsyncComputeNode.hpp` and `CudaComputeNode.hpp` files. This section contains some explanation regarding those.

You typically want to extend the logic around `advance` and `post_advance` methods and you will do this extension by inheritance. The problem is, that when a user uses your `MyExtendedComputeNode` and they override these two methods, they will most likely forget to call your parent implementation. We decided to deal with this problem by creating a separate `__internal__advance` and `__internal__post_advance` methods. These internal varians of these methods are meant to be overriden by users extending the logic of a compute node and they always have to call their parent implementation. The non-internal variants are then dedicated to the end user only and they shouldn't call their parent implementation (since it's either abstract or empty).

The (simplified) call stack for the `advance` method will then look like this:

```
Scheduler::run()
Node::scheduler_update(...)  <--- scheduler wants to advance the node
MyExtendedComputeNode::__internal__advance()  <--- your code
ComputeNode::__internal__advance()  <--- you call the parent implementation
Node::__internal__advance()
MyExtendedComputeNode::advance()  <--- parent implementation calls end-user impl.
```

Since it might be useful to override any of the event methods, all of the methods have their internal variant and you should use it when overriding (`__internal__can_advance`, `__internal__post_advance`, ...).


## Generic pipeline node

All that was said above about compute nodes actually applies to the `Node` class. It also has the same event methods. The only difference is that a `ComputeNode` overrides the `__internal__can_advance` method and it checks all links for being ready before calling the parent implementation (which in turn calls the end-user implementation). To see all the differences, read the `ComputeNode.hpp` file, it's quite short actually.

This also means that the knowledge written in this section can be used to understand the inner workings of a hub, since it's a plain pipeline node as well. It also means you can use the knowledge to build custom nodes that do e.g. load balancing of chunks or broadcasting to multiple consumers, etc...
