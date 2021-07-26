# Core principles


## Basics

Noarr pipelines aims to provide a framework for building computational pipelines for GPGPU computing. These pipelines are designed to process data that won't fit into GPU memory in one batch and needs to be streamed. The user has to define a way to break the data down to a sequence of smaller chunks, process individual chunks and then re-assemble the result from those chunks.

The core conceptual unit of a pipeline is a *compute node*. It's an entity that receives one chunk of data and is triggered to perform a computation on that chunk. The computation can produce a new chunk, modify the existing or perform an aggregation or do any combination of these actions.

```cpp
// create a compute node
auto my_compute_node = LambdaAsyncComputeNode();

// link the compute node to a hub
// with the goal of modifying chunks in the hub
auto& link_to_hub = my_compute_node.link(some_hub.to_modify(Device::HOST_INDEX));

// define what happens when the compute node is triggered
my_compute_node.advance([&](){
    
    // get the envelope holding the actual data chunk
    auto& env = *link.envelope;

    // modify the data in place
    for (std::size_t i = 0; i < env.structure; ++i)
        env.buffer[i] *= 2;
});
```

Chunks, that compute nodes work with, come from so called *hubs*. A hub can naively be thought of as a buffer that can store a chunk of data. When a compute node triggers, it has access to a set of hubs with which it can interact. This set of hubs and the way the node will interact with them is specified in advance as a set of *links*.

A hub is actually not a single buffer but multiple, spread across multiple devices. This is because hub also provides inter-device memory transfer - e.g. one compute node can write to the hub from CPU and another one can read the data from GPU. One of these internal buffers is called an *envelope*. So a compute node accessing a hub through a link actually interacts only with a single envelope, not the entire hub.

An envelope has two main parts: 1) the structure 2) the data. The structure tells the user what shape the data in the envelope has. It is the length for arrays, dimensions for images, etc... The data portion of the envelope contains a continuous binary blob of data that the user has to navigate through based on the structure. The envelope has a pre-allocated buffer with a specific size and the user data may never be larger that the allocated size.

```cpp
// An example envelope containing an array of floats, where the structure of the
// envelope is a std::size_t coding the length of the array.

// API: Envelope<Structure, BufferItem>

Envelope<std::size_t, float> env = ...;
env.structure = 3; // the envelope will contain 3 items
env.buffer[0] = 12.5f; // set these items
env.buffer[1] = 8.2f;
env.buffer[2] = 0.1f;
```


## Producer-consumer example

The following code is a simple producer-consumer example. One compute node produces chunks of data. One chunk of data is a single line of text from the standard input. It writes these chunks into a central hub. The consumer compute node then reads chunks from the central hub and prints them to the screen.

```cpp
// API: Hub<Structure, BufferItem>(envelope_size_in_bytes);
auto line_hub = Hub<std::size_t, char>(sizeof(char) * 1024);
line_hub.allocate_envelopes(Device::HOST_INDEX, 2);

// global variable, used by the producer to stop
bool finished = false;

auto producer = LambdaAsyncComputeNode(); {
    auto& line_link = producer.link(line_hub.to_produce(
        Device::HOST_INDEX, // the link wants to access the data from the CPU
        false // a chunk may sometimes not be produced - disable auto-commit
    ));

    // when can the produer node be triggered?
    producer.can_advance([&](){
        return !finished;
    });

    producer.advance([&](){
        std::string text;
        bool success = std::getline(std::cin, text);
        if (success) {
            line_link.envelope->structure = text.length();
            text.copy(line_link.envelope->buffer, text.length());
            line_link.commit(); // chunk was actually produced
        } else {
            finished = true; // there will be no further chunks
        }
    });
}

auto consumer = LambdaAsyncComputeNode(); {
    auto& line_link = consumer.link(line_hub.to_consume(Device::HOST_INDEX));

    consumer.advance([&](){
        std::string text(
            line_link.envelope->buffer,
            line_link.envelope->structure
        );
        std::cout << "CONSUMER GOT: " << text << std::endl;
    });
}
```

> **Caution:** Be careful about declaring variables in the braced compute node body. Their lifetime may be shorter than the lifetime of provided lambda expressions. Usually, declare only links there and only as references. Passing a reference by reference to a lambda will actually copy the underlying pointer and the lifetime issue is not an issue.

> **Note:** There's also an alternative inheritance-based API if your pipeline is more complicated or you don't like this lambda-based API.


## Scheduling

Once you define your pipeline, you need a scheduler to trigger individual pipeline nodes. Here's the code for the example above:

```cpp
auto scheduler = DebuggingScheduler();
scheduler.add(link_hub);
scheduler.add(producer);
scheduler.add(consumer);

scheduler.run(); // runs the pipeline to completion
```

The scheduler has a list of nodes it tries to trigger, but before it triggers a node, it first asks it, whether it has some work to do. This asking is realized by the `can_advance` function. If the function returns `true`, the scheduler invokes the `advance` method that does the actual computation. When the scheduler has no nodes that can be advanced (all return `false`), it recognizes the situation as the end of computation and finishes.

> **Note:** *Advance* means to *advance data through the pipeline*.

You may have noticed that we didn't provide the `can_advance` function for the consumer node in our example above. This is because a *compute node* is somewhat special, compared to a generic pipeline node. A *compute node* has the added constraint that it cannot be advanced, unless all of its links are ready (have an envelope prepared). This is enough to condition our consumer - it consumes chunks as long as there are chunks to consume.

A *hub* is also a pipeline node, triggered by the scheduler like any other node. But you don't need to worry about this from the user perspective.


## Multithreading

The thread that calls `scheduler.run()` is called the *scheduler thread*. It's important, because it is guaranteed that all important synchronization events (like the `can_advance` invocation) happen on the scheduler thread. This lets you to not worry about locks and other synchronization primitives.

There are different types of compute nodes that run their computation in different contexts, but the `AsyncComputeNode` for example has four imporant methods: `can_advance`, `advance`, `advance_async`, `post_advance`. All of these methods, except for `advance_async` run on the scheduler thread and make it easy for you to access and modify any global state. The `advance_async` method then runs in a background thread and can perform some heavy computation without blocking the scheduler thread. But you have to make sure you don't access global variables, that could be simultaneously accessed by other nodes from there.


## Debugging

Since a pipeline is a complicated computational model, it's oftentimes difficult to troubleshoot problems. One of the tools you have at your disposal is the debugging scheduler:

```cpp
auto scheduler = DebuggingScheduler(std::cout);
```

It has the feature that it never runs two nodes in parallel. It loops over pipeline nodes in the order they were registered and tries to advance them one by one. One iteration of this loop is called a generation.

If you provide an output stream to the scheduler, it will log many interesting events to it. Going through the log may help you diagnose problems quicker, than by using a traditional debugger.

Each pipeline node has a label that is used in the log. You can set a label for a compute node during construction:

```cpp
auto my_node = LambdaAsyncComputeNode("my-node");
```

Sometimes you'd also like to see, what's happening inside a hub (how is the data transfered). You can do this by enabling logging for the hub in a similar way to the scheduler:

```cpp
my_hub.start_logging(std::cout);
```


## Nodes and envelope hosting

From the scheduler's point of view it doesn't matter where the data in the pipeline comes from. It just advances nodes when they can be advanced. But since we typically think of programs as having a datastructure and an algorithm, we devised two types of nodes: hubs and compute nodes. There is currently no need to develop other kinds of data-holding nodes, but if that was to become a need, this section describes the conceptual framework.

Data is exchanged between nodes via links. There is the node that owns the data - the *host*. And the node that acts on the provided data - the *guest* (e.g. the hub hosts envelopes to compute nodes). This terminology governs the structure of a link.

A links starts out with no envelope attached. It's the hosts responsibility to acquire an envelope and attach it to the link. Whether the envelope contains data or not depends on the host and the link type can be used to guide the decision. The moment an envelope is attached by the host, the link transitions to a *fresh* state. Now it's the guest's turn. Guest can look at the envelope and do with it what it sees fit. Once the guest is done, it switches the link to a *processed* state. Now the host can detach the envelope and continue working with it. This is how one node can lend an envelope to another node.

Apart from the link state and the attached envelope, the link also has additional information present that may or may not be used by both interacting sides. The link has a type and it has a *commited* flag. There are four types of links: producing, consuming, peeking and modifying. These types describe the kind of interation the guest wants to make with the data. When producing, the host provides an empty envelope and guest fills it with data. The opposite happens during consumption. During peeking and modification the host provides some existing data to the guest, who can either look at or modify the data. The *commited* flag is set by the guest during production, consumption or modification to signal that the action was in fact performed. This lets the guest to e.g. not consume a chunk of data if it doesn't like the chunk.

A link also has an *autocommit* flag that simply states that the action will be commited automatically when the envelope lending finishes. If this flag is set to `false`, the guest node has to manually call a `commit()` method before finishing the envelope lending.

To learn more about the interplay, read the `Link.hpp` file and the `ComputeNode.hpp` file.
