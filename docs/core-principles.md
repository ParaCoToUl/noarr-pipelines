# Core principles

The noarr pipelines library aims to provide a framework for building computational *pipelines* for GPGPU computing. These *pipelines* are designed to process data that would not fit into GPU memory in one batch and needs to be streamed. The user has to define a way to break the data down into a sequence of smaller chunks, process individual chunks, and then re-assemble the result from those chunks.


## Basics

The *pipeline* is composed of *nodes* - independent units that perform a piece of the computation. Imagine that we have a text file where we want to capitalize all the letters and save the result to another file. The entire file would not fit in memory, but we can say that one line of text easily would. We could process the file line by line, thereby streaming the whole process. The pipeline would have one *node* responsible for reading lines of text, another *node* for performing the capitalization, and another one for writing capitalized lines to the output file. This kind of separation lets all nodes run concurrently and increases the overall throughput of the system. We could also imagine, that the capitalization process was expensive and we would like to run it on the GPU.

> **Note:** The described task is implemented in the `uppercase` example and can be found [here](../examples/uppercase).

We described a scenario where we have *compute nodes* on different devices (GPU, CPU), but we need a way to transfer data between them. For this purpose, we will create a special type of node called a *hub*. A *hub* can be imagined as a queue of chunks of data (lines of text in our example). We can produce new chunks by writing into the hub and consume chunks by reading from it. We can then put one *hub* in between each of our *compute nodes* to serve as the queue in the classic producer-consumer pattern. This gives us the following pipeline:

```txt
[reader] --> {reader_hub} --> [capitalizer] --> {writer_hub} --> [writer]
   |                                                                |
input.txt                                                      output.txt
```

A *compute node* is joined to a *hub* by something called a *link*. A *link* mediates the exchange of data between both sides and it also holds various metadata, like its type (producing / consuming) or the device index on which the *compute node* expects the data to be received. Remember that we imagined the capitalizer to be a GPU kernel. It wants to receive data located in the GPU memory. The *hub* handles the memory transfer for us.


## Envelopes

When a chunk of data enters a *hub*, it is located on one device (say the host). When the same chunk exits the hub, it may be located on another device (say the GPU device). Therefore a chunk of data in a hub might be present on multiple devices simultaneously. We call one of these instances an *envelope*. It is guaranteed that all *envelopes* of one chunk hold the exact same data. When a new chunk is inserted into a hub, it has only one envelope. When the chunk is requested by another device, a new envelope is obtained on that device and the data is copied into it. When the chunk is consumed from the hub, all the envelopes are freed up.

An envelope has five main properties:

- **Buffer pointer**: This pointer points to the buffer containing the data of the envelope. The buffer is typically allocated by the hub, but may also be provided by the user.
- **Structure**: This value describes the structure of the data in the buffer. If the buffer contains a simple C array, this property is of type `std::size_t` and describes the length of the array. But we may choose to use noarr structures here to let the envelope hold arbitrarily complex data. If all the chunks have the same structure (e.g. images with the same resolution), this property might not even be needed, but the chunks are heterogenous, this property is what describes the structure of the chunk.
- **Device index**: This value describes the location of the data (host memory or GPU memory).
- **Size**: This is the size of the allocated buffer in bytes. This value cannot be changed and is set during the allocation of the envelope. All envelopes in a hub have the same size (the size is more like capacity, the actual data size should be stored in the *structure* property).
- **Type**: The type of the envelope is the value of two template parameters, the first specifies the type of the *structure* property (e.g. `std::size_t`, `std::array<std::size_t, 2>`) and the second specifies the type of the *buffer pointer* (e.g. `float`, `pixel_t`, `char`, `void`).

Envelope allocation is handled by *hubs*. Envelopes are allocated when a hub is created and they are reused throughout its lifetime (hubs manage a pool of unused envelopes). Envelopes are not shared between hubs and are destroyed with the hub. All envelopes on all devices within one hub are of the same type and the same size and both are specified during the creation of the hub. In certain situations, the user might provide an existing buffer to a hub to wrap it inside an envelope. This is useful for working with memory-mapped files.

The following code shows us, how to create a hub with two envelopes on each device, that can hold up to 1024 chars in each envelope:

```cpp
// Create a hub with envelopes with the following properties:
// - Structure type:       std::size_t
// - Buffer pointer type:  char
// - Envelope size:        sizeof(char) * 1024
auto my_hub = noarr::pipelines::Hub<std::size_t, char>(
    sizeof(char) * 1024
);

// Allocate 4 envelopes, 2 on each device
// Host = CPU memory (RAM)
// Device = GPU memory
my_hub.allocate_envelopes(noarr::pipelines::Device::HOST_INDEX, 2);
my_hub.allocate_envelopes(noarr::pipelines::Device::DEVICE_INDEX, 2);
```


## Compute nodes

Pipeline *nodes* operate using two methods: `can_advance` and `advance`. Each node has its own implementation of these two methods. The pipeline has one scheduler that monitors all the nodes and periodically asks each of them whether it `can_advance`. If the response is positive, the scheduler will call the `advance` method.

> **Note:** *Advance* means *to advance data through the pipeline*.

The `advance` method is meant to start an asynchronous operation that does not block. When the scheduler calls the `advance` method, it begins to treat the node as *working*. The node will remain in this state until it calls the `callback` method. The `callback` should be called when the asynchronous operation finishes, to signal the node becoming *idle*. The scheduler will never call `can_advance` and `advance` on a *working* node, only on an *idle* one.

> **Note:** The `callback` method can be called from any thread, it is designed to handle that.

> **Note:** Tracking the *idle/working* state of each node is the responsibility of the scheduler. Depending on the implementation of the scheduler, this tracking may be implicit in the scheduling logic.

When we create a custom *node*, we typically want to perform a computation, and in doing so, we want to produce or consume chunks in some hubs. We only want to start our computation when we know that all the linked hubs are ready to serve or accept the data. We could perform these checks in the `can_advance` method, but it would quickly get repetitive. For this reason, we define *compute nodes*. A *compute node* is like a regular *node*, but it knows about all the *links* to hubs, and only *advances* when all of these *links* are ready.

The following code shows a *compute node* linked to the hub from the previous code snippet. The compute node consumes chunks from the hub and prints their content as text to the screen:

```cpp
// create a compute node that has its methods defined
// using lambda expressions
auto my_node = noarr::pipelines::LambdaComputeNode("my_node");

// link the compute node to my_hub
// (to consume chunks from the host device)
auto& my_link = my_node.link(my_hub.to_consume(Device::HOST_INDEX));

// define the advance method:
my_node.advance([&](){
    
    // the hub provides access to an envelope via my_link.envelope,
    // the envelope contains data of the latest chunk
    // (since we want to consume)

    // the structure property holds the length of the char array
    std::size_t array_size = my_link.envelope->structure;

    // the pointer to the char array
    char* buffer_pointer = my_link.envelope->buffer;

    // print to the screen
    std::cout << std::string(buffer_pointer, array_size) << std::endl;

    // The advance method assumes we started an asynchronous operation
    // that signals its completion by calling back. We did not do that,
    // but we still need to call back.
    my_node.callback();
});
```

The *compute node* above does not have the `can_advance` method defined. It does not need one, since it only depends on the presence of chunks in `my_hub`. If we had a producing node, we could define the method like this:

```cpp
std::size_t chunks_produced = 0;
std::size_t total_input_chunks = 420;

my_other_node.can_advance([&](){
    return chunks_produced < total_input_chunks;
});
```


## Scheduling

In the code snippets above, we learned how to define hubs and compute nodes. The last thing that remains is adding a scheduler and letting it run the pipeline to completion:

```cpp
noarr::pipelines::SimpleScheduler scheduler;
scheduler << my_hub
          << my_node
          << my_other_hub
          << my_other_node;

// run the pipeline to completion
scheduler.run();
```

The scheduler calls `can_advance` on all nodes, trying to get them to advance. If all the nodes are *idle* and also respond negatively to `can_advance`, it means there are no nodes to be advanced, and the pipeline terminates.

The library currently provides only the `SimpleScheduler` and the `DebuggingScheduler`. This is because implementing an optimal scheduler is a complicated task and was left as a future improvement. However, the `SimpleScheduler` provides a good enough amount of parallelism in most cases. The `DebuggingScheduler` is described more in a [later section on debugging](#debugging).

Each node can perform some action during the pipeline initialization and termination by using the corresponding event methods:

```cpp
my_other_node.initialize([&](){
    // e.g. open a file to read
});

my_other_node.terminate([&](){
    // e.g. close the file
});
```


## Introductory example

This section talks about the absolute basics of noarr pipelines. You can read through the `uppercase` example located [here](../examples/uppercase) to see all these concepts put together.


## Multithreading

We said that the `advance` method starts an asynchronous non-blocking operation that ends by calling the `callback`, but all the examples so far were synchronous. This was only to get the basic concepts across and to keep the code simple. A more realistic way to create *compute nodes* is to extend the `AsyncComputeNode` or the `CudaComputeNode`. Both of these nodes extend the `advance` method in ways that allow it to be non-blocking. The `AsyncComputeNode` provides an `advance_async` method whose content is executed by a background thread. The `CudaComputeNode` provides an `advance_cuda` method that also runs in a background thread and finishes when a corresponding CUDA stream gets emptied.

By introducing additional threads we need to start worrying about synchronization of access to shared variables. Noarr pipelines solve this in a way that does not require the use of locks.

> **Note:** By a shared variable we mean any variable that is accessed by two different nodes. Hubs are not considered shared variables since they are designed to be accessed by multiple nodes simultaneously.

The thread that calls `scheduler.run()` is called the *scheduler thread*. The methods `can_advance`, `advance`, `initialize` and `terminate` all run in this thread. This makes the code in these methods safe to access any shared variables because executions of these methods will never overlap. On the other hand, methods like `advance_async` and `advance_cuda` are dangerous and should only access variables unique to the node.

> **Note:** A variable accessed by different methods of one node is not a shared variable, because different methods of the node cannot run concurrently (even `can_advance` is not called, when the node is *working*).

By using methods like `advance_async` we gain performance by spreading the load over multiple threads. But we also prevent ourselves from accessing shared variables. To solve this issue we have a method called `post_advance`. This method is guaranteed to be called when the asynchronous operation finishes, just before the node becomes *idle* and it runs in the *scheduler thread*. We can modify any shared variables from this method:

```cpp
std::size_t chunks_produced = 0;

my_other_node.post_advance([&](){
    chunks_produced += 1;
});
```


## Debugging

Since a pipeline is a complicated computational model, it is oftentimes difficult to troubleshoot problems. One of the tools we have at our disposal is the debugging scheduler:

```cpp
auto scheduler = noarr::pipelines::DebuggingScheduler(std::cout);
```

It has the feature that it never runs two nodes simultaneously. It loops over the pipeline nodes in the order they were registered and tries to advance them one by one. One iteration of this loop is called a generation.

Here is roughly how the debugging scheduler operates:

```cpp
void noarr::pipelines::DebuggingScheduler::run() {
    
    // pipeline initialization
    for (auto& node : pipeline_nodes)
        node.initialize();

    // the main loop, one iteration of which is called a generation
    bool generation_advanced_data;
    do {
        generation_advanced_data = false;

        // try to advance each node
        for (auto& node : pipeline_nodes) {
            if (node.can_advance()) {
                node.advance();
                node.wait_for_callback(); // be synchronous
                generation_advanced_data = true;
            }
        }
    } while (generation_advanced_data);

    // pipeline termination
    for (auto& node : pipeline_nodes)
        node.terminate();
}
```

If we provide an output stream to the scheduler constructor, it will log many interesting events to it. Going through the log may help us diagnose problems quicker than by using a traditional debugger.

Each pipeline node has a label that is used in the log. We can set a label for a compute node during construction:

```cpp
auto my_node = LambdaComputeNode("my_node");
```

Sometimes we would also like to see, what is happening inside a hub (how is the data transferred). We can do this by enabling logging for the hub in a similar way to the scheduler:

```cpp
my_hub.start_logging(std::cout);
```


## Overview of the terms

This is a short overview of all the important terms defined in the text of this section.

- **Node**: The basic building block of the pipeline. It can be *advanced* by the scheduler to perform some computation.
- **Compute node**: Node specialized for computation. It can be advanced only when all its *links* are ready.
- **Hub**: Node specialized for memory management. It handles allocations and memory transfers between hardware devices.
- **Chunk**:
    - *Pipeline perspective*: A smaller piece of the input dataset that can fit into memory and can be passed through the pipeline.
    - *Hub perspective:* A set of envelopes on different devices, all holding the same data.
- **Link**: An entity that mediates the exchange of data between a hub and a compute node.
- **Envelope**: Holder for data, with some structure and located on some hardware device. Also, the thing that is accessed through a link. Also the representation of a chunk on one specific device.
- **Advance**: When a *node* advances, it runs some computation. That computation does a piece of the work of *advancing* data through the pipeline.
- **Callback**: The method to call to signal the end of a node's *advance* operation.
- **Scheduler**: Is responsible for calling *advance* methods of pipeline nodes.
- **Scheduler thread**: The thread that calls `scheduler.run()`. It is safe to access shared variables from the code running in this thread.
