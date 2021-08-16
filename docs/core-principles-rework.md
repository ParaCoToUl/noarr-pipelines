# Core principles

The noarr pipelines library aims to provide a framework for building computational *pipelines* for GPGPU computing. These *pipelines* are designed to process data that would not fit into GPU memory in one batch and needs to be streamed. The user has to define a way to break the data down to a sequence of smaller chunks, process individual chunks and then re-assemble the result from those chunks.


## Basics

The *pipeline* is composed of *nodes* - independent units that perform a piece of the computation. Imagine that we have a text file where we want to capitalize all the letters and save the result to another file. The entire file would not fit in memory, but we can say that one line of text easily would. We could process the file line by line, thereby streaming the whole process. The pipeline would have one *node* responsible for reading lines of text, another *node* for performing the capitalization and another one for writing capitalized lines to the output file. This kind of separation lets all nodes run concurrently and increases the overall throughput of the system. We could also imagine, that the capitalization process was expensive and we would like to run it on the GPU.

> **Note:** The described task is implemented in the `upcase` example and can be found [here](../examples/upcase).

We described a scenario where we have *compute nodes* on different devices (GPU, CPU) but we need a way to transfer data between them. For this purpose we will create a special type of node called a *hub*. A *hub* can be imagined as a queue of chunks of data (lines of text in our example). We can produce new chunks by writing into it and consume chunks by reading from it. We can then put one *hub* in between each of our *compute nodes* to serve as the queue in the classic producer-consumer pattern. This gives us the following pipeline:

    [reader] --> {reader_hub} --> [capitalizer] --> {writer_hub} --> [writer]
       |                                                                |
    input.txt                                                      output.txt

A *compute node* is joined to a *hub* by something called a *link*. A *link* mediates the exchange of data between both sides and it also holds various metadata, like its type (producing / consuming) or the device index on which the *compute node* expects the data to be received. Remember that we imagined the capitalizer to be a GPU kernel. It wants to receive data located in the GPU memory. The *hub* handles the memory transfer for us.


## Envelopes

When a chunk of data enters a *hub*, it is located on one device (say the host). When the same chunk exits the hub, it may be located on another device (say the GPU device). Therefore a chunk of data in a hub might be present on multiple devices simultaneously. We call one of these instances an *envelope*. It is guaranteed that all *envelopes* of one chunk hold the exact same data. When a new chunk is inserted into a hub, it has only one envelope. When the chunk is requested by another device, a new envelope is obtained on that device and the data is copied into it. When the chunk is consumed from the hub, all the envelopes are freed up.

An envelope has five main properties:

- **Buffer pointer**: This pointer points to the buffer containing the data of the envelope.
- **Structure**: This value describes the structure of the data in the buffer. If the buffer contains a simple C array, this property is of type `std::size_t` and describes the length of the array. But you may choose to use noarr structures here to let the envelope hold arbitrarily complex data.
- **Device index**: This value describes the location of the data (host memory or GPU memory).
- **Size**: This is the size of the allocated buffer in bytes. This value cannot be changed and is set during the allocation of the envelope.
- **Type**: The type of the envelope is the value of two template parameters, the first specifies the type of the *structure* property (e.g. `std::size_t`, `std::array<std::size_t, 2>`) and the second specifies the type of the *buffer pointer* (e.g. `float`, `pixel_t`, `char`, `void`).

Envelope allocation is handled by *hubs*. Envelopes are allocated when a hub is created and they are reused throughout its lifetime (hubs manage a pool of unused envelopes). Envelopes are not shared between hubs and are destroyed with the hub. All envelopes on all devices within one hub are of the same type and the same size and both are specified during the creation of the hub.

The following code shows you how to create a hub with two envelopes on each device, that can hold up to 1024 chars in each envelope:

```cpp
// Create a hub with envelopes with the following properties:
// - Structure type:       std::size_t
// - Buffer pointer type:  char
// - Envelope size:        sizeof(char) * 1024
auto my_hub = noarr::pipelines::Hub<std::size_t, char>(sizeof(char) * 1024);

// Allocate 4 envelopes, 2 on each device
// Host = CPU memory (RAM)
// Device = GPU memory
my_hub.allocate_envelopes(noarr::pipelines::Device::HOST_INDEX, 2);
my_hub.allocate_envelopes(noarr::pipelines::Device::DEVICE_INDEX, 2);
```


## Compute nodes

Pipeline *nodes* operate using two methods: `can_advance` and `advance`. Each node has its own implementation of these two methods. The pipeline has one scheduler that monitors all the nodes and periodically asks each of them whether it `can_advance`. If the response is positive, the scheduler will call the `advance` method.

> **Note:** *Advance* means *to advance data through the pipeline*.

The `advance` method is meant to start an asychronous operation that does not block. When the scheduler calls the `advance` method, it begins to treat the node as *working*. The node will remain in this state until it calls the `callback` method. The `callback` should be called when the asychronous operation finishes, to signal the node becoming *idle*. The scheduler will never call `can_advance` and `advance` on a *working* node, only on an *idle* one.

> **Note:** Tracking the *idle/working* state of each node is the responsibility of the scheduler. Depending on the implementation of the scheduler, this tracking may be implicit in the scheduling logic.

When we create a custom *node*, we typically want to perform a computation and in doing so we want to produce or consume chunks in some hubs. We only want to start our computation when we know that all the linked hubs are ready to serve or accept the data. We could perform these checks in the `can_advance` method, but it would quickly get repetitive. For this reason we define *compute nodes*. A *compute node* is like a regular *node*, but it knows about all the *links* to hubs and only *advances* when all of these *links* are ready.

The following code shows a *compute node*, linked to the hub from the previous code snippet. The compute node consumes chunks from the hub and prints their content as text to the screen:

```cpp
// create a compute node that has its methods defined using lambda expressions
auto my_node = noarr::pipelines::LambdaComputeNode("writer");

// link the compute node to my_hub
// (to consume chunks from the host device)
auto& my_link = my_node.link(my_hub.to_consume(Device::HOST_INDEX));

// define the advance method:
my_node.advance([&](){
    
    // the hub provides access to an envelope via my_link.envelope,
    // the envelope contains data of the latest chunk (since we want to consume)

    // the structure property holds the length of the char array
    std::size_t array_size = my_link.envelope->structure;

    // the pointer to the char array
    char* buffer_pointer = my_link.envelope->buffer;

    // print to the screen
    std::cout << std::string(buffer_pointer, array_size) << std::endl;

    // The advance method assumes you start an asynchronous operation that
    // will signal its completion by calling back. We did not do that,
    // but we still need to call back.
    writer.callback();
});
```

TODO...

- scheduler in detail (scheduling algorithm)


## Overview of terms

- bullet points of all the terms
    - node
    - compute node
    - hub
    - chunk (abstractly / concretely in a hub)
    - link
    - envelope
    - advance
    - callback
    - scheduler
