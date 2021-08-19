# Hub

*Hubs* are pipeline *nodes* designed to handle memory management. They are connected to *compute nodes* by *links* of various types and the data is passed through these links in the form of *envelopes*. The hub is designed to handle a number of responsibilities:

- **Queuing chunks**: The hub is implemented as a queue of *chunks*. Some *links* produce new *chunks* and other *links* consume *chunks*. This lets the producer and consumer *nodes* run independantly.
- **Memory transfer**: Each one of the *links* may want to access the data on a different device. The *hub* has to perform memory transfers so that these requirements are satisfied. A memory transfer involves creating a copy of one envelope on another device. This means that a *chunk* of data is represented by a set of *envelopes* on different devices, all holding the same data. If the *hub* has enough available *envelopes*, it can overlay these memory transfers with other operations (like producing and consuming) to increase the throughput of the pipeline.
- **Envelope lifetime**: A *hub* manages a pool of *envelopes* and uses them to store the data passing through it. When you create a *hub*, you specify how many envelopes on what devices should be allocated. When a *compute node* wants to produce a chunk, the *hub* will give it an empty *envelope* to be populated by the data and then it inserts the *envelope* into the internal queue as a new *chunk*. Then this *chunk* may be copied to other devices, using up more *envelopes* from the pool of unused envelopes. When a *chunk* is consumed, all of its *envelopes* are returned back to the pool of unused envelopes. An envelope may never move from one *hub* to another.


## Produce-transfer-consume in detail

To get the inner workings of a hub fully across, we demonstrate the following diagram. It shows a hub with one producing link on the CPU and one consuming link on the GPU. The hub starts with two empty envelopes on each device:

    Pool of unused envelopes:
    [Env_A, CPU: <empty>], [Env_B, CPU: <empty>], [Env_C, GPU: <empty>],
    [Env_D, GPU: <empty>]

    Chunk queue:
    <empty>

> **Note:** An empty envelope is not acutally empty by having zero size. It is empty in the same sense an unitialized variable is empty - it contains some data but the data is considered to have no meaning.

> **Note:** The proper names for devices are *the host* and *the device*, but we use the abbreviations CPU and GPU in this example since they are shorter and easier to uderstand.

When the producing link wants to produce a new chunk, it is given an unused envelope on the CPU (the envelope `Env_A`). When the production is commited, the envelope is inserted into the queue of chunks:

    Pool of unused envelopes:
    [Env_B, CPU: <empty>], [Env_C, GPU: <empty>], [Env_D, GPU: <empty>]

    Chunk queue:
    Chunk( [Env_A, CPU: Chunk_1] )

Since the consumer wants to receive the data on the GPU, the hub transfers the chunk to the GPU device:

    Pool of unused envelopes:
    [Env_B, CPU: <empty>], [Env_D, GPU: <empty>]

    Chunk queue:
    Chunk( [Env_A, CPU: Chunk_1] )
         ( [Env_C, GPU: Chunk_1] )

While the transfer was happening, the producer was able to produce another chunk of data, because there was an empty envelope available (`Env_B`):

    Pool of unused envelopes:
    [Env_D, GPU: <empty>]

    Chunk queue:
    Chunk( [Env_B, CPU: Chunk_2] )  Chunk( [Env_A, CPU: Chunk_1] )
         (                       ),      ( [Env_C, GPU: Chunk_1] )

Now the consumer has its data ready and so it receives the envelope `Env_C`. When the consumtion finishes, the entire last chunk is deleted and its envelopes return back to the pool:

    Pool of unused envelopes:
    [Env_D, GPU: <empty>], [Env_A, CPU: <empty>], [Env_C, GPU: <empty>]

    Chunk queue:
    Chunk( [Env_B, CPU: Chunk_2] )

> **Note:** The chunk that will be consumed is called the *top chunk*. It is the oldest chunk in queue. It is the one on the dequeuing end of the queue.


## The flow of envelopes

You already have a decent idea of how envelopes flow though a hub, now we will go over it once more in full detail.

**Allocation:** Envelopes are explicitly allocated right after you create a hub. Their size is the same for all envelopes in one hub and it is provided through the constructor of the hub. By the manual allocation you specify how many of them should be available for what device.

```cpp
auto my_hub = Hub<std::size_t, float>(
    sizeof(float) * 1024 // size of all envelopes in my_hub
);

my_hub.allocate_envelopes(Device::HOST_INDEX, 2); // 2 envelopes on the host
my_hub.allocate_envelopes(Device::DEVICE_INDEX, 4); // 4 envelopes on the device
```

The number of envelopes impacts the behaviour of the hub. If you have no envelopes for a given device and a consuming link requests data for that device, it may never be satisfied. The consuming node will not be able to advance and advancing the hub will not solve the situation either. Therefore no nodes can be advanced and the pipeline terminates as if it has successfully done all the work. Having only one envelope allows the pipeline to function, but it cannot overlay any two operations, so it either produces or transfers or consumes. Having two envelopes is probably optimal in most situations that use the hub as a queue. There are, however, other ways to use the hub that may require different numbers of envelopes for optimal performance.

Although it is typical to allocate all envelopes during the construction of the hub, you may also call the `allocate_envelope(s)` methods anytime you want. The hub will be able to use the new envelope right after the allocation finishes. The only constraint is that allocations should be executed by the scheduler thread only.

**Unused pool:** When an envelope is allocated, it enters the so called *pool of unused envelopes*. This pool contains envelopes that are not present in the chunk queue and can be used for new chunks or for memory transfers. Since each envelope belongs to only one device, you could imagine that this pool is actually multiple pools - one for each device. If this pool becomes empty of envelopes for a given device, no chunk production or memory transfers for that device may occur. The hub waits for chunks in the queue to be consumed, so that envelopes get released and can be reused.

**Chunk queue:** When a producing link receives an envelope from the unused pool and the production is commited (if not, the envelope returns back to the unused pool), the envelope enters the chunk queue as the only envelope in a new chunk. If the consuming link requests the same device, the envelope just sits through the queue and when the chunk is consumed, the envelope is returned to the unused pool. If the consuming link requests a different device, memory transfer has to take place before consumption is allowed.

The hub goes over chunks in the queue from the oldest to the newest and for each one of them tries to make sure it has an envelope for the requested device. If not, a memory transfer begins. First an empty envelope for the target device is taken from the unused pool. If the pool has no such envelope, the transfer does not start and waits for such envelope to become available. Once we have the target envelope, we look at all the envelopes in our chunk and choose the one from which to copy. There has to be at least one available, since each chunk in the queue must have at least on envelope present (the one that produced the chunk). We prioritize the host-device and device-host transfers if possible and if not, a device-device transfer will happen (two different graphics cards).

When the memory transfer finishes, the source envelope is not released. It remains in the queue until its entire chunk is consumed. This is because the chunk may still be read from that device. There are situations, where the hub can be used for holding data that is often read from multiple devices and consumed only rarely from one of these devices.

**Trashed envelopes:** When the *top chunk* is consumed and its envelopes released, they do not immediately go to the unused pool. They first enter the pool of *trashed* envelopes. These are envelopes that are about to become *unused*, but may still have some links looking at them. It may happen that multiple links read the *top chunk*, but only one link consumes it. The other links are still reading the chunk for a while after it has been consumed. The pool of trashed envelopes makes sure all these links finish, before these envelopes are fully released.

**Deallocation:** Consuming the *top chunk* will not deallocate its envelopes. It only releases them back to the unused pool. Proper deallocation happens only during destruction of the hub.


## Manual chunk queue manipulation

We have described how *links* can produce and consume chunks. It might sometimes be needed to do these actions manually. You can imagine a scenario, where you have a compute node that uses one hub as an accumulator. With each iteration it simultanouesly consumes one chunk from the hub and produces one chunk as a replacement. We need to initialize the accumulator hub with some zero-valued chunk when the pipeline starts and then read out the final value when the pipeline finishes.

There are a few functions that should help you in such a scenario:

```cpp
// Takes an envelope from the pool of unused envelopes for the host device,
// inserts it into the chunk queue as a new chunk and returns a reference to it.
auto& envelope = my_hub.push_new_chunk();
envelope.buffer[0] = 0;

// Returns the reference to the host device envelope of the top chunk.
// Fails if the chunk queue is empty. Performs a synchronous memory transfer
// of the chunk to the host device if there is no host envelope for the chunk.
// The memory transfer may fail due to the lack of unused envelopes.
auto& envelope = my_hub.peek_top_chunk();
std::cout << envelope.buffer[0] << std::endl;

// Consumes the top chunk (removes it from the chunk queue).
// Returns void, fails if the queue is empty.
my_hub.consume_top_chunk();

// Returns the current number of chunks in the queue.
// Can be used to check for queue emptines.
std::size_t length = my_hub.get_queue_length();
```

> **Notice:** All of these functions should only be called from the *scheduler thread*.

Below is the example of an accumulator hub, being used by a compute node:

```cpp
auto my_node = noarr::pipelines::LambdaAsyncComputeNode("my_node");
auto acc_hub = noarr::pipelines::Hub<void*, int>(sizeof(int));
acc_hub.allocate_envelopes(Device::HOST_INDEX, 2);

auto& producing_link = my_node.link(acc_hub.to_produce(Device::HOST_INDEX));
auto& consuming_link = my_node.link(acc_hub.to_consume(Device::HOST_INDEX));

my_node.initialize([&](){
    auto& env = acc_hub.push_new_chunk();
    env.buffer[0] = 0;
});

my_node.advance_async([&](){
    consuming_link.envelope->buffer[0] = producing_link.envelope->buffer[0] + 1;
});

my_node.terminate([&](){
    auto& env = acc_hub.peek_top_chunk();
    std::cout << env.buffer[0] << std::endl;
});
```

> **Note:** Using a hub for a single integer value is a bad idea, you should use a regular shared variable for that. This is just to demonstrate the concept. Use hubs only when you actually benefit from their features.

You may also want to push or peek a chunk from another device than the host. The two methods let you specify the device index in the first argument:

```cpp
auto& gpu_envelope = my_hub.push_new_chunk(Device::DEVICE_INDEX);
my_init_kernel<<<...>>>(gpu_envelope.buffer);

auto& gpu_envelope = my_hub.peek_top_chunk(Device::DEVICE_INDEX);
my_other_kernel<<<...>>>(gpu_envelope.buffer);
```


## Peeking and modifying links

We have talked about producing links and consuming links. They are essential for understanding the hub as a queue of chunks. But there are two other types of links that let us work with a hub as if it was a buffer holding one piece of data. These two types are:

- **Peeking link**: Reads the *top chunk* of the queue without modifying it. There may be multiple peeking links accessing the same chunk simultaneously from the same or different devices. This link may be used to provide access to a lookup datastructure, already precomputed by a different compute node.
- **Modifying link**: Accesses the *top chunk* with the intention of modifying its contents. Multiple modifying links may only access one chunk simultanously, if all of them access it from the same device. The user is then responsible for making sure no race conditions occur (e.g. one modifies odd items, the other modifies even items). Modifying from two devices makes it impossible to merge both modifications. When a chunk is modified from one device, all of its envelopes on other devices are immediately released back to the pool of unused envelopes, since they no longer contain the latest chunk data. This is the only other way to release envelopes, other than consuming the entire chunk.

The important thing about these links is that when they finish, the *top chunk* will remain sitting in the queue and can be peeked or modified again (or even consumed).

We could rewrite the accumulator hub example from the previous section using one modifying link:

```cpp
auto my_node = noarr::pipelines::LambdaAsyncComputeNode("my_node");
auto acc_hub = noarr::pipelines::Hub<void*, int>(sizeof(int));
acc_hub.allocate_envelopes(Device::HOST_INDEX, 2);

auto& modifying_link = my_node.link(acc_hub.to_modify(Device::HOST_INDEX));

my_node.initialize([&](){
    auto& env = acc_hub.push_new_chunk();
    env.buffer[0] = 0;
});

my_node.advance_async([&](){
    modifying_link.envelope->buffer[0] += 1;
});

my_node.terminate([&](){
    auto& env = acc_hub.peek_top_chunk();
    std::cout << env.buffer[0] << std::endl;
});
```

You might ponder how does one combine modifying and consuming links, if they both compete for the *top chunk*. This problem is explored in the following section on [dataflow strategy](#dataflow-strategy).


## Committing links

...


## Envelope content swapping

...
- envelope content swapping, when


## Dataflow strategy

...
- dataflow strategy, why, how


## Max queue length

...
- max queue length, why


## Inner workings of links

- the link interface in detail (take from the old docs)
