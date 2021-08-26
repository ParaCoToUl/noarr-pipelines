# Hub

*Hubs* are pipeline *nodes* designed to handle memory management. They are connected to *compute nodes* by *links* of various types and the data is passed through these links in the form of *envelopes*. The hub is designed to handle a number of responsibilities:

- **Queuing chunks**: The hub is implemented as a queue of *chunks*. Some *links* produce new *chunks* and other *links* consume *chunks*. This lets the producer and consumer *nodes* run independently.
- **Memory transfer**: Each one of the *links* may want to access the data on a different device. The *hub* has to perform memory transfers so that these requirements are satisfied. A memory transfer involves creating a copy of one envelope on another device. This means that a *chunk* of data is represented by a set of *envelopes* on different devices, all holding the same data. If the *hub* has enough available *envelopes*, it can overlay these memory transfers with other operations (like producing and consuming) to increase the throughput of the pipeline.
- **Envelope lifetime**: A *hub* manages a pool of *envelopes* and uses them to store the data passing through it. When we create a *hub*, we specify how many envelopes on what devices should be allocated. When a *compute node* wants to produce a chunk, the *hub* will give it an empty *envelope* to be populated by the data, and then it inserts the *envelope* into the internal queue as a new *chunk*. Then this *chunk* may be copied to other devices, using up more *envelopes* from the pool of unused envelopes. When a *chunk* is consumed, all of its *envelopes* are returned to the pool of unused envelopes. An envelope may never move from one *hub* to another.


## Produce-transfer-consume in detail

To get the inner workings of a hub fully across, we demonstrate the following diagram. It shows a hub with one producing a link on the CPU and one consuming link on the GPU. The hub starts with two empty envelopes on each device:

```txt
Pool of unused envelopes:
[Env_A, CPU: <empty>], [Env_B, CPU: <empty>], [Env_C, GPU: <empty>],
[Env_D, GPU: <empty>]

Chunk queue:
<empty>
```

> **Note:** An empty envelope is not actually empty by having a zero size. It is empty in the same sense an uninitialized variable is empty - it contains some data but the data is considered to have no meaning.

> **Note:** The proper names for devices are *the host* and *the device*, but we use the abbreviations CPU and GPU in this example since they are shorter and easier to understand.

When the producing link wants to produce a new chunk, it is given an unused envelope for the CPU (the envelope `Env_A`). When the production is committed, the envelope is inserted into the queue of chunks:

```txt
Pool of unused envelopes:
[Env_B, CPU: <empty>], [Env_C, GPU: <empty>], [Env_D, GPU: <empty>]

Chunk queue:
Chunk( [Env_A, CPU: Chunk_1] )
```

Since the consumer wants to receive the data for the GPU, the hub transfers the chunk to the GPU device:

```txt
Pool of unused envelopes:
[Env_B, CPU: <empty>], [Env_D, GPU: <empty>]

Chunk queue:
Chunk( [Env_A, CPU: Chunk_1] )
     ( [Env_C, GPU: Chunk_1] )
```

While the transfer was happening, the producer was able to produce another chunk of data because there was an empty envelope available (`Env_B`):

```txt
Pool of unused envelopes:
[Env_D, GPU: <empty>]

Chunk queue:
Chunk( [Env_B, CPU: Chunk_2] )  Chunk( [Env_A, CPU: Chunk_1] )
     (                       ),      ( [Env_C, GPU: Chunk_1] )
```

Now the consumer has its data ready, and so it receives the envelope `Env_C`. When the consumption finishes, the entire last chunk is deleted and its envelopes return to the pool:

```txt
Pool of unused envelopes:
[Env_D, GPU: <empty>], [Env_A, CPU: <empty>], [Env_C, GPU: <empty>]

Chunk queue:
Chunk( [Env_B, CPU: Chunk_2] )
```

> **Note:** The chunk that will be consumed is called the *top chunk*. It is the oldest chunk in the queue. It is the one on the dequeuing end of the queue.


## The flow of envelopes

We already have a decent idea of how envelopes flow through a hub, now we will go over it once more in full detail.

**Allocation:** Envelopes are explicitly allocated right after we create a hub. Their size is the same for all envelopes in one hub and it is provided through the constructor of the hub. By the manual allocation, we specify how many of them should be available for what device.

```cpp
auto my_hub = Hub<std::size_t, float>(
    sizeof(float) * 1024 // size of all envelopes in my_hub
);

my_hub.allocate_envelopes(Device::HOST_INDEX, 2); // 2 on the host
my_hub.allocate_envelopes(Device::DEVICE_INDEX, 4); // 4 on the device
```

The number of envelopes impacts the behavior of the hub. If we have no envelopes for a given device and a consuming link requests data for that device, it may never be satisfied. The consuming node will not be able to advance and advancing the hub will not solve the situation either. Therefore no nodes can be advanced and the pipeline terminates as if it has successfully done all the work. Having only one envelope allows the pipeline to function, but it cannot overlay any two operations, so it either produces or transfers or consumes. Having two envelopes is probably optimal in most situations that use the hub as a queue. There are, however, other ways to use the hub that may require different numbers of envelopes for optimal performance.

Although it is typical to allocate all envelopes during the construction of the hub, we may also call the `allocate_envelope(s)` methods anytime we want. The hub will be able to use the new envelope right after the allocation finishes. The only constraint is that allocations should be executed by the scheduler thread only.

**Unused pool:** When an envelope is allocated, it enters the so-called *pool of unused envelopes*. This pool contains envelopes that are not present in the chunk queue and can be used for new chunks or memory transfers. Since each envelope belongs to only one device, we could imagine that this pool is multiple pools - one for each device. If this pool becomes empty of envelopes for a given device, no chunk production or memory transfers for that device may occur. The hub waits for chunks in the queue to be consumed, so that envelopes get released and can be reused.

**Chunk queue:** When a producing link receives an envelope from the unused pool and the production is committed (if not, the envelope returns to the unused pool), the envelope enters the chunk queue as the only envelope in a new chunk. If the consuming link requests the same device, the envelope just sits through the queue and when the chunk is consumed, the envelope is returned to the unused pool. If the consuming link requests a different device, memory transfer has to take place before consumption is allowed.

The hub goes over chunks in the queue from the oldest to the newest and for each one of them tries to make sure it has an envelope for the requested device. If not, a memory transfer begins. First, an empty envelope for the target device is taken from the unused pool. If the pool has no such envelope, the transfer does not start and waits for such an envelope to become available. Once we have the target envelope, we look at all the envelopes in our chunk and choose the one from which to copy. There has to be at least one available since each chunk in the queue must have at least one envelope present (the one that produced the chunk). We prioritize the host-device and device-host transfers if possible, and if not, a device-device transfer will happen (two different graphics cards).

When the memory transfer finishes, the source envelope is not released. It remains in the queue until its entire chunk is consumed. This is because the chunk may still be read from that device. There are situations, where the hub can be used for holding data that is often read from multiple devices and consumed only rarely from one of these devices.

**Trashed envelopes:** When the *top chunk* is consumed and its envelopes released, they do not immediately go to the unused pool. They first enter the pool of *trashed* envelopes. These are envelopes that are about to become *unused* but may still have some links looking at them. It may happen that multiple links read the *top chunk*, but only one link consumes it. The other links are still reading the chunk for a while after it has been consumed. The pool of trashed envelopes makes sure all these links finish before these envelopes are fully released.

**Deallocation:** Consuming the *top chunk* will not deallocate its envelopes. It only releases them back to the unused pool. Proper deallocation happens only during the destruction of the hub.


## Manual chunk queue manipulation

We have described how *links* can produce and consume chunks. It might sometimes be needed to do these actions manually. We can imagine a scenario, where we have a compute node that uses one hub as an accumulator. With each iteration, it simultaneously consumes one chunk from the hub and produces one chunk as a replacement. We need to initialize the accumulator hub with some zero-valued chunk when the pipeline starts and then read out the final value when the pipeline finishes.

There are a few functions that should help us in such a scenario:

```cpp
// Takes an envelope from the pool of unused envelopes for the host
// device, inserts it into the chunk queue as a new chunk and returns
// a reference to it.
auto& envelope = my_hub.push_new_chunk();
envelope.buffer[0] = 0;

// Returns the reference to the host device envelope of the top chunk.
// Fails if the chunk queue is empty. Performs a synchronous memory
// transfer of the chunk to the host device if there is no host envelope
// for the chunk. The memory transfer may fail due to the lack
// of unused envelopes.
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

auto& producing_link = my_node.link(
    acc_hub.to_produce(Device::HOST_INDEX)
);
auto& consuming_link = my_node.link(
    acc_hub.to_consume(Device::HOST_INDEX)
);

my_node.initialize([&](){
    auto& env = acc_hub.push_new_chunk();
    env.buffer[0] = 0;
});

my_node.advance_async([&](){
    consuming_link.envelope->buffer[0]
        = producing_link.envelope->buffer[0] + 1;
});

my_node.terminate([&](){
    auto& env = acc_hub.peek_top_chunk();
    std::cout << env.buffer[0] << std::endl;
});
```

> **Note:** Using a hub for a single integer value is a bad idea, we should use a regularly shared variable for that. This is just to demonstrate the concept. We should use hubs only when we benefit from their features.

We may also want to push or peek a chunk from another device than the host. The two functions let us specify the device index in the first argument:

```cpp
auto& gpu_envelope = my_hub.push_new_chunk(Device::DEVICE_INDEX);
my_init_kernel<<<...>>>(gpu_envelope.buffer);

auto& gpu_envelope = my_hub.peek_top_chunk(Device::DEVICE_INDEX);
my_other_kernel<<<...>>>(gpu_envelope.buffer);
```


## Peeking and modifying links

We have talked about producing links and consuming links. They are essential for understanding the hub as a queue of chunks. But there are two other types of links that let us work with a hub as if it was a buffer holding one piece of data. These two types are:

- **Peeking link**: Reads the *top chunk* of the queue without modifying it. There may be multiple peeking links accessing the same chunk simultaneously from the same or different devices. This link may be used to provide access to a lookup data structure, already precomputed by a different compute node.
- **Modifying link**: Accesses the *top chunk* to modify its contents. Multiple modifying links may only access one chunk simultaneously if all of them access it from the same device. The user is then responsible for making sure no race conditions occur (e.g. one modifies odd items, the other modifies even items). Modifying from two devices makes it impossible to merge both modifications. When a chunk is modified from one device, all of its envelopes on other devices are immediately released back to the pool of unused envelopes, since they no longer contain the latest chunk data. This is the only other way to release envelopes, other than consuming the entire chunk.

The important thing about these links is that when they finish, the *top chunk* will remain sitting in the queue and can be peeked or modified again (or even consumed).

We could rewrite the accumulator hub example from the previous section using one modifying link:

```cpp
auto my_node = noarr::pipelines::LambdaAsyncComputeNode("my_node");
auto acc_hub = noarr::pipelines::Hub<void*, int>(sizeof(int));
acc_hub.allocate_envelopes(Device::HOST_INDEX, 2);

auto& modifying_link = my_node.link(
    acc_hub.to_modify(Device::HOST_INDEX)
);

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

We might ponder how one combine modifying and consuming links if they both compete for the *top chunk*. This problem is explored in the following section on [dataflow strategy](#dataflow-strategy).

Here is the list of all the link types and their construction:

```cpp
bool autocommit = true | false; // true if omitted from arguments

auto& link = node.link(hub.to_produce(Device::HOST_INDEX, autocommit));
auto& link = node.link(hub.to_consume(Device::HOST_INDEX, autocommit));
auto& link = node.link(hub.to_modify(Device::HOST_INDEX, autocommit));
auto& link = node.link(hub.to_peek(Device::HOST_INDEX));
```


## Committing links

When we create a new link, if it is not a peeking link, it has a second boolean argument called `autocommit`. The producing, consuming, and modifying links are special in that they perform an operation that changes the state of the hub (adding, removing, or modifying a chunk). But we sometimes might want to have a producing link, that does not produce always. Or a consuming link that sometimes does not consume. That is why we define the term *committing* as a link. The link tries to perform an action. If that action was indeed performed, the link was *committed*.

An example would be producing chunks from a file for which we do not know the size. We might try to produce a new chunk but we reach the end of the file at that point. Therefore we do not commit the producing link, since we did not even use it.

The `autocommit` argument for links specifies, whether the link needs to be committed manually, or whether it is always committed automatically.

A producing link that has to be manually committed can look like this:

```cpp
auto& link = my_node.link(my_hub.to_produce(
    Device::HOST_INDEX,
    false // disable autocommit
));

my_node.advance_async([&](){
    std::string line;

    if (std::getline(file, line)) {

        // write data into the prepared envelope
        link.envelope->structure = line.size();
        line.copy(link.envelope->buffer, line.size());
        
        // commit the link - we did actually produce a chunk
        link.commit();

    } else {
        // do nothing
    }
});
```

The committing behavior of links was already described. The following list summarizes what happens if a link is not committed.

- **Producing**: When a producing link is not committed, its envelope returns to the pool of unused envelopes, and the chunk queue does not change.
- **Consuming**: When a consuming link is not committed, the *top chunk* is not removed from the queue. We should not modify the provided envelope if we do not consume the chunk, as it would make the top chunk inconsistent between devices.
- **Modifying**: When a modifying link is not committed, envelopes on other devices are not released to the pool of unused envelopes. We should not modify the provided envelope if we do not commit the link, as it would make the top chunk inconsistent between devices.
- **Peeking**: It makes no sense to define commits for peeking links, as they do not modify the state of the hub in any way.


## Envelope content swapping

Envelopes give us the option of swapping their contents by doing:

```cpp
first_envelope.swap_contents_with(other_envelope);
```

This swap is very efficient, as it does not copy the content, but instead swaps the buffer pointers and structures. This, on the other hand, gives us the limitation, that this kind of swap can only be performed, if the two envelopes have the same type, size and belong to the same device.

This trick can be used to speed up the pipeline, if we have a compute node that consumes chunks from one hub and moves them to another hub with minimal modifications:

```cpp
auto& input_link = my_node.link(
    input_hub.to_consume(Device::HOST_INDEX)
);
auto& output_link = my_node.link(
    output_hub.to_produce(Device::HOST_INDEX)
);

my_node.advance_async([&](){
    // perform minimal in-place modifications
    input_link.envelope->buffer[42] *= 6;

    // and then do the swap to avoid expensive copying
    input_link.envelope->swap_contents_with(*output_link.envelope);
});
```

The performed modifications need not be small, a better defining feature is that they are in place.


## Envelopes from existing buffers

Sometimes we have an existing pointer to some data, say a memory-mapped file, an already allocated CUDA pointer, etc... We can provide this pointer to a hub and the hub will wrap it inside a new envelope. This envelope will then flow through the hub just like any other envelope:

```cpp
// an existing C pointer we want to use
int* ptr = (int*) malloc(sizeof(int) * 1024);

// create a hub
auto my_hub = Hub<std::size_t, int>(sizeof(int) * 1024);

// wrap the pointer in a buffer instance first
Buffer my_buffer = Buffer::from_existing(
    // where is the buffer located
    Device::HOST_INDEX,
    
    // the pointer
    ptr,

    // the buffer size, has to match the hub buffer size
    // (the argument of the hub constructor)
    sizeof(int) * 1024
);

// give the buffer instance to the hub
my_hub.create_envelope(std::move(my_buffer));

// ... rest of the pipeline ...

// free the C pointer
free(ptr);
```

The `create_envelope` function works just like the `allocate_envelope` function, but it receives the buffer as an argument, instead of allocating it.

The `Buffer` instance is a lower-level object that envelopes internally use. Typically, it manages an automatically allocated memory and so when the `Buffer` is destroyed, the memory is freed (just like `std::unique_ptr`), but since we already have memory allocated, we want to just wrap it. That is why we use the `Buffer::from_existing` method. In this mode, the `Buffer` instance does not handle memory management and acts only as a wrapper. The underlying wrapped C buffer has to be `free`d manually. The `Buffer` class is documented later in the section on the [Hardware Manager](./hardware-manager.md#buffer).

Envelopes created from an existing buffer act just like any other envelopes with one exception: They cannot have their contents swapped by the `swap_contents_with` function. This is because the swapping transfers buffers between hubs which makes it more difficult to track them through the pipeline, so we decided to disallow it.


## Dataflow strategy

We talked about various link types and how they behave, but when we demonstrated them, we never combined many links of different types together (e.g. consuming and modifying). This section describes why and how to do that.

Say we have a hub with one chunk manually inserted during initialization. One compute node modifies the chunk and the other compute node consumes the chunk when all the modifications are done. The pipeline looks like this:

```txt
        ,----- to_modify  ----> [modifier_node]
{my_hub}
        '----- to_consume ----> [consumer_node]
```

From the perspective of the hub, we do not know which link should we make ready. We could provide the chunk to both links but that would not end up behaving the way we want. The problem is even worse if both links require their data on different devices. We do not know to which device should we transfer the chunk.

To solve this issue, the hub remembers a *dataflow strategy*. A *dataflow strategy* is realized as a set of links, to which we want the data to flow. In our example, we would want the hub to begin by *flowing data* to the modifier node. When the modifier has performed all the modifications we would switch the *dataflow* to the consumer node. In each case, the *dataflow* contains only one link and never both of them, therefore the hub always knows how to behave.

The example could be realized by the following code:

```cpp
auto my_hub = noarr::pipelines::Hub<std::size_t, float>(
    sizeof(float) * 1024
);
my_hub.allocate_envelope(Device::HOST_INDEX);

auto modifier_node = LambdaAsyncComputeNode("modifier_node");
auto consumer_node = LambdaAsyncComputeNode("consumer_node");

auto& modifier_link = modifier_node.link(
    my_hub.to_modify(Device::HOST_INDEX)
);
auto& consumer_link = consumer_node.link(
    my_hub.to_consume(Device::HOST_INDEX)
);

// the hub starts out with a chunk present in the queue
// to be modified and then consumed
my_hub.push_new_chunk();

// at the beginning, the hub flows data to the modifier
my_hub.flow_data_to(modifier_link);

bool all_modifications_done = false;

modifier_node.advance_async([&](){
    // perform some modification via modifier_link
    // and when the conditions are right:
    all_modifications_done = true;
});

// change the dataflow in post_advance since the it can only be changed
// by the scheduler thread
modifier_node.post_advance([&](){
    if (all_modifications_done) {
        hub.flow_data_to(consumer_link);
    }
});

consumer_node.post_advance([&](){
    // consume the chunk via consumer_link
});
```

As we can see, the *dataflow strategy* can be changed by calling:

```cpp
my_hub.flow_data_to(my_link);
```

> **Notice:** Functions specified here (for modifying the dataflow) may only be called from the *scheduler thread*.

We only really need to specify the data flow if we have multiple consuming, modifying, or peeking links. If we have only one such link, the strategy is inferred implicitly by the hub during the pipeline initialization. This inferring can also be triggered explicitly:

```cpp
bool success = my_hub.infer_dataflow();
```

The inference only succeeds if there is only one consuming, modifying, or peeking link present.

Producing links may not be part of a *dataflow policy* since the data does not flow towards them, but away from them. The order in which producing links are made ready is not deterministic and should not be relied on (if we have multiple producing links for a single hub). The synchronization has to be provided by the user.

The order is also non-deterministic for non-producing links if there is more than one of them in the *dataflow strategy*. This might happen when we have multiple peeking links independently reading the same data. The *dataflow strategy* can be defined like this:

```cpp
my_hub.flow_data_to(my_peeking_link);
my_hub.and_flow_data_to(my_other_peeking_link);
```

We could add a variable number of nodes in a for loop by first resetting the dataflow to empty:

```cpp
my_hub.reset_dataflow();
for (auto& link : some_links) {
    my_hub.and_flow_data_to(link);
}
```

By calling `reset_dataflow()`, we empty the set of links to which data should flow. This is a valid state, but may be useless. If there is no link for data to flow to, the hub does not prepare data for any non-consuming links and the pipeline will prematurely terminate.

Lastly, just for convenience, we can provide not only links to the mentioned functions but also nodes. The hub will resolve the correct link from the given node:

```cpp
my_hub.flow_data_to(my_node);
```


## Max queue length

When we have a hub with a producing link, it will try to produce new chunks for as long as it has unused envelopes. This might not be the desired behavior, as it consumes all unused envelopes and that might starve the hub for envelopes for memory transfers. To remedy this problem, we can set a maximum size of the chunk queue:

```cpp
my_hub.set_max_queue_length(2);
```

This will make sure the hub will not accept any more chunks if the queue length reaches the maximum length. Thus there will remain spare unused envelopes, available for memory transfers.

This is only useful in really unusual situations when we create a chunk from the CPU, modify it from the GPU, and then consume it again from the CPU (needs to be transferred back to the CPU). Consider whether it is not advantageous to use multiple hubs instead if we come across needing this feature.

Setting the queue length to zero removes the length limitation:

```cpp
my_hub.set_max_queue_length(0); // remove the limit
```

This function should only be called from the *scheduler thread*.


## Inner workings of links

From the perspective of the scheduler, it does not matter where the data in the pipeline comes from. It just advances nodes when they can be advanced. But since we typically think of programs as having a data structure and an algorithm, we devised two types of nodes: hubs and compute nodes. There is currently no need to develop other kinds of data-holding nodes, but if that were to become a need, this section describes the conceptual framework.

Data is exchanged between nodes via links. There is the node that owns the data - the *host*. And the node that acts on the provided data - the *guest* (e.g. the hub hosts envelopes to compute nodes). This terminology governs the structure of a link.

A link starts out with no envelope attached. It is the responsibility of the host to acquire an envelope and attach it to the link. Whether the envelope contains data or not depends on the host and the link type can be used to guide the decision. The moment an envelope is attached by the host, the link transitions to a *fresh* state. Now, it is time for the guest to act. Guest can look at the envelope and do with it what it sees fit. Once the guest is done, it switches the link to a *processed* state. Now the host can detach the envelope and continue working with it. This is how one node can lend an envelope to another node.

Apart from the link state and the attached envelope, the link also has additional information present, that may or may not be used by both interacting sides. The link has a type and it has a `committed` flag. There are four types of links:

- producing
- consuming
- peeking
- modifying

These types describe the kind of interaction the guest wants to make with the data. When producing, the host provides an empty envelope and the guest fills it with data. The opposite happens during consumption. During peeking and modification, the host provides some existing data to the guest, who can either look at or modify the data. The `committed` flag is set by the guest during production, consumption, or modification to signal that the action was in fact performed. This lets the guest, for example, not consume a chunk of data if it does not like the chunk.

A link also has an `autocommit` flag that simply states that the action will be committed automatically when the envelope lending finishes. If this flag is set to `false`, the guest node has to manually call a `commit()` method before finishing the envelope lending.

The link also has an associated device, that specifies on which device should the hosted envelope exist.

To learn more about the envelope hosting interplay, read the `Link.hpp` file and the `ComputeNode.hpp` file.


## Dataflow in detail

Each time a hub is advanced by the scheduler, it attempts to do three things:

1. satisfy producing links
2. satisfy consuming, modifying, or peeking links
3. transfer data according to the dataflow strategy

*Production satisfaction* is only conditioned by having empty envelopes for the given device. There is no discrimination between producing links - who gets an empty envelope is allowed to produce.

*Data transfer* is also eager. If there is an empty envelope available for a device in the data flow, the hub will try to transfer the data to that device.

> **Note:** Dataflow does not contain devices, it contains links. But we can look at all the links and get a set of devices to transfer to, which is what ultimately interests the memory-transferring part of a hub.

Lastly the *satisfaction of consuming, modifying, and peeking links* is also eager. If the top chunk has an envelope on the device of such link, it will make that link ready. This means one envelope can be put into multiple links (say peeking) and that is not a problem. The same thing may occur for modification or consumption. What prevents race conditions from occurring is the dataflow strategy (it is the user's responsibility to not combine peeking with modifying and if so, dealing with race conditions themselves).

> **Note:** This kind of usage is not disallowed as the user might want, for example, parallel modification of one envelope, where one link accesses only odd items and the other only even items in some array.

If a consumption or modification is committed, it causes some envelopes to be released. If there are other links still looking at those envelopes, the hub also handles this situation: it will wait for these other links to finish their job before releasing those envelopes completely and reusing them. The same behavior might occur during a dataflow strategy change.

If there are multiple modifying (or consuming) links that commit on the same top chunk, the whole execution aborts with an error. It is unclear what should happen in such a scenario and we should make sure it does not happen.

Even though there are many strange situations that may occur (and are handled properly), there are additional constraints on the dataflow strategy that make sure the dataflow makes sense (e.g. data cannot flow to producing links, it cannot flow to multiple consuming links, when having multiple modifying links they have to be on the same device).

> **Note:** Really, these scenarios are edge-cases and not something a typical pipeline would use. Nonetheless, they are explained here so we have a better understanding of how a hub works.
