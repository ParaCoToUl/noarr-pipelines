# Hub

An envelope hub is a pipeline node, that is responsible for data management. It has these primary objectives:

- **Memory transfer:** It moves data from CPU to GPU and vice versa transparently. The hub should externally appear as a plain "data buffer" that can "just be accessed" from anywhere.
- **Transfer pipelining:** Memory transfers should be overlapped with other operations to save on time. This effectively turns the internal hub structure from a single "buffer" to a queue of "buffers" called envelopes.


## Envelopes

The section on core principles already briefly mentioned what envelopes are: An envelope holds a chunk of data that has a structure and the data itself. Also, envelopes are the pieces of data that are shared to other nodes via links.

The separation of envelope content into a structure and a data buffer is not a coincidence. Envelopes are designed to play well with noarr structures. That being said, they can also be used without noarr structures in simple situations. The following three examples demonstrate how an envelope may be used to store data:

```cpp
// Here are three envelopes, first with a simple array of floats,
// second with a grayscale 2D image of floats and third also a 2D image,
// but using noarr structures for the envelope structure.

// API: Envelope<Structure, BufferItem>

// array of floats
Envelope<std::size_t, float> env;
env.structure = 3; // the envelope will contain 3 items
env.buffer[0] = 12.5f; // set these items
env.buffer[1] = 8.2f;
env.buffer[2] = 0.1f;

// 2D array of floats
Envelope<std::tuple<std::size_t, std::size_t>, float> env;
env.structure = std::make_tuple(1920, 1080);
for (std::size_t x = 0; x < std::get<0>(env.structure); ++x)
    for (std::size_t y = 0; y < std::get<1>(env.structure); ++y)
        env.buffer[x + y * std::get<0>(env.structure)] = 42.0f;

// 2D array of floats with noarr structures
using Image = noarr::vector<'x', noarr::vector<'y', noarr::scalar<float>>>;
Envelope<Image> env;
env.structure = Image() | noarr::resize<'x'>(1920) | noarr::resize<'y'>(1080);
for (std::size_t x = 0; x < env.structure | noarr::length<'x'>(); ++x)
    for (std::size_t y = 0; y < env.structure | noarr::length<'y'>(); ++y)
        (env.structure | noarr::get_at<'x', 'y'>(env.buffer, x, y)) = 42.0f;
```

When you create a hub, you specify the type of envelopes it will contain and their size:

```cpp
auto my_hub = Hub<StructureType, BufferItemType>(envelope_sizes_in_bytes);
```

All envelopes within a hub will have the same type (e.g. all will hold 2D images) and the same maximum capacity, but they will be located on different devices and the data transfer between devices is handled automatically.


## Allocation

When you create a hub, you have to immediately specify how many envelopes you want to allocate on what devices. Here is the syntax:

```cpp
auto my_hub = Hub<StructureType, BufferItemType>(envelope_sizes_in_bytes);

// allocate two envelopes on CPU and two on GPU
my_hub.allocate_envelopes(Device::HOST_INDEX, 2);
my_hub.allocate_envelopes(Device::DEVICE_INDEX, 2);
```

These allocated envelopes enter a pool of empty envelopes that are used when new chunks are to be produced into the hub or when a chunk needs to be copied to another device. Having no envelopes for new chunks means the hub cannot be advanced and that might cause the pipeline to terminate prematurely. Having no envelopes for data transfer will abort the entire execution with an error.

The underlying allocation and memory transfer logic is handled by a hardware manager, described in a following section.


## Dataflow strategy

A hub is internally a queue of chunks. Each chunk can have multiple envelopes with the same data on multiple devices. When a producing link produces a chunk of data (e.g. on the CPU), it enters the queue. The chunk then passes through the queue and is copied to proper devices (e.g. the GPU). When it reaches the end of the queue it becomes the *top chunk*. Only the top chunk is subject to consumption, peeking and modification. Consumption is straight forward: when a consuming link finishes, the top chunk is removed from the queue and its envelopes are returned to the pool of empty envelopes. Peeking links access the top chunk for reading only. This means there may be multiple peeking links on multiple devices running simultaneously. Modifying links work like peeking links, but they may only access the chunk from one device at a time and when they finish, they invalidate envelopes of that chunk on other devices. During both peeking and modification, the top chunk remains at the top of the queue.

The hub remembers a *dataflow strategy*. It is a set of links, to which we want the data to flow. This in turn translates to a set of devices and it informs the memory transfer that should occur in the queue. This is how you specify the dataflow:

```cpp
// flow to three links
my_hub.flow_data_to(some_link);
my_hub.and_flow_data_to(some_other_link);
my_hub.and_flow_data_to(some_yet_another_link);

// flow to one link that points to this node
// (has to be non-ambiguous)
my_hub.flow_data_to(some_node);

// reset flow to empty and then set links in a loop
my_hub.reset_dataflow();
for (...) {
    my_hub.and_flow_data_to(...);
}
```

The dataflow may also be empty (if you reset it) and that prevents any memory transfers from happening. This might block the hub, making it unable to advance and thus stopping your pipeline prematurely.

If there is only one consuming, peeking or modifying link and you do not provide an explicit strategy, the hub can infer the dataflow strategy automatically.

Dataflow strategy may change during execution of your pipeline and it is actually the proper thing to do in many cases. You can do that from any method that runs on the scheduler thread (e.g. `post_advance` or `initialize`).


## Direct chunk manipulation

...


## Max queue length

...


## Dataflow in detail

Each time a hub is advanced by the scheduler it attempts to do three things:

1. satisfy producing links
2. satisfy consuming, modifying or peeking links
3. transfer data according to the dataflow strategy

Production satisfaction is only conditioned by having empty envelopes for the given device. There is no discrimination between producing links, who has an empty envelope will be allowed to produce.

Data transfer is also eager. If there is an empty envelope, it will try to transfer the data to that device (according to the dataflow, of course).

Lastly the satisfaction of consuming, modifying and peeking links is also eager. If the top chunk has an envelope on the link's device, it will make that link ready. This means one envelope can be put into multiple links (say peeking) and that is not a problem. Same thing may occur for modification or consumption. What prevents race conditions from occurring is the dataflow strategy (it is the user's responsibility to not combine peeking with modifying and if so, dealing with race conditions themselves).

> **Note:** This kind of usage is not disallowed as the user might want (for example) parallel modification of one envelope where one link accesses only odd items and the other only even items in some array.

If a consumption or modification is committed, it causes some envelopes to be freed up. If there are other links still looking at those envelopes, the hub also handles this situation: it will wait for these other links to finish their job before freeing those envelopes completely and reusing them. Same might occur during a dataflow strategy change.

If there are multiple modifying (or consuming) links that commit on the same top chunk, the whole execution aborts with an error. It is unclear what should happen in such scenario and you should make sure it does not happen.

Even though there are many strange situations that may occur (and are handled properly), there are additional constraints on the dataflow strategy that make sure the dataflow makes sense (e.g. data cannot flow to producing links, it cannot flow to multiple consuming links, when having multiple modifying links they have to be on the same device).

> **Note:** Really, these scenarios are edge-cases and not something a typical pipeline would use. Nonetheless they are explained here so you have a better understanding of how a hub works.
