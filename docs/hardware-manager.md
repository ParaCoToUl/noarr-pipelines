# Hardware manager

This section talks about the `HardwareManager` class, which is a singleton responsible for performing memory allocations, deallocations, and memory transfers between devices.

We can obtain the singleton instance like this:

```cpp
auto& manager = noarr::pipelines::HardwareManager::default_manager();
```

The two main functions of the manager are:

- `.allocate_buffer(device_index, bytes)`: Synchronously allocates a new `Buffer` instance and returns it.
- `.transfer_data(from, to, bytes, callback)`: Asynchronously transfers data from one buffer to another and calls the callback on completion.

These two functions only delegate the call down to `MemoryAllocator`s and `MemoryTransferer`s. The hardware manager also keeps a list of all known allocators and transferers and we can register our own.


## Buffer

An allocation by the `HardwareManager` returns a `Buffer` instance. The class is like a smart pointer, similar to a unique pointer. It can be moved, but not copied. It contains the following information:

- **Buffer pointer**: Points to the address where the allocated buffer resides.
- **Bytes**: The size of the buffer in bytes.
- **Device index**: The device on which the buffer was allocated.
- **Allocator pointer**: Pointer to the allocator to perform deallocation from the destructor. It may be `nullptr` if the buffer was not allocated by the hardware manager but instead created by wrapping an existing pointer.

The `Buffer` instance can exist in two modes:

1. Created by the `HardwareManager` during new data allocation.
2. Created by the user using `Buffer::from_existing(device, pointer, size)`.

In the first mode, the `Buffer` instance automatically deallocates the underlying buffer when destroyed. In the second mode, it only wraps an existing pointer and it will not attempt to deallocate it. The second mode is meant as an adapter between an external memory management system and the internal memory management of noarr pipelines. We can ask the buffer, whether it is in the second mode using `buffer.wraps_existing_buffer()`.

This class is the primary component of an envelope.


## Memory allocator

The `MemoryAllocator` class represents the interface that each specific allocator should implement. It has three required methods, and they can all be seen implemented in the following example:

```cpp
class HostAllocator : public MemoryAllocator {
public:
    Device::index_t device_index() const override {
        return Device::HOST_INDEX;
    };

    void* allocate(std::size_t bytes) const override {
        return malloc(bytes);
    };

    void deallocate(void* buffer) const override {
        free(buffer);
    };
};
```

The code shows that the allocation interface is very low-level. It works with plain void pointers.

We can register such allocator into the hardware manager like this:

```cpp
manager.set_allocator_for(
    Device::HOST_INDEX,
    std::make_unique<HostAllocator>()
);
```


## Memory transferer

The `MemoryTransferer` class represents the interface, that each specific transferer should implement. A memory transferer instance is meant to transfer from one specific device to another in that one direction. Therefore we need two transferers for each pair of devices to have full coverage. This simplifies the interface, leaving us with only one method:

```cpp
class MemoryTransferer {
public:
    virtual void transfer(
        void* from,
        void* to,
        std::size_t bytes,
        std::function<void()> callback
    ) const = 0;
}
```

Also, the transfer is treated as asynchronous, so a callback is provided. It is up to the transferer to decide how to implement the asynchronicity.

The CUDA extension registers its transferers for each device like this:

```cpp
// transferer TO
manager.set_transferer_for(
    Device::HOST_INDEX, Device::DEVICE_INDEX,
    std::make_unique<CudaTransferer>(true, 0)
);

// transferer FROM
manager.set_transferer_for(
    Device::DEVICE_INDEX, Device::HOST_INDEX,
    std::make_unique<CudaTransferer>(false, 0)
);
```

The first `true/false` argument tells the `CudaTransferer` to transfer to or from the GPU device. The second `0` argument is the CUDA device index.


## Synchronous memory transfers

The hardware manager provides a method `transfer_data_sync` which has a similar API to the asynchronous variant and it performs the transfer synchronously. It again delegates the call to a proper `MemoryTransferer` and calls the `transfer_sync` method on it.

This method has a default implementation that simply waits for the asynchronous operation to finish, but we can override it to provide a more suitable implementation.

Synchronous memory transfers are only used in hubs in the `peek_top_chunk` method if the chunk is not yet present in the requested device. Otherwise, hubs transfer data asynchronously to not block the scheduler thread.


## Dummy GPU

Sometimes, we may not have access to a GPU device, yet we want to develop and test our pipeline. We might want to do that to debug all the memory transfers and buffer allocations for the GPU (e.g. our pipeline might stop prematurely if we do not have enough envelopes allocated). We might also want to do that as a fallback in case the computer running the pipeline does not have a GPU available.

The dummy GPU, from the perspective of the pipeline, is yet another device with the index `Device::DUMMY_GPU_INDEX`. Memory transfers from the host to the dummy gpu do happen, like with any other device, but the buffers allocated for the dummy GPU are just plain `malloc` RAM allocations on the host and can be accessed by the CPU just like `Device::HOST_INDEX` buffers.

The corresponding memory transferers are not registered in the hardware manager by default, so we need to call the following method before using the dummy GPU:

```cpp
HardwareManager::default_instance().register_dummy_gpu();
```

Also, note that the dummy GPU does not fake CUDA kernel executions or any other GPU-specific operations. It only simulates the memory transfers within noarr pipelines.
