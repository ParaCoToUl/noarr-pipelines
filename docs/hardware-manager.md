# Hardware manager

This section talks about the `HardwareManager` class, which is a singleton responsible for performing memory allocations, dealocations and memory transfers between devices.

You can obtain the singleton instance like this:

```cpp
auto& manager = noarr::pipelines::HardwareManager::default_manager();
```

The two main function of the manager are:

- `.allocate_buffer(device_index, bytes)`: Synchronously allocates a new `Buffer` instance and returns it.
- `.transfer_data(from, to, bytes, callback)`: Asynchronously transfers data from one buffer to another and calls the callback on completion.

These two functions only delegate the call down to `MemoryAllocator`s and `MemoryTransferer`s. The hardware manager also keeps a list of all known allocators and transferers and you can register your own.


## Buffer

An allocation by the `HardwareManager` returns a `Buffer` instance. The class is like a smart pointer, similar to a unique pointer. It can be moved, but not copied. It contains the following information:

- **Buffer pointer**: Points to the address where the allocated buffer resides.
- **Bytes**: The size of the buffer in bytes.
- **Device index**: The device on which the buffer was allocated.
- **Allocator pointer**: Pointer to the allocator to perform deallocation from the destructor. May be `nullptr`, if the buffer was not allocated by the hardware manager, but instead created by wrapping an existing poitner.

The `Buffer` instance can exist in two modes:

1. Created by the `HardwareManager` during new data allocation.
2. Created by the user using `Buffer::from_existing(device, pointer, size)`.

In the first mode, the `Buffer` instance automatically deallocates the underlying buffer when destroyed. In the second mode it only wraps an existing pointer and it will not attempt to deallocate it. The second mode is meant as an adapter between an external memory management system and the internal memory management of noarr pipelines. You can ask the buffer, whether it is in the second mode using `buffer.wraps_existing_buffer()`.

This class is the primary component of an envelope.


## Memory allocator

The `MemoryAllocator` class represents the interface, that each specific allocator should implement. It has three required methods and they can all be seen implemented in the following example:

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

As you can see, the allocation interface is very low-level. It works with plain void pointers.

You can register such allocator into the hardware manager like this:

```cpp
manager.set_allocator_for(
    Device::HOST_INDEX,
    std::make_unique<HostAllocator>()
);
```


## Memory transferer

The `MemoryTransferer` class represents the interface, that each specific transferer should implement. A memory transferer instance is meant to transfer from one specific device to another in that one direction. Therefore you need two transferers for each pair of devices to have full coverage. This simplifies the interface, leaving us with only one method:

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

The cuda extension registers its transferers for each device like this:

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

The first `true/false` argument tells the `CudaTransferer` to transfer to or from the GPU device. The second `0` argument is the cuda device index.


## Synchronous memory transfers

The hardware manager provides a method `transfer_data_sync` which has similar API to the asynchronous variant and it performs the transfer synchronously. It again delegates the call to a proper `MemoryTransferer` and calls the `transfer_sync` method on it.

This method has a default implementation that simply waits for the asynchronous operation to finish, but you can override it to provide a more suitable implementation.

Synchronous memory transfers are only used in hubs in the `peek_top_chunk` method, if the chunk is not yet present in the requested device. Otherwise hubs transfer data asynchronously to not block the scheduler thread.


## Dummy GPU

The hardware manager lets you to register a dummy GPU device to test all the memory transfers without the need for having and actual GPU available. You register the dummy GPU by calling:

```cpp
HardwareManager::default_instance().register_dummy_gpu();
```

From that point on, you can use `Device::DUMMY_GPU_INDEX` device index for envelope allocations. These envelopes will be allocated in the host memory (RAM), so you can access them just like `Device::HOST_INDEX` envelopes, but they will be properly copied within hubs, as if they were in another device.
