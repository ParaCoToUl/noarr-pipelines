# CUDA pipelines

Noarr pipelines by itself provides only the framework for building pipelines but does not integrate with any GPGPU framework. The *cuda pipelines* extension integrates it with CUDA.

To use the extension, we need to include it and then register it:

```cpp
#include <noarr/cuda-pipelines.hpp>

noarr::pipelines::CudaPipelines::register_extension();
```

The registration adds *memory allocators* and *memory transferers* for CUDA devices into the *hardware manager*. The *hardware manager* is described in more detail in the following section.

With the extension registered, we can now use `Device::DEVICE_INDEX` to represent our primary CUDA capable device and also use the `CudaComputeNode`.


## Cuda compute node

The `CudaComputeNode` builds on top of the `AsyncComputeNode` and replaces the `advance_async` method with the `advance_cuda` method. This method receives a dedicated CUDA stream as an argument:

```cpp
auto my_node = noarr::pipelines::LambdaCudaComputeNode();

my_node.advance_cuda([&](cudaStream_t stream){
    // ...
});
```

We should use this CUDA stream for all kernel invocations and other CUDA operations. The node is built to synchronize this stream before calling `callback`, so we do not need to write the repetitive synchronization code.

The body of the `advance_cuda` also runs in a background thread, just like `advance_async` of an async compute node. In the same way, we can also specify the `advance` method and it will run before `advance_cuda` in the scheduler thread:

```cpp
auto my_node = LambdaCudaComputeNode();

my_node.advance([&](){
    // runs first, in the scheduler thread
});

my_node.advance_cuda([&](cudaStream_t stream){
    // runs second, in a background thread
});
```

The CUDA stream could also be accessed as a member variable:

```cpp
my_node.advance([&](){
    auto stream = my_node.stream; // can be accessed even from 'advance'
});
```


## Multiple GPUs

If our machine has multiple CUDA-capable graphics cards, they are all recognized during the registration of this CUDA extension. They are mapped onto device indices 0, 1, 2, 3, ...

The `Device::DEVICE_INDEX` is the device 0, but to represent other devices we can simply cast an integer to the device index:

```cpp
Device::index_t fifth_device_index = (Device::index_t) 4;
```

When we define a `CudaComputeNode`, it creates the CUDA stream for the device with index 0. We can override this behavior by specifying a constructor parameter:

```cpp
auto my_node = LambdaCudaComputeNode((Device::index_t) 1);
```

The device to which the node belongs could also be accessed as a member variable, either as a noarr pipelines device index or as a CUDA device index:

```cpp
my_node.advance_cuda([&](cudaStream_t stream){
    Device::index_t device_index = my_node.device_index;
    int cuda_device = my_node.cuda_device;
});
```

The CUDA pipelines extension registers *memory transferers* between the host and each device, but it does not *register transferers* directly between two devices. This is a feature that could be added in the future.

The extension also provides a `NOARR_CUCH` macro for checking CUDA errors. The macro can be wrapped around any CUDA function that returns `cudaError_t` and it will throw an exception in the case of an error being present.

```cpp
// check for errors after calling a kernel
my_kernel<<<...>>>(...);
NOARR_CUCH(cudaGetLastError());

// check for errors during allocations
NOARR_CUCH(cudaMalloc(&buffer, size));

// check for errors when synchronizing CUDA streams
NOARR_CUCH(cudaStreamSynchronize(stream));
```
