# Hub

An envelope hub is a *node* within the pipeline, that's responsible for data management. It has these primary objectives:

- **Memory transfer:** It moves data from CPU to GPU and vice versa transparently. The hub should externally appear as a plain "data buffer" that can "just be accessed" from anywhere.
- **Transfer pipelining:** Memory transfers should be overlapped with other operations to save on time. This effectively turns the internal hub structure from a single "buffer" to a queue of "buffers".

envelopes
links
allocation
data transfer
dataflow strategy
direct manipulation




```cpp
// Here are three envelopes, first with a simple array of floats,
// second with a grayscale 2D image of floats and third also a 2D image,
// but using noarr structures for the envelope structure.

// Envelope<Structure, BufferItem>

// array of floats
Envelope<std::size_t, float> env;
env.structure = 3; // the envelope will contain 3 items
env.buffer[0] = 12.5f; // set these items
env.buffer[1] = 8.2f;
env.buffer[2] = 0.1f;

// 2D array of floats
// TODO: check the sytax
Envelope<std::tuple<std::size_t, std::size_t>, float> env;
env.structure = std::make_tuple(1920, 1080);
for (std::size_t x = 0; x < std::get<0>(env.structure); ++x)
    for (std::size_t y = 0; y < std::get<1>(env.structure); ++y)
        env.buffer[x + y * std::get<0>(env.structure)] = 42.0f;

// 2D array of floats with noarr structures
// TODO: check the sytax
using Image = noarr::vector<'x', noarr::vector<'y', noarr::scalar<float>>>;
Envelope<Image> env;
env.structure = Image() | noarr::resize<'x'>(1920) | noarr::resize<'y'>(1080);
for (std::size_t x = 0; x < env.structure | noarr::length<'x'>(); ++x)
    for (std::size_t y = 0; y < env.structure | noarr::length<'y'>(); ++y)
        (env.structure | noarr::get_at<'x', 'y'>(env.buffer, x, y)) = 42.0f;
```
