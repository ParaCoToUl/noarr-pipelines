# Noarr Structures in pipelines

Although noarr pipelines is written as a standalone library, it is also designed to work well with Noarr Structures. Structures are meant to be used for data inside hubs and envelopes.

Here is an example of how to use structures in a hub. Since the structure type is rather complicated, we will use the `using` directive to give it an explicit name:

```cpp
// define the structure of the image
using Image = noarr::vector<'x', noarr::vector<'y',
    noarr::array<'c', 3, noarr::scalar<float>>
>>;

// Define the sized variant, because sized
// variants have different type
// (noarr::vector changes to noarr::sized_vector, etc.)
// (the size we set does not matter, so we can use zero)
using SizedImage = decltype(Image() |
    noarr::set_length<'x'>(0) | noarr::set_length<'y'>(0)
);

// Create a hub to hold one image as one chunk.
//
// The second template argument can be omitted,
// because the structure already defines buffer item types.
//
// We use the sized structure variant, because that
// is what is going to be needed for the data to be
// stored and the type cannot change during the
// lifetime of a hub.
auto my_hub = noarr::pipelines::Hub<SizedImage>(

    // Calculate the size of the allocated memory
    // using noarr structures as well.
    // (store 1920x1080 images at most)
    Image() | noarr::set_length<'x'>(1920) |
    noarr::set_length<'y'>(1080) | noarr::get_size()
);
```

Now we can use the corresponding envelopes in our compute node:

```cpp
auto& my_link = my_node.link(my_hub.to_produce(Device::HOST_INDEX));

my_node.advance_async([&](){
    // set image dimensions
    my_link.envelope->structure = Image()
        | noarr::set_length<'x'>(1280)
        | noarr::set_length<'y'>(720);

    // set each individual pixel color like this
    std::size_t x = 10;
    std::size_t y = 15;
    my_link.envelope->structure
        | noarr::get_at<'x', 'y', 'c'>(
            my_link.envelope->buffer,
            x, y, 0
        ) = 0.8f;
});
```

Noarr pipelines also provide an extension that gives us some helper functions for working with noarr structures via bags:

```cpp
// include the extension
#include <noarr/structures-pipelines.hpp>

auto& my_link = my_node.link(my_hub.to_produce(Device::HOST_INDEX));

my_node.advance_async([&](){
    // set image dimensions (same as before)
    my_link.envelope->structure = Image()
        | noarr::set_length<'x'>(1280)
        | noarr::set_length<'y'>(720);

    // use the envelope in the link as a bag
    auto my_bag = noarr::pipelines::bag_from_link(my_link);

    // set each individual pixel color like this
    // (much shorter syntax than before)
    std::size_t x = 10;
    std::size_t y = 15;
    my_bag.template at<'x', 'y', 'c'>(x, y, 0) = 0.8f;
});
```

The extension provides two self-explanatory functions:

- `bag_from_envelope(envelope)`
- `bag_from_link(link)`

The constructed bag receives a copy of the structure, so we should only create and use the bag after we are done modifying the envelope structure.
