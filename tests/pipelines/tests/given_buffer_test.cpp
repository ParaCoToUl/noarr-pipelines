#include <catch2/catch.hpp>

#include <string>
#include <memory>
#include <iostream>

#include <noarr/pipelines.hpp>

using namespace noarr::pipelines;

TEST_CASE("Given buffer", "[buffer][hub]") {
    
    // an existing C pointer with, say a memory-mapped file
    int* ptr = (int*) malloc(sizeof(int) * 1024);
    ptr[0] = 42;
    ptr[1] = 43;
    ptr[2] = 44;
    
    // create a hub
    auto my_hub = Hub<std::size_t, int>(sizeof(int) * 1024);

    // put our buffer into the hub as an empty envelope
    Buffer my_buffer = Buffer::from_existing(
        Device::HOST_INDEX,
        ptr,
        sizeof(int) * 1024
    );
    my_hub.create_envelope(std::move(my_buffer));

    // and use that envelope in a new chunk
    auto& env = my_hub.push_new_chunk();

    // check that the chunk indeed uses the buffer
    REQUIRE(env.buffer[0] == 42);
    REQUIRE(env.buffer[1] == 43);
    REQUIRE(env.buffer[2] == 44);

    // also create a node to be run once and link it
    auto my_node = LambdaComputeNode();
    auto& link = my_node.link(my_hub.to_modify(Device::HOST_INDEX));

    bool was_run = false;

    my_node.can_advance([&](){
        return !was_run;
    });

    my_node.advance([&](){
        // check that we received the envelope with the proper buffer
        REQUIRE(link.envelope->buffer[0] == 42);
        REQUIRE(link.envelope->buffer[1] == 43);
        REQUIRE(link.envelope->buffer[2] == 44);
        
        was_run = true;
        my_node.callback();
    });

    // run the compute node
    DebuggingScheduler scheduler;
    scheduler << my_hub << my_node;
    scheduler.run();

    // check that the compute node ran
    REQUIRE(was_run);

    // free the underlying C buffer
    // (the "Buffer" buffer was destroyed by the hub)
    free(ptr);
}
