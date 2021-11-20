#include <catch2/catch.hpp>

#include <string>
#include <memory>
#include <iostream>

#include <noarr/pipelines/Device.hpp>
#include <noarr/pipelines/Buffer.hpp>
#include <noarr/pipelines/Hub.hpp>
#include <noarr/pipelines/HardwareManager.hpp>
#include <noarr/pipelines/DebuggingScheduler.hpp>
#include <noarr/pipelines/LambdaComputeNode.hpp>

using namespace noarr::pipelines;

TEST_CASE("Hub peeking with transfer", "[memory_transfer][hub]") {
    auto& manager = HardwareManager::default_manager();
    manager.register_dummy_gpu();

    Hub<void*, int> hub(1024);
    hub.allocate_envelope(Device::HOST_INDEX);
    hub.allocate_envelope(Device::DUMMY_GPU_INDEX);

    bool advanced_once = false;

    auto node = LambdaComputeNode(); {
        auto& link = node.link(hub.to_modify(Device::DUMMY_GPU_INDEX));

        node.initialize([&](){
            // write to hub from the host
            auto& env = hub.push_new_chunk(Device::HOST_INDEX);
            env.buffer[0] = 1;
            env.buffer[1] = 2;
            env.buffer[2] = 3;
        });

        node.can_advance([&](){
            return !advanced_once;
        });

        node.advance([&](){
            auto& env = *link.envelope;

            // modify the hub from the dummy gpu
            env.buffer[0] += 1;
            env.buffer[1] += 1;
            env.buffer[2] += 1;

            node.callback();
        });

        node.post_advance([&](){
            advanced_once = true;
        });

        node.terminate([&](){
            // read the hub from the host again
            // (synchronous memory transfer is triggered here)

            auto& env = hub.peek_top_chunk(Device::HOST_INDEX);

            REQUIRE(env.buffer[0] == 2);
            REQUIRE(env.buffer[1] == 3);
            REQUIRE(env.buffer[2] == 4);
        });
    }

    DebuggingScheduler scheduler;
    scheduler << node << hub;
    scheduler.run();
}
