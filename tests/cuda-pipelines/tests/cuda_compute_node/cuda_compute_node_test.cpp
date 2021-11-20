#include <catch2/catch.hpp>

#include <iostream>
#include <memory>
#include <vector>

#include <noarr/pipelines.hpp>
#include <noarr/cuda-pipelines.hpp>

#include "my_kernel.hpp"

using namespace noarr::pipelines;

TEST_CASE("Cuda compute node", "[cuda_compute_node]") {
    CudaPipelines::register_extension();

    std::vector<int> items = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> expected_items = {1, 4, 9, 16, 25, 36, 49, 64, 81, 100};

    auto modifier = LambdaCudaComputeNode("modifier");
    auto consumer = LambdaComputeNode("consumer");
    
    auto hub = Hub<std::size_t, int>(sizeof(int) * items.size());
    hub.allocate_envelopes(Device::HOST_INDEX, 1);
    hub.allocate_envelopes(Device::DEVICE_INDEX, 1);

    /* define modifier */ {
        auto& link = modifier.link(hub.to_modify(Device::DEVICE_INDEX));

        modifier.advance_cuda([&](cudaStream_t stream){
            auto& env = *link.envelope;
            run_my_kernel(env.buffer, env.structure, stream);
        });

        modifier.post_advance([&](){
            hub.flow_data_to(consumer);
        });
    }

    /* define consumer */ {
        auto& link = consumer.link(hub.to_consume(Device::HOST_INDEX));

        consumer.advance([&](){
            auto& env = *link.envelope;
            items.resize(env.structure);
            std::memcpy(&items[0], env.buffer, sizeof(float) * items.size());

            consumer.callback();
        });
    }

    /* initialize hub */ {
        auto& env = hub.push_new_chunk();
        env.structure = items.size();
        std::memcpy(env.buffer, &items[0], sizeof(float) * items.size());

        hub.flow_data_to(modifier);
    }

    SimpleScheduler scheduler;
    scheduler << hub << modifier << consumer;
    scheduler.run();

    REQUIRE(items.size() == expected_items.size());
    for (std::size_t i = 0; i < items.size(); ++i) {
        REQUIRE(expected_items[i] == items[i]);
    }
}
