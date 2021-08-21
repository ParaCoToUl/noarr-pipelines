#include <catch2/catch.hpp>

#include <string>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

#include <noarr/pipelines/DebuggingScheduler.hpp>
#include <noarr/pipelines/LambdaAsyncComputeNode.hpp>

using namespace noarr::pipelines;

TEST_CASE("Async compute node", "[async_compute_node]") {
    std::vector<std::int32_t> log = {};
    std::vector<std::int32_t> expected_log = {
        1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 6
    };

    std::size_t finished_iterations = 0;
    std::size_t target_iterations = 3;

    std::thread::id scheduler_thread_id = std::this_thread::get_id();

    auto async_node = LambdaAsyncComputeNode(); {
        async_node.initialize([&](){
            log.push_back(1);
        });

        async_node.can_advance([&](){
            log.push_back(2);
            return finished_iterations < target_iterations;
        });

        async_node.advance([&](){
            log.push_back(3);
        });

        async_node.advance_async([&](){
            REQUIRE(scheduler_thread_id != std::this_thread::get_id());
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            log.push_back(4);
        });

        async_node.post_advance([&](){
            log.push_back(5);
            finished_iterations += 1;
        });

        async_node.terminate([&](){
            log.push_back(6);
        });
    }

    DebuggingScheduler scheduler;
    scheduler.add(async_node);
    scheduler.run();

    // check that the log order matches what is expected
    REQUIRE(log.size() == expected_log.size());
    for (std::size_t i = 0; i < log.size(); ++i) {
        REQUIRE(expected_log[i] == log[i]);
    }
}
