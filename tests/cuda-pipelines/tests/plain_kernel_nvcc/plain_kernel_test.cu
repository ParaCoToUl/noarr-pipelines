#include <catch2/catch.hpp>

#include <iostream>

#include "plain_kernel.cu"

TEST_CASE("Plain kernel nvcc can execute", "[plain_kernel][nvcc]") {
    int expected_sum = 0;
    for (int i = 1; i <= 1024; ++i)
        expected_sum += i;

    REQUIRE(run_plain_kernel<int>() == expected_sum);

    REQUIRE((int)run_plain_kernel<float>() == expected_sum);

    REQUIRE((int)run_plain_kernel<double>() == expected_sum);
}
