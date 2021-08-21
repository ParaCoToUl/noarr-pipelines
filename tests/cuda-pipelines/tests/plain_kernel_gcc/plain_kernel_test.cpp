#include <catch2/catch.hpp>

#include <iostream>

#include "plain_kernel.hpp"

TEST_CASE("Plain kernel can execute", "[plain_kernel][gcc]") {
    int expected_sum = 0;
    for (int i = 1; i <= 1024; ++i)
        expected_sum += i;

    int returned_sum = run_plain_kernel<int>();
    
    REQUIRE(returned_sum == expected_sum);
}
