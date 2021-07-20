#include <catch2/catch.hpp>

#include <string>
#include <memory>
#include <iostream>

#include <noarr/cuda-pipelines/foo.hpp>

TEST_CASE("Foo test", "[foo]") {
    REQUIRE(noarr::pipelines::foo() == 42);
}
