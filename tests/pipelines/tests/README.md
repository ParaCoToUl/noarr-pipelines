# Overview of individual tests

- `async_compute_node_test.cpp`: Checks that the `AsyncComputeNode` invokes its event method in the correct order and in the correct thread.
- `buffer_test.cpp`: Check that the `Buffer` class can be constructed from an existing pointer and that it can be `std::move`d properly.
- `given_buffer_test.cpp`: Tests that a hub can be given an existing buffer to be used internally for an envelope.
- `hub_peeking_with_transfer_test.cpp`: Tests that the `Hub::peek_top_chunk` method performs synchronous memory transfer if the top chunk is not available for the requested device.
- `memory_sync_transfer_test.cpp`: Checks that the `HardwareManager` can perform synchronous memory transfers.
- `memory_transfer_test.cpp`: Checks that the `HardwareManger` can perform memory transfers.
- `producer_filter_consumer.cpp`: Tests that a producer-filter-consumer pipeline can be built. Primarily tests manual committing of chunks in the filter node.
- `producer_modifier_consumer.cpp`: Tests that a producer-modifier-consumer pipeline with a single shared hub can be built. The main tested feature is dataflow strategy switching for a hub.

There are also larger tests that simulate a specific usage of pipelines. These tests are tagged `[integration]`. They may reuse logic already tested in the other simple tests and not test anything new.

- `world_simulation`: Tests the usage scenario, where a discrete world is iteratively simulated. Interesting point: The test uses both inheritance and lambda expressions to create the compute node.
