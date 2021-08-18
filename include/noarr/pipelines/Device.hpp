#ifndef NOARR_PIPELINES_DEVICE_HPP
#define NOARR_PIPELINES_DEVICE_HPP

#include <cstddef>

namespace noarr {
namespace pipelines {

/**
 * Represents a device that has memory
 * (host cpu or a gpu device)
 */
struct Device {
    using index_t = std::int8_t;

    index_t device_index = -1;

    // useful constants
    enum : index_t { HOST_INDEX = -1 };
    enum : index_t { DEVICE_INDEX = 0 };
    enum : index_t { DEVICE0_INDEX = 0 };
    enum : index_t { DEVICE1_INDEX = 1 };
    enum : index_t { DUMMY_GPU_INDEX = -2 };

    Device() {
        //
    }

    Device(index_t device_index) : device_index(device_index) { }
};

} // pipelines namespace
} // namespace noarr

#endif
