#ifndef NOARR_PIPELINES_MEMORY_TRANSFERER_HPP
#define NOARR_PIPELINES_MEMORY_TRANSFERER_HPP

#include <cassert>
#include <vector>
#include <map>
#include <functional>
#include <iostream>
#include <mutex>
#include <condition_variable>

#include "noarr/pipelines/Device.hpp"

namespace noarr {
namespace pipelines {

/**
 * Transfers data between two devices
 */
class MemoryTransferer {
public:

    /**
     * Transfers memory between
     */
    virtual ~MemoryTransferer() noexcept = default;

    virtual void transfer(
        void* from,
        void* to,
        std::size_t bytes,
        std::function<void()> callback
    ) const = 0;

    /**
     * Transfers memory between two buffers synchornously.
     * Has a default implementation that simply waits for the async variant.
     */
    virtual void transfer_sync(
        void* from,
        void* to,
        std::size_t bytes
    ) const {
        std::mutex mx; // protects the "done" variable
        std::condition_variable cv; // lets us wait and notify
        bool done = false;
        
        // start the asynchronous operation
        this->transfer(
            from,
            to,
            bytes,

            // when the async operation finishes, toggle the "done" flag
            [&](){
                std::lock_guard<std::mutex> lock(mx);
                done = true;
                cv.notify_one(); // wake up the waiting thread
            }
        );

        // wait for the "done" flag to be set to true
        {
            std::unique_lock<std::mutex> lock(mx);
            cv.wait(lock, [&](){
                return done;
            });
        }
    }
};

} // pipelines namespace
} // namespace noarr

#endif
