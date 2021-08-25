#ifndef NOARR_PIPELINES_SIMPLE_SCHEDULER_HPP
#define NOARR_PIPELINES_SIMPLE_SCHEDULER_HPP

#include <cassert>
#include <vector>
#include <mutex>
#include <condition_variable>

#include "Node.hpp"
#include "Scheduler.hpp"

namespace noarr {
namespace pipelines {

/**
 * A scheduler that advances nodes in generations in parallel,
 * but at the end of each generation a synchronization barrier is present
 */
class SimpleScheduler : public Scheduler {
public:

    SimpleScheduler() {
        //
    }

    /**
     * Runs the pipeline until no nodes can be advanced.
     */
    virtual void run() override {
        assert(this->nodes.size() != 0
            && "Pipeline is empty so cannot be advanced");

        // initialize pipeline
        for (Node* node : this->nodes) {
            node->scheduler_initialize();
        }

        // run generations for as long as they advance data
        while (run_generation()) {};

        // terminate pipeline
        for (Node* node : this->nodes) {
            node->scheduler_terminate();
        }
    }

private:

    /**
     * Executes one generation (launches all nodes in parallel and waits
     * for all to finish)
     * 
     * @return True if data was advanced by at least one node
     */
    bool run_generation() {

        // Each node has one flag that remembers whether the node advanced data.
        // Since each node accesses only its own flag in the critical section,
        // no race condition should occur.
        std::vector<bool> node_advancements(
            nodes.size(), // one flag for each node
            false // all flags to false by default
        );

        // start expecting callbacks from all nodes
        callbacks_will_be_called();

        // update
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            nodes[i]->scheduler_update([this, i, &node_advancements](bool adv){
                node_advancements[i] = adv;
                callback_was_called(i);
            });
        }

        // barrier
        wait_for_callbacks();

        // post-update
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            nodes[i]->scheduler_post_update(node_advancements[i]);
        }
        
        // did the generation advance data?
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            if (node_advancements[i])
                return true;
        }
        return false;
    }

    ///////////////////////////
    // Synchronization logic //
    ///////////////////////////

private:
    std::mutex _callback_mutext; // protects the following variables
    std::condition_variable _callback_cv; // lets us wait and notify
    bool _expecting_callbacks = false;
    std::vector<bool> _callback_was_called;
    std::size_t _remaining_callbacks;

    /**
     * Call this before initializeing a node update
     */
    void callbacks_will_be_called() {
        // this method body is executed only when we acquire the lock
        std::lock_guard<std::mutex> lock(this->_callback_mutext);
        
        assert(!this->_expecting_callbacks
            && "Cannot expect callbacks when still expecting.");

        // we do expect callbacks now
        this->_expecting_callbacks = true;
        
        // no callback has been called yet, reset all to false
        this->_callback_was_called.resize(nodes.size());
        std::fill(
            this->_callback_was_called.begin(),
            this->_callback_was_called.end(),
            false
        );
        this->_remaining_callbacks = nodes.size();
    }

    /**
     * Call this from the node callback
     */
    void callback_was_called(std::size_t node_index) {
        {
            // this code scope is executed only when we acquire the lock
            std::lock_guard<std::mutex> lock(this->_callback_mutext);
            
            // we have the lock, set all the variables

            assert(this->_expecting_callbacks
                && "Callback was called but we did not expect it.");

            assert(node_index < nodes.size()
                && "Given index is invalid");

            assert(!this->_callback_was_called[node_index]
                && "Cannot call back multiple times!");

            this->_callback_was_called[node_index] = true;
            this->_remaining_callbacks -= 1;
        }

        // notify the waiting thread
        this->_callback_cv.notify_one();
    }

    /**
     * Call this to synchronously wait for the callback
     */
    void wait_for_callbacks() {
        // check that we are actually expecting a callback
        {
            // this code scope is executed only when we acquire the lock
            std::lock_guard<std::mutex> lock(this->_callback_mutext);
            
            assert(this->_expecting_callbacks
                && "Cannot wait for callbacks without first expecting them.");
        }

        // wait for someone else to set "_callback_was_called" to true
        {
            // this code scope will get locked when the "wait" call exists
            std::unique_lock<std::mutex> lock(this->_callback_mutext);
            this->_callback_cv.wait(lock, [&](){
                return this->_remaining_callbacks <= 0;
            });

            // we no longer expect any callbacks
            // (we can access the variable here, since we still own the lock
            // thanks to the previous "wait" method call)
            this->_expecting_callbacks = false;
        }
    }

};

} // pipelines namespace
} // namespace noarr

#endif
