#ifndef NOARR_PIPELINES_SCHEDULER_HPP
#define NOARR_PIPELINES_SCHEDULER_HPP

#include <cassert>
#include <vector>

#include "Node.hpp"

namespace noarr {
namespace pipelines {

/**
 * Base class for schedulers to inherit from
 */
class Scheduler {
protected:
    /**
     * Nodes that the scheduler periodically updates
     */
    std::vector<Node*> nodes;

public:

    Scheduler() {
        //
    }

    ///////////////////////////////
    // Pipeline construction API //
    ///////////////////////////////

    /**
     * Registers a node to be updated by the scheduler
     */
    virtual void add(Node& node) {
        this->nodes.push_back(&node);
    }

    Scheduler& operator<<(Node& node) {
        add(node);

        return *this;
    }

    ///////////////////
    // Execution API //
    ///////////////////

public:

    /**
     * Runs the pipeline until no nodes can be advanced.
     */
    virtual void run() = 0;
};

} // pipelines namespace
} // namespace noarr

#endif
