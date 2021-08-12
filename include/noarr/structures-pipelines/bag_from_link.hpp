#ifndef NOARR_STRUCTURES_PIPELINES_BAG_FROM_LINK_HPP
#define NOARR_STRUCTURES_PIPELINES_BAG_FROM_LINK_HPP

#include <noarr/structures_extended.hpp>
#include "bag_from_envelope.hpp"

namespace noarr {
namespace pipelines {

template<typename Link_t>
auto bag_from_link(const Link_t& link) {
    return bag_from_envelope(*link.envelope);
}

} // namespace pipelines
} // namespace noarr

#endif
