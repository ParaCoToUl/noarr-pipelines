#ifndef NOARR_STRUCTURES_PIPELINES_BAG_FROM_ENVELOPE_HPP
#define NOARR_STRUCTURES_PIPELINES_BAG_FROM_ENVELOPE_HPP

#include <noarr/structures_extended.hpp>

namespace noarr {
namespace pipelines {

/**
 * Creates an observer bag that can be used to access the data inside the given envelope
 */
template<typename Envelope_t>
auto bag_from_envelope(const Envelope_t& envelope) {
    return noarr::make_bag(envelope.structure, (char*) envelope.buffer);
}

} // namespace pipelines
} // namespace noarr

#endif
