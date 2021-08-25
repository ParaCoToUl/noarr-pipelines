#ifndef NOARR_PIPELINES_ENVELOPE_HPP
#define NOARR_PIPELINES_ENVELOPE_HPP

#include <cstddef>

#include "Buffer.hpp"
#include "UntypedEnvelope.hpp"

namespace noarr {
namespace pipelines {

template<typename Structure, typename BufferItem = void>
class Envelope : public UntypedEnvelope {
public:
    /**
     * The structure of data contained in the envelope
     */
    Structure structure;

    /**
     * Pointer to the underlying data buffer
     */
    BufferItem* buffer;

    /**
     * Constructs a new envelope from an existing buffer
     */
    Envelope(Buffer allocated_buffer)
        : UntypedEnvelope(
            std::move(allocated_buffer),
            typeid(Structure),
            typeid(BufferItem)
        ),
        buffer((BufferItem*) allocated_buffer_instance.data_pointer)
    { }

    /**
     * Efficiently swaps contents with another envelope of the same type and size, located on the same device
     */
    void swap_contents_with(Envelope &other)
    {
        // check that both envelopes belong to the same device
        assert(
            device_index() == other.device_index()
            && "Envelope contents may only be swapped if located on the same device"
        );

        // The buffer in the envelope may be just a wrapped C buffer given
        // by the user. We do not want this buffer to travel between hubs
        // unexpectedly.
        assert(
            !allocated_buffer_instance.wraps_existing_buffer()
            && !other.allocated_buffer_instance.wraps_existing_buffer()
            && "Envelope contents may not be swapped with wrapped buffer envelopes"
        );

        // check that physical sizes of both envelopes (their buffers) match
        assert(
            size() == other.size()
            && "Envelope contents may only be swapped if the buffers have the same size"
        );

        std::swap(buffer, other.buffer);
        std::swap(allocated_buffer_instance, other.allocated_buffer_instance);
        std::swap(structure, other.structure);
    }
};

} // pipelines namespace
} // namespace noarr

#endif
