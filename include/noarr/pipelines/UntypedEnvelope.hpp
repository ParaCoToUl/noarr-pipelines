#ifndef NOARR_PIPELINES_UNTYPED_ENVELOPE_HPP
#define NOARR_PIPELINES_UNTYPED_ENVELOPE_HPP

#include <string>
#include <typeindex>

#include "Device.hpp"
#include "Buffer.hpp"

namespace noarr {
namespace pipelines {

/**
 * Abstract base class for Envelopes that allows polymorphism
 * 
 * While "untyped" it actually always has a type, only not via
 * a template here, but via runtime variables
 */
class UntypedEnvelope {
public:
    /**
     * The buffer instance that is responsible for memory management.
     * This is what implements the logic behind envelopes. All the other fields
     * are only an external API to the user plus a "structure" field.
     */
    Buffer allocated_buffer_instance;

    /**
     * Size of the data buffer in bytes
     */
    std::size_t size() const { return allocated_buffer_instance.bytes; }

    /**
     * What device this envelope lives on
     */
    Device::index_t device_index() const { return allocated_buffer_instance.device_index; }

    /**
     * Type of the structure value
     */
    const std::type_index structure_type;

    /**
     * Type of the buffer pointer
     */
    const std::type_index buffer_item_type;

protected:
    UntypedEnvelope(
        Buffer allocated_buffer,
        const std::type_index structure_type,
        const std::type_index buffer_item_type
    ) :
        allocated_buffer_instance(std::move(allocated_buffer)),
        structure_type(structure_type),
        buffer_item_type(buffer_item_type)
    { }
};

} // pipelines namespace
} // namespace noarr

#endif
