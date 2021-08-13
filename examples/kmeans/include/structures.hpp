#ifndef EXAMPLES_KMEANS_STRUCTURES_HPP
#define EXAMPLES_KMEANS_STRUCTURES_HPP

/*
    This file contains definitions of various structures used throughout this example.
    
    The reason it is separated like this is:
    1. For readability
    2. To allow me to include it from both the GCC and NVCC compiled parts and
        thus specify an explicit template instantiation.
 */

#include <noarr/structures_extended.hpp>

/**
 * A list of 2D float points arranged as a list of pairs (array of structures)
 */
using PointListAoS = noarr::vector<'i', noarr::array<'d', 2, noarr::scalar<float>>>;

/**
 * A list of 2D points arranged as two lists of coordinates (structure of arrays)
 */
using PointListSoA = noarr::array<'d', 2, noarr::vector<'i', noarr::scalar<float>>>;

// sized variants of structures defined above
using SizedPointListAoS = decltype(PointListAoS() | noarr::set_length<'i'>(0));
using SizedPointListSoA = decltype(PointListSoA() | noarr::set_length<'i'>(0));

// bagged and sized variants of structures defined above
using SizedPointListAoSBag = decltype(noarr::make_bag(SizedPointListAoS(), (char*)nullptr));
using SizedPointListSoABag = decltype(noarr::make_bag(SizedPointListSoA(), (char*)nullptr));

#endif
