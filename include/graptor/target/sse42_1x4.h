// -*- c++ -*-
#ifndef GRAPTOR_TARGET_SSE42_1x4_H
#define GRAPTOR_TARGET_SSE42_1x4_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include "graptor/target/sse42_1x16.h"

namespace target {

/***********************************************************************
 * SSE4.2 4 byte-sized integers. Stored in an %xmm register, which is
 * four times too large, but supports more operations than MMX and scalar
 * ISA.
 ***********************************************************************/
template<unsigned short VL, typename T>
struct sse42_1xL;

template<typename T = uint8_t>
struct sse42_1x4 : public sse42_1xL<4,T> {
    using type = typename sse42_1xL<4,T>::type;

    // Methods to switch easily between mmx_1x8 and sse42_1x8
    static uint32_t asint( type a ) {
	return _mm_extract_epi32( a, 0 );
    }
};

} // namespace target

#endif // GRAPTOR_TARGET_SSE42_1x4_H
