// -*- c++ -*-
#ifndef GRAPTOR_TARGET_SSE42_1x8_H
#define GRAPTOR_TARGET_SSE42_1x8_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include "graptor/target/sse42_1x16.h"

namespace target {

/***********************************************************************
 * SSE4.2 8 byte-sized integers. Stored in an %xmm register, which is
 * twice too large, but supports more operations than MMX.
 ***********************************************************************/
#if !GRAPTOR_USE_MMX

template<unsigned short VL, typename T>
struct sse42_1xL;

template<typename T = uint8_t>
struct sse42_1x8 : public sse42_1xL<8,T> {
    using type = typename sse42_1xL<8,T>::type;

    // Methods to switch easily between mmx_1x8 and sse42_1x8
    static uint64_t asint( type a ) {
	return _mm_extract_epi64( a, 0 );
    }
};

#endif // GRAPTOR_USE_MMX

} // namespace target

#endif // GRAPTOR_TARGET_SSE42_1x8_H
