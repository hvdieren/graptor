// -*- c++ -*-
#ifndef GRAPTOR_TARGET_SSE42_2x4_H
#define GRAPTOR_TARGET_SSE42_2x4_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include "graptor/target/sse42_2x8.h"

namespace target {

/***********************************************************************
 * SSE4.2 4 short integers. Stored in an %xmm register, which is
 * twice too large, but supports more operations than MMX.
 ***********************************************************************/
#if !GRAPTOR_USE_MMX

template<unsigned short VL, typename T>
struct sse42_2xL;

template<typename T = uint8_t>
struct sse42_2x4 : public sse42_2xL<4,T> {
    using type = typename sse42_2xL<4,T>::type;

    // Methods to switch easily between mmx_2x4 and sse42_2x4
    static uint64_t asint( type a ) {
	return _mm_extract_epi64( a, 0 );
    }
};

#endif // GRAPTOR_USE_MMX

} // namespace target

#endif // GRAPTOR_TARGET_SSE42_2x4_H
