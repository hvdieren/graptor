// -*- c++ -*-
#ifndef GRAPTOR_TARGET_CONVERT_2F_2I_H
#define GRAPTOR_TARGET_CONVERT_2F_2I_H

#include <x86intrin.h>
#include <immintrin.h>
#include <type_traits>
#include <stdfloat>

namespace conversion {

#if __AVX512FP16__
template<>
struct fp_conversion_traits<std::float16_t, unsigned short, 8> {
    static __m128i convert( __m128h a ) {
	return _mm_cvtph_epi16( a );
    }
};
#endif

} // namespace conversion

#endif // GRAPTOR_TARGET_CONVERT_2F_2I_H
