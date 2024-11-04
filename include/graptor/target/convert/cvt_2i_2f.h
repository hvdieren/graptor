// -*- c++ -*-
#ifndef GRAPTOR_TARGET_CONVERT_2I_2F_H
#define GRAPTOR_TARGET_CONVERT_2I_2F_H

#include <x86intrin.h>
#include <immintrin.h>
#include <type_traits>
#include <stdfloat>

namespace conversion {

#if __AVX512FP16__
template<>
struct fp_conversion_traits<unsigned short, std::float16_t, 8> {
    static __m128h convert( __m128i a ) {
	return _mm_cvtepu16_ph( a );
    }
};

template<>
struct fp_conversion_traits<short, std::float16_t, 8> {
    static __m128h convert( __m128i a ) {
	return _mm_cvtepi16_ph( a );
    }
};
#endif

} // namespace conversion

#endif // GRAPTOR_TARGET_CONVERT_2I_2F_H
