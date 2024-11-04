// -*- c++ -*-
#ifndef GRAPTOR_TARGET_CONVERT_4I_2F_H
#define GRAPTOR_TARGET_CONVERT_4I_2F_H

#include <x86intrin.h>
#include <immintrin.h>
#include <type_traits>
#include <stdfloat>

namespace conversion {

#if __AVX512FP16__
template<>
struct fp_conversion_traits<unsigned, std::float16_t, 8> {
    static __m128h convert( __m256i a ) {
	return _mm256_cvtepu32_ph( a );
    }
};

template<>
struct fp_conversion_traits<int, std::float16_t, 8> {
    static __m128h convert( __m256i a ) {
	return _mm256_cvtepi32_ph( a );
    }
};
#endif

} // namespace conversion

#endif // GRAPTOR_TARGET_CONVERT_4I_2F_H
