// -*- c++ -*-
#ifndef GRAPTOR_TARGET_CONVERT_4F_4I_H
#define GRAPTOR_TARGET_CONVERT_4F_4I_H

#include <x86intrin.h>
#include <immintrin.h>
#include <type_traits>

#if __AVX2__
#include "graptor/target/avx2_4x8.h"
#include "graptor/target/avx2_4fx8.h"
#endif

namespace conversion {

#if __SSE4_2__
template<>
struct fp_conversion_traits<float, int, 4> {
    static __m128i convert( __m128 a ) {
	return _mm_cvtps_epi32( a );
    }
};
#endif // __SSE4_2__

#if __AVX512VL__
template<>
struct fp_conversion_traits<float, unsigned int, 8> {
    static __m256i convert( __m256 a ) {
	return _mm256_cvtps_epu32( a );
    }
};
#elif __AVX2__
template<>
struct fp_conversion_traits<float, unsigned int, 8> {
    static __m256i convert( __m256 a ) {
	return _mm256_abs_epi32( _mm256_cvtps_epi32( a ) );
    }
};
#endif

#if __AVX2__
template<>
struct fp_conversion_traits<float, int, 8> {
    static __m256i convert( __m256 a ) {
	return _mm256_cvtps_epi32( a );
    }
};
#endif

#if __AVX512F__
template<>
struct fp_conversion_traits<float, int, 16> {
    static __m512i convert( __m512 a ) {
	return _mm512_cvtps_epi32( a );
    }
};

template<>
struct fp_conversion_traits<float, unsigned int, 16> {
    static __m512i convert( __m512 a ) {
	return _mm512_cvtps_epu32( a );
    }
};
#endif

} // namespace conversion

#endif // GRAPTOR_TARGET_CONVERT_4F_4I_H
