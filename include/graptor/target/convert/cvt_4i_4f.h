// -*- c++ -*-
#ifndef GRAPTOR_TARGET_CONVERT_4I_4F_H
#define GRAPTOR_TARGET_CONVERT_4I_4F_H

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
struct fp_conversion_traits<int, float, 4> {
    static __m128 convert( __m128i a ) {
	return _mm_cvtepi32_ps( a );
    }
};
#endif // __SSE4_2__

#if __AVX2__
template<>
struct fp_conversion_traits<unsigned int, float, 8> {
    // Based on
    // https://stackoverflow.com/questions/34066228/how-to-perform-uint32-float-conversion-with-sse
    static __m256 convert( __m256i a ) {
	using tr = target::avx2_4x8<unsigned int>;
	using tf = target::avx2_4fx8<float>;
	__m256i ones = tr::setone();
	__m256i mask = tr::slli( ones, 1 );
	__m256i a0 = tr::srli( a, 1 );
	__m256i a1 = tr::bitwise_andnot( mask, a );
	__m256 f0 = _mm256_cvtepi32_ps( a0 );
	__m256 f1 = _mm256_cvtepi32_ps( a1 );
	__m256 c = tf::add( tf::add( f0, f0 ), f1 );
	return c;
    }
};

template<>
struct fp_conversion_traits<int, float, 8> {
    static __m256 convert( __m256i a ) {
	return _mm256_cvtepi32_ps( a );
    }
};
#endif

#if __AVX512F__
template<>
struct fp_conversion_traits<int, float, 16> {
    static __m512 convert( __m512i a ) {
	return _mm512_cvtepi32_ps( a );
    }
};

template<>
struct fp_conversion_traits<unsigned int, float, 16> {
    static __m512 convert( __m512i a ) {
	return _mm512_cvtepu32_ps( a );
    }
};
#endif

} // namespace conversion

#endif // GRAPTOR_TARGET_CONVERT_4I_4F_H
