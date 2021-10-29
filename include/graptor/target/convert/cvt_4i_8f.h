// -*- c++ -*-
#ifndef GRAPTOR_TARGET_CONVERT_4I_8F_H
#define GRAPTOR_TARGET_CONVERT_4I_8F_H

#include <x86intrin.h>
#include <immintrin.h>
#include <type_traits>

#include "graptor/target/convert/cvt_4i_8i.h"
#include "graptor/target/convert/cvt_8i_8f.h"

namespace conversion {

#if __AVX2__
template<>
struct fp_conversion_traits<unsigned int, double, 4> {
    static __m256d convert( __m128i a ) {
#if __AVX512VL__
	return _mm256_cvtepu32_pd( a );
#elif __AVX512DQ__
	// Just in case the highest bit is set, convert to (un)signed long
	// first, then to double
	auto al = int_conversion_traits<unsigned int, unsigned long, 4>
	    ::convert( a );
	auto bl = fp_conversion_traits<unsigned long, double, 4>
	    ::convert( al );
	return bl;
#else
	// Hope for the best, i.e., most significant bit is always 0.
	// Could craft a code sequence to take out top bit, then add in
	// appropriate double constant
	return _mm256_cvtepi32_pd( a );
#endif
    }
};
#endif

#if __AVX2__
template<>
struct fp_conversion_traits<int, double, 4> {
    static __m256d convert( __m128i a ) {
	return _mm256_cvtepi32_pd( a );
    }
};
#endif

#if __AVX512F__
template<>
struct fp_conversion_traits<int, double, 8> {
    static __m512d convert( __m256i a ) {
	return _mm512_cvtepi32_pd( a );
    }
};

template<>
struct fp_conversion_traits<unsigned int, double, 8> {
    static __m512d convert( __m256i a ) {
	return _mm512_cvtepu32_pd( a );
    }
};
#endif

} // namespace conversion

#endif // GRAPTOR_TARGET_CONVERT_4I_8F_H
