// -*- c++ -*-
#ifndef GRAPTOR_TARGET_CONVERT_8F_4I_H
#define GRAPTOR_TARGET_CONVERT_8F_4I_H

#include <x86intrin.h>
#include <immintrin.h>
#include <type_traits>

namespace conversion {

#if __AVX2__
template<>
struct fp_conversion_traits<double, int, 4> {
    static __m128i convert( __m256d a ) {
	return _mm256_cvtpd_epi32( a );
    }
};
#endif

#if __AVX512VL__
template<>
struct fp_conversion_traits<double, unsigned int, 4> {
    static __m128i convert( __m256d a ) {
	return _mm256_cvtpd_epu32( a );
    }
};
#endif

#if __AVX512F__
template<>
struct fp_conversion_traits<double, int, 8> {
    static __m256i convert( __m512d a ) {
	return _mm512_cvtpd_epi32( a );
    }
};

template<>
struct fp_conversion_traits<double, unsigned int, 8> {
    static __m256i convert( __m512d a ) {
	return _mm512_cvtpd_epu32( a );
    }
};
#endif

} // namespace conversion

#endif // GRAPTOR_TARGET_CONVERT_8F_4I_H
