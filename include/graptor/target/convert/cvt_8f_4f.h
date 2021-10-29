// -*- c++ -*-
#ifndef GRAPTOR_TARGET_CONVERT_8F_4F_H
#define GRAPTOR_TARGET_CONVERT_8F_4F_H

#include <x86intrin.h>
#include <immintrin.h>
#include <type_traits>

namespace conversion {

#if __AVX2__
template<>
struct fp_conversion_traits<double, float, 4> {
    static __m128 convert( __m256d a ) {
	return _mm256_cvtpd_ps( a );
    }
};
#endif

#if __AVX512F__
template<>
struct fp_conversion_traits<double, float, 8> {
    static __m256 convert( __m512d a ) {
	return _mm512_cvtpd_ps( a );
    }
};
#endif

} // namespace conversion

#endif // GRAPTOR_TARGET_CONVERT_8F_4F_H
