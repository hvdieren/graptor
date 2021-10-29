// -*- c++ -*-
#ifndef GRAPTOR_TARGET_CONVERT_8F_8I_H
#define GRAPTOR_TARGET_CONVERT_8F_8I_H

#include <x86intrin.h>
#include <immintrin.h>
#include <type_traits>

namespace conversion {

#if __AVX512VL__ && __AVX512DQ__
template<>
struct fp_conversion_traits<double, long int, 4> {
    static __m256i convert( __m256d a ) {
	return _mm256_cvtpd_epi64( a );
    }
};

template<>
struct fp_conversion_traits<double, unsigned long int, 4> {
    static __m256i convert( __m256d a ) {
	return _mm256_cvtpd_epu64( a );
    }
};
#endif

#if __AVX512DQ__
template<>
struct fp_conversion_traits<double, long int, 8> {
    static __m512i convert( __m512d a ) {
	return _mm512_cvtpd_epi64( a );
    }
};

template<>
struct fp_conversion_traits<double, unsigned long int, 8> {
    static __m512i convert( __m512d a ) {
	return _mm512_cvtpd_epu64( a );
    }
};
#endif

} // namespace conversion

#endif // GRAPTOR_TARGET_CONVERT_8F_8I_H
