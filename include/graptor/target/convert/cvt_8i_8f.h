// -*- c++ -*-
#ifndef GRAPTOR_TARGET_CONVERT_8I_8F_H
#define GRAPTOR_TARGET_CONVERT_8I_8F_H

#include <x86intrin.h>
#include <immintrin.h>
#include <type_traits>

namespace conversion {

#if __AVX2__ && __AVX512DQ__
template<>
struct fp_conversion_traits<unsigned long int, double, 4> {
    static __m256d convert( __m256i a ) {
	return _mm256_cvtepu64_pd( a );
    }
};

template<>
struct fp_conversion_traits<long int, double, 4> {
    static __m256d convert( __m256i a ) {
	return _mm256_cvtepi64_pd( a );
    }
};
#endif

#if __AVX512DQ__
template<>
struct fp_conversion_traits<long int, double, 8> {
    static __m512d convert( __m512i a ) {
	return _mm512_cvtepi64_pd( a );
    }
};

template<>
struct fp_conversion_traits<unsigned long int, double, 8> {
    static __m512d convert( __m512i a ) {
	return _mm512_cvtepu64_pd( a );
    }
};
#endif

} // namespace conversion

#endif // GRAPTOR_TARGET_CONVERT_8I_8F_H
