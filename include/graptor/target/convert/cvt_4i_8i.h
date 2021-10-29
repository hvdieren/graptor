// -*- c++ -*-
#ifndef GRAPTOR_TARGET_CONVERT_4I_8I_H
#define GRAPTOR_TARGET_CONVERT_4I_8I_H

#include <x86intrin.h>
#include <immintrin.h>

namespace conversion {

#if __AVX2__
template<>
struct int_conversion_traits<int32_t, int64_t, 4> {
    static __m256i convert( __m128i a ) {
	return _mm256_cvtepi32_epi64( a );
    }
};

template<>
struct int_conversion_traits<int32_t, uint64_t, 4> {
    static __m256i convert( __m128i a ) {
	return _mm256_cvtepi32_epi64( a );
    }
};

template<>
struct int_conversion_traits<uint32_t, int64_t, 4> {
    static __m256i convert( __m128i a ) {
	return _mm256_cvtepu32_epi64( a );
    }
};

template<>
struct int_conversion_traits<uint32_t, uint64_t, 4> {
    static __m256i convert( __m128i a ) {
	return _mm256_cvtepu32_epi64( a );
    }
};
#endif

#if __AVX512F__
template<>
struct int_conversion_traits<int32_t, int64_t, 8> {
    static __m512i convert( __m256i a ) {
	return _mm512_cvtepi32_epi64( a );
    }
};

template<>
struct int_conversion_traits<int32_t, uint64_t, 8> {
    static __m512i convert( __m256i a ) {
	return _mm512_cvtepi32_epi64( a );
    }
};

template<>
struct int_conversion_traits<uint32_t, int64_t, 8> {
    static __m512i convert( __m256i a ) {
	return _mm512_cvtepu32_epi64( a );
    }
};

template<>
struct int_conversion_traits<uint32_t, uint64_t, 8> {
    static __m512i convert( __m256i a ) {
	return _mm512_cvtepu32_epi64( a );
    }
};
#endif

} // namespace conversion

#endif // GRAPTOR_TARGET_CONVERT_4I_8I_H
