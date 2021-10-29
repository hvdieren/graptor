// -*- c++ -*-
#ifndef GRAPTOR_TARGET_CONVERT_2I_4I_H
#define GRAPTOR_TARGET_CONVERT_2I_4I_H

#include <x86intrin.h>
#include <immintrin.h>

namespace conversion {

#if __SSE4_2__
template<>
struct int_conversion_traits<int16_t, int32_t, 4> {
    static __m128i convert( __m64 a ) {
	auto b = _mm_cvtsi64_si128( _mm_cvtm64_si64( a ) );
	return _mm_cvtepi16_epi32( b );
    }
    // Should implement 2x4 in lower half of __m128i, not in __m64
    static __m128i convert( __m128i a ) {
	return _mm_cvtepi16_epi32( a );
    }
};

template<>
struct int_conversion_traits<int16_t, uint32_t, 4> {
    static __m128i convert( __m64 a ) {
	auto b = _mm_cvtsi64_si128( _mm_cvtm64_si64( a ) );
	return _mm_cvtepi16_epi32( b );
    }
    static __m128i convert( __m128i a ) {
	return _mm_cvtepi16_epi32( a );
    }
};

template<>
struct int_conversion_traits<uint16_t, int32_t, 4> {
    static __m128i convert( __m64 a ) {
	auto b = _mm_cvtsi64_si128( _mm_cvtm64_si64( a ) );
	return _mm_cvtepu16_epi32( b );
    }
    static __m128i convert( __m128i a ) {
	return _mm_cvtepu16_epi32( a );
    }
};

template<>
struct int_conversion_traits<uint16_t, uint32_t, 4> {
    static __m128i convert( __m64 a ) {
	auto b = _mm_cvtsi64_si128( _mm_cvtm64_si64( a ) );
	return _mm_cvtepu16_epi32( b );
    }
    static __m128i convert( __m128i a ) {
	return _mm_cvtepu16_epi32( a );
    }
};
#endif

#if __AVX2__
template<>
struct int_conversion_traits<int16_t, int32_t, 8> {
    static __m256i convert( __m128i a ) {
	return _mm256_cvtepi16_epi32( a );
    }
};

template<>
struct int_conversion_traits<int16_t, uint32_t, 8> {
    static __m256i convert( __m128i a ) {
	return _mm256_cvtepi16_epi32( a );
    }
};

template<>
struct int_conversion_traits<uint16_t, int32_t, 8> {
    static __m256i convert( __m128i a ) {
	return _mm256_cvtepu16_epi32( a );
    }
};

template<>
struct int_conversion_traits<uint16_t, uint32_t, 8> {
    static __m256i convert( __m128i a ) {
	return _mm256_cvtepu16_epi32( a );
    }
};
#endif

#if __AVX512F__
template<>
struct int_conversion_traits<int16_t, int32_t, 16> {
    static __m512i convert( __m256i a ) {
	return _mm512_cvtepi16_epi32( a );
    }
};

template<>
struct int_conversion_traits<int16_t, uint32_t, 16> {
    static __m512i convert( __m256i a ) {
	return _mm512_cvtepi16_epi32( a );
    }
};

template<>
struct int_conversion_traits<uint16_t, int32_t, 16> {
    static __m512i convert( __m256i a ) {
	return _mm512_cvtepu16_epi32( a );
    }
};

template<>
struct int_conversion_traits<uint16_t, uint32_t, 16> {
    static __m512i convert( __m256i a ) {
	return _mm512_cvtepu16_epi32( a );
    }
};
#endif

} // namespace conversion

#endif // GRAPTOR_TARGET_CONVERT_2I_4I_H
