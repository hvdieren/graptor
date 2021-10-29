// -*- c++ -*-
#ifndef GRAPTOR_TARGET_CONVERT_1I_2I_H
#define GRAPTOR_TARGET_CONVERT_1I_2I_H

#include <x86intrin.h>
#include <immintrin.h>

namespace conversion {

#if __AVX2__
template<>
struct int_conversion_traits<uint8_t, uint16_t, 16> {
    static __m256i convert( __m128i a ) {
	return _mm256_cvtepu8_epi16( a );
    }
};

template<>
struct int_conversion_traits<uint8_t, int16_t, 16> {
    static __m256i convert( __m128i a ) {
	return _mm256_cvtepu8_epi16( a );
    }
};

template<>
struct int_conversion_traits<int8_t, int16_t, 16> {
    static __m256i convert( __m128i a ) {
	return _mm256_cvtepi8_epi16( a );
    }
};

template<>
struct int_conversion_traits<int8_t, uint16_t, 16> {
    static __m256i convert( __m128i a ) {
	return _mm256_cvtepi8_epi16( a );
    }
};
#endif

#if __AVX512BW__
template<>
struct int_conversion_traits<uint8_t, uint16_t, 32> {
    static __m512i convert( __m256i a ) {
	return _mm512_cvtepu8_epi16( a );
    }
};

template<>
struct int_conversion_traits<uint8_t, int16_t, 32> {
    static __m512i convert( __m256i a ) {
	return _mm512_cvtepu8_epi16( a );
    }
};

template<>
struct int_conversion_traits<int8_t, int16_t, 32> {
    static __m512i convert( __m256i a ) {
	return _mm512_cvtepi8_epi16( a );
    }
};

template<>
struct int_conversion_traits<int8_t, uint16_t, 32> {
    static __m512i convert( __m256i a ) {
	return _mm512_cvtepi8_epi16( a );
    }
};
#endif

} // namespace conversion

#endif // GRAPTOR_TARGET_CONVERT_1I_2I_H
