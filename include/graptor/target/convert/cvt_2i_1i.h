// -*- c++ -*-
#ifndef GRAPTOR_TARGET_CONVERT_2I_1I_H
#define GRAPTOR_TARGET_CONVERT_2I_1I_H

#include <x86intrin.h>
#include <immintrin.h>

alignas(64) extern const uint8_t conversion_2x8_1x8_shuffle[32];

namespace conversion {

template<>
struct int_conversion_traits<uint16_t, uint8_t, 4> {
    static uint32_t convert( __m64 a ) {
	assert( 0 && "NYI" );
    }
};

template<>
struct int_conversion_traits<uint16_t, int8_t, 4> {
    static uint32_t convert( __m64 a ) {
	assert( 0 && "NYI" );
    }
};

template<>
struct int_conversion_traits<int16_t, uint8_t, 4> {
    static uint32_t convert( __m64 a ) {
	assert( 0 && "NYI" );
    }
};

template<>
struct int_conversion_traits<int16_t, int8_t, 4> {
    static uint32_t convert( __m64 a ) {
	assert( 0 && "NYI" );
    }
};

#if __AVX2__

namespace {
static __m128i avx2_convert_2i_1i( __m256i a ) {
#if __AVX512VL__ && __AVX512BW__
    return _mm256_cvtepi16_epi8( a );
#else
    const __m256i shuf = _mm256_load_si256(
	reinterpret_cast<const __m256i*>( conversion_2x8_1x8_shuffle ) );
    __m256i b = _mm256_shuffle_epi8( a, shuf );
    __m128i hi = _mm256_extracti128_si256( b, 1 );
    __m128i lo = _mm256_castsi256_si128( b );
    return _mm_or_si128( lo, hi );
#endif
}

}

template<>
struct int_conversion_traits<uint16_t, uint8_t, 16> {
    static __m128i convert( __m256i a ) {
	return avx2_convert_2i_1i( a );
    }
};

template<>
struct int_conversion_traits<uint16_t, int8_t, 16> {
    static __m128i convert( __m256i a ) {
	return avx2_convert_2i_1i( a );
    }
};

template<>
struct int_conversion_traits<int16_t, uint8_t, 16> {
    static __m128i convert( __m256i a ) {
	return avx2_convert_2i_1i( a );
    }
};

template<>
struct int_conversion_traits<int16_t, int8_t, 16> {
    static __m128i convert( __m256i a ) {
	return avx2_convert_2i_1i( a );
    }
};
#endif

#if __AVX512BW__
template<>
struct int_conversion_traits<uint16_t, uint8_t, 32> {
    static __m256i convert( __m512i a ) {
	return _mm512_cvtepi16_epi8( a );
    }
};

template<>
struct int_conversion_traits<uint16_t, int8_t, 32> {
    static __m256i convert( __m512i a ) {
	return _mm512_cvtepi16_epi8( a );
    }
};

template<>
struct int_conversion_traits<int16_t, uint8_t, 32> {
    static __m256i convert( __m512i a ) {
	return _mm512_cvtepi16_epi8( a );
    }
};

template<>
struct int_conversion_traits<int16_t, int8_t, 32> {
    static __m256i convert( __m512i a ) {
	return _mm512_cvtepi16_epi8( a );
    }
};

#endif

} // namespace conversion

#endif // GRAPTOR_TARGET_CONVERT_2I_1I_H
