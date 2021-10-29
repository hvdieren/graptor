// -*- c++ -*-
#ifndef GRAPTOR_TARGET_CONVERT_4I_2I_H
#define GRAPTOR_TARGET_CONVERT_4I_2I_H

#include <x86intrin.h>
#include <immintrin.h>

alignas(64) extern const uint8_t conversion_4fx8_cfp16x8_shuffle[32];

namespace conversion {

#if __SSE4_2__
template<>
struct int_conversion_traits<int32_t, int16_t, 4> {
    static auto convert( __m128i a ) {
#if __AVX512VL__
	__m128i b = _mm_cvtepi32_epi16( a );
#else
	const __m128i ctrl = _mm_load_si128(
	    reinterpret_cast<const __m128i*>( conversion_4fx8_cfp16x8_shuffle ) );
	__m128i b = _mm_shuffle_epi8( a, ctrl );
#endif
#if GRAPTOR_USE_MMX
	int64_t c = _mm_extract_epi64( b, 0 );
	__m64 d = _mm_cvtsi64_m64( c );
#else
	__m128i d = b;
#endif
	return d;
    }
};

template<>
struct int_conversion_traits<int32_t, uint16_t, 4> {
    static auto convert( __m128i a ) {
	return int_conversion_traits<int32_t, int16_t, 4>::convert( a );
    }
};

template<>
struct int_conversion_traits<uint32_t, int16_t, 4> {
    static auto convert( __m128i a ) {
	return int_conversion_traits<int32_t, int16_t, 4>::convert( a );
    }
};

template<>
struct int_conversion_traits<uint32_t, uint16_t, 4> {
    static auto convert( __m128i a ) {
	return int_conversion_traits<int32_t, int16_t, 4>::convert( a );
    }
};
#endif

#if __AVX2__
template<>
struct int_conversion_traits<int32_t, int16_t, 8> {
    static __m128i convert( __m256i a ) {
#if __AVX512VL__
	return _mm256_cvtepi32_epi16( a );
#else
	const __m256i ctrl = _mm256_load_si256(
	    reinterpret_cast<const __m256i*>( conversion_4fx8_cfp16x8_shuffle ) );
	__m256i b = _mm256_shuffle_epi8( a, ctrl );
	__m128i bhi = _mm256_extractf128_si256( b, 1 );
	__m128i blo = _mm256_castsi256_si128( b );
	__m128i c = _mm_or_si128( bhi, blo ); // note the mask is tuned for this
	return c;
#endif
    }
};

template<>
struct int_conversion_traits<int32_t, uint16_t, 8> {
    static __m128i convert( __m256i a ) {
	return int_conversion_traits<int32_t, int16_t, 8>::convert( a );
    }
};

template<>
struct int_conversion_traits<uint32_t, int16_t, 8> {
    static __m128i convert( __m256i a ) {
	return int_conversion_traits<int32_t, int16_t, 8>::convert( a );
    }
};

template<>
struct int_conversion_traits<uint32_t, uint16_t, 8> {
    static __m128i convert( __m256i a ) {
	return int_conversion_traits<int32_t, int16_t, 8>::convert( a );
    }
};
#endif

#if __AVX512F__
template<>
struct int_conversion_traits<int32_t, int16_t, 16> {
    static __m256i convert( __m512i a ) {
	return _mm512_cvtepi32_epi16( a );
    }
};

template<>
struct int_conversion_traits<int32_t, uint16_t, 16> {
    static __m256i convert( __m512i a ) {
	return int_conversion_traits<int32_t, int16_t, 16>::convert( a );
    }
};

template<>
struct int_conversion_traits<uint32_t, int16_t, 16> {
    static __m256i convert( __m512i a ) {
	return int_conversion_traits<int32_t, int16_t, 16>::convert( a );
    }
};

template<>
struct int_conversion_traits<uint32_t, uint16_t, 16> {
    static __m256i convert( __m512i a ) {
	return int_conversion_traits<int32_t, int16_t, 16>::convert( a );
    }
};
#endif

} // namespace conversion

#endif // GRAPTOR_TARGET_CONVERT_4I_2I_H
