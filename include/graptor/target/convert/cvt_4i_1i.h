// -*- c++ -*-
#ifndef GRAPTOR_TARGET_CONVERT_4I_1I_H
#define GRAPTOR_TARGET_CONVERT_4I_1I_H

#include <x86intrin.h>
#include <immintrin.h>

alignas(64) extern const uint8_t conversion_4fx8_cfp16x8_shuffle[32];

namespace conversion {

#if __SSE4_2__ && 0
template<>
struct int_conversion_traits<int32_t, int8_t, 4> {
    static __m64 convert( __m128i a ) {
	return _mm_cvtsi128_si32( _mm_cvtepi32_epi8( a ) );
    }
};

template<>
struct int_conversion_traits<int32_t, uint8_t, 4> {
    static __m64 convert( __m128i a ) {
	return _mm_cvtsi128_si32( _mm_cvtepi32_epi8( a ) );
    }
};

template<>
struct int_conversion_traits<uint32_t, int8_t, 4> {
    static __m64 convert( __m128i a ) {
	return _mm_cvtsi128_si32( _mm_cvtepi32_epi8( a ) );
    }
};

template<>
struct int_conversion_traits<uint32_t, uint8_t, 4> {
    static __m64 convert( __m128i a ) {
	return _mm_cvtsi128_si32( _mm_cvtepi32_epi8( a ) );
    }
};
#endif

#if __AVX2__
namespace {
#if GRAPTOR_USE_MMX
static __m64 avx2_convert_4i_1i( __m256i a ) {
#if __AVX512VL__
    __m128i b = _mm256_cvtepi32_epi8( a );
    return _mm_cvtsi64_m64( _mm_cvtsi128_si64( b ) );
#else
    __m128i b = target::convert_4b_1b( a );
    return _mm_cvtsi64_m64( _mm_cvtsi128_si64( b ) );
#endif
}
#else // GRAPTOR_USE_MMX
static __m128i avx2_convert_4i_1i( __m256i a ) {
    // Top half of vector set to zero
#if __AVX512VL__
    __m128i b = _mm256_cvtepi32_epi8( a );
    return b;
#else
    __m128i b = target::convert_4b_1b( a );
    return b;
#endif
}
#endif // GRAPTOR_USE_MMX
}

template<>
struct int_conversion_traits<int32_t, int8_t, 8> {
    static auto convert( __m256i a ) {
	return avx2_convert_4i_1i( a );
    }
};

template<>
struct int_conversion_traits<int32_t, uint8_t, 8> {
    static auto convert( __m256i a ) {
	return avx2_convert_4i_1i( a );
    }
};

template<>
struct int_conversion_traits<uint32_t, int8_t, 8> {
    static auto convert( __m256i a ) {
	return avx2_convert_4i_1i( a );
    }
};

template<>
struct int_conversion_traits<uint32_t, uint8_t, 8> {
    static auto convert( __m256i a ) {
	return avx2_convert_4i_1i( a );
    }
};
#endif

#if __AVX512F__
template<>
struct int_conversion_traits<int32_t, int8_t, 16> {
    static __m128i convert( __m512i a ) {
	return _mm512_cvtepi32_epi8( a );
    }
};

template<>
struct int_conversion_traits<int32_t, uint8_t, 16> {
    static __m128i convert( __m512i a ) {
	return _mm512_cvtepi32_epi8( a );
    }
};

template<>
struct int_conversion_traits<uint32_t, int8_t, 16> {
    static __m128i convert( __m512i a ) {
	return _mm512_cvtepi32_epi8( a );
    }
};

template<>
struct int_conversion_traits<uint32_t, uint8_t, 16> {
    static __m128i convert( __m512i a ) {
	return _mm512_cvtepi32_epi8( a );
    }
};
#endif


} // namespace conversion

#endif // GRAPTOR_TARGET_CONVERT_4I_1I_H
