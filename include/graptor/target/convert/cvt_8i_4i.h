// -*- c++ -*-
#ifndef GRAPTOR_TARGET_CONVERT_8I_4I_H
#define GRAPTOR_TARGET_CONVERT_8I_4I_H

#include <x86intrin.h>
#include <immintrin.h>

#include "graptor/target/avx2_bitwise.h"

alignas(64) extern const uint8_t conversion_4fx8_cfp16x8_shuffle[32];

namespace conversion {

#if __AVX2__

namespace {
static __m128i avx2_convert_8i_4i( __m256i a ) {
#if __AVX512VL__
	return _mm256_cvtepi64_epi32( a );
#else
	// We choose an instruction sequence that does not require loading
	// shuffle masks. A single step would be possible with permutevar8x32
	// followed by cast to extract lower 128 bits, however, the load of
	// the shuffle mask is expensive (possible 7 cycles) on sky lake,
	// and will occupy a register to hold the temporary.
	// This conversion simply truncates the integers to 32 bits.
	const __m256i s = _mm256_shuffle_epi32( a, 0b10001000 );
	const __m256i z = target::avx2_bitwise::setzero();
	const __m256i l = _mm256_blend_epi32( z, s, 0b11000011 );
	const __m128i hi = _mm256_extracti128_si256( l, 1 );
	const __m128i lo = _mm256_castsi256_si128( l );
	return _mm_or_si128( hi, lo );
#endif
}

} // namespace anonymous

template<>
struct int_conversion_traits<int64_t, int32_t, 4> {
    static __m128i convert( __m256i a ) {
	return avx2_convert_8i_4i( a );
    }
};



template<>
struct int_conversion_traits<int64_t, uint32_t, 4> {
    static __m128i convert( __m256i a ) {
	return avx2_convert_8i_4i( a );
    }
};

template<>
struct int_conversion_traits<uint64_t, int32_t, 4> {
    static __m128i convert( __m256i a ) {
	return avx2_convert_8i_4i( a );
    }
};

template<>
struct int_conversion_traits<uint64_t, uint32_t, 4> {
    static __m128i convert( __m256i a ) {
	return avx2_convert_8i_4i( a );
    }
};
#endif

#if __AVX512F__
template<>
struct int_conversion_traits<int64_t, int32_t, 8> {
    static __m256i convert( __m512i a ) {
	return _mm512_cvtepi64_epi32( a );
    }
};

template<>
struct int_conversion_traits<int64_t, uint32_t, 8> {
    static __m256i convert( __m512i a ) {
	return _mm512_cvtepi64_epi32( a );
    }
};

template<>
struct int_conversion_traits<uint64_t, int32_t, 8> {
    static __m256i convert( __m512i a ) {
	return _mm512_cvtepi64_epi32( a );
    }
};

template<>
struct int_conversion_traits<uint64_t, uint32_t, 8> {
    static __m256i convert( __m512i a ) {
	return _mm512_cvtepi64_epi32( a );
    }
};
#endif

} // namespace conversion

#endif // GRAPTOR_TARGET_CONVERT_8I_4I_H
