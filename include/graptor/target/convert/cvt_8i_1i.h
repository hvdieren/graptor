// -*- c++ -*-
#ifndef GRAPTOR_TARGET_CONVERT_8I_1I_H
#define GRAPTOR_TARGET_CONVERT_8I_1I_H

#include <x86intrin.h>
#include <immintrin.h>

alignas(64) extern const uint8_t conversion_8x4_1x4_shuffle[32];

namespace conversion {

#if __AVX2__
template<>
struct int_conversion_traits<int64_t, int8_t, 4> {
    static auto convert( __m256i a ) {
#if __AVX512VL__
	auto c = _mm256_cvtepi64_epi8( a );
#else
	const __m256i ctrl = _mm256_load_si256(
	    reinterpret_cast<const __m256i*>( conversion_8x4_1x4_shuffle ) );
	__m256i b = _mm256_shuffle_epi8( a, ctrl );
	__m128i hi = _mm256_extracti128_si256( b, 1 );
	__m128i lo = _mm256_castsi256_si128( b );
	__m128i c = _mm_or_si128( hi, lo );
#endif
#if GRAPTOR_USE_MMX
	__m64 d = _mm_cvtsi64_m64( _mm_extract_epi64( c, 0 ) & 0xffffffff );
#else
	__m128i d = c;
#endif
	return d;
    }
};

template<>
struct int_conversion_traits<int64_t, uint8_t, 4> {
    static auto convert( __m256i a ) {
	return int_conversion_traits<int64_t, int8_t, 4>::convert( a );
    }
};

template<>
struct int_conversion_traits<uint64_t, int8_t, 4> {
    static auto convert( __m256i a ) {
	return int_conversion_traits<int64_t, int8_t, 4>::convert( a );
    }
};

template<>
struct int_conversion_traits<uint64_t, uint8_t, 4> {
    static auto convert( __m256i a ) {
	return int_conversion_traits<int64_t, int8_t, 4>::convert( a );
    }
};
#endif

#if __AVX512F__
template<>
struct int_conversion_traits<int64_t, int8_t, 8> {
    static __m128i convert( __m512i a ) {
	return _mm512_cvtepi64_epi8( a );
    }
};

template<>
struct int_conversion_traits<int64_t, uint8_t, 8> {
    static __m128i convert( __m512i a ) {
	return _mm512_cvtepi64_epi8( a );
    }
};

template<>
struct int_conversion_traits<uint64_t, int8_t, 8> {
    static __m128i convert( __m512i a ) {
	return _mm512_cvtepi64_epi8( a );
    }
};

template<>
struct int_conversion_traits<uint64_t, uint8_t, 8> {
    static __m128i convert( __m512i a ) {
	return _mm512_cvtepi64_epi8( a );
    }
};

#elif __AVX2__

template<>
struct int_conversion_traits<int64_t, int8_t, 8> {
    static auto convert( vpair<__m256i,__m256i> a ) {
	__m128i lo = int_conversion_traits<int64_t, int8_t, 4>::convert( a.a );
	__m128i hi = int_conversion_traits<int64_t, int8_t, 4>::convert( a.b );
	return _mm_unpacklo_epi32( lo, hi );
    }
};

template<>
struct int_conversion_traits<int64_t, uint8_t, 8> {
    static auto convert( vpair<__m256i,__m256i> a ) {
	return int_conversion_traits<int64_t, int8_t, 8>::convert( a );
    }
};

template<>
struct int_conversion_traits<uint64_t, int8_t, 8> {
    static auto convert( vpair<__m256i,__m256i> a ) {
	return int_conversion_traits<int64_t, int8_t, 8>::convert( a );
    }
};

template<>
struct int_conversion_traits<uint64_t, uint8_t, 8> {
    static auto convert( vpair<__m256i,__m256i> a ) {
	return int_conversion_traits<int64_t, int8_t, 8>::convert( a );
    }
};
#endif

} // namespace conversion

#endif // GRAPTOR_TARGET_CONVERT_8I_1I_H
