// -*- c++ -*-
#ifndef GRAPTOR_TARGET_CONVERT_1I_8I_H
#define GRAPTOR_TARGET_CONVERT_1I_8I_H

#include <x86intrin.h>
#include <immintrin.h>

namespace conversion {

#if __AVX2__
template<>
struct int_conversion_traits<int8_t, int64_t, 4> {
    static __m256i convert( __m64 a ) {
	return _mm256_cvtepi8_epi64(
	    _mm_cvtsi64_si128( _mm_cvtm64_si64( a ) ) );
    }
    static __m256i convert( __m128i a ) {
	return _mm256_cvtepi8_epi64( a );
    }
};

template<>
struct int_conversion_traits<uint8_t, uint64_t, 4> {
    static __m256i convert( __m64 a ) {
	return _mm256_cvtepu8_epi64(
	    _mm_cvtsi64_si128( _mm_cvtm64_si64( a ) ) );
    }
    static __m256i convert( __m128i a ) {
	return _mm256_cvtepu8_epi64( a );
    }
};
#endif // __AVX2__

#if __AVX512F__
template<>
struct int_conversion_traits<int8_t, int64_t, 8> {
    static __m512i convert( __m64 a ) {
	__m128i b = _mm_cvtsi64_si128( _mm_cvtm64_si64( a ) );
	return _mm512_cvtepi8_epi64( b );
    }
    static __m512i convert( __m128i a ) {
	return _mm512_cvtepi8_epi64( a );
    }
};

template<>
struct int_conversion_traits<uint8_t, uint64_t, 8> {
    static __m512i convert( __m64 a ) {
	__m128i b = _mm_cvtsi64_si128( _mm_cvtm64_si64( a ) );
	return _mm512_cvtepu8_epi64( b );
    }
    static __m512i convert( __m128i a ) {
	return _mm512_cvtepu8_epi64( a );
    }
};
#elif __AVX2__ // __AVX512F__
template<>
struct int_conversion_traits<int8_t, int64_t, 8> {
    static vpair<__m256i,__m256i> convert( __m64 a ) {
	__m128i b = _mm_cvtsi64_si128( _mm_cvtm64_si64( a ) );
	return vpair<__m256i,__m256i>(
	    _mm256_cvtepi8_epi64( b ),
	    _mm256_cvtepi8_epi64( _mm_bsrli_si128( b, 4 ) ) );
    }
    static vpair<__m256i,__m256i> convert( __m128i a ) {
	return vpair<__m256i,__m256i>(
	    _mm256_cvtepi8_epi64( a ),
	    _mm256_cvtepi8_epi64( _mm_bsrli_si128( a, 4 ) ) );
    }
};

template<>
struct int_conversion_traits<int8_t, uint64_t, 8> {
    static vpair<__m256i,__m256i> convert( __m64 a ) {
	__m128i b = _mm_cvtsi64_si128( _mm_cvtm64_si64( a ) );
	return vpair<__m256i,__m256i>(
	    _mm256_cvtepi8_epi64( b ),
	    _mm256_cvtepi8_epi64( _mm_bsrli_si128( b, 4 ) ) );
    }
    static vpair<__m256i,__m256i> convert( __m128i a ) {
	return vpair<__m256i,__m256i>(
	    _mm256_cvtepi8_epi64( a ),
	    _mm256_cvtepi8_epi64( _mm_bsrli_si128( a, 4 ) ) );
    }
};

template<>
struct int_conversion_traits<uint8_t, int64_t, 8> {
    static vpair<__m256i,__m256i> convert( __m64 a ) {
	__m128i b = _mm_cvtsi64_si128( _mm_cvtm64_si64( a ) );
	return vpair<__m256i,__m256i>(
	    _mm256_cvtepu8_epi64( b ),
	    _mm256_cvtepu8_epi64( _mm_bsrli_si128( b, 4 ) ) );
    }
    static vpair<__m256i,__m256i> convert( __m128i a ) {
	return vpair<__m256i,__m256i>(
	    _mm256_cvtepu8_epi64( a ),
	    _mm256_cvtepu8_epi64( _mm_bsrli_si128( a, 4 ) ) );
    }
};

template<>
struct int_conversion_traits<uint8_t, uint64_t, 8> {
    static vpair<__m256i,__m256i> convert( __m64 a ) {
	__m128i b = _mm_cvtsi64_si128( _mm_cvtm64_si64( a ) );
	return vpair<__m256i,__m256i>(
	    _mm256_cvtepu8_epi64( b ),
	    _mm256_cvtepu8_epi64( _mm_bsrli_si128( b, 4 ) ) );
    }
    static vpair<__m256i,__m256i> convert( __m128i a ) {
	return vpair<__m256i,__m256i>(
	    _mm256_cvtepu8_epi64( a ),
	    _mm256_cvtepu8_epi64( _mm_bsrli_si128( a, 4 ) ) );
    }
};
#endif // __AVX512F__ / __AVX2__

#if __AVX512F__
template<>
struct int_conversion_traits<uint8_t, uint64_t, 16> {
    static vpair<__m512i,__m512i> convert( __m128i a ) {
	return vpair<__m512i,__m512i>{
	    _mm512_cvtepu8_epi64( a ),
		_mm512_cvtepu8_epi64( _mm_shuffle_epi32( a, 0b11101110 ) ) };
    }
};

template<>
struct int_conversion_traits<uint8_t, int64_t, 16> {
    static vpair<__m512i,__m512i> convert( __m128i a ) {
	return vpair<__m512i,__m512i>{
	    _mm512_cvtepu8_epi64( a ),
		_mm512_cvtepu8_epi64( _mm_shuffle_epi32( a, 0b11101110 ) ) };
    }
};

template<>
struct int_conversion_traits<int8_t, int64_t, 16> {
    static vpair<__m512i,__m512i> convert( __m128i a ) {
	return vpair<__m512i,__m512i>{
	    _mm512_cvtepi8_epi64( a ),
		_mm512_cvtepi8_epi64( _mm_shuffle_epi32( a, 0b11101110 ) ) };
    }
};

template<>
struct int_conversion_traits<int8_t, uint64_t, 16> {
    static vpair<__m512i,__m512i> convert( __m128i a ) {
	return vpair<__m512i,__m512i>{
	    _mm512_cvtepi8_epi64( a ),
		_mm512_cvtepi8_epi64( _mm_shuffle_epi32( a, 0b11101110 ) ) };
    }
};
#endif // __AVX512F__

} // namespace conversion

#endif // GRAPTOR_TARGET_CONVERT_1I_8I_H
