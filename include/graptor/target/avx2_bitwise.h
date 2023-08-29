// -*- c++ -*-
#ifndef GRAPTOR_TARGET_AVX2_BITWISE_H
#define GRAPTOR_TARGET_AVX2_BITWISE_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

alignas(64) extern const uint32_t avx512_singleton_basis_epi32[32];
alignas(64) extern const uint32_t avx512_himask_basis_epi32[32];

namespace target {

/***********************************************************************
 * AVX2 -- bitwise operations, independent of element type
 ***********************************************************************/
#if __AVX2__
struct avx2_bitwise {
    using type = __m256i;

    static type setzero() { return _mm256_setzero_si256(); }
    static bool is_zero( type a ) { return _mm256_testz_si256( a, a ); }

    static type setone() {
	type x;
	return _mm256_cmpeq_epi32( x, x );
    }

    static __m128i lower_half( type a ) {
	return _mm256_castsi256_si128( a );
    }
    static __m128i upper_half( type a ) {
	return _mm256_extracti128_si256( a, 1 );
    }
    static vpair<__m128i,__m128i> decompose( type a ) {
	return vpair<__m128i,__m128i>{ lower_half(a), upper_half(a) };
    }
    static type set_pair( __m128i up, __m128i lo ) {
	return _mm256_inserti128_si256( _mm256_castsi128_si256( lo ), up, 1 );
    }
    static __m128i sse_subvector( type a, int idx ) {
	switch( idx ) {
	case 0: return _mm256_extracti128_si256( a, 0 );
	case 1: return _mm256_extracti128_si256( a, 1 );
	default:
	    assert( 0 && "should not get here" );
	};
    }
    
    static type logical_and( type a, type b ) { return _mm256_and_si256( a, b ); }
    static type logical_andnot( type a, type b ) { return _mm256_andnot_si256( a, b ); }
    static type logical_or( type a, type b ) { return _mm256_or_si256( a, b ); }
    static type logical_invert( type a ) { return _mm256_xor_si256( a, setone() ); }
    static type bitwise_and( type a, type b ) { return _mm256_and_si256( a, b ); }
    static type bitwise_andnot( type a, type b ) { return _mm256_andnot_si256( a, b ); }
    static type bitwise_or( type a, type b ) { return _mm256_or_si256( a, b ); }
    static type bitwise_xor( type a, type b ) { return _mm256_xor_si256( a, b ); }
    static type bitwise_xnor( type a, type b ) {
	return bitwise_xor( bitwise_invert( a ), b );
    }
    static type bitwise_invert( type a ) { return _mm256_xor_si256( a, setone() ); }

    static type setglobaloneval( size_t pos ) {
#if 0
	// https://stackoverflow.com/questions/72424660/best-way-to-mask-a-single-bit-in-avx2
	type ii = _mm256_set1_epi32(pos);
	const type off = _mm256_setr_epi32(0,32,64,96,128,160,192,224);
	type jj = _mm256_sub_epi32( ii, off );
	// http://agner.org/optimize/optimizing_assembly.pdf
	type x;
	x = _mm256_srli_epi32( _mm256_cmpeq_epi32( x, x ), 31 );
	type mask = _mm256_sllv_epi32( x, jj );
	return mask;
#else
	size_t word = pos >> 5;
	const type * s = reinterpret_cast<const type *>(
	    &avx512_singleton_basis_epi32[15-word] );
	type b = _mm256_loadu_si256( s );
	return _mm256_slli_epi32( b, pos & 31 );
#endif
    }

    // Generate a mask where all bits l and above are set, and below l are 0
    static type himask( unsigned l ) {
#if 1
	type k = _mm256_broadcastd_epi32( _mm_cvtsi32_si128( l ) );
	type c = _mm256_set_epi32( 256, 224, 192, 160, 128, 96, 64, 32 );
	type h = _mm256_slli_epi32( setone(), 31 );
	type s = _mm256_srav_epi32( h, _mm256_sub_epi32( c, k ) );
	type m = _mm256_cmpgt_epi32( k, c );
	type r = _mm256_andnot_si256( m, s );
	return r;
#else
	size_t word = l >> 5;
	const type * s = reinterpret_cast<const type *>(
	    &avx512_himask_basis_epi32[15-word] );
	type b = _mm256_loadu_si256( s );
	return _mm256_srai_epi32( b, 31 - ( l & 31 ) );
#endif
    }
};
#endif // __AVX2__


} // namespace target

#endif // GRAPTOR_TARGET_AVX2_BITWISE_H
