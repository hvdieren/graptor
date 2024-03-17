// -*- c++ -*-
#ifndef GRAPTOR_TARGET_AVX512_BITWISE_H
#define GRAPTOR_TARGET_AVX512_BITWISE_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

alignas(64) extern const uint32_t avx512_singleton_basis_epi32[64];
alignas(64) extern const uint32_t avx512_himask_basis_epi32[64];

namespace target {

/***********************************************************************
 * AVX512F -- bitwise operations, independent of element type
 ***********************************************************************/
#if __AVX512F__
struct avx512_bitwise {
    using type = __m512i;

    static type setzero() { return _mm512_setzero_si512(); }
    static bool is_zero( type a ) {
	// __mmask16 e = _mm512_cmpneq_epi32_mask( a, setzero() );
	// return _kortestz_mask16_u8( e, e );
	__mmask8 e = _mm512_test_epi64_mask( a, a );
	return _ktestz_mask8_u8( e, e );
    }
    static bool is_bitwise_and_zero( type a, type b ) {
	__mmask8 e = _mm512_test_epi64_mask( a, b );
	return _ktestz_mask8_u8( e, e );
    }

    static type setone() {
	// Recommended here:
	// https://stackoverflow.com/questions/45105164/set-all-bits-in-cpu-register-to-1-efficiently/45113467#45113467
	// Seems that gcc sets x to zero first using vpxor $xmm
	// Tell compiler we are writing into x to avoid initialisation
	__m512i x;
	__asm__( "" : "=v"(x) : : );
	return _mm512_ternarylogic_epi32( x, x, x, 0xff );
    }

    static __m256i lower_half( type a ) {
	return _mm512_castsi512_si256( a );
    }
    static __m256i upper_half( type a ) {
	return _mm512_extracti64x4_epi64( a, 1 );
    }
    static vpair<__m256i,__m256i> decompose( type a ) {
	return vpair<__m256i,__m256i>{ lower_half(a), upper_half(a) };
    }
    static type set_pair( __m256i up, __m256i lo ) {
	return _mm512_inserti64x4( _mm512_castsi256_si512( lo ), up, 1 );
    }
    static __m128i sse_subvector( type a, int idx ) {
	switch( idx ) {
	case 0: return _mm512_extracti64x2_epi64( a, 0 );
	case 1: return _mm512_extracti64x2_epi64( a, 1 );
	case 2: return _mm512_extracti64x2_epi64( a, 2 );
	case 3: return _mm512_extracti64x2_epi64( a, 3 );
	default:
	    assert( 0 && "should not get here" );
	};
    }

    static constexpr bool has_ternary = true;

    template<unsigned char imm8>
    static type ternary( type a, type b, type c ) {
	return _mm512_ternarylogic_epi64( a, b, c, imm8 );
    }
    
    static type logical_and( type a, type b ) { return _mm512_and_si512( a, b ); }
    static type logical_andnot( type a, type b ) { return _mm512_andnot_si512( a, b ); }
    static type logical_or( type a, type b ) { return _mm512_or_si512( a, b ); }
    static type logical_invert( type a ) { return _mm512_xor_si512( a, setone() ); }
    static type bitwise_and( type a, type b ) { return _mm512_and_si512( a, b ); }
    static type bitwise_andnot( type a, type b ) { return _mm512_andnot_si512( a, b ); }
    static type bitwise_andnot( type a, type b, type c ) {
	return ternary<0x8>( a, b, c );
    }
    static type bitwise_or_and( type a, type b, type c ) {
	return ternary<0xa8>( a, b, c );
    }
    static type bitwise_or( type a, type b ) { return _mm512_or_si512( a, b ); }
    static type bitwise_xor( type a, type b ) { return _mm512_xor_si512( a, b ); }
    static type bitwise_xnor( type a, type b ) {
	// TOOD: use ternary
	return bitwise_xor( bitwise_invert( a ), b );
    }
    static type bitwise_invert( type a ) { return _mm512_andnot_si512( a, setone() ); }

    static type setglobaloneval( size_t pos ) {
#if 0
	// https://stackoverflow.com/questions/72424660/best-way-to-mask-a-single-bit-in-avx2
	assert( 0 && "NYI" );
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
	    &avx512_singleton_basis_epi32[31-word] );
	type b = _mm512_loadu_si512( s );
	return _mm512_slli_epi32( b, pos & 31 );
#endif
    }

    // Generate a mask where all bits l and above are set, and below l are 0
    static type himask( unsigned l ) {
#if 1
	type k = _mm512_broadcastd_epi32( _mm_cvtsi32_si128( l ) );
	type c = _mm512_set_epi32( 512, 480, 448, 416, 384, 352, 320, 288,
				   256, 224, 192, 160, 128, 96, 64, 32 );
	type h = _mm512_slli_epi32( setone(), 31 );
	type s = _mm512_srav_epi32( h, _mm512_sub_epi32( c, k ) );
	__mmask16 m = _mm512_cmpge_epi32_mask( c, k );
	type r = _mm512_maskz_and_epi32( m, s, s );
	return r;
#else
	// erroneous
	size_t word = l >> 5;
	const type * s = reinterpret_cast<const type *>(
	    &avx512_himask_basis_epi32[31-word] );
	type b = _mm512_loadu_si512( s );
	return _mm512_srai_epi32( b, 31 - ( l & 31 ) );
#endif
    }
};
#endif // __AVX512F__


} // namespace target

#endif // GRAPTOR_TARGET_AVX512_BITWISE_H
