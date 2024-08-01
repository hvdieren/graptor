// -*- c++ -*-
#ifndef GRAPTOR_TARGET_SSE42_BITWISE_H
#define GRAPTOR_TARGET_SSE42_BITWISE_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

namespace target {

/***********************************************************************
 * SSE42 -- bitwise operations, independent of element type
 ***********************************************************************/
#if __SSE4_2__
struct sse42_bitwise {
    using type = __m128i;

    static type setzero() { return _mm_setzero_si128(); }
    static bool is_zero( type a ) { return _mm_test_all_zeros( a, a ); }
    static bool is_bitwise_and_zero( type a, type b ) {
	return _mm_testz_si128( a, b );
    }

    static type setone() {
	type x;
	return _mm_cmpeq_epi32( x, x );
    }

#if GRAPTOR_USE_MMX
    static __m64 lower_half( type a ) {
	return (__m64)_mm_extract_epi64( a, 0 );
    }
    static __m64 upper_half( type a ) {
	return (__m64)_mm_extract_epi64( a, 1 );
    }
#else
    static uint64_t lower_half( type a ) {
	return _mm_extract_epi64( a, 0 );
    }
    static uint64_t upper_half( type a ) {
	return _mm_extract_epi64( a, 1 );
    }
#endif
    static type set_pair( __m64 hi, __m64 lo ) {
	return _mm_insert_epi64(
	    _mm_cvtsi64_si128( _mm_cvtm64_si64( lo ) ),
	    _mm_cvtm64_si64( hi ), 1 );
    }
    static type set_pair( uint64_t hi, uint64_t lo ) {
	return _mm_insert_epi64( _mm_cvtsi64_si128( lo ), hi, 1 );
    }

    static type bsrli( type a, unsigned int bs ) {
	return _mm_bsrli_si128( a, bs );
    }

#if __AVX512VL__ && __AVX512F__
    static constexpr bool has_ternary = true;
#else
    static constexpr bool has_ternary = false;
#endif

    template<unsigned char imm8>
    static type ternary( type a, type b, type c ) {
#if __AVX512VL__ && __AVX512F__
	return _mm_ternarylogic_epi32( a, b, c, imm8 );
#else
	assert( 0 && "NYI" );
	return setzero();
#endif
    }

    static type logical_and( type a, type b ) { return _mm_and_si128( a, b ); }
    static type logical_andnot( type a, type b ) { return _mm_andnot_si128( a, b ); }
    static type logical_or( type a, type b ) { return _mm_or_si128( a, b ); }
    static type logical_invert( type a ) { return _mm_xor_si128( a, setone() ); }
    static type bitwise_and( type a, type b ) { return _mm_and_si128( a, b ); }
    static type bitwise_andnot( type a, type b ) { return _mm_andnot_si128( a, b ); }
    static type bitwise_or( type a, type b ) { return _mm_or_si128( a, b ); }
    static type bitwise_xor( type a, type b ) { return _mm_xor_si128( a, b ); }
    static type bitwise_xnor( type a, type b ) {
	return bitwise_xor( bitwise_invert( a ), b );
    }
    static type bitwise_andnot( type a, type b, type c ) {
	if constexpr ( has_ternary )
	    return ternary<0x8>( a, b, c );
	else
	    return bitwise_and( bitwise_andnot( a, b ), c );
    }
    static type bitwise_or_and( type a, type b, type c ) {
	if constexpr ( has_ternary )
	    return ternary<0xa8>( a, b, c );
	else
	    return bitwise_and( bitwise_or( a, b ), c );
    }
    static type bitwise_invert( type a ) { return bitwise_xor( a, setone() ); }

    static type setglobaloneval( size_t pos ) {
	size_t off = pos & 63;
	uint64_t m = uint64_t(1) << off;
	type mm = _mm_cvtsi64_si128( m );
	if( pos != off )
	    mm = _mm_bslli_si128( mm, 8 );
	return mm;
    }

    // Generate a mask where all bits l and above are set, and below l are 0
    static type himask( unsigned l ) {
	type k = _mm_broadcastd_epi32( _mm_cvtsi32_si128( l ) );
	type c = _mm_set_epi32( 128, 96, 64, 32 );
	type h = _mm_slli_epi32( setone(), 31 );
	type d = _mm_sub_epi32( c, k );
#if __AVX2__
	type s = _mm_srav_epi32( h, d );
	type m = _mm_cmpgt_epi32( k, c );
	type r = _mm_andnot_si128( m, s );
	return r;
#else
	// _mm_srav_epi32 is in AVX2
	assert( 0 && "NYI" );
#endif
    }
};
#endif // __SSE4_2__


} // namespace target

#endif // GRAPTOR_TARGET_SSE42_BITWISE_H
