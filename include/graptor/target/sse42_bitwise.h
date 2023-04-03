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

    static type logical_and( type a, type b ) { return _mm_and_si128( a, b ); }
    static type logical_andnot( type a, type b ) { return _mm_andnot_si128( a, b ); }
    static type logical_or( type a, type b ) { return _mm_or_si128( a, b ); }
    static type logical_invert( type a ) { return _mm_xor_si128( a, setone() );}
    static type bitwise_and( type a, type b ) { return _mm_and_si128( a, b ); }
    static type bitwise_andnot( type a, type b ) { return _mm_andnot_si128( a, b ); }
    static type bitwise_or( type a, type b ) { return _mm_or_si128( a, b ); }
    static type bitwise_xor( type a, type b ) { return _mm_xor_si128( a, b ); }
    static type bitwise_xnor( type a, type b ) {
	return bitwise_xor( bitwise_invert( a ), b );
    }
    static type bitwise_invert( type a ) { return bitwise_xor( a, setone() ); }

};
#endif // __SSE4_2__


} // namespace target

#endif // GRAPTOR_TARGET_SSE42_BITWISE_H
