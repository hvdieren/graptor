// -*- c++ -*-
#ifndef GRAPTOR_TARGET_AVX2_BITWISE_H
#define GRAPTOR_TARGET_AVX2_BITWISE_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

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
};
#endif // __AVX2__


} // namespace target

#endif // GRAPTOR_TARGET_AVX2_BITWISE_H
