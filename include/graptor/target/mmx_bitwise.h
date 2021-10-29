// -*- c++ -*-
#ifndef GRAPTOR_TARGET_MMX_BITWISE_H
#define GRAPTOR_TARGET_MMX_BITWISE_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include "graptor/target/decl.h"

namespace target {

/***********************************************************************
 * MMX -- bitwise operations, independent of element type
 ***********************************************************************/
#if __MMX__
struct mmx_bitwise {
    using type = __m64;

    static type setzero() { return _mm_setzero_si64(); }
    static bool is_zero( type x ) { return _mm_cvtm64_si64( x ) == 0ULL; }

    static type setone() {
	return _mm_cvtsi64_m64( 0xffffffffffffffffUL );
    }

    static auto logical_and( type a, type b ) { return _mm_and_si64( a, b ); }
    static auto logical_andnot( type a, type b ) { return _mm_andnot_si64( a, b ); }
    static type logical_or( type a, type b ) { return _mm_or_si64( a, b ); }
    static type logical_invert( type a ) { return _mm_andnot_si64( a, setone() ); }
    static type bitwise_and( type a, type b ) { return _mm_and_si64( a, b ); }
    static type bitwise_andnot( type a, type b ) { return _mm_andnot_si64( a, b ); }
    static type bitwise_or( type a, type b ) { return _mm_or_si64( a, b ); }
    static type bitwise_xor( type a, type b ) { return _mm_xor_si64( a, b ); }
    static type bitwise_xnor( type a, type b ) {
	return bitwise_xor( bitwise_invert( a ), b );
    }
    static type bitwise_invert( type a ) { return _mm_xor_si64( a, setone() ); }
};
#endif // __MMX__


} // namespace target

#endif // GRAPTOR_TARGET_MMX_BITWISE_H
