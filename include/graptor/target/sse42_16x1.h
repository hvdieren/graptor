// -*- c++ -*-
#ifndef GRAPTOR_TARGET_SSE42_16x1_H
#define GRAPTOR_TARGET_SSE42_16x1_H

#include <x86intrin.h>
#include <immintrin.h>

#include "graptor/longint.h"

namespace target {

/***********************************************************************
 * Cases representable in 128 bits (SSEx)
 ***********************************************************************/
#if __SSE4_2__
template<typename T = longint<16>>
struct sse42_16x1 {
    static_assert( sizeof(T) == 16, "size assumption" );
    static constexpr unsigned short W = 16;
    static constexpr unsigned short vlen = 1;
    static constexpr unsigned short size = W * vlen;
    using member_type = T;
    using type = __m128i;
    using vmask_type = __m128i;
    using mask_type = bool;

    static type setzero() { return _mm_setzero_si128(); }
    static type setone() {
	auto zero = setzero();
	return _mm_cmpeq_epi64( zero, zero );
    }
    // error - static type setoneval() { return _mm_srli_si128( setone(), 127 ); }
    static type set1( member_type a ) { return a.get(); }

    static member_type lane( type a, unsigned int l ) {
	return member_type(a);
    }
    static member_type lane0( type a ) {
	return member_type(a);
    }
    
    static bool cmpne( type l, type r, mt_bool ) {
	vmask_type m = cmpne( l, r, mt_vmask() );
	int z = _mm_testz_si128( m, m ); // z is 1 if m == 0
	return (bool)z;
    }
    static bool cmpeq( type l, type r, mt_bool ) {
	return !cmpne( l, r, mt_bool() );
    }
    static vmask_type cmpne( type l, type r, mt_vmask ) {
	// Check using 2 lanes of 64 bits each
	vmask_type eq = _mm_cmpeq_epi64( l, r );
	// Results is (abbreviated) 0xFF, 0xF0, 0x0F or 0x00
	// Need to return false in case of 0xFF only.
	vmask_type sh = _mm_shuffle_epi32( eq, 0b01001110 );
	return _mm_and_si128( eq, sh );
    }
    static vmask_type cmpeq( type l, type r, mt_vmask ) {
	return _mm_xor_si128( setone(), cmpeq( l, r, mt_vmask() ) );
    }
    static mask_type cmpne( type l, type r, mt_mask ) {
	// To return mask_type, treat as 0/1 as scalar
	return (mask_type)cmpne( l, r , mt_bool() );
    }
    static mask_type cmpeq( type l, type r, mt_mask ) {
	// To return mask_type, treat as 0/1 as scalar
	return (mask_type)cmpeq( l, r, mt_bool() );
    }

    static type logical_and( type l, type r ) {
	return _mm_and_si128( l, r );
    }
    static type logical_or( type l, type r ) {
	return _mm_or_si128( l, r );
    }
    static type bitwise_and( type l, type r ) {
	return _mm_and_si128( l, r );
    }
    static type bitwise_or( type l, type r ) {
	return _mm_or_si128( l, r );
    }
    static type bitwise_xor( type l, type r ) {
	return _mm_xor_si128( l, r );
    }
    static type bitwise_invert( type a ) {
	return bitwise_xor( a, setone() );
    }
    static type logical_invert( type a ) {
	assert( 0 && "NYI" );
    }
    static member_type reduce_bitwiseor( type a ) {
	return member_type( a );
    }
    static member_type reduce_logicalor( type a ) {
	return member_type( a );
    }

    static type loadu( const member_type * addr ) { return load( addr ); }
    static type load( const member_type * addr ) {
	return _mm_load_si128( ((const type *)addr) );
    }
    static void storeu( member_type * addr, type val ) {
	store( addr, val );
    }
    static void store( member_type * addr, type val ) {
	_mm_store_si128( ((type *)addr), val );
    }
    template<typename IdxT>
    static type gather( const member_type * addr, IdxT idx ) {
	return _mm_load_si128( ((const type *)addr) + idx );
    }
    template<typename IdxT, typename MaskT>
    static type gather( const member_type * addr, IdxT idx, MaskT mask ) {
	if( mask != 0 )
	    return _mm_load_si128( ((const type *)addr) + idx );
	else
	    return setzero();
    }
    template<typename IdxT>
    static void scatter( member_type * addr, IdxT idx, type val ) {
	store( addr+idx, val );
    }
};
#endif // __SSE4_2__

} // namespace target

#endif //  GRAPTOR_TARGET_SSE42_16x1_H
