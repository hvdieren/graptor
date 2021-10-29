// -*- c++ -*-
#ifndef GRAPTOR_TARGET_MMX_4x2_H
#define GRAPTOR_TARGET_MMX_4x2_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include "graptor/target/decl.h"

namespace target {

/***********************************************************************
 * MMX 2 4-byte integers
 * This is poorly tested
 ***********************************************************************/
#if __MMX__
template<typename T = uint32_t>
struct mmx_4x2 {
    static_assert( sizeof(T) == 4, 
		   "version of template class for 4-byte integers" );
public:
    using member_type = T;
    using type = __m64;
    using vmask_type = __m64;
    using itype = __m64;
    using int_type = uint8_t;

    using mtraits = mask_type_traits<2>;
    using mask_type = typename mtraits::type;

    // using half_traits = vector_type_int_traits<member_type,16>;
    // using recursive_traits = vt_recursive<member_type,1,32,half_traits>;
    // using int_traits = sse42_1x16<int_type>;
    
    static const size_t size = 4;
    static const size_t vlen = 2;

    static type set1( member_type a ) { return _mm_set1_pi32( a ); }
    static type set( member_type a1, member_type a0 ) {
	return _mm_set_pi32( a1, a0 );
    }
    static type setzero() { type x; return _mm_xor_si64( x, x ); }
    static type set1inc( member_type a ) {
	return type(0x0100000000ULL) + set1(a);
    }
    static bool is_zero( type x ) { return lane0( x ) == 0 && lane1( x ) == 0; }

    static member_type lane( type a, int idx ) {
	return (((uint64_t)a) >> idx*32) & ((uint64_t(1)<<32)-1);
    }
    static member_type lane0( type a ) { return lane( a, 0 ); }
    static member_type lane1( type a ) { return lane( a, 1 ); }

    static type load( member_type *addr ) {
	return *(type *)addr;
    }
    static type loadu( member_type *addr ) {
	return *(type *)addr;
    }
    static void store( member_type *addr, type val ) {
	*(type *)addr = val;
    }
    static void storeu( member_type *addr, type val ) {
	*(type *)addr = val;
    }

    template<typename VecT2>
    static auto convert( VecT2 a ) {
	assert( 0 && "NYI" );
	return setzero();
    }

    // Casting here is to do a signed cast of a bitmask for logical masks
    template<typename T2>
    static auto asvector( type a );
    
    static type add( type a, type b ) { return _mm_add_pi32( a, b ); }

    static type cmpeq( type a, type b ) { return _mm_cmpeq_pi32( a, b ); }
    static type cmpne( type a, type b ) { return ~ _mm_cmpeq_pi32( a, b ); }
    static type blend( type c, type a, type b ) {
	assert( 0 && "NYI" );
	return setzero();
    }
    static auto logical_and( type a, type b ) {
	return _mm_and_si64( a, b );
    }
    static auto logicalor_bool( type &a, type b ) {
	auto mod = _mm_andnot_si64( a, b );
	a = _mm_or_si64( a, b );
	return mod;
    }
    static mask_type movemask( vmask_type a ) {
	mask_type r = 0;
	if( lane0( a ) )
	    r |= 1;
	if( lane1( a ) )
	    r |= 2;
	return r;
	// return _mm_movemask_ps( a );
    }
    static member_type reduce_setif( type val ) {
	assert( 0 && "NYI" );
	return setzero(); // TODO
    }
    static member_type reduce_add( type val ) {
	assert( 0 && "NYI" );
	return setzero(); // _mm256_reduce_add_epi64( val );
    }
    static member_type reduce_add( type val, mask_type mask ) {
	assert( 0 && "NYI" );
	return setzero(); // _mm256_reduce_add_epi64( val );
    }

    static member_type reduce_min( type val, vmask_type m ) {
	member_type r = ~member_type(0);
	if( lane0( m ) )
	    r = lane0( val );
	if( lane1( m ) && lane1( val ) < r )
	    r = lane1( val );
	return r;
    }

    static member_type reduce_logicalor( type val ) {
	// Cast to member_type will truncate
	return member_type(
	    _mm_cvtsi64_si32( _mm_or_si64( _m_psrldi( val, 32 ), val ) ) );
    }
    
    static type bitwiseor( type a, type b ) { return _mm_or_si64( a, b ); }
    static member_type reduce_bitwiseor( type a ) {
	return lane0( a ) | lane1( a );
    }

    template<typename IdxT>
    static type gather( member_type *a, IdxT b ) {
	assert( 0 && "NYI" );
	return setzero();
    }

    template<typename IdxT>
    static void scatter( member_type *a, IdxT b, type c ) {
	assert( 0 && "NYI" );
    }
};
#endif // __MMX__

} // namespace target

#endif // GRAPTOR_TARGET_MMX_4x2_H
