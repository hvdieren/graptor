// -*- c++ -*-
#ifndef GRAPTOR_TARGET_MMX_2x4_H
#define GRAPTOR_TARGET_MMX_2x4_H

#if GRAPTOR_USE_MMX || 1

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include "graptor/target/decl.h"
#include "graptor/target/mmx_bitwise.h"

namespace target {

/***********************************************************************
 * MMX 4 2-byte floating-point numbers
 * This is poorly tested
 ***********************************************************************/
#if __MMX__
template<typename T>
struct mmx_2x4 : public mmx_bitwise {
    static_assert( sizeof(T) == 2, 
		   "version of template class for 2-byte floats" );
public:
    static constexpr unsigned short W = 2;
    static constexpr unsigned short vlen = 4;
    static constexpr unsigned short size = W * vlen;

    using member_type = T;
    using type = __m64;
    using vmask_type = __m64;
    using itype = __m64;
    using int_type = uint16_t;

    using mask_traits = mask_type_traits<4>;
    using mask_type = typename mask_traits::type;

    using int_traits = mmx_2x4<int_type>;

    // using half_traits = vector_type_int_traits<member_type,16>;
    // using recursive_traits = vt_recursive<member_type,1,32,half_traits>;
    // using int_traits = sse42_1x16<int_type>;
    
    static type set1( member_type a ) {
	return _mm_set1_pi16( (unsigned short)a );
    }
    static type set( member_type a3, member_type a2,
		     member_type a1, member_type a0 ) {
	return _mm_set_pi16( a3, a2, a1, a0 );
    }
    static type set1inc( member_type a ) {
	return _mm_cvtsi64_m64( 0x03020100UL ) + set1( a );
    }

    static member_type lane( type a, int idx ) {
	// auto b = (_mm_cvtm64_si64(a) >> (idx*16)) & 0xffUL;
	int_type b;
	switch( idx ) {
	case 0: b = _mm_extract_pi16( a, 0 ); break;
	case 1: b = _mm_extract_pi16( a, 1 ); break;
	case 2: b = _mm_extract_pi16( a, 2 ); break;
	case 3: b = _mm_extract_pi16( a, 3 ); break;
	default:
	    assert( 0 && "should not get here" );
	}
	if constexpr ( is_customfp_v<member_type> )
	    return member_type(
		static_cast<typename member_type::int_type>( b ) );
	else
	    return static_cast<member_type>( b );
    }
    static member_type lane0( type a ) { return lane( a, 0 ); }
    static member_type lane1( type a ) { return lane( a, 1 ); }
    static member_type lane2( type a ) { return lane( a, 2 ); }
    static member_type lane3( type a ) { return lane( a, 3 ); }

    static itype castfp( type a ) { return a; }
    static type castint( type a ) { return a; }

    static type load( const member_type *addr ) {
	return *(const type *)addr;
    }
    static type loadu( const member_type *addr ) {
	return *(const type *)addr;
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

    static mask_type asmask( vmask_type v ) {
	assert( 0 && "NYI" );
	return mask_traits::setzero();
    }
    
    static type add( type a, type b ) {
	static_assert( !is_customfp_v<member_type>, "need special operation" );
	return _mm_add_pi16( a, b );
    }

    static type sub( type a, type b ) {
	static_assert( !is_customfp_v<member_type>, "need special operation" );
	return _mm_sub_pi16( a, b );
    }

    static type min( type a, type b ) {
	static_assert( !is_customfp_v<member_type>, "need special operation" );
	return blend( cmpgt( a, b, mt_vmask() ), a, b );
    }
    static type max( type a, type b ) {
	static_assert( !is_customfp_v<member_type>, "need special operation" );
	return blend( cmpgt( a, b, mt_vmask() ), b, a );
    }

    static type cmpeq( type a, type b, mt_vmask ) {
	return _mm_cmpeq_pi16( a, b );
    }
    static type cmpne( type a, type b, mt_vmask ) {
	return ~ _mm_cmpeq_pi16( a, b );
    }
    static type cmpgt( type a, type b, mt_vmask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm_cmpgt_pi16( a, b );
	else {
	    type one = set1( 0x8000 );
	    type ax = bitwise_xor( a, one );
	    type bx = bitwise_xor( b, one );
	    return _mm_cmpgt_pi16( ax, bx );
	}
    }
    static type cmpge( type a, type b, mt_vmask ) {
	return logical_or( cmpgt( a, b, mt_vmask() ),
			   cmpeq( a, b, mt_vmask() ) );
    }
    static type cmplt( type a, type b, mt_vmask ) {
	return cmpgt( b, a, mt_vmask() );
    }
    static type cmple( type a, type b, mt_vmask ) {
	return cmpge( b, a, mt_vmask() );
    }

    static mask_type cmpeq( type a, type b, mt_mask ) {
	return asmask( cmpeq( a, b, mt_vmask() ) );
    }
    static mask_type cmpne( type a, type b, mt_mask ) {
	return asmask( cmpne( a, b, mt_vmask() ) );
    }
    static mask_type cmpgt( type a, type b, mt_mask ) {
	return asmask( cmpgt( a, b, mt_vmask() ) );
    }
    static mask_type cmpge( type a, type b, mt_mask ) {
	return asmask( cmpge( a, b, mt_vmask() ) );
    }
    static mask_type cmplt( type a, type b, mt_mask ) {
	return asmask( cmplt( a, b, mt_vmask() ) );
    }
    static mask_type cmple( type a, type b, mt_mask ) {
	return asmask( cmple( a, b, mt_vmask() ) );
    }
    
    static type blend( type c, type a, type b ) {
	assert( 0 && "NYI" );
	return setzero();
    }
    static mask_type movemask( vmask_type a ) {
	assert( 0 && "NYI" );
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
	assert( 0 && "NYI" );
    }

    static type bitwiseor( type a, type b ) { return _mm_or_si64( a, b ); }

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

#else // GRAPTOR_USE_MMX

#include "graptor/target/sse42_2x4.h"

#endif // GRAPTOR_USE_MMX

#endif // GRAPTOR_TARGET_MMX_2x4_H
