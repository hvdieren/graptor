// -*- c++ -*-
#ifndef GRAPTOR_TARGET_SCALARFP_H
#define GRAPTOR_TARGET_SCALARFP_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>
#include <cmath>

#include "graptor/itraits.h"
#include "graptor/target/decl.h"
#include "graptor/target/scalar_int.h"
#include "graptor/target/scalar_bool.h"

namespace target {

/***********************************************************************
 * Scalar floating point values
 ***********************************************************************/
template<typename T>
struct scalar_fp {
    static constexpr size_t W = sizeof(T);
    static constexpr size_t B = 8*W;
    static constexpr size_t vlen = 1;
    static constexpr size_t size = W * vlen;

    typedef T member_type;
    typedef T type;
    using int_type = typename int_type_of_size<sizeof(T)>::type;
    using itype = typename int_type_of_size<sizeof(T)>::type;
    typedef itype vmask_type;

    static_assert( size * vlen == sizeof(type), "type size error" );

    using int_traits = scalar_int<int_type>;
    using mask_traits = mask_type_traits<vlen>;
    using mask_type = typename mask_traits::type;

    static member_type lane0( type a ) { return a; }
    static member_type lane( type a, int idx ) { return a; }

    static type setzero() { return member_type(0.0); }
    static type setone() { return member_type(1.0); } // should be ~0
    static type setoneval() { return member_type(1.0); }

    static type set1( member_type a ) { return a; }
    static type abs( type a ) { return std::abs( a ); }
    static type sqrt( type a ) { return ::sqrt(a); /* a * a;*/ }

    static type add( type a, type b ) { return a + b; }
    static type sub( type a, type b ) { return a - b; }
    static type add( type s, mask_type m, type a, type b ) {
	return m ? a + b : s;
    }
    static type mul( type a, type b ) { return a * b; }
    static type div( type a, type b ) { return a / b; }
    static type mul( type src, mask_type m, type a, type b ) {
	return m ? src : (a * b);
    }

    static vmask_type cmpne( type a, type b, target::mt_vmask ) { return a != b ? ~vmask_type(0) : vmask_type(0); }
    static vmask_type cmpeq( type a, type b, target::mt_vmask ) { return a == b ? ~vmask_type(0) : vmask_type(0); }
    static vmask_type cmplt( type a, type b, target::mt_vmask ) { return a < b ? ~vmask_type(0) : vmask_type(0); }
    static vmask_type cmple( type a, type b, target::mt_vmask ) { return a <= b ? ~vmask_type(0) : vmask_type(0); }
    static vmask_type cmpgt( type a, type b, target::mt_vmask ) { return a > b ? ~vmask_type(0) : vmask_type(0); }
    static vmask_type cmpge( type a, type b, target::mt_vmask ) { return a >= b ? ~vmask_type(0) : vmask_type(0); }

    static mask_type cmpne( type a, type b, target::mt_mask ) { return a != b; }
    static mask_type cmpeq( type a, type b, target::mt_mask ) { return a == b; }
    static mask_type cmplt( type a, type b, target::mt_mask ) { return a < b; }
    static mask_type cmple( type a, type b, target::mt_mask ) { return a <= b; }
    static mask_type cmpgt( type a, type b, target::mt_mask ) { return a > b; }
    static mask_type cmpge( type a, type b, target::mt_mask ) { return a >= b; }

    static bool cmpne( type a, type b, target::mt_bool ) { return a != b; }
    static bool cmpeq( type a, type b, target::mt_bool ) { return a == b; }
    static bool cmplt( type a, type b, target::mt_bool ) { return a < b; }
    static bool cmple( type a, type b, target::mt_bool ) { return a <= b; }
    static bool cmpgt( type a, type b, target::mt_bool ) { return a > b; }
    static bool cmpge( type a, type b, target::mt_bool ) { return a >= b; }

    static type blendm( mask_type m, type a, type b ) { return m ? b : a; }
    static type blend( bool m, type a, type b ) { return m ? b : a; }

    static member_type reduce_add( type val ) { return val; }
    static member_type reduce_add( type val, mask_type mask ) {
	return mask ? val : setzero();
    }
    static member_type reduce_mul( type val ) { return val; }
    static member_type reduce_mul( type val, mask_type mask ) {
	return mask ? val : setone();
    }

    static type asvector( mask_type mask ) {
	return (type)~member_type(0);
    }

    static type load( const member_type *a ) { return *a; }
    static type loadu( const member_type *a ) { return *a; }
    static void store( member_type *addr, type val ) { *addr = val; }
    static void storeu( member_type *addr, type val ) { *addr = val; }
    static type ntload( const member_type *a ) {
#if 0 // This is simply inefficient.
	// Need to use wider load
	void * addr = reinterpret_cast<__m128 *>(
	    uintptr_t(a) & ~uintptr_t(0xf) );
	__m128i ival;
	__asm__ __volatile__( "\n\tmovntdqa (%1),%0"
			      : "=x"(ival) : "r"(addr) : );
	type ret;
	if constexpr ( size == 4 ) {
	    __m128 fval = _mm_castsi128_ps( ival );
	    __m128 val;
	    uintptr_t off = ( uintptr_t(a) >> 2 ) & uintptr_t(0x3);
	    switch( off ) {
	    case 0: val = fval; break;
	    case 1: val = _mm_shuffle_ps( fval, fval, 1 ); break;
	    case 2: val = _mm_shuffle_ps( fval, fval, 2 ); break;
	    case 3: val = _mm_shuffle_ps( fval, fval, 3 ); break;
	    default:
		assert( 0 && "error" );
	    }
	    ret = _mm_cvtss_f32( val );
	} else
	    assert( 0 && "error" );

	return ret;
#else
	return load( a );
#endif
    }
    static void ntstore( member_type *addr, type val ) {
	if constexpr ( size == 4 ) {
	    __asm__ __volatile__ ( "\n\tmovnti %1,%0"
				   : "=m"(*addr) : "r"(val)
				   : "memory" );
	} else
	    store( addr, val );
	// _mm_stream_i32( addr, member_type(val) );
    }

    template<typename IdxT>
    static type gather( member_type *a, IdxT b ) { // IdxT should be scalar
	return load( &a[b] );
    }
    static type gather( member_type *a, itype b, mask_type mask ) {
	return mask ? load( &a[b] ) : setzero();
    }
    static void scatter( member_type *a, itype b, type c ) {
	a[b] = c;
    }
    static void scatter( member_type *a, itype b, type c, mask_type mask ) {
	if( mask )
	    a[b] = c;
    }

    static bool
    cas( volatile member_type * addr, member_type oldval, member_type newval ) {
	int_type * o = reinterpret_cast<int_type *>( &oldval );
	int_type * n = reinterpret_cast<int_type *>( &newval );
	return __sync_bool_compare_and_swap(
	    const_cast<int_type *>(
		reinterpret_cast<volatile int_type *>( addr ) ), *o, *n );
    }
};

} // namespace target

#endif // GRAPTOR_TARGET_SCALARFP_H
