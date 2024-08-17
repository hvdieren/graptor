// -*- c++ -*-
#ifndef GRAPTOR_TARGET_SCALARBOOL_H
#define GRAPTOR_TARGET_SCALARBOOL_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include "graptor/itraits.h"
#include "graptor/target/decl.h"
#include "graptor/target/bitmask.h"

namespace target {

template<typename T>
struct scalar_int;

/***********************************************************************
 * Scalar bool
 ***********************************************************************/
struct scalar_bool {
public:
    using member_type = bool;
    using type = bool;
    using vmask_type = typename int_type_of_size<sizeof(bool)>::type;
    using itype = vmask_type;
    using int_type = vmask_type;

    using mtraits = mask_type_traits<1>;
    using mask_type = bool;

    // half_traits not defined (already down to scalar)
    using int_traits = scalar_int<int_type>;
    
    static const size_t size = sizeof(bool);
    static const size_t vlen = 1;

    static void print( std::ostream & os, type v ) {
	os << '(' << lane0(v) << ')';
    }

    static type setone() { return type(0xff); }
    static type setoneval() { return true; }
    
    static type create( member_type a0_ ) { return a0_; }
    static type set1( member_type a ) { return a; }
    static type set( member_type a0 ) { return a0; }
    static GG_INLINE type setzero() { return false; }
    static type set1inc( member_type a ) { return a; } // TODO: error
    static type set1inc0() { return 0; }

    static member_type lane( type a, int idx ) { return member_type(a); }
    static member_type lane0( type a ) { return member_type(a); }

    template<typename U>
    type convert( U u ) {
	static_assert( std::is_integral_v<U>, "requires integral type" );
	return u;
    }
    
    // Casting here is to do a signed cast of a bitmask for logical masks
    template<typename T2>
    static auto convert_to( type a ) {
	return T2(a);
    }

    template<typename U>
    static auto asvector( type a ) { return U(a); }

    static type abs( type a ) { return a; }
    
    static type add( type s, mask_type m, type a, type b ) {
	return m ? a + b : s;
    }
    static type add( type a, type b ) { return a + b; }
    static type sub( type a, type b ) { return a - b; }
    static type mul( type a, type b ) { return a * b; }
    static type div( type a, type b ) { return a / b; }

    static member_type reduce_add( type a ) { return a; }
    static member_type reduce_bitwiseor( type a ) { return a; }

    static type logical_and( type a, type b ) { return a & b; }
    static type logical_andnot( type a, type b ) { return !a & b; }
    static type logical_or( type a, type b ) { return a | b; }
    static type logical_invert( type a ) { return !a; }
    static type bitwise_and( type a, type b ) { return a & b; }
    static type bitwise_or( type a, type b ) { return a | b; }
    static type bitwise_invert( type a ) { return !a; }

    static vmask_type cmpeq( type a, type b, target::mt_vmask ) { return a == b ? member_type(0xff) : member_type(0); }
    static vmask_type cmpne( type a, type b, target::mt_vmask ) { return a == b ? member_type(0) : member_type(0xff); }
    static bool cmpne( type a, type b, target::mt_bool ) { return a != b; }
    static bool cmpeq( type a, type b, target::mt_bool ) { return a == b; }
    static bool cmplt( type a, type b, target::mt_bool ) { return a < b; }
    static bool cmple( type a, type b, target::mt_bool ) { return a <= b; }
    static bool cmpgt( type a, type b, target::mt_bool ) { return a > b; }
    static bool cmpge( type a, type b, target::mt_bool ) { return a >= b; }
    static vmask_type cmplt( type a, type b, target::mt_vmask ) { return a < b ? member_type(0xff) : member_type(0); }
    static vmask_type cmpgt( type a, type b, target::mt_vmask ) { return a > b ? member_type(0xff) : member_type(0); }
    static vmask_type cmpge( type a, type b, target::mt_vmask ) { return a >= b ? member_type(0) : member_type(0); }

    static type blend( type c, type a, type b ) { return c != 0 ? b : a; }
    static member_type blendm( mask_type m, type l, type r ) {
	return m ? r : l;
    }
    static member_type reduce_setif( type val ) { return val; }
    static mask_type from_int( type a ) { return movemask( a ); }
    static mask_type movemask( type a ) { return a ? true : false; }
    static mask_type asvector( type a ) { return a; }
    static mask_type asmask( type a ) { return movemask( a ); }
    static member_type reduce_logicalor( type val ) { return member_type(val != 0); }
    static member_type reduce_min( type val ) { return val; }
    static member_type reduce_max( type val ) { return val; }
    static type load( const member_type *a ) { return *a; }

    // TODO: temporal loads of bools
    static type ntload( const member_type *a ) { return *a; }
    // static type ntloadu( const member_type *a ) { return ntload( a ); }
    static type loadu( const member_type *a ) { return *a; }
    static void store( member_type *addr, type val ) { *addr = member_type(val); }
    static void storeu( member_type *addr, type val ) { *addr = member_type(val); }

    // Byte-width non-temporal stores not supported by target hardware
    static void ntstore( member_type *addr, type val ) { store( addr, val ); }

    template<typename IdxT>
    static type
    gather( member_type *addr, IdxT idx ) {
	return addr[idx];
    }
    template<typename IdxT>
    static type
    gather( member_type *addr, IdxT idx, mask_type mask ) {
	return mask ? addr[idx] : 0;
    }
    template<typename IdxT>
    static void
    scatter( member_type *a, IdxT b, type c ) {
	a[b] = lane0(c);
    }
    template<typename IdxT>
    static void
    scatter( member_type *a, IdxT b, type c, vmask_type mask ) {
	if( lane0(mask) )
	    a[b] = lane0(c);
    }

    static bool
    cas( volatile member_type * addr, member_type oldval, member_type newval ) {
	return __sync_bool_compare_and_swap(
	    const_cast<member_type *>( addr ), oldval, newval );
    }
};

template<>
inline auto scalar_bool::convert_to<bool>( type a ) {
    return a;
}

} // namespace target

#endif // GRAPTOR_TARGET_SCALARBOOL_H
