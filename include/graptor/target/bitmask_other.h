// -*- c++ -*-
#ifndef GRAPTOR_TARGET_BITMASK_OTHER_H
#define GRAPTOR_TARGET_BITMASK_OTHER_H

#include "graptor/itraits.h"
#include "graptor/longint.h"
#include "graptor/utils.h"

namespace target {

/***********************************************************************
 * Bitmask traits with non-power-of-2 lanes.
 ***********************************************************************/
template<unsigned short VL>
struct mask_type_traits {
    using type = int_type_of_size_t<next_ipow2((VL+7)/8)>;
    using pointer_type = unsigned char;
    using member_type = bool;

    static constexpr unsigned short vlen = VL;
    static constexpr unsigned short lo_VL = next_ipow2(VL/2);
    static constexpr unsigned short hi_VL = VL - lo_VL;

    static bool lane( type m, unsigned short l ) {
	if constexpr ( is_longint_v<type> )
	    return m.getbit( l );
	else
	    return ( m >> l ) & 1;
    }

    static bool lane0( type m ) { return m&1; }
    static bool lane1( type m ) { return m&2; }
    static bool lane2( type m ) { return m&4; }
    static bool lane3( type m ) { return m&8; }
    static bool lane4( type m ) { return m&16; }
    static bool lane5( type m ) { return m&32; }
    static bool lane6( type m ) { return m&64; }
    static bool lane7( type m ) { return m&128; }

    static type setzero() {
	if constexpr ( is_longint_v<type> )
	    return type::setzero();
	else {
	    type k;
	    return k ^ k;
	}
    }
    static type setone() { return ~type(0); }
    static type setl0( member_type a ) { return a ? type(1) : setzero(); }
    static type set1( bool a ) { return a ? setone() : setzero(); }

    static typename mask_type_traits<lo_VL>::type lower_half( type m ) {
	if constexpr ( is_longint<type>::value )
	    return m.get_lo();
	else
	    return m & ( (type(1) << lo_VL) - 1 );
    }
    static typename mask_type_traits<hi_VL>::type upper_half( type m ) {
	if constexpr ( is_longint<type>::value )
	    return m.get_hi();
	else
	    return m >> lo_VL;
    }

    // Order of arguments !!??
    static type set_pair( typename mask_type_traits<lo_VL>::type hi,
			  typename mask_type_traits<hi_VL>::type lo ) {
	if constexpr ( is_longint_v<type> )
	    return type::set_pair( hi, lo );
	else
	    return ( type(hi) << lo_VL ) | type(lo);
    }

    static type logical_and( type l, type r ) {
	return l & r;
    }
    static type logical_or( type l, type r ) {
	return l | r;
    }
    static auto logical_invert( type k ) {
	if constexpr ( VL == 8*sizeof(type) )
	    return ~k;
	else
	    return ( ~k ) & ( (type(1) << VL) - 1 );
    }
    static auto reduce_logicalor( type k ) {
	return k != setzero();
    }

    static auto is_all_false( type k ) {
	return k == setzero();
    }

    static type cmpne( type l, type r, target::mt_mask ) {
	return l ^ r;
    }
    static type cmpne( type l, type r, target::mt_bool ) {
	return l != r;
    }
    static type cmpeq( type l, type r, target::mt_mask ) {
	return ( l ^ r ) ^ setone();
    }
    static type cmpeq( type l, type r, target::mt_bool ) {
	return l == r;
    }

    template<typename T>
    static auto asvector( type m ) {
	using vtraits = vector_type_traits_vl<T,VL>;
	return vtraits::asvector( m );
    }

    static type from_int( unsigned m ) {
	static_assert( sizeof(m) >= VL );
	return m & ((type(1)<<VL)-1);
    }

    static member_type loads( const unsigned char * addr, unsigned int idx ) {
	return loads( reinterpret_cast<const pointer_type *>( addr ), idx );
    }
    static member_type loads( const type * addr, unsigned int idx ) {
	constexpr size_t W = sizeof(type)*8;
	static_assert( sizeof(pointer_type)*8 == VL, "sanity" );
	return ( addr[idx/W] >> (idx % W) ) & 1;
    }
    static type load( const type * addr, unsigned int idx ) {
	// Aligned property implies full word is returned
	// return addr[idx/8];
	assert( 0 && "NYI" );
	return setzero();
    }
    static type loadu( const type * addr, unsigned int idx ) {
	// unsigned int shift = idx % 8;
	// type a = addr[idx/8];
	// type b = addr[idx/8+1];
	// return ( ( a >> shift ) & ( ( type(1) << ( 8 - shift ) ) - type(1) ) )
	    // | ( b << ( 8 - shift ) );
	assert( 0 && "NYI" );
	return setzero();
    }
    static void store( pointer_type * addr, unsigned int idx, type val ) {
	// Aligned property implies full word is stored
	// addr[idx/8] = val;
	assert( 0 && "NYI" );
    }
    static void storeu( type * addr, unsigned int idx, type val ) {
	// unsigned int shift = idx % 8;
	// type a = addr[idx/8];
	// type b = addr[idx/8+1];
	// a |= ( val << shift );
	// b |= ( val >> shift ) & ( ( type(1) << ( 8 - shift ) ) - type(1) );
	// addr[idx/8] = a;
	// addr[idx/8+1] = b;
	assert( 0 && "NYI" );
    }
    
    template<typename vindex_type>
    static type gather( type * addr, vindex_type idx ) {
	assert( 0 && "NYI" );
	return setzero();
    }
    template<typename vindex_type>
    static void scatter( type * addr, vindex_type idx, type val,
			typename std::enable_if<sizeof(vindex_type) % 8 == 0>::type * = nullptr ) {
	assert( 0 && "NYI" );
    }
    template<typename vindex_type>
    static void scatter( type * addr, vindex_type idx, type val, type mask,
			typename std::enable_if<sizeof(vindex_type) % 16 == 0>::type * = nullptr ) {
	assert( 0 && "NYI" );
    }
};


} // namespace target

#endif // GRAPTOR_TARGET_BITMASK_OTHER_H
