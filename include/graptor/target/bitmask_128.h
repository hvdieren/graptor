// -*- c++ -*-
#ifndef GRAPTOR_TARGET_BITMASK128_H
#define GRAPTOR_TARGET_BITMASK128_H

#if __SSE4_2__

#include "graptor/target/sse42_bitwise.h"

namespace target {

/***********************************************************************
 * Bitmask traits with 128 lanes.
 ***********************************************************************/
template<>
struct mask_type_traits<128> {
    typedef __m128i type;
    typedef uint64_t pointer_type;
    typedef bool member_type;

    static constexpr unsigned short width = sizeof(pointer_type)*8;
    static constexpr unsigned short vlen = 128;
    
    static bool lane( type m, unsigned short l ) {
	unsigned short ll = l / width;
	pointer_type w = _mm_extract_epi64( m, ll );
	uint64_t mm = uint64_t(1) << ( l % width );
	return ( w & mm ) != 0;
    }

    static bool lane0( type m ) { return lane( m, 0 ); }

    static type setzero() {
	return sse42_bitwise::setzero();
    }
    static type setone() {
	return sse42_bitwise::setone();
    }
    static type setl0( member_type a ) {
	return _mm_set_epi64x( 0, a ? 1 : 0 );
    }
    static type set1( bool a ) { return a ? setone() : setzero(); }
    // static type setalternating() { return 0x55; }

    static typename mask_type_traits<64>::type lower_half( type m ) {
	return sse42_bitwise::lower_half( m );
    }
    static typename mask_type_traits<64>::type upper_half( type m ) {
	return sse42_bitwise::upper_half( m );
    }

    static type set_pair( typename mask_type_traits<64>::type hi,
			  typename mask_type_traits<64>::type lo ) {
	return sse42_bitwise::set_pair( hi, lo );
    }

    // static uint32_t find_first( type a ) { }

    // static uint32_t find_first( type a, type m ) { }

    static type logical_and( type l, type r ) {
	return sse42_bitwise::bitwise_and( l, r );
    }
    static type logical_andnot( type l, type r ) {
	return sse42_bitwise::bitwise_andnot( l, r );
    }
    static type logical_or( type l, type r ) {
	return sse42_bitwise::bitwise_or( l, r );
    }
    static type logical_invert( type a ) {
	return sse42_bitwise::bitwise_invert( a );
    }
    static auto reduce_logicalor( type k ) {
	return mask_type_traits<64>::reduce_logicalor(
	    mask_type_traits<64>::logical_or(
		upper_half( k ), lower_half( k ) ) );
    }

    static type cmpne( type l, type r, target::mt_mask ) {
	return sse42_bitwise::bitwise_xor( l, r );
    }
    static bool cmpne( type l, type r, target::mt_bool ) {
	return !sse42_bitwise::is_zero( sse42_bitwise::bitwise_xor( l, r ) );
    }
    static type cmpeq( type l, type r, target::mt_mask ) {
	return sse42_bitwise::bitwise_xnor( l, r );
    }
    static bool cmpeq( type l, type r, target::mt_bool ) {
	return sse42_bitwise::is_zero( sse42_bitwise::bitwise_xor( l, r ) );
    }

    template<typename T>
    static auto asvector( type m ); /* {
	using vtraits = vector_type_traits_vl<T,vlen>;
	return vtraits::asvector( m );
    } */

    // No int is wide enough
    // static type from_int( unsigned m ) { }

#if 0
    static member_type loads( const pointer_type * addr, unsigned int idx ) {
	return ( addr[idx/8] >> (idx % 8) ) & 1;
    }
    static type load( const pointer_type * addr, unsigned int idx ) {
	// Aligned property implies full word is returned
	return addr[idx/8];
    }
    static type loadu( const pointer_type * addr, unsigned int idx ) {
	unsigned int shift = idx % 8;
	type a = addr[idx/8];
	type b = addr[idx/8+1];
	return ( ( a >> shift ) & ( ( type(1) << ( 8 - shift ) ) - type(1) ) )
	    | ( b << ( 8 - shift ) );
    }
    static void store( pointer_type * addr, unsigned int idx, type val ) {
	// Aligned property implies full word is stored
	addr[idx/8] = val;
    }
    static void storeu( pointer_type * addr, unsigned int idx, type val ) {
	unsigned int shift = idx % 8;
	type a = addr[idx/8];
	type b = addr[idx/8+1];
	a |= ( val << shift );
	b |= ( val >> shift ) & ( ( type(1) << ( 8 - shift ) ) - type(1) );
	addr[idx/8] = a;
	addr[idx/8+1] = b;
    }
    
    template<typename vindex_type>
    static type gather( pointer_type * addr, vindex_type idx ) {
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
#endif
};


} // namespace target

#endif // __SSE4_2__

#endif // GRAPTOR_TARGET_BITMASK128_H
