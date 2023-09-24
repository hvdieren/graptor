// -*- c++ -*-
#ifndef GRAPTOR_TARGET_BITMASK4_H
#define GRAPTOR_TARGET_BITMASK4_H

namespace target {

/***********************************************************************
 * Bitmask traits with 4 lanes
 ***********************************************************************/
template<>
struct mask_type_traits<4> {
    static constexpr unsigned short vlen = 4;
    using type = unsigned char;
    using pointer_type = type;
    using member_type = bool;

    static bool lane( type m, unsigned short l ) { return (m>>l)&1; }

    static bool lane0( type m ) { return m&1; }
    static bool lane1( type m ) { return m&2; }
    static bool lane2( type m ) { return m&4; }
    static bool lane3( type m ) { return m&8; }

    static type setzero() { return 0; }
    static type setone() { return 0xf; }

    static bool is_zero( type a ) { return a == 0; }
    static bool is_ones( type a ) { return a == 0xf; }

    static type set1( bool v ) { return v ? setone() : setzero(); }

    static bool cmpne( type l, type r, target::mt_bool ) {
	return l != r;
    }
    static bool cmpeq( type l, type r, target::mt_bool ) {
	return l == r;
    }
    static type cmpeq( type l, type r, target::mt_mask ) {
	return ~l ^ r;
    }

    static uint32_t find_first( type a ) {
	return _tzcnt_u32( ~(uint32_t)a );
    }

    static uint32_t find_first( type a, type m ) {
	// see bitmask_8.h for explanation
	return _tzcnt_u32( (uint32_t)logical_andnot( a, m ) );
    }

    static type logical_and( type l, type r ) { return l & r; }
    static type logical_andnot( type l, type r ) { return ~l & r; }
    static type logical_or( type l, type r ) { return l | r; }
    static type logical_invert( type a ) {
	return a ^ ( (type(1) << vlen) - 1 );
    }

    static typename mask_type_traits<2>::type lower_half( type m ) {
	return m & ( (type(1) << 2) - 1 );
    }
    static typename mask_type_traits<2>::type upper_half( type m ) {
	return m >> 2;
    }

    static type blendm( type sel, type l, type r ) {
	return ( sel & r ) | ( ~sel & l );
    }

    template<typename T>
    static auto asvector( type m ) {
	using vtraits = vector_type_traits_vl<T,4>;
	return vtraits::asvector( m );
    }

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
	using index_type
	    = typename int_type_of_size<sizeof(vindex_type)/vlen>::type;
	using itraits = vector_type_traits_vl<index_type,vlen>;
	type m = 0;
	index_type lidx3 = itraits::lane( idx, 3 );
	m |= ( (addr[lidx3/8] >> (lidx3 & 7)) & 1 );
	m <<= 1;
	index_type lidx2 = itraits::lane( idx, 2 );
	m |= ( (addr[lidx2/8] >> (lidx2 & 7)) & 1 );
	m <<= 1;
	index_type lidx1 = itraits::lane( idx, 1 );
	m |= ( (addr[lidx1/8] >> (lidx1 & 7)) & 1 );
	m <<= 1;
	index_type lidx0 = itraits::lane( idx, 0 );
	m |= ( (addr[lidx0/8] >> (lidx0 & 7)) & 1 );
	return m;
    }
};


} // namespace target

#endif // GRAPTOR_TARGET_BITMASK4_H
