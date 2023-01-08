// -*- c++ -*-
#ifndef GRAPTOR_TARGET_BITMASKAVX512F8_H
#define GRAPTOR_TARGET_BITMASKAVX512F8_H

namespace target {

/***********************************************************************
 * Bitmask traits with 8 lanes, using AVX512F extensions
 ***********************************************************************/
#if __AVX512F__

template<>
struct mask_type_traits<8> {
    static constexpr unsigned short vlen = 8;
    typedef __mmask8 type;
    typedef type pointer_type;
    typedef bool member_type;

    static bool lane( type m, unsigned short l ) { return (m>>l)&1; }

    static bool lane0( type m ) { return m&1; }
    static bool lane1( type m ) { return m&2; }
    static bool lane2( type m ) { return m&4; }
    static bool lane3( type m ) { return m&8; }
    static bool lane4( type m ) { return m&16; }
    static bool lane5( type m ) { return m&32; }
    static bool lane6( type m ) { return m&64; }
    static bool lane7( type m ) { return m&128; }

    static type setzero() {
	type k;
	return _kxor_mask8( k, k );
    }
    static type setone() {
	// A bit of extra effort to help the compiler generate good code
	type k;
	__asm__ ( "\n\t kxnorw %%k0,%%k0,%0" : "=k"(k) : );
	return k;
    }
    static type set1( bool a ) { return a ? setone() : setzero(); }

    static typename mask_type_traits<4>::type lower_half( type m ) {
	return m & ( (type(1) << 4) - 1 );
    }
    static typename mask_type_traits<4>::type upper_half( type m ) {
	return m >> 4;
    }

    static uint32_t find_first( type a ) {
    	return _tzcnt_u32( ~(uint32_t)a );
    }
    static uint32_t find_first( type a, type m ) {
	return _tzcnt_u32( (uint32_t)logical_andnot( a, m ) );
    }

    static auto logical_invert( type k ) {
	return _knot_mask8( k );
    }
    static type logical_and( type l, type r ) {
	return _kand_mask8( l, r );
    }
    static type logical_andnot( type l, type r ) {
	return _kandn_mask8( l, r );
    }
    static type logical_or( type l, type r ) {
	return _kor_mask8( l, r );
    }
    static auto reduce_logicalor( type k ) {
	return ! _kortestz_mask8_u8( k, k );
    }

    static type cmpne( type l, type r, target::mt_mask ) {
	return _kxor_mask8( l, r ); // CHECK
    }
    static type cmpne( type l, type r, target::mt_bool ) {
	return l != r;
    }
    static type cmpeq( type l, type r, target::mt_mask ) {
	return _kxnor_mask8( l, r );
    }
    static type cmpeq( type l, type r, target::mt_bool ) {
	return l == r;
    }

    template<typename T>
    static auto asvector( type m ) {
	using vtraits = vector_type_traits<T,sizeof(T)*8>;
	return vtraits::asvector( m );
    }

    static type from_int( unsigned m ) {
	return m & 0xffU;
    }

    static auto is_all_false( type k ) {
	return _kortestz_mask8_u8( k, k );
    }

    static type blendm( type sel, type l, type r ) {
	type t = _kand_mask8( sel, r );
	type f = _kandn_mask8( sel, l );
	return _kor_mask8( t, f );
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

#endif // __AVX512F__

} // namespace target

#endif // GRAPTOR_TARGET_BITMASKAVX512F8_H
