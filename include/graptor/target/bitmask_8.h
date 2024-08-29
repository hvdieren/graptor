// -*- c++ -*-
#ifndef GRAPTOR_TARGET_BITMASK8_H
#define GRAPTOR_TARGET_BITMASK8_H

namespace target {

/***********************************************************************
 * Bitmask traits with 8 lanes.
 ***********************************************************************/
template<>
struct mask_type_traits<8> {
    typedef uint8_t type;
    typedef type pointer_type;
    typedef bool member_type;

    static constexpr unsigned short vlen = 8;
    
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
	return k ^ k;
    }
    static type setone() { return 0xff; }
    static type setl0( member_type a ) { return type(!!a); }
    static type set1( bool a ) { return a ? setone() : setzero(); }
    static type setalternating() { return 0x55; }

    static typename mask_type_traits<4>::type lower_half( type m ) {
	return m & ( (type(1) << 4) - 1 );
    }
    static typename mask_type_traits<4>::type upper_half( type m ) {
	return m >> 4;
    }

    static type set_pair( typename mask_type_traits<4>::type hi,
			  typename mask_type_traits<4>::type lo ) {
	return ( hi << 4 ) | lo;
    }

    static bool is_zero( type a ) { return a == 0; }
    static bool is_ones( type a ) { return a == 0xff; }

    static uint32_t find_first( type a ) {
	return _tzcnt_u32( ~(uint32_t)a );
    }

    static uint32_t find_first( type a, type m ) {
	// a m | eligible
	// 0 0 | - 1
	// 0 1 | + 0
	// 1 0 | - 1
	// 1 1 | - 1
	// tzcnt( ~( a | ~m ) ) = tzcnt( ~a & m )
	// return find_first( logical_or( a, logical_invert( m ) ) );
	return _tzcnt_u32( (uint32_t)logical_andnot( a, m ) );
    }

    static uint32_t popcnt( type m ) {
	return _popcnt32( m );
    }
    static uint32_t tzcnt( type m ) {
	return _tzcnt_u32( m );
    }

    static type blend( type c, type no, type yes ) {
	return ( ~c & no ) | ( c & yes );
    }
    static type logical_and( type l, type r ) {
	return l & r;
    }
    static type logical_andnot( type l, type r ) {
	return ~l & r;
    }
    static type logical_or( type l, type r ) {
	return l | r;
    }
    static type logical_invert( type a ) {
	return ~a;
    }
    static auto reduce_logicalor( type k ) {
	return k != 0;
    }
    static auto reduce_logicaland( type k ) {
	// return k == (type)255;
	static_assert( sizeof(type) == 1, "pre-requisite for inversion" );
	return ~k == 0;
    }

    static type cmpne( type l, type r, target::mt_mask ) {
	return l ^ r;
    }
    static type cmpne( type l, type r, target::mt_bool ) {
	return l != r;
    }
    static type cmpeq( type l, type r, target::mt_mask ) {
	return ( l ^ r ) ^ (type)255;
    }
    static type cmpeq( type l, type r, target::mt_bool ) {
	return l == r;
    }

    static type slli( type a, size_t s ) {
	return (type)( a << s ); // truncate, please
    }
    static type srli( type a, size_t s ) {
	return (type)( a >> s ); // truncate, please
    }

    template<typename T>
    static auto asvector( type m ) {
	using vtraits = vector_type_traits_vl<T,8>;
	return vtraits::asvector( m );
    }

    static type from_int( unsigned m ) {
	return m & 0xffU;
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
    static void scatter( type * addr, vindex_type idx, type val ) {
	assert( 0 && "NYI" );
    }
    template<typename vindex_type>
    static void scatter( type * addr, vindex_type idx, type val, type mask ) {
	assert( 0 && "NYI" );
    }
};


} // namespace target

#endif // GRAPTOR_TARGET_BITMASK8_H
