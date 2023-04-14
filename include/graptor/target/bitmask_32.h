// -*- c++ -*-
#ifndef GRAPTOR_TARGET_BITMASK32_H
#define GRAPTOR_TARGET_BITMASK32_H

namespace target {

/***********************************************************************
 * Bitmask traits with 32 lanes
 ***********************************************************************/
template<>
struct mask_type_traits<32> {
    using type = uint32_t;
    using pointer_type = unsigned char;
    using member_type = bool;

    static constexpr unsigned short vlen = 32;
    
    static bool lane( type m, unsigned short l ) { return (m>>l)&1; }

    static bool lane0( type m ) { return m&1; }
    static bool lane1( type m ) { return m&2; }
    static bool lane2( type m ) { return m&4; }
    static bool lane3( type m ) { return m&8; }
    static bool lane4( type m ) { return m&16; }
    static bool lane5( type m ) { return m&32; }
    static bool lane6( type m ) { return m&64; }
    static bool lane7( type m ) { return m&128; }
    static bool lane8( type m ) { return lane0( m>>8 ); }
    static bool lane9( type m ) { return lane1( m>>8 ); }
    static bool lane10( type m ) { return lane2( m>>8 ); }
    static bool lane11( type m ) { return lane3( m>>8 ); }
    static bool lane12( type m ) { return lane4( m>>8 ); }
    static bool lane13( type m ) { return lane5( m>>8 ); }
    static bool lane14( type m ) { return lane6( m>>8 ); }
    static bool lane15( type m ) { return lane7( m>>8 ); }

    static type setzero() { return type(0); }
    static type setone() { return 0xffffffffU; }
    static type setalternating() { return 0x55555555U; }
    static type set1( bool v ) { return ~(type(v)-1); }

    static type logical_and( type l, type r ) {
	return l & r;
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

    template<typename T>
    static auto asvector( type m ) {
	using vtraits = vector_type_traits_vl<T,32>;
	return vtraits::asvector( m );
    }

    static type from_int( unsigned m ) {
	static_assert( sizeof(unsigned) == sizeof(type),
		       "requirement to convert int to mask verbatim" );
	return (type)m;
    }

    static typename mask_type_traits<16>::type lower_half( type m ) {
	return m & ( (type(1) << 16) - 1 );
    }
    static typename mask_type_traits<16>::type upper_half( type m ) {
	return m >> 16;
    }

    static type set_pair( typename mask_type_traits<16>::type hi,
			  typename mask_type_traits<16>::type lo ) {
	return ( hi << 16 ) | lo;
    }

    static member_type loads( const unsigned char * addr, unsigned int idx ) {
	constexpr size_t W = sizeof(unsigned char)*8;
	return ( addr[idx/W] >> (idx % W) ) & 1;
    }
    static member_type loads( const type * addr, unsigned int idx ) {
	constexpr size_t W = sizeof(type)*8;
	return ( addr[idx/W] >> (idx % W) ) & 1;
    }
};

} // namespace target

#endif // GRAPTOR_TARGET_BITMASK32_H
