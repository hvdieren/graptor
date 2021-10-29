// -*- c++ -*-
#ifndef GRAPTOR_TARGET_BITMASK16_H
#define GRAPTOR_TARGET_BITMASK16_H

namespace target {

/***********************************************************************
 * Bitmask traits with 16 lanes
 ***********************************************************************/
template<>
struct mask_type_traits<16> {
    using type = uint16_t;
    using pointer_type = type;
    using member_type = bool;

    static constexpr unsigned short vlen = 16;

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
    static type setone() { return ~ type(0); }
    static type setalternating() { return 0x5555U; }

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
	using vtraits = vector_type_traits_vl<T,16>;
	return vtraits::asvector( m );
    }

    static type from_int( unsigned m ) {
	return m & 0xffffU;
    }

    static typename mask_type_traits<8>::type lower_half( type m ) {
	return m & ( (type(1) << 8) - 1 );
    }
    static typename mask_type_traits<8>::type upper_half( type m ) {
	return m >> 8;
    }

    static type set_pair( typename mask_type_traits<8>::type hi,
			  typename mask_type_traits<8>::type lo ) {
	return ( type(hi) << 8 ) | type(lo);
    }
};

} // namespace target

#endif // GRAPTOR_TARGET_BITMASK16_H
