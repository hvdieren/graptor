// -*- c++ -*-
#ifndef GRAPTOR_TARGET_BITMASK2_H
#define GRAPTOR_TARGET_BITMASK2_H

namespace target {

/***********************************************************************
 * Bit mask traits with 2 lanes
 ***********************************************************************/
template<>
struct mask_type_traits<2> {
    typedef unsigned char type;

    static bool lane( type m, unsigned short l ) { return (m>>l)&1; }

    static bool lane0( type m ) { return m & 1; }
    static bool lane1( type m ) { return m & 2; }

    static type setzero() { return 0; }
    static type setone() { return 0x3; }

    static type setalternating() { return 0x2; }

    static type logical_and( type l, type r ) { return l & r; }
    static type logical_andnot( type l, type r ) { return ~l & r; }
    static type logical_or( type l, type r ) { return l | r; }
    static auto logical_invert( type k ) { return ~k & 3; }

    static type blendm( type sel, type l, type r ) {
	type t = logical_and( sel, r );
	type f = logical_andnot( sel, l );
	return logical_or( t, f );
    }

    static typename mask_type_traits<1>::type lower_half( type m ) {
	return m & ( (type(1) << 1) - 1 );
    }
    static typename mask_type_traits<1>::type upper_half( type m ) {
	return m >> 1;
    }
};

} // namespace target

#endif // GRAPTOR_TARGET_BITMASK2_H
