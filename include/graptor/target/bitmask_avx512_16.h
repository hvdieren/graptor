// -*- c++ -*-
#ifndef GRAPTOR_TARGET_BITMASKAVX512F16_H
#define GRAPTOR_TARGET_BITMASKAVX512F16_H

namespace target {

/***********************************************************************
 * Bitmask traits with 16 lanes, using AVX512F extensions
 ***********************************************************************/
#if __AVX512F__
template<>
struct mask_type_traits<16> {
    static constexpr unsigned short vlen = 16;
    typedef __mmask16 type;
    typedef type pointer_type;
    typedef bool member_type;

    static bool lane( type v, unsigned short l ) { return (v>>l) & 1; }

    static bool lane0( type m ) { return m&1; }
    static bool lane1( type m ) { return m&2; }
    static bool lane2( type m ) { return m&4; }
    static bool lane3( type m ) { return m&8; }
    static bool lane4( type m ) { return m&16; }
    static bool lane5( type m ) { return m&32; }
    static bool lane6( type m ) { return m&64; }
    static bool lane7( type m ) { return m&128; }
    static bool lane8( type m ) { return m&256; }
    static bool lane9( type m ) { return m&512; }
    static bool lane10( type m ) { return m&1024; }
    static bool lane11( type m ) { return m&2048; }
    static bool lane12( type m ) { return m&4096; }
    static bool lane13( type m ) { return m&8192; }
    static bool lane14( type m ) { return m&16384; }
    static bool lane15( type m ) { return m&32768; }

    static type setzero() {
	// A bit of extra effort to help the compiler generate good code.
	type k;
	// return _kxor_mask16( k, k );
	__asm__ ( "\n\t kxorw %%k0,%%k0,%0" : "=k"(k) : );
	return k;
    }
    static type setone() {
	// A bit of extra effort to help the compiler generate good code.
	// type k;
	// return _kxor_mask16( k, k );
	type k;
	__asm__ ( "\n\t kxnorw %%k0,%%k0,%0" : "=k"(k) : );
	return k;
    }
    static type setl0( member_type a ) { return _mm512_int2mask(!!a); }
    static type set1( bool v ) { return _mm512_int2mask( ~(int(v)-1) ); }
    static type setalternating() { return 0x5555U; }

    static bool is_zero( type a ) { return _kortestz_mask16_u8( a, a ); }
    static bool is_ones( type a ) { return _kortestc_mask16_u8( a, a ); }

    static bool reduce_logicalandz( type a, type b ) {
	return _ktestz_mask16_u8( a, b );
    }

    static type cmpne( type l, type r, target::mt_mask ) {
	return _kxor_mask16( l, r );
    }
    static bool cmpne( type l, type r, target::mt_bool ) {
	type ne = cmpne( l, r, target::mt_mask() ); // any not equal
	return ! _kortestz_mask16_u8( ne, ne );
    }
    static type cmpeq( type l, type r, target::mt_mask ) {
	return _kxnor_mask16( l, r );
    }
    static bool cmpeq( type l, type r, target::mt_bool ) {
	type eq = cmpeq( l, r, target::mt_mask() ); // equal
	type ne = _knot_mask16( eq );
	return _kortestz_mask16_u8( ne, ne );
    }
    static type logical_invert( type k ) {
	return _knot_mask16( k );
    }
    static type logical_and( type l, type r ) {
	return _kand_mask16( l, r );
    }
    static type logical_and( type l, type m, type r ) {
	return _kand_mask16( l, _kand_mask16( m, r ) );
    }
    static type logical_andnot( type l, type r ) {
	return _kandn_mask16( l, r );
    }
    static type logical_or( type l, type r ) {
	return _kor_mask16( l, r );
    }
    static auto reduce_logicalor( type k ) {
	return ! _kortestz_mask16_u8( k, k );
    }
    static auto reduce_bitwiseor( type k ) {
	return reduce_logicalor( k );
    }

    static bool is_all_false( type k ) {
	return _kortestz_mask16_u8( k, k );
    }

    static bool is_all_true( type k ) {
	return _kortestc_mask16_u8( k, k );
    }

    static bool is_all_same( type k ) {
	unsigned char o;
	unsigned char z = _kortest_mask16_u8( k, k, &o );
	return z | o;
    }

    static bool is_logical_andnot_zero( type a, type b ) {
	return _ktestc_mask16_u8( a, b );
    }

    static type blend( type sel, type l, type r ) {
	return blendm( sel, l, r );
    }

    static type blendm( type sel, type l, type r ) {
	type t = logical_and( sel, r );
	type f = logical_andnot( sel, l );
	return logical_or( t, f );
    }

    static member_type loads( const pointer_type * addr, unsigned int idx ) {
	return ( addr[idx/16] >> (idx % 16) ) & 1;
    }
    static member_type loads( const uint8_t * addr, unsigned int idx ) {
	return loads( reinterpret_cast<const pointer_type *>( addr ), idx );
    }
    static type load( const pointer_type * addr, unsigned int idx ) {
	// Aligned property implies full word is returned
	return addr[idx/16];
    }
    static type loadu( const pointer_type * addr, unsigned int idx ) {
	unsigned int shift = idx % 16;
	type a = addr[idx/16];
	type b = addr[idx/16+1];
	return ( ( a >> shift ) & ( ( type(1) << ( 16 - shift ) ) - type(1) ) )
	    | ( b << ( 16 - shift ) );
    }
    static void store( pointer_type * addr, unsigned int idx, type val ) {
	// Aligned property implies full word is stored
	addr[idx/16] = val;
    }
    static void storeu( pointer_type * addr, unsigned int idx, type val ) {
	unsigned int shift = idx % 16;
	type a = addr[idx/16];
	type b = addr[idx/16+1];
	a |= ( val << shift );
	b |= ( val >> shift ) & ( ( type(1) << ( 16 - shift ) ) - type(1) );
	addr[idx/16] = a;
	addr[idx/16+1] = b;
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
	return m;
    }
    static typename mask_type_traits<8>::type upper_half( type m ) {
	return _kshiftri_mask16( m, 8 );
    }

    static type set_pair( typename mask_type_traits<8>::type hi,
			  typename mask_type_traits<8>::type lo ) {
	return _kor_mask16( _kshiftli_mask16( hi, 8 ), lo );
    }

    static uint32_t popcnt( type m ) {
	return _popcnt32( m );
    }
    static uint32_t tzcnt( type m ) {
	return _tzcnt_u32( m );
    }
};

#endif // __AVX512F__

} // namespace target

#endif // GRAPTOR_TARGET_BITMASKAVX512F16_H
