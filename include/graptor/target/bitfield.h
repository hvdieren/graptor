// -*- c++ -*-
#ifndef GRAPTOR_TARGET_BITFIELD_H
#define GRAPTOR_TARGET_BITFIELD_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include "graptor/target/decl.h"
#include "graptor/bitfield.h"

namespace target {


/***********************************************************************
 * bitfields of 2 or 4 bits; vector fits in scalar register
 ***********************************************************************/
template<unsigned short Bits, unsigned short nbytes, typename Enable = void>
struct bitfield_24_byte {
    static_assert( Bits == 1 || Bits == 2 || Bits == 4,
		   "assuming a whole number of bitfields per byte" );
    static_assert( nbytes <= 8,
		   "restrict to using scalar registers" );
public:
    static constexpr unsigned short bits = Bits;
    // static constexpr size_t W = nbytes;
    static constexpr size_t vlen = 8 * nbytes / bits;
    static constexpr unsigned short size = nbytes;
    static constexpr unsigned short factor = (8*size) / bits;

    using member_type = bitfield<bits>;
    using type = int_type_of_size_t<size>;
    using vmask_type = type;
    using pointer_type = unsigned char;
    // using itype = __m256i;
    // using int_type = uint8_t;

    using mt_preferred = target::mt_vmask;

    using mtraits = mask_type_traits<vlen>;
    using mask_type = typename mtraits::type;

    // using half_traits = bitfield_24_byte<bits,nbytes/2>;
    // using recursive_traits = vt_recursive<member_type,1,32,half_traits>;
    // using int_traits = avx2_1x32<int_type>;
    
    static type setzero() { return 0; }
    static type setone() { return ~(type)0; }
    static type set1( member_type a ) {
	if constexpr ( bits == 1 ) {
	    return a ? ~type(0) : type(0);
	} else if constexpr ( bits == 2 ) {
	    static const uint64_t pattern[4] = {
		0x0,
		0x5555555555555555,
		0xaaaaaaaaaaaaaaaa,
		0xffffffffffffffff
	    };
	    uint64_t val = pattern[a.get() & 3];
	    return (type)val; // shorten as needed
	} else
	    assert( 0 && "NYI" );
    }

    static auto lower_half( type a ) {
	// Halt recursive template instantiation
	if constexpr ( size > 1 ) {
	    return int_type_of_size_t<size/2>(
		a & ((type(1)<<((vlen/2)*bits))-1) );
	} else {
	    assert( 0 && "NYI" );
	    return 0;
	}
    }
    static auto upper_half( type a ) {
	// Halt recursive template instantiation
	if constexpr ( size > 1 ) {
	    return int_type_of_size_t<size/2>( a >> (vlen/2) );
	} else {
	    assert( 0 && "NYI" );
	    return 0;
	}
    }
    
    template<typename T>
    static type set_pair( T hi,
			  int_type_of_size_t<size-sizeof(T)> lo ) {
	return ( type(hi) << ((vlen/2)*bits) ) | type(lo);
    }

    static bool is_all_false( type a ) {
	// TODO: consider only top bit
	if constexpr ( bits == 1 )
	    return a == 0;
	else
	    assert( 0 && "NYI" );
    }

    static member_type lane( type a, unsigned short lane ) {
	type b = a >> ( lane * bits );
	type m = ( type(1) << bits ) - 1;
	return member_type( b & m );
    }

    static type setlane( type a, member_type b, unsigned short lane ) {
	auto c = (type)b.get();
	const type m = ( type(1) << bits ) - 1;
	type d = c & m;
	type e = a & ~( m << ( lane * bits ) );
	type f = e | ( d << ( lane * bits ) );
	return f;
    }

    static mask_type asmask( vmask_type a ) {
	if constexpr ( bits == 1 ) {
	    return a;
	} else if constexpr ( bits == 2 ) {
	    constexpr uint64_t mask = 0xaaaaaaaaaaaaaaaaULL;
	    return (mask_type) _pext_u64( a, mask );
	}
	assert( 0 && "NYI" );
    }

    template<typename T>
    static auto asvector( type m ) {
	using vtraits = vector_type_traits_vl<T,8>;
	return vtraits::asvector( m );
    }

    static vmask_type asvector( mask_type mask ) {
	if constexpr ( bits == 1 ) {
	    return mask;
	} else if constexpr ( bits == 2 ) {
	    constexpr uint64_t select = 0xaaaaaaaaaaaaaaaaULL;
	    auto a = _pdep_u64( (uint64_t)mask, select );
	    return a;
	}
	assert( 0 && "NYI" );
    }

    static uint32_t find_first( type a ) {
	if constexpr ( bits == 1 && size <= 4 ) {
	    return _tzcnt_u32( ~(uint32_t)a );
	} else if constexpr ( bits == 1 && size <= 8 ) {
	    return _tzcnt_u64( ~(uint64_t)a );
	} else {
	    assert( 0 && "NYI" );
	    return 0;
	}
    }
    static uint32_t find_first( type a, vmask_type m ) {
	if constexpr ( bits == 1 && size <= 4 ) {
	    return _tzcnt_u32( (uint32_t)logical_andnot( a, m ) );
	} else if constexpr ( bits == 1 && size <= 8 ) {
	    return _tzcnt_u64( (uint64_t)logical_andnot( a, m ) );
	} else {
	    assert( 0 && "NYI" );
	    return 0;
	}
    }

    static type blendm( vmask_type m, type a, type b ) {
	return blend( m, a, b );
    }
    static type blend( vmask_type m, type a, type b ) {
	// First ensure top bit in lane is copied over to other bit positions
	// in lane
	auto mm = m;
	if constexpr ( bits == 1 ) {
	    // mm is ok
	} else if constexpr ( bits == 2 ) {
	    constexpr uint64_t hi_mask = 0xaaaaaaaaaaaaaaaaULL;
	    mm = mm & hi_mask;
	    mm = mm | ( mm >> 1 );
	} else
	    assert( 0 && "NYI" );

	return ( mm & b ) | ( ~mm & a );
    }

    static mask_type cmpne( type a, type b, mt_mask ) {
	if constexpr ( bits == 1 )
	    return a ^ b;
	else
	    return asmask( cmpne( a, b, mt_vmask() ) );
    }
    
    static vmask_type cmpne( type a, type b, mt_vmask ) {
	if constexpr ( bits == 1 ) {
	    return a ^ b;
	} else if constexpr ( bits == 2 ) {
	    // only top bit is meaningful
	    auto ne = a ^ b;
	    auto c = ( ne << 1 ) | ne;
	    return c;
	}
	assert( 0 && "NYI" );
    }
    static vmask_type cmpeq( type a, type b, mt_vmask ) {
	return ~cmpne( a, b, mt_vmask() );
    }
    static vmask_type cmplt( type a, type b, mt_vmask ) {
	if constexpr ( bits == 2 ) {
	    // only top bit is meaningful
	    auto hi_lt = ~a & b;
	    auto hi_eq = ~a ^ b;
	    auto lo_lt = hi_lt;
	    auto c = hi_lt | ( hi_eq & ( lo_lt << 1 ) );
	    return c;
	}
	assert( 0 && "NYI" );
    }
    static vmask_type cmpgt( type a, type b, mt_vmask ) {
	return cmplt( b, a, mt_vmask() );
    }

    static vmask_type cmpneg( type a, mt_vmask ) { return a; }

    // Only top bit matters in logical (bit-)masks
    static type logical_and( type a, type b ) { return a & b; }
    static type logical_andnot( type a, type b ) { return ~a & b; }
    static type logical_or( type a, type b ) { return a | b; }
    static type logical_invert( type a ) { return ~a; }

    static type bitwise_and( type a, type b ) { return a & b; }
    static type bitwise_or( type a, type b ) { return a | b; }
    static type bitwise_invert( type a ) { return ~a; }

    // This incorrectly requires that all non-top-bits mimick the top bit.
    // Corrects if bits == 1, otherwise maybe not.
    static member_type reduce_logicaland( type a ) { return ~a == 0; }

    // Contrary to other implementations, the address passed into load and store
    // is a vector address, not a member address.
    static member_type loads( const pointer_type * a, unsigned int idx ) {
	return lane( a[idx/factor], idx % factor );
    }
    static type load( const type * a ) {
	return *a;
    }
    static type loadu( const type * a ) {
	return *a;
    }
    static void store( type *addr, type val ) {
	*addr = val;
    }
};

/***********************************************************************
 * bitfields of 2 or 4 bits; vector fits in vector register
 ***********************************************************************/
template<unsigned short Bits, unsigned short nbytes>
struct bitfield_24_byte<Bits,nbytes,std::enable_if_t<(nbytes>8)>> {
    static_assert( Bits == 1 || Bits == 2 || Bits == 4,
		   "assuming a whole number of bitfields per byte" );
    static_assert( nbytes > 8,
		   "specialised to non-scalar registers" );
public:
    static constexpr unsigned short bits = Bits;
    // static constexpr size_t W = nbytes;
    static constexpr size_t vlen = 8 * nbytes / bits;
    static constexpr unsigned short size = nbytes;
    static constexpr unsigned short factor = (8*size) / bits;

    using intm_type = uint32_t;
    using base_traits = vector_type_traits<intm_type,nbytes>;
    using bitf_traits = bitfield_24_byte<bits,sizeof(intm_type)>;

    using member_type = bitfield<bits>;
    using type = typename base_traits::type;
    using vmask_type = type;
    using pointer_type = unsigned char;
    using itype = type;
    // using int_type = uint8_t;

    using mtraits = mask_type_traits<vlen>;
    using mask_type = typename mtraits::type;

    using half_traits = bitfield_24_byte<bits,nbytes/2>;
    // using recursive_traits = vt_recursive<member_type,1,32,half_traits>;
    // using int_traits = avx2_1x32<int_type>;
    
    static type setzero() { return base_traits::setzero(); }
    static type setone() { return base_traits::setone(); }
    static type set1( member_type a ) {
	intm_type val = bitf_traits::set1( a );
	return base_traits::set1( (intm_type)val );
    }

    static auto lower_half( type a ) {
	auto b = base_traits::lower_half( a );
#if GRAPTOR_USE_MMX
	if constexpr ( half_traits::size <= 8 )
	    return _mm_cvtm64_si64( b );
	else
#endif
	    return b;
    }
    static auto upper_half( type a ) {
	auto b = base_traits::upper_half( a );
#if GRAPTOR_USE_MMX
	if constexpr ( half_traits::size <= 8 )
	    return _mm_cvtm64_si64( b );
	else
#endif
	    return b;
    }
    static type set_pair( typename half_traits::type hi,
			  typename half_traits::type lo ) {
	return base_traits::set_pair( hi, lo );
    }

    static bool is_all_false( type a ) {
	// TODO: consider only top bit
	if constexpr ( bits == 1 )
	    return a == 0;
	else
	    assert( 0 && "NYI" );
    }

    static member_type lane( type a, unsigned short lane ) {
	constexpr unsigned short F = (base_traits::W*8) / bits;
	auto l = base_traits::lane( a, lane / F );
	return bitf_traits::lane( l, lane % F );
    }

    static type setlane( type a, member_type b, unsigned short lane ) {
	constexpr unsigned short F = (base_traits::W*8) / bits;
	auto l = base_traits::lane( a, lane / F );
	auto ls = bitf_traits::setlane( l, b, lane % F );
	auto ll = base_traits::setlane( a, ls, lane / F );
	return ll;
    }

    static mask_type asmask( vmask_type a ) {
	auto lo = half_traits::asmask( lower_half( a ) );
	auto hi = half_traits::asmask( upper_half( a ) );
	if constexpr ( is_longint_v<mask_type> )
	    return mask_type( mtraits::set_pair( hi, lo ) );
	else
	    return (mask_type(hi) << (vlen/2)) | mask_type(lo);
    }

    static vmask_type asvector( mask_type mask ) {
	auto lo = half_traits::asvector( mtraits::lower_half( mask ) );
	auto hi = half_traits::asvector( mtraits::upper_half( mask ) );
	return set_pair( hi, lo );
    }

    static type blendm( vmask_type m, type a, type b ) {
	return blend( m, a, b );
    }
    static type blend( vmask_type m, type a, type b ) {
	// First ensure top bit in lane is copied over to other bit positions
	// in lane
	auto mm = m;
	if constexpr ( bits == 2 ) {
	    vmask_type hi_mask = base_traits::set1( 0xaaaaaaaa );
	    mm = base_traits::bitwise_and( mm, hi_mask );
	    mm = base_traits::bitwise_or( mm, base_traits::srli( mm, 1 ) );
	} else
	    assert( 0 && "NYI" );

	return base_traits::bitblend( mm, a, b );
    }

    static vmask_type cmpne( type a, type b, mt_vmask ) {
	if constexpr ( bits == 2 ) {
	    // only top bit is meaningful
	    auto eq = base_traits::bitwise_xor( a, b );
	    auto eqs = base_traits::slli( eq, 1 );
	    auto c = base_traits::bitwise_or( eqs, eq );
	    return c;
	}
	assert( 0 && "NYI" );
    }
    static vmask_type cmpeq( type a, type b, mt_vmask ) {
	return base_traits::bitwise_invert( cmpne( a, b, mt_vmask() ) );
    }
    static vmask_type cmplt( type a, type b, mt_vmask ) {
	if constexpr ( bits == 2 ) {
	    // only top bit is meaningful
	    // TODO: optimise with ternary logic
	    auto hi_lt = base_traits::bitwise_andnot( a, b );
	    auto hi_eq = base_traits::bitwise_xnor( a, b );
	    auto lo_lt = hi_lt;
	    auto lo_lt_s = base_traits::slli( lo_lt, 1 );
	    auto hi_eq_lo_lt = base_traits::bitwise_and( hi_eq, lo_lt_s );
	    auto c = base_traits::bitwise_or( hi_lt, hi_eq_lo_lt );
	    return c;
	}
	assert( 0 && "NYI" );
    }
    static vmask_type cmpgt( type a, type b, mt_vmask ) {
	return cmplt( b, a, mt_vmask() );
    }

    // Only top bit matters in logical (bit-)masks; so bitwise ops are enough
    static type logical_and( type a, type b ) {
	return base_traits::bitwise_and( a, b );
    }
    static type logical_or( type a, type b ) {
	return base_traits::bitwise_or( a, b );
    }
    static type logical_invert( type a ) {
	return base_traits::bitwise_invert( a );
    }

    static type bitwise_and( type a, type b ) {
	return base_traits::bitwise_and( a, b );
    }
    static type bitwise_or( type a, type b ) {
	return base_traits::bitwise_or( a, b );
    }
    static type bitwise_invert( type a ) {
	return base_traits::bitwise_invert( a );
    }

    // Contrary to other implementations, the address passed into load and store
    // is a vector address, not a member address.
    static type load( const type * a ) {
	return base_traits::load(
	    reinterpret_cast<const typename base_traits::member_type *>( a ) );
    }
    static type loadu( const type * a ) {
	return base_traits::loadu(
	    reinterpret_cast<const typename base_traits::member_type *>( a ) );
    }
    static void store( type *addr, type val ) {
	return base_traits::store(
	    reinterpret_cast<typename base_traits::member_type *>( addr ),
	    val );
    }
};


/***********************************************************************
 * scalar bitfields
 ***********************************************************************/
template<unsigned short Bits>
struct bitfield_scalar {
    static_assert( Bits == 1 || Bits == 2 || Bits == 4,
		   "assuming a whole number of bitfields per byte" );
public:
    static constexpr unsigned short bits = Bits;
    static constexpr unsigned short factor = 8 / bits;
    // static constexpr size_t W = nbytes;
    static constexpr size_t vlen = 1;
    static constexpr unsigned short size = 1;

    using member_type = bitfield<bits>;
    using type = member_type;
    using vmask_type = type;
    using itype = type;
    using int_type = type;

    using mtraits = mask_type_traits<1>;
    using mask_type = bool;

    using int_traits = bitfield_scalar<bits>;

    using vtraits = bitfield_24_byte<bits,1>;
    using pointer_type = typename vtraits::pointer_type;
    
    static type setzero() { return 0; }
    static type setone() { return ~(member_type)0; }
    static type set1( member_type a ) { return a; }

    static member_type lane( type a, unsigned short ) {
	return member_type( a );
    }
    static member_type setlane( type a, member_type b, unsigned short ) {
	return type( b );
    }

    static type bitwise_and( type a, type b ) { return a & b; }
    static type bitwise_andnot( type a, type b ) { return ~a & b; }
    static type bitwise_or( type a, type b ) { return a | b; }
    static type bitwise_xor( type a, type b ) { return a ^ b; }
    static type bitwise_invert( type a ) { return ~a; }
    static type logical_and( type a, type b ) { return a & b; }
    static type logical_or( type a, type b ) { return a | b; }
    static type logical_xor( type a, type b ) { return a ^ b; }
    static type logical_invert( type a ) { return ~a; }

    static bool cmpeq( type a, type b, mt_bool ) {
	return a.get() == b.get();
    }
    static bool cmpne( type a, type b, mt_bool ) {
	return a.get() != b.get();
    }
    static bool cmplt( type a, type b, mt_bool ) {
	return a.get() < b.get();
    }
    static vmask_type cmpneg( type a, mt_vmask ) { return a; }
    static vmask_type cmpneg( type a, mt_bool ) { return a != setzero(); }

    static type blendm( mask_type m, type a, type b ) {
	return blend( m, a, b );
    }
    static type blendm( vmask_type m, type a, type b ) {
	return blend( m, a, b );
    }
    static type blend( mask_type m, type a, type b ) {
	return m ? b : a;
    }
    static type blend( vmask_type m, type a, type b ) {
	return m ? b : a;
    }

    // Contrary to other implementations, the address passed into load and store
    // is a vector address, not a member address.
    template<typename index_type>
    static member_type loads( const pointer_type * a, index_type idx ) {
	return vtraits::lane( vtraits::load( a+(idx/factor) ), idx % factor );
    }
    template<typename index_type>
    static member_type load( const pointer_type * a, index_type idx ) {
	return vtraits::lane( vtraits::load( a+(idx/factor) ), idx % factor );
    }
    template<typename index_type>
    static member_type loadu( const pointer_type * a, index_type idx ) {
	return vtraits::lane( vtraits::loadu( a+(idx/factor) ), idx % factor );
    }
    static member_type load( const pointer_type * a ) { return load( a, 0 ); }
    static member_type loadu( const pointer_type * a ) { return loadu( a, 0 ); }
    template<typename index_type>
    static void store( pointer_type * a, index_type idx, member_type val ) {
	auto v = vtraits::load( a+(idx/factor) );
	auto w = vtraits::setlane( v, val, idx % factor );
	vtraits::store( a+(idx/factor), w );
    }
};


} // namespace target

#endif // GRAPTOR_TARGET_BITFIELD_H
