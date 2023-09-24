// -*- c++ -*-
#ifndef GRAPTOR_TARGET_ALGO_H
#define GRAPTOR_TARGET_ALGO_H

#include "graptor/target/vector.h"
#include "graptor/target/conversion.h"

namespace target {

template<typename ResultTy, typename T, unsigned short VL>
struct tzcnt {
    using arg_traits = vector_type_traits_vl<T,VL>;
    using ret_traits = vector_type_traits_vl<ResultTy,VL>;

    static typename ret_traits::type compute( typename arg_traits::type a ) {
	// Cases where we have encoded a solution
	if constexpr ( ( sizeof(T) == 8 && ( VL == 4 || VL == 8 ) )
		       || ( sizeof(T) == 4 && ( VL == 8 || VL == 16 ) )
		       || VL == 1 )
	    return arg_traits::template tzcnt<ResultTy>( a );
	
	// Scale up width to 4 bytes such that we can use the available
	// implementations
	if constexpr ( sizeof(T) < 4 ) {
	    auto b = conversion_traits<T,uint32_t,VL>::convert( a );
	    return tzcnt<ResultTy,uint32_t,VL>::compute( b );
	}

	// Recursive case
	if constexpr ( is_vpair_v<decltype(a)> ) { // is_vt_recursive<arg_traits>::value ) {
	    auto lo = tzcnt<ResultTy,T,arg_traits::lo_half_traits::vlen>
		::compute( arg_traits::lower_half( a ) );
	    auto hi = tzcnt<ResultTy,T,arg_traits::hi_half_traits::vlen>
		::compute( arg_traits::upper_half( a ) );
	    return ret_traits::set_pair( hi, lo );
	}

	assert( 0 && "NYI" );
    }
};

template<typename ResultTy, typename T, unsigned short VL>
struct lzcnt {
    using arg_traits = vector_type_traits_vl<T,VL>;
    using ret_traits = vector_type_traits_vl<ResultTy,VL>;

    static typename ret_traits::type compute( typename arg_traits::type a ) {
	// Cases where we have encoded a solution
	if constexpr ( ( sizeof(T) == 8 && ( VL == 4 || VL == 8 ) )
		       || ( sizeof(T) == 4 && ( VL == 8 || VL == 16 ) )
		       || VL == 1 )
	    return arg_traits::template lzcnt<ResultTy>( a );
	
	// Scale up width to 4 bytes such that we can use the available
	// implementations
	if constexpr ( sizeof(T) < 4 ) {
	    auto b = conversion_traits<T,uint32_t,VL>::convert( a );
	    return lzcnt<ResultTy,uint32_t,VL>::compute( b );
	}

	// Recursive case
	if constexpr ( is_vpair_v<decltype(a)> ) { // is_vt_recursive<arg_traits>::value ) {
	    auto lo = lzcnt<ResultTy,T,arg_traits::lo_half_traits::vlen>
		::compute( arg_traits::lower_half( a ) );
	    auto hi = lzcnt<ResultTy,T,arg_traits::hi_half_traits::vlen>
		::compute( arg_traits::upper_half( a ) );
	    return ret_traits::set_pair( hi, lo );
	}

	assert( 0 && "NYI" );
    }
};

// Count across all lanes, returning a single scalar
template<typename ResultTy, typename T, unsigned short VL>
struct allpopcnt {
    using arg_traits = vector_type_traits_vl<T,VL>;
    using ret_traits = vector_type_traits_vl<ResultTy,1>;

    // The lane width doesn't really matter. So use width 8 always.
    static typename ret_traits::type compute( typename arg_traits::type a ) {
	// TODO:
	// * consider scalar variant: store vector in memory; perform scalar
	//   popcnt on each lane as fetched from memory/array and add up.
	if constexpr ( sizeof(T) == 8 && VL == 8 ) {
	    return arg_traits::reduce_add( arg_traits::popcnt( a ) );
	} else if constexpr ( sizeof(T) == 8 && VL == 4 ) {
	    // AVX2 8x4 has an algorithm due to Mula for per-lane popcount
	    auto cnt = arg_traits::popcnt( a );
	    // Add up the four counts
	    // Note: there is scope to adjust the summing procedures
	    //       if a non-default ResultTy is required.
	    // Note: by nature, the four counts are between 0 and 64
	    //      (boundaries inclusive). Hence, we can use a specialised
	    //      approach to adding up 4 counts, knowing that only
	    //      4 of the 32 bytes in the vector are non-zero.
	    // Interpret the vector as a vector of 16 2-byte ints (the ones
	    // in lanes not multiples of 4 are zero).
	    // Marginally faster than the simple
	    // return arg_traits::reduce_add( cnt );
	    using tr = vector_type_traits_vl<uint16_t,16>;
	    auto lo = tr::lower_half( cnt );
	    auto hi = tr::upper_half( cnt );
	    auto lh = _mm_add_epi16( lo, hi );  // or _mm_add_epi32
	    auto sh = _mm_shuffle_epi32( lh, 0b11011000 );
	    auto rd = _mm_hadd_epi32( sh, sh );
	    return tr::half_traits::lane0( rd );
	} else if constexpr ( sizeof(T) == 8 && VL == 2 ) {
	    // TODO: similar implementation to Mula's?
	    return _popcnt64( arg_traits::lane0( a ) )
		+ _popcnt64( arg_traits::lane( a, 1 ) );
	} else if constexpr ( sizeof(T) == 8 && VL == 1 ) {
	    return _popcnt64( a );
	} else if constexpr ( sizeof(T) == 8 ) {
	    assert( 0 && "NYI" );
	} else if constexpr ( sizeof(T) == 4 && VL == 1 ) {
	    return _popcnt32( a );
	} else {
	    return allpopcnt<ResultTy, uint64_t, arg_traits::size/8>
		::compute( a );
	}

	assert( 0 && "NYI" );
    }
};

// Trailing zero across all lanes, returning a single scalar
template<typename ResultTy, typename T, unsigned short VL>
struct alltzcnt {
    using arg_traits = vector_type_traits_vl<T,VL>;
    using ret_traits = vector_type_traits_vl<ResultTy,1>;

    static typename ret_traits::type compute( typename arg_traits::type a ) {
	if constexpr ( sizeof(T) == 8 && VL >= 2 ) {
	    using trw = vector_type_traits_vl<uint32_t,arg_traits::size/4>;
	    auto neq = trw::cmpne( a, trw::setzero(), target::mt_mask() );
	    if( neq == 0 ) [[unlikely]]
		return arg_traits::size * 8;
	    uint32_t l = alltzcnt<uint32_t,decltype(neq),1>::compute( neq );
	    uint32_t w = trw::lane( a, l );
	    return _tzcnt_u32( w ) + l * 32;
#if 0 // The variants below are slower on AMD EPYC 7702
	} else if constexpr ( sizeof(T) == 8 && VL > 2 ) {
	    // Recursively decompose into SSE42 subvector to minimize
	    // cross-subvector movement in unpacking into 64-bit words.
	    // Check for zero on a half vector as it is not slower than
	    // extracting half of the vector for the recursive step, or
	    // taking its tzcnt.
#if 0
	    auto h = arg_traits::lower_half( a );
	    ResultTy c = 0;
	    if( arg_traits::lo_half_traits::is_zero( h ) ) {
		h = arg_traits::upper_half( a );
		c = arg_traits::size * 8 / 2;
	    }
	    c += alltzcnt<ResultTy,T,VL/2>::compute( h );
	    return c;
#else
	    auto lo = arg_traits::lower_half( a );
	    ResultTy c = alltzcnt<ResultTy,T,VL/2>::compute( lo );
	    if( c == arg_traits::size * 8 / 2 ) {
		c += alltzcnt<ResultTy,T,VL/2>::compute(
		    arg_traits::upper_half( a ) );
	    }
	    return c;
#endif
#endif
	} else if constexpr ( sizeof(T) == 8 && VL > 1 ) {
	    // Unpack into 64-bit words
	    // For some targets, lane0(.) is faster than lane(.,0)
	    ResultTy cnt = _tzcnt_u64( arg_traits::lane0( a ) );
	    if( cnt < 64 )
		return cnt;

	    for( unsigned i=1; i < VL; ++i ) {
		ResultTy cnt = _tzcnt_u64( arg_traits::lane( a, i ) );
		if( cnt < 64 )
		    return cnt + i * 64;
	    }
	    return 64 * VL;
	} else if constexpr ( sizeof(T) == 8 && VL == 1 ) {
	    // Scalar
	    return _tzcnt_u64( a );
	} else if constexpr ( sizeof(T) <= 4 && VL == 1 ) {
	    // Scalar, short
	    return _tzcnt_u32( a );
	} else if constexpr ( sizeof(T) == 4 && VL > 1 ) {
	    // Recast 4-byte lanes into 8-byte lanes
	    return alltzcnt<ResultTy,uint64_t,VL/2>::compute( a );
	}

	assert( 0 && "NYI" );
    }
};

template<typename T, typename V, unsigned short VL>
struct confused_convert {
    using src_traits = vector_type_traits_vl<T, VL>;
    using dst_traits = vector_type_traits_vl<V, VL>;

    static typename dst_traits::type compute( typename src_traits::type a ) {
	if constexpr ( sizeof(V) == 2*sizeof(T)
		       && std::is_integral_v<V> && std::is_unsigned_v<V>
		       && std::is_integral_v<T> && std::is_unsigned_v<T> ) {
	    using ht = typename dst_traits::lo_half_traits;
	    // We will create two half vectors out of a, one having the
	    // even lanes, the other with the odd lanes. Then fuse together.
	    auto mask = ht::srli( ht::setone(), ht::B/2 );
	    auto even = ht::bitwise_and( mask, a );
	    // For upper half: 32 lower bits will be shifted out, no mask needed
	    auto odd = ht::srli( a, ht::B/2 );
	    // Will typically use vt_recursive for dst_traits.
	    return dst_traits::set_pair( odd, even );
	} else if constexpr ( is_vpair_v<decltype(a)> ) {
	    // Recursive case
	    auto lo = confused_convert<T,V,src_traits::lo_half_traits::vlen>
		::compute( src_traits::lower_half( a ) );
	    auto hi = confused_convert<T,V,src_traits::hi_half_traits::vlen>
		::compute( src_traits::upper_half( a ) );
	    return dst_traits::set_pair( hi, lo );
	} else {
	    // If the custom rules above do not apply, do a normal conversion.
	    return conversion_traits<T,V,VL>::convert( a );
	}
    }
};

template<typename T, typename V, unsigned short VL>
struct expand_bitset {
    using btr = vector_type_traits_vl<T, VL>;

    static V * compute( typename btr::type b, size_t nbits, V * a,
			V off ) {
	// TODO: in case of AVX512, operating on 16 bits at a time is possible
	
	if constexpr ( btr::size < 16 ) { // scalar bitmask
	    // same native vector length as btr but accessed using
	    // byte operations
	    using ctr = vector_type_traits_vl<uint8_t,btr::size>;
	    using vtr = vector_type_traits_vl<V, 8>;
	    using mtr = typename vtr::mask_traits;

	    // Take 8 bits out of b and expand indices
	    size_t groups = ( nbits + vtr::vlen - 1 ) / vtr::vlen;
	    typename btr::type brot = b;
	    typename btr::type bmask = (typename btr::type)0xff;
	    typename vtr::type v = vtr::set1inc( off );
	    typename vtr::type vinc
		= vtr::slli( vtr::setoneval(), 3 ); // splat 8
	    for( size_t i=0; i < groups; ++i ) {
		// Take 8 bits and rotate remaining
		uint8_t m8 = brot & bmask;
		brot >>= 8;

		// Place values as indicated by bits
		a = vtr::cstoreu_p( a, m8, v );
		v = vtr::add( v, vinc );
	    }
	    return a;
	}
#if 0
	else if constexpr ( btr::size == 16 ) { // SSE subvector
	    // same native vector length as btr but accessed using
	    // byte operations
	    using ctr = vector_type_traits_vl<uint8_t,btr::size>;
	    using vtr = vector_type_traits_vl<V, 8>;
	    using mtr = typename vtr::mask_traits;

	    // Take 8 bits out of b and expand indices
	    size_t groups = ( nbits + vtr::vlen - 1 ) / vtr::vlen;
	    typename btr::type brot = b;
	    typename vtr::type v = vtr::set1inc( off );
	    typename vtr::type vinc
		= vtr::slli( vtr::setoneval(), 3 ); // splat 8
	    for( size_t i=0; i < groups; ++i ) {
		// Take 8 bits and rotate remaining
		uint8_t m8 = ctr::lane0( brot );
		brot = btr::bsrli( brot, 1 );

		// Place values as indicated by bits
		a = vtr::cstoreu_p( a, m8, v );
		v = vtr::add( v, vinc );
	    }
	    return a;
	}
#endif
	else {
	    // Multiple SSE words. Recursively decompose and process.
	    constexpr size_t hbits = 8*btr::size/2;
	    if( nbits >= hbits ) {
		a = expand_bitset<T,V,VL/2>::compute(
		    btr::lower_half( b ), hbits, a, off );
		a = expand_bitset<T,V,VL/2>::compute(
		    btr::upper_half( b ), nbits - hbits, a, off + hbits );
	    } else {
		a = expand_bitset<T,V,VL/2>::compute(
		    btr::lower_half( b ), nbits, a, off );
	    }
	    return a;
	}
    }
    
    static V * compute_if( typename btr::type b, size_t nbits,
			   V * a, V off, size_t length ) {
	size_t nset = allpopcnt<size_t,T,VL>::compute( b );
	if( nset < length )
	    return a;

	return compute( b, nbits, a, off );
    }
};

} // namespace target

#endif // GRAPTOR_TARGET_ALGO_H

