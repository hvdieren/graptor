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
	    return _popcnt64( arg_traits::lane( a, 0 ) )
		+ _popcnt64( arg_traits::lane( a, 1 ) );
	} else if constexpr ( sizeof(T) == 8 && VL == 1 ) {
	    return _popcnt64( arg_traits::lane( a, 0 ) );
	} else if constexpr ( sizeof(T) == 8 ) {
	    assert( 0 && "NYI" );
	} else if constexpr ( sizeof(T) == 4 && VL == 1 ) {
	    return _popcnt32( arg_traits::lane( a, 0 ) );
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
	if constexpr ( sizeof(T) == 8 ) {
	    for( unsigned i=0; i < VL; ++i ) {
		ResultTy cnt = _tzcnt_u64( arg_traits::lane( a, i ) );
		if( cnt < 64 )
		    return cnt + i * 64;
	    }
	    return 64 * VL;
	} else if constexpr ( sizeof(T) <= 4 && VL == 1 ) {
	    return _tzcnt_u32( arg_traits::lane( a, 0 ) );
	} else if constexpr ( sizeof(T) == 4 && VL > 1 ) {
	    return vector_type_traits<int_type_of_size_t<sizeof(T)*2>,VL/2>
		::compute( a );
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

} // namespace target

#endif // GRAPTOR_TARGET_ALGO_H

