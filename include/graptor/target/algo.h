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


} // namespace target

#endif // GRAPTOR_TARGET_ALGO_H

