// -*- c++ -*-
#ifndef GRAPTOR_DSL_PRIMITIVES_H
#define GRAPTOR_DSL_PRIMITIVES_H

#include "graptor/partitioner.h"
#include "graptor/dsl/vertexmap.h"

template<typename ToTy, typename FromTy>
void __attribute__((noinline))
copy_cast( const partitioner & part, ToTy * to, FromTy * from ) {
    if constexpr ( is_bitfield<ToTy>::value ) {
	using Enc = array_encoding_bit<ToTy::bits>;
	expr::array_ro<FromTy,VID,expr::aid_frontier_old> af( from );
	expr::array_ro<FromTy,VID,expr::aid_frontier_new,Enc>
	    at( reinterpret_cast<typename Enc::storage_type *>( to ) );
	make_lazy_executor( part )
	    .vertex_map( [&]( auto v ) { return at[v] = af[v]; } )
	    .materialize();
    } else if constexpr ( is_bitfield<FromTy>::value ) {
	using Enc = array_encoding_bit<FromTy::bits>;
	expr::array_ro<ToTy,VID,expr::aid_frontier_old,Enc>
	    af( reinterpret_cast<typename Enc::storage_type *>( from ) );
	expr::array_ro<ToTy,VID,expr::aid_frontier_new> at( to );
	make_lazy_executor( part )
	    .vertex_map( [&]( auto v ) { return at[v] = af[v]; } )
	    .materialize();
    } else {
	expr::array_ro<FromTy,VID,expr::aid_frontier_old> af( from );
	expr::array_ro<ToTy,VID,expr::aid_frontier_new> at( to );
	make_lazy_executor( part )
	    .vertex_map( [&]( auto v ) {
		    return at[v] = expr::cast<ToTy>( af[v] );
		} )
	    .materialize();
    }
}

#endif // GRAPTOR_DSL_PRIMITIVES_H

