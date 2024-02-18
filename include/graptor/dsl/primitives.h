// -*- c++ -*-
#ifndef GRAPTOR_DSL_PRIMITIVES_H
#define GRAPTOR_DSL_PRIMITIVES_H

#include <bit>

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

template<typename T, typename I, typename F>
T sum_reduce_sequential( I from, I to, F && fn ) {
    T s = 0;
    for( I i=from; i < to; ++i )
	s += fn( i );
    return s;
}

template<typename T, typename I, typename F>
T sum_scan_sequential( T * sums, T in, I from, I to, F && fn ) {
    T s = in;
    for( I i=from; i < to; ++i ) {
	T v = fn( i );
	sums[i-from] = s;
	s += v;
    }
    return s;
}

template<typename T, typename I, typename F>
T sum_scan( T * sums, I from, I to, F && fn ) {
    static constexpr I BLOCK_SIZE = 1024;

    const I n = to - from;
    const I max_blocks = 16 * graptor_num_threads();
    const I nblocks0 = std::min( max_blocks, 1 + ( n - 1 ) / BLOCK_SIZE );

    if( nblocks0 <= 2 )
	return sum_scan_sequential( sums, T(0), from, to, fn );

    const I block_size = std::bit_ceil( ( n + nblocks0 - 1 ) / nblocks0 );
    const I nblocks = std::min( max_blocks, 1 + ( n - 1 ) / block_size );

    assert( nblocks * block_size >= n );
    assert( ( nblocks - 1 ) * block_size < n );
    
    T * psums = new T[nblocks];
    parallel_loop( I(0), nblocks, [&]( I b ) {
	I ps = from + b * block_size;
	I pe = std::min( ps + block_size, to );
	psums[b] = sum_reduce_sequential<T>( ps, pe, fn );
    } );
    const T total = sum_scan_sequential( psums, T(0), I(0), nblocks,
					 [=]( I b ) { return psums[b]; } );
    parallel_loop( I(0), nblocks, [&]( I b ) {
	I ps = from + b * block_size;
	I pe = std::min( ps + block_size, to );
	sum_scan_sequential( &sums[ps], psums[b], ps, pe, fn );
    } );

    delete[] psums;

    return total;
}

#endif // GRAPTOR_DSL_PRIMITIVES_H

