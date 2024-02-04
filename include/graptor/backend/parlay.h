// -*- C++ -*-
#ifndef GRAPHGRIND_BACKEND_PARLAY_H
#define GRAPHGRIND_BACKEND_PARLAY_H

#include "config.h"
#include "graptor/partitioner.h"
#include "graptor/legacy/parallel.h"
#include "graptor/utils.h"

#include <parlay/parallel.h>
#include <parlay/thread_specific.h>

inline uint32_t graptor_num_threads() {
    return parlay::num_workers();
}

template<typename T, typename Fn>
void parallel_loop( const T it_start, const T it_end, Fn fn ) {
    parlay::parallel_for( it_start, it_end, fn );
}

template<typename T, typename Fn>
void parallel_loop( const T it_start, const T it_end, const int grain, Fn fn ) {
    parlay::parallel_for( it_start, it_end, fn, grain );
}

template<typename Fn>
void map_partitionL( const partitioner & part, Fn fn ) {
    VID _np = part.get_num_partitions();
    parallel_loop( VID(0), _np, 1, [&]( uint64_t i ) { fn( i ); } );
}

template<typename Fn>
void map_vertexL( const partitioner & part, Fn fn ) {
    using VID = typename partitioner::VID;
    unsigned int np = part.get_num_partitions();
    VID ps = part.start_of( 0 );
    VID pe = part.end_of( np-1 );
    parallel_loop( ps, pe, [&]( VID v ) { fn( v ); } );
}

template<typename Fn>
void map_edgeL( const partitioner & part, Fn fn ) {
    using EID = typename partitioner::EID;
    unsigned int np = part.get_num_partitions();
    EID ps = part.edge_start_of( 0 );
    EID pe = part.edge_end_of( np-1 );
    parallel_loop( ps, pe, [&]( EID e ) { fn( e ); } );
}

template<typename Fn>
void map_workers( Fn && fn ) {
    uint32_t num_threads = graptor_num_threads();

    if( num_threads == 1 ) {
	fn( 0 );
	return;
    }
    
    volatile uint32_t num_wait = num_threads;
    
    parallel_loop( (uint32_t)0, num_threads, 1, [&]( uint32_t t ) {
	// Ensure each iteration of this loop is mapped onto a different
	// worker. Needs some busy waiting to trigger work stealing.
	__sync_fetch_and_add( &num_wait, -1 );
	while( num_wait != 0 ); // busy wait

	// Now we are certain that all workers are active on a different
	// iteration of this loop.
	fn( t );
    } );
}

// Define in the end so to not affect parlay::parallel_for above
#define parallel_for for

#endif // GRAPHGRIND_BACKEND_PARLAY_H
