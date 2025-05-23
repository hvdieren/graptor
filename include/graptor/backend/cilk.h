// -*- C++ -*-
#ifndef GRAPHGRIND_BACKEND_CILK_H
#define GRAPHGRIND_BACKEND_CILK_H

#include "config.h"
#include "graptor/partitioner.h"
#include "graptor/legacy/parallel.h"
#include "graptor/utils.h"

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <sched.h>
#include <errno.h>
#include <cstring>
#include <utility>
#include <algorithm>

#define USE_CILK_API

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

#if defined(__cilk)
#define parallel_for cilk_for
#else
#define parallel_for for
#endif

inline uint32_t graptor_num_threads() {
    return __cilkrts_get_nworkers();
}

// Copied from Cilk include/internal/abi.h:
typedef uint64_t cilk64_t;
typedef void (*__cilk_abi_f64_t)(void *data, cilk64_t low, cilk64_t high);

extern "C" {
    void __cilkrts_cilk_for_64(__cilk_abi_f64_t body, void *data,
			       cilk64_t count, int grain);
#ifdef USE_NUMA
    void __cilkrts_cilk_for_numa_64(__cilk_abi_f64_t body, void *data,
				    cilk64_t count, int grain);
#endif // USE_NUMA
}

template<typename Fn>
struct PartitionOp {
    const partitioner & part;
    Fn data;

    PartitionOp( const partitioner & part_, Fn data_ )
	: part( part_ ), data( data_ ) { }

    static void func(void *data, uint64_t low, uint64_t high) {
	PartitionOp<Fn> * datap = reinterpret_cast<PartitionOp<Fn> *>( data );
	parallel_loop( datap->part.numa_start_of( low ),
		       datap->part.numa_start_of( high ),
		       1, // Iterate over few partitions per NUMA node
		       [&]( uint64_t n ) { datap->data( n ); } );
    }
};

template<typename Fn>
struct VertexOp {
    const partitioner & part;
    Fn data;

    VertexOp( const partitioner & part_, Fn data_ )
	: part( part_ ), data( data_ ) { }

    static void func(void *data, uint64_t low, uint64_t high) {
	using VID = typename partitioner::VID;
	VertexOp<Fn> * datap = reinterpret_cast<VertexOp<Fn> *>( data );
	// TODO: single parallel_loop over all vertices in partition?
	//       how to deal with padding vertices at end of each partition?
	const partitioner & part = datap->part;
	parallel_loop(
	    part.numa_start_of(low),
	    part.numa_start_of(high),
	    1, // Iterate over few partitions
	    [&]( auto p ) {
		VID ps = part.start_of_vbal( p );
		VID pe = part.end_of_vbal( p );
		for( VID v=ps; v < pe; ++v ) {
		    datap->data( v );
		}
	    } );
    }
};

template<typename Fn>
struct EdgeOp {
    const partitioner & part;
    Fn data;

    EdgeOp( const partitioner & part_, Fn data_ )
	: part( part_ ), data( data_ ) { }

    static void func(void *data, uint64_t low, uint64_t high) {
	using EID = typename partitioner::EID;
	EdgeOp<Fn> * datap = reinterpret_cast<EdgeOp<Fn> *>( data );
	parallel_loop(
	    datap->part.numa_start_of(low),
	    datap->part.numa_start_of(high),
	    1, // Iterate over few partitions
	    [&]( auto p ) {
		EID ps = datap->part.edge_start_of( p );
		EID pe = datap->part.edge_end_of( p );
		for( EID e=ps; e < pe; ++e ) {
		    datap->data( e );
		}
	    } );
    }
};

template<typename Fn>
struct LoopOp {
    uint64_t it_start;
    Fn data;

    LoopOp( uint64_t it_start_, Fn data_ )
	: it_start( it_start_ ), data( data_ ) { }

    static void func(void *data, uint64_t low, uint64_t high) {
	LoopOp<Fn> * datap = reinterpret_cast<LoopOp<Fn> *>( data );
	const uint64_t start = datap->it_start;
	Fn & data_fn = datap->data;
	for( uint64_t i=low; i < high; ++i ) {
	    data_fn( start + i );
	}
    }
};

template<typename T, typename Fn>
void parallel_loop( const T it_start, const T it_end, Fn fn ) {
    LoopOp<Fn> op( static_cast<uint64_t>( it_start ), fn );
    __cilkrts_cilk_for_64( &LoopOp<Fn>::func,
			   reinterpret_cast<void *>( &op ),
			   static_cast<uint64_t>( it_end - it_start ),
			   2048 );
}

template<typename T, typename Fn>
void parallel_loop( const T it_start, const T it_end, const int grain, Fn fn ) {
    LoopOp<Fn> op( static_cast<uint64_t>( it_start ), fn );
    __cilkrts_cilk_for_64( &LoopOp<Fn>::func,
			   reinterpret_cast<void *>( &op ),
			   static_cast<uint64_t>( it_end - it_start ),
			   grain );
}

#if NUMA
template<typename Fn>
void map_partitionL( const partitioner & part, Fn fn ) {
    PartitionOp<Fn> op( part, fn );
    __cilkrts_cilk_for_numa_64( &PartitionOp<Fn>::func,
				reinterpret_cast<void *>( &op ),
				num_numa_node, 1 );
    
}
#else // not NUMA
template<typename Fn>
void map_partitionL( const partitioner & part, Fn fn ) {
    VID _np = part.get_num_partitions();
    parallel_loop( VID(0), _np, 1, [&]( uint64_t i ) { fn( i ); } );
}
#endif // NUMA

#if NUMA
template<typename Fn>
void map_vertexL( const partitioner & part, Fn fn ) {
    VertexOp<Fn> op( part, fn );
    __cilkrts_cilk_for_numa_64( &VertexOp<Fn>::func,
				reinterpret_cast<void *>( &op ),
				num_numa_node, 1 );
    
}
#else
template<typename Fn>
void map_vertexL( const partitioner & part, Fn fn ) {
    using VID = typename partitioner::VID;
    unsigned int np = part.get_num_partitions();
    VID ps = part.start_of( 0 );
    VID pe = part.end_of( np-1 );
    parallel_loop( ps, pe, [&]( VID v ) { fn( v ); } );
}
#endif

#if NUMA
template<typename Fn>
void map_edgeL( const partitioner & part, Fn fn ) {
    EdgeOp<Fn> op( part, fn );
    __cilkrts_cilk_for_numa_64( &EdgeOp<Fn>::func,
				reinterpret_cast<void *>( &op ),
				num_numa_node, 1 );
    
}
#else
template<typename Fn>
void map_edgeL( const partitioner & part, Fn fn ) {
    using EID = typename partitioner::EID;
    unsigned int np = part.get_num_partitions();
    EID ps = part.edge_start_of( 0 );
    EID pe = part.edge_end_of( np-1 );
    parallel_loop( ps, pe, [&]( EID e ) { fn( e ); } );
}
#endif

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

#endif // GRAPHGRIND_BACKEND_CILK_H
