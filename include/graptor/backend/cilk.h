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

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

#define parallel_for cilk_for

inline uint32_t graptor_num_threads() {
    return __cilkrts_get_nworkers();
}

#ifdef USE_NUMA
// Copied from Cilk include/internal/abi.h:
typedef uint64_t cilk64_t;
typedef void (*__cilk_abi_f64_t)(void *data, cilk64_t low, cilk64_t high);

extern "C" {
    void __cilkrts_cilk_for_numa_64(__cilk_abi_f64_t body, void *data,
				    cilk64_t count, int grain);
}
#endif // USE_NUMA

template<typename Fn>
struct PartitionOp {
    const partitioner & part;
    Fn data;

    PartitionOp( const partitioner & part_, Fn data_ )
	: part( part_ ), data( data_ ) { }

    static void func(void *data, uint64_t low, uint64_t high) {
	PartitionOp<Fn> * datap = reinterpret_cast<PartitionOp<Fn> *>( data );
	parallel_for( uint64_t n=datap->part.numa_start_of(low);
		      n < datap->part.numa_start_of(high); ++n )
	    datap->data( n );
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
	parallel_for( uint64_t p=datap->part.numa_start_of(low);
		      p < datap->part.numa_start_of(high); ++p ) {
	    VID ps = datap->part.start_of( p );
	    VID pe = datap->part.end_of( p );
	    _Pragma( STRINGIFY(cilk grainsize = _SCAN_BSIZE) ) parallel_for(
		VID v=ps; v < pe; ++v )
		datap->data( v );
	}
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
	parallel_for( uint64_t p=datap->part.numa_start_of(low);
		      p < datap->part.numa_start_of(high); ++p ) {
	    EID ps = datap->part.edge_start_of( p );
	    EID pe = datap->part.edge_end_of( p );
	    _Pragma( STRINGIFY(cilk grainsize = _SCAN_BSIZE) ) parallel_for(
		EID e=ps; e < pe; ++e )
		datap->data( e );
	}
    }
};

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
    parallel_for( VID vname=0; vname < _np; ++vname ) {
	fn( vname );
    }
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
    VID np = part.get_num_partitions();
    parallel_for( unsigned int p=0; p < np; ++p ) {
	VID s = part.start_of( p );
	VID e = part.end_of( p );
	parallel_for( VID v=s; v < e; ++v )
	    fn( v );
    }
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
    parallel_for( unsigned int p=0; p < np; ++p ) {
	EID ps = part.edge_start_of( p );
	EID pe = part.edge_end_of( p );
	parallel_for( EID e=ps; e < pe; ++e )
	    fn( e );
    }
}
#endif

#endif // GRAPHGRIND_BACKEND_CILK_H
