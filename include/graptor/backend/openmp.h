// -*- C++ -*-
#ifndef GRAPHGRIND_BACKEND_OPENMP_H
#define GRAPHGRIND_BACKEND_OPENMP_H

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

#ifdef _OPENMP
#include <omp.h>
#endif

#define parallel_for _Pragma("omp parallel for") for

inline uint32_t graptor_num_threads() {
    return omp_get_max_threads();
}

template<typename T, typename Fn>
void parallel_loop( const T it_start, const T it_end, Fn fn ) {
#pragma omp parallel for
    for( T i=it_start; i != it_end; ++i ) {
	fn( i );
    }
}

template<typename T, typename Fn>
void parallel_loop( const T it_start, const T it_end, const int grain, Fn fn ) {
#pragma omp parallel for schedule(static,grain)
    for( T i=it_start; i != it_end; ++i ) {
	fn( i );
    }
}

#if NUMA
template<typename Fn>
void map_partitionL( const partitioner & part, Fn fn ) {
#if 0
#pragma omp parallel for num_threads(num_numa_node) proc_bind(spread)
    for( unsigned n=0; n < num_numa_node; ++n ) {
	auto lo = part.numa_start_of(n);
	auto hi = part.numa_end_of(n);
	std::cerr << "nested parallel case...\n";
// #pragma omp parallel for num_threads(??) schedule(static,1) proc_bind(close)
#pragma omp parallel for schedule(static,1) proc_bind(close)
	for( auto p=lo; p < hi; ++p )
	    fn( p );
    }
#else
    // Shoud result in desired mapping if assuming static scheduling and
    // #partitions multiple of threads, however, no option for dynamic
    // scheduling to achieve automatic load balancing
    auto np = part.get_num_partitions();
#pragma omp parallel for
    for( unsigned p=0; p < np; ++p ) {
	fn( p );
    }
#endif
}
#else // not NUMA
template<typename Fn>
void map_partitionL( const partitioner & part, Fn fn ) {
    auto num_partitions = part.get_num_partitions();
#pragma omp parallel for
    for( decltype(num_partitions) p=0; p < num_partitions; ++p )
	fn( p );
}
#endif // NUMA

#if NUMA
template<typename Fn>
void map_vertexL( const partitioner & part, Fn fn ) {
    map_partitionL( part, [&]( typename partitioner::PID p ) {
	auto lo = part.start_of(p);
	auto hi = part.end_of(p);
	for( auto v=lo; v < hi; ++v )
	    fn( v );
    } );
}
#else
template<typename Fn>
void map_vertexL( const partitioner & part, Fn fn ) {
    auto n = part.get_num_vertices();
#pragma omp parallel for
    for( typename partitioner::VID v=0; v < n; ++v )
	fn( v );
}
#endif

#endif // GRAPHGRIND_BACKEND_OPENMP_H
