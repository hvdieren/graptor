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
    return omp_get_num_threads();
}

#if NUMA
template<typename Fn>
void map_partitionL( const partitioner & part, Fn fn ) {
#pragma omp parallel for num_threads(num_numa_node) proc_bind(spread)
    for( unsigned n=0; n < num_numa_node; ++n ) {
	auto lo = part.numa_start_of(n);
	auto hi = part.numa_end_of(n);
#pragma omp parallel for schedule(static)
	for( auto p=lo; p < hi; ++p )
	    fn( p );
    }
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
#pragma omp parallel for schedule(dynamic) proc_bind(master)
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
