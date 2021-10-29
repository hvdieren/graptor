// -*- c++ -*-
#ifndef GRAPTOR_GRAPTOR_H
#define GRAPTOR_GRAPTOR_H

/***********************************************************************
 * \mainpage Graptor
 *
 * \section sec_intro Introduction
 * Graptor is a shared-memory graph processing system that uses an
 * embedded domain-specific language and compiler to generate vectorized
 * code for graph processing problems. Its programming interface is
 * organized through higher-order methods (map and scan) that may be applied
 * to the vertex set or the edge set of a graph.
 * 
 * \section sec_install Installation instructions
 * See the README.md file in the top-level directory
 *
 * \section sec_api Programming interface
 * This section links to the main interface components: lazy executors
 * (\ref page_lazy), vertex map (\ref page_vmap), edge map (\ref page_emap)
 * and the mapped function language (\ref page_ast).
 ***********************************************************************/

#define unlikely(x) __builtin_expect((x),0)
#define likely(x) __builtin_expect((x),1)

#include "config.h"

/*======================================================================*
 * Basic utilities and data types
 *======================================================================*/
#include "graptor/utils.h"
#include "graptor/legacy/gettime.h"
#include "graptor/itraits.h"

/*======================================================================*
 * Partitioning and parallelism control
 *======================================================================*/
#if HAS_NUMA(GRAPTOR_PARALLEL)
#define USE_NUMA
#define NUMA 1
#include <numa.h>
#include <numaif.h>
static const int num_numa_node = numa_num_configured_nodes();
#else
#ifdef USE_NUMA
#undef USE_NUMA
#endif // USE_NUMA
#if NUMA
#warning NUMA set to true
#endif
#define NUMA 0
static const int num_numa_node = 1;
#endif

#include "graptor/partitioner.h"
#if GRAPTOR_PARALLEL == BACKEND_cilk || GRAPTOR_PARALLEL == BACKEND_cilk_numa
#include "graptor/backend/cilk.h"
#define CILK 1
#elif GRAPTOR_PARALLEL == BACKEND_openmp \
    || GRAPTOR_PARALLEL == BACKEND_openmp_numa
#include "graptor/backend/openmp.h"
#define CILK 0
#else
#include "graptor/backend/sequential.h"
#define CILK 0
#endif

template<bool parallel, typename Fn>
void map_partition( const partitioner & part, Fn fn ) {
    if constexpr ( parallel )
	map_partitionL( part, fn );
    else
	map_partition_serialL( part, fn );
}

template<typename Fn>
void map_partition( const partitioner & part, Fn fn ) {
    map_partitionL( part, fn );
};

/*======================================================================*
 * Basics
 *======================================================================*/
#include "graptor/mm/mm.h"
#include "graptor/mm.h"

/*======================================================================*
 * Legacy stuff
 *======================================================================*/
#include "graptor/legacy/utils.h"
#include "graptor/legacy/parallel.h"
#include "graptor/legacy/parseCommandLine.h"
#include "graptor/legacy/graph-numa.h"

/*======================================================================*
 * Graptor internals
 *======================================================================*/
#include "graptor/target/vector.h"
#include "graptor/simd/decl.h"
#include "graptor/encoding.h"

/*======================================================================*
 * Graptor data types and API
 *======================================================================*/
#include "graptor/graph/cgraph.h"
#include "graptor/frontier.h"
#include "graptor/dsl/vmap_utils.h"
#include "graptor/dsl/edgemap.h"

/*======================================================================*
 * Graptor API implementation
 *======================================================================*/
#include "graptor/frontier_impl.h"
#include "graptor/longint_impl.h"

/*======================================================================*
 * Legacy API, partially retargeted to Graptor
 *======================================================================*/
#ifndef PAPI_CACHE
#define PAPI_CACHE 0
#endif

#if PAPI_CACHE
#include "legacy/papi_code.h"
#endif

// #include "graptor/legacy/ligra-numa.h"

#endif // GRAPTOR_GRAPTOR_H
