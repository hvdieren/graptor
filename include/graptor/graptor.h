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
#include "graptor/customfp.h"
#include "graptor/vcustomfp.h"

/*======================================================================*
 * Partitioning and parallelism control
 *======================================================================*/
#include "graptor/partitioner.h"
#include "graptor/backend/backend.h"

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
#include "graptor/frontier/frontier.h"
#include "graptor/dsl/vmap_utils.h"
#include "graptor/dsl/edgemap.h"

/*======================================================================*
 * Graptor API implementation
 *======================================================================*/
#include "graptor/frontier/impl.h"
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
