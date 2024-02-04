// -*- c++ -*-
#ifndef GRAPTOR_BACKEND_BACKEND_H
#define GRAPTOR_BACKEND_BACKEND_H

#if GRAPTOR_PARALLEL == BACKEND_cilk || GRAPTOR_PARALLEL == BACKEND_cilk_numa
#include "graptor/backend/cilk.h"
#define CILK 1
#elif GRAPTOR_PARALLEL == BACKEND_openmp \
    || GRAPTOR_PARALLEL == BACKEND_openmp_numa
#include "graptor/backend/openmp.h"
#define CILK 0
#elif GRAPTOR_PARALLEL == BACKEND_parlay
#define CILK 0
#include "graptor/backend/parlay.h"
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

#endif // GRAPTOR_BACKEND_BACKEND_H

