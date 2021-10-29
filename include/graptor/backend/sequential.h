// -*- C++ -*-
#ifndef GRAPHGRIND_BACKEND_SEQUENTIAL_H
#define GRAPHGRIND_BACKEND_SEQUENTIAL_H

#include "graptor/partitioner.h"
#include "graptor/utils.h"

#define parallel_for for

inline uint32_t graptor_num_threads() {
    return 1;
}

template<typename Fn>
void map_partitionL( const partitioner & part, Fn fn ) {
    map_partition_serialL( part, fn );
}

template<typename Fn>
void map_vertexL( const partitioner & part, Fn fn ) {
    map_vertex_serialL( part, fn );
}

template<typename Fn>
void map_edgeL( const partitioner & part, Fn fn ) {
    map_edge_serialL( part, fn );
}
#endif // GRAPHGRIND_BACKEND_SEQUENTIAL_H
