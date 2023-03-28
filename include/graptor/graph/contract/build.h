// -*- c++ -*-
#ifndef GRAPTOR_GRAPH_CONTRACT_BUILD_H
#define GRAPTOR_GRAPH_CONTRACT_BUILD_H

#include "graptor/graph/contract/contract.h"
#include "graptor/dsl/emap/edgechunk.h"

namespace contract {

namespace clique {
} // namespace clique

template<typename lVID, typename lEID, typename XX>
bool find_clique( const GraphCSx & G, lVID v, CliqueCollector & cc ) {
    ;
}


template<typename lVID, typename lEID>
GraphContract<lVID,lEID>
contract_graph( const GraphCSx & G, lVID npart ) {
    // Extract graph properties
    lVID n = G.numVertices();
    lEID m = G.numEdges();
    const lEID * const index = G.getIndex();
    const lVID * const degree = G.getDegree();
    const lVID * const edges = G.getEdges();

    // Partition the vertex set. This isolates the write-set per partition
    // and ensures no overlap of the write sets when partitions are processed
    // in parallel.
    partitioner part( npart, n );
    partition_vertex_list(
	nullptr, n, index, index, m, npart,
	[&]( lVID p, lVID from, lVID to ) {
	    part.as_array()[p] = to - from;
	} );
    part.as_array()[npart] = m;
    // After this call, vertex boundaries have been initialised, but edge
    // boundaries are undefined.
    part.compute_starts_inuse();

    // Safety check - all vertices accounted for
    assert( part.start_of( npart ) == n );
    // assert( part.edge_start_of( npart ) == m );

    // Find patterns in each partition, independently.
    // TODO: deal with overlaps
    for( lVID p=0; p < npart; ++p ) {
	lVID vs = part.start_of( p );
	lVID ve = part.end_of( p );

	// Find all patterns for each vertex in the partition
	for( lVID v=vs; v < ve; ++v ) {
	    find_clique( G, v );
	}
    }
}

} // namespace contract

#endif // GRAPTOR_GRAPH_CONTRACT_BUILD_H
