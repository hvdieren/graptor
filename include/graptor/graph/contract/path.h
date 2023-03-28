// -*- c++ -*-
#ifndef GRAPTOR_GRAPH_CONTRACT_PATH_H
#define GRAPTOR_GRAPH_CONTRACT_PATH_H

namespace contract {

namespace path {

/**
 * Find paths in the graph. A path consists of a sequence of connected edges
 * where the intermediate vertices have exactly degree 2.
 */
    
// p will never contain end-points; ecov will contain all edges
template<typename VID, typename EID>
VID trace_path( const GraphCSx & G,
		VID prev,
		VID cur,
		vertex_set<VID> & p,
		edge_cover<VID,EID> & ecov ) {
    const EID * const index = G.getIndex();
    const VID * const edges = G.getEdges();

    EID e = index[cur];
    EID deg = index[cur+1] - e;
    if( deg != 2 )
	return cur;

    p.push( cur );

    VID nxt = edges[e] == prev ? edges[e+1] : edges[e];
    ecov.cover( cur, nxt );
    return trace_path( G, cur, nxt, p );
}

// Parallel version:
// An alternative would be a connected-components-like algorithm where
// initially ID[v] == v if degree[v] == 2; else ID[v] = infty
// then, propagate finite degrees to neighbours until convergence.
// Only propagate a degree out of a vertex if the degree of the vertex is 2.
// Then, identify all unique non-infty IDs, which identifies the paths.
// Count unique IDs, relabel path IDs, identify IDs, cover edges.
    
template<typename VID, typename EID, typename Fn>
void find_paths( const GraphCSx & G,
		 bool * covered_edges,
		 Fn & record_path,
		 edge_cover<VID,EID> & ecov ) {
    VID n = G.numVertices();
    const EID * const index = G.getIndex();
    const VID * const edges = G.getEdges();
    
    // Iterate over all degree-2 vertices (would be more efficient on a
    // degree-sorted list of vertices).
    for( VID v=0; v < n; ++v ) {
	EID ve = index[v];
	if( index[v+1] - ve == 2 ) {
	    if( !covered_edges[ve] && !covered_edges[ve+1] ) {
		VID u = edges[ve];
		VID w = edges[ve+1];

		// Requires at least 4 edges in a path, with 3 intermediate
		// vertices (u, v, w).
		if( index[u+1] - index[u] == 2
		    && index[v+1] - index[v] == 2 ) {
		    // We have a path, now extend it
		    vertex_set<VID> p;
		    p.push( v );

		    ecov.cover( u, v );
		    ecov.cover( v, w );
		    
		    VID end_u = trace_path( G, v, u, p );
		    VID end_w = trace_path( G, v, w, p );

		    // Record path consisting of all vertices in p and the
		    // end-points end_u and end_w. The path is registered
		    // against the end-points for retrieval, not against the
		    // intermediate vertices.
		    // Although, perhaps, for supporting frontiers effectively
		    // it may be useful to register the path against all points,
		    // in case an intermediate vertex is activated but not any
		    // of the end-points.
		    record_path( end_u, end_w, p );
		}
	}
    }
}

} // namespace path

} // namespace contract

#endif // GRAPTOR_GRAPH_CONTRACT_PATH_H
