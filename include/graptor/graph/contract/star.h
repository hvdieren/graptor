// -*- c++ -*-
#ifndef GRAPTOR_GRAPH_CONTRACT_STAR_H
#define GRAPTOR_GRAPH_CONTRACT_STAR_H

namespace contract {

namespace star {

/**
 * Find stars (and claws) in the graph. A path consists of a central vertex
 * and a subset of its neighbours, where none of the neighbours are connected.
 * A claw is a special case of a star with 1+3 vertices.
 *
 * Not clear if the star pattern is really useful to speed up processing:
 * - supernodes: what algorithms benefit? CC? TC, clique count
 * - vertex expansion: would still need to activate central vertex for most algo
 *   unless if degree(central) == #terminals in star AND no frontier needed for
 *   a subsequent vertexmap.
 * - edge expansion: for central vertex: could vectorize; for terminals: just
 *   one edge is processed (to central vertex).
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
void find_stars( const GraphCSx & G,
		 bool * covered_edges,
		 Fn & record_star,
		 edge_cover<VID,EID> & ecov ) {
    VID n = G.numVertices();
    const EID * const index = G.getIndex();
    const VID * const edges = G.getEdges();
    
    // Iterate over all vertices with degree >= 3 (would be more efficient on a
    // degree-sorted list of vertices).
    for( VID v=0; v < n; ++v ) {
	EID vs = index[v];
	EID ve = index[v+1];
	if( ve - vs >= 3 ) {
	    vertex_set<VID> s;
	    s.push( &edges[vs], &edges[ve] );

	    for( EID e=vs; e < ve; ++e ) {
		VID u = edges[e];
		if( intersect_is_empty( s, &edges[index[u]],
					index[u+1]-index[u] ) )
		    s.remove( u );
		if( s.size() < 3 )
		    break;
	    }
		
	    if( s.size() >= 3 ) {
		allocate all edges in ecov; drop neighbour if not possible;
		if( s.size() >= 3 )
		    record_star( v, s );
		else
		    release edges ???;
	    }
	}
    }
}

} // namespace star

} // namespace contract

#endif // GRAPTOR_GRAPH_CONTRACT_STAR_H
