// -*- C++ -*-
#ifndef GRAPTOR_GRAPH_TRANSFORM_RMSELF_H
#define GRAPTOR_GRAPH_TRANSFORM_RMSELF_H

#include "graptor/itraits.h"

namespace graptor {

namespace graph {

/*!======================================================================*
 * Remove self edges
 *=======================================================================*/

::GraphCSx
remove_self_edges( const ::GraphCSx & G, bool verbose = true ) {
    VID n = G.numVertices();
    EID m = G.numEdges();
    const EID * index = G.getIndex();
    const VID * edges = G.getEdges();
    const float * weights = G.getWeights() ? G.getWeights()->get() : nullptr;

    if( verbose ) {
	std::cerr << "Removing self edges\n";
	std::cerr << "  |V|=" << n << "\n";
	std::cerr << "  |E|=" << m << "\n";
    }

    mmap_ptr<EID> new_index( n );

    if( verbose )
	std::cerr << "Finding self-edges\n";
    parallel_for( VID u=0; u < n; ++u ) {
	EID es = index[u];
	EID ee = index[u+1];
	EID n_self = 0;
	for( EID e=es; e < ee; ++e ) {
	    VID v = edges[e];
	    if( v == u )
		++n_self;
	}
	new_index[u] = n_self;
    }

    if( verbose )
	std::cerr << "Rebuilding index\n";
    EID tmp = 0;
    for( VID u=0; u < n; ++u ) {
	EID nxt = index[u+1] - index[u] - new_index[u];
	new_index[u] = tmp;
	tmp += nxt;
    }
    new_index[n] = tmp;

    if( verbose )
	std::cerr << "Creating new graph object\n";
    ::GraphCSx UG( n, tmp, -1, G.isSymmetric() );
    EID * uidx = UG.getIndex();
    VID * uedge = UG.getEdges();

    if( verbose )
	std::cerr << "Rebuilding edge list (" << tmp << " edges)\n";
    
    parallel_for( VID u=0; u < n; ++u ) {
	EID es = index[u];
	EID ee = index[u+1];
	EID j = new_index[u];
	uidx[u] = j;
	for( EID e=es; e < ee; ++e ) {
	    VID v = edges[e];
	    if( v != u )
		uedge[j++] = v;
	}
    }
    uidx[n] = new_index[n];

    new_index.del();

    UG.build_degree();

    return UG;
}

} // namespace graph

} // namespace graptor

#endif // GRAPTOR_GRAPH_TRANSFORM_RMSELF_H
