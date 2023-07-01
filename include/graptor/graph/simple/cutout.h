// -*- C++ -*-
#ifndef GRAPHGRIND_GRAPH_SIMPLE_CUTOUT_H
#define GRAPHGRIND_GRAPH_SIMPLE_CUTOUT_H

#include <type_traits>
#include <algorithm>

#include "graptor/graph/GraphCSx.h"

namespace graptor {

namespace graph {

template<typename lVID, typename lEID>
class NeighbourCutOutXP {
public:
    using VID = lVID;
    using EID = lEID;

public:
    // For maximal clique enumeration: all vertices regardless of coreness
    // Sort neighbour list in increasing order
    NeighbourCutOutXP( const ::GraphCSx & G, VID v,
		       const lVID * const core_order )
	: NeighbourCutOutXP( G, v, G.getIndex()[v+1] - G.getIndex()[v],
			     core_order ) { }
    NeighbourCutOutXP( const ::GraphCSx & G, VID v, VID deg,
		       const lVID * const core_order )
	: m_iset( &G.getEdges()[G.getIndex()[v]] ),
	  m_s2g( new lVID[deg] ),
	  m_n2s( new lVID[deg] ),
	  m_num_iset( deg ) {
	lVID n = G.numVertices();
	lEID m = G.numEdges();
	const lEID * const gindex = G.getIndex();
	const lVID * const gedges = G.getEdges();

	// Set of eligible neighbours
	lVID ns = deg;
	const lVID * const neighbours = &gedges[gindex[v]];
	// std::copy( &neighbours[0], &neighbours[ns], m_s2g );

	std::iota( &m_s2g[0], &m_s2g[ns], 0 );

	// Sort by increasing core_order
	std::sort( &m_s2g[0], &m_s2g[ns],
		   [&]( lVID u, lVID v ) {
		       return core_order[neighbours[u]]
			   < core_order[neighbours[v]];
		   } );
	// Invert permutation into n2s and create mapping for m_s2g
	for( lVID su=0; su < ns; ++su ) {
	    lVID x = m_s2g[su];
	    m_s2g[su] = neighbours[x]; // create mapping
	    m_n2s[x] = su; // invert permutation
	}

	// Determine start position, i.e., vertices less than start_pos
	// are in X by default
	lVID * sp2_pos = std::upper_bound(
	    &m_s2g[0], &m_s2g[ns], v,
	    [&]( lVID a, lVID b ) {
		return core_order[a] < core_order[b];
	    } );
	m_start_pos = sp2_pos - &m_s2g[0];
    }

    ~NeighbourCutOutXP() {
	if( m_s2g )
	    delete[] m_s2g;
	if( m_n2s )
	    delete[] m_n2s;
    }

    lVID get_num_vertices() const { return m_num_iset; }
    const lVID * get_vertices() const { return m_iset; }

    lVID get_start_pos() const { return m_start_pos; }
    const lVID * get_s2g() const { return m_s2g; }
    const lVID * get_n2s() const { return m_n2s; }

private:
    const VID * m_iset;
    VID * m_s2g;
    VID * m_n2s;
    VID m_num_iset;
    VID m_start_pos;
};

} // namespace graph

} // namespace graptor

#endif // GRAPHGRIND_GRAPH_SIMPLE_CUTOUT_H
