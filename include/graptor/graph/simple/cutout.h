// -*- C++ -*-
#ifndef GRAPHGRIND_GRAPH_SIMPLE_CUTOUT_H
#define GRAPHGRIND_GRAPH_SIMPLE_CUTOUT_H

#include <type_traits>
#include <algorithm>

#include "graptor/graph/GraphCSx.h"
#include "graptor/container/array_slice.h"

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

template<typename lVID, typename lEID>
class NeighbourCutOutDegeneracyOrder {
public:
    using VID = lVID;
    using EID = lEID;

public:
    // For maximal clique enumeration: all vertices regardless of coreness
    // Sort neighbour list in increasing order
    NeighbourCutOutDegeneracyOrder( const ::GraphCSx & G, VID v )
	: NeighbourCutOutDegeneracyOrder(
	    G, v, G.getIndex()[v+1] - G.getIndex()[v] ) { }
    NeighbourCutOutDegeneracyOrder( const ::GraphCSx & G, VID v, VID deg )
	: m_iset( &G.getEdges()[G.getIndex()[v]] ),
	  m_num_iset( deg ) {
	const lVID * pos = std::lower_bound( m_iset, m_iset+deg, v );
	m_start_pos = pos - m_iset;
    }

    lVID get_num_vertices() const { return m_num_iset; }
    const lVID * get_vertices() const { return m_iset; }

    lVID get_start_pos() const { return m_start_pos; }

private:
    const VID * const m_iset;
    VID m_num_iset;
    VID m_start_pos;
};

template<typename lVID, typename lEID>
class NeighbourCutOutDegeneracyOrderFiltered {
public:
    // For maximal clique enumeration: all vertices regardless of coreness
    // Sort neighbour list in increasing order
    template<typename FilterFn>
    NeighbourCutOutDegeneracyOrderFiltered(
	const ::GraphCSx & G, lVID v, FilterFn && fn )
	: NeighbourCutOutDegeneracyOrderFiltered(
	    G, v, G.getIndex()[v+1] - G.getIndex()[v],
	    std::forward<FilterFn>( fn ) ) { }
    template<typename GraphTy, typename FilterFn>
    NeighbourCutOutDegeneracyOrderFiltered(
	const GraphTy & G, lVID v, lVID deg,
	FilterFn && fn )
	: m_iset( new lVID[deg] ) {
	const lVID * const ngh = G.get_neighbours( v );
	// Skip left neighbourhood
	const lVID * pos = std::upper_bound( ngh, ngh+deg, v );
	// Filter remaining vertices
	lVID * end = std::copy_if( pos, ngh+deg, m_iset,
				   std::forward<FilterFn>( fn ) );
	m_num_iset = end - m_iset;
    }

    template<typename FilterFn>
    NeighbourCutOutDegeneracyOrderFiltered(
	const lVID * set, lVID num,
	FilterFn && fn )
	: m_iset( new lVID[num] ) {
	// Filter vertices
	lVID * end = std::copy_if( set, set+num, m_iset,
				   std::forward<FilterFn>( fn ) );
	m_num_iset = end - m_iset;
    }
    template<typename SetType, typename FilterFn>
    NeighbourCutOutDegeneracyOrderFiltered(
	SetType && set, FilterFn && fn )
	: m_iset( new lVID[set.size()] ) {
	// Filter vertices
	lVID * end = std::copy_if( set.begin(), set.end(), m_iset,
				   std::forward<FilterFn>( fn ) );
	m_num_iset = end - m_iset;
    }

    ~NeighbourCutOutDegeneracyOrderFiltered() {
	delete[] m_iset;
    }

    template<typename FilterFn>
    void filter( FilterFn && fn, lVID min_size = 0 ) {
	// Remove vertices where fn(v) is false. Make sure that at each step,
	// get_vertices() and get_num_vertices() return correct values.
	// Working from the start considers low-degeneracy vertices first, which
	// are more likely to be pruned away, allowing higher-degeneracy
	// vertices to be pruned too.
	// std::make_signed_t<lVID> j = m_num_iset - 1;
	// while( j >= 0 ) {
	lVID j = 0;
	while( j < m_num_iset ) {
	    if( !fn( m_iset[j] ) ) {
		std::copy( &m_iset[j+1], &m_iset[m_num_iset], &m_iset[j] );
		--m_num_iset;
		// If the set becomes too small, then we are no longer
		// interested.
		if( m_num_iset < min_size )
		    break;
	    }
	    // --j;
	    else
		++j;
	}
    }

    lVID get_num_vertices() const { return m_num_iset; }
    const lVID * get_vertices() const { return m_iset; }

    lVID at( lVID pos ) const { return m_iset[pos]; }

    array_slice<lVID,lVID> get_slice() const {
	return array_slice<lVID,lVID>( m_iset, m_num_iset );
    }

    const lVID * begin() const { return m_iset; }
    const lVID * end() const { return m_iset+m_num_iset; }

private:
    lVID * m_iset;
    lVID m_num_iset;
};


} // namespace graph

} // namespace graptor

#endif // GRAPHGRIND_GRAPH_SIMPLE_CUTOUT_H
