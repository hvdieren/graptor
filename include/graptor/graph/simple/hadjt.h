// -*- C++ -*-
#ifndef GRAPHGRIND_GRAPH_SIMPLE_HADJT_H
#define GRAPHGRIND_GRAPH_SIMPLE_HADJT_H

#include "graptor/itraits.h"
#include "graptor/utils.h"
#include "graptor/mm.h"
#include "graptor/mm/mm.h"

#include "graptor/container/range_iterator.h"
#include "graptor/container/generic_edge_iterator.h"
#include "graptor/container/hash_table.h"
#include "graptor/container/difference_iterator.h"

namespace graptor {

namespace graph {

template<typename lVID, typename lEID, typename lHash = std::hash<lVID>>
class GraphHAdjTable {
public:
    using VID = lVID;
    using EID = lEID;
    using Hash = lHash;

    using vertex_iterator = range_iterator<VID>;
    using edge_iterator = generic_edge_iterator<VID,EID>;

    static constexpr bool has_dual_rep = false;

public:
    explicit GraphHAdjTable( VID n ) :
	m_n( n ),
	m_m( 0 ),
	m_hashes( new hash_table<VID,Hash>[m_n]() ) {
	// hashes initialised?
    }
    GraphHAdjTable( const ::GraphCSx & G ) :
	m_n( G.numVertices() ),
	m_m( G.numEdges() ),
	m_hashes( new hash_table<VID,Hash>[m_n]() ) {
	const EID * const index = G.getIndex();
	const VID * const edges = G.getEdges();

	parallel_loop( (VID)0, m_n, [&]( VID v ) {
	    auto & adj = get_adjacency( v );
	    EID ee = index[v+1];
	    for( EID e=index[v]; e != ee; ++e ) {
		VID u = edges[e];
		adj.insert( u );
	    }
	} );
    }
    GraphHAdjTable( const GraphHAdjTable & ) = delete;

    ~GraphHAdjTable() {
	// for( VID v=0; v < m_n; ++v )
	// delete m_hashes[v];
	delete[] m_hashes;
    }

    VID numVertices() const { return m_n; }
    EID numEdges() const { return m_m; }
    float density() const {
	return float(m_m) / ( float(m_n) * float(m_n) );
    }

    void sum_up_edges() {
	EID m = 0;
	for( VID v=0; v < m_n; ++v )
	    m += get_adjacency( v ).size();
	m_m = m;
    }

    VID getDegree( VID v ) { return m_hashes[v].size(); }
    hash_table<VID,Hash> & get_adjacency( VID v ) { return m_hashes[v]; }
    const hash_table<VID,Hash> & get_adjacency( VID v ) const {
	return m_hashes[v];
    }

    vertex_iterator vbegin() { return vertex_iterator( 0 ); }
    vertex_iterator vbegin() const { return vertex_iterator( 0 ); }
    vertex_iterator vend() { return vertex_iterator( numVertices() ); }
    vertex_iterator vend() const { return vertex_iterator( numVertices() ); }

/*
    edge_iterator ebegin() {
	return edge_iterator( 0, 0, m_index.get(), m_edges.get() );
    }
    edge_iterator ebegin() const {
	return edge_iterator( 0, 0, m_index.get(), m_edges.get() );
    }
    edge_iterator eend() {
	return edge_iterator( m_n, m_m, m_index.get(), m_edges.get() );
    }
    edge_iterator eend() const {
	return edge_iterator( m_n, m_m, m_index.get(), m_edges.get() );
    }
*/

private:
    VID m_n;
    EID m_m;
    hash_table<VID,Hash> * m_hashes;
};

} // namespace graph

} // namespace graptor

#endif // GRAPHGRIND_GRAPH_SIMPLE_HADJT_H
