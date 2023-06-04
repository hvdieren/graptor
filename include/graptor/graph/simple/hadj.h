// -*- C++ -*-
#ifndef GRAPHGRIND_GRAPH_SIMPLE_HADJ_H
#define GRAPHGRIND_GRAPH_SIMPLE_HADJ_H

#include "graptor/itraits.h"
#include "graptor/utils.h"
#include "graptor/mm.h"
#include "graptor/mm/mm.h"

#include "graptor/graph/simple/range_iterator.h"
#include "graptor/graph/simple/generic_edge_iterator.h"
#include "graptor/graph/simple/hashed_set.h"
#include "graptor/graph/simple/utils.h"

namespace graptor {

namespace graph {

template<typename lVID, typename lEID>
class GraphHAdj {
public:
    using VID = lVID;
    using EID = lEID;

    using vertex_iterator = range_iterator<VID>;
    using edge_iterator = generic_edge_iterator<VID,EID>;

public:
    explicit GraphHAdj( const GraphCSx<VID,EID> & G ) :
	GraphHAdj( G, numa_allocation_interleaved() ) { }
    explicit GraphHAdj( const GraphCSx<VID,EID> & G,
			numa_allocation && alloc ) :
	m_G( G ),
	m_index( G.numVertices()+1, alloc ),
	m_hashes( get_hash_slots( G.getIndex(), G.numVertices() ), alloc ) {
	VID n = G.numVertices();
	EID h = 0;
	EID * index = m_index.get();
	VID * hashes = m_hashes.get();
	EID * gindex = getIndex();
	VID * gedges = getEdges();
	for( VID v=0; v < n; ++v ) {
	    VID deg = gindex[v+1] - gindex[v];
	    VID s = get_hash_slots( deg );
	    auto a = hashed_set<VID>( &hashes[h], deg, s );
	    a.clear(); // initialise to invalid element
	    a.insert( &gedges[gindex[v]], &gedges[gindex[v+1]] );
	    index[v] = h;
	    h += s;
	}
	index[n] = h;
    }
    GraphHAdj( const GraphHAdj & ) = delete;

    ~GraphHAdj() {
	m_index.del();
	m_hashes.del();
    }

    VID numVertices() const { return m_G.numVertices(); }
    EID numEdges() const { return m_G.numEdges(); }

    EID * getIndex() { return m_G.getIndex(); }
    EID * const getIndex() const { return m_G.getIndex(); }
    VID * getEdges() { return m_G.getEdges(); }
    VID * const getEdges() const { return m_G.getEdges(); }
    EID * getHashIndex() { return m_index.get(); }
    EID * const getHashIndex() const { return m_index.get(); }
    VID * getHashes() { return m_hashes.get(); }
    VID * const getHashes() const { return m_hashes.get(); }

    vertex_iterator vbegin() { return vertex_iterator( 0 ); }
    vertex_iterator vbegin() const { return vertex_iterator( 0 ); }
    vertex_iterator vend() { return vertex_iterator( numVertices() ); }
    vertex_iterator vend() const { return vertex_iterator( numVertices() ); }

    hashed_set<VID> get_adjacency( VID v ) const {
	VID deg = getIndex()[v+1] - getIndex()[v];
	VID h = getHashIndex()[v+1] - getHashIndex()[v];
	return hashed_set<VID>( &getHashes()[getHashIndex()[v]], deg, h );
    }


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
    static EID get_hash_slots( VID deg ) {
	return next_ipow2( deg+1 ) << 1;
    }
    static EID get_hash_slots( EID * index, VID n ) {
	EID h = 0;
	for( VID v=0; v < n; ++v )
	    h += get_hash_slots( index[v+1] - index[v] );
	return h;
    }

private:
    const GraphCSx<VID,EID> & m_G;
    mm::buffer<EID> m_index;
    mm::buffer<VID> m_hashes;
};

} // namespace graph

} // namespace graptor

#endif // GRAPHGRIND_GRAPH_SIMPLE_HADJ_H
