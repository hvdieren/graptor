// -*- C++ -*-
#ifndef GRAPHGRIND_GRAPH_SIMPLE_CSX_H
#define GRAPHGRIND_GRAPH_SIMPLE_CSX_H

#include "graptor/itraits.h"
#include "graptor/mm.h"
#include "graptor/mm/mm.h"

#include "graptor/container/range_iterator.h"
#include "graptor/container/generic_edge_iterator.h"
#include "graptor/container/difference_iterator.h"

namespace graptor {

namespace graph {

template<typename lVID, typename lEID>
class GraphCSx {
public:
    using VID = lVID;
    using EID = lEID;

    using vertex_iterator = range_iterator<VID>;
    using edge_iterator = generic_edge_iterator<VID,EID>;

public:
    // GraphCSx() = delete;
    GraphCSx() { }
    GraphCSx( VID n, EID m, numa_allocation && alloc ) :
	m_n( n ),
	m_m( m ),
	m_index( n+1, alloc ),
	m_edges( m, alloc ) { }
    GraphCSx( VID n, EID m )
	: GraphCSx( n, m, numa_allocation_interleaved() ) { }

    ~GraphCSx() {
	m_index.del();
	m_edges.del();
    }

    VID numVertices() const { return m_n; }
    EID numEdges() const { return m_m; }

    EID * getIndex() { return m_index.get(); }
    EID * const getIndex() const { return m_index.get(); }
    VID * getEdges() { return m_edges.get(); }
    VID * const getEdges() const { return m_edges.get(); }

    vertex_iterator vbegin() { return vertex_iterator( 0 ); }
    vertex_iterator vbegin() const { return vertex_iterator( 0 ); }
    vertex_iterator vend() { return vertex_iterator( m_n ); }
    vertex_iterator vend() const { return vertex_iterator( m_n ); }

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

    VID max_degree() const {
	return find_maximum(
	    difference_iterator<EID *>( &m_index.get()[0] ),
	    difference_iterator<EID *>( &m_index.get()[m_n] ) );
    }

private:
    VID m_n;
    EID m_m;
    mm::buffer<EID> m_index;
    mm::buffer<VID> m_edges;
};

} // namespace graph

} // namespace graptor

#endif // GRAPHGRIND_GRAPH_SIMPLE_CSX_H
