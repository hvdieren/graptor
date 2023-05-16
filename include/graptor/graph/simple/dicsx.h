// -*- C++ -*-
#ifndef GRAPHGRIND_GRAPH_SIMPLE_DICSX_H
#define GRAPHGRIND_GRAPH_SIMPLE_DICSX_H

#include <vector>

#include "graptor/itraits.h"
#include "graptor/mm.h"
#include "graptor/mm/mm.h"

#include "graptor/graph/simple/range_iterator.h"
#include "graptor/graph/simple/double_index_edge_iterator.h"
#include "graptor/graph/simple/utils.h"

namespace graptor {

namespace graph {

template<typename lVID, typename lEID>
class GraphDoubleIndexCSx {
public:
    using VID = lVID;
    using EID = lEID;

    using vertex_iterator = range_iterator<VID>;
    using edge_iterator = double_index_edge_iterator<VID,EID>;
    using neighbour_iterator = VID *;
    using const_neighbour_iterator = const VID *;

    class checkpoint_type {
    public:
	checkpoint_type( const GraphDoubleIndexCSx & G )
	    : m_old_degree( G.numVertices() ),
	      m_old_m( G.numEdges() ) {
	    std::copy( G.dbegin(), G.dend(), &m_old_degree[0] );
	}

	VID get_degree( VID v ) const {
	    return m_old_degree[v];
	}

	EID get_num_edges() const {
	    return m_old_m;
	}

    private:
	std::vector<VID> m_old_degree;
	EID m_old_m;
    };

public:
    GraphDoubleIndexCSx() { }
    GraphDoubleIndexCSx( VID n, EID m, numa_allocation && alloc ) :
	m_n( n ),
	m_m( m ),
	m_begin_index( n+1, alloc ), // n elements should be sufficient
	m_end_index( n+1, alloc ), // n elements should be sufficient
	m_edges( m, alloc ) { }
    GraphDoubleIndexCSx( VID n, EID m )
	: GraphDoubleIndexCSx( n, m, numa_allocation_interleaved() ) { }
    GraphDoubleIndexCSx( GraphDoubleIndexCSx && ) = delete;
    GraphDoubleIndexCSx( const GraphDoubleIndexCSx & ) = delete;

    ~GraphDoubleIndexCSx() {
	m_begin_index.del();
	m_end_index.del();
	m_edges.del();
    }

    VID numVertices() const { return m_n; }
    EID numEdges() const { return m_m; }

    EID * getBeginIndex() { return m_begin_index.get(); }
    EID * const getBeginIndex() const { return m_begin_index.get(); }
    EID * getEndIndex() { return m_end_index.get(); }
    EID * const getEndIndex() const { return m_end_index.get(); }
    VID * getEdges() { return m_edges.get(); }
    VID * const getEdges() const { return m_edges.get(); }

    EID getDegree( VID v ) const {
	return m_end_index[v] - m_begin_index[v];
    }

    vertex_iterator vbegin() { return vertex_iterator( 0 ); }
    vertex_iterator vbegin() const { return vertex_iterator( 0 ); }
    vertex_iterator vend() { return vertex_iterator( m_n ); }
    vertex_iterator vend() const { return vertex_iterator( m_n ); }

    edge_iterator ebegin() {
	return edge_iterator( 0, 0, m_begin_index.get(),
			      m_end_index.get(), m_edges.get() );
    }
    edge_iterator ebegin() const {
	return edge_iterator( 0, 0, m_begin_index.get(),
			      m_end_index.get(), m_edges.get() );
    }
    edge_iterator eend() {
	return edge_iterator( m_n, m_end_index[m_n], m_begin_index.get(),
			      m_end_index.get(), m_edges.get() );
    }
    edge_iterator eend() const {
	return edge_iterator( m_n, m_end_index[m_n], m_begin_index.get(),
			      m_end_index.get(), m_edges.get() );
    }

    neighbour_iterator nbegin( VID v ) {
	return &m_edges[m_begin_index[v]];
    }
    const_neighbour_iterator nbegin( VID v ) const {
	return &m_edges[m_begin_index[v]];
    }
    neighbour_iterator nend( VID v ) {
	return &m_edges[m_end_index[v]];
    }
    const_neighbour_iterator nend( VID v ) const {
	return &m_edges[m_end_index[v]];
    }

    auto dbegin() const {
	return pairwise_difference_iterator<EID *>(
	    &m_begin_index.get()[0], &m_end_index.get()[0] );
    }
    auto dend() const {
	return pairwise_difference_iterator<EID *>(
	    &m_begin_index.get()[m_n], &m_end_index.get()[m_n] );
    }

    VID max_degree_vertex() const {
	return std::max_element(
	    pairwise_difference_iterator<EID *>(
		&m_begin_index.get()[0], &m_end_index.get()[0] ),
	    pairwise_difference_iterator<EID *>(
		&m_begin_index.get()[m_n], &m_end_index.get()[m_n] ) )
	    - pairwise_difference_iterator<EID *>(
		&m_begin_index.get()[0], &m_end_index.get()[0] );
    }
    std::pair<VID,VID> max_degree() const {
	VID v = max_degree_vertex();
	return std::make_pair( v, getDegree( v ) );
    }

    void sort_neighbours( VID v ) {
	std::sort( nbegin( v ), nend( v ) );
    }

    void sort_neighbour_lists() {
	// TODO: add boolean flag to check whether already sorted or not
	for( VID v=0; v < m_n; ++v )
	    std::sort(
		&m_edges[m_begin_index[v]], &m_edges[m_end_index[v]] );
    }

    template<typename Iter>
    void erase_incident_edges( Iter start, Iter end ) {
	auto to_remove = [&]( VID v ) {
	    Iter pos = std::lower_bound( start, end, v );
	    return pos != end && *pos == v;
	};
	erase_incident_edges( to_remove );
    }

    template<typename Fn>
    void erase_incident_edges( Fn && to_remove ) {
	EID new_m = 0;
	for( VID v=0; v < m_n; ++v ) {
	    if( to_remove( v ) ) {
		m_end_index[v] = m_begin_index[v];
	    } else {
		EID se=m_begin_index[v];
		EID ee=m_end_index[v];
		VID * pos = &m_edges[se];
		for( EID e=se; e != ee; ++e ) {
		    VID u = m_edges[e];
		    if( !to_remove( u ) )
			*pos++ = u;
		}
		EID deg = pos - &m_edges[se]; 
		m_end_index[v] = se + deg;
		new_m += deg;
	    }
	}
	m_m = new_m;
	assert( m_m % 2 == 0 );
    }

    // Allows to undo the actions of disable_incident_edges, except
    // for the loss of sort order of the neighbour list
    checkpoint_type checkpoint() const {
	return checkpoint_type( *this );
    }
    void restore_checkpoint( const checkpoint_type & chkpt ) {
	for( VID v=0; v < m_n; ++v )
	    m_end_index[v] = m_begin_index[v] + chkpt.get_degree( v );
	m_m = chkpt.get_num_edges();
    }

    template<typename Iter>
    void disable_incident_edges( Iter start, Iter end ) {
	auto to_remove = [&]( VID v ) {
	    Iter pos = std::lower_bound( start, end, v );
	    return pos != end && *pos == v;
	};
	disable_incident_edges( to_remove );
    }

    template<typename Fn>
    void disable_incident_edges( Fn && to_remove ) {
	// Removes sort order, if any
	EID new_m = 0;
	for( VID v=0; v < m_n; ++v ) {
	    if( to_remove( v ) ) {
		m_end_index[v] = m_begin_index[v];
	    } else {
		EID se = m_begin_index[v];
		EID ee = m_end_index[v];
		for( EID e=se; e != ee; ) {
		    VID u = m_edges[e];
		    if( to_remove( u ) ) {
			std::swap( m_edges[e], m_edges[--ee] );
		    } else
			++e;
		}
		m_end_index[v] = ee;
		new_m += ee - se;
	    }
	}
	m_m = new_m;
	assert( m_m % 2 == 0 );
    }

private:
    VID m_n;
    EID m_m;
    mm::buffer<EID> m_begin_index;
    mm::buffer<EID> m_end_index;
    mm::buffer<VID> m_edges;
};

} // namespace graph

} // namespace graptor

#endif // GRAPHGRIND_GRAPH_SIMPLE_DICSX_H

