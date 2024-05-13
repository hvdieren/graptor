// -*- C++ -*-
#ifndef GRAPHGRIND_GRAPH_SIMPLE_CSXD_H
#define GRAPHGRIND_GRAPH_SIMPLE_CSXD_H

#include <vector>

#include "graptor/itraits.h"
#include "graptor/mm.h"
#include "graptor/mm/mm.h"

#include "graptor/container/range_iterator.h"
// #include "graptor/container/edge_iterator.h"
#include "graptor/container/conditional_iterator.h"
#include "graptor/container/difference_iterator.h"

namespace graptor {

namespace graph {

/*! CSR/CSC style graph with vertices and edges filtered based on a depth
 *  parameter.
 *
 * \tparam lVID type of a vertex identifier
 * \tparam lEID type of an edge identifier
 */
template<typename lVID, typename lEID>
class GraphCSxDepth {
public:
    using VID = lVID;
    using EID = lEID;

    struct depth_check {
	depth_check( const mm::buffer<lVID> & depth, lVID cur_depth )
	    : m_depth( depth.get() ), m_cur_depth( cur_depth ) { }

	bool operator() ( lVID v ) const { return m_depth[v] > m_cur_depth; }

    private:
	const lVID * m_depth;
	lVID m_cur_depth;
    };

    using vertex_iterator =
	conditional_iterator<range_iterator<VID>,depth_check>;
    using edge_iterator =
	conditional_iterator<const VID*,depth_check>; // TODO: all edges
    using neighbour_iterator =
	conditional_iterator<const VID*,depth_check>;
    // using const_neighbour_iterator = const VID *;

    static constexpr VID initial_depth = std::numeric_limits<VID>::max();

    class checkpoint_type {
    public:
	checkpoint_type( const GraphCSxDepth & G )
	    : m_degree(),
	      m_cur_m( G.m_cur_m ),
	      m_cur_depth( G.m_cur_depth ) {
	    m_degree.insert( m_degree.end(), G.dbegin(), G.dend() );
	}

	VID get_degree( VID v ) const { return m_degree[v]; }

	EID get_num_edges() const { return m_cur_m; }

	EID get_cur_depth() const { return m_cur_depth; }

    private:
	std::vector<VID> m_degree;
	EID m_cur_m;
	VID m_cur_depth;
    };

    class single_vertex_checkpoint_type {
    public:
	single_vertex_checkpoint_type( VID v ) : m_vid( v ) { }

	VID get_vertex() const { return m_vid; }

    private:
	VID m_vid;
    };

public:
    GraphCSxDepth() : m_n( 0 ), m_m( 0 ) { }
    GraphCSxDepth( VID n, EID m, numa_allocation && alloc ) :
	m_n( n ),
	m_m( m ),
	m_cur_m( m ),
	m_cur_depth( 0 ),
	m_index( n+1, alloc ),
	m_depth( n, alloc ),
	m_degree( n, alloc ),
	m_edges( m, alloc ) { }
    GraphCSxDepth( VID n, EID m )
	: GraphCSxDepth( n, m, numa_allocation_interleaved() ) { }
    GraphCSxDepth( GraphCSxDepth && ) = delete;
    GraphCSxDepth( const GraphCSxDepth & ) = delete;

    ~GraphCSxDepth() {
	m_index.del();
	m_depth.del();
	m_degree.del();
	m_edges.del();
    }

    VID numVertices() const { return m_n; }
    EID numEdges() const { return m_cur_m; }

    EID * getIndex() { return m_index.get(); }
    EID * const getIndex() const { return m_index.get(); }
    VID * getEdges() { return m_edges.get(); }
    VID * const getEdges() const { return m_edges.get(); }
    VID * getDegree() { return m_degree.get(); }
    VID * const getDegree() const { return m_degree.get(); }
    VID * getDepth() { return m_depth.get(); }
    VID * const getDepth() const { return m_depth.get(); }

    EID getDegree( VID v ) const { return m_degree[v]; }

#if 0
    vertex_iterator vbegin() {
	return vertex_iterator( range_iterator( 0 ),
				range_iterator( m_n ),
				depth_check( m_depth, m_cur_depth ) );
    }
    vertex_iterator vbegin() const {
	return vertex_iterator( range_iterator( 0 ),
				range_iterator( m_n ),
				depth_check( m_depth, m_cur_depth ) );
    }
    vertex_iterator vend() { 
	return vertex_iterator( range_iterator( m_n ),
				range_iterator( m_n ),
				depth_check( m_depth, m_cur_depth ) );
    }
    vertex_iterator vend() const {
	return vertex_iterator( range_iterator( m_n ),
				range_iterator( m_n ),
				depth_check( m_depth, m_cur_depth ) );
    }

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
#endif

    neighbour_iterator nbegin( VID v ) {
	return neighbour_iterator( &m_edges[m_index[v]],
				   &m_edges[m_index[v+1]],
				   depth_check( m_depth, m_cur_depth ) );
    }
    neighbour_iterator nbegin( VID v ) const {
	return neighbour_iterator( &m_edges[m_index[v]],
				   &m_edges[m_index[v+1]],
				   depth_check( m_depth, m_cur_depth ) );
    }
    neighbour_iterator nend( VID v ) {
	return neighbour_iterator( &m_edges[m_index[v+1]],
				   &m_edges[m_index[v+1]],
				   depth_check( m_depth, m_cur_depth ) );
    }
    neighbour_iterator nend( VID v ) const {
	return neighbour_iterator( &m_edges[m_index[v+1]],
				   &m_edges[m_index[v+1]],
				   depth_check( m_depth, m_cur_depth ) );
    }

    auto dbegin() const { return &m_degree[0]; }
    auto dend() const { return &m_degree[m_n]; }

    VID max_degree_vertex() const {
	return std::max_element( dbegin(), dend() ) - dbegin();
    }
    std::pair<VID,VID> max_degree() const {
	VID v = max_degree_vertex();
	return std::make_pair( v, getDegree( v ) );
    }

    /*! Sort neighbour list of \p v. This is a no-op.
     *
     * \param v Vertex whose neighbour lists should be placed in sort order.
     */
    void sort_neighbours( VID v ) { }

#if 0
    void sort_neighbour_lists() {
	// TODO: add boolean flag to check whether already sorted or not
	for( VID v=0; v < m_n; ++v )
	    std::sort(
		&m_edges[m_begin_index[v]], &m_edges[m_end_index[v]] );
    }
#endif

    // Allows to undo the actions of disable_incident_edges, except
    // for the loss of sort order of the neighbour list
    checkpoint_type checkpoint() {
	return checkpoint_type( *this );
    }
    void restore_checkpoint( const checkpoint_type & chkpt ) {
	for( VID v=0; v < m_n; ++v ) {
	    if( m_depth[v] == m_cur_depth )
		m_depth[v] = initial_depth;
	    
	    m_degree[v] = chkpt.get_degree( v );
	}

	m_cur_depth = chkpt.get_cur_depth();
	m_cur_m = chkpt.get_num_edges();
    }
    void restore_checkpoint( const single_vertex_checkpoint_type & chkpt ) {
	restore_vertex( chkpt.get_vertex() );
	--m_cur_depth;
    }

    template<typename Fn>
    void disable_incident_edges( Fn && to_remove ) {
	assert( 0 && "NYI" );
    }

    template<typename Iterator>
    checkpoint_type disable_incident_edges( Iterator I, Iterator E ) {
	auto cp = checkpoint();
	++m_cur_depth;
	for( ; I != E; ++I )
	    disable_incident_edges_per_vertex( *I );
	return cp;
    }

    single_vertex_checkpoint_type
    disable_incident_edges_for( VID v ) {
	++m_cur_depth;
	disable_incident_edges_per_vertex( v );
	return single_vertex_checkpoint_type( v );
    }

private:
    void disable_incident_edges_per_vertex( VID v ) {
	if( m_depth[v] <= m_cur_depth )
	    return;

	m_depth[v] = m_cur_depth;
	// VID rm = 0;

	EID se = m_index[v];
	EID ee = m_index[v+1];
	for( EID e=se; e != ee; ++e ) {
	    VID u = m_edges[e];
	    if( m_depth[u] > m_cur_depth ) {
		// assert( m_degree[u] > 0 );
		--m_degree[u];
		// ++rm;
	    }
	}

	// assert( m_degree[v] == rm );
	m_cur_m -= 2 * m_degree[v];
	m_degree[v] = 0;
    }

    void restore_vertex( VID v ) {
	assert( m_depth[v] == m_cur_depth );

	m_depth[v] = initial_depth;

	EID se = m_index[v];
	EID ee = m_index[v+1];
	VID cnt = 0;
	for( EID e=se; e != ee; ++e ) {
	    VID u = m_edges[e];
	    if( m_depth[u] >= m_cur_depth ) {
		++m_degree[u];
		++cnt;
	    }
	}

	m_cur_m += 2 * cnt;
	m_degree[v] = cnt;
    }

private:
    const VID m_n;
    const EID m_m;
    EID m_cur_m;
    VID m_cur_depth;
    mm::buffer<EID> m_index;
    mm::buffer<VID> m_depth;
    mm::buffer<VID> m_degree;
    mm::buffer<VID> m_edges;
};

} // namespace graph

} // namespace graptor

#endif // GRAPHGRIND_GRAPH_SIMPLE_CSXD_H

