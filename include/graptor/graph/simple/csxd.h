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

// TODO: redefine as a wrapper around another graph representation.
//       the other representation may provide hashing also, and we could
//       adapt to check depth upon hash access. This may accelerate creating
//       a dense cutout

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
	conditional_iterator<range_iterator<lVID>,depth_check>;
    using edge_iterator =
	conditional_iterator<const lVID*,depth_check>; // TODO: all edges
    using neighbour_iterator =
	conditional_iterator<const lVID*,depth_check>;
    // using const_neighbour_iterator = const lVID *;

    static constexpr lVID initial_depth = std::numeric_limits<lVID>::max();

    class checkpoint_type {
    public:
	checkpoint_type( const GraphCSxDepth & G )
	    : m_degree(),
	      m_cur_m( G.m_cur_m ),
	      m_cur_depth( G.m_cur_depth ),
	      m_n_remain( G.m_n_remain ) {
	    m_degree.insert( m_degree.end(), G.dbegin(), G.dend() );
	}

	lVID get_degree( lVID v ) const { return m_degree[v]; }

	lEID get_num_edges() const { return m_cur_m; }

	lVID get_cur_depth() const { return m_cur_depth; }

	lVID get_num_remaining_vertices() const { return m_n_remain; }

    private:
	std::vector<lVID> m_degree;
	lEID m_cur_m;
	lVID m_cur_depth;
	lVID m_n_remain;
    };

    class single_vertex_checkpoint_type {
    public:
	single_vertex_checkpoint_type( lVID v ) : m_vid( v ) { }

	lVID get_vertex() const { return m_vid; }

    private:
	lVID m_vid;
    };

public:
    GraphCSxDepth() : m_n( 0 ), m_m( 0 ) { }
    GraphCSxDepth( lVID n, lEID m, numa_allocation && alloc ) :
	m_n( n ),
	m_n_remain( n ),
	m_m( m ),
	m_cur_m( m ),
	m_cur_depth( 0 ),
	m_index( n+1, alloc ),
	m_depth( n, alloc ),
	m_degree( n, alloc ),
	m_edges( m, alloc ) { }
    GraphCSxDepth( lVID n, lEID m )
	: GraphCSxDepth( n, m, numa_allocation_interleaved() ) { }
    GraphCSxDepth( GraphCSxDepth && ) = delete;
    GraphCSxDepth( const GraphCSxDepth & ) = delete;

    ~GraphCSxDepth() {
	m_index.del();
	m_depth.del();
	m_degree.del();
	m_edges.del();
    }

    void complete_init() {
	m_n_remain = 0;
	for( lVID v=0; v < m_n; ++v )
	    if( m_degree[v] > 0 )
		++m_n_remain;
    }

    // capitalised interface
    lVID numVertices() const { return m_n; }
    lEID numEdges() const { return m_cur_m; }

    // readable interface
    lVID get_num_vertices() const { return m_n; }
    lEID get_num_edges() const { return m_cur_m; }
    lVID get_num_remaining_vertices() const { return m_n_remain; }

    lVID get_cur_depth() const { return m_cur_depth; }
    lVID get_depth( lVID v ) const { return m_depth[v]; }
    lVID get_degree( lVID v ) const { return m_degree[v]; }

    lEID * getIndex() { return m_index.get(); }
    lEID * const getIndex() const { return m_index.get(); }
    lVID * getEdges() { return m_edges.get(); }
    lVID * const getEdges() const { return m_edges.get(); }
    lVID * getDegree() { return m_degree.get(); }
    lVID * const getDegree() const { return m_degree.get(); }
    lVID * getDepth() { return m_depth.get(); }
    lVID * const getDepth() const { return m_depth.get(); }

    lVID getDegree( lVID v ) const { return m_degree[v]; }

    // Returns neighbours without considering depth-based filtering
    const lVID * get_neighbours( lVID v ) const {
	return &m_edges[m_index[v]];
    }

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

    neighbour_iterator nbegin( lVID v ) {
	return neighbour_iterator( &m_edges[m_index[v]],
				   &m_edges[m_index[v+1]],
				   depth_check( m_depth, m_cur_depth ) );
    }
    neighbour_iterator nbegin( lVID v ) const {
	return neighbour_iterator( &m_edges[m_index[v]],
				   &m_edges[m_index[v+1]],
				   depth_check( m_depth, m_cur_depth ) );
    }
    neighbour_iterator nend( lVID v ) {
	return neighbour_iterator( &m_edges[m_index[v+1]],
				   &m_edges[m_index[v+1]],
				   depth_check( m_depth, m_cur_depth ) );
    }
    neighbour_iterator nend( lVID v ) const {
	return neighbour_iterator( &m_edges[m_index[v+1]],
				   &m_edges[m_index[v+1]],
				   depth_check( m_depth, m_cur_depth ) );
    }

    auto dbegin() const { return &m_degree[0]; }
    auto dend() const { return &m_degree[m_n]; }

    lVID min_degree_vertex() const {
	return min_degree().first;
    }
    std::pair<lVID,lVID> min_degree() const {
	lVID min_deg = m_n;
	lVID min_v = m_n;
	for( lVID v=0; v < m_n; ++v ) {
	    lVID deg = getDegree( v );
	    if( deg < min_deg && deg > 0 ) {
		min_deg = deg;
		min_v = v;
		if( min_deg == 1 )
		    break;
	    }
	}
	return std::make_pair( min_v, min_deg );
    }

    lVID max_degree_vertex() const {
	return std::max_element( dbegin(), dend() ) - dbegin();
    }
    std::pair<lVID,lVID> max_degree() const {
	lVID v = max_degree_vertex();
	return std::make_pair( v, getDegree( v ) );
    }

    /*! Sort neighbour list of \p v. This is a no-op.
     *
     * \param v Vertex whose neighbour lists should be placed in sort order.
     */
    void sort_neighbours( lVID v ) { }

#if 0
    void sort_neighbour_lists() {
	// TODO: add boolean flag to check whether already sorted or not
	for( lVID v=0; v < m_n; ++v )
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
	assert( m_cur_depth == chkpt.get_cur_depth()+1 );
	
	// TODO: checkpoint only removed vertices and reconstruct as single vertex
	lVID r = 0;
	for( lVID v=0; v < m_n; ++v ) {
	    if( m_depth[v] == m_cur_depth )
		m_depth[v] = initial_depth;

	    m_degree[v] = chkpt.get_degree( v );
	    if( m_degree[v] > 0 )
		++r;
	}

	m_n_remain = chkpt.get_num_remaining_vertices();

	if( r != m_n_remain )
	    std::cout << "r=" << r << " m_n_remain=" << m_n_remain << "\n";
	assert( r == m_n_remain );

	m_cur_depth = chkpt.get_cur_depth();
	m_cur_m = chkpt.get_num_edges();

	assert( m_n_remain <= m_n );
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
	lVID dd = std::distance( I, E );
	for( ; I != E; ++I )
	    disable_incident_edges_per_vertex( *I );

	lVID r = 0, d = 0;
	for( lVID v=0; v < m_n; ++v ) {
	    if( m_depth[v] == m_cur_depth )
		++d;
	    
	    if( m_degree[v] > 0 )
		++r;
	}
	if( r != m_n_remain )
	    std::cout << "r=" << r << " m_n_remain=" << m_n_remain << "\n";
	if( dd != d )
	    std::cout << "dd=" << dd << " d=" << d << "\n";
	assert( r == m_n_remain );
	assert( dd == d );

	assert( m_n_remain <= m_n );
	
	return cp;
    }

    single_vertex_checkpoint_type
    disable_incident_edges_for( lVID v ) {
	++m_cur_depth;
	disable_incident_edges_per_vertex( v );

	lVID dd = 1;
	lVID r = 0, d = 0;
	for( lVID v=0; v < m_n; ++v ) {
	    if( m_depth[v] == m_cur_depth )
		++d;
	    
	    if( m_degree[v] > 0 )
		++r;
	}
	if( r != m_n_remain )
	    std::cout << "r=" << r << " m_n_remain=" << m_n_remain << "\n";
	if( dd != d )
	    std::cout << "dd=" << dd << " d=" << d << "\n";
	assert( r == m_n_remain );
	assert( dd == d );

	assert( m_n_remain <= m_n );
	
	return single_vertex_checkpoint_type( v );
    }

private:
    void disable_incident_edges_per_vertex( lVID v ) {
	if( m_depth[v] <= m_cur_depth )
	    return;

	lVID r = 0;
	if( m_degree[v] > 0 )
	    ++r;

	m_depth[v] = m_cur_depth;
	// lVID rm = 0;

	lEID se = m_index[v];
	lEID ee = m_index[v+1];
	for( lEID e=se; e != ee; ++e ) {
	    lVID u = m_edges[e];
	    if( m_depth[u] > m_cur_depth ) {
		// assert( m_degree[u] > 0 );
		if( --m_degree[u] == 0 )
		    ++r;
	    }
	}

	// assert( m_degree[v] == rm );
	m_cur_m -= 2 * m_degree[v];
	m_n_remain -= r;
	m_degree[v] = 0;
    }

    void restore_vertex( lVID v ) {
	assert( m_depth[v] == m_cur_depth );

	m_depth[v] = initial_depth;

	lVID r = 0;
	if( m_degree[v] == 0 )
	    ++r;

	lEID se = m_index[v];
	lEID ee = m_index[v+1];
	lVID cnt = 0;
	for( lEID e=se; e != ee; ++e ) {
	    lVID u = m_edges[e];
	    if( m_depth[u] >= m_cur_depth ) {
		if( m_degree[u]++ == 0 )
		    ++r;
		++cnt;
	    }
	}

	m_cur_m += 2 * cnt;
	m_degree[v] = cnt;
	m_n_remain += r;
    }

private:
    const lVID m_n;
    lVID m_n_remain;
    const lEID m_m;
    lEID m_cur_m;
    lVID m_cur_depth;
    mm::buffer<lEID> m_index;
    mm::buffer<lVID> m_depth;
    mm::buffer<lVID> m_degree;
    mm::buffer<lVID> m_edges;
};

} // namespace graph

} // namespace graptor

#endif // GRAPHGRIND_GRAPH_SIMPLE_CSXD_H

