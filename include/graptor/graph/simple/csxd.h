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
 * The purpose of this class is to support algorithms where vertices are
 * repeatedly removed from the graph, requiring an update on statistics such
 * as degree and vertex and edge counts. We call removed vertices inactive,
 * those still present as active. Vertices may become active again
 * but only in the reverse order from which they were made inactive.
 *
 * The class supports only undirected graphs. The graph is represented
 * immutably through a CSR/CSC representation of the full adjacency matrix
 * (both upper and lower triangular parts). Self-edges are assumed to be absent.
 * Additional data is maintained to track which vertices are present in the
 * graph: a depth parameter is associated to each vertex to track presence
 * in the graph; the degree of each vertex tracking present neighbours
 * is stored (the CSR/CSC maintains the original degree of the vertex).
 *
 * Invariants:
 *  + a vertex v has been removed from the graph if m_depth[v] <= m_cur_depth
 *  + m_degree[v] tracks the number of active neighbours of v
 *  + m_degree[v] == 0 iff m_depth[v] <= m_cur_depth
 *    (vertices must be made inactive when their last incident edge is made
 *     inactive)
 *  + m_n_remain tracks number of active vertices
 *  + m_n_remain equals number of vertices with m_degree[v] > 0
 *  + m_n_remain equals number of vertices with m_depth[v] > m_cur_depth
 *  + 0 <= m_n_remain and m_n_remain <= m_n
 *  + m_m_remain equals number of edges between active vertices
 *  + 0 <= m_m_remain and m_m_remain <= m_m
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
	      m_m_remain( G.m_m_remain ),
	      m_cur_depth( G.m_cur_depth ),
	      m_n_remain( G.m_n_remain ) {
	    m_degree.insert( m_degree.end(), G.dbegin(), G.dend() );
	}

	lVID get_degree( lVID v ) const { return m_degree[v]; }

	lEID get_num_edges() const { return m_m_remain; }

	lVID get_cur_depth() const { return m_cur_depth; }

	lVID get_num_remaining_vertices() const { return m_n_remain; }

    private:
	std::vector<lVID> m_degree;
	lEID m_m_remain;
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
	m_m_remain( m ),
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
    lEID numEdges() const { return m_m_remain; }

    // readable interface
    lVID get_num_vertices() const { return m_n; }
    lEID get_num_edges() const { return m_m_remain; }
    lVID get_num_remaining_vertices() const { return m_n_remain; }
    lEID get_num_remaining_edges() const { return m_m_remain; }

    lVID get_cur_depth() const { return m_cur_depth; }
    lVID get_depth( lVID v ) const { return m_depth[v]; }
    lVID get_degree( lVID v ) const { return m_index[v+1] - m_index[v]; }
    lVID get_remaining_degree( lVID v ) const { return m_degree[v]; }

    lEID * getIndex() { return m_index.get(); }
    lEID * const getIndex() const { return m_index.get(); }
    lVID * getEdges() { return m_edges.get(); }
    lVID * const getEdges() const { return m_edges.get(); }

    /*! Get access to the degree array.
     * Warning: the degree array reflects the remaining degree and may be
     * inconsistent with the differences between successive values in m_index.
     */
    lVID * getDegree() { return m_degree.get(); }
    lVID * const getDegree() const { return m_degree.get(); }
    lVID * getDepth() { return m_depth.get(); }
    lVID * const getDepth() const { return m_depth.get(); }

    lVID getDegree( lVID v ) const { return get_degree( v ); }

    // Returns neighbours without considering depth-based filtering
    const lVID * get_neighbours( lVID v ) const {
	return &m_edges[m_index[v]];
    }

    auto get_neighbours_set( lVID v ) const {
	return graptor::make_array_slice(
	    &m_edges[m_index[v]], &m_edges[m_index[v+1]] );
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
	    lVID deg = get_remaining_degree( v );
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
	return std::make_pair( v, get_remaining_degree( v ) );
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
	
	// TODO: checkpoint only inactive vertices and reconstruct as single vertex
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
	m_m_remain = chkpt.get_num_edges();

	assert( m_n_remain <= m_n );
#if 0
	lEID mm = count_edges();
	assert( m_m_remain == mm );
#endif
    }
    void restore_checkpoint( const single_vertex_checkpoint_type & chkpt ) {
	restore_vertex( chkpt.get_vertex() );
	--m_cur_depth;
    }

    template<typename Fn>
    [[deprecated]]
    void disable_incident_edges( Fn && to_remove ) {
	assert( 0 && "NYI" );
    }

    // TODO: make sparse representation of checkpointed data to save
    //       space and possibly computation time (if sparsity of m_depth
    //       is apparent)
    template<typename Iterator>
    checkpoint_type disable_incident_edges( Iterator I, Iterator E ) {
	auto cp = checkpoint();
	++m_cur_depth;
	for( ; I != E; ++I )
	    disable_incident_edges_per_vertex( *I );

#if 0
	// Debugging code
	lVID r = 0, d = 0;
	for( lVID v=0; v < m_n; ++v ) {
	    if( m_depth[v] == m_cur_depth )
		++d;
	    
	    if( m_degree[v] > 0 )
		++r;
	}
	if( r != m_n_remain )
	    std::cout << "r=" << r << " m_n_remain=" << m_n_remain << "\n";
	assert( r == m_n_remain );
#endif

	assert( m_n_remain <= m_n );
	
	return cp;
    }

    single_vertex_checkpoint_type
    disable_incident_edges_for( lVID v ) {
	++m_cur_depth;
	disable_incident_edges_per_vertex( v );

#if 0
	// Debugging code
	lVID r = 0, d = 0;
	for( lVID v=0; v < m_n; ++v ) {
	    assert( ( m_depth[v] != initial_depth ) == ( m_degree[v] == 0 ) );
	    
	    if( m_degree[v] > 0 )
		++r;
	}
	if( r != m_n_remain )
	    std::cout << "r=" << r << " m_n_remain=" << m_n_remain << "\n";
	assert( r == m_n_remain );
#endif

	assert( m_n_remain <= m_n );
	
	return single_vertex_checkpoint_type( v );
    }

private:
    void disable_incident_edges_per_vertex( lVID v ) {
	assert( ( m_depth[v] != initial_depth ) == ( m_degree[v] == 0 ) );

	if( m_depth[v] <= m_cur_depth )
	    return;

	m_depth[v] = m_cur_depth;

	lVID rv = 0;
	lVID re = 0;

	lEID se = m_index[v];
	lEID ee = m_index[v+1];
	for( lEID e=se; e != ee; ++e ) {
	    lVID u = m_edges[e];

#if 0
	    assert( ( m_depth[u] != initial_depth ) == ( m_degree[u] == 0 ) );
	    assert( m_depth[u] <= m_cur_depth || m_depth[u] == initial_depth ); 
#endif

	    if( m_depth[u] > m_cur_depth ) {
		if( --m_degree[u] == 0 ) {
		    m_depth[u] = m_cur_depth;
		    ++rv;
		}
		++re;
	    }
	}

	assert( m_degree[v] == re );

	m_m_remain -= 2 * (lEID)re;
	m_n_remain -= rv + 1; // +1 for vertex v
	m_degree[v] = 0;

#if 0
	lEID mm = count_edges();
	assert( m_m_remain == mm );
#endif
    }

    void restore_vertex( lVID v ) {
	assert( m_depth[v] == m_cur_depth );

	m_depth[v] = initial_depth;

	lVID rv = 0;
	lVID re = 0;
	assert( m_degree[v] == 0 );

	lEID se = m_index[v];
	lEID ee = m_index[v+1];
	for( lEID e=se; e != ee; ++e ) {
	    lVID u = m_edges[e];
	    // The test >= is such that we consider all vertices at the
	    // current depth (which is restored / popped from checkpoint stack)
	    // to be / become active. This method is specific to restore a
	    // single-vertex checkpoint, hence all vertices at depth m_cur_depth
	    // must be neighbours of v.
	    if( m_depth[u] >= m_cur_depth ) {
#if 0
		assert( ( m_depth[u] != initial_depth ) == ( m_degree[u] == 0 ) );
#endif
		if( m_degree[u] == 0 ) {
		    m_depth[u] = initial_depth;
		    ++rv;
		}
		++m_degree[u];
		++re;
	    }
	}

	m_degree[v] = re; // total number of active neihbours
	m_m_remain += 2 * (lEID)re;
	m_n_remain += rv + 1; // +1 for vertex v

#if 0
	lEID mm = count_edges();
	assert( m_m_remain == mm );
#endif
    }

    /*! Count the number of remaining edges in the graph.
     *
     * This method is provided for debugging purposes. It also tests a
     * number of invariants that need to hold.
     *
     * @return The number of remaining edges in the graph.
     */
    lEID count_edges() const {
	lEID m = 0;
	lEID e = 0;
	for( lVID v=0; v < m_n; ++v ) {
	    lEID ee = m_index[v+1];
	    assert( ( m_depth[v] != initial_depth ) == ( m_degree[v] == 0 ) );
	    assert( ( m_depth[v] > m_cur_depth ) == ( m_degree[v] != 0 ) );
	    assert( m_depth[v] <= m_cur_depth || m_depth[v] == initial_depth );
	    lVID d = 0;
	    if( m_depth[v] > m_cur_depth ) {
		for( ; e < ee; ++e ) {
		    lVID u = m_edges[e];
		    if( m_depth[u] > m_cur_depth ) {
			++m;
			++d;
		    }
		}
	    } else {
		e = ee;
	    }
	    assert( m_degree[v] == d );
	}
	return m;
    }

private:
    const lVID m_n;		//!< Number of vertices in complete graph
    lVID m_n_remain;		//!< Number of remaining vertices in graph
    const lEID m_m;		//!< Number of edges in complete graph
    lEID m_m_remain;		//!< Number of remaining edges in graph
    lVID m_cur_depth;		//!< Current depth of stack of inactive vertices
    mm::buffer<lEID> m_index;	//!< Index parameter of CSx representation
    mm::buffer<lVID> m_edges;	//!< Neighbour list of CSx representation
    mm::buffer<lVID> m_depth;	//!< Depth at which vertex deactivated
    mm::buffer<lVID> m_degree;	//!< Remaining degree of vertex
};

} // namespace graph

} // namespace graptor

#endif // GRAPHGRIND_GRAPH_SIMPLE_CSXD_H

