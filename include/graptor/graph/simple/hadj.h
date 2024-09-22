// -*- C++ -*-
#ifndef GRAPHGRIND_GRAPH_SIMPLE_HADJ_H
#define GRAPHGRIND_GRAPH_SIMPLE_HADJ_H

#include "graptor/itraits.h"
#include "graptor/utils.h"
#include "graptor/mm.h"
#include "graptor/mm/mm.h"

#include "graptor/container/hash_set.h"
#include "graptor/container/hash_set_hopscotch.h"
#include "graptor/container/hash_set_hopscotch_delta.h"
#include "graptor/container/hash_table.h"
#include "graptor/container/intersect.h"

#include "graptor/container/range_iterator.h"
#include "graptor/container/double_index_edge_iterator.h"
#include "graptor/container/difference_iterator.h"
#include "graptor/container/generic_edge_iterator.h"
#include "graptor/container/array_slice.h"
#include "graptor/container/dual_set.h"
#include "graptor/container/maybe_dual_set.h"

#ifndef LAZY_HASH_FILTER
#define LAZY_HASH_FILTER 0
#endif

namespace graptor {

namespace graph {

template<bool dual_rep, bool left_base, typename HashSet>
struct hash_pa_insert_iterator {
    hash_pa_insert_iterator(
	HashSet & table, VID * list, const VID * start )
	: m_table( table ), m_list( list ), m_start( start ) { }

    template<typename LSet, typename RSet>
    void swap( LSet && lset, RSet && rset ) { }

    size_t return_value() { return 0; } // whatever

    void push_back( const VID * lt, const VID * rt = nullptr ) {
	VID v = ( left_base ? lt : rt ) - m_start;
	m_table.insert( v );
	if constexpr ( dual_rep )
	    *m_list++ = v;
    }

    //! record function used in case of merge_scalar 
    template<bool rhs>
    bool record( const VID * l, const VID * r, bool ins ) {
	// rhs == true: l points into adjacency, r points into XP
	// rhs == false: l points into XP, r points into adjacency
	if( ins ) {
	    VID v = ( rhs ? r : l ) - m_start;
	    m_table.insert( v );
	    if constexpr ( dual_rep )
		*m_list++ = v;
	}

	return true;
    }

    //! record function used in case of scalar
    template<bool rhs>
    bool record( const VID * p, VID xlat, bool ins ) {
	if( ins ) {
	    if constexpr ( rhs ) {
		// RHS is XP dual set. xlat contains translated ID
		m_table.insert( xlat );
		if constexpr ( dual_rep )
		    *m_list++ = xlat;
	    } else {
		// RHS is adjacency info. xlat is 0/1 value indicating presence.
		// Need to translate using pointer arithmetic
		VID v = p - m_start;
		m_table.insert( v );
		if constexpr ( dual_rep )
		    *m_list++ = v;
	    }
	}
	return true;
    }

    bool terminated() const { return false; }

private:
    HashSet & m_table;
    VID * m_list;
    const VID * m_start;
};

template<typename lVID, typename lEID,
	 bool dual_rep,
	 bool preallocate,
	 bool lazyhash,
	 typename lHashSet>
class GraphHAdjPA {
public:
    using self_type = GraphHAdjPA<lVID,lEID,dual_rep,preallocate,lazyhash,lHashSet>;
    using VID = lVID;
    using EID = lEID;
    using hash_set_type = lHashSet;
    using Hash = typename hash_set_type::hash_type;
    using hash_table_type = graptor::hash_table<VID,VID,Hash>;
    using ngh_set_type = graptor::array_slice<VID,VID>;
    using dual_set_type = graptor::dual_set<ngh_set_type,hash_set_type>;

    using vertex_iterator = range_iterator<VID>;
    using edge_iterator = generic_edge_iterator<VID,EID>;
    using neighbour_iterator = VID *;
    using const_neighbour_iterator = const VID *;

    static constexpr bool has_dual_rep = dual_rep;
    static constexpr bool preallocate_storage = preallocate;
    static constexpr bool lazy_hashing = lazyhash;

    static_assert( !preallocate_storage || !lazy_hashing,
		   "Lazy hashing implemented only for non-preallocated "
		   "storage" );

public:
    explicit GraphHAdjPA( const ::GraphCSx & G,
			  numa_allocation && alloc,
			  size_t lazy_threshold = 16 ) :
	m_n( G.numVertices() ),
	m_index( G.numVertices()+1, alloc ),
	m_adjacency( G.numVertices(), alloc ) {

	VID n = m_n;
	const EID * const gindex = G.getIndex();
	const VID * const gedges = G.getEdges();

	EID * index = m_index.get();
	EID h = 0;
	parallel_loop( VID(0), n, [&]( VID v ) {
	    if constexpr ( preallocate_storage ) {
		EID deg = gindex[v+1] - gindex[v];
		index[v] = get_hash_slots( deg );
		if constexpr ( dual_rep )
		    index[v] += ( deg == 0 ? 0 : index[v] );
	    } else {
		if constexpr ( dual_rep )
		    index[v] = gindex[v];
		else
		    index[v] = 0;
	    }
	} );
	if constexpr ( preallocate_storage )
	    index[n] = sequence::plusScan( index, index, n );
	else
	    index[n] = gindex[n];

	new ( &m_storage ) mm::buffer<VID>( index[n], alloc );
	VID * hashes = m_storage.get();
	parallel_loop( VID(0), n, [&]( VID v ) {
	    EID deg = gindex[v+1] - gindex[v];
	    VID s = index[v+1] - index[v];
	    if constexpr ( preallocate_storage ) {
		VID t = dual_rep && deg != 0 ? s/2 : 0;
		if constexpr ( dual_rep )
		    if( deg != 0 )
			s /= 2;
		VID logs = s == 0 ? 0 : rt_ilog2( s );
		hash_set_type & a = m_adjacency[v];
		new ( &a ) hash_set_type( &hashes[index[v]+t], 0, logs );
		a.insert( &gedges[gindex[v]], &gedges[gindex[v+1]] );
		if constexpr ( graptor::is_hash_set_hopscotch_delta_v<hash_set_type> )
		    a.finalise();
	    } else {
		hash_set_type & a = m_adjacency[v];
		if( lazy_hashing && gindex[v+1] - gindex[v] < lazy_threshold ) {
		    new ( &a ) hash_set_type(); // size hint empty
		} else {
		    new ( &a ) hash_set_type( deg ); // size hint
		    a.insert( &gedges[gindex[v]], &gedges[gindex[v+1]] );
		    if constexpr ( graptor::is_hash_set_hopscotch_delta_v<hash_set_type> )
			a.finalise();
		    assert( a.size() == gindex[v+1] - gindex[v] );
		    assert( a.size() == index[v+1] - index[v] );
		}
	    }
	    if constexpr ( dual_rep ) {
		std::copy( &gedges[gindex[v]], &gedges[gindex[v+1]],
			   &hashes[index[v]] );
		assert( gindex[v] == gindex[v+1] || hashes[index[v]] != ~(VID)0 );
	    }
	} );

	// Discard if not needed
	if constexpr ( preallocate_storage || !has_dual_rep )
	    m_index.del();
    }
    // CSxGraphTy: some kind of graph storing neighbour list
    // HGraphTy: some kind of hashed graph
    template<typename CSxGraphTy, typename HGraphTy>
    explicit GraphHAdjPA( const CSxGraphTy & G,
			  const HGraphTy & H,
			  const VID * const XP,
			  VID ne,
			  VID ce,
			  numa_allocation && alloc ) :
	m_n( ce ),
	m_adjacency( ce, alloc ) {
	// Constructor taking cut-out and remapping vertex IDs

	assert( preallocate_storage );

	// TODO: allocate using StackLikeAllocator
	EID h = 0;
	for( VID su=0; su < ce; ++su ) {
	    VID u = XP[su];
	    VID deg = std::min( (VID)G.getDegree( u ), ce );
	    VID s = get_hash_slots( deg );
	    if constexpr ( has_dual_rep )
		h += 2 * s;
	    else
		h += s;
	}

	// Place XP in hash table for fast intersection
#if !ABLATION_HADJPA_DISABLE_XP_HASH
#if 0
	hash_table_type XP_hash( ce );
	for( VID i=0; i < ce; ++i )
	    XP_hash.insert( XP[i], i );
#else
	hash_set_type XP_hash( XP, XP+ce );
#endif
#endif

	new ( &m_storage ) mm::buffer<VID>( h, alloc );

	VID * hashes = m_storage.get();
	EID s_next = 0;
	for( VID su=0; su < ce; ++su ) {
	    VID u = XP[su];
	    hash_set_type & a = m_adjacency[su];

	    // Note: ce >> degree( u ). Use dual representation so that the
	    // main list for iteration can be swapped.
	    const VID * n = G.get_neighbours( u );
	    VID deg = G.getDegree( u );
	    if( ce > 2*deg && n != nullptr ) [[likely]] {
#if !ABLATION_HADJPA_DISABLE_XP_HASH
#if 0
		// Alternative: use hash table to translate vertex IDs
		VID * arr = &hashes[s_next];
		const VID * n_start = n;
		VID sdeg = std::min( deg, ce );
		VID logs = get_log_hash_slots( sdeg );
		VID s = 1 << logs;
		new ( &a ) hash_set_type( &hashes[s_next+s], 0, logs );

		for( const VID * p=n_start; p != n+deg; ++p ) {
		    VID v = *p;
		    VID sv;
		    if( XP_hash.contains( v, sv ) ) {
			if( sv >= ne || su >= ne ) {
			    a.insert( sv );
			    if constexpr ( has_dual_rep )
				*arr++ = sv;
			}
		    }
		}
		s_next += 2*s;
#else
		// Alternative: first place intersection in sequential storage,
		// using hash of XP, then translate and insert in adjacency a
		// Seems best-performing
		// Note that this leaves the neighbourhood list in a non-sorted
		// state: the order of neighbours if originally sorted is
		// different from the order of remapped vertices due to the
		// distinction between X and P: X neighbours get lower IDs than
		// P neighbours, but that may change the order.
		// As such, need to include the full range of the neighbour
		// list and starting at lower_bound(n,n+deg,XP[ne]) when
		// su >= ne is incorrect.
		assert( has_dual_rep && "otherwise need temporary space" );
		VID * arr = &hashes[s_next];
		VID * e = graptor::hash_scalar::intersect(
		    n, n+deg, XP_hash, arr );
		VID logs = get_log_hash_slots( e - arr );
		VID s = 1 << logs;
		new ( &a ) hash_set_type(
		    &hashes[has_dual_rep?(s_next+s):s_next], 0, logs );
		s_next += has_dual_rep ? 2*s : s;
		
		VID * j = arr;
		for( VID * i=arr; i != e; ++i ) {
		    const VID * pos = std::lower_bound( XP, XP+ne, *i );
		    if( *pos != *i )
			pos = std::lower_bound( XP+ne, XP+ce, *i );
		    VID sv = pos - XP;
		    if( sv >= ne || su >= ne ) {
			assert( sv != su );
			if constexpr ( has_dual_rep )
			    *j++ = sv;
			a.insert( sv );
		    }
		}
#endif
#else
#if ABLATION_HADJPA_DISABLE_XP_HASH == 1
		VID sdeg = std::min( deg, ce );
		VID logs = get_log_hash_slots( sdeg );
		VID s = 1 << logs;
		new ( &a ) hash_set_type(
		    &hashes[has_dual_rep ? (s_next+s) : s_next], 0, logs );
		VID * arr = has_dual_rep ? &hashes[s_next] : nullptr;
		s_next += has_dual_rep ? 2*s : s;

		// advance n to lower_bound of XP[ne] in n if su < ne,
		// i.e. when looking at an X vertex, ignore X neighbours of u
		const VID * n_start = n;
		if( su < ne )
		    n_start = std::lower_bound( n, n+deg, XP[ne] );

		if constexpr ( has_dual_rep ) {
		    // X-P edges: include edges linking to X
		    if( su >= ne ) { // su is a P vertex
			hash_pa_insert_iterator<dual_rep,false,hash_set_type>
			    out( a, arr, XP );
			graptor::merge_scalar::intersect<true>(
			    n, n+deg,	// X+P; shorter list
			    XP, XP+ne,	// X
			    out );
		    }

		    // P-P edges
		    // New iterator to count elements already included
		    hash_pa_insert_iterator<dual_rep,false,hash_set_type>
			out( a, arr+a.size(), XP );
		    graptor::merge_scalar::intersect<true>(
			n_start, n+deg, // P; shorter list
			XP+ne, XP+ce,   // P
			out );
		} else {
		    // X-P edges: include edges linking to X
		    if( su >= ne ) { // su is a P vertex
			hash_insert_iterator<hash_set_type> out( a, XP );
			graptor::merge_scalar::intersect<true>(
			    XP, XP+ne,	// X
			    n, n+deg,	// X+P; shorter list
			    out );
		    }

		    // P-P edges
		    // New iterator to count elements already included
		    hash_insert_iterator<hash_set_type> out( a, XP );
		    graptor::merge_scalar::intersect<true>(
			XP+ne, XP+ce,   // P
			n_start, n+deg, // P; shorter list
			out );
		}
#else
		// Same code as adjacency hashed
		VID sdeg = std::min( deg, ce );
		VID logs = get_log_hash_slots( sdeg );
		VID s = 1 << logs;
		new ( &a ) hash_set_type(
		    &hashes[has_dual_rep?(s_next+s):s_next], 0, logs );
		hash_pa_insert_iterator<dual_rep,true,hash_set_type>
		    out( a, &hashes[s_next], XP );
		graptor::hash_scalar::intersect<true>(
		    su < ne ? XP+ne : XP, XP+ce,
		    H.get_adjacency( u ), out );
		s_next += has_dual_rep ? 2*s : s;
#endif
#endif
	    } else {
		VID sdeg = std::min( deg, ce );
		VID logs = get_log_hash_slots( sdeg );
		VID s = 1 << logs;
		new ( &a ) hash_set_type(
		    &hashes[has_dual_rep?(s_next+s):s_next], 0, logs );
		hash_pa_insert_iterator<dual_rep,true,hash_set_type>
		    out( a, &hashes[s_next], XP );
		graptor::hash_scalar::intersect<true>(
		    su < ne ? XP+ne : XP, XP+ce,
		    H.get_adjacency( u ), out );
		s_next += has_dual_rep ? 2*s : s;
	    }
	}
    }
    // CSxGraphTy: some kind of graph storing neighbour list
    // HGraphTy: some kind of hashed graph
    // This constructor constructs a "PSet" only, i.e., does not distinguish
    // X from P vertices and avoid X-X edges.
    //
    // Note: we only use lazy hashing for the full graph
    template<typename CSxGraphTy, typename HGraphTy>
    explicit GraphHAdjPA( const CSxGraphTy & G,
			  const HGraphTy & H,
			  const VID * const XP,
			  VID ce,
			  numa_allocation && alloc ) :
	m_n( ce ),
	m_adjacency( ce, alloc ) {
	// Constructor taking cut-out and remapping vertex IDs

	if constexpr ( !preallocate_storage && has_dual_rep )
	    new ( &m_index ) mm::buffer<EID>( ce+1, alloc );

	EID h = 0;
	// TODO: allocate using StackLikeAllocator
	for( VID su=0; su < ce; ++su ) {
	    VID u = XP[su];
	    VID deg = std::min( (VID)G.getDegree( u ), ce );
	    VID s = get_hash_slots( deg );
	    if constexpr ( preallocate_storage ) {
		if constexpr ( has_dual_rep )
		    h += 2 * s;
		else
		    h += s;
	    } else {
		if constexpr ( has_dual_rep )
		    h += deg;
	    }
	}

	// Note: if this is only called from top level, and in this case
	// elements of XP have been selected down to have degrees only
	// greater than the size of XP, than XP is smaller than
	// the adjacency lists of its elements, except in rare circumstances.
	auto XP_slice = ngh_set_type( XP, XP+ce );

	if constexpr ( preallocate_storage || has_dual_rep )
	    new ( &m_storage ) mm::buffer<VID>( h, alloc );

	VID * hashes = m_storage.get();
	EID s_next = 0;
	for( VID su=0; su < ce; ++su ) {
	    VID u = XP[su];
	    hash_set_type & a = m_adjacency[su];

	    const VID * n = G.get_neighbours( u );
	    VID deg = G.getDegree( u );
	    const auto & u_adj = H.get_neighbours_set( u );

	    // Intersection u_adj and XP_slice
	    // Common elements need to be remapped to position in XP
	    VID sdeg = std::min( deg, ce );
	    VID logs = get_log_hash_slots( sdeg );
	    VID s = 1 << logs;

	    if constexpr ( preallocate_storage ) {
		new ( &a ) hash_set_type(
		    &hashes[has_dual_rep?(s_next+s):s_next], 0, logs );
		// Remap values: XP[i] -> i, same as hash table mapping
		hash_pa_insert_iterator<dual_rep,true,hash_set_type>
		    out( a, &hashes[s_next], XP );
		graptor::set_operations<graptor::hash_scalar>::intersect_ds(
		    u_adj, XP_slice, out );
		if constexpr ( graptor::is_hash_set_hopscotch_delta_v<hash_set_type> )
		    a.finalise();
		s_next += has_dual_rep ? 2*s : s;
	    } else {
		static_assert( has_dual_rep, "To be completed" );
		new ( &a ) hash_set_type( deg ); // size hint
		// Remap values: XP[i] -> i, same as hash table mapping
		hash_pa_insert_iterator<dual_rep,true,hash_set_type>
		    out( a, &hashes[s_next], XP );
		if constexpr ( has_dual_rep )
		    m_index[su] = s_next;
		graptor::set_operations<graptor::hash_scalar>::intersect_ds(
		    u_adj, XP_slice, out );
		if constexpr ( graptor::is_hash_set_hopscotch_delta_v<hash_set_type> )
		    a.finalise();
		s_next += a.size();
	    }
	}

	if constexpr ( !preallocate_storage && has_dual_rep )
	    m_index[ce] = s_next;
    }

    GraphHAdjPA( const GraphHAdjPA & ) = delete;

    ~GraphHAdjPA() {
	if( !preallocate_storage ) {
	    for( VID v=0; v < m_n; ++v )
		// Do not use get_adjacency() as it will needlessly create
		// the hash set.
		m_adjacency[v].~hash_set_type();
	}
	m_adjacency.del();
	if constexpr ( preallocate_storage || has_dual_rep )
	    m_storage.del();
	if constexpr ( !preallocate_storage && has_dual_rep )
	    m_index.del();
    }

    VID numVertices() const { return m_n; }
    VID get_num_vertices() const { return m_n; }
    VID get_degree( VID v ) const {
	if constexpr ( lazy_hashing )
	    return m_index[v+1] - m_index[v];
	else
	    return get_adjacency( v ).size();
    }
    VID getDegree( VID v ) const { return get_degree( v ); }
    VID get_right_degree( VID v ) const {
	auto pos = std::lower_bound( nbegin( v ), nend( v ), v );
	return std::distance( pos, nend( v ) );
    }

    vertex_iterator vbegin() { return vertex_iterator( 0 ); }
    vertex_iterator vbegin() const { return vertex_iterator( 0 ); }
    vertex_iterator vend() { return vertex_iterator( get_num_vertices() ); }
    vertex_iterator vend() const { return vertex_iterator( get_num_vertices() ); }

    hash_set_type & get_adjacency( VID v ) const {
	return const_cast<self_type *>( this )->get_adjacency( v );
    }
    hash_set_type & get_adjacency( VID v ) {
	hash_set_type & a = m_adjacency[v];
	if constexpr ( lazy_hashing ) {
	    // Relies on a linearisation order where an element is added
	    // before the size counter is incremented.
	    // This does not work correctly for delta hashing which requires
	    // an additional finalisation stage.
	    if( a.size() != m_index[v+1] - m_index[v] ) {
		std::lock_guard<std::mutex> guard( a.get_lock() );
		if( a.size() == 0 ) {
		    a.insert( &m_storage[m_index[v]], &m_storage[m_index[v+1]] );
		    if constexpr ( graptor::is_hash_set_hopscotch_delta_v<hash_set_type> )
			a.finalise();
		    assert( a.size() == m_index[v+1] - m_index[v] );
		}
	    }
	}
	return a;
    }

    bool is_adjacency_initialised( VID v ) const {
	if constexpr ( lazy_hashing )
	    return m_adjacency[v].size() !=0 || m_index[v+1] == m_index[v];
	else
	    return true;
    }

    const VID * get_neighbours( VID v ) const {
	if constexpr ( dual_rep ) {
	    if constexpr ( preallocate_storage )
		return m_adjacency[v].get_table() - m_adjacency[v].capacity();
	    else
		return &m_storage[m_index[v]];
	} else
	    return nullptr;
    }

    auto get_lazy_neighbours_set( VID v ) const {
	if constexpr ( lazy_hashing )
	    return ngh_set_type( get_neighbours( v ), get_degree( v ) );
	else
	    return get_neighbours_set( v );
    }
    dual_set_type get_neighbours_set( VID v ) const {
	return dual_set_type(
	    ngh_set_type( get_neighbours( v ), get_degree( v ) ),
	    get_adjacency( v ) );
    }
    dual_set_type get_left_neighbours_set( VID v ) const {
	auto b = nbegin( v );
	auto e = nend( v );
	auto r = std::lower_bound( b, e, v );
	return dual_set_type( ngh_set_type( b, r ), get_adjacency( v ) );
    }
    dual_set_type get_right_neighbours_set( VID v ) const {
	auto b = nbegin( v );
	auto e = nend( v );
	auto r = std::upper_bound( b, e, v );
	return dual_set_type( ngh_set_type( r, e ), get_adjacency( v ) );
    }
    size_t get_num_right_neighbours( VID v ) const {
	auto b = nbegin( v );
	auto e = nend( v );
	auto r = std::upper_bound( b, e, v );
	return std::distance( r, e );
    }

    neighbour_iterator nbegin( VID v ) {
	return get_neighbours( v );
    }
    const_neighbour_iterator nbegin( VID v ) const {
	return get_neighbours( v );
    }
    neighbour_iterator nend( VID v ) {
	return get_neighbours( v ) + get_degree( v );
    }
    const_neighbour_iterator nend( VID v ) const {
	return get_neighbours( v ) + get_degree( v );
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
    static VID get_hash_slots( VID deg ) {
	return VID(1) << hash_set_type::required_log_size( deg );
    }
    static VID get_log_hash_slots( VID deg ) {
	return hash_set_type::required_log_size( deg );
    }
    static EID get_hash_slots( const EID * index, VID n ) {
	EID h = 0;
	for( VID v=0; v < n; ++v )
	    h += get_hash_slots( index[v+1] - index[v] );
	return h;
    }

private:
    VID m_n;
    mm::buffer<VID> m_storage;
    mm::buffer<EID> m_index;
    mm::buffer<hash_set_type> m_adjacency;
};

template<typename lVID, typename lEID,
	 bool lazyhash,
	 typename lHashSet>
class GraphLazyHashedAdj {
public:
    using self_type =
	GraphLazyHashedAdj<lVID,lEID,lazyhash,lHashSet>;
    using VID = lVID;
    using EID = lEID;
    using hash_set_type = lHashSet;
    using Hash = typename hash_set_type::hash_type;
    using ngh_set_type = graptor::array_slice<VID,VID>;
    using dual_set_type = graptor::dual_set<ngh_set_type,hash_set_type>;

    using vertex_iterator = range_iterator<VID>;
    using edge_iterator = generic_edge_iterator<VID,EID>;
    using neighbour_iterator = VID *;
    using const_neighbour_iterator = const VID *;

    static constexpr bool has_dual_rep = true;
    static constexpr bool preallocate_storage = false;
    static constexpr bool lazy_hashing = lazyhash;

    static_assert( !preallocate_storage || !lazy_hashing,
		   "Lazy hashing implemented only for non-preallocated "
		   "storage" );

public:
    explicit GraphLazyHashedAdj( const ::GraphCSx & G,
				 numa_allocation && alloc,
				 size_t lazy_threshold = 16 ) :
	m_graph( G ),
	m_adjacency( G.numVertices(), alloc ) {

	VID n = m_graph.get_num_vertices();
	const EID * const gindex = G.getIndex();
	const VID * const gedges = G.getEdges();

	parallel_loop( VID(0), n, [&]( VID v ) {
	    EID deg = gindex[v+1] - gindex[v];
	    hash_set_type & a = m_adjacency[v];
	    if( lazy_hashing && deg < lazy_threshold ) {
		new ( &a ) hash_set_type(); // size hint empty
	    } else {
		new ( &a ) hash_set_type( deg ); // size hint
		a.insert( &gedges[gindex[v]], &gedges[gindex[v+1]] );
		if constexpr ( graptor::is_hash_set_hopscotch_delta_v<hash_set_type> )
		    a.finalise();
		assert( a.size() == gindex[v+1] - gindex[v] );
	    }
	} );
    }

    GraphLazyHashedAdj( const GraphLazyHashedAdj & ) = delete;

    ~GraphLazyHashedAdj() {
	VID n = m_graph.get_num_vertices();
	for( VID v=0; v < n; ++v )
	    // Do not use get_adjacency() as it will needlessly create
	    // the hash set.
	    m_adjacency[v].~hash_set_type();
	m_adjacency.del();
    }

    VID numVertices() const { return m_graph.get_num_vertices(); }
    VID get_num_vertices() const { return m_graph.get_num_vertices(); }
    VID get_degree( VID v ) const { return m_graph.get_degree( v ); }
    VID getDegree( VID v ) const { return get_degree( v ); }
    VID get_right_degree( VID v ) const {
	auto pos = std::lower_bound( nbegin( v ), nend( v ), v );
	return std::distance( pos, nend( v ) );
    }

    vertex_iterator vbegin() { return vertex_iterator( 0 ); }
    vertex_iterator vbegin() const { return vertex_iterator( 0 ); }
    vertex_iterator vend() { return vertex_iterator( get_num_vertices() ); }
    vertex_iterator vend() const { return vertex_iterator( get_num_vertices() ); }

    hash_set_type & get_adjacency( VID v ) const {
	return const_cast<self_type *>( this )->get_adjacency( v );
    }
    hash_set_type & get_adjacency( VID v ) {
	hash_set_type & a = m_adjacency[v];
	if constexpr ( lazy_hashing ) {
	    // Relies on a linearisation order where an element is added
	    // before the size counter is incremented.
	    // This does not work correctly for delta hashing which requires
	    // an additional finalisation stage.
	    if( a.size() != get_degree( v ) ) {
		std::lock_guard<std::mutex> guard( a.get_lock() );
		if( a.size() == 0 ) {
		    a.insert( m_graph.get_neighbours( v ),
			      m_graph.get_neighbours( v ) + get_degree( v ) );
		    if constexpr ( graptor::is_hash_set_hopscotch_delta_v<hash_set_type> )
			a.finalise();
		    assert( a.size() == get_degree( v ) );
		}
	    }
	}
	return a;
    }

    bool is_adjacency_initialised( VID v ) const {
	if constexpr ( lazy_hashing )
	    return m_adjacency[v].size() == get_degree( v );
	else
	    return true;
    }

    const VID * get_neighbours( VID v ) const {
	return m_graph.get_neighbours( v );
    }

    auto get_lazy_neighbours_set( VID v ) const {
	if constexpr ( lazy_hashing )
	    return ngh_set_type( get_neighbours( v ), get_degree( v ) );
	else
	    return get_neighbours_set( v );
    }
    dual_set_type get_neighbours_set( VID v ) const {
	return dual_set_type(
	    ngh_set_type( get_neighbours( v ), get_degree( v ) ),
	    get_adjacency( v ) );
    }
    dual_set_type get_left_neighbours_set( VID v ) const {
	auto b = nbegin( v );
	auto e = nend( v );
	auto r = std::lower_bound( b, e, v );
	return dual_set_type( ngh_set_type( b, r ), get_adjacency( v ) );
    }
    dual_set_type get_right_neighbours_set( VID v ) const {
	auto b = nbegin( v );
	auto e = nend( v );
	auto r = std::upper_bound( b, e, v );
	return dual_set_type( ngh_set_type( r, e ), get_adjacency( v ) );
    }
    size_t get_num_right_neighbours( VID v ) const {
	auto b = nbegin( v );
	auto e = nend( v );
	auto r = std::upper_bound( b, e, v );
	return std::distance( r, e );
    }

    neighbour_iterator nbegin( VID v ) {
	return get_neighbours( v );
    }
    const_neighbour_iterator nbegin( VID v ) const {
	return get_neighbours( v );
    }
    neighbour_iterator nend( VID v ) {
	return get_neighbours( v ) + get_degree( v );
    }
    const_neighbour_iterator nend( VID v ) const {
	return get_neighbours( v ) + get_degree( v );
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
    const ::GraphCSx & m_graph;
    mm::buffer<hash_set_type> m_adjacency;
};

template<typename lVID, typename lEID,
	 typename lHashSet>
class GraphLazyRemappedHashedAdj {
public:
    using self_type =
	GraphLazyRemappedHashedAdj<lVID,lEID,lHashSet>;
    using VID = lVID;
    using EID = lEID;
    using hash_set_type = lHashSet;
    using Hash = typename hash_set_type::hash_type;
    using ngh_set_type = graptor::array_slice<VID,VID>;
    using maybe_dual_set_type =
	graptor::maybe_dual_set<ngh_set_type,hash_set_type>;

    using vertex_iterator = range_iterator<VID>;
    using edge_iterator = generic_edge_iterator<VID,EID>;
    using neighbour_iterator = VID *;
    using const_neighbour_iterator = const VID *;

    static constexpr uint8_t fl_zero = 0;
    static constexpr uint8_t fl_seq = 1;
    static constexpr uint8_t fl_hash = 2;

public:
    template<typename Fn>
    explicit GraphLazyRemappedHashedAdj(
	const ::GraphCSx & G,
	const VID * const remap_to_orig,
	const VID * const orig_to_remap,
	const VID * const coreness,
	numa_allocation && alloc,
	size_t hash_threshold,
	Fn && lazy_fn ) :
	m_orig_graph( G ),
	m_remap_to_orig( remap_to_orig ),
	m_orig_to_remap( orig_to_remap ),
	m_coreness( coreness ),
	m_remap_graph( G.get_num_vertices(), G.get_num_edges(), -1,
		       true, false ),
	m_hash_threshold( hash_threshold ),
	m_adjacency( G.numVertices(), alloc ),
	m_flags( G.numVertices(), alloc )
#if LAZY_HASH_FILTER
	,
	m_degree( G.numVertices(), alloc )
#endif
	{

	VID n = m_orig_graph.get_num_vertices();
	const EID * const gindex = m_orig_graph.getIndex();
	const VID * const gedges = m_orig_graph.getEdges();

#if LAZY_HASH_FILTER
	// At least a 2-clique can be/has been found
	m_highest_threshold = 2;
#endif

	EID * const index = m_remap_graph.getIndex();
	VID * const edges = m_remap_graph.getEdges();

	index[n] = sum_scan(
	    index, VID(0), n, [&]( VID rv ) {
		VID ov = m_remap_to_orig[rv];
		return (VID)( gindex[ov+1] - gindex[ov] );
	    } );

	parallel_loop( VID(0), n, [&]( VID rv ) {
	    m_flags[rv].store( fl_zero );
	    VID ov = m_remap_to_orig[rv];
	    EID deg = gindex[ov+1] - gindex[ov];
#if LAZY_HASH_FILTER
	    m_degree[rv] = deg;
#endif
	    hash_set_type & a = m_adjacency[rv];
	    if( !lazy_fn( rv ) ) {
		new ( &a ) hash_set_type(); // size hint empty
	    } else if( deg >= m_hash_threshold ) {
		new ( &a ) hash_set_type( deg ); // size hint
		build_hash_set( rv, ov, 0 );
	    } else {
		new ( &a ) hash_set_type(); // empty size hint
		build_seq( rv, ov, 0 );
	    }
	} );
    }

    GraphLazyRemappedHashedAdj( const GraphLazyRemappedHashedAdj & ) = delete;
    GraphLazyRemappedHashedAdj( GraphLazyRemappedHashedAdj && ) = delete;

    ~GraphLazyRemappedHashedAdj() {
	VID n = m_remap_graph.get_num_vertices();
	for( VID v=0; v < n; ++v )
	    // Do not use get_adjacency() as it will needlessly create
	    // the hash set.
	    m_adjacency[v].~hash_set_type();
	m_adjacency.del();
	m_remap_graph.del();
#if LAZY_HASH_FILTER
	m_degree.del();
#endif
    }

    VID numVertices() const { return m_remap_graph.get_num_vertices(); }
    VID get_num_vertices() const { return m_remap_graph.get_num_vertices(); }
#if LAZY_HASH_FILTER
    VID get_degree( VID v ) const { return m_degree[v]; }
#else
    VID get_degree( VID v ) const { return m_remap_graph.get_degree( v ); }
#endif
    VID getDegree( VID v ) const { return get_degree( v ); }

    vertex_iterator vbegin() { return vertex_iterator( 0 ); }
    vertex_iterator vbegin() const { return vertex_iterator( 0 ); }
    vertex_iterator vend() { return vertex_iterator( get_num_vertices() ); }
    vertex_iterator vend() const { return vertex_iterator( get_num_vertices() ); }

    hash_set_type & get_adjacency( VID v, VID cth = -1 ) const {
	return const_cast<self_type *>( this )->get_adjacency( v, cth );
    }
    hash_set_type & get_adjacency( VID v, VID cth = -1 ) {
	return get_hash_set( v, cth );
    }

public:
    const VID * get_neighbours( VID v, VID cth = -1 ) const {
	return get_seq( v, cth );
    }

    maybe_dual_set_type get_lazy_neighbours_set( VID v, VID cth = -1 ) const {
	return get_neighbours_set( v, cth, false );
    }

    maybe_dual_set_type get_neighbours_set( VID v, VID cth = -1 ) const {
	return get_neighbours_set(
	    v, cth, get_degree( v ) >= m_hash_threshold );
    }
    
private:
    maybe_dual_set_type
    get_neighbours_set( VID v, VID cth, bool create_hash ) const {
	bool has_h = is_hash_set_initialised( v );
	bool has_s = is_seq_initialised( v );
	if( has_h && has_s )
	    return maybe_dual_set_type(
		ngh_set_type( m_remap_graph.get_neighbours( v ),
			      get_degree( v ) ),
		m_adjacency[v] );
	else if( has_h )
	    return maybe_dual_set_type( m_adjacency[v] );
	else if( has_s )
	    return maybe_dual_set_type(
		ngh_set_type( m_remap_graph.get_neighbours( v ),
			      get_degree( v ) ) );
	else {
	    // Create something. A hash set does not require sorting,
	    // but a small set is compact sequentially and quick to sort,
	    // and benefits from sequential access.
	    if( create_hash )
		return maybe_dual_set_type( get_hash_set( v, cth ) );
	    else
		return maybe_dual_set_type(
		    ngh_set_type( get_seq( v, cth ), get_degree( v ) ) );
	}
    }
public:

#if 0
    // Not using this on this graph
    //! Get sequence of left neighbours
    // This method must materialise both representations (colouring - check)
    maybe_dual_set_type get_left_neighbours_set( VID v ) const {
	VID * ngh = get_seq( v );
	VID deg = m_remap_graph.get_degree( v );
	
	auto b = ngh;
	auto e = ngh + v;
	auto r = std::lower_bound( b, e, v );
	return maybe_dual_set_type( ngh_set_type( b, r ), get_hash_set( v ) );
    }
#endif

    //! Get sequence of right neighbours
    // This method must materialise a sequential list. Return the hash set
    // also if available.
    maybe_dual_set_type get_right_neighbours_set( VID v, VID cth = -1 ) const {
	VID * ngh = get_seq( v, cth );
	VID deg = get_degree( v );
	
	auto b = ngh;
	auto e = ngh + deg;
	auto r = std::upper_bound( b, e, v );
	return is_hash_set_initialised( v )
	    ? maybe_dual_set_type( ngh_set_type( r, e ), m_adjacency[v] )
	    : maybe_dual_set_type( ngh_set_type( r, e ) );
    }

/*
    neighbour_iterator nbegin( VID v ) {
	return get_neighbours( v );
    }
    const_neighbour_iterator nbegin( VID v ) const {
	return get_neighbours( v );
    }
    neighbour_iterator nend( VID v ) {
	return get_neighbours( v ) + get_degree( v );
    }
    const_neighbour_iterator nend( VID v ) const {
	return get_neighbours( v ) + get_degree( v );
    }
*/

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
    hash_set_type & get_hash_set( VID rv, VID cth ) const {
	if( is_hash_set_initialised( rv ) )
	    return m_adjacency[rv];
	else
	    return build_hash_set( rv, m_remap_to_orig[rv], cth );
    }
    hash_set_type & build_hash_set( VID rv, VID ov, VID cth ) const {
	hash_set_type & a = m_adjacency[rv];

	std::lock_guard<std::mutex> guard( a.get_lock() );
	if( is_hash_set_initialised( rv ) )
	    return a;

	const EID * const gindex = m_orig_graph.getIndex();
	const VID * const gedges = m_orig_graph.getEdges();

	// Try to get allocation size right from the start
	a.create_if_uninitialised( gindex[ov+1] - gindex[ov] );

#if LAZY_HASH_FILTER
	if( cth == -1 )
	    cth = m_highest_threshold.load();
	else {
	    VID cur = m_highest_threshold.load();
	    if( cth > cur )
		while( !m_highest_threshold.compare_exchange_weak(
			   cur, cth,
			   std::memory_order_release,
			   std::memory_order_relaxed ) ) { }
	}
#endif

	for( EID oe=gindex[ov], oee=gindex[ov+1]; oe != oee; ++oe ) {
	    VID ou = gedges[oe];
	    VID ru = m_orig_to_remap[ou];
#if LAZY_HASH_FILTER
	    if( m_coreness[ru] >= cth )
		a.insert( ru );
#else
	    a.insert( ru );
#endif
	}

	assert( a.size() <= gindex[ov+1] - gindex[ov] );

#if LAZY_HASH_FILTER
	m_degree[rv] = a.size();
#endif
	set_hash_set( rv );

	return a;
    }

    VID * get_seq( VID rv, VID cth ) const {
	if( is_seq_initialised( rv ) )
	    return const_cast<VID *>( m_remap_graph.get_neighbours( rv ) );
	else
	    return build_seq( rv, m_remap_to_orig[rv], cth );
    }
    VID * build_seq( VID rv, VID ov, VID cth ) const {
	hash_set_type & a = m_adjacency[rv];

	const EID * const index = m_remap_graph.getIndex();
	VID * const edges = m_remap_graph.getEdges();

	std::lock_guard<std::mutex> guard( a.get_lock() );
	if( is_seq_initialised( rv ) )
	    return &edges[index[rv]];

	const EID * const gindex = m_orig_graph.getIndex();
	const VID * const gedges = m_orig_graph.getEdges();

#if LAZY_HASH_FILTER
	if( cth == -1 )
	    cth = m_highest_threshold.load();
	else {
	    VID cur = m_highest_threshold.load();
	    if( cth > cur )
		while( !m_highest_threshold.compare_exchange_weak(
			   cur, cth,
			   std::memory_order_release,
			   std::memory_order_relaxed ) ) { }
	}
#endif

	EID re = index[rv];
	for( EID oe=gindex[ov], oee=gindex[ov+1]; oe != oee; ++oe ) {
	    VID ou = gedges[oe];
	    VID ru = m_orig_to_remap[ou];
#if LAZY_HASH_FILTER
	    if( m_coreness[ru] >= cth )
		edges[re++] = ru;
#else
	    edges[re++] = ru;
#endif
	}

	assert( re <= index[rv+1] );

	// TODO: ips4o
	std::sort( &edges[index[rv]], &edges[re] );

#if LAZY_HASH_FILTER
	m_degree[rv] = re - index[rv];
#endif
	set_sequential( rv );

	return &edges[index[rv]];
    }

public:
    bool is_hash_set_initialised( VID v ) const {
	// return m_adjacency[v].size() == m_remap_graph.get_degree( v );
	return ( m_flags[v].load( std::memory_order_acquire ) & fl_hash ) != 0;
    }
    bool is_seq_initialised( VID v ) const {
    return ( m_flags[v].load( std::memory_order_acquire ) & fl_seq ) != 0;
    }
private:
    void set_hash_set( VID v ) const {
	m_flags[v].fetch_or( fl_hash );
    }
    void set_sequential( VID v ) const {
	m_flags[v].fetch_or( fl_seq );
    }

private:
    const ::GraphCSx & m_orig_graph;
    mutable ::GraphCSx m_remap_graph;
    const VID * const m_remap_to_orig;
    const VID * const m_orig_to_remap;
    const VID * const m_coreness;
    VID m_hash_threshold;
    mutable mm::buffer<hash_set_type> m_adjacency;
    mutable mm::buffer<std::atomic<uint8_t>> m_flags;
#if LAZY_HASH_FILTER
    mutable mm::buffer<VID> m_degree;
    mutable std::atomic<VID> m_highest_threshold;
#endif
};


} // namespace graph

} // namespace graptor

#endif // GRAPHGRIND_GRAPH_SIMPLE_HADJ_H
