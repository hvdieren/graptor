// -*- C++ -*-
#ifndef GRAPHGRIND_GRAPH_SIMPLE_HADJ_H
#define GRAPHGRIND_GRAPH_SIMPLE_HADJ_H

#include "graptor/itraits.h"
#include "graptor/utils.h"
#include "graptor/mm.h"
#include "graptor/mm/mm.h"

#include "graptor/container/hash_set.h"
#include "graptor/container/hash_table.h"
#include "graptor/container/intersect.h"

#include "graptor/container/range_iterator.h"
#include "graptor/container/double_index_edge_iterator.h"
#include "graptor/container/difference_iterator.h"


namespace graptor {

namespace graph {

/*!======================================================================*
 * GraphHadj: A graph data structure that represents adjacency lists as
 * hash tables using a single pre-allocated amount of memory.
 *
 * Note: some hash functions (e.g. graptor::rand_hash) require members that
 *       impacts on the location of elements. As such, the instance of the
 *       hash function needs to be retained across constructions of the
 *       hashed adjacency list.
 *=======================================================================*/

template<typename lVID, typename lEID, typename UGCSx,
	 typename lHash = std::hash<lVID>>
class GraphHAdj {
public:
    using VID = lVID;
    using EID = lEID;
    using Hash = lHash;
    using UnderlyingGraphCSx = UGCSx; // GraphCSx or graptor::graph::GraphCSx<>
    using self_type = GraphHAdj<VID,EID,UnderlyingGraphCSx,Hash>;
    using hash_set_type = graptor::hash_set<VID,Hash>;

    using vertex_iterator = range_iterator<VID>;
    using edge_iterator = generic_edge_iterator<VID,EID>;

    static constexpr bool has_dual_rep = false;

public:
    explicit GraphHAdj( const GraphCSx<VID,EID> & G ) :
	GraphHAdj( G, numa_allocation_interleaved() ) { }
    explicit GraphHAdj( const GraphCSx<VID,EID> & G,
			numa_allocation && alloc ) :
	m_G( G ),
	m_index( G.numVertices()+1, alloc ),
	m_hashes( get_hash_slots( G.getIndex(), G.numVertices() ), alloc ),
	m_hash_fn( G.numVertices() ) {
	VID n = G.numVertices();
	EID h = 0;
	EID * index = m_index.get();
	VID * hashes = m_hashes.get();
	EID * gindex = getIndex();
	VID * gedges = getEdges();
	Hash * hf = m_hash_fn.get();
	for( VID v=0; v < n; ++v ) {
	    VID deg = gindex[v+1] - gindex[v];
	    VID logs = get_log_hash_slots( deg );
	    VID s = VID(1) << logs;
	    new ( &hf[v] ) Hash( logs ); // rand_hash function requires storage
	    hash_set_type a( &hashes[h], 0, logs, hf( v ) );
	    a.clear(); // initialise to invalid element
	    a.insert( &gedges[gindex[v]], &gedges[gindex[v+1]] );
	    index[v] = h;
	    h += s;
	}
	index[n] = h;
    }
    explicit GraphHAdj( const ::GraphCSx & G,
			bool parallel,
			numa_allocation && alloc ) :
	m_G( G ),
	m_index( G.numVertices()+1, alloc ),
	// m_hashes( get_hash_slots( G.getIndex(), G.numVertices() ), alloc ),
	m_hash_fn( G.numVertices(), alloc ) {
	VID n = G.numVertices();
	EID h = 0;
	EID * index = m_index.get();
	const EID * const gindex = getIndex();
	const VID * const gedges = getEdges();
	parallel_loop( VID(0), n, [&]( VID v ) {
	    VID deg = gindex[v+1] - gindex[v];
	    index[v] = get_hash_slots( deg );
	} );
	index[n] = sequence::plusScan( index, index, n );

	new ( &m_hashes ) mm::buffer<VID>( index[n], alloc );
	VID * hashes = m_hashes.get();
	Hash * hf = m_hash_fn.get();
	parallel_loop( VID(0), n, [&]( VID v ) {
	    VID deg = gindex[v+1] - gindex[v];
	    VID s = index[v+1] - index[v];
	    VID logs = rt_ilog2( s );
	    new ( &hf[v] ) Hash( logs ); // rand_hash function requires storage
	    hash_set_type a( &hashes[index[v]], 0, logs, hf[v] );
	    a.clear(); // initialise to invalid element
	    a.insert( &gedges[gindex[v]], &gedges[gindex[v+1]] );
	} );
    }
    GraphHAdj( const GraphHAdj & ) = delete;

    ~GraphHAdj() {
	m_index.del();
	m_hashes.del();
    }

    const auto & get_graph() const { return m_G; }

    VID numVertices() const { return m_G.numVertices(); }
    EID numEdges() const { return m_G.numEdges(); }

    const EID * getIndex() { return m_G.getIndex(); }
    const EID * const getIndex() const { return m_G.getIndex(); }
    const VID * getEdges() { return m_G.getEdges(); }
    const VID * const getEdges() const { return m_G.getEdges(); }
    EID * getHashIndex() { return m_index.get(); }
    const EID * getHashIndex() const { return m_index.get(); }
    VID * getHashes() { return m_hashes.get(); }
    const VID * getHashes() const { return m_hashes.get(); }

    vertex_iterator vbegin() { return vertex_iterator( 0 ); }
    vertex_iterator vbegin() const { return vertex_iterator( 0 ); }
    vertex_iterator vend() { return vertex_iterator( numVertices() ); }
    vertex_iterator vend() const { return vertex_iterator( numVertices() ); }

    hash_set_type get_adjacency( VID v ) const {
	return const_cast<self_type *>( this )->get_adjacency( v );
    }
    hash_set_type get_adjacency( VID v ) {
	VID deg = getIndex()[v+1] - getIndex()[v];
	VID h = rt_ilog2( getHashIndex()[v+1] - getHashIndex()[v] );
	const Hash * hf = m_hash_fn.get();
	return hash_set_type(
	    &getHashes()[getHashIndex()[v]], deg, h, hf[v] );
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
    const UnderlyingGraphCSx & m_G;
    mm::buffer<EID> m_index;
    mm::buffer<VID> m_hashes;
    mm::buffer<Hash> m_hash_fn;
};

template<bool dual_rep, bool left_base, typename Hash>
struct hash_pa_insert_iterator {
    hash_pa_insert_iterator(
	hash_set<VID,Hash> & table, VID * list, const VID * start )
	: m_table( table ), m_list( list ), m_start( start ) { }

    void push_back( const VID * lt, const VID * rt = nullptr ) {
	VID v = ( left_base ? lt : rt ) - m_start;
	m_table.insert( v );
	if constexpr ( dual_rep )
	    *m_list++ = v;
    }

private:
    hash_set<VID,Hash> & m_table;
    VID * m_list;
    const VID * m_start;
};

template<typename lVID, typename lEID,
	 bool dual_rep,
	 typename lHash = std::hash<lVID>>
class GraphHAdjPA {
public:
    using VID = lVID;
    using EID = lEID;
    using Hash = lHash;
    using self_type = GraphHAdjPA<VID,EID,dual_rep,Hash>;
    using hash_set_type = graptor::hash_set<VID,Hash>;
    using hash_table_type = graptor::hash_table<VID,VID,Hash>;

    using vertex_iterator = range_iterator<VID>;
    using edge_iterator = generic_edge_iterator<VID,EID>;

    static constexpr bool has_dual_rep = dual_rep;

public:
    explicit GraphHAdjPA( const ::GraphCSx & G,
			  numa_allocation && alloc ) :
	m_n( G.numVertices() ),
	m_adjacency( G.numVertices(), alloc ) {
	VID n = m_n;
	const EID * const gindex = G.getIndex();
	const VID * const gedges = G.getEdges();

	mm::buffer<EID> index_buf( G.numVertices()+1, alloc );
	EID * index = index_buf.get();
	EID h = 0;
	parallel_loop( VID(0), n, [&]( VID v ) {
	    EID deg = gindex[v+1] - gindex[v];
	    index[v] = get_hash_slots( deg );
	    if constexpr ( dual_rep )
		index[v] *= 2;
	} );
	index[n] = sequence::plusScan( index, index, n );

	new ( &m_storage ) mm::buffer<VID>( index[n], alloc );
	VID * hashes = m_storage.get();
	parallel_loop( VID(0), n, [&]( VID v ) {
	    VID s = index[v+1] - index[v];
	    if constexpr ( dual_rep )
		s /= 2;
	    VID logs = rt_ilog2( s );
	    hash_set_type & a = m_adjacency[v];
	    new ( &a ) hash_set_type( &hashes[index[v]+(dual_rep?s:0)], 0, logs );
	    a.clear(); // initialise to invalid element
	    a.insert( &gedges[gindex[v]], &gedges[gindex[v+1]] );
	    if constexpr ( dual_rep )
		std::copy( &gedges[gindex[v]], &gedges[gindex[v+1]],
			   &hashes[index[v]] );
	} );

	index_buf.del();
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
			hash_pa_insert_iterator<dual_rep,false,Hash>
			    out( a, arr, XP );
			graptor::merge_scalar::intersect<true>(
			    n, n+deg,	// X+P; shorter list
			    XP, XP+ne,	// X
			    out );
		    }

		    // P-P edges
		    // New iterator to count elements already included
		    hash_pa_insert_iterator<dual_rep,false,Hash>
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
		hash_pa_insert_iterator<dual_rep,true,Hash>
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
		hash_pa_insert_iterator<dual_rep,true,Hash>
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
    template<typename CSxGraphTy, typename HGraphTy, typename FilterFn>
    explicit GraphHAdjPA( const CSxGraphTy & G,
			  const HGraphTy & H,
			  const VID * const XP,
			  VID ce,
			  numa_allocation && alloc,
			  FilterFn && fn ) :
	m_n( ce ),
	m_adjacency( ce, alloc ) {
	// Constructor taking cut-out and remapping vertex IDs

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
	hash_set_type XP_hash( XP, XP+ce );

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
		// Alternative: first place intersection in sequential storage,
		// using hash of XP, then translate and insert in adjacency a
		// Seems best-performing
		assert( has_dual_rep && "otherwise need temporary space" );
		VID * arr = &hashes[s_next];
		VID * e = graptor::hash_scalar::intersect(
		    n, n+deg, XP_hash, arr );
		VID logs = get_log_hash_slots( e - arr );
		VID s = 1 << logs;
		new ( &a ) hash_set_type(
		    &hashes[has_dual_rep?(s_next+s):s_next], 0, logs );
		s_next += has_dual_rep ? 2*s : s;
		
		for( VID * i=arr; i != e; ++i ) {
		    const VID * pos = std::lower_bound( XP, XP+ce, *i );
		    assert( *pos == *i );
		    VID sv = pos - XP;
		    assert( sv != su );
		    if constexpr ( has_dual_rep )
			*i = sv;
		    a.insert( sv );
		}
	    } else {
		VID sdeg = std::min( deg, ce );
		VID logs = get_log_hash_slots( sdeg );
		VID s = 1 << logs;
		new ( &a ) hash_set_type(
		    &hashes[has_dual_rep?(s_next+s):s_next], 0, logs );
		hash_pa_insert_iterator<dual_rep,true,Hash>
		    out( a, &hashes[s_next], XP );
		graptor::hash_scalar::intersect<true>(
		    XP, XP+ce, H.get_adjacency( u ), out );
		s_next += has_dual_rep ? 2*s : s;
	    }
	}
    }

    GraphHAdjPA( const GraphHAdjPA & ) = delete;

    ~GraphHAdjPA() {
	m_adjacency.del();
	m_storage.del();
    }

    VID numVertices() const { return m_n; }
    VID getDegree( VID v ) const { return get_adjacency( v ).size(); }

    vertex_iterator vbegin() { return vertex_iterator( 0 ); }
    vertex_iterator vbegin() const { return vertex_iterator( 0 ); }
    vertex_iterator vend() { return vertex_iterator( numVertices() ); }
    vertex_iterator vend() const { return vertex_iterator( numVertices() ); }

    hash_set_type & get_adjacency( VID v ) const {
	return const_cast<self_type *>( this )->get_adjacency( v );
    }
    hash_set_type & get_adjacency( VID v ) { return m_adjacency[v]; }

    const VID * get_neighbours( VID v ) const {
	if constexpr ( dual_rep )
	    return m_adjacency[v].get_table() - m_adjacency[v].capacity();
	else
	    return nullptr;
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
    mm::buffer<hash_set_type> m_adjacency;
};


} // namespace graph

} // namespace graptor

#endif // GRAPHGRIND_GRAPH_SIMPLE_HADJ_H
