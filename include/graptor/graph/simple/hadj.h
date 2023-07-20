// -*- C++ -*-
#ifndef GRAPHGRIND_GRAPH_SIMPLE_HADJ_H
#define GRAPHGRIND_GRAPH_SIMPLE_HADJ_H

#include "graptor/itraits.h"
#include "graptor/utils.h"
#include "graptor/mm.h"
#include "graptor/mm/mm.h"

#include "graptor/container/hash_table.h"

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
    using hash_table_type = graptor::hash_table<VID,Hash>;

    using vertex_iterator = range_iterator<VID>;
    using edge_iterator = generic_edge_iterator<VID,EID>;

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
	    hash_table_type a( &hashes[h], 0, logs, hf( v ) );
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
	    hash_table_type a( &hashes[index[v]], 0, logs, hf[v] );
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

    hash_table_type get_adjacency( VID v ) const {
	return const_cast<self_type *>( this )->get_adjacency( v );
    }
    hash_table_type get_adjacency( VID v ) {
	VID deg = getIndex()[v+1] - getIndex()[v];
	VID h = rt_ilog2( getHashIndex()[v+1] - getHashIndex()[v] );
	const Hash * hf = m_hash_fn.get();
	return hash_table_type(
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
	return VID(1) << hash_table_type::required_log_size( deg );
    }
    static VID get_log_hash_slots( VID deg ) {
	return hash_table_type::required_log_size( deg );
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

} // namespace graph

} // namespace graptor

#endif // GRAPHGRIND_GRAPH_SIMPLE_HADJ_H
