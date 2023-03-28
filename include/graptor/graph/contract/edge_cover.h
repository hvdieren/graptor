// -*- c++ -*-
#ifndef GRAPTOR_GRAPH_CONTRACT_EDGE_COVER_H
#define GRAPTOR_GRAPH_CONTRACT_EDGE_COVER_H

#include <algorithm>
#include <cstdint>
#include "graptor/mm/buffer.h"
#include "graptor/graph/contract/vertex_set.h"

namespace contract {
    
template<typename VID, typename EID>
class edge_cover {
    using counter_ty = int8_t;
    
public:
    edge_cover( const GraphCSx & G )
	: m_G( G ),
	  m_covered( G.numEdges(), numa_allocation_interleaved() ) {
	parallel_loop( (EID)0, G.numEdges(),
		       [&]( auto e ) { m_covered[e] = 0; } );
    }
    ~edge_cover() {
	m_covered.del();
    }

    /**
     * Try to allocate both edges between a pair of vertices. If the allocation
     * fails, lift both allocations.
     * Allocations may fail if an edge was previously allocated, or when
     * another thread concurrently aims to allocate the same edges.
     * Return true if the allocation of both edges was successful.
     * TODO: could simplify to allocate edges only in one direction, e.g., u<v
     */
    bool cover( VID u, VID v ) {
	// This assumes an undirected graph
	// TODO: consider CAS to avoid counters going up (max value ~ #threads)
	//       CAS could also admit bit mask instead of counter, as well as
	//       bulk setting of multiple bits in the same word (edge_cover).
	counter_ty c0
	    = __sync_fetch_and_add( &m_covered[get_eid(u,v)], (counter_ty)1 );
	if( c0 != 0 ) {
	    __sync_fetch_and_add( &m_covered[get_eid(u,v)], (counter_ty)-1 );
	    return false;
	}
	counter_ty c1
	    = __sync_fetch_and_add( &m_covered[get_eid(v,u)], (counter_ty)1 );
	if( c1 != 0 ) {
	    __sync_fetch_and_add( &m_covered[get_eid(u,v)], (counter_ty)-1 );
	    __sync_fetch_and_add( &m_covered[get_eid(v,u)], (counter_ty)-1 );
	    return false;
	}
	return true;
    }

    /**
     * The vertex set s is a subset of the neighbours of u. Allocate all
     * of the edges between u and a member of s.
     * If not all edges can be allocated, then release all of them.
     * Return true if all edges were allocated, false otherwise.
     */
    bool cover_edges( VID u, const vertex_set<VID> & s ) {
	auto B = s.begin();
	auto E = s.end();
	auto I = B;
	for( ; I != E; ++I ) {
	    if( !cover( u, *I ) )
		break;
	}

	if( I != E ) { // failed to allocate all edges, drop vertex
	    E = I;
	    for( I=B; I != E; ++I )
		release( u, *I );
	    return false;
	} else
	    return true;
    }

    /**
     * Release all edges between u and a member of s.
     */
    void release_edges( VID u, const vertex_set<VID> & s ) {
	auto I = s.begin();
	auto E = s.end();
	for( ; I != E; ++I )
	    release( u, *I );
    }

    bool is_covered( VID u, VID v ) const {
	return m_covered[get_eid(u,v)] != 0;
    }
    bool is_covered( EID e ) const {
	return m_covered[e] != 0;
    }

private:
    EID get_eid( VID u, VID v ) const {
	const EID * const idx = m_G.getIndex();
	const VID * const edges = m_G.getEdges();
	const VID * const p
	    = std::lower_bound( &edges[idx[u]], &edges[idx[u+1]], v );
	assert( *p == v );
	return p - edges;
    }

    void release( VID u, VID v ) {
	// This assumes an undirected graph
	__sync_fetch_and_add( &m_covered[get_eid(u,v)], (counter_ty)-1 );
	__sync_fetch_and_add( &m_covered[get_eid(v,u)], (counter_ty)-1 );
    }

private:
    const GraphCSx & m_G;
    // Could also store pattern type + pattern ID within EID space
    // (EID space follows from #patterns <= #edges)
    mm::buffer<counter_ty> m_covered;
};

} // namespace contract

#endif // GRAPTOR_GRAPH_CONTRACT_EDGE_COVER_H
