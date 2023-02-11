// -*- C++ -*-
#ifndef GRAPHGRIND_GRAPH_GRAPHCSR_H
#define GRAPHGRIND_GRAPH_GRAPHCSR_H

#include "graptor/partitioner.h"
#include "graptor/graph/GraphCSx.h"
#include "graptor/graph/EIDRemapper.h"


class GraphCSRAdaptor {
    using EIDRetriever = IdempotentEIDRetriever<VID,EID>;

public:
    GraphCSRAdaptor( const GraphCSx & csx )
	: GraphCSRAdaptor( csx, graptor_num_threads() * 8 ) { }
    GraphCSRAdaptor( const GraphCSx & csx, unsigned short npart )
	: m_csr( csx ), m_part( npart, m_csr.numVertices() ) {
	// Partition vertex set
	partitionBalanceEdges( m_csr, m_part );

	// set up edge partitioner - determines how to allocate edge properties
	EID * counts = m_part.edge_starts();
	const EID * const index = m_csr.getIndex();
	EID ne = 0;
	for( unsigned short p=0; p < npart; ++p ) {
	    counts[p] = ne;
	    ne += index[m_part.end_of(p)] - index[m_part.start_of(p)];
	}
	counts[npart] = ne;	
	assert( ne == m_csr.numEdges() );
    }

    const EID *getIndex() const { return m_csr.getIndex(); }
    const VID *getEdges() const { return m_csr.getEdges(); }
    const VID *getOutDegree() const { return m_csr.getDegree(); }

    VID numVertices() const { return m_csr.numVertices(); }
    EID numEdges() const { return m_csr.numEdges(); }

    VID getOutDegree( VID v ) const { return m_csr.getDegree( v ); }
    VID getOutNeighbor( VID v, VID pos ) const {
	return m_csr.getNeighbor( v, pos );
    }

    bool isSymmetric() const { return m_csr.isSymmetric(); }

    bool hasEdge( VID s, VID d ) const { return m_csr.hasEdge( s, d ); }
    std::pair<bool,float> getEdgeWeight( VID s, VID d ) const {
	return m_csr.getEdgeWeight( s, d );
    }

    const mm::buffer<float> * getWeights() const {
	return m_csr.getWeights();
    }

    const partitioner & get_partitioner() const { return m_part; }
    const EIDRetriever & get_eid_retriever() const { return m_eid_retriever; }
    const GraphCSx & getCSR() const { return m_csr; }
    
    // This graph only supports scalar processing in our system as destinations
    // are not laid out in a way that excludes conflicts across vector lanes
    // in the COO representation.
    static constexpr unsigned short getPullVLBound() { return 1; }
    static constexpr unsigned short getPushVLBound() { return 1; }
    static constexpr unsigned short getIRegVLBound() { return 1; }

    graph_traversal_kind select_traversal(
	bool fsrc_strong,
	bool fdst_strong,
	bool adst_strong,
	bool record_strong,
	frontier F,
	bool is_sparse ) const {

	if( is_sparse )
	    return graph_traversal_kind::gt_sparse;

	return graph_traversal_kind::gt_push;
    }

    static constexpr bool is_privatized( graph_traversal_kind gtk ) {
#if OWNER_READS
	return gtk == graph_traversal_kind::gt_push;
#else
	return gtk == graph_traversal_kind::gt_pull;
#endif
    }

private:
    const GraphCSx & m_csr;
    partitioner m_part;
    EIDRetriever m_eid_retriever;
};

#endif // GRAPHGRIND_GRAPH_GRAPHCSR_H

