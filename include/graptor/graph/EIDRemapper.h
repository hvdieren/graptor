// -*- c++ -*-
#ifndef GRAPTOR_GRAPH_EIDREMAPPER_H
#define GRAPTOR_GRAPH_EIDREMAPPER_H

#include "graptor/graph/GraphCSx.h"

template<typename fVID, typename fEID>
class IdempotentEIDRetriever {
public:
    using VID = fVID;
    using EID = fEID;

    EID get_edge_eid( EID seq ) const {
	return seq;
    }
};

template<typename fVID, typename fEID>
class PartitionedCOOEIDRetriever {
public:
    using VID = fVID;
    using EID = fEID;

    PartitionedCOOEIDRetriever( EID first_edge )
	: m_first_edge( first_edge ) { }

    EID get_edge_eid( EID seq ) const {
	// std::cout << "COO EID " << seq << " + " << m_first_edge
	// << " -> " << ( seq + m_first_edge ) << "\n";
	return seq + m_first_edge;
    }

private:
    EID m_first_edge;
};

template<typename fVID, typename fEID>
class CSxEIDRetriever {
public:
    using VID = fVID;
    using EID = fEID;

    CSxEIDRetriever() { }
    CSxEIDRetriever( const GraphCSx & csr, mmap_ptr<EID> remap )
	: m_csr( &csr ), m_remap( remap ) { }

    /**
     * get_edge_eid - retrieve EID for specific edge
     */
    EID get_edge_eid( VID src, VID seq ) const {
	// std::cout << "CSx EID " << src << " #" << seq
	// << " comp " << (m_csr->getIndex()[src]+seq)
	// << " xlat " << (m_csr->getIndex()[src]+seq)
	// << m_remap[m_csr->getIndex()[src]+seq] << "\n";
	return m_remap[m_csr->getIndex()[src]+seq];
    }

private:
    mmap_ptr<EID> m_remap;  //!< Remapped edge IDs
    const GraphCSx * m_csr; //!< Reference CSR graph
};

template<typename fVID, typename fEID>
class CSxEIDRemapper {
public:
    using VID = fVID;
    using EID = fEID;

    CSxEIDRemapper( const GraphCSx & csr )
	: m_remap( csr.numEdges(), numa_allocation_interleaved() ),
	  m_dst( csr.numEdges(), numa_allocation_interleaved() ),
	  m_idx( csr.numVertices()+1, numa_allocation_interleaved() ),
	  m_csr( csr ) {
	VID n = csr.numVertices()+1;
	const EID * idx = m_csr.getIndex();
	parallel_for( VID v=0; v < n; ++v )
	    m_idx[v] = idx[v];
    }

    /**
     * set - Record that edge (s,d) has EID e in a remapped graph.
     */
    void set( VID s, VID d, EID e ) {
	EID w = __sync_fetch_and_add( &m_idx[s], 1 );
	m_remap[w] = e;
	m_dst[w] = d;
    }

    /**
     * finalize - sort data and perform checks
     */
    void finalize( const partitioner & part ) {
	const EID * oidx = m_csr.getIndex();
	const EID * nidx = m_idx.get();
	map_vertexL( part, [&]( VID v ) {
			       // timer tm;
			       // tm.start();
	       assert( nidx[v] == oidx[v+1] && "Setting all edges correctly" );
	       VID d = oidx[v+1]-oidx[v];
	       EID * eid = &m_remap[oidx[v]];
	       VID * dst = &m_dst[oidx[v]];
	       for( VID s=1; s < d; ++s ) {
		   EID ie = eid[s];
		   VID id = dst[s];
		   VID k = s;
		   while( k > 0 ) {
		       if( dst[k-1] < ie ) {
			   break;
		       } else {
			   eid[k] = eid[k-1];
			   dst[k] = dst[k-1];
			   --k;
		       }
		   }
		   eid[k] = ie;
		   dst[k] = id;

		   // std::cerr << "deg: " << d << " tm: " << tm.next() << "\n";
	       }
	   } );

	// Clean up temporaries
	m_dst.del();
	m_idx.del();
    }

    CSxEIDRetriever<VID,EID> create_retriever() const {
	return CSxEIDRetriever<VID,EID>( m_csr, m_remap );
    }

private:
    mmap_ptr<EID> m_remap;  //!< Remapped edge IDs
    mmap_ptr<VID> m_dst;    //!< Remapped destinations (temporary)
    mmap_ptr<EID> m_idx;    //!< Copy of index array (temporary)
    const GraphCSx & m_csr; //!< Reference CSR graph
};

#endif // GRAPTOR_GRAPH_EIDREMAPPER_H
