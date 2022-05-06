// -*- c++ -*-
#ifndef GRAPTOR_GRAPH_CGRAPHVEBOGRAPTOR_H
#define GRAPTOR_GRAPH_CGRAPHVEBOGRAPTOR_H

#ifndef GRAPTOR
#define GRAPTOR 1
#endif // GRAPTOR

#ifndef GRAPTOR_CACHED
#define GRAPTOR_CACHED 1
#endif // GRAPTOR_CACHED

#ifndef GRAPTOR_CSC
#define GRAPTOR_CSC 1
#endif // GRAPTOR_CSC

#include "graptor/graph/CGraphCSx.h"
#include "graptor/graph/VEBOReorder.h"
#include "graptor/graph/CGraphCSxSIMDDegree.h"
#include "graptor/graph/CGraphCSxSIMDDegreeDelta.h"
#include "graptor/graph/CGraphCSxSIMDDegreeMixed.h"
#include "graptor/graph/CGraphCSRSIMDDegreeMixed.h"
#include "graptor/graph/CGraphCSxSIMDDegreeDeltaMixed.h"
#include "graptor/frontier.h"

enum graptor_mode_t {
    gm_csc_vreduce_not_cached = 0,
    gm_csc_vreduce_cached = 1,
    gm_csc_datapar_not_cached = 2,
    gm_csc_datapar_cached = 3,
    gm_csr_vpush_not_cached = 4,
    gm_csr_vpush_cached = 5,
    gm_csr_datapar_not_cached = 6,
    gm_csr_datapar_cached = 7
};

#define GRAPTOR_MODE (graptor_mode_t) \
    ( (GRAPTOR_CACHED == 1 ? 1 : 0)   \
      | (GRAPTOR == 1 ? 0 : 2)	      \
      | (GRAPTOR_CSC == 1 ? 0 : 4) )

struct GraptorEIDRetriever {
    mmap_ptr<EID> edge_offset;
    unsigned short logVL;

    void del() {
	edge_offset.del();
    }

    void init( VID nv, unsigned short VL_ ) {
	edge_offset.allocate( nv/(VID)VL_, numa_allocation_interleaved() );
	logVL = rt_ilog2( VL_ );
    }

    EID get_edge_eid( VID src, VID ngh ) const {
	VID idx = src >> logVL;
	VID off = src & ((VID(1)<<logVL)-1);
	return edge_offset[idx] + off + ( ngh << logVL );
    }
};

//template<typename lVID, typename lEID>
template<graptor_mode_t Mode_>
class GraphVEBOGraptor {
    static constexpr graptor_mode_t Mode = Mode_;
    
    // using GraphPartitionType = GraphCSxSIMDDegreeDeltaMixed;
    // using GraphPartitionType = GraphCSxSIMDDegree;
#if GRAPTOR_CSC
    using GraphPartitionType = GraphCSxSIMDDegreeMixed;
#else
    using GraphPartitionType = GraphCSRSIMDDegreeMixed;
#endif

    GraphCSx csr;
    GraphPartitionType * csc;
#if GRAPTOR_CSC
    VEBOReorder remap;
#else
#if VEBO_DISABLE
    VEBOReorderIdempotent<VID,EID> remap;
#else
    VEBOReorderSIMD<VID,EID> remap;
#endif
#endif
    partitioner part;
    unsigned short minVL, maxVL;
    VID maxdiff;
    GraptorEIDRetriever eid_retriever;

public:
    GraphVEBOGraptor( const GraphCSx & Gcsr,
		      int npart,
		      unsigned short minVL_,
		      unsigned short maxVL_,
		      VID maxdiff_ )
	: csc( new GraphPartitionType[npart] ),
	  csr(), // csr( Gcsr.numVertices(), Gcsr.numEdges(), -1 ),
	  part( npart, Gcsr.numVertices() ),
	  minVL( minVL_ ),
	  maxVL( maxVL_ ),
	  maxdiff( maxdiff_ ) {

	// Setup temporary CSC, try to be space-efficient
	std::cerr << "Transposing CSR...\n";
	GraphCSx * csc_tmp;
	if( Gcsr.isSymmetric() ) {
	    // Symmetric, so identical to CSR
	    csc_tmp = const_cast<GraphCSx *>( &Gcsr );
	} else {
	    csc_tmp = new GraphCSx( Gcsr.numVertices(), Gcsr.numEdges(), -1 );
	    csc_tmp->import_transpose( Gcsr );
	}

#if GRAPTOR_CSC
	// Calculate remapping table. Do not use the feature to interleave
	// subsequent destinations over per-lane partitions.
	std::cerr << "VEBO...\n";
	remap = VEBOReorder( *csc_tmp, part, 1, false, maxVL );

	std::cerr << "Highest-degree vertex: deg=" << csc_tmp->max_degree()
		  << "...\n";
	csc_tmp->setMaxDegreeVertex( csc_tmp->findHighestDegreeVertex() );

	// Setup CSR
	std::cerr << "Reorder CSR...\n";
	new (&csr) GraphCSx( part.get_num_elements(), Gcsr.numEdges(), -1 );
	csr.import_expand( Gcsr, part, remap.remapper() );

	// Setup CSC partitions
	std::cerr << "Graptor: "
		  << " n=" << Gcsr.numVertices()
		  << " e=" << Gcsr.numEdges()
		  << "\n";
	map_partitionL( part, [&]( int p ) {
		assert( part.start_of(p) % maxVL == 0 );
		new (&csc[p]) GraphPartitionType();
		VID lo = part.start_of(p);
		VID hi = part.end_of(p);
		// if( p == npart-1 && hi % maxVL != 0 )
		// hi += maxVL - ( hi % maxVL );
		csc[p].import( *csc_tmp, lo, hi, part.get_num_elements(),
			       maxVL, remap.remapper(),
			       part.numa_node_of( p ) );

/*
		std::cerr << "Graptor part " << p
			  << " s=" << part.start_of(p)
			  << " e=" << part.end_of(p)
			  << " nv=" << csc[p].numSIMDVertices()
			  << " ne=" << csc[p].numSIMDEdges()
			  << "\n";
*/
	    } );
#else
	// Calculate remapping table.
#if VEBO_DISABLE
	remap = VEBOReorderIdempotent<VID,EID>();
	partitionBalanceEdges( *csc_tmp,
			       gtraits_getoutdegree<GraphCSx>( *csc_tmp ),
			       part, maxVL );

	// Setup CSR
	std::cerr << "Remapping CSR...\n";
	new (&csr) GraphCSx( part.get_num_elements(), Gcsr.numEdges(), -1 );
	csr.import_expand( Gcsr, part, remap.remapper() );

	VID hideg = csc_tmp->findHighestDegreeVertex();
	csr.setMaxDegreeVertex( remap.remappedID( hideg ) );
	csc_tmp->setMaxDegreeVertex( hideg );
#else
	// remap = VEBOReorder( *csc_tmp, part, maxVL, false, 1 ); // true, 1 );
	remap = VEBOReorderSIMD<VID,EID>( *csc_tmp, part, 1, false, maxVL ); // maxVL, true, maxVL );

	// Setup CSR
	new (&csr) GraphCSx( part.get_num_elements(), Gcsr.numEdges(), -1 );
	csr.import_expand( Gcsr, part, remap.remapper() );

	csr.setMaxDegreeVertex( 0 );
	csc_tmp->setMaxDegreeVertex( remap.originalID( 0 ) );
#endif

#if GRAPTOR == 1 // VPush
	// Clean up intermediates - no longer needed
	if( !Gcsr.isSymmetric() && csc_tmp ) {
	    csc_tmp->del();
	    delete csc_tmp;
	    csc_tmp = nullptr;
	}

	// Setup CSR partitions
	std::cerr << "Graptor: "
		  << " n=" << Gcsr.numVertices()
		  << " e=" << Gcsr.numEdges()
		  << "\n";

	// Pre-calculate space requirements for all partitions
	std::cerr << "Calculating space requirements...\n";
	std::pair<const EID *, const VID *> sz
	    = GraphPartitionType::space_from_csr( csr, part, maxVL );

	// Create partitions
	std::cerr << "Creating VPush partitions...\n";
	map_partitionL( part, [&]( int p ) {
		assert( part.start_of(p) % maxVL == 0 );
		new (&csc[p]) GraphPartitionType();
		csc[p].import_csr( csr, part, p, maxVL,
				   sz.first[p], sz.second[p],
				   numa_allocation_local( part.numa_node_of( p ) ) );
	    } );

	delete[] sz.first;
	delete[] sz.second;
	
#else
	// Setup CSR partitions
	std::cerr << "Graptor: "
		  << " n=" << Gcsr.numVertices()
		  << " e=" << Gcsr.numEdges()
		  << "\n";
	map_partitionL( part, [&]( int p ) {
		assert( part.start_of(p) % maxVL == 0 );
		new (&csc[p]) GraphPartitionType();
		VID lo = part.start_of(p);
		// VID hi = part.end_of(p);
		VID hi = part.start_of(p+1);
		// if( p == npart-1 && hi % maxVL != 0 )
		// hi += maxVL - ( hi % maxVL );
		csc[p].import( *csc_tmp, part, p,
			       maxVL, remap.remapper(),
			       part.numa_node_of( p ) );

/*
		std::cerr << "Graptor part " << p
			  << " s=" << part.start_of(p)
			  << " e=" << part.end_of(p)
			  << " nv=" << csc[p].numSIMDVertices()
			  << " ne=" << csc[p].numSIMDEdges()
			  << "\n";
*/
	    } );
#endif
#endif

	// set up edge partitioner - determines how to allocate edge properties
	EID * counts = part.edge_starts();
	EID ne = 0;
	for( unsigned short p=0; p < npart; ++p ) {
	    counts[p] = ne;
	    ne += csc[p].numSIMDEdges();
	}
	counts[npart] = ne;
	
	// Setup EID retriever - map vertices to EID index
	// WARNING: UNTESTED
	eid_retriever.init( part.get_vertex_range(), maxVL );
	map_partitionL( part, [&]( int p ) {
		VID lo = part.start_of(p);
		VID hi = part.end_of(p);

		EID eid = 0;
		for( int pp=0; pp < p; ++pp )
		    eid += csc[p].numSIMDEdges();

		const EID * starts = csc[p].getStarts();

		for( VID v=lo; v < hi; v += maxVL ) {
		    eid_retriever.edge_offset[v/maxVL]
			= starts[(v-lo)/maxVL] + part.edge_start_of( p );
		}
	  } );

	// Clean up intermediates
	if( !Gcsr.isSymmetric() && csc_tmp ) {
	    csc_tmp->del();
	    delete csc_tmp;
	}
    }
    void del() {
	csr.del();
	for( int p=0; p < part.get_num_partitions(); ++p )
	    csc[p].del();
	delete[] csc;
	csc = nullptr;
	remap.del();
	eid_retriever.del();
    }

public:
    void fragmentation() const {
	std::cerr << "GraphVEBOGraptor: [TODO]\n";
	for( int p=0; p < part.get_num_partitions(); ++p ) {
	    std::cerr << "\t" << p << "\t"
		      << " nv=" << csc[p].numSIMDVertices()
		      << " mv=" << csc[p].numSIMDEdges()
		      << " mv2=" << csc[p].numSIMDEdgesDeg2()
		      << " mv1=" << csc[p].numSIMDEdgesDeg1()
		      << " inactd1=" << csc[p].numSIMDEdgesInvDelta1()
		      << " inactdpar=" << csc[p].numSIMDEdgesInvDeltaPar()
#if GRAPTOR_CSR_INDIR
		      << " indir_nnz=" << csc[p].getRedirNNZ()
#endif
		      << "\n";
	}
    }
    
public:
#if 0
    template<unsigned short W>
    frontier createValidFrontier() const {
	frontier f = frontier::template dense<W>( part );
	logical<W> *d = f.getDenseL<W>();
	VID norig = part.get_num_vertices();
	map_vertexL( part, [&]( VID v ) {
		d[v] = logical<W>::get_val( remap.originalID( v ) < norig ); } );
	f.setActiveCounts( norig, (EID)0 );
	return f;
    }
    bool isValid( VID v ) const {
	return remap.originalID( v ) < part.get_num_vertices();
    }
#endif
    
    VID numVertices() const { return csr.numVertices(); }
    EID numEdges() const { return csr.numEdges(); }

    auto get_remapper() const { return remap.remapper(); }

    bool transposed() const { return false; }
    void transpose() { assert( 0 && "Not supported" ); }

    const GraphCSx & getCSR() const { return csr; }
    const GraphVEBOGraptor<Mode> & getCSC() const { return *this; }
    const GraphPartitionType & getCSC( int p ) const { return csc[p]; }

    const VID * getOutDegree() const { return getCSR().getDegree(); }

    const partitioner & get_partitioner() const { return part; }
    const GraptorEIDRetriever & get_eid_retriever() const {
	return eid_retriever;
    }

    VID originalID( VID v ) const { return remap.originalID( v ); }
    VID remapID( VID v ) const { return remap.remapID( v ); }

    unsigned short getMaxVL() const { return maxVL; }
    static constexpr unsigned short getVLCOOBound() { return VLUpperBound; }
    static constexpr unsigned short getVLCSCBound() { return VLUpperBound; }
    static constexpr unsigned short getPullVLBound() { return VLUpperBound; }
    static constexpr unsigned short getPushVLBound() { return VLUpperBound; }
    static constexpr unsigned short getIRegVLBound() { return VLUpperBound; }

    graph_traversal_kind select_traversal(
	bool fsrc_strong,
	bool fdst_strong,
	bool adst_strong,
	bool record_strong,
	frontier F,
	bool is_sparse ) const {

	if( is_sparse )
	    return graph_traversal_kind::gt_sparse;
	else {
#if GRAPTOR_CSC
	    return graph_traversal_kind::gt_pull;
#else
	    return graph_traversal_kind::gt_push;
#endif
	}
    }

    static constexpr bool is_privatized( graph_traversal_kind gtk ) {
	return gtk == graph_traversal_kind::gt_pull;
    }

#if GRAPTOR_CSC
    static constexpr bool getRndRd() { return true; }
    static constexpr bool getRndWr() { return false; }
#else
#if GRAPTOR_CSR_INDIR
    static constexpr bool getRndRd() { return true; }
    static constexpr bool getRndWr() { return true; }
#else
    static constexpr bool getRndRd() { return false; }
    static constexpr bool getRndWr() { return true; }
#endif
#endif

    VID getOutDegree( VID v ) const { return getCSR().getDegree( v ); }

    bool isSymmetric() const { return getCSR().isSymmetric(); }
};

#endif // GRAPTOR_GRAPH_CGRAPHVEBOGRAPTOR_H
