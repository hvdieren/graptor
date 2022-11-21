// -*- C++ -*-
#ifndef GRAPTOR_GRAPH_GRAPHGG_H
#define GRAPTOR_GRAPH_GRAPHGG_H

// This class is not up to date with the latest requirements around
// - frontier selection: getPullVLBound, select_traversal, etc
// - EID retrievers (which are being phased out)
// Note: a fixup has been made w/ little testing
//
// Should consider making this class a specialisation of GraphGGVEBO

template<typename COOType>
class GraphGG_tmpl {
    using EIDRemapper = NullEIDRemapper<VID,EID>;
    using EIDRetriever = IdempotentEIDRetriever<VID,EID>;

    GraphCSx * csr, * csc; // for transpose
    GraphCSx csr_act, csc_act;
    COOType * coo;
    partitioner part;
    EIDRetriever eid_retriever;

public:
    template<class vertex>
    GraphGG_tmpl( const wholeGraph<vertex> & WG,
		  int npart, bool balance_vertices )
	: csr( &csr_act ),
	  csc( std::is_same<vertex,symmetricVertex>::value
	       ? &csr_act : &csc_act ),
	  csr_act( WG, -1 ),
	  csc_act( std::is_same<vertex,symmetricVertex>::value ? 0 : WG.n,
		   std::is_same<vertex,symmetricVertex>::value ? 0 : WG.m, -1 ),
	  coo( new COOType[npart] ),
	  part( npart, WG.numVertices() ) {
	// Setup CSR and CSC
	if( !std::is_same<vertex,symmetricVertex>::value ) {
	    wholeGraph<vertex> * WGc = const_cast<wholeGraph<vertex> *>( &WG );
	    WGc->transpose();
	    csc_act.import( WG );
	    WGc->transpose();
	}
	
	// Decide what partition each vertex goes into
	if( balance_vertices )
	    partitionBalanceDestinations( WG, part ); 
	else
	    partitionBalanceEdges( WG, part ); 

	// Create COO partitions in parallel
	map_partitionL( part, [&]( int p ) {
		COOType & el = coo[p];
		new ( &el ) COOType( WG, part, p );
	    } );
    }
    GraphGG_tmpl( const GraphCSx & Gcsr,
		  int npart, bool balance_vertices )
	: csr( &csr_act ),
	  csc( Gcsr.isSymmetric() ? &csr_act : &csc_act ),
	  csr_act( Gcsr, -1 ),
	  csc_act( Gcsr.isSymmetric() ? 0 : Gcsr.numVertices(),
		   Gcsr.isSymmetric() ? 0 : Gcsr.numEdges(), -1 ),
	  coo( new COOType[npart] ),
	  part( npart, Gcsr.numVertices() ) {
	// Setup CSR and CSC
	std::cerr << "Transposing CSR...\n";
	const GraphCSx * csc_tmp_ptr = &csc_act;
	if( Gcsr.isSymmetric() )
	    csc_tmp_ptr = &Gcsr;
	else
	    csc_act.import_transpose( Gcsr );
	const GraphCSx & csc_tmp = *csc_tmp_ptr;
	
	// Decide what partition each vertex goes into
	if( balance_vertices )
	    partitionBalanceDestinations( Gcsr, part ); 
	else
	    partitionBalanceEdges( csc_tmp, part ); 

	// Create COO partitions in parallel
	map_partitionL( part, [&]( int p ) {
		COOType & el = coo[p];
		new ( &el ) COOType( csc_tmp, part, p );
	    } );

	for( unsigned p=0; p < npart; ++p ) {
	    std::cerr << "partition " << p
		      << ": s=" << part.start_of( p )
		      << ": e=" << part.end_of( p )
		      << ": nv=" << ( part.end_of( p ) - part.start_of( p ) )
		      << ": ne=" << coo[p].numEdges()
		      << "\n";
	}
    }


    void del() {
	csr_act.del();
	csc_act.del();
	for( int p=0; p < part.get_num_partitions(); ++p )
	    coo[p].del();
	delete[] coo;
	coo = nullptr;
	csr = nullptr;
	csc = nullptr;
    }

    void fragmentation() const {
	// TODO
    }

public:
    VID numVertices() const { return csr_act.numVertices(); }
    EID numEdges() const { return csr_act.numEdges(); }

    bool transposed() const { return csr != &csr_act; }
    void transpose() {
	std::swap( csc, csr );
	int np = part.get_num_partitions();
	for( int p=0; p < np; ++p )
	    coo[p].transpose();
    }

    const partitioner & get_partitioner() const { return part; }
    const EIDRetriever & get_eid_retriever() const { return eid_retriever; }

    const GraphCSx & getCSC() const { return *csc; }
    const GraphCSx & getCSR() const { return *csr; }
    const COOType & get_edge_list_partition( int p ) const { return coo[p]; }

    PartitionedCOOEIDRetriever<VID,EID>
    get_edge_list_eid_retriever( int p ) const {
	return PartitionedCOOEIDRetriever<VID,EID>( part.edge_start_of( p ) );
    }

    VID originalID( VID v ) const { return v; }
    VID remapID( VID v ) const { return v; }

    auto get_remapper() const { return RemapVertexIdempotent<VID>(); }

    VID getOutDegree( VID v ) const { return getCSR().getDegree( v ); }
    VID getOutNeighbor( VID v, VID pos ) const { return getCSR().getNeighbor( v, pos ); }
    const VID * getOutDegree() const { return getCSR().getDegree(); }

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

#if GG_ALWAYS_MEDIUM
#if OWNER_READS
	return graph_traversal_kind::gt_push;
#else
	return graph_traversal_kind::gt_pull;
#endif
#endif

	// threshold not configurable
	EID nactv = (EID)F.nActiveVertices();
	EID nacte = F.nActiveEdges();
	EID threshold2 = numEdges() / 2;
	if( nactv + nacte <= threshold2 ) {
#if OWNER_READS
	    return graph_traversal_kind::gt_push;
#else
	    return graph_traversal_kind::gt_pull;
#endif
	} else
	    return graph_traversal_kind::gt_ireg;
    }

    static constexpr bool is_privatized( graph_traversal_kind gtk ) {
#if OWNER_READS
	return gtk == graph_traversal_kind::gt_push;
#else
	return gtk == graph_traversal_kind::gt_pull;
#endif
    }

};

using GraphGG = GraphGG_tmpl<GraphCOO>;
using GraphGGIntlv = GraphGG_tmpl<GraphCOOIntlv>;

#endif // GRAPTOR_GRAPH_GRAPHGG_H
