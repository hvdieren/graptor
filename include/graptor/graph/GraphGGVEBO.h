// -*- C++ -*-
#ifndef GRAPTOR_GRAPH_GRAPHGGVEBO_H
#define GRAPTOR_GRAPH_GRAPHGGVEBO_H

#include "graptor/mm.h"
#include "graptor/partitioner.h"
#include "graptor/frontier.h"
#include "graptor/graph/EIDRemapper.h"
#include "graptor/graph/VEBOReorder.h"
#include "graptor/graph/GraphCOO.h"

class GraphGGVEBO {
#if GGVEBO_COOINTLV
    using ThisGraphCOO = GraphCOOIntlv;
#else
    using ThisGraphCOO = GraphCOO;
#endif
    // using EIDRetriever = CSxEIDRetriever<VID,EID>;
    // using EIDRemapper = CSxEIDRemapper<VID,EID>;
    using EIDRemapper = NullEIDRemapper<VID,EID>;
    using EIDRetriever = IdempotentEIDRetriever<VID,EID>;

    GraphCSx * csr, * csc; // for transpose
    GraphCSx csr_act, csc_act;
    ThisGraphCOO * coo;
    partitioner part;
    VEBOReorder remap;
    EIDRetriever eid_retriever;

public:
    template<class vertex>
    GraphGGVEBO( const wholeGraph<vertex> & WG, int npart )
	: csr( &csr_act ),
	  csc( std::is_same<vertex,symmetricVertex>::value
	       ? &csr_act : &csc_act ),
	  csr_act( WG.n, WG.m, -1, WG.isSymmetric() ),
	  csc_act( std::is_same<vertex,symmetricVertex>::value ? 0 : WG.n,
		   std::is_same<vertex,symmetricVertex>::value ? 0 : WG.m, -1 ),
	  coo( new ThisGraphCOO[npart] ),
	  part( npart, WG.numVertices() ) {

#if OWNER_READS
	assert( WG.isSymmetric() && "OWNER_READS requires symmetric graphs" );
#endif

	// Setup temporary CSC, try to be space-efficient
	GraphCSx & csc_tmp = csr_act;
	if( std::is_same<vertex,symmetricVertex>::value )
	    csc_tmp.import( WG );
	else {
	    wholeGraph<vertex> * WGc = const_cast<wholeGraph<vertex> *>( &WG );
	    WGc->transpose();
	    csc_tmp.import( WG );
	    WGc->transpose();
	}

	// Calculate remapping table
	remap = VEBOReorder( csc_tmp, part );

	// Setup CSR
	std::cerr << "Remapping CSR...\n";
	csr_act.import( WG, remap.maps() );

	// Setup CSC
	std::cerr << "Remapping CSC...\n";
	if( !std::is_same<vertex,symmetricVertex>::value ) {
	    wholeGraph<vertex> * WGc = const_cast<wholeGraph<vertex> *>( &WG );
	    WGc->transpose();
	    csc_act.import( WG, remap.maps() );
	    WGc->transpose();
	}
	
	// Create COO partitions in parallel
	std::cerr << "Creating and remapping COO partitions...\n";
	map_partitionL( part, [&]( int p ) {
#if GGVEBO_COO_CSC_ORDER
		createCOOPartitionFromCSC( WG.numVertices(),
					   p, part.numa_node_of( p ) );
#else
		createCOOPartitionFromCSR( WG.numVertices(),
					   p, p / part.numa_node_of( p ) );
#endif
	    } );
	std::cerr << "GraphGGVEBO loaded\n";
    }

    GraphGGVEBO( const GraphCSx & Gcsr, int npart )
	: csr( &csr_act ),
	  csc( &csc_act ),
	  csr_act( Gcsr.numVertices(), Gcsr.numEdges(), -1, Gcsr.isSymmetric(),
		   Gcsr.getWeights() != nullptr ),
	  csc_act( Gcsr.numVertices(), Gcsr.numEdges(), -1, Gcsr.isSymmetric(),
		   Gcsr.getWeights() != nullptr  ),
	  coo( new ThisGraphCOO[npart] ),
	  part( npart, Gcsr.numVertices() ) {

#if OWNER_READS
	assert( Gcsr.isSymmetric() && "OWNER_READS requires symmetric graphs" );
#endif

	// Setup temporary CSC, try to be space-efficient
	std::cerr << "Transposing CSR...\n";
	const GraphCSx * csc_tmp_ptr = &csr_act;
	if( Gcsr.isSymmetric() )
	    csc_tmp_ptr = &Gcsr;
	else
	    csr_act.import_transpose( Gcsr );
	const GraphCSx & csc_tmp = *csc_tmp_ptr;

	// Calculate remapping table
	remap = VEBOReorder( csc_tmp, part );

	// Setup CSC
	std::cerr << "Remapping CSC...\n";
	csc_act.import( csc_tmp, remap.maps() );
	
	// Setup CSR (overwrites csc_tmp)
	// TODO: if symmetric, we don't need a different copy for CSR and CSC
	std::cerr << "Remapping CSR...\n";
	csr_act.import( Gcsr, remap.maps() );

	// Create COO partitions in parallel
#if GG_ALWAYS_MEDIUM
	std::cerr << "Skipping COO partitions...\n";

	// Cleanup / indicate non-existant
	delete[] coo;
	coo = nullptr;

	// set up edge partitioner - determines how to allocate edge properties
	EID * counts = part.edge_starts();
	EID ne = 0;
	EID * idx = csc_act.getIndex();
	for( unsigned short p=0; p < npart; ++p ) {
	    VID vs = part.start_of( p );
	    VID ve = part.end_of( p );
	    EID pe = idx[ve] - idx[vs];
	    counts[p] = ne;
	    ne += pe;
	}
	counts[npart] = ne;
#else
	// Setup EID remapper based on remapped (padded) vertices
	EIDRemapper eid_remapper( csr_act );

	std::cerr << "Creating and remapping COO partitions...\n";
#if 0
	map_partitionL( part, [&]( int p ) {
#if GGVEBO_COO_CSC_ORDER
		createCOOPartitionFromCSC( Gcsr.numVertices(),
					   p, part.numa_node_of( p ),
					   eid_remapper );
#else
		createCOOPartitionFromCSR( Gcsr.numVertices(),
					   p, part.numa_node_of( p ),
					   eid_remapper );
#endif
	    } );
	eid_remapper.finalize( part );
	// eid_retriever = eid_remapper.create_retriever();
#else
	createCOOPartitionsFromCSR();
#endif

	// set up edge partitioner - determines how to allocate edge properties
	EID * counts = part.edge_starts();
	EID ne = 0;
	for( unsigned short p=0; p < npart; ++p ) {
	    counts[p] = ne;
	    ne += coo[p].numEdges();
	}
	counts[npart] = ne;
#endif // GG_ALWAYS_MEDIUM

	std::cerr << "GraphGGVEBO loaded\n";
    }


    void del() {
	remap.del();
	csr_act.del();
	csc_act.del();
	if( coo ) {
	    for( int p=0; p < part.get_num_partitions(); ++p )
		coo[p].del();
	    delete[] coo;
	}
	coo = nullptr;
	csr = nullptr;
	csc = nullptr;
    }

    void fragmentation() const {
	std::cerr << "GraphGGVEBO:\n";
	getCSR().fragmentation();
	getCSC().fragmentation();
	std::cerr << "COO partitions:\ntotal-size: "
		  << ( numEdges()*2*sizeof(VID) ) << "\n";
    }

public:
    VID numVertices() const { return csr_act.numVertices(); }
    EID numEdges() const { return csr_act.numEdges(); }

    bool isSymmetric() const { return csr_act.isSymmetric(); }

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
    const ThisGraphCOO & get_edge_list_partition( int p ) const {
	return coo[p];
    }
    PartitionedCOOEIDRetriever<VID,EID>
    get_edge_list_eid_retriever( int p ) const {
#if GGVEBO_COO_CSC_ORDER
	static_assert( false,
		       "EID retrievers are incorrect for COO in CSC ORDER" );
#endif 
	return PartitionedCOOEIDRetriever<VID,EID>( part.edge_start_of( p ) );
    }

    VID originalID( VID v ) const { return remap.originalID( v ); }
    VID remapID( VID v ) const { return remap.remapID( v ); }

    auto get_remapper() const { return remap.remapper(); }

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

    VID getOutDegree( VID v ) const { return getCSR().getDegree( v ); }
    VID getOutNeighbor( VID v, VID pos ) const { return getCSR().getNeighbor( v, pos ); }

    const VID * getOutDegree() const { return getCSR().getDegree(); }

public:
    GraphCSx::vertex_iterator part_vertex_begin( const partitioner & part, unsigned p ) const {
	return csc->vertex_begin( part.start_of( p ) );
    }
    GraphCSx::vertex_iterator part_vertex_end( const partitioner & part, unsigned p ) const {
	return csc->vertex_begin( part.end_of( p ) );
    }

private:
    void createCOOPartitionFromCSC( VID n, int p, int allocation,
				    EIDRemapper & eid_remapper ) {
	// In w = remap.first[v], v is the new vertex ID, w is the old one
	// In v = remap.second[w], v is the new vertex ID, w is the old one

	// Range of destination vertices to include
	VID rangeLow = part.start_of(p);
	VID rangeHi = part.start_of(p+1);
	
	// Short-hands
	EID *idx = csc->getIndex();
	VID *edge = csc->getEdges();

	EID num_edges = idx[rangeHi] - idx[rangeLow];

	ThisGraphCOO & el = coo[p];
	new ( &el ) GraphCOO( n, num_edges, allocation );
	
        EID k = 0;
        for( VID v=rangeLow; v < rangeHi; v++ ) {
	    VID deg = idx[v+1] - idx[v];
            for( VID j=0; j < deg; ++j ) {
		el.setEdge( k, edge[idx[v]+j], v );
		eid_remapper.set( edge[idx[v]+j], v, idx[rangeLow]+k );
#ifdef WEIGHTED
		wgh[k] = ...;
#endif
		k++;
	    }
	}
	assert( k == num_edges );

	// Edges are now stored in CSC traversal order
#if EDGES_HILBERT
	el.hilbert_sort();
#else
	el.CSR_sort();
#endif
    }
    void createCOOPartitionFromCSR( VID n, int p, int allocation,
				    EIDRemapper & eid_remapper ) {
	// In w = remap.first[v], v is the new vertex ID, w is the old one
	// In v = remap.second[w], v is the new vertex ID, w is the old one

	// Range of destination vertices to include
	VID rangeLow = part.start_of(p);
	VID rangeHi = part.end_of(p); // part.start_of(p+1);
	
	// Short-hands
	EID *csc_idx = csc->getIndex();
	EID *idx = csr->getIndex();
	VID *edge = csr->getEdges();

	EID num_edges = csc_idx[rangeHi] - csc_idx[rangeLow];

	ThisGraphCOO & el = coo[p];
	new ( &el ) ThisGraphCOO( n, num_edges, allocation, csc->getWeights() );

	// std::cerr << "COO from CSR: n=" << n << " p=" << p
	// << " alloc=" << allocation << " nE=" << num_edges << "\n";

	// Convenience defs
	const bool has_weights = el.getWeights() != nullptr;
	float * const Tweights = has_weights ? el.getWeights() : nullptr;
	assert( ( ( csr->getWeights() != nullptr ) || !has_weights )
		&& "expecting weights in graph" );
	const float * const Gweights = csr->getWeights()
	    ? csr->getWeights()->get() : nullptr;

        EID k = 0;
        for( VID v=0; v < n; v++ ) {
	    VID deg = idx[v+1] - idx[v];
            for( VID j=0; j < deg; ++j ) {
		VID d = edge[idx[v]+j];
		if( rangeLow <= d && d < rangeHi ) {
		    el.setEdge( k, v, d );
		    eid_remapper.set( v, d, csc_idx[rangeLow]+k );
#ifdef WEIGHTED
		    wgh[k] = ...;
#endif
		    if( has_weights )
			Tweights[k] = Gweights[idx[v]+j];
		    k++;
		}
	    }
	}
	assert( k == num_edges );

	// Edges are now stored in CSR traversal order
#if EDGES_HILBERT
	el.hilbert_sort();
#endif
    }
    void createCOOPartitionsFromCSR() {
	// In w = remap.first[v], v is the new vertex ID, w is the old one
	// In v = remap.second[w], v is the new vertex ID, w is the old one

	// Short-hands
	VID n = csr->numVertices();
	EID *csc_idx = csc->getIndex();
	EID *idx = csr->getIndex();
	VID *edge = csr->getEdges();
	unsigned P = part.get_num_partitions();

	// 1. Determine how many edges each partition will receive. Can be
	//    inferred from partition boundaries (sources) and CSC index
	//    structure, and create COO data structures
	for( unsigned p=0; p < P; ++p ) {
	    // Range of destination vertices to include
	    VID rangeLow = part.start_of(p);
	    VID rangeHi = part.end_of(p);
	    EID n_edges = csc_idx[rangeHi] - csc_idx[rangeLow];

	    // Create COO data structures
	    new ( &coo[p] )
		ThisGraphCOO( n, n_edges, part.numa_node_of( p ),
			      csc->getWeights() );
	}
	
	// 3. Considering a parallel assignment of edges based on the
	//    partitioner, determine how many edges will be sent to each
	//    COO partition from each CSR partition (P x P info).
	//    Align counters on block boundaries to minimise false sharing.
	constexpr unsigned wP = 64 / sizeof(EID);
	unsigned wPm = P % wP;
	unsigned L = wPm == 0 ? P : ( P + wP - wPm );
	EID * p2p = new EID[L*P]();
	parallel_for( unsigned p=0; p < P; ++p ) {
	    VID vs = part.start_of(p);
	    VID ve = part.end_of(p);
	    EID es = idx[vs];
	    EID ee = idx[ve];
	    for( EID e=es; e < ee; ++e ) {
		unsigned q = part.part_of( edge[e] );
		p2p[L*p+q]++;
	    }
	}

	// Scan across all p for each q to determine the insertion points
	// in the edge lists.
	parallel_for( unsigned q=0; q < P; ++q ) {
	    EID s = 0;
	    for( unsigned p=0; p < P; ++p ) {
		EID t = p2p[L*p+q];
		p2p[L*p+q] = s;
		s += t;
	    }

	    // Correctness check
	    assert( s == coo[q].numEdges() );
	}

	// 4. Assign edges in parallel
	// Convenience defs
	const float * const Gweights = csr->getWeights()
	    ? csr->getWeights()->get() : nullptr;
	if( Gweights ) {
	    parallel_for( unsigned p=0; p < P; ++p ) {
		VID vs = part.start_of(p);
		VID ve = part.end_of(p);
		for( VID v=vs; v < ve; ++v ) {
		    EID es = idx[v];
		    EID ee = idx[v+1];
		    for( EID e=es; e < ee; ++e ) {
			unsigned q = part.part_of( edge[e] );
			EID k = p2p[L*p+q]++;
			coo[q].setEdge( k, v, edge[e], Gweights[e] );
		    }
		}
	    }
	} else {
	    parallel_for( unsigned p=0; p < P; ++p ) {
		VID vs = part.start_of(p);
		VID ve = part.end_of(p);
		for( VID v=vs; v < ve; ++v ) {
		    EID es = idx[v];
		    EID ee = idx[v+1];
		    for( EID e=es; e < ee; ++e ) {
			unsigned q = part.part_of( edge[e] );
			EID k = p2p[L*p+q]++;
			coo[q].setEdge( k, v, edge[e] );
		    }
		}
	    }
	}

	// Edges are now stored in CSR traversal order
	delete[] p2p;
    }
};


#endif // GRAPTOR_GRAPH_GRAPHGGVEBO_H

