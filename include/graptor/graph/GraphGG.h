// -*- C++ -*-
#ifndef GRAPHGRIND_GRAPH_GRAPHGG_H
#define GRAPHGRIND_GRAPH_GRAPHGG_H

// This class is not up to date with the latest requirements around
// - frontier selection: getPullVLBound, select_traversal, etc
// - EID retrievers (which are being phased out)
//
// Should consider making this class a specialisation of GraphGGVEBO

template<typename COOType>
class GraphGG_tmpl {
    GraphCSx * csr, * csc; // for transpose
    GraphCSx csr_act, csc_act;
    COOType * coo;
    partitioner part;

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

    const GraphCSx & getCSC() const { return *csc; }
    const GraphCSx & getCSR() const { return *csr; }
    const COOType & get_edge_list_partition( int p ) const { return coo[p]; }

    VID originalID( VID v ) const { return v; }
    VID remapID( VID v ) const { return v; }

    VID getOutDegree( VID v ) const { return getCSR().getDegree( v ); }

    // This graph only supports scalar processing in our system as destinations
    // are not laid out in a way that excludes conflicts across vector lanes
    // in the COO representation.
    // unsigned short getMaxVLCOO() const { return 1; }
    // unsigned short getMaxVLCSC() const { return VLUpperBound; }
    // static constexpr unsigned short getVLCOOBound() { return 1; }
    // static constexpr unsigned short getVLCSCBound() { return VLUpperBound; }

    static constexpr unsigned short getMaxVLCOO() { return 1; }
    static constexpr unsigned short getMaxVLCSC() { return 1; }
    static constexpr unsigned short getVLCOOBound() { return 1; }
    static constexpr unsigned short getVLCSCBound() { return 1; }

    // Really, we could be true/false or false/true, depending on frontier.
    // Main point is that a frontier bitmask is unlikely to be useful
    static constexpr bool getRndRd() { return true; }
    static constexpr bool getRndWr() { return true; }
};

using GraphGG = GraphGG_tmpl<GraphCOO>;
using GraphGGIntlv = GraphGG_tmpl<GraphCOOIntlv>;

#endif // GRAPHGRIND_GRAPH_GRAPHGG_H
