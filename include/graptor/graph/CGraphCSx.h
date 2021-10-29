// -*- c++ -*-
#ifndef GRAPTOR_GRAPH_CGRAPHCSX_H
#define GRAPTOR_GRAPH_CGRAPHCSX_H

template<typename lVID, typename lEID>
class CGraphCSx {
    lVID n;
    lEID m;
    mmap_ptr<lEID> index;
    mmap_ptr<lVID> edges;
    bool symmetric;

public:
    CGraphCSx() { }
    CGraphCSx( const std::string & infile, int allocation = -1,
	       bool _symmetric = false ) 
	: symmetric( _symmetric ) {
	if( allocation == -1 ) {
	    numa_allocation_interleaved alloc;
	    readFromBinaryFile( infile, alloc );
	} else {
	    numa_allocation_local alloc( allocation );
	    readFromBinaryFile( infile, alloc );
	}
    }
    CGraphCSx( const std::string & infile, const numa_allocation & allocation,
	       bool _symmetric )
	: symmetric( _symmetric ) {
	readFromBinaryFile( infile, allocation );
    }
    CGraphCSx( lVID n_, lEID m_, int allocation )
	: n( n_ ), m( m_ ), symmetric( false ) {
	if( allocation == -1 )
	    allocateInterleaved();
	else
	    allocateLocal( allocation );
    }
    void del() {
	index.del();
	edges.del();
    }
    template<typename llVID, typename llEID>
    CGraphCSx( const CGraphCSx<llVID,llEID> & WG, int allocation )
	: n( WG.numVertices() ), m( WG.numEdges() ),
	  symmetric( WG.isSymmetric() ) {
	if( allocation == -1 )
	    allocateInterleaved();
	else
	    allocateLocal( allocation );
	import( WG );
    }
    template<typename llVID, typename llEID>
    void import( const CGraphCSx<llVID,llEID> & WG ) {
	assert( n == WG.numVertices() && m == WG.numEdges() );

	llVID vfail = 0;
	parallel_for( llVID v=0; v <= WG.numVertices(); ++v ) {
	    lVID lv = v;
	    index[lv] = WG.getIndex()[lv];
	    if( llVID(index[lv]) != WG.getIndex()[lv] )
		__sync_fetch_and_add( &vfail, 1 );
	}

	llEID efail = 0;
	parallel_for( llEID e=0; e < WG.numEdges(); ++e ) {
	    lEID le = e;
	    edges[le] = WG.getEdges()[le];
	    if( llEID(edges[le]) != WG.getEdges()[le] )
		__sync_fetch_and_add( &efail, 1 );
	}

	if( vfail != 0 || efail != 0 ) {
	    std::cerr << "CGraphCSx<> copy: "
		      << vfail << " VID conversion failures; "
		      << efail << " EID conversion failures\n";
	}
    }

    void writeToBinaryFile( const std::string & ofile ) {
	ofstream file( ofile, ios::out | ios::trunc | ios::binary );
	uint64_t header[8] = {
	    2, // version
	    1,
	    (uint64_t)n, // num nodes
	    (uint64_t)m, // num edges
	    sizeof(lVID), // sizeof(lVID)
	    sizeof(lEID), // sizeof(lEID)
	    0, // unused
	    0, // unused
	};

	file.write( (const char *)header, sizeof(header) );
	file.write( (const char *)&index[0], sizeof(index[0])*n );
	file.write( (const char *)&edges[0], sizeof(edges[0])*m );

	file.close();
    }

    void readFromBinaryFile( const std::string & ifile,
			     const numa_allocation & alloc ) {
	// We messed up with the organization of the file. Ideally we should
	// just mmap the contents in memory, however, for vectorization purposes
	// we need alignment in some places. This has not been taken into
	// account and may affect us adversely. Moreover, the n+1-th index
	// value has not been written to disk and we need a place to store it

	// TODO: extend mmap_ptr to mmap a file using specified allocation
	//       policy. Will avoid current copy of data
	std::cerr << "Reading (using mmap) file " << ifile << std::endl;
	int fd;

	if( (fd = open( ifile.c_str(), O_RDONLY )) < 0 ) {
	    std::cerr << "Cannot open file '" << ifile << "': "
		      << strerror( errno ) << "\n";
	    exit( 1 );
	}
	off_t len = lseek( fd, 0, SEEK_END );
	if( len == off_t(-1) ) {
	    std::cerr << "Cannot lseek to end of file '" << ifile << "': "
		      << strerror( errno ) << "\n";
	    exit( 1 );
	}
	const char * data = (const char *)mmap( 0, len, PROT_READ,
						MAP_SHARED, fd, 0 );
	if( data == (const char *)-1 ) {
	    std::cerr << "Cannot mmap file '" << ifile << "' read-only: "
		      << strerror( errno ) << "\n";
	    exit( 1 );
	}

	const uint64_t * header = reinterpret_cast<const uint64_t*>( &data[0] );
	if( header[0] != 2 ) {
	    std::cerr << "Only accepting version 2 files\n";
	    exit( 1 );
	}
	n = header[2];
	m = header[3];
	assert( sizeof(lVID) == header[4] );
	assert( sizeof(lEID) == header[5] );

	allocate( alloc );

	const lEID *index_p = reinterpret_cast<const lEID *>(
	    data+sizeof(uint64_t)*8 );
	parallel_for( lVID v=0; v < n; ++v )
	    index[v] = index_p[v];
	index[n] = m;

	const lVID *edges_p = reinterpret_cast<const lVID *>(
	    data+sizeof(uint64_t)*8+n*sizeof(lEID) );
	parallel_for( lEID e=0; e < m; ++e )
	    edges[e] = edges_p[e];

	close( fd );
	std::cerr << "Reading file done" << std::endl;
    }

public:
    void fragmentation() const {
	std::cerr << "CGraphCSx:\ntotal-size: "
		  << ( numVertices()*sizeof(lEID)+numEdges()*sizeof(lVID) )
		  << "\n";
    }
    
public:
    lEID *getIndex() { return index.get(); }
    const lEID *getIndex() const { return index.get(); }
    lVID *getEdges() { return edges.get(); }
    const lVID *getEdges() const { return edges.get(); }

    lVID numVertices() const { return n; }
    lEID numEdges() const { return m; }

    lVID getDegree( lVID v ) const { return index[v+1] - index[v]; }

    bool isSymmetric() const { return symmetric; }

private:
    void allocate( const numa_allocation & alloc ) {
	// Note: only interleaved and local supported
	index.allocate( n+1, alloc );
	edges.allocate( m, alloc );
    }

    void allocateInterleaved() {
	allocate( numa_allocation_interleaved() );
	// index.Interleave_allocate( n+1 );
	// edges.Interleave_allocate( m );
    }
    void allocateLocal( int numa_node ) {
	allocate( numa_allocation_local( numa_node ) );
	// index.local_allocate( n+1, numa_node );
	// edges.local_allocate( m, numa_node );
    }
};

#endif // GRAPTOR_GRAPH_CGRAPHCSX_H
