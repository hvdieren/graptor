// -*- C++ -*-
#ifndef GRAPHGRIND_GRAPH_GRAPHCSX_H
#define GRAPHGRIND_GRAPH_GRAPHCSX_H

#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <string>
#include <algorithm>

#include "graptor/itraits.h"
#include "graptor/mm.h"
#include "graptor/mm/mm.h"
#include "graptor/graph/gtraits.h"
#include "graptor/graph/remap.h"

enum class graph_traversal_kind {
    gt_sparse = 0,
    gt_pull = 1,
    gt_push = 2,
    gt_ireg = 3,
    gt_N = 4
};

extern const char * graph_traversal_kind_names[
    static_cast<std::underlying_type_t<graph_traversal_kind>>(
	graph_traversal_kind::gt_N )+1];

static std::ostream &
operator << ( std::ostream & os, graph_traversal_kind gtk ) {
    using T = std::underlying_type_t<graph_traversal_kind>;
    T igtk = (T) gtk;
    if( igtk >= 0 && igtk < (T)graph_traversal_kind::gt_N )
	return os << graph_traversal_kind_names[igtk];
    else
	return os << graph_traversal_kind_names[(int)graph_traversal_kind::gt_N];
}

struct VertexInfo {
    VID v;
    VID degree;
    const VID * neighbours;

    VertexInfo( VID v_, VID degree_, const VID * neighbours_ ) :
	v( v_ ), degree( degree_ ), neighbours( neighbours_ ) { }
};

template<typename VID, typename EID>
struct getVDeg {
    getVDeg( const EID * idx_ ) : idx( idx_ ) { }
    std::pair<VID,VID> operator() ( VID v ) {
	return std::make_pair( (VID)( idx[v+1] - idx[v] ), v );
    }
private:
    const EID * idx;
};

class GraphCSx {
    VID n;
    VID nmaxdeg;
    EID m;
    mmap_ptr<EID> index;
    mmap_ptr<VID> edges;
    mmap_ptr<VID> degree;
    bool symmetric;

    mm::buffer<float> * weights;

public:
    GraphCSx() { }
    GraphCSx( const std::string & infile, int allocation = -1,
	      bool _symmetric = false, const char * wfile = nullptr ) 
	: symmetric( _symmetric ), weights( nullptr ) {
	if( allocation == -1 ) {
	    numa_allocation_interleaved alloc;
	    readFromBinaryFile( infile, alloc );
	    if( wfile )
		readWeightsFromBinaryFile( wfile, alloc );
	} else {
	    numa_allocation_local alloc( allocation );
	    readFromBinaryFile( infile, alloc );
	    if( wfile )
		readWeightsFromBinaryFile( wfile, alloc );
	}
    }
    GraphCSx( const std::string & infile, const numa_allocation & allocation,
	      bool _symmetric )
	: symmetric( _symmetric ), weights( nullptr ) {
	readFromBinaryFile( infile, allocation );
    }
    GraphCSx( VID n_, EID m_, int allocation, bool symmetric_ = false,
	      bool weights_ = false )
	: n( n_ ), nmaxdeg( ~VID(0) ), m( m_ ), symmetric( symmetric_ ),
	  weights( nullptr ) {
	if( allocation == -1 ) {
	    allocateInterleaved();
	    if( weights_ )
		weights = new mm::buffer<float>( m, numa_allocation_interleaved() );
	} else {
	    allocateLocal( allocation );
	    if( weights_ )
		weights = new mm::buffer<float>(
		    m, numa_allocation_local( allocation ) );
	}
    }
    void del() {
	index.del();
	edges.del();
	degree.del();
    }
    template<typename vertex>
    GraphCSx( const wholeGraph<vertex> & WG, int allocation )
	: n( WG.n ), m( WG.m ),
	  symmetric( std::is_same<vertex,symmetricVertex>::value ),
	  weights( nullptr ) {
	if( allocation == -1 )
	    allocateInterleaved();
	else
	    allocateLocal( allocation );
	import( WG );
    }
    template<typename vertex>
    void import( const wholeGraph<vertex> & WG ) {
	assert( n == WG.n && m == WG.m );

	const vertex * V = WG.V.get();
	EID nxt = 0;
	for( VID v=0; v < n; ++v ) {
	    index[v] = nxt;
	    for( VID j=0; j < V[v].getOutDegree(); ++j )
		edges[nxt++] = V[v].getOutNeighbor(j);
	}
	index[n] = m;
	build_degree();
    }
    GraphCSx( const GraphCSx & WG, int allocation )
	: n( WG.n ), m( WG.m ), symmetric( WG.isSymmetric() ),
	  weights( nullptr ) {
	if( allocation == -1 )
	    allocateInterleaved();
	else
	    allocateLocal( allocation );
	import( WG );
    }
    void import( const GraphCSx & WG ) {
	assert( n == WG.n && m == WG.m );

	std::copy( &WG.index[0], &WG.index[n+1], &index[0] );
	std::copy( &WG.edges[0], &WG.edges[m], &edges[0] );

	build_degree();
    }
/*
    void import_expand( const GraphCSx & WG ) {
	assert( n >= WG.n && m == WG.m );

	std::copy( &WG.index[0], &WG.index[WG.n+1], &index[0] );
	std::fill( &index[WG.n+1], &index[n+1], m );
	std::copy( &WG.edges[0], &WG.edges[m], &edges[0] );
    }
*/

    template<typename vertex>
    GraphCSx( const wholeGraph<vertex> & WG, int allocation,
	      std::pair<const VID *, const VID *> remap )
	: n( WG.n ), m( WG.m ),
	  symmetric( std::is_same<vertex,symmetricVertex>::value ),
	  weights( nullptr ) {
	if( allocation == -1 )
	    allocateInterleaved();
	else
	    allocateLocal( allocation );
	import( WG, remap );
	build_degree();
    }
    template<typename vertex>
    void import( const wholeGraph<vertex> & WG,
		 std::pair<const VID *, const VID *> remap ) {
	// In w = remap.first[v], v is the new vertex ID, w is the old one
	// In v = remap.second[w], v is the new vertex ID, w is the old one
	assert( n == WG.n && m == WG.m );

	const vertex * V = WG.V.get();
	// 1. Build array for each v its degree (in parallel)
	parallel_for( VID v=0; v < n; ++v ) {
	    VID w = remap.first[v];
	    index[v] = V[w].getOutDegree();
	}
	index[n] = m;
	// 2. Prefix-sum (parallel) => index array
	EID mm = sequence::plusScan( index.get(), index.get(), n );
	assert( mm == m && "Index array count mismatch" );

	// 3. Fill out edge array (parallel)
	parallel_for( VID v=0; v < n; ++v ) {
	    VID w = remap.first[v];
	    EID nxt = index[v];
	    for( VID j=0; j < V[w].getOutDegree(); ++j )
		edges[nxt++] = remap.second[V[w].getOutNeighbor(j)];
	    std::sort( &edges[index[v]], &edges[nxt] );
	}
	build_degree();
    }
    void import( const GraphCSx & Gcsr, const RemapVertexIdempotent<VID> & ) {
	import( Gcsr );
    }
    void import( const GraphCSx & Gcsr,
		 std::pair<const VID *, const VID *> remap ) {
	// In w = remap.first[v], v is the new vertex ID, w is the old one
	// In v = remap.second[w], v is the new vertex ID, w is the old one
	assert( n == Gcsr.n && m == Gcsr.m );

	// 1. Build array for each v its degree (in parallel)
	parallel_for( VID v=0; v < n; ++v ) {
	    VID w = remap.first[v];
	    index[v] = Gcsr.index[w+1] - Gcsr.index[w];
	}
	index[n] = m;
	// 2. Prefix-sum (parallel) => index array
	EID mm = sequence::plusScan( index.get(), index.get(), n );
	assert( mm == m && "Index array count mismatch" );

	// Convenience defs
	const bool has_weights = weights != nullptr;
	float * const Tweights = has_weights ? weights->get() : nullptr;
	assert( ( ( Gcsr.getWeights() != nullptr ) || !has_weights )
		&& "expecting weights in graph" );
	const float * const Gweights = Gcsr.getWeights()
	    ? Gcsr.getWeights()->get() : nullptr;

	// 3. Fill out edge array (parallel)
	parallel_for( VID v=0; v < n; ++v ) {
	    VID w = remap.first[v];
	    EID nxt = index[v];
	    VID deg = Gcsr.index[w+1] - Gcsr.index[w];
	    for( VID j=0; j < deg; ++j ) {
		edges[nxt] = remap.second[Gcsr.edges[Gcsr.index[w]+j]];
		if( has_weights )
		    Tweights[nxt] = Gweights[Gcsr.index[w]+j];
		++nxt;
	    }
	    if( has_weights )
		paired_sort( &edges[index[v]], &edges[nxt],
			     &Tweights[index[v]] );
	    else
		std::sort( &edges[index[v]], &edges[nxt] );
	}
	build_degree();
    }
    template<typename Remapper>
    void import_expand( const GraphCSx & Gcsr,
			const partitioner & part,
			Remapper remap ) {
	symmetric = Gcsr.isSymmetric();
	
	// In this version, Gcsr is the original graph with n vertices, but
	// the current graph and remap expand that to n + npad vertices.
	
	// In w = remap.first[v], v is the new vertex ID, w is the old one
	// In v = remap.second[w], v is the new vertex ID, w is the old one
	assert( n == part.get_num_elements() && m == Gcsr.m );
	VID npad = part.get_num_padding();
	VID norig = Gcsr.n;
	assert( n - Gcsr.n == npad );

	const bool has_weights = weights != nullptr;
	float * const Tweights = has_weights ? weights->get() : nullptr;
	assert( ( ( Gcsr.getWeights() != nullptr ) || !has_weights )
		&& "expecting weights in graph" );
	float * const Gweights = Gcsr.getWeights()
	    ? Gcsr.getWeights()->get() : nullptr;

	// 1. Build array for each v its degree (in parallel)
	parallel_for( VID v=0; v < n; ++v ) {
	    VID w = remap.origID( v );
	    index[v] = w < norig ? Gcsr.index[w+1] - Gcsr.index[w] : 0;
	}
	index[n] = m;
	// 2. Prefix-sum (parallel) => index array
	EID mm = sequence::plusScan( index.get(), index.get(), n );
	assert( mm == m && "Index array count mismatch" );

	// 3. Fill out edge array (parallel)
	parallel_for( VID v=0; v < n; ++v ) {
	    VID w = remap.origID( v );
	    EID nxt = index[v];
	    VID deg = w < norig ? Gcsr.index[w+1] - Gcsr.index[w] : 0;
	    for( VID j=0; j < deg; ++j ) {
		edges[nxt] = remap.remapID( Gcsr.edges[Gcsr.index[w]+j] );
		if( has_weights )
		    Tweights[nxt] = Gweights[Gcsr.index[w]+j];
		++nxt;
	    }
	    if( has_weights )
		paired_sort( &edges[index[v]], &edges[nxt],
			     &Tweights[index[v]] );
	    else
		std::sort( &edges[index[v]], &edges[nxt] );
	}
	build_degree();
    }
    void import_transpose( const GraphCSx & Gcsr ) {
	assert( n == Gcsr.numVertices() );
	assert( m == Gcsr.numEdges() );

	mmap_ptr<EID> aux( n+1, numa_allocation_interleaved() );

	std::cerr << "transpose: init\n";
	parallel_for( VID v=0; v < n+1; ++v )
	    aux[v] = 0;

	std::cerr << "transpose: count edges\n";
	parallel_for( EID i=0; i < m; ++i )
	    __sync_fetch_and_add( &aux[Gcsr.edges[i]], 1 );

	std::cerr << "transpose: scan (seq)\n";
	index[0] = 0;
	for( VID s=0; s < n; ++s ) {
	    index[s+1] = index[s] + aux[s];
	    aux[s] = index[s];
	}
	assert( index[n] == m );

	float * w = nullptr, * wg = nullptr;
	if( weights ) {
	    w = weights->get();
	    wg = Gcsr.weights->get();
	    assert( w && wg );
	}

	std::cerr << "transpose: place\n";
	parallel_for( VID s=0; s < n; ++s ) {
	    EID i = Gcsr.index[s];
	    EID j = Gcsr.index[s+1];
	    for( ; i < j; ++i ) {
		auto idx = __sync_fetch_and_add( &aux[Gcsr.edges[i]], 1 );
		edges[idx] = s;

		if( w )
		    w[idx] = wg[i];
	    }
	}

	std::cerr << "transpose: sort\n";
	parallel_for( VID s=0; s < n; ++s ) {
	    assert( aux[s] == index[s+1] );
	    if( w )
		paired_sort( &edges[index[s]], &edges[index[s+1]],
			     &w[index[s]] );
	    else
		std::sort( &edges[index[s]], &edges[index[s+1]] );
	}

	aux.del();
	build_degree();
    }
/*
    void import_transpose_expand( const GraphCSx & Gcsr ) {
	assert( n >= Gcsr.numVertices() );
	assert( m == Gcsr.numEdges() );

	mmap_ptr<EID> aux( n+1 );

	std::cerr << "transpose-expand: init\n";
	parallel_for( VID v=0; v < n+1; ++v )
	    aux[v] = 0;

	std::cerr << "transpose-expand: count edges\n";
	parallel_for( EID i=0; i < m; ++i )
	    __sync_fetch_and_add( &aux[Gcsr.edges[i]], 1 );

	std::cerr << "transpose-expand: scan (seq)\n";
	index[0] = 0;
	for( VID s=0; s < n; ++s ) {
	    index[s+1] = index[s] + aux[s];
	    aux[s] = index[s];
	}
	assert( index[n] == m );

	std::cerr << "transpose-expand: place\n";
	parallel_for( VID s=0; s < Gcsr.n; ++s ) {
	    EID i = Gcsr.index[s];
	    EID j = Gcsr.index[s+1];
	    for( ; i < j; ++i ) {
		auto idx = __sync_fetch_and_add( &aux[Gcsr.edges[i]], 1 );
		edges[idx] = s;
	    }
	}

	std::cerr << "transpose-expand: sort\n";
	parallel_for( VID s=0; s < n; ++s ) {
	    assert( aux[s] == index[s+1] );
	    std::sort( &edges[index[s]], &edges[index[s+1]] );
	}

	aux.del();
    }
*/

    template<typename vertex>
    GraphCSx( const wholeGraph<vertex> & WG,
	      const partitioner & part, int p, int allocation )
	: n( WG.numVertices() ),
	  symmetric( std::is_same<vertex,symmetricVertex>::value ),
	  weights( nullptr ) {
	// Range of destination vertices to include
	VID rangeLow = part.start_of(p);
	VID rangeHi = part.start_of(p+1);
	
	// Short-hands
	vertex *V = WG.V.get();

	m = 0;
        for( VID i=rangeLow; i < rangeHi; i++ )
	    m += V[i].getInDegree();

	if( allocation == -1 )
	    allocateInterleaved();
	else
	    allocateLocal( allocation );
	
        EID nxt = 0;
        for( VID s=0; s < n; s++ ) {
	    index[s] = nxt;
	    VID deg = V[s].getOutDegree();
	    for( VID i=0; i < deg; i++ ) {
		VID d = V[s].getOutNeighbor(i);
		if( rangeLow <= d && d < rangeHi )
		    edges[nxt++] = d;
	    }
	}
	assert( nxt == m );
	index[n] = nxt;
	build_degree();
    }

    template<typename vertex>
    GraphCSx( const wholeGraph<vertex> & WG,
	      const partitioner & part, int p,
	      std::pair<const VID *, const VID *> remap )
	: n( WG.numVertices() ),
	  symmetric( std::is_same<vertex,symmetricVertex>::value ),
	  weights( nullptr )  {
	// Range of destination vertices to include
	VID rangeLow = part.start_of(p);
	VID rangeHi = part.start_of(p+1);
	
	// Short-hands
	vertex *V = WG.V.get();

	m = 0;
        for( VID i=rangeLow; i < rangeHi; i++ )
	    m += V[remap.first[i]].getInDegree();

	allocateLocal( part.numa_node_of( p ) );
	
        EID nxt = 0;
        for( VID s=0; s < n; s++ ) {
	    VID sr = remap.first[s];
	    index[s] = nxt;
	    VID deg = V[sr].getOutDegree();
	    for( VID i=0; i < deg; i++ ) {
		VID d = remap.second[V[sr].getOutNeighbor(i)];
		if( rangeLow <= d && d < rangeHi )
		    edges[nxt++] = d;
	    }
	    std::sort( &edges[index[s]], &edges[nxt] );
	}
	assert( nxt == m );
	index[n] = nxt;
	build_degree();
    }
    GraphCSx( const GraphCSx & WG,
	      const partitioner & part, int p,
	      std::pair<const VID *, const VID *> remap )
	: n( WG.numVertices() ),
	  symmetric( false ),
	  weights( nullptr )  { // as these are partitions, won't be symmetric
	// Range of destination vertices to include
	VID rangeLow = part.start_of(p);
	VID rangeHi = part.start_of(p+1);
	
	// Count number of edges in partition -- already done when partitioning!
	EID mm = WG.numEdges();
	m = 0;
        for( VID i=0; i < mm; i++ ) {
	    VID d = remap.second[WG.edges[i]];
	    if( rangeLow <= d && d < rangeHi )
		++m;
	}

	allocateLocal( part.numa_node_of( p ) );
	
        EID nxt = 0;
        for( VID s=0; s < n; s++ ) {
	    VID sr = remap.first[s];
	    index[s] = nxt;
	    for( VID i=WG.index[sr]; i < WG.index[sr+1]; i++ ) {
		VID d = remap.second[WG.edges[i]];
		if( rangeLow <= d && d < rangeHi )
		    edges[nxt++] = d;
	    }
	    std::sort( &edges[index[s]], &edges[nxt] );
	}
	assert( nxt == m );
	index[n] = nxt;
	build_degree();
    }

    void writeToBinaryFile( const std::string & ofile ) {
	ofstream file( ofile, ios::out | ios::trunc | ios::binary );
	uint64_t header[8] = {
	    2, // version
	    1,
	    (uint64_t)n, // num nodes
	    (uint64_t)m, // num edges
	    sizeof(VID), // sizeof(VID)
	    sizeof(EID), // sizeof(EID)
	    0, // unused
	    0, // unused
	};

	file.write( (const char *)header, sizeof(header) );
	file.write( (const char *)&index[0], sizeof(index[0])*n );
	file.write( (const char *)&edges[0], sizeof(edges[0])*m );

	file.close();
    }
    void writeToTextFile( const std::string & ofile ) {
	ofstream file( ofile, ios::out | ios::trunc );

	file << ( weights ? "WeightedAdjacencyGraph\n" : "AdjacencyGraph\n" );
	file << (uint64_t)n << "\n"
	     << (uint64_t)m << "\n";

	for( VID v=0; v < n; ++v )
	    file << index[v] << '\n';

	for( EID e=0; e < m; ++e )
	    file << edges[e] << '\n';

	if( weights ) {
	    float * w = weights->get();
	    file.precision( std::numeric_limits<float>::max_digits10 );
	    file << std::fixed;
	    for( EID e=0; e < m; ++e )
		file << w[e] << '\n';
	}
	    
	file.close();
    }

    void readFromBinaryFile( const std::string & ifile,
			     const numa_allocation & alloc ) {
	// We messed up with the organization of the file. Ideally we should
	// just mmap the contents in memory, however, for vectorization purposes
	// we need alignment in some places. This has not been taken into
	// account and may affect us adversely. Moreover, the n+1-th index
	// value has not been written to disk and we need a place to store it
#if 0
	// Simple solution
	std::cerr << "Reading file " << ifile << std::endl;
	ifstream file( ifile, ios::in | ios::binary );
	uint64_t header[8];
	file.read( (char *)header, sizeof(header) );
	if( header[0] != 2 ) {
	    std::cerr << "Only accepting version 2 files\n";
	    exit( 1 );
	}
	n = header[2];
	m = header[3];
	assert( sizeof(VID) == header[4] );
	assert( sizeof(EID) == header[5] );

	allocate( alloc );

	file.read( (char *)&index[0], sizeof(index[0])*n );
	index[n] = m;
	file.read( (char *)&edges[0], sizeof(edges[0])*m );

	file.close();
	std::cerr << "Reading file done" << std::endl;
#else
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
	assert( sizeof(VID) == header[4] );
	assert( sizeof(EID) == header[5] );

	allocate( alloc );

	const EID *index_p = reinterpret_cast<const EID *>(
	    data+sizeof(uint64_t)*8 );
	parallel_for( VID v=0; v < n; ++v )
	    index[v] = index_p[v];
	index[n] = m;

	const VID *edges_p = reinterpret_cast<const VID *>(
	    data+sizeof(uint64_t)*8+n*sizeof(EID) );
	parallel_for( EID e=0; e < m; ++e )
	    edges[e] = edges_p[e];

	munmap( (void *)data, len );
	close( fd );
	std::cerr << "Reading file done" << std::endl;
#endif
	build_degree();
    }

    void readWeightsFromBinaryFile( const std::string & wfile,
				    const numa_allocation & alloc ) {
	std::cerr << "Reading (using mmap) weights file " << wfile << std::endl;
	int fd;

	if( (fd = open( wfile.c_str(), O_RDONLY )) < 0 ) {
	    std::cerr << "Cannot open file '" << wfile << "': "
		      << strerror( errno ) << "\n";
	    exit( 1 );
	}
	off_t len = lseek( fd, 0, SEEK_END );
	if( len == off_t(-1) ) {
	    std::cerr << "Cannot lseek to end of file '" << wfile << "': "
		      << strerror( errno ) << "\n";
	    exit( 1 );
	}

	assert( len == m * sizeof(float)
		&& "weights file size mismatch" );

	weights = new mm::buffer<float>( (size_t)m, fd, (off_t)0, alloc );

	close( fd );
	std::cerr << "Reading file done" << std::endl;
    }

public:
    void fragmentation() const {
	std::cerr << "GraphCSx:\ntotal-size: "
		  << ( numVertices()*sizeof(EID)+numEdges()*sizeof(VID) )
		  << "\n";
    }

    VID max_degree() const {
	VID max = 0;
	for( VID v=0; v < n; ++v ) {
	    VID deg = index[v+1] - index[v];
	    if( max < deg )
		max = deg;
	}
	return max;
    }

    VID findHighestDegreeVertex() const {
	using T = std::pair<VID,VID>;
	T s = sequence::reduce<T>( (VID)0, n, sequence::argmaxF<T>(),
				   getVDeg<VID,EID>(index.get()) );
	return s.second;
    }

public:
    class edge_iterator {
    public:
	edge_iterator( VID zid, EID eid, const GraphCSx * graph )
	    : m_zid( zid ), m_eid( eid ), m_graph( graph ) { }
	edge_iterator( const edge_iterator & it )
	    : m_zid( it.m_zid ), m_eid( it.m_eid ), m_graph( it.m_graph ) { }

	edge_iterator & operator = ( const edge_iterator & it ) {
	    m_zid = it.m_zid;
	    m_eid = it.m_eid;
	    m_graph = it.m_graph;
	    return *this;
	}

	std::pair<VID,VID> operator * () const {
	    return std::make_pair( m_zid, m_graph->getEdges()[m_eid] );
	}

	edge_iterator & operator ++ () {
	    ++m_eid;
	    if( m_eid == m_graph->getIndex()[m_zid+1] )
		++m_zid;
	    return *this;
	}

	edge_iterator operator ++ ( int ) {
	    edge_iterator cp( *this );
	    ++*this;
	    return cp;
	}

	bool operator == ( edge_iterator it ) const {
	    return m_zid == it.m_zid && m_eid == it.m_eid
		&& m_graph == it.m_graph;
	}
	bool operator != ( edge_iterator it ) const {
	    return !( *this == it );
	}

	EID operator - ( edge_iterator it ) const {
	    return m_eid - it.m_eid;
	}

    private:
	VID m_zid;
	EID m_eid;
	const GraphCSx * m_graph;
    };

    edge_iterator edge_begin() const {
	return edge_iterator( 0, 0, this );
    }
    edge_iterator edge_end() const {
	return edge_iterator( n, m, this );
    }

public:
    class neighbour_iterator : edge_iterator {
    public:
	neighbour_iterator( VID zid, EID eid, const GraphCSx * graph )
	    : edge_iterator( zid, eid, graph ) { }
	neighbour_iterator( const edge_iterator & it )
	    : edge_iterator( it ) { }

	neighbour_iterator & operator = ( const neighbour_iterator & it ) {
	    *static_cast<edge_iterator *>( this )
		= *static_cast<const edge_iterator *>( &it );
	    return *this;
	}

	VID operator * () const {
	    return static_cast<const edge_iterator *>( this )
		->operator *().second;
	}

	neighbour_iterator & operator ++ () {
	    static_cast<edge_iterator *>( this )->operator ++();
	    return *this;
	}

	neighbour_iterator operator ++ ( int ) {
	    neighbour_iterator cp( *this );
	    ++*this;
	    return cp;
	}

	bool operator == ( neighbour_iterator it ) const {
	    return *static_cast<const edge_iterator *>( this )
		== *static_cast<const edge_iterator *>( &it );
	}
	bool operator != ( neighbour_iterator it ) const {
	    return !( *this == it );
	}

	EID operator - ( neighbour_iterator it ) const {
	    return static_cast<const edge_iterator *>( this )
		->operator - ( *static_cast<const edge_iterator *>( &it ) );
	}
    };

    neighbour_iterator neighbour_begin( VID v ) const {
	return neighbour_iterator( v, getIndex()[v], this );
    }
    neighbour_iterator neighbour_end( VID v ) const {
	return neighbour_iterator( v+1, getIndex()[v+1], this );
    }
    
public:
    class vertex_iterator {
    public:
	vertex_iterator( VID zid, const GraphCSx * graph )
	    : m_zid( zid ), m_graph( graph ) { }
	vertex_iterator( const vertex_iterator & it )
	    : m_zid( it.m_zid ), m_graph( it.m_graph ) { }

	vertex_iterator & operator = ( const vertex_iterator & it ) {
	    m_zid = it.m_zid;
	    m_graph = it.m_graph;
	    return *this;
	}

	VertexInfo operator * () const {
	    EID idx = m_graph->getIndex()[m_zid];
	    return VertexInfo( m_zid,
			       m_graph->getIndex()[m_zid+1] - idx,
			       &m_graph->getEdges()[idx] );
	}

	vertex_iterator & operator ++ () {
	    ++m_zid;
	    return *this;
	}

	vertex_iterator operator ++ ( int ) {
	    vertex_iterator cp( *this );
	    ++*this;
	    return cp;
	}

	bool operator == ( vertex_iterator it ) const {
	    return m_zid == it.m_zid && m_graph == it.m_graph;
	}
	bool operator != ( vertex_iterator it ) const {
	    return !( *this == it );
	}
	VID operator - ( vertex_iterator it ) const {
	    return m_zid - it.m_zid;
	}

    private:
	VID m_zid;
	const GraphCSx * m_graph;
    };

    vertex_iterator vertex_begin() const {
	return vertex_iterator( 0, this );
    }
    vertex_iterator vertex_begin( VID v ) const {
	return vertex_iterator( v, this );
    }
    vertex_iterator vertex_end() const {
	return vertex_iterator( n, this );
    }
    vertex_iterator vertex_end( VID v ) const {
	return vertex_iterator( v+1, this );
    }

    vertex_iterator part_vertex_begin( const partitioner & part, unsigned p ) const {
	return vertex_iterator( part.start_of( p ), this );
    }
    vertex_iterator part_vertex_end( const partitioner & part, unsigned p ) const {
	return vertex_iterator( part.end_of( p ), this );
    }

public:
    mm::buffer<EID> buildInverseEdgeMap() const {
	mm::buffer<EID> buf( m, numa_allocation_interleaved() );
	EID * invert = buf.get();
	const EID * idx = index.get();
	const VID * edg = edges.get();

	// TODO: could optimise for symmetric graphs by limiting the scan
	//       to, e.g., j < i and setting invert[j] = i; at the same time
	//       as invert[i] = j;
	if( symmetric ) {
	    parallel_for( VID u=0; u < n; ++u ) {
		EID s = idx[u];
		EID e = idx[u+1];
		for( EID i=s; i < e; ++i ) {
		    VID v = edg[i];
		    if( v > u )
			break;
		    EID vs = idx[v];
		    EID ve = idx[v+1];

		    // Note: this is an optimisation based on the
		    //       assumption that the neighbour list is
		    //       sorted in increasing order.
		    if( ve - vs > (EID)30 ) {
			const VID * ij
			    = std::lower_bound( &edg[vs], &edg[ve], u );
			assert( u == *ij
				&& "need to find value in symmetric graph" );
			EID j = ij - edg;
			invert[i] = j;
			invert[j] = i;
		    } else {
			for( EID j=vs; j < ve; ++j ) {
			    if( edg[j] == u ) {
				invert[i] = j;
				invert[j] = i;
				break;
			    } else if( edg[j] > u ) {
				// Assuming sorted list
				break;
			    }
			}
		    }
		}
	    }
	} else {
	    parallel_for( VID u=0; u < n; ++u ) {
		EID s = idx[u];
		EID e = idx[u+1];
		for( EID i=s; i < e; ++i ) {
		    VID v = edg[i];
		    invert[i] = ~(EID)0;
		    EID vs = idx[v];
		    EID ve = idx[v+1];
		    for( EID j=vs; j < ve; ++j ) {
			if( edg[j] == u ) {
			    invert[i] = j;
			    break;
			} else if( edg[j] > u ) {
			    // Note: this is an optimisation based on the
			    //       assumption that the neighbourlist is
			    //       sorted in increasing order.
			    break;
			}
		    }
		}
	    }
	}

	return buf;
    }

public:
    EID *getIndex() { return index.get(); }
    const EID *getIndex() const { return index.get(); }
    VID *getEdges() { return edges.get(); }
    const VID *getEdges() const { return edges.get(); }
    VID *getDegree() { return degree.get(); }
    const VID *getDegree() const { return degree.get(); }

    VID numVertices() const { return n; }
    EID numEdges() const { return m; }

    // Very specific interface
    [[deprecated("This interface should not normally be used!")]]
    void setNumEdges( EID um ) { m = um; }

    VID getDegree( VID v ) const { return index[v+1] - index[v]; }
    VID getNeighbor( VID v, VID pos ) const { return edges[index[v]+pos]; }

    VID getMaxDegreeVertex() const { return nmaxdeg; }
    void setMaxDegreeVertex( VID v ) { nmaxdeg = v; }

    bool isSymmetric() const { return symmetric; }

    bool hasEdge( VID s, VID d ) const {
	bool ret = false;
	for( EID e=index[s]; e < index[s+1]; ++e )
	    if( edges[e] == d )
		return true;
	return false;
    }

    const mm::buffer<float> * getWeights() const { return weights; }

private:
    void allocate( const numa_allocation & alloc ) {
	// Note: only interleaved and local supported
	index.allocate( n+1, alloc );
	edges.allocate( m, alloc );
	degree.allocate( n, alloc );
    }

    void allocateInterleaved() {
	allocate( numa_allocation_interleaved() );
    }
    void allocateLocal( int numa_node ) {
	allocate( numa_allocation_local( numa_node ) );
    }
    void build_degree() {
	parallel_for( VID v=0; v < n; ++v )
	    degree[v] = index[v+1] - index[v];
    }
};

// Obtaining the degree of a vertex
template<>
struct gtraits_getoutdegree<GraphCSx> {
    using traits = gtraits<GraphCSx>;
    using VID = typename traits::VID;

    gtraits_getoutdegree( const GraphCSx & G_ ) : G( G_ ) { }
    
    VID operator() ( VID v ) {
	return G.getDegree( v );
    }

private:
    const GraphCSx & G;
};

// Obtaining the vertex with maximum degree
template<>
struct gtraits_getmaxoutdegree<GraphCSx> {
    using traits = gtraits<GraphCSx>;
    using VID = typename traits::VID;

    gtraits_getmaxoutdegree( const GraphCSx & G_ ) : G( G_ ) { }
    
    VID getMaxOutDegreeVertex() const { return G.getMaxDegreeVertex(); }
    VID getMaxOutDegree() const { return G.getDegree( G.getMaxDegreeVertex() ); }

private:
    const GraphCSx & G;
};

#endif // GRAPHGRIND_GRAPH_GRAPHCSX_H

