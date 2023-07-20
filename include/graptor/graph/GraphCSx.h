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

#if EXTERNAL_SPTRANS
#include "../../sptrans/sptrans.h"
#endif

void partitionBalanceEdges( const GraphCSx & Gcsc, partitioner &part );

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

template<bool IsSorted, typename lVID>
lVID merge_count( const lVID * ls, const lVID * le,
		  const lVID * rs, const lVID * re ) {
    const lVID * l = ls;
    const lVID * r = rs;

    lVID cnt = 0;
    while( l != le && r != re ) {
	if constexpr ( !IsSorted ) {
	    if( l > ls && *l <= *(l-1) )
		return std::numeric_limits<lVID>::max();
	    if( r > rs && *r <= *(r-1) )
		return std::numeric_limits<lVID>::max();
	}
	    
	if( *l < *r ) {
	    ++cnt;
	    ++l;
	} else if( *l > *r ) {
	    ++cnt;
	    ++r;
	} else {
	    ++cnt;
	    ++l;
	    ++r;
	}
    }
    cnt += ( le - l );
    cnt += ( re - r );
    return cnt;
}

template<typename lVID>
void merge_place( const lVID * ls, const lVID * le,
		  const lVID * rs, const lVID * re,
		  lVID * u ) {
    const lVID * l = ls;
    const lVID * r = rs;

    while( l != le && r != re ) {
	if( *l < *r )
	    *u++ = *l++;
	else if( *l > *r )
	    *u++ = *r++;
	else {
	    *u++ = *l;
	    ++l;
	    ++r;
	}
    }
    while( l != le )
	*u++ = *l++;
    while( r != re )
	*u++ = *r++;
}

template<typename T>
class compact_list {
public:
    using member_type = T;
    using storage_type = int_type_of_size_t<sizeof(T)>; // unsigned type
    using counter_type = storage_type;
    static_assert( sizeof(member_type)*2 == sizeof(member_type*),
		   "assumption on member type" );

private:
    static constexpr storage_type flag_position = sizeof(storage_type)*8 - 1;
    static constexpr storage_type direct_mask =
	storage_type(1) << flag_position;
    static constexpr storage_type invalid_mask = ~(storage_type)0;

public:
    compact_list() {
	// data.direct.f0 = invalid_mask;
	// data.direct.f1 = invalid_mask;
	*reinterpret_cast<intptr_t *>( &data.vec ) = ~(intptr_t)0;
    }
    ~compact_list() {
	if( !is_direct() )
	    delete data.vec;
    }

    size_t size() const {
	if( is_direct() ) {
	    if( data.direct.f0 == invalid_mask )
		return 0;
	    else if( data.direct.f1 == invalid_mask )
		return 1;
	    else
		return 2;
	} else
	    return data.vec->size();
    }

    void push_back( member_type v ) {
	if( is_direct() ) {
	    if( data.direct.f0 == invalid_mask )
		data.direct.f0 = v | direct_mask;
	    else if( data.direct.f1 == invalid_mask )
		data.direct.f1 = v | direct_mask;
	    else {
		// Convert to vector
		std::vector<member_type> * vec
		    = new std::vector<member_type>();
		assert( ( reinterpret_cast<intptr_t>( vec )
			  & ( intptr_t(1) << (2*8*sizeof(storage_type)-1) )  )
			== 0 );
		vec->push_back( get_value( data.direct.f0 ) );
		vec->push_back( get_value( data.direct.f1 ) );
		vec->push_back( v );
		data.vec = vec;
	    }
	} else
	    data.vec->push_back( v );
    }

    void copy_to( member_type * a ) const {
	if( is_direct() ) {
	    if( data.direct.f0 != invalid_mask ) {
		a[0] = get_value( data.direct.f0 );
		if( data.direct.f1 != invalid_mask )
		    a[1] = get_value( data.direct.f1 );
	    }
	} else
	    std::copy( data.vec->cbegin(), data.vec->cend(), a );
    }
    
private:
    static constexpr member_type get_value( storage_type f ) {
	return (member_type)( f & ~direct_mask );
    }
    static constexpr bool is_direct_value( storage_type f ) {
	return (f >> flag_position) != 0;
    }
    constexpr bool is_direct() const {
	// check f1 assuming little endian byte order and less likely to
	// have 1 bit set in highest position
	return is_direct_value( data.direct.f1 );
    }
    
private:
    union { 
	struct {
	    storage_type f0;
	    storage_type f1;
	} direct;
	std::vector<member_type> * vec;
    } data;
};

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

template<typename VID, typename EID>
struct selectVDeg {
    selectVDeg( const EID * idx_, VID threshold_ )
	: idx( idx_ ), threshold( threshold_ ) { }
    VID operator() ( VID v ) {
	return ( idx[v+1] - idx[v] > threshold ) ? 1 : 0;
    }
private:
    const EID * idx;
    const VID threshold;
};

inline void parallel_read( int fd, size_t off, void * vptr, size_t len ) {
    timer tm;
    tm.start();
    uint8_t * ptr = reinterpret_cast<uint8_t *>( vptr );

    constexpr size_t BLOCK = size_t(128) << 20; // 128 MiB
    unsigned num_threads = graptor_num_threads();
    size_t nblock = ( len + BLOCK - 1 ) / BLOCK;
    parallel_loop( (unsigned)0, num_threads, [&]( unsigned t ) { 
	size_t blk_from = ( nblock * t ) / num_threads;
	size_t blk_to = ( nblock * (t+1) ) / num_threads;
	size_t from = blk_from * BLOCK;
	size_t to = std::min( blk_to * BLOCK, len );
	for( size_t b=from; b < to; b += BLOCK ) {
	    size_t sz = std::min( BLOCK, to - b );
	    ssize_t res = pread( fd, &ptr[b], sz, off+b );
	    if( res != sz ) {
		std::cerr << "Reading file fd='" << fd << "' failed: "
			  << strerror( errno ) << "\n";
		exit( 1 );
	    }
	}
    } );
    double delay = tm.stop();
    std::cerr << "Read " << pretty_size( len ) << " in "
	      << delay << " seconds, "
	      << pretty_size( double(len)/delay ) << "/s\n";
}

class GraphCSx {
    VID n;
    VID nmaxdeg;
    EID m;
    mm::buffer<EID> index;
    mm::buffer<VID> edges;
    mm::buffer<VID> degree;
    bool symmetric;

    mm::buffer<float> * weights;
    mutable mm::buffer<bool> flags;

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
	if( weights ) {
	    weights->del();
	    delete weights;
	}
	flags.del( "GraphCSx flags array" );
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
    GraphCSx( const GraphCSx & G,
	      std::pair<const VID *, const VID *> remap,
	      int allocation = -1 )
	: n( G.n ), m( G.m ), symmetric( G.isSymmetric() ),
	  weights( nullptr ) {
	if( allocation == -1 )
	    allocateInterleaved();
	else
	    allocateLocal( allocation );
	import( G, remap );
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
	parallel_loop( (VID)0, n, [&]( VID v ) { 
	    VID w = remap.first[v];
	    index[v] = V[w].getOutDegree();
	} );
	index[n] = m;
	// 2. Prefix-sum (parallel) => index array
	EID mm = sequence::plusScan( index.get(), index.get(), n );
	assert( mm == m && "Index array count mismatch" );

	// 3. Fill out edge array (parallel)
	parallel_loop( (VID)0, n, [&]( VID v ) { 
	    VID w = remap.first[v];
	    EID nxt = index[v];
	    for( VID j=0; j < V[w].getOutDegree(); ++j )
		edges[nxt++] = remap.second[V[w].getOutNeighbor(j)];
	    std::sort( &edges[index[v]], &edges[nxt] );
	} );
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
	parallel_loop( (VID)0, n, [&]( VID v ) { 
	    VID w = remap.first[v];
	    index[v] = Gcsr.index[w+1] - Gcsr.index[w];
	} );
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
	parallel_loop( (VID)0, n, [&]( VID v ) { 
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
	} );
	build_degree();
    }
    template<typename Remapper>
    void import_expand( const GraphCSx & Gcsr,
			const partitioner & part,
			Remapper remap ) {
	timer tm;
	tm.start();
	std::cerr << "GraphCSx::import_expand...\n";

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
	parallel_loop( (VID)0, n, [&]( VID v ) { 
	    VID w = remap.origID( v );
	    index[v] = w < norig ? Gcsr.index[w+1] - Gcsr.index[w] : 0;
	} );
	index[n] = m;
	// 2. Prefix-sum (parallel) => index array
	EID mm = sequence::plusScan( index.get(), index.get(), n );
	assert( mm == m && "Index array count mismatch" );

	std::cerr << "GraphCSx::import_expand: init and remapped index[]: "
		  << tm.next() << "\n";
	
	// 3. Fill out edge array (parallel)
	if( has_weights ) {
	    // Rather than copying weights prior to paired_sort, could also
	    // merge into the remapping step in paired_sort to avoid the
	    // additional copy.
	    parallel_loop( (VID)0, n, [&]( VID v ) { 
		VID w = remap.origID( v );
		if( w < norig ) {
		    EID nxt = index[v];
		    EID js = Gcsr.index[w];
		    EID je = Gcsr.index[w+1];
		    EID deg = je - js;
		    assert( deg <= n );
		    if( deg > 0 )
			std::copy( &Gweights[js], &Gweights[je],
				   &Tweights[nxt] );
		    for( EID j=js; j < je; ++j )
			edges[nxt+(j-js)] = remap.remapID( Gcsr.edges[j] );

		    if( w & 1 )
			paired_sort( &edges[nxt], &edges[nxt+deg], &Tweights[nxt] );
		}
	    } );
	} else {
	    parallel_loop( (VID)0, n, [&]( VID v ) { 
		VID w = remap.origID( v );
		if( w < norig ) {
		    EID nxt = index[v];
		    EID js = Gcsr.index[w];
		    EID je = Gcsr.index[w+1];
		    EID deg = je - js;
		    for( EID j=js; j < je; ++j ) {
			edges[nxt+(j-js)] = remap.remapID( Gcsr.edges[j] );
			// heap_insert( &edges[nxt], j-js, 
			// remap.remapID( Gcsr.edges[j] ) );
		    }
		    if( deg > 1 )
			std::sort( &edges[nxt], &edges[nxt+deg] );
			// heap_sort( &edges[nxt], &edges[nxt+deg] );
		}
	    } );
	}

	std::cerr << "GraphCSx::import_expand: remapping edges and weights: "
		  << tm.next() << "\n";
	
	build_degree();

	std::cerr << "GraphCSx::import_expand: building degree[]: "
		  << tm.next() << "\n";
    }
    void import_transpose( const GraphCSx & Gcsr ) {
	const VID P = atoi( getenv( "GRAPTOR_P" ) ); // 128;

	// import_transpose_partitioned( Gcsr, P );
	// import_transpose_adjacency( Gcsr, P );
	import_transpose_hybrid( Gcsr, P );
	// import_transpose_pull( Gcsr, P );
	// import_transpose_sptrans_scan( Gcsr, P );
	// import_transpose_sptrans_merge( Gcsr, P );
	return;

	timer tm;
	tm.start();
	
	assert( n == Gcsr.numVertices() );
	assert( m == Gcsr.numEdges() );

	mmap_ptr<EID> aux( n+1, numa_allocation_interleaved() );
	std::cerr << "transpose: setup: " << tm.next() << "\n";

	parallel_loop( (VID)0, n+1, [&]( VID v ) { 
	    aux[v] = 0;
	} );
	std::cerr << "transpose: init: " << tm.next() << "\n";

	// This loop could have better cache usage with VID aux[] instead of EID
	parallel_loop( (EID)0, m, [&]( EID i ) { 
	    __sync_fetch_and_add( &aux[Gcsr.edges[i]], 1 );
	} );
	std::cerr << "transpose: count edges: " << tm.next() << "\n";

	index[0] = 0;
	for( VID s=0; s < n; ++s ) {
	    index[s+1] = index[s] + aux[s];
	    aux[s] = index[s];
	}
	assert( index[n] == m );
	std::cerr << "transpose: scan (seq): " << tm.next() << "\n";

	float * w = nullptr, * wg = nullptr;
	if( weights ) {
	    w = weights->get();
	    wg = Gcsr.weights->get();
	    assert( w && wg );
	}

	parallel_loop( (VID)0, n, [&]( VID s ) { 
	    EID i = Gcsr.index[s];
	    EID j = Gcsr.index[s+1];
	    for( ; i < j; ++i ) {
		// Lots of contention on the counters for vertices with
		// many in-edges (in-hubs) -> privatize those counters?
		// Need per-partition index values, so per-partition degrees,
		// but we cannot identify these vertices without knowing
		// their degrees. Identify by out-degree?
		auto idx = __sync_fetch_and_add( &aux[Gcsr.edges[i]], 1 );
		edges[idx] = s;

		if( w )
		    w[idx] = wg[i];
	    }
	} );
	std::cerr << "transpose: place: " << tm.next() << "\n";

	// Because we are doing a transpose and keeping VIDs the same (no
	// remapping), sorting would be unnecessary in a sequential
	// implementation (insert sources in order traversed).
	// So in a partitioned transpose, with pre-defined per-partition
	// insertion points, we don't need sorting either.
	parallel_loop( (VID)0, n, [&]( VID s ) { 
	    assert( aux[s] == index[s+1] );
	    if( w )
		paired_sort( &edges[index[s]], &edges[index[s+1]],
			     &w[index[s]] );
	    else
		std::sort( &edges[index[s]], &edges[index[s+1]] );
	} );
	std::cerr << "transpose: sort: " << tm.next() << "\n";

	aux.del();
	build_degree();
	std::cerr << "transpose: build degree: " << tm.next() << "\n";

	std::cerr << "transpose: total: " << tm.total() << "\n";
    }

    static GraphCSx create_union( const GraphCSx & G1, const GraphCSx & G2 ) {
	// This code ignores weights
	timer tm;
	tm.start();

	assert( G1.numVertices() == G2.numVertices() );
	VID n = G1.numVertices();

	const EID * idx1 = G1.getIndex();
	const EID * idx2 = G2.getIndex();
	VID * edges1 = const_cast<VID *>( G1.getEdges() );
	VID * edges2 = const_cast<VID *>( G2.getEdges() );

	// Step 1: Determine incident edge count per vertex for union
	mm::buffer<EID> new_idx( n+1, numa_allocation_interleaved() );
	parallel_loop( (VID)0, n, [&]( VID v ) {
	    VID cnt = merge_count<false>( &edges1[idx1[v]], &edges1[idx1[v+1]],
					  &edges2[idx2[v]], &edges2[idx2[v+1]] );
	    if( cnt == std::numeric_limits<VID>::max() ) {
		std::sort( &edges1[idx1[v]], &edges1[idx1[v+1]] );
		std::sort( &edges2[idx2[v]], &edges2[idx2[v+1]] );
		cnt = merge_count<true>( &edges1[idx1[v]], &edges1[idx1[v+1]],
					 &edges2[idx2[v]], &edges2[idx2[v+1]] );
	    }
	    new_idx[v] = (EID)cnt;
	} );
	std::cerr << "union: merge-count edges: " << tm.next() << "\n";

	// Step 2: prefix scan
	EID off = 0;
	for( VID s=0; s < n; ++s ) {
	    EID deg = new_idx[s];
	    new_idx[s] = off;
	    off += deg;
	}
	new_idx[n] = off;
	std::cerr << "union: scan (seq): " << tm.next() << "\n";

	// Step 3: Construct graph object
	GraphCSx Gu( n, off, -1, G1.isSymmetric() && G2.isSymmetric() );
	std::cerr << "union: allocate new graph: " << tm.next() << "\n";
	
	// Step 4: Replace index list
	Gu.index.del();
	Gu.index = new_idx;
	std::cerr << "union: initialise indices: " << tm.next() << "\n";
	
	// Step 5: Merge again, now placing vertices, and knowing neighbour
	//         lists are sorted
	const EID * idx = Gu.getIndex();
	VID * edges = Gu.getEdges();
	parallel_loop( (VID)0, n, [&]( VID v ) { 
	    merge_place( &edges1[idx1[v]], &edges1[idx1[v+1]],
			 &edges2[idx2[v]], &edges2[idx2[v+1]],
			 &edges[idx[v]] );
	} );
	std::cerr << "union: merge-place edges: " << tm.next() << "\n";
	std::cerr << "union: total: " << tm.total() << "\n";

	return Gu;
    }

#if EXTERNAL_SPTRANS
    void import_transpose_sptrans_scan( const GraphCSx & Gcsr, unsigned P ) {
	sptrans_scanTrans<VID,int>(
	    n, n, m,
	    const_cast<EID *>( Gcsr.getIndex() ),
	    const_cast<VID *>( Gcsr.getEdges() ),
	    nullptr,
	    getIndex(),
	    getEdges(),
	    nullptr );
	getIndex()[n] = m;
    }

    void import_transpose_sptrans_merge( const GraphCSx & Gcsr, unsigned P ) {
	sptrans_mergeTrans<VID,int>(
	    n, n, m,
	    const_cast<EID *>( Gcsr.getIndex() ),
	    const_cast<VID *>( Gcsr.getEdges() ),
	    nullptr,
	    getIndex(),
	    getEdges(),
	    nullptr );
	getIndex()[n] = m;
    }
#endif

    void import_transpose_partitioned( const GraphCSx & Gcsr, unsigned P ) {
	timer tm;
	tm.start();
	
	assert( n == Gcsr.numVertices() );
	assert( m == Gcsr.numEdges() );
	assert( P >= 1 );

	// TODO: optimise partitioning by doing prefix sum followed by
	//       binary search to find the vertices closest to the cut points
	partitioner part( P, n );
	partitionBalanceEdges( Gcsr, part );
	std::cerr << "transpose: create partitioner: " << tm.next() << "\n";

	const EID * const g_index = Gcsr.getIndex();
	const VID * const g_edges = Gcsr.getEdges();

	// Lots of space, but this will be hypersparse. Should we first
	// figure out the per-partition sparsity pattern?
	mm::buffer<VID> * ctrs = new mm::buffer<VID>[P];
	for( unsigned p=0; p < P; ++p )
	    new ( &ctrs[p] ) mm::buffer<VID>(
		n+1, numa_allocation_local( part.numa_node_of( p ) ) );
	mm::buffer<VID> xref( m, numa_allocation_interleaved() );
	std::cerr << "transpose: setup: " << tm.next() << "\n";

	parallel_loop( (unsigned)0, P, [&]( unsigned p ) { 
	    std::fill( &ctrs[p][0], &ctrs[p][n+1], VID(0) );
	} );
	std::cerr << "transpose: init: " << tm.next() << "\n";

	map_partition( part, [&]( unsigned p ) {
	    VID * const ctrs_p = ctrs[p].get();
	    VID vs = part.start_of( p );
	    VID ve = part.end_of( p );
	    EID es = g_index[vs];
	    EID ee = g_index[ve];
	    for( EID e=es; e < ee; ++e )
		xref[e] = ctrs_p[g_edges[e]]++;
	} );
	std::cerr << "transpose: count: " << tm.next() << "\n";

	map_partition( part, [&]( unsigned q ) {
	    VID vs = part.start_of_vbal( q );
	    VID ve = part.end_of_vbal( q );
#if 0
	    for( VID v=vs; v < ve; ++v ) {
		VID deg = 0;
		for( unsigned p=0; p < P; ++p ) {
		    VID pdeg = ctrs[p][v];
		    // This conditional avoids *a lot* of memory traffic
		    // due to the hypersparsity of per-partition info.
		    // Omitting the store when ctrs[p][v] == 0 is correct, as
		    // this indicates no edges were mapped here, and we will
		    // not read this location in the future
		    if( pdeg != 0 ) {
			ctrs[p][v] = deg;
			deg += pdeg;
		    }
		}
		index[v] = (EID)deg;
	    }
#else
	    constexpr unsigned VL = 8; // MAX_VL
	    using vid_type = simd::ty<VID,VL>;
	    using eid_type = simd::ty<EID,VL>;
	    auto zero = simd::template create_zero<vid_type>();
	    VID v = vs;
	    for( ; v+VL <= ve; v += VL ) {
		auto deg = vid_type::traits::setzero();
		for( unsigned p=0; p < P; ++p ) {
		    auto pdeg = vid_type::traits::loadu( &ctrs[p][v] );
		    if( !vid_type::traits::is_zero( pdeg ) ) {
			vid_type::traits::storeu( &ctrs[p][v], deg );
			deg = vid_type::traits::add( deg, pdeg );
		    }
		}
		eid_type::traits::storeu(
		    &index[v],
		    conversion_traits<VID,EID,VL>::convert( deg ) );
	    }
	    for( ; v < ve; ++v ) {
		VID deg = 0;
		for( unsigned p=0; p < P; ++p ) {
		    VID pdeg = ctrs[p][v];
		    // This conditional avoids *a lot* of memory traffic
		    // due to the hypersparsity of per-partition info.
		    // Omitting the store when ctrs[p][v] == 0 is correct, as
		    // this indicates no edges were mapped here, and we will
		    // not read this location in the future
		    if( pdeg != 0 ) {
			ctrs[p][v] = deg;
			deg += pdeg;
		    }
		}
		index[v] = (EID)deg;
	    }
#endif
	} );
	std::cerr << "transpose: reduce degree: " << tm.next() << "\n";

	EID off = 0;
	for( VID s=0; s < n; ++s ) {
	    EID deg = index[s];
	    index[s] = off;
	    off += deg;
	}
	index[n] = off;
	assert( index[n] == m );
	std::cerr << "transpose: scan (seq): " << tm.next() << "\n";

/*
	map_partition( part, [&]( unsigned p ) {
	    VID * const ctrs_p = ctrs[p].get();
	    const VID * const g_edges = Gcsr.getEdges();
	    VID vs = part.start_of( p );
	    VID ve = part.end_of( p );
	    EID es = Gcsr.index[vs];
	    EID ee = Gcsr.index[ve];
	    for( EID e=es; e < ee; ++e ) {
		VID u = g_edges[e];
		// xref[e] += index[u] + ctrs_p[u];
		xref[e] = index[u] + ctrs_p[u]++;
	    }
	} );
	std::cerr << "transpose: cross-ref: " << tm.next() << "\n";
*/

	float * w = nullptr, * wg = nullptr;
	if( weights ) {
	    w = weights->get();
	    wg = Gcsr.weights->get();
	    assert( w && wg );
	}

	map_partition( part, [&]( unsigned p ) {
	    VID * const ctrs_p = ctrs[p].get();
	    VID vs = part.start_of( p );
	    VID ve = part.end_of( p );
	    EID e = g_index[vs];
	    for( VID v=vs; v < ve; ++v ) {
		// EID es = g_index[v];
		EID ee = g_index[v+1];
		for( ; e < ee; ++e ) {
		    VID u = g_edges[e];
		    VID off = xref[e] + ctrs_p[u]; // avoids random writes
		    EID idx = index[u] + off;
		    edges[idx] = v;

		    if( w )
			w[idx] = wg[e];
		}
	    }
	} );
	std::cerr << "transpose: place: " << tm.next() << "\n";

	// Because we are doing a transpose and keeping VIDs the same (no
	// remapping), sorting would be unnecessary in a sequential
	// implementation (insert sources in order traversed).
	// So in a partitioned transpose, with pre-defined per-partition
	// insertion points, we don't need sorting either.
	// Note that this is true even if the adjacency lists of the CSR are
	// not sorted!
	parallel_loop( (VID)0, n, [&]( VID s ) {
	    // assert( index[s] + (EID)ctrs[P-1][s] == index[s+1] );
	    assert( std::is_sorted( &edges[index[s]], &edges[index[s+1]] ) );
/*
	    assert( index[s] + (EID)ctrs[0][s] == index[s+1] );
	    if( w )
		paired_sort( &edges[index[s]], &edges[index[s+1]],
			     &w[index[s]] );
	    else
		std::sort( &edges[index[s]], &edges[index[s+1]] );
*/
	} );
	std::cerr << "transpose: sort (verify): " << tm.next() << "\n";

	xref.del();
	for( unsigned p=0; p < P; ++p )
	    ctrs[p].del();
	delete[] ctrs;
	build_degree();
	std::cerr << "transpose: build degree: " << tm.next() << "\n";

	std::cerr << "transpose: total: " << tm.total() << "\n";
    }

    void import_transpose_adjacency( const GraphCSx & Gcsr, unsigned P ) {
	timer tm;
	tm.start();
	
	assert( n == Gcsr.numVertices() );
	assert( m == Gcsr.numEdges() );
	assert( P >= 1 );

	// TODO: optimise partitioning by doing prefix sum followed by
	//       binary search to find the vertices closest to the cut points
	partitioner part( P, n );
	partitionBalanceEdges( Gcsr, part );
	std::cerr << "transpose: create partitioner: " << tm.next() << "\n";

	const EID * const g_index = Gcsr.getIndex();
	const VID * const g_edges = Gcsr.getEdges();

	// Lots of space, but this will be hypersparse. Should we first
	// figure out the per-partition sparsity pattern?
	using list_type = compact_list<VID>;
	mm::buffer<list_type> * adj = new mm::buffer<list_type>[P];
	for( unsigned p=0; p < P; ++p )
	    new ( &adj[p] ) mm::buffer<list_type>(
		n, numa_allocation_local( part.numa_node_of( p ) ) );
	std::cerr << "transpose: setup: " << tm.next() << "\n";

	map_partition( part, [&]( unsigned p ) {
	    std::fill( reinterpret_cast<intptr_t *>( &adj[p][0] ),
		       reinterpret_cast<intptr_t *>( &adj[p][n] ),
		       ~(intptr_t)0 );
	    /*
	    for( VID v=0; v < n; ++v ) {
		new ( &adj[p][v] ) list_type();
		// adj[p][v].reserve( g_index[v+1] - g_index[v] );
	    }
	    */
	} );
	std::cerr << "transpose: init: " << tm.next() << "\n";

	map_partition( part, [&]( unsigned p ) {
	    VID vs = part.start_of( p );
	    VID ve = part.end_of( p );
	    EID e = g_index[vs];
	    for( VID v=vs; v < ve; ++v ) {
		EID ee = g_index[v+1];
		for( ; e < ee; ++e )
		    adj[p][g_edges[e]].push_back( v );
	    }
	} );
	std::cerr << "transpose: place: " << tm.next() << "\n";

	map_partition( part, [&]( unsigned q ) {
	    VID vs = part.start_of_vbal( q );
	    VID ve = part.end_of_vbal( q );
	    for( VID v=vs; v < ve; ++v ) {
		VID deg = 0;
		for( unsigned p=0; p < P; ++p )
		    deg += adj[p][v].size();
		index[v] = (EID)deg;
	    }
	} );
	std::cerr << "transpose: count: " << tm.next() << "\n";

	EID off = 0;
	for( VID s=0; s < n; ++s ) {
	    EID deg = index[s];
	    index[s] = off;
	    off += deg;
	}
	index[n] = off;
	assert( index[n] == m );
	std::cerr << "transpose: scan (seq): " << tm.next() << "\n";

	float * w = nullptr, * wg = nullptr;
	if( weights ) {
	    w = weights->get();
	    wg = Gcsr.weights->get();
	    assert( w && wg );
	}

	map_partition( part, [&]( unsigned q ) {
	    VID vs = part.start_of( q );
	    VID ve = part.end_of( q );
	    EID e = index[vs];
	    for( VID v=vs; v < ve; ++v ) {
		for( unsigned p=0; p < P; ++p ) {
		    const auto & A = adj[p][v];
		    // std::copy( A.cbegin(), A.cend(), &edges[e] );
		    A.copy_to( &edges[e] );
		    e += adj[p][v].size();
		    // adj[p][v].~vector(); // cleanup
		    adj[p][v].~compact_list(); // cleanup
		}
		assert( e == index[v+1] );
	    }
	} );
	std::cerr << "transpose: re-assemble: " << tm.next() << "\n";

	// Because we are doing a transpose and keeping VIDs the same (no
	// remapping), sorting would be unnecessary in a sequential
	// implementation (insert sources in order traversed).
	// So in a partitioned transpose, with pre-defined per-partition
	// insertion points, we don't need sorting either.
	// Note that this is true even if the adjacency lists of the CSR are
	// not sorted!
	parallel_loop( (VID)0, n, [&]( VID s ) { 
	    // assert( index[s] + (EID)ctrs[P-1][s] == index[s+1] );
	    assert( std::is_sorted( &edges[index[s]], &edges[index[s+1]] ) );
/*
	    assert( index[s] + (EID)ctrs[0][s] == index[s+1] );
	    if( w )
		paired_sort( &edges[index[s]], &edges[index[s+1]],
			     &w[index[s]] );
	    else
		std::sort( &edges[index[s]], &edges[index[s+1]] );
*/
	} );
	std::cerr << "transpose: sort (verify): " << tm.next() << "\n";

	for( unsigned p=0; p < P; ++p )
	    adj[p].del();
	delete[] adj;
	build_degree();
	std::cerr << "transpose: build degree: " << tm.next() << "\n";

	std::cerr << "transpose: total: " << tm.total() << "\n";
    }

    void import_transpose_hybrid( const GraphCSx & Gcsr, unsigned P ) {
	// 1. Estimate in-degree by out-degree of v
	// 2. Select D
	// 3. if deg+(v) > D, then use per-partition counter
	// 4. if deg+(v) <= D, then use shared counter
	std::cerr << "transpose (hybrid)...\n";

	timer tm;
	tm.start();

	using SVID = std::make_signed_t<VID>;
	using SEID = std::make_signed_t<EID>;
	
	assert( n == Gcsr.numVertices() );
	assert( m == Gcsr.numEdges() );
	assert( P >= 1 );

	partitioner part( P, n );
	partitionBalanceEdges( Gcsr, part );
	std::cerr << "transpose: create partitioner: " << tm.next() << "\n";

	const EID * idx = Gcsr.getIndex();
	const VID * edges = Gcsr.getEdges();

	EID * tr_idx = getIndex();
	VID * tr_edges = getEdges();
	
	// Determine cut-off degree. Estimate in-degree by out-degree of vertex.
	// Count number of 'high-degree' vertices
	const VID D = atoi( getenv( "GRAPTOR_D" ) ); // 4096;
	VID n_high = sequence::reduce<VID>(
	    (VID)0, n, addF<VID>(), selectVDeg<VID,EID>( idx, D ) );

	// Create counters. First allocate compacted arrays per partition
	// with n_high counters.
	mm::buffer<VID> * hcnt = new mm::buffer<VID>[P+1];
	map_partition( part, [&]( unsigned p ) {
	    new ( &hcnt[p] ) mm::buffer<VID>(
		n_high,
		numa_allocation_local( part.part_of( p ) ) );
	    
	    std::fill( &hcnt[p][0], &hcnt[p][n_high], (EID)0 );
	} );
	new ( &hcnt[P] ) mm::buffer<VID>(
	    n_high, numa_allocation_interleaved() );
	std::fill( &hcnt[P][0], &hcnt[P][n_high], (EID)0 );

	mm::buffer<VID> lcnt( n+1, numa_allocation_partitioned( part ) );
	mm::buffer<VID> xref( m, numa_allocation_interleaved() );
	// mm::buffer<uint64_t> vkind( (m+63)/64, numa_allocation_interleaved() );
#if 0
	mm::buffer<VID> rxp( m, numa_allocation_interleaved() );
#endif
	std::cerr << "transpose: setup: " << tm.next() << "\n";

	VID next_hi = 0;
#if 0
	VID vv;
	for( vv=0; vv+63 < n; vv += 64 ) {
	    uint64_t bmp = 0;
	    for( VID vi=0; vi < 64; ++vi ) {
		VID v = vv+vi;
		bool is_high = idx[v+1] - idx[v] > D;
		if( is_high ) {
		    lcnt[v] = next_hi++;
		    bmp |= uint64_t(1) << vi;
		} else
		    lcnt[v] = 0;
	    }
	    vkind[vv/64] = bmp;
	}
	uint64_t bmp = 0;
	for( VID v=vv; v < n; ++v ) {
	    bool is_high = idx[v+1] - idx[v] > D;
	    if( is_high ) {
		lcnt[v] = next_hi++;
		bmp |= uint64_t(1) << ( vv - v );
	    } else
		lcnt[v] = 0;
	}
	vkind[vv/64] = bmp;
#endif
	constexpr VID hibit_mask = VID(1) << ( sizeof(VID)*8-1 );
	constexpr EID hibit_emask = EID(1) << ( sizeof(EID)*8-1 );
	constexpr EID sort_emask = EID(1) << ( sizeof(EID)*8-2 );
	constexpr EID bits_emask = EID(3) << ( sizeof(EID)*8-2 );
#if 0 // sequential init
	for( VID v=0; v < n; ++v ) {
	    bool is_high = idx[v+1] - idx[v] > D;
	    if( is_high )
		lcnt[v] = next_hi++ | hibit_mask;
	    else
		lcnt[v] = 0;
	}
	assert( next_hi == n_high );
#else // parallel init
	VID * n_high_part = new VID[P];
	map_partition( part, [=,&part]( unsigned p ) {
	    VID vs = part.start_of_vbal( p );
	    VID ve = part.end_of_vbal( p );
	    VID cnt = 0;
	    for( VID v=vs; v < ve; ++v ) {
		bool is_high = idx[v+1] - idx[v] > D;
		if( is_high )
		    ++cnt;
	    }
	    n_high_part[p] = cnt;
	} );
	VID cum = 0;
	for( unsigned p=0; p < P; ++p ) {
	    VID tmp = n_high_part[p];
	    n_high_part[p] = cum;
	    cum += tmp;
	}
	assert( cum == n_high );
	map_partition( part, [=,&part]( unsigned p ) mutable {
	    VID vs = part.start_of_vbal( p );
	    VID ve = part.end_of_vbal( p );
	    VID next_hi = n_high_part[p];
	    for( VID v=vs; v < ve; ++v ) {
		bool is_high = idx[v+1] - idx[v] > D;
		if( is_high )
		    lcnt[v] = next_hi++ | hibit_mask;
		else
		    lcnt[v] = 0;
	    }
	} );
	delete[] n_high_part;
#endif
	std::cerr << "transpose: D=" << D << " n_high=" << n_high
		  << " n=" << n << " m=" << m << "\n";
	std::cerr << "transpose: init (partially sequential): "
		  << tm.next() << "\n";

#if 0
	// Expand row indices
	map_partition( part, [=,&part]( unsigned p ) mutable {
	    VID vs = part.start_of( p );
	    VID ve = part.end_of( p );
	    EID e = idx[vs];
	    for( VID v=vs; v < ve; ++v ) {
		EID ee = idx[v+1];
		std::fill( &rxp[e], &rxp[ee], v );
		e = ee;
	    }
	} );
	std::cerr << "transpose: expand row indices: " << tm.next() << "\n";
#endif

	// Count occurences of each vertex
	// Options:
	//  - use read-only array for indices; updated array for counts
	//  - split loop, iterate twice, once for HDV, once for LDV
	//    (branch hard to predict; false sharing on read-only data)
	//  - clzero
	//  - 
	map_partition( part, [=,&part]( unsigned p ) mutable {
	    VID vs = part.start_of( p );
	    VID ve = part.end_of( p );
	    EID es = idx[vs];
	    EID ee = idx[ve];
	    VID * hcnt_p = &hcnt[p][0];
	    for( EID e=es; e < ee; ++e ) {
		// if( (e & 15) == 0 ) clzero( &xref[e] );
		if( (e & 15) == 0 ) {
		    // __asm__( "\n\tclzero" : : "a"(&xref[e]) : );
		    char * p = reinterpret_cast<char *>( &xref[e] );
		    __m128i z = _mm_setzero_si128();
		    _mm_stream_si128( (__m128i *)&p[0], z );
		    _mm_stream_si128( (__m128i *)&p[16], z );
		    _mm_stream_si128( (__m128i *)&p[32], z );
		    _mm_stream_si128( (__m128i *)&p[48], z );
		}
		// if( (e & 15) == 0 ) _mm_prefetch( &edges[e+TR_DIST], _MM_HINT_NTA );
		VID v = edges[e];
		VID lcnt_v = lcnt[v];
		if( (SVID)lcnt_v < (SVID)0 )
		    xref[e] = hcnt_p[lcnt_v & ~hibit_mask]++;
		else {
		    // Subject to false sharing as well as conflicts
		    xref[e] = __sync_fetch_and_add( &lcnt[v], 1 );
		}
	    }
	} );
	std::cerr << "transpose: count: " << tm.next() << "\n";

	// Do per-vertex vertical scans
#if 0
	parallel_loop( (VID)0, n, [&]( VID v ) { 
	    if( (SVID)lcnt[v] < (SVID)0 ) {
		VID sum = 0;
		VID vv = lcnt[v] & ~hibit_mask;
		for( unsigned p=0; p < P; ++p ) {
		    VID tmp = hcnt[p][vv];
		    if( tmp != 0 ) { // avoid unnecessary stores
			hcnt[p][vv] = sum;
			sum += tmp;
		    }
		}
		tr_idx[v] = sum;
	    } else
		tr_idx[v] = lcnt[v];
	} );
	std::cerr << "transpose: vertical scan: " << tm.next() << "\n";
#else
	// TODO: Easy to vectorize
	parallel_loop( (VID)0, n_high, [&]( VID vv ) { 
	    VID sum = 0;
	    for( unsigned p=0; p < P; ++p ) {
		VID tmp = hcnt[p][vv];
		if( tmp != 0 ) { // avoid unnecessary stores
		    hcnt[p][vv] = sum;
		    sum += tmp;
		}
	    }
	    hcnt[P][vv] = sum;
	} );
	
/*
	const VID * hcnt_P = &hcnt[P][0];
	parallel_for( VID v=0; v < n; ++v ) {
	    if( (SVID)lcnt[v] < (SVID)0 ) {
		VID vv = lcnt[v] & ~hibit_mask;
		tr_idx[v] = hcnt_P[vv];
	    } else
		tr_idx[v] = lcnt[v];
	}
*/
	std::cerr << "transpose: vertical scan: " << tm.next() << "\n";
#endif
	
	// Do prefix sums to determine insertion points
#if 0 // sequential prefix sum
	EID ins = 0;
	for( VID v=0; v < n; ++v ) {
	    EID deg = tr_idx[v];
	    tr_idx[v] = ins;
	    if( (SVID)lcnt[v] < (SVID)0 ) {
		tr_idx[v] |= hibit_emask;
		lcnt[v] &= ~hibit_mask;
	    }
	    ins += deg;
	}
	assert( ins == m );
	tr_idx[n] = m;
	std::cerr << "transpose: prefix sum (seq): " << tm.next() << "\n";
#else // parallel prefix sum
	EID * ps_cnt = new EID[P];
	map_partition( part, [=,&part]( unsigned p ) {
	    VID vs = part.start_of( p );
	    VID ve = part.end_of( p );
	    EID cnt = 0;
	    const VID * hcnt_P = &hcnt[P][0];
	    for( VID v=vs; v < ve; ++v ) {
		VID val;
		VID lc = lcnt[v];
		if( (SVID)lc < (SVID)0 ) {
		    VID vv = lc & ~hibit_mask;
		    val = hcnt_P[vv];
		} else
		    val = lc;
		cnt += val;
		tr_idx[v] = val;
	    }
	    ps_cnt[p] = cnt;
	} );
	EID ps_sum = 0;
	for( unsigned p=0; p < P; ++p ) {
	    EID tmp = ps_cnt[p];
	    ps_cnt[p] = ps_sum;
	    ps_sum += tmp;
	}
	assert( ps_sum == m );
	map_partition( part, [=,&part]( unsigned p ) mutable {
	    VID vs = part.start_of( p );
	    VID ve = part.end_of( p );
	    EID ins = ps_cnt[p];
	    for( VID v=vs; v < ve; ++v ) {
		EID deg = tr_idx[v];
		tr_idx[v] = ins;
		if( (SVID)lcnt[v] < (SVID)0 ) {
		    tr_idx[v] |= hibit_emask;
		    lcnt[v] &= ~hibit_mask;
		}
		ins += deg;
	    }
	} );
	tr_idx[n] = m;
	delete[] ps_cnt;
	
	std::cerr << "transpose: prefix sum (par): " << tm.next() << "\n";
#endif

#if 0
	// Pre-study: idea is to have only one random access pattern per loop.
	// That doesn't seem important...
	map_partition( part, [&]( unsigned p ) {
	    VID vs = part.start_of( p );
	    VID ve = part.end_of( p );
	    EID es = idx[vs];
	    EID ee = idx[ve];
	    for( EID e=es; e < ee; ++e ) {
		VID u = edges[e];
		if( ( vkind[u/64] >> (u & 63) ) & 1 ) { // random read
		    // xref[e] += tr_idx[u] + hcnt[p][lcnt[u]]; // random read
		    xref[e] += hcnt[p][lcnt[u]]; // random read
		    // } else {
		    // xref[e] += tr_idx[u]; // random read
		}
	    }
	} );
	std::cerr << "transpose: place pre-study: " << tm.next() << "\n";

	// Populate adjacency lists
#if 0
	constexpr unsigned short VL = MAX_VL;
	using vid_type = simd::ty<VID,VL>;
	using eid_type = simd::ty<EID,VL>;

	map_partition( part, [=]( unsigned p ) {
	    VID vs = part.start_of( p );
	    VID ve = part.end_of( p );
	    EID es = idx[vs];
	    EID ee = idx[ve];
	    EID eo = es;
	    for( ; eo+VL-1 < ee; eo += VL ) {
		auto v_edges = vid_type::traits::loadu( &edges[eo] );
		auto v_xref = vid_type::traits::loadu( &xref[eo] );
		auto v_tr_idx = eid_type::traits::gather( &tr_idx[0], v_edges );
		auto e_xref = conversion_traits<VID,EID,VL>::convert( v_xref );
		auto v_rxp = vid_type::traits::loadu( &rxp[eo] );
		auto e_ins = eid_type::traits::add( e_xref, v_tr_idx );
		vid_type::traits::scatter( &tr_edges[0], e_ins, v_rxp );
	    }
	    for( EID e=eo; e < ee; ++e ) {
		VID u = edges[e];
		EID ins = xref[e] + tr_idx[u]; // random read
		tr_edges[ins] = rxp[e]; // random write
	    }
	} );
#else // rxp
	map_partition( part, [=,&part]( unsigned p ) mutable {
	    VID vs = part.start_of( p );
	    VID ve = part.end_of( p );
	    EID e = idx[vs];
	    for( VID v=vs; v < ve; ++v ) {
		// EID es = idx[v];
		EID ee = idx[v+1];
		for( ; e < ee; ++e ) {
		    VID u = edges[e];
		    EID ins = xref[e] + tr_idx[u]; // random read
		    tr_edges[ins] = v; // random write
		}
	    }
	    assert( e == idx[ve] );
	} );
#endif
	
#else
#if 1 // scalar place
	// Populate adjacency lists
	map_partition( part, [=,&part]( unsigned p ) {
	    VID vs = part.start_of( p );
	    VID ve = part.end_of( p );
	    EID e = idx[vs];
	    const auto * hcnt_p = &hcnt[p][0];
	    for( VID v=vs; v < ve; ++v ) {
		EID ee = idx[v+1];
		for( ; e < ee; ++e ) {
		    VID u = edges[e];
		    EID ins = xref[e] + tr_idx[u];
		    if( (SEID)tr_idx[u] < (SEID)0 ) {
			ins &= ~hibit_emask; // remove flag bit
			ins += hcnt_p[lcnt[u]]; // avoids random writes to lcnt
		    }
		    tr_edges[ins] = v;
		}
	    }
	    assert( e == idx[ve] );
	} );
#else // vector place
	constexpr unsigned short VL = MAX_VL;
	using vid_type = simd::ty<VID,VL>;
	using svid_type = simd::ty<SVID,VL>;
	using eid_type = simd::ty<EID,VL>;
	using vt = typename vid_type::traits;
	using et = typename eid_type::traits;

	map_partition( part, [=,&part]( unsigned p ) {
	    auto e_hibit = et::slli( et::setone(), et::B-1 );
	    const auto * hcnt_p = &hcnt[p][0];
	    VID vs = part.start_of( p );
	    VID ve = part.end_of( p );
	    EID es = idx[vs];
	    EID ee = idx[ve];
	    EID eo = es;
	    for( ; eo+VL-1 < ee; eo += VL ) {
		auto v_edges = vt::loadu( &edges[eo] );
		auto v_xref = vt::loadu( &xref[eo] );
		auto e_tr_idx = et::gather( &tr_idx[0], v_edges );
		auto e_tr_idx_c = et::bitwise_andnot( e_hibit, e_tr_idx );
		auto v_rxp = vt::loadu( &rxp[eo] );
		auto hh = et::cmpge( e_tr_idx, e_hibit, et::mt_preferred() );
#if __AVX512F__
		auto h = hh;
#else
		auto h = conversion_traits<
		    logical<sizeof(EID)>,logical<sizeof(VID)>,VL>
		    ::convert( e_tr_idx );
#endif
		typename vt::type v_ins0;
		// if( vt::is_all_false( h ) ) {
		if( vid_type::prefmask_traits::traits::is_all_false( h ) ) {
		    v_ins0 = v_xref;
		} else {
		    auto v_lcnt = vt::gather( &lcnt[0], v_edges, h );
		    auto v_hcnt0 = vt::gather( hcnt_p, v_lcnt, h );
		    auto v_hcnt = vt::blend( h, vt::setzero(), v_hcnt0 );
		    v_ins0 = vt::add( v_xref, v_hcnt );
		}
		auto e_ins0 = conversion_traits<VID,EID,VL>::convert( v_ins0 );
		auto e_ins = et::add( e_tr_idx_c, e_ins0 );
		vt::scatter( &tr_edges[0], e_ins, v_rxp );
	    }
	    for( EID e=eo; e < ee; ++e ) {
		VID u = edges[e];
		EID ins = xref[e] + tr_idx[u]; // random read
		if( (SEID)tr_idx[u] < (SEID)0 ) {
		    ins &= ~hibit_emask; // remove flag bit
		    ins += hcnt_p[lcnt[u]]; // avoids random writes to lcnt
		}
		tr_edges[ins] = rxp[e];
	    }
	} );
#endif // scalar/vector place
#endif // place with pre-study
	std::cerr << "transpose: place: " << tm.next() << "\n";

	// Sort short adjacency lists
	// Using edge balanced works better here than vertex balanced
	// Free-form parallel for is slightly better still
	parallel_loop( (VID)0, n, [&]( VID v ) { 
	    // if( idx[v+1] - idx[v] <= D ) { // check type on source graph
	    // if( !( ( vkind[v/64] >> (v & 63) ) & 1 ) ) {
	    // if( !( (SVID)lcnt[v] < (SVID)0 ) ) {
	    if( !( (SEID)tr_idx[v] < (SEID)0 ) ) {
		tr_idx[v] &= ~hibit_emask;
		EID vs = tr_idx[v];
		EID ve = tr_idx[v+1] & ~hibit_emask;
		EID tr_deg = ve - vs;
		if( tr_deg < 2 ) {
		} else if( tr_deg == 2 ) { // marginal benefit of special case
		    VID a = tr_edges[vs];
		    VID b = tr_edges[vs+1];
		    if( a > b ) {
			tr_edges[vs] = b;
			tr_edges[vs+1] = a;
		    }
		} else {
		    // consider https://arxiv.org/pdf/1704.08579.pdf
		    // and ips4o
		    std::sort( &tr_edges[vs], &tr_edges[ve] );
		}
	    }
	} );
	std::cerr << "transpose: sort: " << tm.next() << "\n";

#if 0
	// Verify sortedness
	parallel_loop( (VID)0, n, [&]( VID s ) { 
	    assert( std::is_sorted(
			&tr_edges[tr_idx[s]], &tr_edges[tr_idx[s+1]] ) );
	} );
	std::cerr << "transpose: sort (verify): " << tm.next() << "\n";
#endif

	// Cleanup
	// rxp.del();
	// vkind.del();
	xref.del();
	for( unsigned p=0; p <= P; ++p )
	    hcnt[p].del();
	delete[] hcnt;
	lcnt.del();

	build_degree();
	std::cerr << "transpose: build degree: " << tm.next() << "\n";

	std::cerr << "transpose: total: " << tm.total() << "\n";
    }

    void import_transpose_pull( const GraphCSx & Gcsr, unsigned P ) {
	// 1. Estimate in-degree by out-degree of v
	// 2. Select D
	// 3. if deg+(v) > D, then use per-partition counter
	// 4. if deg+(v) <= D, then use shared counter
	std::cerr << "transpose (pull)...\n";

	timer tm;
	tm.start();
	
	assert( n == Gcsr.numVertices() );
	assert( m == Gcsr.numEdges() );
	assert( P >= 1 );

	partitioner part( P, n );
	partitionBalanceEdges( Gcsr, part );
	std::cerr << "transpose: create partitioner: " << tm.next() << "\n";

	const EID * idx = Gcsr.getIndex();
	const VID * edges = Gcsr.getEdges();

	EID * tr_idx = getIndex();
	VID * tr_edges = getEdges();
	
	// Determine cut-off degree. Estimate in-degree by out-degree of vertex.
	// Count number of 'high-degree' vertices
	const VID D = atoi( getenv( "GRAPTOR_D" ) ); // 4096;
	VID n_high = sequence::reduce<VID>(
	    (VID)0, n, addF<VID>(), selectVDeg<VID,EID>( idx, D ) );

	// Create counters. First allocate compacted arrays per partition
	// with n_high counters.
	mm::buffer<VID> * hcnt = new mm::buffer<VID>[P];
	map_partition( part, [&]( unsigned p ) {
	    new ( &hcnt[p] ) mm::buffer<VID>(
		n_high,
		numa_allocation_local( part.part_of( p ) ) );
	    
	    std::fill( &hcnt[p][0], &hcnt[p][n_high], (EID)0 );
	} );
	mm::buffer<VID> lcnt( n+1, numa_allocation_partitioned( part ) );
	mm::buffer<EID> last( n, numa_allocation_partitioned( part ) );
	mm::buffer<EID> link( m, numa_allocation_interleaved() );
	mm::buffer<VID> xref( m, numa_allocation_interleaved() );
	mm::buffer<uint64_t> vkind( (m+63)/64, numa_allocation_interleaved() );
	mm::buffer<VID> rxp( m, numa_allocation_interleaved() );
	std::cerr << "transpose: setup: " << tm.next() << "\n";

	VID next_hi = 0;
	VID vv;
	for( vv=0; vv+63 < n; vv += 64 ) {
	    uint64_t bmp = 0;
	    for( VID vi=0; vi < 64; ++vi ) {
		VID v = vv+vi;
		bool is_high = idx[v+1] - idx[v] > D;
		if( is_high ) {
		    lcnt[v] = next_hi++;
		    bmp |= uint64_t(1) << vi;
		} else
		    lcnt[v] = 0;
		last[v] = std::numeric_limits<EID>::max();
	    }
	    vkind[vv/64] = bmp;
	}
	uint64_t bmp = 0;
	for( VID v=vv; v < n; ++v ) {
	    bool is_high = idx[v+1] - idx[v] > D;
	    if( is_high ) {
		lcnt[v] = next_hi++;
		bmp |= uint64_t(1) << ( vv - v );
	    } else
		lcnt[v] = 0;
	    last[v] = std::numeric_limits<EID>::max();
	}
	vkind[vv/64] = bmp;
	assert( next_hi == n_high );
	std::cerr << "transpose: D=" << D << " n_high=" << n_high
		  << " n=" << n << " m=" << m << "\n";
	std::cerr << "transpose: init (partially sequential): "
		  << tm.next() << "\n";

#if 1
	// Expand row indices
	map_partition( part, [&]( unsigned p ) {
	    VID vs = part.start_of( p );
	    VID ve = part.end_of( p );
	    EID e = idx[vs];
	    for( VID v=vs; v < ve; ++v ) {
		EID ee = idx[v+1];
		std::fill( &rxp[e], &rxp[ee], v );
		e = ee;
	    }
	} );
	std::cerr << "transpose: expand row indices: " << tm.next() << "\n";
#endif

	// Count occurences of each vertex
	map_partition( part, [&]( unsigned p ) {
	    VID vs = part.start_of( p );
	    VID ve = part.end_of( p );
	    EID es = idx[vs];
	    EID ee = idx[ve];
	    VID * xref_p = xref.get();
	    EID * link_p = link.get();
	    EID * last_p = last.get();
	    VID * lcnt_p = lcnt.get();
	    VID * hcnt_p = hcnt[p].get();
	    uint64_t * vkind_p = vkind.get();
	    for( EID e=es; e < ee; ++e ) {
		VID v = edges[e];
		// if( idx[v+1] - idx[v] > D )
		if( __builtin_expect( ( vkind_p[v/64] >> (v & 63) ) & 1, 1 ) )
		    xref_p[e] = hcnt_p[lcnt_p[v]]++;
		else {
		    EID old_link = __atomic_exchange_n(
			&last_p[v], e, __ATOMIC_RELAXED );
		    link_p[e] = old_link;
		    // ... consider counting in different array to avoid sharing with read-only high-degree info in lcnt (false sharing) -- applies to hybrid version also
		    __sync_fetch_and_add( &lcnt_p[v], 1 );
		}
	    }
	} );
	std::cerr << "transpose: count: " << tm.next() << "\n";

	// Do per-vertex vertical scans
	parallel_loop( (VID)0, n, [&]( VID v ) { 
	    if( ( vkind[v/64] >> (v & 63) ) & 1 ) {
		// if( idx[v+1] - idx[v] > D ) {
		VID sum = 0;
		VID vv = lcnt[v];
		for( unsigned p=0; p < P; ++p ) {
		    VID tmp = hcnt[p][vv];
		    if( tmp != 0 ) { // avoid unnecessary stores
			hcnt[p][vv] = sum;
			sum += tmp;
		    }
		}
		tr_idx[v] = sum;
	    } else
		tr_idx[v] = lcnt[v];
	} );
	std::cerr << "transpose: vertical scan: " << tm.next() << "\n";
	
	// Do prefix sums to determine insertion points
	EID ins = 0;
	for( VID v=0; v < n; ++v ) {
	    EID deg = tr_idx[v];
	    tr_idx[v] = ins;
	    ins += deg;
	}
	assert( ins == m );
	tr_idx[n] = m;
	std::cerr << "transpose: prefix sum (seq): " << tm.next() << "\n";

	// Populate adjacency lists
	map_partition( part, [&]( unsigned p ) {
	    VID vs = part.start_of( p );
	    VID ve = part.end_of( p );
	    EID e = idx[vs];
	    for( VID v=vs; v < ve; ++v ) {
		// EID es = idx[v];
		EID ee = idx[v+1];
		for( ; e < ee; ++e ) {
		    VID u = edges[e];
		    if( ( vkind[u/64] >> (u & 63) ) & 1 ) {
		    // if( idx[u+1] - idx[u] > D ) {
			VID off = xref[e] + hcnt[p][lcnt[u]]; // avoids random writes
			EID ins = tr_idx[u] + off;
			// VID ins = xref[e] + hcnt[p][lcnt[u]]; // avoids random writes

			// EID ins = hcnt[p][lcnt[u]]++;
			tr_edges[ins] = v;
		    } else {
			// skip - pull-based loop
			// EID ins = tr_idx[u] + xref[e];
			// tr_edges[ins] = v;
		    }
		}
	    }
	    assert( e == idx[ve] );
	} );
	std::cerr << "transpose: place (push): " << tm.next() << "\n";

	map_partition( part, [&]( unsigned q ) {
	    EID * sublists = new EID[P];
	    const EID undef = std::numeric_limits<EID>::max();
	    std::fill( &sublists[0], &sublists[P], undef );
	    VID vs = part.start_of( q );
	    VID ve = part.end_of( q );
	    for( VID v=vs; v < ve; ++v ) {
		if( last[v] == undef ) {
		    // nothing to do
		    // Either 0-degree or high-degree type
		} else if( lcnt[v] == 1 ) {
		    tr_edges[tr_idx[v]] = rxp[last[v]];
		} else if( lcnt[v] == 2 ) {
		    VID a = rxp[last[v]];
		    VID b = rxp[link[last[v]]];
		    if( a < b ) {
			tr_edges[tr_idx[v]] = a;
			tr_edges[tr_idx[v]+1] = b;
		    } else {
			tr_edges[tr_idx[v]] = b;
			tr_edges[tr_idx[v]+1] = a;
		    }
		} else {
#if 0
		    EID next_e;
		    for( EID e=last[v]; e != undef; e=next_e ) {
			unsigned p = part.part_of( rxp[e] );
			EID nxt = sublists[p];
			sublists[p] = e;
			next_e = link[e];
			link[e] = nxt;
		    }

		    EID v_idx = tr_idx[v];
		    for( unsigned p=0; p < P; ++p ) {
			for( EID e=sublists[p]; e != undef; e=link[e] ) {
			    tr_edges[v_idx] = rxp[e];
			    v_idx++;
			}
			sublists[p] = undef;
		    }
#else
		    EID p = tr_idx[v];
		    for( EID e=last[v]; e != undef; e=link[e] )
			tr_edges[p++] = rxp[e];
		    std::sort( &tr_edges[tr_idx[v]], &tr_edges[p] );
#endif
		}
	    }
	    delete[] sublists;
	} );

	std::cerr << "transpose: place (pull): " << tm.next() << "\n";

	// Sort short adjacency lists
	// Using edge balanced works better here than vertex balanced
	// Free-form parallel for is slightly better still
	/*
	map_partition( part, [&]( unsigned p ) {
	    VID vs = part.start_of( p );
	    VID ve = part.end_of( p );
	    for( VID v=vs; v < ve; ++v ) {
		EID deg = idx[v+1] - idx[v];
		if( deg > 1 && deg <= D )
		    std::sort( &tr_edges[tr_idx[v]], &tr_edges[tr_idx[v+1]] );
	    }
	} );
	*/
#if 0
	parallel_loop( (VID)0, n, [&]( VID v ) { 
	    // if( idx[v+1] - idx[v] <= D ) { // check type on source graph
	    if( !( ( vkind[v/64] >> (v & 63) ) & 1 ) ) {
		EID vs = tr_idx[v];
		EID ve = tr_idx[v+1];
		EID tr_deg = ve - vs;
		if( tr_deg < 2 ) {
		} else if( tr_deg == 2 ) { // marginal benefit of special case
		    VID a = tr_edges[vs];
		    VID b = tr_edges[vs+1];
		    if( a > b ) {
			tr_edges[vs] = b;
			tr_edges[vs+1] = a;
		    }
		} else {
		    std::sort( &tr_edges[vs], &tr_edges[ve] );
		}
	    }
	} );
	std::cerr << "transpose: sort: " << tm.next() << "\n";
#endif

	// Verify sortedness
	parallel_loop( (VID)0, n, [&]( VID s ) { 
/* not when using xref
	    if( idx[s+1] - idx[s] > D )
		assert( hcnt[P-1][lcnt[s]] == tr_idx[s+1] );
	    else
		assert( lcnt[s] == tr_idx[s+1] );
*/
	    // assert( std::is_sorted( &edges[idx[s]], &edges[idx[s+1]] ) );
	    assert( std::is_sorted(
			&tr_edges[tr_idx[s]], &tr_edges[tr_idx[s+1]] ) );
	} );
	std::cerr << "transpose: sort (verify): " << tm.next() << "\n";

	// Cleanup
	rxp.del();
	vkind.del();
	xref.del();
	link.del();
	last.del();
	for( unsigned p=0; p < P; ++p )
	    hcnt[p].del();
	delete[] hcnt;
	lcnt.del();

	build_degree();
	std::cerr << "transpose: build degree: " << tm.next() << "\n";

	std::cerr << "transpose: total: " << tm.total() << "\n";
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

	constexpr size_t THRESHOLD = size_t(1) << 32; // 4 GiB
	std::stringstream buffer;

	buffer << ( weights ? "WeightedAdjacencyGraph\n" : "AdjacencyGraph\n" );
	buffer << (uint64_t)n << "\n"
	       << (uint64_t)m << "\n";

	for( VID v=0; v < n; ++v ) {
	    if( (size_t)buffer.tellp() >= THRESHOLD ) {
		file << buffer.rdbuf();
		std::stringstream().swap(buffer);
	    }
	    buffer << index[v] << '\n';
	}

	for( EID e=0; e < m; ++e ) {
	    if( (size_t)buffer.tellp() >= THRESHOLD ) {
		file << buffer.rdbuf();
		std::stringstream().swap(buffer);
	    }
	    buffer << edges[e] << '\n';
	}

	if( weights ) {
	    float * w = weights->get();
	    buffer.precision( std::numeric_limits<float>::max_digits10 );
	    buffer << std::fixed;
	    for( EID e=0; e < m; ++e ) {
		if( (size_t)buffer.tellp() >= THRESHOLD ) {
		    file << buffer.rdbuf();
		    std::stringstream().swap(buffer);
		}
		buffer << w[e] << '\n';
	    }
	}
	    
	file << buffer.rdbuf();
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
#if 0
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
	parallel_loop( (VID)0, n, [&]( VID v ) { 
	    index[v] = index_p[v];
	} );
	index[n] = m;

	const VID *edges_p = reinterpret_cast<const VID *>(
	    data+sizeof(uint64_t)*8+n*sizeof(EID) );
	parallel_loop( (EID)0, m, [&]( EID e ) { 
	    edges[e] = edges_p[e];
	} );

	munmap( (void *)data, len );
	close( fd );
	std::cerr << "Reading file done" << std::endl;
#else
	// Read using read() with large chunks
	std::cerr << "Reading (using parallel read) file "
		  << ifile << std::endl;
	int fd;

	if( (fd = open( ifile.c_str(), O_RDONLY )) < 0 ) {
	    std::cerr << "Cannot open file '" << ifile << "': "
		      << strerror( errno ) << "\n";
	    exit( 1 );
	}

	uint64_t header[8];
	ssize_t sz = read( fd, (char *)header, sizeof(header) );
	if( sz != sizeof(header) ) {
	    std::cerr << "Reading file '" << ifile << "' failed: "
		      << strerror( errno ) << "\n";
	    exit( 1 );
	}
	    
	if( header[0] != 2 && header[0] != 3 ) {
	    std::cerr << "Only accepting version 2 or 3 files\n";
	    exit( 1 );
	}
	n = header[2];
	m = header[3];
	assert( sizeof(VID) == header[4] );
	assert( sizeof(EID) == header[5] );

	allocate( alloc );

	parallel_read( fd, sizeof(header), index.get(), sizeof(EID)*n );
	index[n] = m;
	size_t off = sizeof(header)+sizeof(EID)*n;
	if( header[0] == 3 )
	    off += sizeof(EID);
	parallel_read( fd, off, edges.get(), sizeof(VID)*m );

	close( fd );
	std::cerr << "Reading file done" << std::endl;
#endif
#endif
	build_degree();
    }

    void readWeightsFromBinaryFile( const std::string & wfile,
				    const numa_allocation & alloc ) {
#if 0
	std::cerr << "Reading (using mmap) edge weights file "
		  << wfile << std::endl;
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
	std::cerr << "Reading edge weights file done" << std::endl;
#else

	// Read using read() with large chunks
	std::cerr << "Reading (using parallel read) edge weights file "
		  << wfile << std::endl;
	int fd;

	if( (fd = open( wfile.c_str(), O_RDONLY )) < 0 ) {
	    std::cerr << "Cannot open file '" << wfile << "': "
		      << strerror( errno ) << "\n";
	    exit( 1 );
	}

	weights = new mm::buffer<float>( (size_t)m, alloc, "edge weights" );

	parallel_read( fd, 0, weights->get(), sizeof(float)*m );

	close( fd );
	std::cerr << "Reading edge weights file done" << std::endl;
#endif
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
	    parallel_loop( (VID)0, n, [&]( VID u ) { 
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
				// Assuming sorted list + symmetric
				assert( 0 && "need to find value in "
					"symmetric graph" );
				break;
			    }
			}
		    }
		}
	    } );
	} else {
	    parallel_loop( (VID)0, n, [&]( VID u ) {
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
	    } );
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
    bool hasNeighbor( VID v, VID ngh ) const {
	return std::binary_search( &edges[index[v]], &edges[index[v+1]], ngh );
    }

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
    std::pair<bool,float> getEdgeWeight( VID s, VID d ) const {
	bool ret = false;
	for( EID e=index[s]; e < index[s+1]; ++e )
	    if( edges[e] == d )
		return std::make_pair(
		    true, weights ? weights->get()[e]
		    : std::numeric_limits<float>::infinity() );
	return std::make_pair( false, 0.0f );
    }

    const mm::buffer<float> * getWeights() const { return weights; }

    bool * get_flags( const partitioner & part ) const {
	if( !flags.get() ) {
	    using std::swap;
	    mm::buffer<bool> buf( n, numa_allocation_partitioned( part ),
				  "GraphCSx flags array" );
	    map_vertexL( part, [&]( auto v ) { buf[v] = false; } );
	    swap( flags, buf );
	    buf.del();
	}
	return flags.get();
    }

private:
    void allocate( const numa_allocation & alloc ) {
	// Note: only interleaved and local supported
	// index.allocate( n+1, alloc );
	// edges.allocate( m, alloc );
	// degree.allocate( n, alloc );
	new (&index) mm::buffer<EID>( n+1, alloc, "CSx index" );
	new (&edges) mm::buffer<VID>( m, alloc, "CSx edges" );
	new (&degree) mm::buffer<VID>( n, alloc, "CSx degree" );
    }

    void allocateInterleaved() {
	allocate( numa_allocation_interleaved() );
    }
    void allocateLocal( int numa_node ) {
	allocate( numa_allocation_local( numa_node ) );
    }
public:
    void build_degree() {
	parallel_loop( (VID)0, n, [&]( VID v ) { 
	    degree[v] = index[v+1] - index[v];
	} );
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

