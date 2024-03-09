// -*- c++ -*-
// Specialised to MC

// TODO:
// * online machine learning
// * Look at Blocked and Binary matrix design:
//   + col_start and row_start redundant to each other
// * VIDs of 8 or 16 bits
// * Consider sorting vertices first by non-increasing degeneracy, secondly
//   by non-increasing degree within a group of equal degeneracy.
//   The non-increasing degree means faster reduction of size of P?

// Consider:
// + StackLikeAllocator PAGE_SIZE => mmap => high overhead, USAroad not needed
//   look at retaining chunks in persistent allocator, i.e., insert layer
//   that hands out pages to the SLAs?

#ifndef ABLATION_HADJPA_DISABLE_XP_HASH
#define ABLATION_HADJPA_DISABLE_XP_HASH 0
#endif

#ifndef ABLATION_BLOCKED_DISABLE_XP_HASH
#define ABLATION_BLOCKED_DISABLE_XP_HASH 0
#endif

#ifndef ABLATION_DENSE_DISABLE_XP_HASH
#define ABLATION_DENSE_DISABLE_XP_HASH 0
#endif

#ifndef ABLATION_BITCONSTRUCT_XP_VEC
#define ABLATION_BITCONSTRUCT_XP_VEC 0
#endif

#ifndef ABLATION_BLOCKED_EXCEED
#define ABLATION_BLOCKED_EXCEED 0
#endif

#ifndef ABLATION_DENSE_EXCEED
#define ABLATION_DENSE_EXCEED 0
#endif

#ifndef ABLATION_GENERIC_EXCEED
#define ABLATION_GENERIC_EXCEED 0
#endif

#ifndef ABLATION_DISABLE_LEAF
#define ABLATION_DISABLE_LEAF 1
#endif

#ifndef ABLATION_DISABLE_TOP_TINY
#define ABLATION_DISABLE_TOP_TINY 1
#endif

#ifndef ABLATION_DISABLE_TOP_DENSE
#define ABLATION_DISABLE_TOP_DENSE 1
#endif

// Not effective, so disable by default
#ifndef ABLATION_PDEG
#define ABLATION_PDEG 1
#endif

#ifndef ABLATION_DENSE_NO_PIVOT_TOP
#define ABLATION_DENSE_NO_PIVOT_TOP 0
#endif

#ifndef ABLATION_BLOCKED_NO_PIVOT_TOP
#define ABLATION_BLOCKED_NO_PIVOT_TOP 0
#endif

#ifndef ABLATION_DENSE_FILTER_FULLY_CONNECTED
#define ABLATION_DENSE_FILTER_FULLY_CONNECTED 1
#endif

#ifndef ABLATION_BLOCKED_FILTER_FULLY_CONNECTED
#define ABLATION_BLOCKED_FILTER_FULLY_CONNECTED 1
#endif

#ifndef ABLATION_DENSE_ITERATE
#define ABLATION_DENSE_ITERATE 0
#endif

#ifndef ABLATION_BLOCKED_ITERATE
#define ABLATION_BLOCKED_ITERATE 0
#endif

#ifndef ABLATION_DENSE_PIVOT_FILTER
#define ABLATION_DENSE_PIVOT_FILTER 0
#endif

#ifndef ABLATION_BLOCKED_PIVOT_FILTER
#define ABLATION_BLOCKED_PIVOT_FILTER 0
#endif

#ifndef USE_512_VECTOR
#if __AVX512F__
#define USE_512_VECTOR 1
#else
#define USE_512_VECTOR 0
#endif
#endif

#ifndef PAPI_REGION
#define PAPI_REGION 0
#endif

#include <signal.h>
#include <sys/time.h>

#include <thread>
#include <mutex>
#include <map>
#include <exception>
#include <numeric>

#include <pthread.h>

#if PAPI_REGION == 1
#include <papi.h>
#endif

#include "graptor/graptor.h"
#include "graptor/api.h"

#include "graptor/graph/contract/vertex_set.h"
#include "graptor/graph/GraphCSx.h"
#include "graptor/graph/partitioning.h"

#include "graptor/graph/simple/csx.h"
#include "graptor/graph/simple/dicsx.h"
#include "graptor/graph/simple/hadj.h"
#include "graptor/graph/simple/hadjt.h"
#include "graptor/graph/simple/cutout.h"
#include "graptor/graph/simple/dense.h"
#include "graptor/graph/simple/blocked.h"
#include "graptor/graph/simple/xp_set.h"
#include "graptor/graph/transform/rmself.h"

#include "graptor/container/bitset.h"
#include "graptor/container/hash_fn.h"
#include "graptor/container/intersect.h"

#ifndef TUNABLE_SMALL_AVOID_CUTOUT_LEAF
#define TUNABLE_SMALL_AVOID_CUTOUT_LEAF 0
#endif

#define NOBENCH
#define MD_ORDERING 0
#define VARIANT 11
#define FUSION 1
#include "../bench/KC_bucket.C"
#undef FUSION
#undef VARIANT
#undef MD_ORDERING
#undef NOBENCH

//! Choice of hash function for compilation unit
using hash_fn = graptor::rand_hash<uint32_t>;

#if ABLATION_BLOCKED_DISABLE_XP_HASH	\
    && ABLATION_HADJPA_DISABLE_XP_HASH
using HGraphTy = graptor::graph::GraphHAdjPA<VID,EID,false,hash_fn>;
#else
using HGraphTy = graptor::graph::GraphHAdjPA<VID,EID,true,hash_fn>;
#endif
// using HFGraphTy = graptor::graph::GraphHAdjPA<VID,EID,false,hash_fn>;
using HFGraphTy = graptor::graph::GraphHAdjPA<VID,EID,true,hash_fn>;

using graptor::graph::DenseMatrix;
using graptor::graph::BinaryMatrix;
using graptor::graph::BlockedBinaryMatrix;
using graptor::graph::PSet;

static constexpr size_t X_MIN_SIZE = 5;
static constexpr size_t X_DIM = 9 - X_MIN_SIZE + 1;
static constexpr size_t P_MIN_SIZE = 5;
static constexpr size_t P_DIM = 9 - P_MIN_SIZE + 1;
static constexpr size_t N_MIN_SIZE = 5;
static constexpr size_t N_DIM = 9 - N_MIN_SIZE + 1;

#if USE_512_VECTOR
static constexpr size_t X_MAX_SIZE = 9;
static constexpr size_t P_MAX_SIZE = 9;
static constexpr size_t N_MAX_SIZE = 9;
#else
static constexpr size_t X_MAX_SIZE = 8;
static constexpr size_t P_MAX_SIZE = 8;
static constexpr size_t N_MAX_SIZE = 8;
#endif


static bool verbose = false;

static std::mutex io_mux;
static constexpr bool io_trace = false;

class MC_Enumerator {
public:
    MC_Enumerator( size_t degen = 0 )
	: m_degeneracy( degen ),
	  m_best( 0 ) {
	m_timer.start();
    }

    // Record solution
    void record( size_t s ) {
	assert( s <= m_degeneracy+1 );
	if( s > m_best )
	    update_best( s );
    }

    // Feasability check
    bool is_feasible( size_t upper_bound ) const {
	return upper_bound > m_best.load( std::memory_order_relaxed );
    }

    size_t get_max_clique_size() const {
	return m_best.load( std::memory_order_relaxed );
    }

    std::ostream & report( std::ostream & os ) const {
	return os << "Maximum clique size: " << m_best.load() << "\n";
    }

private:
    void update_best( size_t s ) {
	size_t prior = m_best.load( std::memory_order_relaxed );
	while( s > prior )  {
	    if( m_best.compare_exchange_weak(
		    prior, s, 
		    std::memory_order_release,
		    std::memory_order_relaxed ) ) {
		std::cout << "max_clique: " << s << " at "
			  << m_timer.elapsed() << '\n';
		break;
	    }
	    prior = m_best.load( std::memory_order_relaxed );
	}
    }

private:
    size_t m_degeneracy;
    std::atomic<size_t> m_best;
    timer m_timer;
};

struct variant_statistics {
    variant_statistics()
	: m_tm( 0 ), m_max( std::numeric_limits<double>::min() ),
	  m_build( 0 ), m_calls( 0 ) { }
    variant_statistics( double tm, double mx, double bld, size_t calls )
	: m_tm( tm ), m_max( mx ), m_build( bld ), m_calls( calls ) { }

    variant_statistics operator + ( const variant_statistics & s ) const {
	return variant_statistics( m_tm + s.m_tm,
				   std::max( m_max, s.m_max ),
				   m_build + s.m_build,
				   m_calls + s.m_calls );
    }

    void record_build( double abld ) {
	m_build += abld;
    }
    void record( double atm ) {
	m_tm += atm;
	if( m_max < atm )
	    m_max = atm;
	++m_calls;
    }

    ostream & print( ostream & os ) const {
	return os << m_tm << " seconds in "
		  << m_calls << " calls @ "
		  << ( m_tm / double(m_calls) )
		  << " s/call; max " << m_max
		  << "; build " << ( m_build / double(m_calls) )
		  << "\n";
    }
    
    double m_tm, m_max, m_build;
    size_t m_calls;
};

struct all_variant_statistics {
    all_variant_statistics
    operator + ( const all_variant_statistics & s ) const {
	all_variant_statistics sum;
	for( size_t n=0; n < N_DIM; ++n ) {
	    sum.m_dense[n] = m_dense[n] + s.m_dense[n];
	    sum.m_leaf_dense[n] = m_leaf_dense[n] + s.m_leaf_dense[n];
	}
	for( size_t x=0; x < X_DIM; ++x )
	    for( size_t p=0; p < P_DIM; ++p ) {
		sum.m_blocked[x][p] = m_blocked[x][p] + s.m_blocked[x][p];
		sum.m_leaf_blocked[x][p]
		    = m_leaf_blocked[x][p] + s.m_leaf_blocked[x][p];
	    }
	sum.m_tiny = m_tiny + s.m_tiny;
	sum.m_gen = m_gen + s.m_gen;
	return sum;
    }

    void record_tiny( double atm ) { m_tiny.record( atm ); }
    void record_gen( double atm ) { m_gen.record( atm ); }
    void record_genbuild( double atm ) { m_gen.record_build( atm ); }

    variant_statistics & get( size_t n ) {
	return m_dense[n-N_MIN_SIZE];
    }
    variant_statistics & get_leaf( size_t n ) {
	return m_leaf_dense[n-N_MIN_SIZE];
    }
    variant_statistics & get( size_t x, size_t p ) {
	return m_blocked[x-X_MIN_SIZE][p-P_MIN_SIZE];
    }
    variant_statistics & get_leaf( size_t x, size_t p ) {
	return m_leaf_blocked[x-X_MIN_SIZE][p-P_MIN_SIZE];
    }
    
    variant_statistics m_dense[N_DIM];
    variant_statistics m_blocked[X_DIM][P_DIM];
    variant_statistics m_leaf_dense[N_DIM];
    variant_statistics m_leaf_blocked[X_DIM][P_DIM];
    variant_statistics m_tiny, m_gen;

};

struct per_thread_statistics {
    all_variant_statistics & get_statistics() {
	static thread_local all_variant_statistics * local_stats = nullptr;
	if( local_stats != nullptr )
	    return *local_stats;

	const pthread_t tid = pthread_self();
	std::lock_guard<std::mutex> guard( m_mutex );
	auto it = m_stats.find( tid );
	if( it == m_stats.end() ) {
	    auto it2 = m_stats.emplace(
		std::make_pair( tid, all_variant_statistics() ) );
	    return it2.first->second;
	}
	local_stats = &it->second;
	return it->second;
    }
    
    all_variant_statistics sum() const {
	return std::accumulate(
	    m_stats.begin(), m_stats.end(), all_variant_statistics(),
	    []( const all_variant_statistics & s,
		const std::pair<pthread_t,all_variant_statistics> & p ) {
		return s + p.second;
	    } );
    }
    
    std::mutex m_mutex;
    std::map<pthread_t,all_variant_statistics> m_stats;
};

per_thread_statistics mc_stats;

/*! Direct solution for tiny problems.
 *
 * HGraph is a graph type that supports a get_adjacency(VID) method that returns
 * a type with contains method.
 */
template<typename HGraph>
void
mc_tiny(
    const HGraph & H,
    const VID * const ngh,
    const VID start_pos,
    const VID num,
    MC_Enumerator & E ) {
    if( num == 0 ) {
	E.record( 0 );
    } else if( num == 1 ) {
	if( start_pos == 0 )
	    E.record( 2 ); // v, 0
    } else if( num == 2 ) {
	bool n01 = H.get_adjacency( ngh[0] ).contains( ngh[1] );
	if( start_pos == 0 ) {
	    if( n01 ) // Two neighbours of v are neighbours themselves
		E.record( 3 ); // v, 0, 1
	    else {
		E.record( 2 ); // v, 0
		E.record( 2 ); // v, 1
	    }
	} else if( start_pos == 1 ) {
	    // No maximal clique in case start_pos == 1
	    if( !n01 ) // triangle v, 0, 1 does not exist
		E.record( 2 ); // v, 1
	}
    } else if( num == 3 ) {
	int n01 = H.get_adjacency( ngh[0] ).contains( ngh[1] ) ? 1 : 0;
	int n02 = H.get_adjacency( ngh[0] ).contains( ngh[2] ) ? 1 : 0;
	int n12 = H.get_adjacency( ngh[1] ).contains( ngh[2] ) ? 1 : 0;
	if( start_pos == 0 ) {
	    // v < ngh[0] < ngh[1] < ngh[2]
	    if( n01 + n02 + n12 == 3 )
		E.record( 4 );
	    else if( n01 + n02 + n12 == 2 ) {
		E.record( 3 );
		E.record( 3 );
	    } else if( n01 + n02 + n12 == 1 ) {
		E.record( 3 );
		E.record( 2 );
	    } else if( n01 + n02 + n12 == 0 ) {
		E.record( 2 );
		E.record( 2 );
		E.record( 2 );
	    }
	} else if( start_pos == 1 ) {
	    // ngh[0] < v < ngh[1] < ngh[2]
	    if( n01 + n02 + n12 == 3 )
		; // duplicate
	    else if( n01 + n02 + n12 == 2 ) { // wedge
		if( n12 == 1 )
		    E.record( 3 );
	    } else if( n01 + n02 + n12 == 1 ) {
		if( n12 == 1 )
		    E.record( 3 ); // v, 1, 2
		else if( n01 == 1 )
		    E.record( 2 ); // v, 2
		else if( n02 == 1 )
		    E.record( 2 ); // v, 1
	    } else if( n01 + n02 + n12 == 0 ) {
		E.record( 2 ); // v, 1
		E.record( 2 ); // v, 2
	    }
	} else if( start_pos == 2 ) {
	    // ngh[0] < ngh[1] < v < ngh[2]
	    if( n02 + n12 == 0 ) // not part of a triangle or more
		E.record( 2 );
	}
    }
}    

bool is_member( VID v, VID C_size, const VID * C_set ) {
    const VID * const pos = std::lower_bound( C_set, C_set+C_size, v );
    if( pos == C_set+C_size || *pos != v )
	return false;
    return true;
}

#if 0
#if 0
class StackLikeAllocator {
public:
    StackLikeAllocator( size_t min_chunk_size = 0 ) { }

    template<typename T>
    T * allocate( size_t n_elements ) {
	return new T[n_elements];
    }
    template<typename T>
    void deallocate_to( T * p ) {
	delete[] p;
    }
};
#else
class StackLikeAllocator {
    static constexpr size_t PAGE_SIZE = size_t(1) << 20; // 1MiB
    struct chunk_t {
	static constexpr size_t MAX_BYTES =
	    ( size_t(1) << (8*sizeof(uint32_t)) ) - 2 * sizeof( uint32_t );

	chunk_t( size_t sz ) : m_size( sz ), m_end( 0 ) {
	    assert( sz <= MAX_BYTES );
	    // assert( sz == (size_t)m_size );
	    // assert( (size_t)m_size >= PAGE_SIZE );
	}

	char * allocate( size_t sz ) {
	    // assert( (size_t)m_size >= PAGE_SIZE );
	    // assert( sz <= MAX_BYTES );
	    // assert( m_end + sz <= m_size );
	    char * p = get_ptr() + m_end;
	    m_end += sz;
	    return p;
	}

	bool has_available_space( size_t sz ) const {
	    // assert( (size_t)m_size >= PAGE_SIZE );
	    // assert( sz <= MAX_BYTES );
	    uint32_t new_end = m_end + sz;
	    return new_end - m_end == sz && new_end <= m_size;
	    
	}

	char * get_ptr() const {
	    char * me = const_cast<char *>(
		reinterpret_cast<const char *>( this ) );
	    me += sizeof( m_size );
	    me += sizeof( m_end );
	    return me;
	}

	bool release_to( char * p ) {
	    // assert( (size_t)m_size >= PAGE_SIZE );
	    char * q = get_ptr();
	    if( q <= p && p < q+m_end ) {
		m_end = p - q;
		return true;
	    } else {
		m_end = 0;
		return false;
	    }
	}

    private:
	uint32_t m_size;
	uint32_t m_end;
    };

public:
    StackLikeAllocator( size_t min_chunk_size = PAGE_SIZE )
	: m_min_chunk_size( min_chunk_size ), m_current( 0 ) {
	// if( verbose )
	// std::cerr << "sla " << this << ": constructor\n";
    }
    ~StackLikeAllocator() {
	for( chunk_t * c : m_chunks ) {
	    // if( verbose )
	    // std::cerr << "sla " << this << ": delete chunk "
	    // << c << "\n";
	    delete[] reinterpret_cast<char *>( c );
	}
	// if( verbose )
	// std::cerr << "sla " << this << ": destructor done\n";
    }

    template<typename T>
    T * allocate( size_t n_elements ) {
	size_t sz = n_elements * sizeof(T);
	sz = ( sz + 3 ) & ~size_t(3); // multiple of 4 bytes
	T * p = reinterpret_cast<T*>( allocate_private( sz ) );
	// if( verbose )
	// std::cerr << "sla " << this << ": allocate " << n_elements
	// << ' ' << (void *)p << "\n";
	return p;
    }
    template<typename T>
    void deallocate_to( T * p ) {
	// if( verbose )
	// std::cerr << "sla " << this << ": deallocate-to "
	// << (void *)p << "\n";
	release_chunks( reinterpret_cast<char *>( p ) );
    }

private:
    char * allocate_private( size_t nbytes ) {
	// Do we have any available chunks?
	if( m_chunks.empty() )
	    return allocate_from_new_chunk( nbytes );
	
	// Check if any free chunk has sufficient space
	// Might be better to insert a larger chunk in the sequence if
	// the current cannot hold it, as future calls will require smaller
	// allocations which may be served from the available chunks
	do {
	    chunk_t * c = m_chunks[m_current];
	    if( c->has_available_space( nbytes ) )
		return c->allocate( nbytes );
	} while( ++m_current < m_chunks.size() );

	// No chunk can hold this
	return allocate_from_new_chunk( nbytes );
    }

    char * allocate_from_new_chunk( size_t nbytes ) {
	size_t sz = std::max( nbytes, m_min_chunk_size );
	sz = ( sz + PAGE_SIZE - 1 ) & ~( PAGE_SIZE - 1 );
	// assert( sz >= nbytes );
	char * cc = new char[sz];
	chunk_t * c = new ( cc ) chunk_t( sz );
	m_chunks.push_back( c );
	m_current = m_chunks.size() - 1;
	// if( verbose )
	// std::cerr << "sla " << this << ": new chunk " << c << "\n";
	return c->allocate( nbytes );
    }

    void release_chunks( char * p ) {
	for( size_t i=0; i <= m_current; ++i ) {
	    size_t j = m_current - i;
	    chunk_t * const c = m_chunks[j];
	    if( c->release_to( p ) ) {
		m_current = j;
		return;
	    }
	}
	assert( false && "deallocation error - should not reach here" );
    }
    
private:
    std::vector<chunk_t *> m_chunks;
    size_t m_min_chunk_size;
    size_t m_current;
};
#endif
#endif

template<typename T, typename S>
S insert_sorted( T * p, S sz, T u ) {
    T * q = std::lower_bound( p, p+sz, u );
    if( q == p+sz || *q != u ) { // not already present
	std::copy_backward( q, p+sz, p+sz+1 );
	*q = u;
	return sz+1;
    }
    return sz;
}

template<typename T, typename S>
S remove_sorted( T * p, S sz, T u ) {
    T * q = std::lower_bound( p, p+sz, u );
    if( q != p+sz && *q == u ) {
	std::copy( q+1, p+sz, q );
	return sz-1;
    }
    return sz;
}

void
check_clique( const graptor::graph::GraphDoubleIndexCSx<VID,EID> & G,
	      VID size,
	      VID * clique ) {
    std::sort( clique, clique+size );
    VID n = G.numVertices();
    const EID * const bindex = G.getBeginIndex();
    const EID * const eindex = G.getEndIndex();
    const VID * const edges = G.getEdges();

    for( VID i=0; i < size; ++i ) {
	VID v = clique[i];
	for( VID j=0; j < size; ++j ) {
	    if( j == i )
		continue;
	    VID u = clique[j];
	    const VID * const pos
		= std::lower_bound( &edges[bindex[v]], &edges[eindex[v]], u );
	    if( pos == &edges[eindex[v]] || *pos != u )
		abort();
	}
    }
}

void
check_clique( const GraphCSx & G,
	      VID size,
	      VID * clique ) {
    std::sort( clique, clique+size );
    VID n = G.numVertices();
    const EID * const index = G.getIndex();
    const VID * const edges = G.getEdges();

    VID v0 = clique[0];
    contract::vertex_set<VID> ins;
    ins.push( &edges[index[v0]], &edges[index[v0+1]] );

    for( VID i=1; i < size; ++i ) {
	VID v = clique[i];
	for( VID j=0; j < size; ++j ) {
	    if( j == i )
		continue;
	    VID u = clique[j];
	    const VID * const pos
		= std::lower_bound( &edges[index[v]], &edges[index[v+1]], u );
	    if( pos == &edges[index[v+1]] || *pos != u )
		abort();
	}
	ins = ins.intersect( &edges[index[v]], index[v+1] - index[v] );
    }
    assert( ins.size() == 0 ); // check if maximal
}

bool
is_maximal_clique(
    const graptor::graph::GraphCSx<VID,EID> & G,
    VID size,
    VID * clique ) {
    // std::sort( clique, clique+size );
    VID n = G.numVertices();
    const EID * const index = G.getIndex();
    const VID * const edges = G.getEdges();

    VID vs = clique[size-1];
    contract::vertex_set<VID> ins;
    ins.push( &edges[index[vs]], &edges[index[vs+1]] );

    for( VID i=0; i < size-1; ++i ) {
	VID v = clique[size-2-i];
	ins = ins.intersect( &edges[index[v]], index[v+1] - index[v] );
	if( ins.size() == 0 )
	    break;
    }
    return ins.size() == 0;
}

bool
is_maximal_clique(
    const graptor::graph::GraphCSx<VID,EID> & G,
    contract::vertex_set<VID> & R ) {
    return is_maximal_clique( G, R.size(), &*R.begin() );
}

bool is_subset( VID S_size, const VID * S_set,
		VID C_size, const VID * C_set ) {
    for( VID i=0; i < S_size; ++i ) {
	VID v = S_set[i];
	if( !is_member( v, C_size, C_set ) )
	    return false;
    }
    return true;
}

template<typename VID, typename EID>
bool mc_leaf(
    const HGraphTy & H,
    MC_Enumerator & E,
    VID r,
    const PSet<VID> & xp_set,
    VID ce );

template<typename GraphType>
class GraphBuilderInduced;

// For maximal clique enumeration - all vertices regardless of coreness
// Sort/relabel vertices by decreasing coreness
// TODO: make hybrid between hash table / adj list for lowest degrees?
template<typename VID, typename EID, typename Hash>
class GraphBuilderInduced<graptor::graph::GraphHAdjTable<VID,EID,Hash>> {
public:
    template<typename HGraph>
    GraphBuilderInduced(
	const GraphCSx & G,
	const HGraph & H,
	VID v,
	const graptor::graph::NeighbourCutOutDegeneracyOrder<VID,EID> & cut )
	: S( cut.get_num_vertices() ),
	  start_pos( cut.get_start_pos() ) {
	const VID * const s2g = cut.get_vertices();
	VID n = G.numVertices();
	EID m = G.numEdges();
	const EID * const gindex = G.getIndex();
	const VID * const gedges = G.getEdges();

	// Indices:
	// global (g): original graph's vertex IDs
	// short (s): relabeled vertex IDs in induced graph S
	// neighbours (n): position of vertex in neighbour list, which is
	//                 sorted by global IDs, facilitating lookup
	VID ns = cut.get_num_vertices();

	for( VID su=0; su < ns; ++su ) {
	    VID u = s2g[su];
	    VID k = 0;
	    auto & adj = S.get_adjacency( su );
	    for( EID e=gindex[u], ee=gindex[u+1]; e != ee && k != ns; ++e ) {
		VID w = gedges[e];
		while( k != ns && s2g[k] < w )
		    ++k;
		if( k == ns )
		    break;
		// If neighbour is selected in cut-out, add to induced graph.
		// Skip self-edges.
		// Skip edges between vertices in X.
		if( s2g[k] == w && w != u
		    && ( su >= start_pos || k >= start_pos ) )
		    adj.insert( k );
	    }
	}

	S.sum_up_edges();
    }
    template<typename HGraph>
    GraphBuilderInduced(
	const HGraph & H,
	const VID * const XP,
	VID ne,
	VID ce )
	: S( ce ),
	  start_pos( ne ) {
	VID n = H.numVertices();

	for( VID su=0; su < ce; ++su ) {
	    VID u = XP[su];
	    VID k = 0;
	    auto & Hadj = H.get_adjacency( u );
	    auto & Sadj = S.get_adjacency( su );
	    graptor::hash_insert_iterator<typename HGraph::hash_set_type>
		out( Sadj, XP );
	    graptor::hash_scalar::intersect<true>(
		su < ne ? XP+ne : XP, XP+ce, Hadj, out );
	}

	S.sum_up_edges(); // necessary?
    }

    const auto & get_graph() const { return S; }
    VID get_start_pos() const { return start_pos; }

private:
    graptor::graph::GraphHAdjTable<VID,EID,Hash> S;
    VID start_pos;
};

template<typename VID, typename EID, bool dual_rep, typename Hash>
class GraphBuilderInduced<graptor::graph::GraphHAdjPA<VID,EID,dual_rep,Hash>> {
public:
    template<typename HGraph>
    GraphBuilderInduced(
	const GraphCSx & G,
	const HGraph & H,
	VID v,
	const graptor::graph::NeighbourCutOutDegeneracyOrder<VID,EID> & cut )
	: S( G, H, cut.get_vertices(), cut.get_start_pos(),
	     cut.get_num_vertices(),
	     numa_allocation_interleaved() ),
	  start_pos( cut.get_start_pos() ) { }
    template<typename HGraph>
    GraphBuilderInduced(
	const HGraph & H,
	const VID * const XP,
	VID ne,
	VID ce )
	: S( H, H, XP, ne, ce, numa_allocation_small() ),
	  start_pos( ne ) { }
    template<typename HGraph, typename FilterFn>
    GraphBuilderInduced(
	const GraphCSx & G,
	const HGraph & H,
	VID v,
	const graptor::graph::NeighbourCutOutDegeneracyOrderFiltered<VID,EID> & cut,
	FilterFn && fn )
	: S( G, H, cut.get_vertices(), cut.get_num_vertices(),
	     numa_allocation_interleaved(), std::forward<FilterFn>( fn ) ),
	  start_pos( 0 ) { }

    const auto & get_graph() const { return S; }
    VID get_start_pos() const { return start_pos; }

private:
    graptor::graph::GraphHAdjPA<VID,EID,dual_rep,Hash> S;
    std::vector<VID> s2g;
    VID start_pos;
};

//! recursively parallel version of Bron-Kerbosch w/ pivoting
//
void
mc_bron_kerbosch_recpar_xps(
    const HGraphTy & G,
    VID degeneracy,
    MC_Enumerator & E,
    PSet<VID> & xp,
    VID ce,
    int depth );

void
bk_recursive_call(
    const HGraphTy & G,
    VID degeneracy,
    MC_Enumerator & E,
    PSet<VID> & xp_new,
    VID ce_new,
    int depth ) {
    // Check if the best possible clique we can construct would improve
    // over the current best known clique.
    if( !E.is_feasible( depth + ce_new ) )
	return;

    // Reached leaf of search tree
    if( ce_new == 0 ) {
	E.record( depth );
	return;
    }

#if TUNABLE_SMALL_AVOID_CUTOUT_LEAF != 0
    if( ce_new - ne_new >= TUNABLE_SMALL_AVOID_CUTOUT_LEAF )
#endif
    {
	if( mc_leaf<VID,EID>( G, E, depth, xp_new, ce_new ) )
	    return;
    }

    // Large sub-problem; search recursively
    // Tuning point: do we cut out a subgraph or not?
    // Tuning point: do we proceed with MC or switch to VC?
    mc_bron_kerbosch_recpar_xps( G, degeneracy, E, xp_new, ce_new, depth );
}

// XP may be modified by the method. It is not required to be in sort order.
void
mc_bron_kerbosch_recpar_xps(
    const HGraphTy & G,
    VID degeneracy,
    MC_Enumerator & E,
    PSet<VID> & xp,
    VID ce,
    int depth ) {
    // Termination condition
    if( 0 == ce ) {
	E.record( depth );
	return;
    }
    const VID n = G.numVertices();

    if constexpr ( io_trace ) {
	std::lock_guard<std::mutex> guard( io_mux );
	std::cout << "XPS loop: ce=" << ce << " depth=" << depth << "\n";
    }

    parallel_loop( (VID)0, ce, (VID)1, [&,ce]( VID i ) {
	VID v = xp.at( i );

	const auto & adj = G.get_adjacency( v ); 
	VID deg = adj.size();

	if constexpr ( io_trace ) {
	    std::lock_guard<std::mutex> guard( io_mux );
	    std::cout << "XP2: X=" << i << " P=" << (ce - (i+1)) << " adj="
		      << adj.size() << " depth=" << depth << "\n";
	}

	if( deg == 0 ) [[unlikely]] { // implies ne == ce == 0
	    // avoid overheads of copying and cutout
	    E.record( depth+1 );
	} else {
	    // Some complexity:
	    // + Need to consider all vertices prior to v in XP are now
	    //   in the X set. Could set ne to i, however:
	    // + Vertices that are filtered due to pivoting,
	    //   i.e., neighbours of pivot, are still in P.
	    // + In sequential execution, we can update XP incrementally,
	    //   however in parallel execution we cannot.
	    const VID * ngh = G.get_neighbours( v ); 
	    VID ce_new;
	    PSet<VID> xp_new
		= xp.intersect( G.numVertices(), i, ce, adj, ngh, ce_new );

	    bk_recursive_call( G, degeneracy, E, xp_new, ce_new, depth+1 );
	}
    } );
}

void
mc_bron_kerbosch_recpar_top_xps(
    const HGraphTy & G,
    VID degeneracy,
    MC_Enumerator & E ) {
    const VID n = G.numVertices();

    parallel_loop( (VID)0, n, (VID)1, [&,n]( VID v ) {
	VID deg = G.getDegree( v ); 

	if( deg == 0 ) {
	    // avoid overheads of copying and cutout
	    E.record( 1 );
	} else {
	    const VID * ngh = G.get_neighbours( v ); 
	    VID ce_new;
	    // TODO: cutout must ignore left-neighbourhood
	    // TODO: intersect-and-filter based on largest clique so far
	    PSet<VID> xp_new
		= PSet<VID>::intersect_top_level( n, v, ngh, deg, ce_new );

	    bk_recursive_call( G, degeneracy, E, xp_new, ce_new, 2 );
	}
    } );
}

void check_clique_edges( EID m, const VID * assigned_clique, EID ce ) {
    EID cce = 0;
    for( EID e=0; e != m; ++e )
	if( ~assigned_clique[e] != 0 )
	    ++cce;
    assert( cce == ce );
}

#if 0

template<unsigned XBits, unsigned PBits, typename HGraph, typename Enumerator>
void mc_blocked_fn(
    const GraphCSx & G,
    const HGraph & H,
    Enumerator & E,
    VID v,
    const graptor::graph::NeighbourCutOutDegeneracyOrder<VID,EID> & cut,
    variant_statistics & stats ) {

    timer tm;
    tm.start();

    // Build induced graph
    BlockedBinaryMatrix<XBits,PBits,VID,EID>
	IG( G, H, cut.get_vertices(), cut.get_start_pos(),
	    cut.get_num_vertices() );

    stats.record_build( tm.next() );

    mce_bron_kerbosch( IG, E );

    stats.record( tm.stop() );
}

template<unsigned Bits, typename HGraph, typename Enumerator>
void mc_dense_fn(
    const GraphCSx & G,
    const HGraph & H,
    Enumerator & E,
    VID v,
    const graptor::graph::NeighbourCutOutDegeneracyOrder<VID,EID> & cut,
    variant_statistics & stats ) {

    timer tm;
    tm.start();

    VID num = cut.get_num_vertices();

    // Build induced graph
    DenseMatrix<Bits,VID,EID>
	IG( G, H, cut.get_vertices(), cut.get_start_pos(),
	    cut.get_num_vertices() );

    stats.record_build( tm.next() );

    IG.mce_bron_kerbosch( E );

    double t = tm.stop();
    if( false && t >= 3.0 ) {
	std::cerr << "dense " << Bits << " v=" << v
		  << " num=" << cut.get_num_vertices()
		  << " start=" << cut.get_start_pos()
		  << " t=" << t
		  << "\n";
    }

    stats.record( t );
}

typedef void (*mc_func)(
    const GraphCSx &, 
    const HFGraphTy &,
    MC_Enumerator &,
    VID,
    const graptor::graph::NeighbourCutOutDegeneracyOrder<VID,EID> & cut,
    variant_statistics & );
    
static mc_func mc_dense_func[N_DIM+1] = {
    &mc_dense_fn<32,HFGraphTy,MC_Enumerator>,  // N=32
    &mc_dense_fn<64,HFGraphTy,MC_Enumerator>,  // N=64
    &mc_dense_fn<128,HFGraphTy,MC_Enumerator>, // N=128
    &mc_dense_fn<256,HFGraphTy,MC_Enumerator>, // N=256
    &mc_dense_fn<512,HFGraphTy,MC_Enumerator>  // N=512
};

static mc_func mc_blocked_func[X_DIM+1][P_DIM+1] = {
    // X == 2**5
    { &mc_blocked_fn<32,32,HFGraphTy,MC_Enumerator>,  // X=32, P=32
      &mc_blocked_fn<32,64,HFGraphTy,MC_Enumerator>,  // X=32, P=64
      &mc_blocked_fn<32,128,HFGraphTy,MC_Enumerator>, // X=32, P=128
      &mc_blocked_fn<32,256,HFGraphTy,MC_Enumerator>, // X=32, P=256
      &mc_blocked_fn<32,512,HFGraphTy,MC_Enumerator>  // X=32, P=512
    },
    // X == 2**6
    { &mc_blocked_fn<64,32,HFGraphTy,MC_Enumerator>,  // X=64, P=32
      &mc_blocked_fn<64,64,HFGraphTy,MC_Enumerator>,  // X=64, P=64
      &mc_blocked_fn<64,128,HFGraphTy,MC_Enumerator>, // X=64, P=128
      &mc_blocked_fn<64,256,HFGraphTy,MC_Enumerator>, // X=64, P=256
      &mc_blocked_fn<64,512,HFGraphTy,MC_Enumerator>  // X=64, P=512
    },
    // X == 2**7
    { &mc_blocked_fn<128,32,HFGraphTy,MC_Enumerator>,  // X=128, P=32
      &mc_blocked_fn<128,64,HFGraphTy,MC_Enumerator>,  // X=128, P=64
      &mc_blocked_fn<128,128,HFGraphTy,MC_Enumerator>, // X=128, P=128
      &mc_blocked_fn<128,256,HFGraphTy,MC_Enumerator>, // X=128, P=256
      &mc_blocked_fn<128,512,HFGraphTy,MC_Enumerator>  // X=128, P=512
    },
    // X == 2**8
    { &mc_blocked_fn<256,32,HFGraphTy,MC_Enumerator>,  // X=256, P=32
      &mc_blocked_fn<256,64,HFGraphTy,MC_Enumerator>,  // X=256, P=64
      &mc_blocked_fn<256,128,HFGraphTy,MC_Enumerator>, // X=256, P=128
      &mc_blocked_fn<256,256,HFGraphTy,MC_Enumerator>, // X=256, P=256
      &mc_blocked_fn<256,512,HFGraphTy,MC_Enumerator>  // X=256, P=512
    },
    // X == 2**9
    { &mc_blocked_fn<512,32,HFGraphTy,MC_Enumerator>,  // X=512, P=32
      &mc_blocked_fn<512,64,HFGraphTy,MC_Enumerator>,  // X=512, P=64
      &mc_blocked_fn<512,128,HFGraphTy,MC_Enumerator>, // X=512, P=128
      &mc_blocked_fn<512,256,HFGraphTy,MC_Enumerator>, // X=512, P=256
      &mc_blocked_fn<512,512,HFGraphTy,MC_Enumerator>  // X=512, P=512
    }
};

#endif

size_t get_size_class( uint32_t v ) {
    size_t b = _lzcnt_u32( v-1 );
    size_t cl = 32 - b;
    assert( v <= (1<<cl) );
    return cl;
}

void mc_top_level(
    const GraphCSx & G,
    const HFGraphTy & H,
    MC_Enumerator & E,
    VID v,
    VID degeneracy,
    const VID * const remap_coreness ) {

    VID best = E.get_max_clique_size();

    // No point analysing a vertex of too low degree
    if( remap_coreness[v] < best )
	return;

    // Filter out vertices where degree in main graph < best.
    // With degree == best, we can make a clique of size best+1 at best.
    graptor::graph::NeighbourCutOutDegeneracyOrderFiltered<VID,EID>
	cut( G, v, [&]( VID u ) { return remap_coreness[u] > best; } );

    // If size of cut-out graph is less than best, then there is no point
    // in analysing it, nor constructing cut-out.
    if( cut.get_num_vertices() < best )
	return;

    all_variant_statistics & stats = mc_stats.get_statistics();

    VID num = cut.get_num_vertices();

#if !ABLATION_DISABLE_TOP_TINY
    if( num <= 3 ) [[unlikely]] {
	timer tm;
	tm.start();
	mc_tiny( H, cut.get_vertices(), cut.get_start_pos(),
		 cut.get_num_vertices(), E );
	stats.record_tiny( tm.stop() );
	return;
    }
#endif

#if !ABLATION_DISABLE_TOP_DENSE
    VID xnum = cut.get_start_pos();
    VID pnum = num - xnum;

    VID nlg = get_size_class( num );
    if( nlg < N_MIN_SIZE )
	nlg = N_MIN_SIZE;

    if( nlg <= N_MIN_SIZE+1 ) { // up to 64 bits
	return mc_dense_func[nlg-N_MIN_SIZE](
	    G, H, E, v, cut, stats.get( nlg ) );
    }

    VID xlg = get_size_class( xnum );
    if( xlg < X_MIN_SIZE )
	xlg = X_MIN_SIZE;

    VID plg = get_size_class( pnum );
    if( plg < P_MIN_SIZE )
	plg = P_MIN_SIZE;

    if constexpr ( io_trace ) {
	std::lock_guard<std::mutex> guard( io_mux );
	std::cout << "nlg=" << nlg << " xlg=" << xlg << " plg=" << plg << "\n";
    }

    if( nlg <= xlg + plg && nlg <= N_MAX_SIZE ) {
	return mc_dense_func[nlg-N_MIN_SIZE](
	    G, H, E, v, cut, stats.get( nlg ) );
    }

    if( xlg <= X_MAX_SIZE && plg <= P_MAX_SIZE ) {
	return mc_blocked_func[xlg-X_MIN_SIZE][plg-P_MIN_SIZE](
	    G, H, E, v, cut, stats.get( xlg, plg ) );
    }

    if( nlg <= N_MAX_SIZE ) {
	return mc_dense_func[nlg-N_MIN_SIZE](
	    G, H, E, v, cut, stats.get( nlg ) );
    }
#endif

    timer tm;
    tm.start();
    // TODO: it should help if the cut-out only contains valid vertices,
    //       but additionally need the filter-func to quickly determine
    //       eligibility of neighbours. This should include comparison against v
    GraphBuilderInduced<HGraphTy> ibuilder(
	G, H, v, cut,
	[&]( VID u ) { return u > v && remap_coreness[u] >= best; } );
    const auto & HG = ibuilder.get_graph();

    stats.record_genbuild( tm.stop() );

    tm.start();
    mc_bron_kerbosch_recpar_top_xps( HG, degeneracy, E );
    double t = tm.stop();
    stats.record_gen( t );
}

template<unsigned Bits, typename VID, typename EID>
void leaf_dense_fn(
    const HGraphTy & H,
    MC_Enumerator & E,
    VID r,
    const PSet<VID> & xp_set,
    VID ne,
    VID ce ) {
    variant_statistics & stats
	= mc_stats.get_statistics().get_leaf( ilog2( Bits ) );
    timer tm;
    tm.start();
    DenseMatrix<Bits,VID,EID> D( H, H, xp_set, ne, ce );
    stats.record_build( tm.next() );
    D.mce_bron_kerbosch( E );
    stats.record( tm.next() );
}

#if 0
template<unsigned XBits, unsigned PBits, typename VID, typename EID>
void leaf_blocked_fn(
    const HGraphTy & H,
    MC_Enumerator & E,
    VID r,
    const PSet<VID> & xp_set,
    VID ne,
    VID ce ) {
    variant_statistics & stats
	= mc_stats.get_statistics().get_leaf( ilog2( XBits ), ilog2( PBits ) );
    timer tm;
    tm.start();
    BlockedBinaryMatrix<XBits,PBits,VID,EID>
	D( H, H, xp_set, ne, ce );
    stats.record_build( tm.next() );
    mce_bron_kerbosch( D, E );
    stats.record( tm.next() );
}

typedef void (*mc_leaf_func)(
    const HGraphTy &,
    MC_Enumerator &,
    VID,
    const PSet<VID> &,
    VID,
    VID );
    
static mc_leaf_func leaf_dense_func[N_DIM+1] = {
    &leaf_dense_fn<32,VID,EID>,  // N=32
    &leaf_dense_fn<64,VID,EID>,  // N=64
    &leaf_dense_fn<128,VID,EID>, // N=128
    &leaf_dense_fn<256,VID,EID>, // N=256
    &leaf_dense_fn<512,VID,EID>  // N=512
};

static mc_leaf_func leaf_blocked_func[X_DIM+1][P_DIM+1] = {
    // X == 2**5
    { &leaf_blocked_fn<32,32,VID,EID>,  // X=32, P=32
      &leaf_blocked_fn<32,64,VID,EID>,  // X=32, P=64
      &leaf_blocked_fn<32,128,VID,EID>, // X=32, P=128
      &leaf_blocked_fn<32,256,VID,EID>, // X=32, P=256
      &leaf_blocked_fn<32,512,VID,EID>  // X=32, P=512
    },
    // X == 2**6
    { &leaf_blocked_fn<64,32,VID,EID>,  // X=64, P=32
      &leaf_blocked_fn<64,64,VID,EID>,  // X=64, P=64
      &leaf_blocked_fn<64,128,VID,EID>, // X=64, P=128
      &leaf_blocked_fn<64,256,VID,EID>, // X=64, P=256
      &leaf_blocked_fn<64,512,VID,EID>  // X=64, P=512
    },
    // X == 2**7
    { &leaf_blocked_fn<128,32,VID,EID>,  // X=128, P=32
      &leaf_blocked_fn<128,64,VID,EID>,  // X=128, P=64
      &leaf_blocked_fn<128,128,VID,EID>, // X=128, P=128
      &leaf_blocked_fn<128,256,VID,EID>, // X=128, P=256
      &leaf_blocked_fn<128,512,VID,EID>  // X=128, P=512
    },
    // X == 2**8
    { &leaf_blocked_fn<256,32,VID,EID>,  // X=256, P=32
      &leaf_blocked_fn<256,64,VID,EID>,  // X=256, P=64
      &leaf_blocked_fn<256,128,VID,EID>, // X=256, P=128
      &leaf_blocked_fn<256,256,VID,EID>, // X=256, P=256
      &leaf_blocked_fn<256,512,VID,EID>  // X=256, P=512
    },
    // X == 2**9
    { &leaf_blocked_fn<512,32,VID,EID>,  // X=512, P=32
      &leaf_blocked_fn<512,64,VID,EID>,  // X=512, P=64
      &leaf_blocked_fn<512,128,VID,EID>, // X=512, P=128
      &leaf_blocked_fn<512,256,VID,EID>, // X=512, P=256
      &leaf_blocked_fn<512,512,VID,EID>  // X=512, P=512
    }
};
#endif

template<typename VID, typename EID>
bool mc_leaf(
    const HGraphTy & H,
    MC_Enumerator & E,
    VID r,
    const PSet<VID> & xp_set,
    VID ce ) {
#if ABLATION_DISABLE_LEAF
    return false;
#else
    VID num = ce;
    VID pnum = ce - ne;
    VID * XP = xp_set.get_set();

    if( ce <= 3 ) {
	mc_tiny( H, XP, ne, ce, E );
	return true;
    }

    VID nlg = get_size_class( num );
    if( nlg < N_MIN_SIZE )
	nlg = N_MIN_SIZE;

    if( nlg <= N_MIN_SIZE+1 ) { // up to 64 bits
	leaf_dense_func[nlg-N_MIN_SIZE]( H, E, R, r, xp_set, ne, ce );
	return true;
    }

    VID plg = get_size_class( pnum );
    if( plg < P_MIN_SIZE )
	plg = P_MIN_SIZE;

    if( nlg <= xlg + plg && nlg <= N_MAX_SIZE ) {
	leaf_dense_func[nlg-N_MIN_SIZE]( H, E, R, r, xp_set, ne, ce );
	return true;
    }

    if( xlg <= X_MAX_SIZE && plg <= P_MAX_SIZE ) {
	leaf_blocked_func[xlg-X_MIN_SIZE][plg-P_MIN_SIZE](
	    H, E, R, r, xp_set, ne, ce );
	return true;
    }

    if( nlg <= N_MAX_SIZE ) {
	leaf_dense_func[nlg-N_MIN_SIZE]( H, E, R, r, xp_set, ne, ce );
	return true;
    }

    return false;
#endif // ABLATION_DISABLE_LEAF
}

int main( int argc, char *argv[] ) {
    commandLine P( argc, argv, " help" );
    bool symmetric = P.getOptionValue("-s");
    VID npart = P.getOptionLongValue( "-c", 256 );
    verbose = P.getOptionValue("-v");
    const char * ifile = P.getOptionValue( "-i" );

    timer tm;
    tm.start();

    GraphCSx G0( ifile, -1, symmetric );

    std::cout << "Reading graph: " << tm.next() << "\n";

    GraphCSx G = graptor::graph::remove_self_edges( G0, true );
    G0.del();
    std::cout << "Removed self-edges: " << tm.next() << "\n";

    VID n = G.numVertices();
    EID m = G.numEdges();

    assert( G.isSymmetric() );
    double density = double(m) / ( double(n) * double(n) );
    VID dmax_v = G.findHighestDegreeVertex();
    VID dmax = G.getDegree( dmax_v );
    double davg = (double)m / (double)n;
    std::cout << "Undirected graph: n=" << n << " m=" << m
	      << " density=" << density
	      << " dmax=" << dmax
	      << " davg=" << davg
	      << std::endl;

    GraphCSRAdaptor GA( G, npart );
    KCv<GraphCSRAdaptor> kcore( GA, P );
    kcore.run();
    auto & coreness = kcore.getCoreness();
    std::cout << "Calculating coreness: " << tm.next() << "\n";
    std::cout << "coreness=" << kcore.getLargestCore() << "\n";

    mm::buffer<VID> order( n, numa_allocation_interleaved() );
    mm::buffer<VID> rev_order( n, numa_allocation_interleaved() );
    sort_order( order.get(), rev_order.get(),
		coreness, n, kcore.getLargestCore() );
    std::cout << "Determining sort order: " << tm.next() << "\n";

    mm::buffer<VID> remap_coreness( n, numa_allocation_interleaved() );
    parallel_loop( (VID)0, n, [&]( VID v ) {
	remap_coreness[v] = coreness[order[v]];
    } );
    std::cout << "Remapping coreness data: " << tm.next() << "\n";

    GraphCSx R( G, std::make_pair( order.get(), rev_order.get() ) );
    std::cout << "Remapping graph: " << tm.next() << "\n";

    HFGraphTy H( R, numa_allocation_interleaved() );
    std::cout << "Building hashed graph: " << tm.next() << "\n";

    std::cout << "Options:"
	      << "\n\tABLATION_PDEG=" << ABLATION_PDEG
	      << "\n\tABLATION_DENSE_EXCEED=" << ABLATION_DENSE_EXCEED
	      << "\n\tABLATION_GENERIC_EXCEED=" << ABLATION_GENERIC_EXCEED
	      << "\n\tABLATION_BLOCKED_EXCEED=" << ABLATION_BLOCKED_EXCEED
	      << "\n\tABLATION_DISABLE_LEAF=" << ABLATION_DISABLE_LEAF
	      << "\n\tABLATION_DISABLE_TOP_TINY=" << ABLATION_DISABLE_TOP_TINY
	      << "\n\tABLATION_DISABLE_TOP_DENSE=" << ABLATION_DISABLE_TOP_DENSE
	      << "\n\tABLATION_HADJPA_DISABLE_XP_HASH="
	      << ABLATION_HADJPA_DISABLE_XP_HASH
	      << "\n\tABLATION_BLOCKED_DISABLE_XP_HASH="
	      << ABLATION_BLOCKED_DISABLE_XP_HASH
	      << "\n\tABLATION_DENSE_DISABLE_XP_HASH="
	      << ABLATION_DENSE_DISABLE_XP_HASH
	      << "\n\tTUNABLE_SMALL_AVOID_CUTOUT_LEAF="
	      << TUNABLE_SMALL_AVOID_CUTOUT_LEAF
	      << "\n\tDENSE_THRESHOLD_SEQUENTIAL_BITS="
	      << DENSE_THRESHOLD_SEQUENTIAL_BITS
	      << "\n\tDENSE_THRESHOLD_SEQUENTIAL="
	      << DENSE_THRESHOLD_SEQUENTIAL
	      << "\n\tDENSE_THRESHOLD_DENSITY="
	      << DENSE_THRESHOLD_DENSITY
	      << "\n\tBLOCKED_THRESHOLD_SEQUENTIAL_PBITS="
	      << BLOCKED_THRESHOLD_SEQUENTIAL_PBITS
	      << "\n\tBLOCKED_THRESHOLD_SEQUENTIAL="
	      << BLOCKED_THRESHOLD_SEQUENTIAL
	      << "\n\tBLOCKED_THRESHOLD_DENSITY="
	      << BLOCKED_THRESHOLD_DENSITY
	      << "\n\tABLATION_DENSE_NO_PIVOT_TOP="
	      << ABLATION_DENSE_NO_PIVOT_TOP
	      << "\n\tABLATION_DENSE_FILTER_FULLY_CONNECTED="
	      << ABLATION_DENSE_FILTER_FULLY_CONNECTED 
	      << "\n\tABLATION_BLOCKED_FILTER_FULLY_CONNECTED="
	      << ABLATION_BLOCKED_FILTER_FULLY_CONNECTED 
	      << "\n\tABLATION_DENSE_ITERATE="
	      <<  ABLATION_DENSE_ITERATE
	      << "\n\tABLATION_BLOCKED_ITERATE="
	      <<  ABLATION_BLOCKED_ITERATE
	      << "\n\tABLATION_DENSE_PIVOT_FILTER="
	      <<  ABLATION_DENSE_PIVOT_FILTER
	      << "\n\tABLATION_BLOCKED_PIVOT_FILTER="
	      <<  ABLATION_BLOCKED_PIVOT_FILTER
	      << "\n\tUSE_512_VECTOR="
	      <<  USE_512_VECTOR
	      << '\n';
    
    system( "hostname" );
    system( "date" );

    std::cout << "Start enumeration: " << tm.next() << std::endl;


#if PAPI_REGION == 1 
    map_workers( [&]( uint32_t t ) {
	if( PAPI_OK != PAPI_hl_region_begin( "MCE" ) ) {
	    std::cerr << "Error initialising PAPI\n";
	    exit(1);
	}
    } );
#endif

    VID degeneracy = kcore.getLargestCore();
    MC_Enumerator E( degeneracy );

    // Number of partitions is tunable. A fairly large number is helpful
    // to help load balancing.
    parallel_loop( VID(0), npart, 1, [&,npart,degeneracy,n]( VID p ) {
	for( VID v=p; v < n; v += npart )
	    mc_top_level( R, H, E, v, degeneracy, remap_coreness.get() );
    } );

#if PAPI_REGION == 1
    map_workers( [&]( uint32_t t ) {
	if( PAPI_OK != PAPI_hl_region_end( "MCE" ) ) {
	    std::cerr << "Error initialising PAPI\n";
	    exit(1);
	}
    } );
#endif

    std::cout << "Enumeration: " << tm.next() << "\n";

    all_variant_statistics stats = mc_stats.sum();

    double duration = tm.total();
    std::cout << "Completed MCE in " << duration << " seconds\n";
#if 0
    for( size_t n=N_MIN_SIZE; n <= N_MAX_SIZE; ++n ) {
	std::cout << (1<<n) << "-bit dense: ";
	stats.get( n ).print( std::cout ); 
    }
    for( size_t x=X_MIN_SIZE; x <= X_MAX_SIZE; ++x )
	for( size_t p=P_MIN_SIZE; p <= P_MAX_SIZE; ++p ) {
	    std::cout << (1<<x) << ',' << (1<<p) << "-bit blocked: ";
	    stats.get( x, p ).print( std::cout );
	}
    for( size_t n=N_MIN_SIZE; n <= N_MAX_SIZE; ++n ) {
	std::cout << "leaf-" << (1<<n) << "-bit dense: ";
	stats.get_leaf( n ).print( std::cout ); 
    }
    for( size_t x=X_MIN_SIZE; x <= X_MAX_SIZE; ++x )
	for( size_t p=P_MIN_SIZE; p <= P_MAX_SIZE; ++p ) {
	    std::cout << "leaf-" << (1<<x) << ',' << (1<<p) << "-bit blocked: ";
	    stats.get_leaf( x, p ).print( std::cout );
	}
    std::cout << "tiny: ";
    stats.m_tiny.print( std::cout );
#endif
    std::cout << "generic: ";
    stats.m_gen.print( std::cout );

    E.report( std::cout );

    remap_coreness.del();
    rev_order.del();
    order.del();
    G.del();

    return 0;
}
