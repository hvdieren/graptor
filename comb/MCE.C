// -*- c++ -*-
// Specialised to MCE

// TODO:
// * online machine learning
// * MCE_Enumerator thread-local (no sync-fetch-and-add)
// * Look at Blocked and Binary matrix design:
//   + col_start and row_start redundant to each other
// * VIDs of 8 or 16 bits
// * Consider sorting vertices first by non-increasing degeneracy, secondly
//   by non-increasing degree within a group of equal degeneracy.
//   The non-increasing degree means faster reduction of size of P?

// Novelties:
// + find pivot -> abort intersection if seen to be too small
// + small sub-problems -> dense matrix; O(1) operations

// Experiments:
// + Check that 32-bit is faster than 64-bit for same-sized problems;
//   same for SSE vs AVX

// Experiments observations:
// + sometimes sequentially faster; sometimes does not translate in parallel
//   efficiency (e.g. wiki-talk, USA)
// + need to check load balancing in detail (e.g. warwiki)
// + sometimes sequentially slower (e.g. warwiki, 10x)
//
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

#ifndef ABLATION_PIVOT_DISABLE_XP_HASH
#define ABLATION_PIVOT_DISABLE_XP_HASH 0
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
#define ABLATION_DISABLE_LEAF 0
#endif

#ifndef ABLATION_DISABLE_TOP_TINY
#define ABLATION_DISABLE_TOP_TINY 0
#endif

#ifndef ABLATION_DISABLE_TOP_DENSE
#define ABLATION_DISABLE_TOP_DENSE 0
#endif

#ifndef ABLATION_SORT_ORDER_TIES
#define ABLATION_SORT_ORDER_TIES 0
#endif

#ifndef ABLATION_RECPAR_CUTOUT
#define ABLATION_RECPAR_CUTOUT 1
#endif

// Not effective, so disable by default
#ifndef ABLATION_PDEG
#define ABLATION_PDEG 1
#endif

#ifndef PAR_LOOP
#define PAR_LOOP 2
#endif

#ifndef PAR_DENSE
#define PAR_DENSE 1
#endif

#ifndef PAR_BLOCKED
#define PAR_BLOCKED 1
#endif

#include <signal.h>
#include <sys/time.h>

#include <thread>
#include <mutex>
#include <map>
#include <exception>
#include <numeric>

#include <pthread.h>

#include <cilk/cilk.h>

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
    && ABLATION_HADJPA_DISABLE_XP_HASH	\
    && ABLATION_PIVOT_DISABLE_XP_HASH
using HGraphTy = graptor::graph::GraphHAdjPA<VID,EID,false,hash_fn>;
#else
using HGraphTy = graptor::graph::GraphHAdjPA<VID,EID,true,hash_fn>;
#endif
// using HFGraphTy = graptor::graph::GraphHAdjPA<VID,EID,false,hash_fn>;
using HFGraphTy = graptor::graph::GraphHAdjPA<VID,EID,true,hash_fn>;

using graptor::graph::DenseMatrix;
using graptor::graph::BinaryMatrix;
using graptor::graph::BlockedBinaryMatrix;
using graptor::graph::XPSet;

static constexpr size_t X_MIN_SIZE = 5;
static constexpr size_t X_MAX_SIZE = 8; // 9;
static constexpr size_t X_DIM = X_MAX_SIZE - X_MIN_SIZE + 1;
static constexpr size_t P_MIN_SIZE = 5;
static constexpr size_t P_MAX_SIZE = 8; // 9;
static constexpr size_t P_DIM = P_MAX_SIZE - P_MIN_SIZE + 1;
static constexpr size_t N_MIN_SIZE = 5;
static constexpr size_t N_MAX_SIZE = 8; // 9;
static constexpr size_t N_DIM = N_MAX_SIZE - N_MIN_SIZE + 1;

static bool verbose = false;

static VID * global_coreness = nullptr;

static std::mutex io_mux;
static constexpr bool io_trace = false;

class MCE_Enumerator {
public:
    MCE_Enumerator( size_t degen = 0 )
	: m_degeneracy( degen ),
	  m_histogram( degen+1 ) { }

    // Recod clique of size s
    void record( size_t s ) {
	assert( m_degeneracy+1 == m_histogram.size() );
	assert( s <= m_degeneracy+1 );
	__sync_fetch_and_add( &m_histogram[s-1], 1 );
    }

    MCE_Enumerator
    operator + ( const MCE_Enumerator & e ) const {
	MCE_Enumerator sum( m_degeneracy );
	for( size_t n=0; n < m_histogram.size(); ++n )
	    sum.m_histogram[n] = m_histogram[n] + e.m_histogram[n];
	return sum;
    }

    std::ostream & report( std::ostream & os ) const {
	assert( m_histogram.size() >= m_degeneracy+1 );

	size_t num_maximal_cliques = 0;
	for( size_t i=0; i < m_histogram.size(); ++i )
	    num_maximal_cliques += m_histogram[i];
	
	os << "Number of maximal cliques: " << num_maximal_cliques
	   << "\n";
	os << "Clique histogram: clique_size, num_of_cliques\n";
	for( size_t i=0; i <= m_degeneracy; ++i ) {
	    if( m_histogram[i] != 0 ) {
		os << (i+1) << ", " << m_histogram[i] << "\n";
	    }
	}
	return os;
    }

private:
    size_t m_degeneracy;
    std::vector<size_t> m_histogram;
};

struct MCE_Enumerator_stage2 {
    MCE_Enumerator_stage2( MCE_Enumerator & _E, size_t surplus = 1 )
	: E( _E ), m_surplus( surplus ) { }
    MCE_Enumerator_stage2( MCE_Enumerator_stage2 & _E, size_t surplus = 1 )
	: E( _E.E ), m_surplus( _E.m_surplus + surplus ) { }

    template<unsigned Bits>
    void operator() ( const bitset<Bits> & c, size_t sz ) {
	E.record( m_surplus + sz );
    }
    
    void record( size_t sz ) {
	E.record( m_surplus + sz );
    }

    size_t get_surplus() const { return m_surplus; }

private:
    MCE_Enumerator & E;
    const size_t m_surplus;
};

class MCE_Enumerator_Farm {
public:
    MCE_Enumerator_Farm( size_t d ) : m_degeneracy( d ) { }

    MCE_Enumerator_stage2 get_enumerator( size_t surplus ) {
	return MCE_Enumerator_stage2( get_enumerator_ref(), surplus );
    }
    
    MCE_Enumerator sum() const {
	return std::accumulate(
	    m_enum.begin(), m_enum.end(), MCE_Enumerator( m_degeneracy ),
	    []( const MCE_Enumerator & s,
		const std::pair<pthread_t,MCE_Enumerator> & p ) {
		return s + p.second;
	    } );
    }

    MCE_Enumerator & get_enumerator_ref() {
	static thread_local MCE_Enumerator * local_enum = nullptr;
	if( local_enum != nullptr )
	    return *local_enum;
	
	const pthread_t tid = pthread_self();
	std::lock_guard<std::mutex> guard( m_mutex );
	auto it = m_enum.find( tid );
	if( it == m_enum.end() ) {
	    auto it2 = m_enum.emplace(
		std::make_pair( tid, MCE_Enumerator( m_degeneracy ) ) );
	    return it2.first->second;
	}
	local_enum = &it->second;
	return it->second;
    }

private:
    size_t m_degeneracy;
    std::mutex m_mutex;
    std::map<pthread_t,MCE_Enumerator> m_enum;
};

class MCE_Parallel_Enumerator {
public:
    MCE_Parallel_Enumerator( MCE_Enumerator_Farm & farm, size_t surplus = 0 )
	: m_farm( farm ), m_surplus( surplus ) { }
    MCE_Parallel_Enumerator( MCE_Parallel_Enumerator & E, size_t surplus = 0 )
	: m_farm( E.m_farm ), m_surplus( E.m_surplus + surplus ) { }

    MCE_Enumerator_stage2 get_enumerator( size_t more_surplus = 1 ) {
	return m_farm.get_enumerator( m_surplus + more_surplus );
    }
    
    MCE_Enumerator & get_enumerator_ref() {
	assert( m_surplus == 0 );
	return m_farm.get_enumerator_ref();
    }
    
    // Recod clique of size s
    void record( size_t s ) {
	m_farm.get_enumerator_ref().record( m_surplus + s );
    }

private:
    MCE_Enumerator_Farm & m_farm;
    size_t m_surplus;
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
	for( size_t n=0; n < N_DIM; ++n )
	    sum.m_dense[n] = m_dense[n] + s.m_dense[n];
	for( size_t x=0; x < X_DIM; ++x )
	    for( size_t p=0; p < P_DIM; ++p )
		sum.m_blocked[x][p] = m_blocked[x][p] + s.m_blocked[x][p];
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
    variant_statistics & get( size_t x, size_t p ) {
	return m_blocked[x-X_MIN_SIZE][p-P_MIN_SIZE];
    }
    
    variant_statistics m_dense[N_DIM];
    variant_statistics m_blocked[X_DIM][P_DIM];
    variant_statistics m_tiny, m_gen;

};

// thread_local static all_variant_statistics * mce_pt_stats = nullptr;

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

per_thread_statistics mce_stats;

/*! Direct solution for tiny problems.
 *
 * HGraph is a graph type that supports a get_adjacency(VID) method that returns
 * a type with contains method.
 */
template<typename HGraph>
void
mce_tiny(
    const HGraph & H,
    const VID * const ngh,
    const VID start_pos,
    const VID num,
    MCE_Enumerator_stage2 & E ) {
    if( num == 0 ) {
	if( E.get_surplus() > 1 ) // min clique size counted is 2
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

template<typename HGraph>
std::pair<VID,VID>
__attribute__((noinline))
mc_get_pivot_xp(
    const HGraph & G,
    const XPSet<VID> & xp_set,
    VID ne,
    VID ce ) {

    assert( ce - ne != 0 );
    const VID * const XP = xp_set.get_set();

#if !ABLATION_GENERIC_EXCEED
    // Tunable (|P| and selecting vertex from X or P)
    if( ce - ne <= 3 )
	return std::make_pair( XP[ne], 0 );
#endif

    VID v_max = ~VID(0);
    VID tv_max = std::numeric_limits<VID>::min();

    for( VID i=0; i < ce; ++i ) {
	VID v = XP[i];
	// VID v = XP[ce-1-i]; -- slower
	// VID v = XP[(i+ne)%ce]; -- makes no difference
	auto & hadj = G.get_adjacency( v );
	VID deg = hadj.size();
#if !ABLATION_GENERIC_EXCEED
	if( deg <= tv_max )
	    continue;
#endif

	// Abort during intersection_size if size will be less than tv_max
	// Note: hash_vector is much slower in this instance
#if !ABLATION_GENERIC_EXCEED
#if !ABLATION_DISABLE_PIVOT_XP_HASH
	size_t tv;
	if( ce-ne > deg ) {
	    const VID * n = G.get_neighbours( v );
	    tv = graptor::hash_scalar::intersect_size_exceed(
		n, n+deg, xp_set.P_hash_set( ne ), tv_max );
	} else {
	    tv = graptor::hash_scalar::intersect_size_exceed(
		XP+ne, XP+ce, hadj, tv_max );
	}
#else
	size_t tv = graptor::hash_scalar::intersect_size_exceed(
	    XP+ne, XP+ce, hadj, tv_max );
#endif
#else
#if !ABLATION_DISABLE_PIVOT_XP_HASH
	size_t tv;
	if( ce-ne > deg ) {
	    const VID * n = G.get_neighbours( v );
	    tv = graptor::hash_scalar::intersect_size(
		n, n+deg, xp_set.P_hash_set( ne ) );
	} else {
	    tv = graptor::hash_scalar::intersect_size(
		XP+ne, XP+ce, hadj );
	}
#else
	size_t tv = graptor::hash_scalar::intersect_size(
	    XP+ne, XP+ce, hadj );
#endif
#endif
	    
	if( tv > tv_max ) {
	    tv_max = tv;
	    v_max = v;
	}
    }

    // return first element of P if nothing good found
    return std::make_pair( ~v_max == 0 ? XP[ne] : v_max, tv_max );
}

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
bool mce_leaf(
    const HGraphTy & H,
    MCE_Enumerator_stage2 & E,
    VID r,
    const XPSet<VID> & xp_set,
    VID ne,
    VID ce );

#if 0
template<bool with_leaf = true, typename HGraph = HGraphTy>
void
mce_iterate_xp_iterative(
    const HGraph & G,
    MCE_Enumerator_stage2 & Ee,
    StackLikeAllocator & alloc,
    VID degeneracy,
    VID * XP,
    VID ne, // not edges
    VID ce ) { // candidate edges

    struct frame_t {
	VID * XP;
	VID ne;
	VID ce;
	VID pivot;
	VID * XP_new;
	VID * prev_tgt;
	VID i;
    };

    // Base case - trivial problem
    if( ce == ne ) {
	if( ne == 0 )
	    Ee.record( 0 );
	return;
    }

    frame_t * frame = new frame_t[degeneracy+1];
    int depth = 1;

    // If space is an issue, could put alloc/dealloc inside loop and tune
    // space depending on neighbour list length (each of X and P can not
    // be longer than number of neighbours of u, nor longer than their
    // current size, so allocate std::min( ce, degree(u) ).

    new ( &frame[0] ) frame_t { XP, ne, ce, 0, nullptr, XP, ne };
    frame[0].pivot = mc_get_pivot_xp( G, XP, ne, ce ).first;
    frame[0].XP_new = alloc.template allocate<VID>( ce );

    while( depth >= 1 ) {
	frame_t & fr = frame[depth-1];
	assert( fr.ce >= fr.ne );

	// Loop iteration control
	if( fr.i >= fr.ce ) {
	    // Pop frame
	    assert( depth > 0 );
	    // assert( R.size() == depth );
	    --depth;
	    alloc.deallocate_to( fr.XP_new );
	    fr.XP_new = 0;

	    // Finish off vertex in higher-level frame
	    if( depth > 0 ) {
		frame_t & fr = frame[depth-1];
		if( fr.i+1 < fr.ce ) {
		    VID u = fr.XP[fr.i];
		
		    // R.pop();

		    // Move candidate (u) from original position to appropriate
		    // place in X part (maintaining sort order).
		    // Cache tgt for next iteration as next iteration's u
		    // will be strictly larger.
		    VID * tgt = std::upper_bound( fr.prev_tgt, fr.XP+fr.ne, u );
		    if( tgt != &fr.XP[fr.i] ) { // equality when u moves to tgt == XP+ne
			std::copy_backward( tgt, &fr.XP[fr.i], &fr.XP[fr.i+1] );
			*tgt = u;
		    }
		    fr.prev_tgt = tgt+1;
		    ++fr.ne;
		}
		++fr.i;
	    }
	    continue;
	}

	// Next step on frame
	VID u = fr.XP[fr.i];

	// Is u in the neighbour list of the pivot? If so, skip
	auto & p_ngh = G.get_adjacency( fr.pivot );
	if( !p_ngh.contains( u ) ) {
	    auto & adj = G.get_adjacency( u );
	    VID deg = adj.size();
	    VID * XP = fr.XP;
	    VID ne = fr.ne;
	    VID ce = fr.ce;
	    VID * XP_new = fr.XP_new;
	    VID ne_new = graptor::hash_vector::intersect(
		XP, XP+ne, adj, XP_new ) - XP_new;
	    VID ce_new = graptor::hash_vector::intersect(
		XP+ne, XP+ce, adj, XP_new+ne_new ) - XP_new;
	    assert( ce_new <= ce );
	    // R.push( u );
	    // assert( R.size() == depth+1 );

	    // Recursion, check base case (avoid pushing on stack)
	    if( ce_new == ne_new ) {
		if( ne_new == 0 )
		    Ee.record( depth ); // R.size() );
		// done
	    // Tunable
	    } else {
		bool ok = false;

		// Clear min/max size requirements to attempt
		if constexpr ( with_leaf ) {
		    if( ce_new - ne_new >= TUNABLE_SMALL_AVOID_CUTOUT_LEAF
			&& ( ce_new <= (1<<N_MAX_SIZE)
			     || ( ne_new <= (1<<X_MAX_SIZE)
				  && ce_new - ne_new <= (1<<P_MAX_SIZE) ) )
			) {

			/* xp_iterative retains sort order
			if constexpr ( HGraphTy::has_dual_rep ) {
			    assert( std::is_sorted( XP_new, XP_new+ne_new ) );
			    std::sort( XP_new, XP_new+ne_new );
			    assert( std::is_sorted(
					XP_new+ne_new, XP_new+ce_new ) );
			    std::sort( XP_new+ne_new, XP_new+ce_new );
			}
			*/

			// ok = mce_leaf<VID,EID>(
			// G, Ee, depth, XP_new, ne_new, ce_new );
			ok = false; // tmp
		    }
		}

		if( !ok ) { // Need recursion
		    // Recursion - push new frame
		    assert( depth+1 < degeneracy+1 );
		    frame_t & nfr = frame[depth++];
		    new ( &nfr ) frame_t {
			fr.XP_new, ne_new, ce_new, 0, nullptr, nullptr, ne_new };

		    nfr.pivot = mc_get_pivot_xp( G, XP_new, ne_new, ce_new ).first;
		    nfr.XP_new = alloc.template allocate<VID>( ce_new );
		    nfr.prev_tgt = XP_new;
		    // assert( R.size() == depth );
		    // Go to handle top frame
		    continue;
		}
	    }

	    // R.pop();
	    
	    // Move candidate (u) from original position to appropriate
	    // place in X part (maintaining sort order).
	    // Cache tgt for next iteration as next iteration's u
	    // will be strictly larger.
	    VID * tgt = std::upper_bound( fr.prev_tgt, XP+ne, u );
	    if( tgt != &XP[fr.i] ) { // equality when u moves to tgt == XP+ne
		std::copy_backward( tgt, &XP[fr.i], &XP[fr.i+1] );
		*tgt = u;
	    }
	    fr.prev_tgt = tgt+1;
	    ++fr.ne;
	}

	++fr.i;
    }

    // alloc.deallocate_to( frame[0].XP_new );
    assert( frame[0].XP_new == 0 );
    delete[] frame;
}
#endif


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

    const auto & get_graph() const { return S; }
    VID get_start_pos() const { return start_pos; }

private:
    graptor::graph::GraphHAdjPA<VID,EID,dual_rep,Hash> S;
    std::vector<VID> s2g;
    VID start_pos;
};

//! recursively parallel version of Bron-Kerbosch w/ pivoting
//
// XP may be modified by the method. It is not required to be in sort order.
void
mce_bron_kerbosch_recpar_xps(
    const HGraphTy & G,
    VID degeneracy,
    MCE_Parallel_Enumerator & E,
    XPSet<VID> & xp,
    VID ne,
    VID ce,
    int depth ) {
    // Termination condition
    if( ne == ce ) {
	if( ne == 0 )
	    E.record( depth );
	return;
    }
    const VID n = G.numVertices();

    // Get pivot and its number of common neighbours with [XP+ne,XP+ce)
    // The number of neighbours may be zero of it is not considered worthwhile
    // to apply pivoting (few candidates).
    const auto pp = mc_get_pivot_xp( G, xp, ne, ce );
    const VID pivot = pp.first;
    const VID sum = pp.second;

    const auto & padj = G.get_adjacency( pivot );

    // Pre-sort, affects vertex selection order in recursive calls
    // We own XP and are entitled to modify it.
    VID pe = ce - sum; // neighbours of pivot moved to end.

    // Special case, occurs in warwiki: pe == ne, i.e., all vertices
    // to be postponed to different top-level vertex. This is very strong
    // filtering as a result of pivoting. This results only when selecting
    // an X vertex.
    // however occurs only if
    // we don't make a cut-out, i.e., it results from selecting an X vertex.
    // The method will stop immediately (zero iterations in reorderig loop,
    // zero iterations in main parallel loop); no specific checks are useful.
    
    if( sum > 0 ) { // sum == 0 disables pivoting
	// Semisort P into P\N(pivot) and P\cap N(pivot)
	xp.semisort_pivot( ne, pe, ce, padj );
    }

    if constexpr ( io_trace ) {
	std::lock_guard<std::mutex> guard( io_mux );
	std::cout << "XPS loop: ne=" << ne << " pe=" << pe
		  << " ce=" << ce << " depth=" << depth << "\n";
    }

    parallel_loop( ne, pe, 1, [&,ne,pe,ce]( VID i ) {
	VID v = xp.at( i );

	// Not keeping track of R

	const auto & adj = G.get_adjacency( v ); 
	VID deg = adj.size();

	if constexpr ( io_trace ) {
	    std::lock_guard<std::mutex> guard( io_mux );
	    std::cout << "XP2: X=" << i << " P=" << (ce - (i+1)) << " adj="
		      << adj.size() << " depth=" << depth << "\n";
	}

	if( deg == 0 ) { // implies ne == ce == 0
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
	    VID ne_new, ce_new;
	    XPSet<VID> xp_new
		= xp.intersect( G.numVertices(),
				i, ce, adj, ngh, ne_new, ce_new );

	    bool ok = false;
	    if( ce_new - ne_new == 0 ) {
		// Reached leaf of search tree
		if( ne_new == 0 )
		    E.record( depth+1 );
		ok = true;
	    } else if( ce_new - ne_new >= TUNABLE_SMALL_AVOID_CUTOUT_LEAF
		       // very small -> just keep going
		       && ( ce_new - ne_new <= /*4**/(1<<P_MAX_SIZE)
			    && ne_new <= (1<<X_MAX_SIZE)
			    // on way to a clique with deep recursion?
			    // || ( ne_new == 0 && ( ce - ce_new < ce_new / 100 ) )
			       ) ) {
		MCE_Enumerator_stage2 E2 = E.get_enumerator( 0 );
		ok = mce_leaf<VID,EID>(
		    G, E2, depth+1, xp_new, ne_new, ce_new );
/*
		if( false && !ok ) {
		    StackLikeAllocator alloc;
		    MCE_Enumerator_stage2 E2 = E.get_enumerator( depth+1 );
		    mce_iterate_xp_iterative( G, E2, alloc, degeneracy,
					      xp_new.get_set(), ne_new, ce_new );
		}
*/
	    }
	    // else
	    if( !ok )
	    {
#if ABLATION_RECPAR_CUTOUT == 0
		bool cutout
		    = G.getDegree( xp_new.at( ne_new ) ) > 2 * ( ce_new - ne_new );
#elif ABLATION_RECPAR_CUTOUT == 1
		constexpr bool cutout = false;
#elif ABLATION_RECPAR_CUTOUT == 2
		constexpr bool cutout = true;
#endif
		if( cutout ) {
		    // large sub-problem; search recursively, and also
		    // construct cut-out
		    // this cut-out needs sorted XP set...
		    xp_new.sort( ne_new );
		    GraphBuilderInduced<HGraphTy>
			builder( G, xp_new.get_set(), ne_new, ce_new );
		    const auto & Gc = builder.get_graph();
		    std::iota( xp_new.get_set(), xp_new.get_set()+ce_new, 0 );
		    mce_bron_kerbosch_recpar_xps(
			Gc, degeneracy, E, xp_new, ne_new, ce_new, depth+1 );
		} else {
		    // large sub-problem; search recursively
		    mce_bron_kerbosch_recpar_xps(
			G, degeneracy, E, xp_new, ne_new, ce_new, depth+1 );
		}
	    }
	}
    } );
}


void
mce_bron_kerbosch_recpar_top(
    const HGraphTy & G,
    VID start_pos,
    VID degeneracy,
    MCE_Parallel_Enumerator & E ) {
    VID n = G.numVertices();

    // Specialised version of pivoting: avoid creating full XP list (0..ce)
    // and performing intersections to find pivot. Instead look at highest
    // degree.
    // TODO: in hadj, keep track of adjacency list size for X vs P neighbours.
    // Note: for some reason, pivoting here is very inefficient for warwiki
/*
    VID pivot = 0;
    VID pivot_degree = 0;
    for( VID v=0; v < n; ++v ) {
	VID deg = G.getDegree( v );
	VID pdeg = deg;
	if( v >= start_pos ) {
	    const VID * n = G.get_neighbours( v );
	    const VID * pos = std::upper_bound( n, n+deg, start_pos );
	    pdeg = std::distance( pos, n+deg );
	}
	if( pdeg > pivot_degree ) {
	    pivot_degree = pdeg;
	    pivot = v;
	}
    }

    const auto & padj = G.get_adjacency( pivot );
*/

    // start_pos calculated to avoid revisiting vertices ordered before the
    // reference vertex of this cut-out
#if 1
    parallel_loop( start_pos, n, 1, [&]( VID v ) {
	// Note: push v into R
	VID ce = G.getDegree( v );
#if !ABLATION_DISABLE_TOP_TINY
	// Consider this case before construction of XPSet. Note that dense
	// and blocked leaf tasks require XPSet for cutout construction.
	if( ce == 0 ) {
	    // This is a 2-clique (top-level vertex plus our v). mce_tiny
	    // handles his special case incorrectly for this call site.
	    E.record( 1 );
	}  else
#endif
	{
	    const VID * const ngh = G.get_neighbours( v );
	    VID ne = std::upper_bound( ngh, ngh+ce, v ) - ngh;

	    if constexpr ( io_trace ) {
		std::lock_guard<std::mutex> guard( io_mux );
		std::cout << "XPTOP loop: ne=" << ne << " ce=" << ce << "\n";
	    }

#if !ABLATION_DISABLE_TOP_TINY
	    if( ce <= 3 ) {
		MCE_Enumerator_stage2 E2 = E.get_enumerator( 0 );
		mce_tiny( G, ngh, ne, ce, E2 );
	    } else
#endif
	    {
		XPSet<VID> xp = XPSet<VID>::create_top_level( G, v );

		bool ok = false;
		if( ne <= (1<<P_MAX_SIZE) && ne <= (1<<X_MAX_SIZE) ) {
		    MCE_Enumerator_stage2 E2 = E.get_enumerator( 0 );
		    ok = mce_leaf<VID,EID>( G, E2, 1, xp, ne, ce );
		}

		if( !ok )
		    mce_bron_kerbosch_recpar_xps( G, degeneracy, E, xp, ne, ce, 1 );
	    }
	}
    } );
#elif 1
    parallel_loop( start_pos, n, 1, [&]( VID v ) {
	// Note: push v into R
	VID ce = G.getDegree( v );
	const VID * const ngh = G.get_neighbours( v );
	VID ne = std::upper_bound( ngh, ngh+ce, v ) - ngh;

	if constexpr ( io_trace ) {
	    std::lock_guard<std::mutex> guard( io_mux );
	    std::cout << "XPTOP loop: ne=" << ne << " ce=" << ce << "\n";
	}

	XPSet<VID> xp = XPSet<VID>::create_top_level( G, v );
	mce_bron_kerbosch_recpar_xps( G, degeneracy, E, xp, ne, ce, 1 );
    } );
#else
    VID * v_parallel = new VID[n-start_pos];
    VID n_parallel = 0;
    
    for( VID v=start_pos; v < n; ++v ) {
	VID ce = G.getDegree( v );
#if !ABLATION_DISABLE_TOP_TINY
	// Consider this case before construction of XPSet. Note that dense
	// and blocked leaf tasks require XPSet for cutout construction.
	if( ce == 0 ) {
	    // This is a 2-clique (top-level vertex plus our v). mce_tiny
	    // handles his special case incorrectly for this call site.
	    E.record( 1 );
	}  else
#endif
	{
	    const VID * const ngh = G.get_neighbours( v );
	    VID ne = std::upper_bound( ngh, ngh+ce, v ) - ngh;

	    if constexpr ( io_trace ) {
		std::lock_guard<std::mutex> guard( io_mux );
		std::cout << "XPTOP loop: ne=" << ne << " ce=" << ce << "\n";
	    }

#if !ABLATION_DISABLE_TOP_TINY
	    if( ce <= 3 ) {
		MCE_Enumerator_stage2 E2 = E.get_enumerator( 0 );
		mce_tiny( G, ngh, ne, ce, E2 );
	    } else
#endif
	    {
		bool ok = false;
		if( ce - ne <= (1<<P_MAX_SIZE) && ne <= (1<<X_MAX_SIZE)
		    && ce < degeneracy / 2 ) {
		    MCE_Enumerator_stage2 E2 = E.get_enumerator( 0 );
		    XPSet<VID> xp = XPSet<VID>::create_top_level( G, v );
		    ok = mce_leaf<VID,EID>( G, E2, 1, xp, ne, ce );
		}

		if( !ok )
		    v_parallel[n_parallel++] = v;
	    }
	}
    }

    parallel_loop( VID(0), n_parallel, 1, [&]( VID i ) {
	VID v = v_parallel[i];
	VID ce = G.getDegree( v );
	const VID * const ngh = G.get_neighbours( v );
	VID ne = std::upper_bound( ngh, ngh+ce, v ) - ngh;
	XPSet<VID> xp = XPSet<VID>::create_top_level( G, v );

	bool ok = false;
	if( ce - ne <= (1<<P_MAX_SIZE) && ne <= (1<<X_MAX_SIZE) ) {
	    MCE_Enumerator_stage2 E2 = E.get_enumerator( 0 );
	    ok = mce_leaf<VID,EID>( G, E2, 1, xp, ne, ce );
	}

	if( !ok )
	    mce_bron_kerbosch_recpar_xps( G, degeneracy, E, xp, ne, ce, 1 );
    } );

    delete[] v_parallel;

#endif
}

#if 0
template<typename VID, typename EID, typename Hash>
void
mce_bron_kerbosch_par_xp(
    const graptor::graph::GraphHAdjTable<VID,EID,Hash> & G,
    VID start_pos,
    VID degeneracy,
    MCE_Enumerator_stage2 & E ) {
    VID n = G.numVertices();

    // start_pos calculated to avoid revisiting vertices ordered before the
    // reference vertex of this cut-out
    parallel_loop( start_pos, n, 1, [&]( VID v ) {
	StackLikeAllocator alloc;
	// contract::vertex_set<VID> R;

	// R.push( v );

	// Consider as candidates only those neighbours of v that are larger
	// than u to avoid revisiting the vertices unnecessarily.
	auto & adj = G.get_adjacency( v ); 

	VID deg = adj.size();
	VID * XP = alloc.template allocate<VID>( deg );
	auto end = std::copy_if(
	    adj.begin(), adj.end(), XP,
	    [&]( VID v ) { return v != adj.invalid_element; } );
	assert( end - XP == deg );
	std::sort( XP, XP+deg );
	const VID * const start = std::upper_bound( XP, XP+deg, v );
	VID ne = start - XP;
	VID ce = deg;

	// TODO: further parallel decomposition
	// mce_iterate_xp( G, E, alloc, R, XP, ne, ce, 1 );
	MCE_Enumerator_stage2 Ee( E, 1 );
	mce_iterate_xp_iterative( G, Ee, alloc, degeneracy, XP, ne, ce );
    } );
}
#endif


void check_clique_edges( EID m, const VID * assigned_clique, EID ce ) {
    EID cce = 0;
    for( EID e=0; e != m; ++e )
	if( ~assigned_clique[e] != 0 )
	    ++cce;
    assert( cce == ce );
}

template<unsigned XBits, unsigned PBits, typename HGraph, typename Enumerator>
void mce_blocked_fn(
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

    MCE_Enumerator_stage2 E2( E );
    mce_bron_kerbosch( IG, E2 );

    stats.record( tm.stop() );
}

template<unsigned Bits, typename HGraph, typename Enumerator>
void mce_dense_fn(
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

    MCE_Enumerator_stage2 E2( E );
    IG.mce_bron_kerbosch( E2 );

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

typedef void (*mce_func)(
    const GraphCSx &, 
    const HFGraphTy &,
    MCE_Enumerator &,
    VID,
    const graptor::graph::NeighbourCutOutDegeneracyOrder<VID,EID> & cut,
    variant_statistics & );
    
static mce_func mce_dense_func[N_DIM+1] = {
    &mce_dense_fn<32,HFGraphTy,MCE_Enumerator>,  // N=32
    &mce_dense_fn<64,HFGraphTy,MCE_Enumerator>,  // N=64
    &mce_dense_fn<128,HFGraphTy,MCE_Enumerator>, // N=128
    &mce_dense_fn<256,HFGraphTy,MCE_Enumerator>, // N=256
    &mce_dense_fn<512,HFGraphTy,MCE_Enumerator>  // N=512
};

static mce_func mce_blocked_func[X_DIM+1][P_DIM+1] = {
    // X == 2**5
    { &mce_blocked_fn<32,32,HFGraphTy,MCE_Enumerator>,  // X=32, P=32
      &mce_blocked_fn<32,64,HFGraphTy,MCE_Enumerator>,  // X=32, P=64
      &mce_blocked_fn<32,128,HFGraphTy,MCE_Enumerator>, // X=32, P=128
      &mce_blocked_fn<32,256,HFGraphTy,MCE_Enumerator>, // X=32, P=256
      &mce_blocked_fn<32,512,HFGraphTy,MCE_Enumerator>  // X=32, P=512
    },
    // X == 2**6
    { &mce_blocked_fn<64,32,HFGraphTy,MCE_Enumerator>,  // X=64, P=32
      &mce_blocked_fn<64,64,HFGraphTy,MCE_Enumerator>,  // X=64, P=64
      &mce_blocked_fn<64,128,HFGraphTy,MCE_Enumerator>, // X=64, P=128
      &mce_blocked_fn<64,256,HFGraphTy,MCE_Enumerator>, // X=64, P=256
      &mce_blocked_fn<64,512,HFGraphTy,MCE_Enumerator>  // X=64, P=512
    },
    // X == 2**7
    { &mce_blocked_fn<128,32,HFGraphTy,MCE_Enumerator>,  // X=128, P=32
      &mce_blocked_fn<128,64,HFGraphTy,MCE_Enumerator>,  // X=128, P=64
      &mce_blocked_fn<128,128,HFGraphTy,MCE_Enumerator>, // X=128, P=128
      &mce_blocked_fn<128,256,HFGraphTy,MCE_Enumerator>, // X=128, P=256
      &mce_blocked_fn<128,512,HFGraphTy,MCE_Enumerator>  // X=128, P=512
    },
    // X == 2**8
    { &mce_blocked_fn<256,32,HFGraphTy,MCE_Enumerator>,  // X=256, P=32
      &mce_blocked_fn<256,64,HFGraphTy,MCE_Enumerator>,  // X=256, P=64
      &mce_blocked_fn<256,128,HFGraphTy,MCE_Enumerator>, // X=256, P=128
      &mce_blocked_fn<256,256,HFGraphTy,MCE_Enumerator>, // X=256, P=256
      &mce_blocked_fn<256,512,HFGraphTy,MCE_Enumerator>  // X=256, P=512
    },
    // X == 2**9
    { &mce_blocked_fn<512,32,HFGraphTy,MCE_Enumerator>,  // X=512, P=32
      &mce_blocked_fn<512,64,HFGraphTy,MCE_Enumerator>,  // X=512, P=64
      &mce_blocked_fn<512,128,HFGraphTy,MCE_Enumerator>, // X=512, P=128
      &mce_blocked_fn<512,256,HFGraphTy,MCE_Enumerator>, // X=512, P=256
      &mce_blocked_fn<512,512,HFGraphTy,MCE_Enumerator>  // X=512, P=512
    }
};

size_t get_size_class( uint32_t v ) {
    size_t b = _lzcnt_u32( v-1 );
    size_t cl = 32 - b;
    assert( v <= (1<<cl) );
    return cl;
}

void mce_top_level(
    const GraphCSx & G,
    const HFGraphTy & H,
    MCE_Parallel_Enumerator & E,
    VID v,
    VID degeneracy ) {
    graptor::graph::NeighbourCutOutDegeneracyOrder<VID,EID> cut( G, v );

    all_variant_statistics & stats = mce_stats.get_statistics();

    VID num = cut.get_num_vertices();

    // Do not count 1-cliques for isolated vertices
    // Blanusa's algorithm does sometimes count these, but not always.
    if( num == 0 )
	return;

#if !ABLATION_DISABLE_TOP_TINY
    if( num <= 3 ) {
	timer tm;
	tm.start();
	MCE_Enumerator_stage2 E2 = E.get_enumerator( 0 );
	mce_tiny( H, cut.get_vertices(), cut.get_start_pos(),
		  cut.get_num_vertices(), E2 );
	stats.record_tiny( tm.stop() );
	return;
    }
#endif

    VID xnum = cut.get_start_pos();
    VID pnum = num - xnum;

#if !ABLATION_DISABLE_TOP_DENSE
    VID nlg = get_size_class( num );
    if( nlg < N_MIN_SIZE )
	nlg = N_MIN_SIZE;

    MCE_Enumerator & E2 = E.get_enumerator_ref();

    if( nlg <= N_MIN_SIZE+1 ) { // up to 64 bits
	return mce_dense_func[nlg-N_MIN_SIZE](
	    G, H, E2, v, cut, stats.get( nlg ) );
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
	return mce_dense_func[nlg-N_MIN_SIZE](
	    G, H, E2, v, cut, stats.get( nlg ) );
    }

    if( xlg <= X_MAX_SIZE && plg <= P_MAX_SIZE ) {
	return mce_blocked_func[xlg-X_MIN_SIZE][plg-P_MIN_SIZE](
	    G, H, E2, v, cut, stats.get( xlg, plg ) );
    }

    if( nlg <= N_MAX_SIZE ) {
	return mce_dense_func[nlg-N_MIN_SIZE](
	    G, H, E2, v, cut, stats.get( nlg ) );
    }
#endif

    timer tm;
    tm.start();
    GraphBuilderInduced<HGraphTy> ibuilder( G, H, v, cut );
    const auto & HG = ibuilder.get_graph();

    stats.record_genbuild( tm.stop() );

    tm.start();
    MCE_Parallel_Enumerator Ee( E, 1 );
    mce_bron_kerbosch_recpar_top( HG, ibuilder.get_start_pos(),
				  degeneracy, Ee );
    double t = tm.stop();
    if( io_trace /* false && t >= 3.0 */ ) {
	// P-P edges more informative than X-P edges...
	EID edges = 0, sq = 0;
	for( VID u=ibuilder.get_start_pos(); u < HG.numVertices(); ++u ) {
	    EID d = HG.get_adjacency( u ).size();
	    edges += d;
	    sq += d * d;
	}
	std::cerr << "generic v=" << v << " num=" << num
		  << " xnum=" << xnum << " pnum=" << pnum
	    // << " density=" << HG.density()
	    // << " m=" << HG.numEdges()
		  << " t=" << t
		  << " e=" << edges
		  << " sq=" << sq
		  << " sq/e=" << float(sq)/float(edges)
		  << " density=" << float(edges)/(float(HG.numVertices())*float(HG.numVertices()))
		  << "\n";
    }
    stats.record_gen( t );
}

template<unsigned Bits, typename VID, typename EID>
void leaf_dense_fn(
    const HGraphTy & H,
    MCE_Enumerator_stage2 & Ee,
    VID r,
    const XPSet<VID> & xp_set,
    VID ne,
    VID ce ) {
    DenseMatrix<Bits,VID,EID> D( H, H, xp_set, ne, ce );
    D.mce_bron_kerbosch( [&]( const bitset<Bits> & c, size_t sz ) {
	Ee.record( r + sz );
    } );
}

template<unsigned XBits, unsigned PBits, typename VID, typename EID>
void leaf_blocked_fn(
    const HGraphTy & H,
    MCE_Enumerator_stage2 & Ee,
    VID r,
    const XPSet<VID> & xp_set,
    VID ne,
    VID ce ) {
    BlockedBinaryMatrix<XBits,PBits,VID,EID>
	D( H, H, xp_set, ne, ce );
    mce_bron_kerbosch( D, [&]( const bitset<PBits> & c, size_t sz ) {
	Ee.record( r + sz );
    } );
}

typedef void (*mce_leaf_func)(
    const HGraphTy &,
    MCE_Enumerator_stage2 & Ee,
    VID,
    const XPSet<VID> &,
    VID,
    VID );
    
static mce_leaf_func leaf_dense_func[N_DIM+1] = {
    &leaf_dense_fn<32,VID,EID>,  // N=32
    &leaf_dense_fn<64,VID,EID>,  // N=64
    &leaf_dense_fn<128,VID,EID>, // N=128
    &leaf_dense_fn<256,VID,EID>, // N=256
    &leaf_dense_fn<512,VID,EID>  // N=512
};

static mce_leaf_func leaf_blocked_func[X_DIM+1][P_DIM+1] = {
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

template<typename VID, typename EID>
bool mce_leaf(
    const HGraphTy & H,
    MCE_Enumerator_stage2 & E,
    VID r,
    const XPSet<VID> & xp_set,
    VID ne,
    VID ce ) {
#if ABLATION_DISABLE_LEAF
    return false;
#else
    VID num = ce;
    VID xnum = ne;
    VID pnum = ce - ne;
    VID * XP = xp_set.get_set();

    if( ce <= 3 ) {
	MCE_Enumerator_stage2 E2( E, r-1 );
	mce_tiny( H, XP, ne, ce, E2 );
	return true;
    }

    VID nlg = get_size_class( num );
    if( nlg < N_MIN_SIZE )
	nlg = N_MIN_SIZE;

#if 0
    if( nlg <= N_MIN_SIZE+1 ) { // up to 64 bits
	leaf_dense_func[nlg-N_MIN_SIZE]( H, E, r, xp_set, ne, ce );
	return true;
    }
#endif

    VID xlg = get_size_class( xnum );
    if( xlg < X_MIN_SIZE )
	xlg = X_MIN_SIZE;

    VID plg = get_size_class( pnum );
    if( plg < P_MIN_SIZE )
	plg = P_MIN_SIZE;

    if( nlg <= xlg + plg && nlg <= N_MAX_SIZE ) {
	leaf_dense_func[nlg-N_MIN_SIZE]( H, E, r, xp_set, ne, ce );
	return true;
    }

    if( xlg <= X_MAX_SIZE && plg <= P_MAX_SIZE ) {
	leaf_blocked_func[xlg-X_MIN_SIZE][plg-P_MIN_SIZE](
	    H, E, r, xp_set, ne, ce );
	return true;
    }

    if( nlg <= N_MAX_SIZE ) {
	leaf_dense_func[nlg-N_MIN_SIZE]( H, E, r, xp_set, ne, ce );
	return true;
    }

    return false;
#endif // ABLATION_DISABLE_LEAF
}

int main( int argc, char *argv[] ) {
    commandLine P( argc, argv, " help" );
    bool symmetric = P.getOptionValue("-s");
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

    GraphCSRAdaptor GA( G, 256 );
    KCv<GraphCSRAdaptor> kcore( GA, P );
    kcore.run();
    auto & coreness = kcore.getCoreness();
    std::cout << "Calculating coreness: " << tm.next() << "\n";
    std::cout << "coreness=" << kcore.getLargestCore() << "\n";

    mm::buffer<VID> order( n, numa_allocation_interleaved() );
    mm::buffer<VID> rev_order( n, numa_allocation_interleaved() );
#if ABLATION_SORT_ORDER_TIES
    sort_order( order.get(), rev_order.get(),
		coreness.get_ptr(), n, kcore.getLargestCore() );
#else
    sort_order_ties( order.get(), rev_order.get(),
		     coreness.get_ptr(), n, kcore.getLargestCore(),
		     GA.getOutDegree() );
#endif
    global_coreness = coreness.get_ptr();
    std::cout << "Determining sort order: " << tm.next() << "\n";

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
	      << "\n\tABLATION_PIVOT_DISABLE_XP_HASH="
	      << ABLATION_PIVOT_DISABLE_XP_HASH
	      << "\n\tTUNABLE_SMALL_AVOID_CUTOUT_LEAF="
	      << TUNABLE_SMALL_AVOID_CUTOUT_LEAF
	      << "\n\tABLATION_SORT_ORDER_TIES=" << ABLATION_SORT_ORDER_TIES
	      << "\n\tABLATION_RECPAR_CUTOUT=" << ABLATION_RECPAR_CUTOUT
	      << "\n\tPAR_LOOP=" << PAR_LOOP
	      << "\n\tPAR_DENSE=" << PAR_DENSE
	      << "\n\tPAR_BLOCKED=" << PAR_BLOCKED
	      << '\n';
    
    MCE_Enumerator_Farm farm( kcore.getLargestCore() );
    MCE_Parallel_Enumerator E( farm );

    system( "date" );
    std::cout << "Start enumeration: " << tm.next() << "\n";

    // Number of partitions is tunable. A fairly large number is helpful
    // to help load balancing.
    VID nthreads = graptor_num_threads();
    VID npart = nthreads * 128;
#if PAR_LOOP == 0
    parallel_loop( VID(0), npart, 1, [&,npart]( VID p ) {
	VID k = n / npart; // round down as lower-numbered vertices take longer
	VID from = p == 0 ? 0 : (p-1) * k;
	VID to = p == npart-1 ? n : p * k;
	for( VID i=from; i < to; i++ ) {
	    VID v = i; // order[i];
	    mce_top_level( R, H, E, v, kcore.getLargestCore() );
	}
    } );
#elif PAR_LOOP == 1
    parallel_loop( VID(0), npart, 1, [&,npart]( VID p ) {
	for( VID i=p; i < n; i += npart ) {
	    VID v = i; // order[i];
	    mce_top_level( R, H, E, v, kcore.getLargestCore() );
	}
    } );
#elif PAR_LOOP == 2
    parallel_loop( VID(0), npart, 1, [&,npart]( VID p ) {
	VID k = ( n + npart - 1 ) / npart;
	VID degeneracy = kcore.getLargestCore();
	while( p + k * npart >= n )
	    --k;
	// Split out vertices in higher-degree ones and lower-degree ones,
	// processing the latter sequentially for efficiency reasons
	VID kpar = k;
	for( VID v=p; v < n; v += npart ) {
	    if( R.getDegree( v ) < degeneracy ) {
		kpar = ( v - p ) / npart;
		break;
	    }
	}
	// Do sequential part first, because work stealing cannot help to
	// obtain load balance if we keep this to the end of the computation.
	for( VID v=p+kpar*npart; v < n; v += npart )
	    mce_top_level( R, H, E, v, kcore.getLargestCore() );
	parallel_loop( VID(0), kpar, 1, [&,p,k]( VID j ) {
	    VID v = p + j * npart;
	    mce_top_level( R, H, E, v, kcore.getLargestCore() );
	} );
    } );
#endif

    std::cout << "Enumeration: " << tm.next() << "\n";

    all_variant_statistics stats = mce_stats.sum();

    double duration = tm.total();
    std::cout << "Completed MCE in " << duration << " seconds\n";
    for( size_t n=N_MIN_SIZE; n <= N_MAX_SIZE; ++n ) {
	std::cout << (1<<n) << "-bit dense: ";
	stats.get( n ).print( std::cout ); 
    }
    for( size_t x=X_MIN_SIZE; x <= X_MAX_SIZE; ++x )
	for( size_t p=P_MIN_SIZE; p <= P_MAX_SIZE; ++p ) {
	    std::cout << (1<<x) << ',' << (1<<p) << "-bit blocked: ";
	    stats.get( x, p ).print( std::cout );
	}
    std::cout << "tiny: ";
    stats.m_tiny.print( std::cout );
    std::cout << "generic: ";
    stats.m_gen.print( std::cout );

    farm.sum().report( std::cout );

    rev_order.del();
    order.del();
    G.del();

    return 0;
}
