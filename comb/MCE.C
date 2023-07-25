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
#include "graptor/graph/GraphPDG.h"
#include "graptor/graph/partitioning.h"

#include "graptor/graph/simple/csx.h"
#include "graptor/graph/simple/dicsx.h"
#include "graptor/graph/simple/hadj.h"
#include "graptor/graph/simple/hadjt.h"
#include "graptor/graph/simple/cutout.h"
#include "graptor/graph/simple/dense.h"
#include "graptor/graph/simple/blocked.h"
#include "graptor/graph/transform/rmself.h"

#include "graptor/container/bitset.h"
#include "graptor/container/hash_fn.h"
#include "graptor/container/intersect.h"

#ifndef TUNABLE_SMALL_AVOID_CUTOUT
#define TUNABLE_SMALL_AVOID_CUTOUT 12
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

// using HGraphTy = graptor::graph::GraphHAdjTable<VID,EID,hash_fn>;
using HGraphTy = graptor::graph::GraphHAdjPA<VID,EID,true,hash_fn>;
// using HFGraphTy = graptor::graph::GraphHAdj<VID,EID,GraphCSx,hash_fn>;
using HFGraphTy = graptor::graph::GraphHAdjPA<VID,EID,false,hash_fn>;

using graptor::graph::DenseMatrix;
using graptor::graph::BinaryMatrix;
using graptor::graph::BlockedBinaryMatrix;

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

class MCE_Enumerator {
public:
    MCE_Enumerator( size_t degen = 0 )
	: m_degeneracy( degen ),
	  m_histogram( degen+1 ) { }

    // Recod clique of size s
    void record( size_t s ) {
	assert( s <= m_degeneracy+1 );
	__sync_fetch_and_add( &m_histogram[s-1], 1 );
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


struct variant_statistics {
    variant_statistics()
	: m_tm( 0 ), m_max( std::numeric_limits<double>::min() ),
	  m_calls( 0 ) { }
    variant_statistics( double tm, double mx, size_t calls )
	: m_tm( tm ), m_max( mx ), m_calls( calls ) { }

    variant_statistics operator + ( const variant_statistics & s ) const {
	return variant_statistics( m_tm + s.m_tm,
				   std::max( m_max, s.m_max ),
				   m_calls + s.m_calls );
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
		  << " s/call; max " << m_max << "\n";
    }
    
    double m_tm, m_max;
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
	sum.m_genbuild = m_genbuild + s.m_genbuild;
	return sum;
    }

    void record_tiny( double atm ) { m_tiny.record( atm ); }
    void record_gen( double atm ) { m_gen.record( atm ); }
    void record_genbuild( double atm ) { m_genbuild.record( atm ); }

    variant_statistics & get( size_t n ) {
	return m_dense[n-N_MIN_SIZE];
    }
    variant_statistics & get( size_t x, size_t p ) {
	return m_blocked[x-X_MIN_SIZE][p-P_MIN_SIZE];
    }
    
    variant_statistics m_dense[N_DIM];
    variant_statistics m_blocked[X_DIM][P_DIM];
    variant_statistics m_tiny, m_gen, m_genbuild;

};

// thread_local static all_variant_statistics * mce_pt_stats = nullptr;

struct per_thread_statistics {
    all_variant_statistics & get_statistics() {
	const pthread_t tid = pthread_self();
	std::lock_guard<std::mutex> guard( m_mutex );
	auto it = m_stats.find( tid );
	if( it == m_stats.end() ) {
	    auto it2 = m_stats.emplace(
		std::make_pair( tid, all_variant_statistics() ) );
	    return it2.first->second;
	}
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

template<unsigned XBits, unsigned PBits, typename sVID, typename sEID,
	 typename Enumerate>
void
mce_vertex_cover_iterate(
    const BlockedBinaryMatrix<XBits,PBits,sVID,sEID> & mtx,
    Enumerate && EE,
    sVID n_remaining,
    sVID k,
    typename BinaryMatrix<PBits,sVID,sEID>::row_type rm,
    typename BinaryMatrix<PBits,sVID,sEID>::row_type vc,
    typename BinaryMatrix<XBits,sVID,sEID>::row_type Xx,
    typename BinaryMatrix<PBits,sVID,sEID>::row_type Xp );

template<unsigned XBits, unsigned PBits, typename sVID, typename sEID>
void
trace_path(
    const BlockedBinaryMatrix<XBits,PBits,sVID,sEID> & mtx,
    typename BinaryMatrix<PBits,sVID,sEID>::row_type & visited,
    typename BinaryMatrix<PBits,sVID,sEID>::row_type & rm,
    typename BinaryMatrix<PBits,sVID,sEID>::row_type & vc,
    sVID cur, sVID nxt, bool incl ) {

    using ptr = typename BinaryMatrix<PBits,sVID,sEID>::tr;
    using prow_type = typename BinaryMatrix<PBits,sVID,sEID>::row_type;
    using xtr = typename BinaryMatrix<XBits,sVID,sEID>::tr;
    using xrow_type = typename BinaryMatrix<XBits,sVID,sEID>::row_type;

    const BinaryMatrix<PBits,sVID,sEID> & xp = mtx.get_xp();
    const BinaryMatrix<XBits,sVID,sEID> & px = mtx.get_px();

    sVID cs = xp.get_col_start();
    prow_type mask = ptr::bitwise_invert( ptr::himask( xp.numCols()+1 ) );

    prow_type v_set = xp.create_singleton_rel( nxt );

    if( !ptr::is_zero( ptr::bitwise_and( visited, v_set ) ) )
	return;

    std::cerr << "trace_path: " << cur << ", " << nxt
	      << ( incl ? " incl" : " excl" ) << "\n";

    visited = ptr::bitwise_or( visited, v_set );

    if( incl )
	vc = ptr::bitwise_or( vc, v_set );
    // Maintain invariant that vc is a subset of rm
    rm = ptr::bitwise_or( rm, v_set );

    // Note: nxt already included in rm, so self-edge not present in act_ngh
    prow_type nxt_ngh = ptr::bitwise_xor( mask, xp.get_row( nxt + cs ) );
    prow_type act_ngh = ptr::bitwise_andnot( rm, act_ngh );
    if( xp.get_size( act_ngh ) == 2 ) {
	bitset<PBits> b( act_ngh );
	auto I = b.begin();
	sVID ngh1 = cs + *I;
	++I;;
	sVID ngh2 = cs + *I;
	sVID ngh = ngh1 == cur ? ngh2 : ngh1;

	trace_path( mtx, visited, rm, vc, nxt, ngh, !incl );
    }
}

template<unsigned XBits, unsigned PBits, typename sVID, typename sEID,
	 typename Enumerate>
void
mce_vertex_cover_poly(
    const BlockedBinaryMatrix<XBits,PBits,sVID,sEID> & mtx,
    Enumerate && EE,
    sVID n_remaining,
    sVID k,
    typename BinaryMatrix<PBits,sVID,sEID>::row_type rm,
    typename BinaryMatrix<PBits,sVID,sEID>::row_type vc,
    typename BinaryMatrix<XBits,sVID,sEID>::row_type Xx,
    typename BinaryMatrix<PBits,sVID,sEID>::row_type Xp ) {

    using ptr = typename BinaryMatrix<PBits,sVID,sEID>::tr;
    using prow_type = typename BinaryMatrix<PBits,sVID,sEID>::row_type;
    using xtr = typename BinaryMatrix<XBits,sVID,sEID>::tr;
    using xrow_type = typename BinaryMatrix<XBits,sVID,sEID>::row_type;

    const BinaryMatrix<PBits,sVID,sEID> & xp = mtx.get_xp();
    const BinaryMatrix<XBits,sVID,sEID> & px = mtx.get_px();
    sVID n = mtx.numVertices();

    sVID cs = xp.get_col_start();
    sVID cn = xp.numCols();

    prow_type mask = ptr::bitwise_invert( ptr::himask( xp.numCols()+1 ) );

    prow_type visited = ptr::setzero();

    // Find paths
    bitset<PBits> b( ptr::bitwise_invert( rm ) );
    for( auto I=b.begin(), E=b.end(); I != E && *I < cn; ++I ) {
	sVID i = *I;
	sVID v = i + cs;
	prow_type v_set = xp.create_singleton_rel( i );
	prow_type v_inv = ptr::bitwise_xor( mask, xp.get_row( v ) );
	prow_type v_row = ptr::bitwise_andnot( v_set, v_inv );
	sVID deg = xp.get_size( ptr::bitwise_andnot( rm, v_row ) );
	if( deg == 1
	    && ptr::is_zero( ptr::bitwise_and( visited, v_set ) ) ) {
	    visited = ptr::bitwise_or( visited, v_set );
	    trace_path( mtx, visited, rm, vc,
			i, *bitset<PBits>( v_row ).begin(), true );
	    rm = ptr::bitwise_or( rm, visited );
	}
    }

    // Find cycles
    for( auto I=b.begin(), E=b.end(); I != E && *I < cn; ++I ) {
	sVID i = *I;
	sVID v = i + cs;
	prow_type v_set = xp.create_singleton_rel( i );
	prow_type v_inv = ptr::bitwise_xor( mask, xp.get_row( v ) );
	prow_type v_abc = ptr::bitwise_andnot( v_set, v_inv );
	prow_type v_row = ptr::bitwise_andnot( rm, v_abc );
	sVID deg = xp.get_size( v_row );
	if( deg == 2
	    && ptr::is_zero( ptr::bitwise_and( visited, v_set ) ) ) {
	    visited = ptr::bitwise_or( visited, v_set );
	    trace_path( mtx, visited, rm, vc,
			i, *bitset<PBits>( v_row ).begin(), false );
	    rm = ptr::bitwise_or( rm, visited );
	}
    }

    if( xp.get_size( visited ) <= k ) {
	prow_type fvc = ptr::bitwise_or( vc, visited );
	xrow_type Xxv = Xx;
	prow_type Xpv = Xp;
	bitset<PBits> b( // all vertices not in rm and not in visited
	    ptr::bitwise_invert( ptr::bitwise_or( rm, visited ) ) );
	for( auto I=b.begin(), E=b.end(); I != E && *I < cn; ++I ) {
	    Xxv = xtr::bitwise_and( Xxv, px.get_row( *I + cs ) );
	    Xpv = ptr::bitwise_and( Xpv, xp.get_row( *I + cs ) );
	}
	if( ptr::is_zero( Xpv ) && xtr::is_zero( Xxv ) )
	    EE( bitset<PBits>( ptr::bitwise_xor( mask, fvc ) ) );
    }
}

template<unsigned XBits, unsigned PBits, typename sVID, typename sEID,
	 typename Enumerate>
void
mce_vertex_cover_buss(
    const BlockedBinaryMatrix<XBits,PBits,sVID,sEID> & mtx,
    Enumerate && EE,
    sVID n_remaining,
    sVID k,
    typename BinaryMatrix<PBits,sVID,sEID>::row_type rm,
    typename BinaryMatrix<PBits,sVID,sEID>::row_type vc,
    typename BinaryMatrix<XBits,sVID,sEID>::row_type Xx,
    typename BinaryMatrix<PBits,sVID,sEID>::row_type Xp ) {

    using ptr = typename BinaryMatrix<PBits,sVID,sEID>::tr;
    using prow_type = typename BinaryMatrix<PBits,sVID,sEID>::row_type;
    using xtr = typename BinaryMatrix<XBits,sVID,sEID>::tr;
    using xrow_type = typename BinaryMatrix<XBits,sVID,sEID>::row_type;

    const BinaryMatrix<PBits,sVID,sEID> & xp = mtx.get_xp();
    const BinaryMatrix<XBits,sVID,sEID> & px = mtx.get_px();
    sVID n = mtx.numVertices();

    sVID cs = xp.get_col_start();
    sVID cn = xp.numCols();

    prow_type mask = ptr::bitwise_invert( ptr::himask( xp.numCols()+1 ) );

    // Count vertices with degree higher than k
    prow_type u = ptr::setzero();
    sVID u_size = 0;
    bitset<PBits> b( ptr::bitwise_invert( rm ) );
    for( auto I=b.begin(), E=b.end(); I != E && *I < cn; ++I ) {
	sVID i = *I;
	sVID v = i + cs;
	prow_type ngh = ptr::bitwise_xor( mask, xp.get_row( v ) );
	prow_type valid = ptr::bitwise_andnot( rm, ngh );
	sVID deg = xp.get_size( valid ) - 1;
	if( deg > k ) {
	    ++u_size;
	    u = ptr::bitwise_or( u, xp.create_singleton_rel( i ) );
	}
    }

    // Count number of edges in graph. Need to know full set U before
    // doing this, so requires second pass.
    bitset<PBits> bu( u );
    sEID u_m = 0;
    xrow_type Xxv = Xx;
    prow_type Xpv = Xp;
    for( auto I=bu.begin(), E=bu.end(); I != E; ++I ) {
	sVID i = *I;
	sVID v = i + cs;
	prow_type row = xp.get_row( v );
	prow_type ngh = ptr::bitwise_xor( mask, row );
	prow_type valid = ptr::bitwise_andnot( u, ngh );
	sVID deg = xp.get_size( valid ) - 1;
	u_m += deg;
	Xxv = xtr::bitwise_and( Xxv, px.get_row( v ) );
	Xpv = ptr::bitwise_and( Xpv, row );
    }

    // If kernel graph has more than k(k-|U|) edges, reject
    if( u_m > sEID(k) * sEID( k - u_size ) )
	return;

    // Find a cover for the remaining vertices
    sVID r_remaining = n_remaining - u_size;
    sVID r_k = k - u_size;
    prow_type r_rm = ptr::bitwise_or( rm, u );
    prow_type r_vc = ptr::bitwise_or( vc, u );
    mce_vertex_cover_iterate( mtx, EE, r_remaining, r_k, r_rm, r_vc,
			      Xxv, Xpv );
}

template<unsigned XBits, unsigned PBits, typename sVID, typename sEID,
	 typename Enumerate>
void
mce_vertex_cover_iterate(
    const BlockedBinaryMatrix<XBits,PBits,sVID,sEID> & mtx,
    Enumerate && EE,
    sVID n_remaining,
    sVID k,
    typename BinaryMatrix<PBits,sVID,sEID>::row_type rm,
    typename BinaryMatrix<PBits,sVID,sEID>::row_type vc,
    typename BinaryMatrix<XBits,sVID,sEID>::row_type Xx,
    typename BinaryMatrix<PBits,sVID,sEID>::row_type Xp ) {

    using ptr = typename BinaryMatrix<PBits,sVID,sEID>::tr;
    using prow_type = typename BinaryMatrix<PBits,sVID,sEID>::row_type;
    using xtr = typename BinaryMatrix<XBits,sVID,sEID>::tr;
    using xrow_type = typename BinaryMatrix<XBits,sVID,sEID>::row_type;

    const BinaryMatrix<PBits,sVID,sEID> & xp = mtx.get_xp();
    const BinaryMatrix<XBits,sVID,sEID> & px = mtx.get_px();
    sVID n = mtx.numVertices();

    sVID cs = xp.get_col_start();
    sVID cn = xp.numCols();

    prow_type mask = ptr::bitwise_invert( ptr::himask( xp.numCols()+1 ) );

    // Find vertex with maximum degree
    sVID max_i = 0;
    sVID max_deg = 0;
    prow_type max_ngh = ptr::setzero();
    bitset<PBits> b( ptr::bitwise_invert( rm ) );
    sEID m = 0;
    for( auto I=b.begin(), E=b.end(); I != E && *I < cn; ++I ) {
	sVID i = *I;
	sVID v = i + cs;
	prow_type ngh = ptr::bitwise_xor( mask, xp.get_row( v ) );
	prow_type valid = ptr::bitwise_andnot( rm, ngh );
	sVID deg = xp.get_size( valid ) - 1;
	m += deg;
	if( deg > max_deg ) {
	    max_deg = deg;
	    max_i = i;
	    max_ngh = valid;
	}
    }

    if( m == 0 ) {
	// All vertices not in rm have degree 0 in the reduced graph.
	// These are not part of the vertex cover, hence elements of the
	// max clique.
	// MCE requires that they have no common X neighbours.
	xrow_type Xxv = Xx;
	prow_type Xpv = Xp;
	for( auto I=b.begin(), E=b.end(); I != E && *I < cn; ++I ) {
	    Xxv = xtr::bitwise_and( Xxv, px.get_row( *I + cs ) );
	    Xpv = ptr::bitwise_and( Xpv, xp.get_row( *I + cs ) );
	}
	if( ptr::is_zero( Xpv ) && xtr::is_zero( Xxv ) )
	    EE( bitset<PBits>( ptr::bitwise_xor( mask, vc ) ) );
	return;
    }

    /* Polynomial specialisation works for finding single minimum but
       enumeration is awkward (especially if there are multiple paths/cycles
       in the residue graph.
       if( max_deg <= 2 ) {
       mce_vertex_cover_poly( mtx, EE, n_remaining, k, rm, vc, Xx, Xp );
       return;
       }
    */

    if( k == 0 ) // m != 0, hence uncovered edges remain
	return;

    static constexpr sVID c = 1;
    if( m/2 > c * k * k && max_deg > k ) {
	mce_vertex_cover_buss( mtx, EE, n_remaining, k, rm, vc, Xx, Xp );
	return;
    }

    prow_type v_set = xp.create_singleton_rel( max_i );

    // Create two sub-problems by branching on max_v
    // 1. Exclude max_v from the vertex cover. All its neighbours must be
    //    included, but only those neighbours not already branched on.
    prow_type x_sel = ptr::bitwise_andnot( v_set, max_ngh );
    prow_type x_rm = ptr::bitwise_or( rm, ptr::bitwise_or( x_sel, v_set ) );
    // -1 for vertex max_i
    sVID x_remaining = n_remaining - xp.get_size( x_sel ) - 1;
    prow_type x_vc = ptr::bitwise_or( vc, x_sel );
    sVID x_k = std::min( n_remaining-1-max_deg, k-max_deg );
    // Vertex max_v is excluded from VC, hence included in max clique
    xrow_type Xxv = xtr::bitwise_and( Xx, px.get_row( max_i + cs ) );
    prow_type Xpv = ptr::bitwise_and( Xp, xp.get_row( max_i + cs ) );
    if( k >= max_deg )
	mce_vertex_cover_iterate(
	    mtx, EE, x_remaining, x_k, x_rm, x_vc, Xxv, Xpv );

    // 2. Include max_v in the vertex cover.
    prow_type i_rm = ptr::bitwise_or( rm, v_set );
    prow_type i_vc = ptr::bitwise_or( vc, v_set );
    sVID i_remaining = n_remaining - 1;
    sVID i_k = k - 1;
    // if( k >= 1 )
    mce_vertex_cover_iterate( mtx, EE, i_remaining, i_k, i_rm, i_vc, Xx, Xp );
}

template<unsigned XBits, unsigned PBits, typename sVID, typename sEID,
	 typename Enumerate>
void
mce_vertex_cover_iterate_v1(
    const BlockedBinaryMatrix<XBits,PBits,sVID,sEID> & mtx,
    Enumerate && EE,
    sVID v,
    typename BinaryMatrix<PBits,sVID,sEID>::row_type cin,
    typename BinaryMatrix<PBits,sVID,sEID>::row_type cout,
    typename BinaryMatrix<PBits,sVID,sEID>::row_type unc,
    typename BinaryMatrix<XBits,sVID,sEID>::row_type Xx,
    typename BinaryMatrix<PBits,sVID,sEID>::row_type Xp,
    sVID cin_sz ) {

    using ptr = typename BinaryMatrix<PBits,sVID,sEID>::tr;
    using prow_type = typename BinaryMatrix<PBits,sVID,sEID>::row_type;
    using xtr = typename BinaryMatrix<XBits,sVID,sEID>::tr;
    using xrow_type = typename BinaryMatrix<XBits,sVID,sEID>::row_type;

    const BinaryMatrix<PBits,sVID,sEID> & xp = mtx.get_xp();
    const BinaryMatrix<XBits,sVID,sEID> & px = mtx.get_px();
    sVID n = mtx.numVertices();

    sVID cs = xp.get_col_start();

    // Leaf node
    if( v == xp.numCols() ) {
	// Check if all edges are covered: unc \ cin == empty
	if( ptr::is_zero( ptr::bitwise_andnot( cin, unc ) )
	    && xtr::is_zero( Xx ) && xtr::is_zero( Xp ) ) {
	    prow_type mask
		= ptr::bitwise_invert( ptr::himask( xp.numCols()+1 ) );
	    prow_type cinv = ptr::bitwise_xor( mask, cin );
	    EE( bitset<PBits>( cinv ) );
	}
	return;
    }

    // Early termination: bit < v in unc is set...
    // prow_type unc_past = ptr::bitwise_andnot( xp.get_himask( v+cs ), unc );
    // if( !ptr::is_zero( ptr::bitwise_andnot( cin, unc_past ) ) )
    // return;

    prow_type v_set = xp.create_singleton_rel( v );

    // isolated vertex?
    prow_type v_raw = xp.get_row( v+cs );
    prow_type mask = ptr::bitwise_invert( ptr::himask( xp.numCols()+1 ) );
    prow_type v_row = ptr::bitwise_andnot(
	v_set, ptr::bitwise_xor( v_raw, mask ) );
    VID deg = xp.get_size( v_row );
    xrow_type Xxv = xtr::bitwise_and( Xx, px.get_row( v+cs ) );
    prow_type Xpv = ptr::bitwise_and( Xp, v_raw );
    if( deg == 0 ) {
	mce_vertex_cover_iterate(
	    mtx, EE, v+1, cin, ptr::bitwise_or( cout, v_set ), unc,
	    Xxv, Xpv, cin_sz );
	return;
    }

    // Count number of covered neighbours.
    // All X vertices are automatically covered, but they are
    // not compared to deg
    VID num_covered = xp.get_size( ptr::bitwise_and( v_row, cin ) );

    // All neighbours included, so this vertex is not needed
    // Any neighbour not included, then this vertex must be included
    if( num_covered == deg ) {
	mce_vertex_cover_iterate(
	    mtx, EE, v+1, cin, ptr::bitwise_or( cout, v_set ), unc,
	    Xxv, Xpv, cin_sz );
	return;
    }

    // Count number of uncovered neighbours; only chance we have any
    // if cout_sz is non-zero
    VID num_uncovered = xp.get_size( ptr::bitwise_and( v_row, cout ) );
    if( num_uncovered > 0 ) {
	mce_vertex_cover_iterate(
	    mtx, EE, v+1, ptr::bitwise_or( cin, v_set ), cout,
	    unc, Xx, Xp, cin_sz+1 );
	return;
    }

    // Otherwise, try both ways.
    mce_vertex_cover_iterate(
	mtx, EE, v+1, cin, ptr::bitwise_or( cout, v_set ),
	ptr::bitwise_or( unc, v_row ), Xxv, Xpv, cin_sz );

    mce_vertex_cover_iterate(
	mtx, EE, v+1, ptr::bitwise_or( cin, v_set ), cout,
	unc, Xx, Xp, cin_sz+1 );
}


template<unsigned XBits, unsigned PBits, typename sVID, typename sEID,
	 typename Enumerate>
void
mce_vertex_cover(
    const BlockedBinaryMatrix<XBits,PBits,sVID,sEID> & mtx,
    Enumerate && EE ) {

    using ptr = typename BinaryMatrix<PBits,sVID,sEID>::tr;
    using prow_type = typename BinaryMatrix<PBits,sVID,sEID>::row_type;
    using xtr = typename BinaryMatrix<XBits,sVID,sEID>::tr;
    using xrow_type = typename BinaryMatrix<XBits,sVID,sEID>::row_type;

    const BinaryMatrix<PBits,sVID,sEID> & xp = mtx.get_xp();
    const BinaryMatrix<XBits,sVID,sEID> & px = mtx.get_px();
    assert( px.numRows() + px.numCols() == xp.numRows() );
    assert( px.numRows() == xp.numCols() );

    sVID n = mtx.numVertices();
    sVID cs = xp.get_col_start();

    // Now consider, recursively, all candidates to make minimal vertex covers
    prow_type z = ptr::setzero();
    prow_type Xp = ptr::setone();
    xrow_type Xx = xtr::setone();
    mce_vertex_cover_iterate( mtx, EE, n - cs, n - cs, z, z, Xx, Xp );
}


class timeout_exception : public std::exception {
public:
    explicit timeout_exception( uint64_t usec = 0, int idx = -1 )
	: m_usec( usec ), m_idx( idx ) { }
    timeout_exception( const timeout_exception & e )
	: m_usec( e.m_usec ), m_idx( e.m_idx ) { }
    timeout_exception & operator = ( const timeout_exception & e ) {
	m_idx = e.m_idx;
	m_usec = e.m_usec;
	return *this;
    }

    uint64_t usec() const noexcept { return m_usec; }
    int idx() const noexcept { return m_idx; }

    const char * what() const noexcept {
	return "timeout exception";
    }

private:
    uint64_t m_usec;
    int m_idx;
};

bool is_member( VID v, VID C_size, const VID * C_set ) {
    const VID * const pos = std::lower_bound( C_set, C_set+C_size, v );
    if( pos == C_set+C_size || *pos != v )
	return false;
    return true;
}

template<typename HGraph>
std::pair<VID,VID> mc_get_pivot_xp(
    const HGraph & G,
    const VID * XP,
    VID ne,
    VID ce ) {

    assert( ce - ne != 0 );

    // Tunable (|P| and selecting vertex from X or P)
    if( ce - ne <= 3 )
	return std::make_pair( XP[ne], 0 );

    VID v_max = ~VID(0);
    VID tv_max = std::numeric_limits<VID>::min();

    for( VID i=0; i < ce; ++i ) {
	VID v = XP[i];
	// VID v = XP[ce-1-i]; -- slower
	// VID v = XP[(i+ne)%ce]; -- makes no difference
	auto & hadj = G.get_adjacency( v );
	VID deg = hadj.size();
	if( deg <= tv_max )
	    continue;

	// Abort during intersection_size if size will be less than tv_max
	// Note: hash_vector is much slower in this instance
	size_t tv = graptor::hash_scalar::intersect_size_exceed(
	    XP+ne, XP+ce, hadj, tv_max );
	    
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
	if( verbose )
	    std::cerr << "sla " << this << ": constructor\n";
    }
    ~StackLikeAllocator() {
	for( chunk_t * c : m_chunks ) {
	    if( verbose )
		std::cerr << "sla " << this << ": delete chunk "
			  << c << "\n";
	    delete[] reinterpret_cast<char *>( c );
	}
	if( verbose )
	    std::cerr << "sla " << this << ": destructor done\n";
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
	if( verbose )
	    std::cerr << "sla " << this << ": new chunk " << c << "\n";
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
    const VID * XP,
    VID ne,
    VID ce );

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
		    if( ce_new - ne_new >= TUNABLE_SMALL_AVOID_CUTOUT
			&& ( ce_new <= (1<<N_MAX_SIZE)
			     || ( ne_new <= (1<<X_MAX_SIZE)
				  && ce_new - ne_new <= (1<<P_MAX_SIZE) ) )
			) {
			ok = mce_leaf<VID,EID>(
			    G, Ee, depth, XP_new, ne_new, ce_new );
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
	    graptor::hash_insert_iterator<graptor::hash_table<VID,Hash>>
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
	: S( H, H, XP, ne, ce, numa_allocation_interleaved() ),
	  start_pos( ne ) { }

    const auto & get_graph() const { return S; }
    VID get_start_pos() const { return start_pos; }

private:
    graptor::graph::GraphHAdjPA<VID,EID,dual_rep,Hash> S;
    std::vector<VID> s2g;
    VID start_pos;
};


void
mce_bron_kerbosch_seq_xp(
    const HGraphTy & G,
    VID start_pos,
    VID degeneracy,
    MCE_Enumerator_stage2 & E ) {
    VID n = G.numVertices();

    StackLikeAllocator alloc;

    // start_pos calculate to avoid revisiting vertices ordered before the
    // reference vertex of this cut-out
    for( VID v=start_pos; v < n; ++v ) {
	// contract::vertex_set<VID> R;
	// R.push( v );

	// Consider as candidates only those neighbours of v that are larger
	// than u to avoid revisiting the vertices unnecessarily.
	auto & adj = G.get_adjacency( v ); 

	VID deg = adj.size();
	VID * XP = alloc.template allocate<VID>( deg );
	if constexpr ( HGraphTy::has_dual_rep ) {
	    const VID * ngh = G.get_neighbours( v );
	    std::copy( ngh, ngh+deg, XP );
	} else {
	    auto end = std::copy_if(
		adj.begin(), adj.end(), XP,
		[&]( VID v ) { return v != adj.invalid_element; } );
	    assert( end - XP == deg );
	    std::sort( XP, XP+deg );
	}
	const VID * const start = std::upper_bound( XP, XP+deg, v );
	VID ne = start - XP;
	VID ce = deg;

	// Following method may modify XP contents, hence need our own copy.
	MCE_Enumerator_stage2 Ee( E, 1 );
	mce_iterate_xp_iterative( G, Ee, alloc, degeneracy, XP, ne, ce );

	if( ce > 0 )
	    alloc.deallocate_to( XP );
    }
}

void
mce_bron_kerbosch_recpar_xp2(
    const HGraphTy & G,
    VID degeneracy,
    MCE_Enumerator_stage2 & E,
    const VID * XP,
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
    const auto pp = mc_get_pivot_xp( G, XP, ne, ce );
    const VID pivot = pp.first;
    const VID sum = pp.second;

    const auto & padj = G.get_adjacency( pivot );

    // Pre-sort, affects vertex selection order in recursive calls
    const VID * XP_piv = XP;
    VID pe = ce - sum; // neighbours of pivot moved to end.
    if( sum > 0 ) {
	VID * XP_prep = new VID[ce];
	VID P_ins = pe;
	std::copy( XP, XP+ne, XP_prep );
	VID ne_i = ne;
	for( VID i=ne; i < ce; ++i ) {
	    if( padj.contains( XP[i] ) )
		XP_prep[P_ins++] = XP[i];
	    else
		XP_prep[ne_i++] = XP[i];
	}
	assert( P_ins == ce );
	assert( ne_i == pe );
	XP_piv = XP_prep;
    }

    parallel_loop( ne, pe, 1, [&]( VID i ) { // TODO: [=] ?
	VID v = XP_piv[i];

	// Not keeping track of R

	const auto & adj = G.get_adjacency( v ); 
	VID deg = adj.size();
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

	    // Now work with modified XP. Could consider intersect in-place.
	    VID * XP_new = new VID[std::min(ce,deg)];
	    VID ne_new = graptor::hash_vector::intersect(
		XP_piv, XP_piv+i, adj, XP_new ) - XP_new;
	    VID ce_new = graptor::hash_vector::intersect(
		XP_piv+i, XP_piv+ce, adj, XP_new+ne_new ) - XP_new;

	    if( ce_new - ne_new == 0 ) {
		// Reached leaf of search tree
		if( ne_new == 0 )
		    E.record( depth+1 );
	    } else if( ce_new - ne_new >= TUNABLE_SMALL_AVOID_CUTOUT
		       // very small -> just keep going
		       && ( ce_new - ne_new <= (1<<P_MAX_SIZE)
			    // on way to a clique with deep recursion?
			    || ( ne_new == 0 && ( ce - ce_new < ce_new / 100 ) )
			   ) ) {
		// direct cut-out of dense graph
		bool ok = mce_leaf<VID,EID>(
		    G, E, depth+1, XP_new, ne_new, ce_new );
		if( !ok ) {
		    // Restore sort order of P set to support merge intersect.
		    // P was disrupted by sorting pivot neighbours to the end.
		    // TODO: is P a merge of two sorted sub-arrays?
		    // Restore sort order of X set. This may be disrupted as
		    // vertices are added out of order, specifically
		    // pivot neighbours.
		    if constexpr ( HGraphTy::has_dual_rep ) {
			std::sort( XP_new, XP_new+ne_new );
			std::sort( XP_new+ne_new, XP_new+ce_new );
		    }
		
		    GraphBuilderInduced<HGraphTy>
			builder( G, XP_new, ne_new, ce_new );
		    const auto & Gc = builder.get_graph();
		    StackLikeAllocator alloc;
		    MCE_Enumerator_stage2 E2( E, depth+1 );
		    std::iota( XP_new, XP_new+ce_new, 0 );
		    mce_iterate_xp_iterative( Gc, E2, alloc, degeneracy,
					      XP_new, ne_new, ce_new );
		}
	    } else {
		// large sub-problem; search recursively
		mce_bron_kerbosch_recpar_xp2(
		    G, degeneracy, E, XP_new, ne_new, ce_new, depth+1 );
	    }

	    delete[] XP_new;
	}
    } );

    if( sum > 0 )
	delete[] XP_piv;
}


void
mce_bron_kerbosch_recpar_xp2_top(
    const HGraphTy & G,
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
	const VID * XP;
	if constexpr ( HGraphTy::has_dual_rep ) {
	    XP = G.get_neighbours( v );
	} else {
	    VID * XP_prep = alloc.template allocate<VID>( deg );
	    auto end = std::copy_if(
		adj.begin(), adj.end(), XP_prep,
		[&]( VID v ) { return v != adj.invalid_element; } );
	    assert( end - XP_prep == deg );
	    std::sort( XP_prep, XP_prep+deg );
	    XP = XP_prep;
	}
	const VID * const start = std::upper_bound( XP, XP+deg, v );
	VID ne = start - XP;
	VID ce = deg;

	mce_bron_kerbosch_recpar_xp2( G, degeneracy, E, XP, ne, ce, 1 );
    } );
}

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


void check_clique_edges( EID m, const VID * assigned_clique, EID ce ) {
    EID cce = 0;
    for( EID e=0; e != m; ++e )
	if( ~assigned_clique[e] != 0 )
	    ++cce;
    assert( cce == ce );
}

class TimeLimitedExecution {
    struct thread_info {
	timeval m_expired_time;
	volatile bool m_termination_flag;
	bool m_active;
	std::mutex m_lock;
    };

public:
    static TimeLimitedExecution & getInstance() {
	// Guaranteed to be destroyed and instantiated on first use.
	static TimeLimitedExecution instance;
	return instance;
    }
private:
    TimeLimitedExecution() : m_terminated( false ), m_thread( guard_thread ) {
	// install_signal_handler();
	// set_timer();
    }
    ~TimeLimitedExecution() {
	m_terminated = true; // causes guard_thread to terminate
	m_thread.join(); // wait until it has terminated
	// clear_timer();
	// remove_signal_handler();
    }

public:
    TimeLimitedExecution( TimeLimitedExecution const& ) = delete;
    void operator = ( TimeLimitedExecution const& )  = delete;

public:
    template<typename Fn, typename... Args>
    static auto execute_time_limited( uint64_t usec, Fn && fn, Args && ... args ) {
	// The singleton object
	TimeLimitedExecution & tlexec = getInstance();
	
	// Who am I?
	pthread_t self = pthread_self();

	// Look up my record
	thread_info & ti = tlexec.m_thread_info[self];

	// Check current time and calculate expiry time
	if( gettimeofday( &ti.m_expired_time, NULL ) < 0 ) {
	    std::cerr << "Error getting current time: "
		      << strerror( errno ) << "\n";
	    exit( 1 );
	}

	// Lock the record
	{
	    std::lock_guard<std::mutex> g( ti.m_lock );
	    uint64_t mln = 1000000ull;
	    ti.m_expired_time.tv_sec += usec / mln;
	    ti.m_expired_time.tv_usec += usec % mln;
	    if( ti.m_expired_time.tv_usec >= mln ) {
		ti.m_expired_time.tv_sec
		    += ti.m_expired_time.tv_usec / mln;
		ti.m_expired_time.tv_usec
		    = ti.m_expired_time.tv_usec % mln;
	    }

	    // Set active
	    ti.m_termination_flag = false;
	    ti.m_active = true;
	} // releases lock

	// std::cerr << "set to expire at sec=" << ti.m_expired_time.tv_sec
	// << " usec=" << ti.m_expired_time.tv_usec << "\n";

	decltype( fn( &ti.m_termination_flag, args... ) ) ret;
	try {
	    ret = fn( &ti.m_termination_flag, args... );
	} catch( const timeout_exception & e ) {
	    // std::cerr << "reached timeout; invalid result\n";

	    // Disable - no need to lock
	    ti.m_active = false;

	    // Rethrow exception
	    throw timeout_exception( usec );
	}
	
	// Disable - no need to lock
	ti.m_active = false;

	return ret;
    }

private:
    static void guard_thread() {
	getInstance().process_loop();
    }
    static void alarm_signal_handler( int ) {
	getInstance().process_periodically();
    }
    
    void process_loop() {
	while( !m_terminated ) {
	    std::this_thread::sleep_for( 10us );
	    process_periodically();
	}
    }
    
    void process_periodically() {
	// Lock map
	std::lock_guard<std::mutex> g( m_lock );
	
	// Get gurrent time
	timeval now;
	if( gettimeofday( &now, NULL ) < 0 ) {
	    std::cerr << "Error getting current time: "
		      << strerror( errno ) << "\n";
	    exit( 1 );
	}
	
	for( auto & tip : m_thread_info ) {
	    thread_info & ti = tip.second;

	    // Avoid deadlock in case we are manipulating the record in the
	    // same thread that executes the signal handler. If the record
	    // is being manipulated, then the computation is not in progress
	    // and need not be interrupted.
	    if( ti.m_lock.try_lock() ) {
		std::lock_guard<std::mutex> g( ti.m_lock, std::adopt_lock );
		if( !ti.m_active )
		    continue;
		if( ti.m_expired_time.tv_sec < now.tv_sec
		    || ( ti.m_expired_time.tv_sec == now.tv_sec
			 && ti.m_expired_time.tv_usec < now.tv_usec ) ) {
		    ti.m_termination_flag = true;

		    /*
		    std::cerr << "set to expire at sec=" << ti.m_expired_time.tv_sec
			      << " usec=" << ti.m_expired_time.tv_usec << "\n";
		    std::cerr << "triggering at sec=" << now.tv_sec
			      << " usec=" << now.tv_usec << "\n";
		    */
		}
	    }
	}
    }

    void install_signal_handler() {
	struct sigaction act;

	act.sa_handler = alarm_signal_handler;
	act.sa_flags = 0;
	
	int ret = sigaction( SIGALRM, &act, NULL );
	if( ret < 0 ) {
	    std::cerr << "Error setting signal handler: "
		      << strerror( errno ) << "\n";
	    exit( 1 );
	}
    }

    void remove_signal_handler() {
	struct sigaction act;

	act.sa_handler = SIG_DFL;
	act.sa_flags = 0;
	
	int ret = sigaction( SIGALRM, &act, NULL );
	if( ret < 0 ) {
	    std::cerr << "Error removing signal handler: "
		      << strerror( errno ) << "\n";
	    exit( 1 );
	}
    }

    void set_timer() {
	struct itimerval when;

	when.it_interval.tv_sec = 0;
	when.it_interval.tv_usec = 100000;
	when.it_value.tv_sec = 0;
	when.it_value.tv_usec = 100000;

	int ret = setitimer( ITIMER_REAL, &when, NULL );
	if( ret < 0 ) {
	    std::cerr << "Error setting timer: "
		      << strerror( errno ) << "\n";
	    exit( 1 );
	}

	ret = getitimer( ITIMER_REAL, &when );
    }

    void clear_timer() {
	struct itimerval when;

	when.it_interval.tv_sec = 0;
	when.it_interval.tv_usec = 0;
	when.it_value.tv_sec = 0;
	when.it_value.tv_usec = 0;

	int ret = setitimer( ITIMER_REAL, &when, NULL );
	if( ret < 0 ) {
	    std::cerr << "Error clearing timer: "
		      << strerror( errno ) << "\n";
	    exit( 1 );
	}
    }

private:
    volatile bool m_terminated;
    std::mutex m_lock;
    std::thread m_thread;
    std::map<pthread_t,thread_info> m_thread_info;
};

template<typename Fn, typename... Args>
auto execute_time_limited( uint64_t usec, Fn && fn, Args && ... args ) {
    return TimeLimitedExecution::execute_time_limited( usec, fn, args... );
}

template<typename... Fn>
class AlternativeSelector {
    static constexpr size_t num_fns = sizeof...( Fn );
    
public:
    AlternativeSelector( Fn && ... fn )
	: m_fn( std::forward<Fn>( fn )... ) {
	std::fill( &m_success[0], &m_success[num_fns], 0 );
	std::fill( &m_fail[0], &m_fail[num_fns], 0 );
	std::fill( &m_best[0], &m_best[num_fns], 0 );
	std::fill( &m_success_time_total[0], &m_success_time_total[num_fns], 0 );
	std::fill( &m_success_time_max[0], &m_success_time_max[num_fns], 0 );
	std::fill( &m_best_time_total[0], &m_best_time_total[num_fns], 0 );
    }
    ~AlternativeSelector() {
	report( std::cerr );
    }

    template<typename... Args>
    auto execute( uint64_t base_usec, Args && ... args ) {
#if 1
	using return_type = decltype( std::get<0>( m_fn )( 0ull, args... ) );
	return_type ret;

	for( uint64_t rep=1; rep <= 24; ++rep ) {
	    uint64_t usec = base_usec << rep;
	    try {
		return attempt_fn<0>( usec, std::forward<Args>( args )... );
	    } catch( timeout_exception & e ) {
	    }
	}

	// None of the alternatives completed in time limit
	abort();
#else
	try {
	    // uint64_t usec = 800000000ull; // 800sec
	    uint64_t usec = 50000000ull << 13; // 50sec
	    return attempt_all_fn( usec, std::forward<Args>( args )... );
	} catch( timeout_exception & e ) {
	    std::cerr << "timeout: usec=" << e.usec() << " idx=" << e.idx() << "\n";
	    throw;
	}
#endif
    }

    std::ostream & report( std::ostream & os ) {
	os << "Success of alternatives (#=" << num_fns << "):\n";
	for( size_t i=0; i < num_fns; ++i ) {
	    os << "alternative " << i
	       << ": success=" << m_success[i]
	       << " avg-success-tm="
	       << ( m_success_time_total[i] / double(m_success[i]) )
	       << " max-success-tm=" << m_success_time_max[i] 
	       << " fail=" << m_fail[i]
	       << " best=" << m_best[i]
	       << " avg-best-time="
	       << ( m_best_time_total[i] / double(m_best[i]) )
	       << "\n";
	}
	return os;
    }

private:
    template<size_t idx, typename... Args>
    auto attempt_fn( uint64_t usec, Args && ... args ) {
	using return_type = decltype( std::get<0>( m_fn )( 0ull, args... ) );
	return_type ret;

	timer tm;
	tm.start();

	try {
	    ret = execute_time_limited(
		usec, std::get<idx>( m_fn ), std::forward<Args>( args )... );
	    auto dly = tm.stop();
	    std::cerr << "   alt #" << idx << " succeeded after "
		      << dly << "\n";
	    m_success_time_total[idx] += dly;
	    m_success_time_max[idx] = std::max( m_success_time_max[idx], dly );
	    ++m_success[idx];
	    return ret;
	} catch( timeout_exception & e ) {
	    std::cerr << "   alt #" << idx << " failed after "
		      <<  tm.stop() << "\n";
	    ++m_fail[idx];
	    if constexpr ( idx >= num_fns-1 )
		throw timeout_exception( usec, idx );
	    else
		return attempt_fn<idx+1>( usec, std::forward<Args>( args )... );
	}
    }

    template<typename... Args>
    auto attempt_all_fn( uint64_t usec, Args && ... args ) {
	std::array<double,num_fns> tms = { std::numeric_limits<double>::max() };
	using return_type = decltype( std::get<0>( m_fn )( 0ull, args... ) );
	return_type ret;
	bool repeat = true;

	while( repeat ) {
	    try {
		ret = attempt_all_fn_aux<0>(
		    usec, tms, std::forward<Args>( args )... );
		repeat = false;
	    } catch( timeout_exception & e ) {
		usec *= 2;
		std::cerr << "timeout on all variants; doubling time to "
			  << usec << "\n";
	    }
	}

	for( size_t idx=0; idx < num_fns; ++idx ) {
	    double dly = tms[idx];
	    if( dly != std::numeric_limits<double>::max() ) {
		m_success_time_total[idx] += dly;
		m_success_time_max[idx] = std::max( m_success_time_max[idx], dly );
		++m_success[idx];
	    } else
		++m_fail[idx];
	}

	size_t best = std::distance(
	    tms.begin(), std::min_element( tms.begin(), tms.end() ) );
	++m_best[best];
	m_best_time_total[best] += tms[best];

	return ret;
    }

    template<typename Arg0, typename... Args>
    void check_clique( size_t size, VID * clique,
		       Arg0 && arg0, Args && ... args ) {
	::check_clique( std::forward<Arg0>( arg0 ), size, clique );
    }
    
    template<size_t idx, typename... Args>
	auto attempt_all_fn_aux( uint64_t usec, std::array<double,num_fns> & tms, Args && ... args ) {
	using return_type = decltype( std::get<0>( m_fn )( 0ull, args... ) );
	return_type ret;

	timer tm;
	tm.start();

	try {
	    // TODO: pass in ret as argument and use any contents filled in
	    //       even in case of timeout.
	    if( verbose )
		std::cerr << "as: alternative " << idx
			  << " timeout " << usec << "\n";
	    ret = execute_time_limited(
		usec, std::get<idx>( m_fn ), std::forward<Args>( args )... );
	    tms[idx] = tm.stop();
	    // check_clique( ret.size(), &ret[0], std::forward<Args>( args )... );

	    if constexpr ( idx+1 < num_fns ) {
		try {
		    auto r = attempt_all_fn_aux<idx+1>(
			usec, tms, std::forward<Args>( args )... );
		    assert( is_equal( ret, r ) );
		} catch( timeout_exception & e ) {
		    return ret;
		}
	    }

	    return ret;
	} catch( timeout_exception & e ) {
	    tms[idx] = std::numeric_limits<double>::max();
	    if constexpr ( idx+1 < num_fns )
		return attempt_all_fn_aux<idx+1>(
		    usec, tms, std::forward<Args>( args )... );
	    else
		throw timeout_exception( usec, idx );
	}
    }

    template<typename T>
    static bool is_equal( const T & a, const T & b ) {
	return true;
    }
    static bool is_equal( bool a, bool b ) {
	return a == b;
    }
    static bool
    is_equal( const std::vector<VID> & a, const std::vector<VID> & b ) {
	return a.size() == b.size();
    }

private:
    std::tuple<Fn...> m_fn;
    size_t m_success[num_fns];
    size_t m_fail[num_fns];
    size_t m_best[num_fns];
    double m_success_time_total[num_fns];
    double m_success_time_max[num_fns];
    double m_best_time_total[num_fns];
};

template<typename... Fn>
auto make_alternative_selector( Fn && ... fn ) {
    return AlternativeSelector<Fn...>( std::forward<Fn>( fn )... );
}


static std::mutex io_mux;

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
    // TODO: Make merge method depend on size of neighbour lists
    BlockedBinaryMatrix<XBits,PBits,VID,EID>
	// IG( G, H, v, cut, graptor::hash_scalar() );
	// IG( G, H, v, cut, graptor::merge_scalar() );
	IG( G, H, v, cut, graptor::hash_vector() );

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

    // Make merge method depend on size of neighbour lists
    // merge_scalar is fastest for pokec, but not for orkut
    if( num <= 8 ) {
	// Build induced graph
	DenseMatrix<Bits,VID,EID> IG( G, H, v, cut, graptor::merge_scalar() );

	MCE_Enumerator_stage2 E2( E );
	IG.mce_bron_kerbosch( E2 );
    } else {
	// Build induced graph
	DenseMatrix<Bits,VID,EID> IG( G, H, v, cut, graptor::hash_vector() );

	MCE_Enumerator_stage2 E2( E );
	IG.mce_bron_kerbosch( E2 );
    }

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

void mce_variable(
    const HGraphTy & HG,
    MCE_Enumerator & E,
    VID v,
    VID start_pos,
    VID degeneracy
    ) {
    MCE_Enumerator_stage2 Ee( E, 1 ); // 1 for top-level vertex re: cutout
    VID n = HG.numVertices();
    if( n - start_pos >= (1<<P_MAX_SIZE) )
	mce_bron_kerbosch_recpar_xp2_top( HG, start_pos, degeneracy, Ee );
    else
	mce_bron_kerbosch_seq_xp( HG, start_pos, degeneracy, Ee );
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
    MCE_Enumerator & E,
    VID v,
    VID degeneracy ) {
    graptor::graph::NeighbourCutOutDegeneracyOrder<VID,EID> cut( G, v );

    all_variant_statistics & stats = mce_stats.get_statistics();

    VID num = cut.get_num_vertices();

    if( num <= 3 ) {
	timer tm;
	tm.start();
	MCE_Enumerator_stage2 E2( E, 0 );
	mce_tiny( H, cut.get_vertices(), cut.get_start_pos(),
			 cut.get_num_vertices(), E2 );
	stats.record_tiny( tm.stop() );
	return;
    }

    // Threshold is tunable and depends on cost of creating a cut-out vs the
    // cost of merge and hash intersections.
    if( num <= TUNABLE_SMALL_AVOID_CUTOUT ) {
	timer tm;
	tm.start();
	MCE_Enumerator_stage2 E2( E, 1 );
	StackLikeAllocator alloc;
	VID ce = cut.get_num_vertices();
	VID * XP = alloc.template allocate<VID>( ce );
	const VID * ngh = cut.get_vertices();
	std::copy( ngh, ngh+ce, XP );
	VID ne = cut.get_start_pos();
	mce_iterate_xp_iterative<false>( H, E2, alloc, degeneracy, XP, ne, ce );
	stats.record_tiny( tm.stop() );
	return;
    }
    
    VID xnum = cut.get_start_pos();
    VID pnum = num - xnum;

    VID nlg = get_size_class( num );
    if( nlg < N_MIN_SIZE )
	nlg = N_MIN_SIZE;

    if( nlg <= N_MIN_SIZE+1 ) { // up to 64 bits
	return mce_dense_func[nlg-N_MIN_SIZE](
	    G, H, E, v, cut, stats.get( nlg ) );
    }

    VID xlg = get_size_class( xnum );
    if( xlg < X_MIN_SIZE )
	xlg = X_MIN_SIZE;

    VID plg = get_size_class( pnum );
    if( plg < P_MIN_SIZE )
	plg = P_MIN_SIZE;

    if( nlg <= xlg + plg && nlg <= N_MAX_SIZE ) {
	return mce_dense_func[nlg-N_MIN_SIZE](
	    G, H, E, v, cut, stats.get( nlg ) );
    }

    if( xlg <= X_MAX_SIZE && plg <= P_MAX_SIZE ) {
	return mce_blocked_func[xlg-X_MIN_SIZE][plg-P_MIN_SIZE](
	    G, H, E, v, cut, stats.get( xlg, plg ) );
    }

    if( nlg <= N_MAX_SIZE ) {
	return mce_dense_func[nlg-N_MIN_SIZE](
	    G, H, E, v, cut, stats.get( nlg ) );
    }

    timer tm;
    tm.start();
    GraphBuilderInduced<HGraphTy> ibuilder( G, H, v, cut );
    const auto & HG = ibuilder.get_graph();

    stats.record_genbuild( tm.stop() );

    tm.start();
    mce_variable( HG, E, v, ibuilder.get_start_pos(), degeneracy );
    double t = tm.stop();
    if( false && t >= 3.0 ) {
	std::cerr << "generic v=" << v << " num=" << num
		  << " xnum=" << xnum << " pnum=" << pnum
	    // << " density=" << HG.density()
	    // << " m=" << HG.numEdges()
		  << " t=" << t
		  << "\n";
    }
    stats.record_gen( t );
}

template<unsigned Bits, typename VID, typename EID>
void leaf_dense_fn(
    const HGraphTy & H,
    MCE_Enumerator_stage2 & Ee,
    VID r,
    const VID * XP,
    VID ne,
    VID ce ) {
    DenseMatrix<Bits,VID,EID> D( H, XP, ne, ce );
    D.mce_bron_kerbosch( [&]( const bitset<Bits> & c, size_t sz ) {
	Ee.record( r + sz );
    } );
}

template<unsigned XBits, unsigned PBits, typename VID, typename EID>
void leaf_blocked_fn(
    const HGraphTy & H,
    MCE_Enumerator_stage2 & Ee,
    VID r,
    const VID * XP,
    VID ne,
    VID ce ) {
    BlockedBinaryMatrix<XBits,PBits,VID,EID> D( H, XP, ne, ce );
    mce_bron_kerbosch( D, [&]( const bitset<PBits> & c, size_t sz ) {
	Ee.record( r + sz );
    } );
}

typedef void (*mce_leaf_func)(
    const HGraphTy &,
    MCE_Enumerator_stage2 & Ee,
    VID,
    const VID *,
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
    const VID * XP,
    VID ne,
    VID ce ) {
    VID num = ce;
    VID xnum = ne;
    VID pnum = ce - ne;

    if( ce <= 3 ) {
	MCE_Enumerator_stage2 E2( E, r-1 );
	mce_tiny( H, XP, ne, ce, E2 );
	return true;
    }

    VID nlg = get_size_class( num );
    if( nlg < N_MIN_SIZE )
	nlg = N_MIN_SIZE;

    if( nlg <= N_MIN_SIZE+1 ) { // up to 64 bits
	leaf_dense_func[nlg-N_MIN_SIZE]( H, E, r, XP, ne, ce );
	return true;
    }

    VID xlg = get_size_class( xnum );
    if( xlg < X_MIN_SIZE )
	xlg = X_MIN_SIZE;

    VID plg = get_size_class( pnum );
    if( plg < P_MIN_SIZE )
	plg = P_MIN_SIZE;

/*
    if( nlg <= xlg + plg && nlg <= N_MAX_SIZE ) {
	leaf_dense_func[nlg-N_MIN_SIZE]( H, E, r, XP, ne, ce );
	return true;
    }
*/

    if( xlg <= X_MAX_SIZE && plg <= P_MAX_SIZE ) {
	leaf_blocked_func[xlg-X_MIN_SIZE][plg-P_MIN_SIZE](
	    H, E, r, XP, ne, ce );
	return true;
    }

    if( nlg <= N_MAX_SIZE ) {
	leaf_dense_func[nlg-N_MIN_SIZE]( H, E, r, XP, ne, ce );
	return true;
    }

    return false;
}

int main( int argc, char *argv[] ) {
    commandLine P( argc, argv, " help" );
    bool symmetric = P.getOptionValue("-s");
    verbose = P.getOptionValue("-v");
    const char * ifile = P.getOptionValue( "-i" );

    timer tm;
    tm.start();

    GraphCSx G0( ifile, -1, symmetric );

    std::cerr << "Reading graph: " << tm.next() << "\n";

    GraphCSx G = graptor::graph::remove_self_edges( G0, true );
    G0.del();
    std::cerr << "Removed self-edges: " << tm.next() << "\n";

    VID n = G.numVertices();
    EID m = G.numEdges();

    assert( G.isSymmetric() );
    double density = double(m) / ( double(n) * double(n) );
    VID dmax_v = G.findHighestDegreeVertex();
    VID dmax = G.getDegree( max_v );
    double davg = (double)m / (double)n;
    std::cerr << "Undirected graph: n=" << n << " m=" << m
	      << " density=" << density
	      << " dmax=" << dmax
	      << " davg=" << davg
	      << std::endl;

    GraphCSRAdaptor GA( G, 256 );
    KCv<GraphCSRAdaptor> kcore( GA, P );
    kcore.run();
    auto & coreness = kcore.getCoreness();
    std::cerr << "Calculating coreness: " << tm.next() << "\n";
    std::cerr << "coreness=" << kcore.getLargestCore() << "\n";

    mm::buffer<VID> order( n, numa_allocation_interleaved() );
    mm::buffer<VID> rev_order( n, numa_allocation_interleaved() );
    sort_order( order.get(), rev_order.get(),
		coreness.get_ptr(), n, kcore.getLargestCore() );
    global_coreness = coreness.get_ptr();
    std::cerr << "Determining sort order: " << tm.next() << "\n";

    GraphCSx R( G, std::make_pair( order.get(), rev_order.get() ) );
    std::cerr << "Remapping graph: " << tm.next() << "\n";

    // graptor::graph::GraphHAdjTable<VID,EID,hash_fn> H( R );
    // graptor::graph::GraphHAdj<VID,EID,GraphCSx,hash_fn>
	// H( R, true, numa_allocation_interleaved() );
    HFGraphTy H( R, numa_allocation_interleaved() );
    std::cerr << "Building hashed graph: " << tm.next() << "\n";

    MCE_Enumerator E( kcore.getLargestCore() );

    system( "date" );
    std::cerr << "Start enumeration: " << tm.next() << "\n";

    // Number of partitions is tunable. A fairly large number is helpful
    // to help load balancing.
    VID nthreads = graptor_num_threads();
    VID npart = nthreads * 128;
    parallel_loop( VID(0), npart, 1, [&]( VID p ) {
	for( VID i=p; i < n; i += npart ) {
	    VID v = order[i];
	    mce_top_level( R, H, E, v, kcore.getLargestCore() );
	}
    } );

    std::cerr << "Enumeration: " << tm.next() << "\n";

    all_variant_statistics stats = mce_stats.sum();

    double duration = tm.total();
    std::cerr << "Completed MCE in " << duration << " seconds\n";
    for( size_t n=N_MIN_SIZE; n <= N_MAX_SIZE; ++n ) {
	std::cerr << (1<<n) << "-bit dense: ";
	stats.get( n ).print( std::cerr ); 
    }
    for( size_t x=X_MIN_SIZE; x <= X_MAX_SIZE; ++x )
	for( size_t p=P_MIN_SIZE; p <= P_MAX_SIZE; ++p ) {
	    std::cerr << (1<<x) << ',' << (1<<p) << "-bit blocked: ";
	    stats.get( x, p ).print( std::cerr );
	}
    std::cerr << "tiny: ";
    stats.m_tiny.print( std::cerr );
    std::cerr << "generic: ";
    stats.m_gen.print( std::cerr );
    std::cerr << "generic build: ";
    stats.m_genbuild.print( std::cerr );

    E.report( std::cerr );

    rev_order.del();
    order.del();
    G.del();

    return 0;
}
