// -*- c++ -*-
// Specialised to MC

// TODO:
// * Add TERM signal handler to flush all output after a timeout during
//   measurements.

#ifndef ABLATION_HADJPA_DISABLE_XP_HASH
#define ABLATION_HADJPA_DISABLE_XP_HASH 0
#endif

#ifndef ABLATION_DENSE_DISABLE_XP_HASH
#define ABLATION_DENSE_DISABLE_XP_HASH 0
#endif

#ifndef ABLATION_BITCONSTRUCT_XP_VEC
#define ABLATION_BITCONSTRUCT_XP_VEC 0
#endif

#ifndef ABLATION_DISABLE_LEAF
#define ABLATION_DISABLE_LEAF 0
#endif

#ifndef ABLATION_DISABLE_TOP_DENSE
#define ABLATION_DISABLE_TOP_DENSE 0
#endif

#ifndef ABLATION_GROW_CLIQUE
#define ABLATION_GROW_CLIQUE 0
#endif

// Not effective, so disable by default
#ifndef ABLATION_PDEG
#define ABLATION_PDEG 1
#endif

#ifndef ABLATION_DENSE_NO_PIVOT_TOP
#define ABLATION_DENSE_NO_PIVOT_TOP 0
#endif

#ifndef ABLATION_DENSE_PIVOT_FILTER
#define ABLATION_DENSE_PIVOT_FILTER 0
#endif

#ifndef ABLATION_FILTER_STEPS
#define ABLATION_FILTER_STEPS 3
#endif 

#ifndef ABLATION_DISABLE_CONNECTED_FILTERING
#define ABLATION_DISABLE_CONNECTED_FILTERING 1
#endif 

#ifndef BK_MIN_LEAF
#define BK_MIN_LEAF 8
#endif

#ifndef TOP_DENSE_SELECT
#define TOP_DENSE_SELECT 0
#endif

#ifndef ABLATION_DISABLE_VC
#define ABLATION_DISABLE_VC 0
#endif

#ifndef ABLATION_DISABLE_BK
#define ABLATION_DISABLE_BK 0
#endif

#ifndef PIVOT_COLOUR
#define PIVOT_COLOUR 0
#endif

#ifndef PIVOT_COLOUR_DENSE
#define PIVOT_COLOUR_DENSE 0
#endif

#ifndef CLIQUER_PRUNE
#define CLIQUER_PRUNE 0
#endif

#ifdef FILTER_NEIGHBOUR_CLIQUE
#undef FILTER_NEIGHBOUR_CLIQUE
#endif
#define FILTER_NEIGHBOUR_CLIQUE 0

#if CLIQUER_PRUNE || FILTER_NEIGHBOUR_CLIQUE
#define MEMOIZE_MC_PER_VERTEX 1
#else
#define MEMOIZE_MC_PER_VERTEX 0
#endif

#ifndef VERTEX_COVER_COMPONENTS
#define VERTEX_COVER_COMPONENTS 0
#endif

#ifndef VERTEX_COVER_ABSOLUTE
#define VERTEX_COVER_ABSOLUTE 0
#endif

/*!
 * Profile impact of various incumbent sizes on enumeration time.
 *
 * Values:
 * 0: disabled
 * 1: measure bron-kerbosch-style enumeration (leafs may use vertex cover)
 * 2: measure vertex cover
 */
#ifndef PROFILE_INCUMBENT_SIZE
#define PROFILE_INCUMBENT_SIZE 0
#endif

#if PROFILE_INCUMBENT_SIZE == 1
#undef ABLATION_DISABLE_VC 
#define ABLATION_DISABLE_VC 1
#undef ABLATION_GROW_CLIQUE
#define ABLATION_GROW_CLIQUE 1
#elif PROFILE_INCUMBENT_SIZE == 2
#undef ABLATION_DISABLE_BK 
#define ABLATION_DISABLE_BK 1
#undef ABLATION_GROW_CLIQUE
#define ABLATION_GROW_CLIQUE 1
#endif

#ifndef PROFILE_DENSITY
#define PROFILE_DENSITY 0
#endif

#ifndef USE_512_VECTOR
#if __AVX512F__
#define USE_512_VECTOR 1
#else
#define USE_512_VECTOR 1
#endif
#endif

/* (SORT,TRAVERSAL) execution time findings (<: clearly faster;
 * ~< sometimes marginally faster):
 * (4,1) ~< (4,3) < (5,1)
 * Preferred on basis of completer measurement: (4,3)
 */
#ifndef SORT_ORDER
#define SORT_ORDER 4
#endif

#ifndef TRAVERSAL_ORDER
#define TRAVERSAL_ORDER 3
#endif

#ifndef PAPI_REGION
#define PAPI_REGION 0
#endif

#include <ranges>

#include <sched.h>
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
#include "graptor/graph/simple/csxd.h"
#include "graptor/graph/simple/hadj.h"
#include "graptor/graph/simple/hadjt.h"
#include "graptor/graph/simple/cutout.h"
#include "graptor/graph/simple/dense.h"
#include "graptor/graph/simple/xp_set.h"
#include "graptor/graph/transform/rmself.h"

#include "graptor/container/bitset.h"
#include "graptor/container/hash_fn.h"
#include "graptor/container/hash_set_hopscotch.h"
#include "graptor/container/hash_set_hopscotch_delta.h"
#include "graptor/container/intersect.h"
#include "graptor/container/transform_iterator.h"
#include "graptor/container/concatenate_iterator.h"
#include "graptor/stat/welford.h"
#include "graptor/cmdline.h"
#include "reorder_kcore.h"

float density_threshold = 0.9f;

#if TOP_DENSE_SELECT > 0
#include "graptor/stat/vfdt.h"

using timing_attribute = graptor::numeric_float_dense_range<
    float,std::ratio<0,1>,std::ratio<1,1>,8>;
using input_attribute = graptor::numeric_float_dense_range<
    float,std::ratio<0,1>,std::ratio<1,1>,8>;

static std::vector<graptor::binary_vfdt<
		       timing_attribute,
		       input_attribute, // #vertices / 512
		       input_attribute, // k_max / #vertices
		       input_attribute // density
		       >> algo_predictor;
#endif

//! Choice of hash function for compilation unit
// The random hash function was used by Blanusa VLDB 2020 (MCE), however,
// it shows to be inferior for MC especially in the context of hopscotch
// hashing (also used by Blanusa).
// using hash_fn = graptor::rand_hash<uint32_t>;
using hash_fn = graptor::java_hash<uint32_t>;

#ifndef HOPSCOTCH_HASHING
#define HOPSCOTCH_HASHING 1
#endif

#ifndef ABLATION_DISABLE_LAZY_HASHING
#define ABLATION_DISABLE_LAZY_HASHING 0
#endif

#ifndef ABLATION_DISABLE_LAZY_REMAPPING
#define ABLATION_DISABLE_LAZY_REMAPPING 0
#endif

//! Hopscotch hashing is a more efficient hash set than the general
// open-addressed hashing in graptor::hash_set. However, it is hard to
// predict the required size of hash table in advance for hopscotch hashing
// as it depends on how conflicts play out for the particular values used.
// By consequence, we cannot pre-allocate all storage for the full graph
// and need to work with many small allocations.
// One option to mitigate this would be to construct the hash sets on the
// fly, only for those vertices that need them. Post heuristic search, only
// the high-degeneracy vertices would need a hash set representation of their
// neighbourhood. We can postpone the construction of the remapped graph
// in the same way.
#if HOPSCOTCH_HASHING == 1
using hash_set_type = graptor::hash_set_hopscotch<VID,hash_fn>;
static constexpr bool hash_set_prealloc = false;
#if ABLATION_DISABLE_LAZY_HASHING
static constexpr bool lazy_hashing = false;
#else
static constexpr bool lazy_hashing = true;
#endif
#elif HOPSCOTCH_HASHING == 2
using hash_set_type = graptor::hash_set_hopscotch_delta<VID,hash_fn>;
static constexpr bool hash_set_prealloc = false;
#if ABLATION_DISABLE_LAZY_HASHING
static constexpr bool lazy_hashing = false;
#else
static constexpr bool lazy_hashing = true;
#endif
#else
using hash_set_type = graptor::hash_set<VID,hash_fn>;
static constexpr bool hash_set_prealloc = true;
static constexpr bool lazy_hashing = false;
#endif

#if ABLATION_HADJPA_DISABLE_XP_HASH
using HGraphTy = graptor::graph::GraphHAdjPA<VID,EID,false,hash_set_prealloc,false,hash_set_type>;
#else
using HGraphTy = graptor::graph::GraphHAdjPA<VID,EID,true,hash_set_prealloc,false,hash_set_type>;
#endif

#if !ABLATION_DISABLE_LAZY_REMAPPING
using HFGraphTy = graptor::graph::GraphLazyRemappedHashedAdj<VID,EID,hash_set_type>;
#else
using HFGraphTy = graptor::graph::GraphLazyHashedAdj<VID,EID,lazy_hashing,hash_set_type>;
#endif

using graptor::graph::DenseMatrix;
using graptor::graph::PSet;

static constexpr size_t N_MIN_SIZE = 5;
static constexpr size_t N_DIM = 9 - N_MIN_SIZE + 1;

#if USE_512_VECTOR
static constexpr size_t N_MAX_SIZE = 9;
#else
static constexpr size_t N_MAX_SIZE = 8;
#endif

size_t get_size_class( uint32_t v ) {
    size_t b = _lzcnt_u32( v-1 );
    size_t cl = 32 - b;
    assert( v <= (1<<cl) );
    if( cl < N_MIN_SIZE )
	cl = N_MIN_SIZE;
    return cl;
}

enum algo_variant {
    av_bk = 0,
    av_vc = 1,
    N_VARIANTS = 2
};

/*! Verbosity level
 * Higher values provide more verbose output
 */
static int verbose = 0;

static std::mutex io_mux;
static constexpr bool io_trace = false;

enum filter_reason {
    fr_pset = 0,
    fr_colour_ub = 1,
    fr_colour_greedy = 2,
    fr_rdeg = 3,
    fr_maxdeg = 4,
    fr_unknown = 5,
    fr_cover = 6,
    fr_outer = 7,
    fr_cliquer = 8,
    filter_reason_num = 9
};

template<typename T>
class clique_set {
public:
    clique_set( T val, const clique_set * next = nullptr ) :
	m_value( val ), m_next( next ) { }

    class iterator {
    public:
	using type = T;

	// iterator traits
	using iterator_category = std::input_iterator_tag;
	using value_type = T;
	using difference_type = std::intptr_t;
	using pointer = const T *;
	using reference = const T &;

    public:
	explicit iterator( const clique_set * cs ) : m_set( cs ) { }
	iterator& operator++() { m_set = m_set->get_next(); return *this; }
	iterator operator++( int ) {
	    iterator retval = *this; ++(*this); return retval;
	}
	bool operator == ( iterator other ) const {
	    return m_set == other.m_set;
	}
	bool operator != ( iterator other ) const {
	    return !( *this == other );
	}
	const typename iterator::reference operator*() const {
	    return m_set->get_reference();
	}

    private:
	const clique_set * m_set;
    };

    iterator begin() const { return iterator( this ); }
    iterator end() const { return iterator( nullptr ); }

    T get_value() const { return m_value; }
    const T & get_reference() const { return m_value; }
    const clique_set * get_next() const { return m_next; }
    
private:
    const T m_value; //!< value in this link list element
    const clique_set * m_next; //!< previously selected element in clique
};

class MC_Enumerator {
public:
    MC_Enumerator( const GraphCSx & G )
	: m_graph( G ),
	  m_degeneracy( std::numeric_limits<VID>::max()-1 ),
	  m_best( 0 ),
	  m_order( nullptr ) {
	m_timer.start();
    }
    MC_Enumerator(  const GraphCSx & G, size_t degen, const VID * const order,
		    const VID * const coreness = nullptr )
	: m_graph( G ),
	  m_degeneracy( degen ),
	  m_best( degen > 0 ? 1 : 0 ),
	  m_order( order ),
	  m_coreness( coreness ) {
	m_timer.start();
    }

    void rebase( size_t degen, const VID * const order,
		 const VID * const coreness = nullptr ) {
	m_degeneracy = degen;
	m_order = order;
	m_coreness = coreness;
	if( m_best == 0 && degen > 0 )
	    m_best = 1;
    }

    void reset() {
	m_best = m_degeneracy > 0 ? 1 : 0;
	std::lock_guard<std::mutex> guard( m_mutex );
	m_max_clique.clear();
    }

    // Record solution
    template<typename It>
    void record( size_t s, VID top_v, It && begin, It && end ) {
#if PROFILE_INCUMBENT_SIZE == 0
	assert( s <= m_degeneracy+1 );
#endif
	if( s > m_best )
	    update_best( s, top_v, begin, end );
    }

    // Feasability check
    bool is_feasible_bool( bool cond, filter_reason r = fr_unknown ) {
	if( !cond )
	    ++m_reason[(int)r];
	return cond;
    }

    bool is_feasible( size_t upper_bound, filter_reason r = fr_unknown ) {
	return is_feasible_bool( upper_bound > m_best, r );
    }

    size_t get_max_clique_size() const {
	return m_best;
    }

    // Modifies m_max_clique to adjust sort order
    std::ostream & report( std::ostream & os ) {
	std::lock_guard<std::mutex> guard( m_mutex );

	os << "Maximum clique size: " << m_best
	   << " from top-level vertex " << m_max_clique[0]
	   << "\n";

	// Sort clique, except top-level vertex
	if( !m_max_clique.empty() )
	    std::sort( std::next( m_max_clique.begin() ), m_max_clique.end() );

	os << "Maximum clique:";
	for( auto v : m_max_clique )
	    os << ' ' << v;
	os << "\n";

	os << "  filter pset: " << m_reason[(int)fr_pset].load() << "\n";
	os << "  filter colour-ub: "
	   << m_reason[(int)fr_colour_ub].load() << "\n";
	os << "  filter colour-greedy: "
	   << m_reason[(int)fr_colour_greedy].load() << "\n";
	os << "  filter rdeg: " << m_reason[(int)fr_rdeg].load() << "\n";
	os << "  filter maxdeg: " << m_reason[(int)fr_maxdeg].load() << "\n";
	os << "  filter unknown: " << m_reason[(int)fr_unknown].load() << "\n";
	os << "  filter cover: " << m_reason[(int)fr_cover].load() << "\n";
	os << "  filter outer: " << m_reason[(int)fr_outer].load() << "\n";
	os << "  filter cliquer: " << m_reason[(int)fr_cliquer].load() << "\n";

	return os;
    }

    auto sort_and_get_max_clique() {
	if( !m_max_clique.empty() )
	    std::sort( m_max_clique.begin(), m_max_clique.end() );
	return graptor::make_array_slice( m_max_clique );
    }

private:
    template<typename It>
    void update_best( size_t s, VID top_v, It && begin, It && end ) {
	// Map all vertex IDs to original graph IDs
	std::vector<VID> clique;
	clique.reserve( s );
	if( m_order != nullptr ) {
	    clique.push_back( m_order[top_v] );
	    for( It b=begin; b != end; ++b )
		clique.push_back( m_order[*b] );
	} else {
	    clique.push_back( top_v );
	    for( It b=begin; b != end; ++b )
		clique.push_back( *b );
	}

	// Try to extend clique to a larger clique (in case it is not
	// maximal). This is possible as the clique was searched using
	// the right-neighbourhood of top_v. Now we also consider the
	// left-neighbourhood.
	VID rs = s;
#if !ABLATION_GROW_CLIQUE
	if( clique.size() == s )
	    rs = grow_clique( top_v, clique );
#endif
	
	// Acquire mutex to update m_max_clique and m_best atomically.
	std::lock_guard<std::mutex> guard( m_mutex );

	if( rs > m_best )  {
	    m_best = rs;

	    if( verbose > 0 ) {
		std::cout << "max_clique: " << rs << " (found "
			  << s << ") at "
			  << m_timer.elapsed()
			  << " top-level vertex: " << top_v;
		if( m_coreness != nullptr )
		    std::cout << " coreness: " << m_coreness[top_v];
		if( m_order != nullptr )
		    std::cout << " (" << m_order[top_v] << ')';
		std::cout << '\n';
	    }

	    // Copy clique
	    using std::swap;
	    swap( clique, m_max_clique );
	}
    }

    size_t grow_clique( VID top_v, std::vector<VID> & clique ) const {
	VID csz = clique.size();
	std::sort( clique.begin(), clique.end() );

	auto top_adj = m_graph.get_neighbours_set( top_v );
	std::vector<VID> ins;
	ins.reserve( top_adj.size() + 16 );
	ins.insert( ins.end(), top_adj.begin(), top_adj.end() );
	std::vector<VID> reconstruct( top_adj.size()+16 );
	
	for( VID v : clique ) {
	    auto ins_slice = graptor::make_array_slice( ins );
	    VID * start = &*reconstruct.begin();
	    VID * end = graptor::set_operations<graptor::MC_intersect>::
		intersect_ds( ins_slice, m_graph.get_neighbours_set( v ),
			      start );
	    size_t sz = end - start;
	    reconstruct.resize( sz );
	    ins.resize( sz );
	    std::swap( ins, reconstruct );
	}

	if( ins.size() > 0 ) {
	    clique.reserve( csz + ins.size() );
	    clique.insert( clique.end(), ins.begin(), ins.end() );
	}

	return clique.size();
    }

private:
    const GraphCSx & m_graph;
    size_t m_degeneracy;
    size_t m_best;
    std::array<std::atomic<size_t>,filter_reason_num> m_reason;
    timer m_timer;
    const VID * m_order; //!< to translate IDs back to input file IDs
    const VID * m_coreness;
    std::mutex m_mutex;
    std::vector<VID> m_max_clique; //!< max clique contents, not thread-safe
};

class MC_CutOutEnumerator {
public:
    MC_CutOutEnumerator( MC_Enumerator & E, VID top_v, const VID * const order )
	: m_E( E ), m_top_vertex( top_v ), m_order( order ) { }

    // Record solution
    template<typename It>
    void record( size_t s, It && begin, It && end ) {
	auto fn = [&]( VID v ) { return m_order != nullptr ? m_order[v] : v; };
	m_E.record( s, m_top_vertex,
		    graptor::make_transform_iterator( begin, fn ),
		    graptor::make_transform_iterator( end, fn ) );
    }

    // Feasability check
    bool is_feasible( size_t upper_bound, filter_reason r = fr_unknown ) {
	return m_E.is_feasible( upper_bound, r );
    }
    bool is_feasible_bool( bool cond, filter_reason r = fr_unknown ) {
	return m_E.is_feasible_bool( cond, r );
    }

    size_t get_max_clique_size() const {
	return m_E.get_max_clique_size();
    }

private:
    MC_Enumerator & m_E;
    VID m_top_vertex;
    const VID * const m_order;
};


class MC_DenseEnumerator {
public:
    MC_DenseEnumerator( MC_CutOutEnumerator & E,
			const clique_set<VID> * R,
			const VID * const remap = nullptr )
	: m_E( E ), m_R( R ), m_remap( remap ) { }

    // Record solution
    template<typename It>
    void record( size_t s, It && begin, It && end ) {
	if( m_R != nullptr ) {
	    if( m_remap ) {
		auto fn = [&]( VID v ) { return m_remap[v]; };
		auto b = graptor::make_transform_iterator( begin, fn );
		auto e = graptor::make_transform_iterator( end, fn );
		m_E.record(
		    s,
		    graptor::make_concatenate_iterator( b, e, m_R->begin() ),
		    graptor::make_concatenate_iterator( e, e, m_R->end() ) );
	    } else {
		m_E.record(
		    s,
		    graptor::make_concatenate_iterator( begin, end, m_R->begin() ),
		    graptor::make_concatenate_iterator( end, end, m_R->end() ) );
	    }
	} else {
	    if( m_remap ) {
		auto fn = [&]( VID v ) { return m_remap[v]; };
		auto b = graptor::make_transform_iterator( begin, fn );
		auto e = graptor::make_transform_iterator( end, fn );
		m_E.record( s, b, e );
	    } else {
		m_E.record( s, begin, end );
	    }
	}
    }

    // Feasability check
    bool is_feasible( size_t upper_bound, filter_reason r = fr_unknown ) {
	return m_E.is_feasible( upper_bound, r );
    }
    bool is_feasible_bool( bool cond, filter_reason r = fr_unknown ) {
	return m_E.is_feasible_bool( cond, r );
    }

    size_t get_max_clique_size() const {
	return m_E.get_max_clique_size();
    }

private:
    MC_CutOutEnumerator & m_E;
    const clique_set<VID> * const m_R;
    const VID * const m_remap;
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
	for( size_t v=0; v < N_VARIANTS; ++v ) {
	    for( size_t n=0; n < N_DIM; ++n ) {
		sum.m_dense[v][n] = m_dense[v][n] + s.m_dense[v][n];
		sum.m_leaf_dense[v][n] =
		    m_leaf_dense[v][n] + s.m_leaf_dense[v][n];
	    }
	    sum.m_gen[v] = m_gen[v] + s.m_gen[v];
	}
	sum.m_filter0 = m_filter0 + s.m_filter0;
	sum.m_filter1 = m_filter1 + s.m_filter1;
	sum.m_filter2 = m_filter2 + s.m_filter2;
	sum.m_heuristic = m_heuristic + s.m_heuristic;
	return sum;
    }

    void record_gen( algo_variant var, double atm ) {
	m_gen[(size_t)var].record( atm );
    }
    void record_genbuild( algo_variant var, double atm ) {
	m_gen[(size_t)var].record_build( atm );
    }
    void record_filter0( double atm ) { m_filter0.record( atm ); }
    void record_filter1( double atm ) { m_filter1.record( atm ); }
    void record_filter2( double atm ) { m_filter2.record( atm ); }
    void record_heuristic( double atm ) { m_heuristic.record( atm ); }

    variant_statistics & get( algo_variant var, size_t n ) {
	return m_dense[(size_t)var][n-N_MIN_SIZE];
    }
    variant_statistics & get_leaf( algo_variant var, size_t n ) {
	return m_leaf_dense[(size_t)var][n-N_MIN_SIZE];
    }
    
    variant_statistics m_dense[N_VARIANTS][N_DIM];
    variant_statistics m_leaf_dense[N_VARIANTS][N_DIM];
    variant_statistics m_gen[N_VARIANTS];
    variant_statistics m_filter0, m_filter1, m_filter2, m_heuristic;

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

/*======================================================================*
 * Exception for timeout on variants executing too long
 *======================================================================*/

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
	// report( std::cerr );
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
	    // std::cerr << "   alt #" << idx << " succeeded after "
	    // << dly << "\n";
	    m_success_time_total[idx] += dly;
	    m_success_time_max[idx] = std::max( m_success_time_max[idx], dly );
	    ++m_success[idx];
	    return ret;
	} catch( timeout_exception & e ) {
	    // std::cerr << "   alt #" << idx << " failed after "
	    // <<  tm.stop() << "\n";
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

    template<size_t idx, typename... Args>
	auto attempt_all_fn_aux( uint64_t usec, std::array<double,num_fns> & tms, Args && ... args ) {
	using return_type = decltype( std::get<0>( m_fn )( 0ull, args... ) );
	return_type ret;

	timer tm;
	tm.start();

	try {
	    // TODO: pass in ret as argument and use any contents filled in
	    //       even in case of timeout.
	    if( verbose > 0 )
		std::cerr << "as: alternative " << idx
			  << " timeout " << usec << "\n";
	    ret = execute_time_limited(
		usec, std::get<idx>( m_fn ), std::forward<Args>( args )... );
	    tms[idx] = tm.stop();

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


/*======================================================================*
 * Induced subgraph builder
 *======================================================================*/

template<typename GraphType>
class GraphBuilderInduced;

template<typename VID, typename EID, bool dual_rep, bool preallocate, typename HashSet>
class GraphBuilderInduced<graptor::graph::GraphHAdjPA<VID,EID,dual_rep,preallocate,false,HashSet>> {
public:
    template<typename HGraph>
    GraphBuilderInduced(
	const GraphCSx & G,
	const HGraph & H,
	VID v,
	const graptor::graph::NeighbourCutOutDegeneracyOrder<VID,EID> & cut )
	: S( G, H, cut.get_vertices(), cut.get_start_pos(),
	     cut.get_num_vertices(),
	     numa_allocation_small() ),
	  start_pos( cut.get_start_pos() ) { }
    template<typename HGraph>
    GraphBuilderInduced(
	const HGraph & H,
	const VID * const XP,
	VID ne,
	VID ce )
	: S( H, H, XP, ne, ce, numa_allocation_small() ),
	  start_pos( ne ) { }
    template<typename HGraph>
    GraphBuilderInduced(
	const GraphCSx & G,
	const HGraph & H,
	VID v,
	const graptor::graph::NeighbourCutOutDegeneracyOrderFiltered<VID,EID> & cut )
	: S( G, H, cut.get_vertices(), cut.get_num_vertices(),
	     numa_allocation_small() ),
	  start_pos( 0 ) { }
    template<typename HGraph>
    GraphBuilderInduced(
	const HGraph & H,
	VID v,
	const graptor::graph::NeighbourCutOutDegeneracyOrderFiltered<VID,EID> & cut )
	: S( H, H, cut.get_vertices(), cut.get_num_vertices(),
	     numa_allocation_small() ),
	  start_pos( 0 ) { }

    const auto & get_graph() const { return S; }
    VID get_start_pos() const { return start_pos; }

private:
    graptor::graph::GraphHAdjPA<VID,EID,dual_rep,preallocate,false,HashSet> S;
    VID start_pos;
};

template<typename lVID, typename lEID>
class GraphBuilderInduced<graptor::graph::GraphCSxDepth<lVID,lEID>> {
public:
    template<typename gVID, typename gEID>
    GraphBuilderInduced(
	const graptor::graph::GraphCSx<gVID,gEID> & G,
	const lVID * cut,
	lVID num_cut ) {
	gVID n = G.numVertices();
	lVID ns = num_cut;

	auto cut_slice = graptor::make_array_slice( cut, cut+num_cut );

	// Count induced edges
	std::vector<lEID> tmp_index( ns+1 );
	for( lVID p=0; p < ns; ++p ) {
	    VID v = cut[p]; // error at p=46 w/ trim?

	    // Degree of vertex
	    lVID deg = graptor::set_operations<graptor::MC_intersect>::
		intersect_size_ds( cut_slice, G.get_neighbours_set( v ) );
	    tmp_index[p] = deg;

	    assert( n != ns || deg == G.getDegree( v ) );
	    assert( deg != 0 ); // because of components
	}
	std::exclusive_scan( &tmp_index[0], &tmp_index[ns+1],
			     &tmp_index[0], 0 );

	// Construct selected graph
	// Edges: complement, not including diagonal
	lEID ms = tmp_index[ns];
	new ( &S ) graptor::graph::GraphCSxDepth(
	    ns, ms /*, numa_allocation_unbound()*/ );
	lEID * index = S.getIndex();
	lVID * edges = S.getEdges();
	lVID * depth = S.getDepth();
	lVID * degree = S.getDegree();

	// Set up index array
	std::copy( &tmp_index[0], &tmp_index[ns+1], index );
	index[ns] = ms;

	// Set edges
	for( lVID vs=0; vs < ns; ++vs ) {
	    gVID v = cut[vs];
	    lEID e = index[vs];
	    const gVID * gedges = G.get_neighbours( v );
	    gVID deg = G.getDegree( v );
	    lVID le = 0;

	    for( gVID ge=0; ge < deg; ++ge ) {
		gVID u = gedges[ge];
		while( le != ns && cut[le] < u )
		    ++le;
		assert( le <= ns || cut[le] >= u );
		if( le < ns && u == cut[le] )
		    edges[e++] = le;
	    }
	    assert( e == index[vs+1] );

	    degree[vs] = index[vs+1] - index[vs];
	    depth[vs] = S.initial_depth;
	}

	S.complete_init();
    }

    auto & get_graph() { return S; }

private:
    graptor::graph::GraphCSxDepth<lVID,lEID> S;
};

/*======================================================================*
 * Induced subgraph and complement builder
 *======================================================================*/
template<typename GraphType>
class GraphBuilderInducedComplement;

template<typename VID, typename EID>
class GraphBuilderInducedComplement<
    graptor::graph::GraphDoubleIndexCSx<VID,EID>> {
public:
    template<typename HGraph>
    GraphBuilderInducedComplement(
	const HGraph & G,
	const PSet<VID> & pset ) {
	VID n = G.numVertices();
	VID ns = pset.get_fill();

	// Count induced edges
	std::vector<VID> tmp_index( ns+1 );
	for( VID p=0; p < ns; ++p ) {
	    VID v = pset.at( p );

	    // Degree of vertex
	    VID deg = pset.intersect_size( G.get_neighbours_set( v ) );

	    tmp_index[p] = EID(ns) - 1 - deg;
	}
	std::exclusive_scan( &tmp_index[0], &tmp_index[ns+1],
			     &tmp_index[0], 0 );

	// Construct selected graph
	// Edges: complement, not including diagonal
	EID ms = tmp_index[ns];
	new ( &S ) graptor::graph::GraphDoubleIndexCSx(
	    ns, ms /*, numa_allocation_unbound()*/ );
	EID * sindex = S.getBeginIndex();
	EID * eindex = S.getEndIndex();
	VID * edges = S.getEdges();

	// Set up index array
	std::copy( &tmp_index[0], &tmp_index[ns+1], sindex );
	std::copy( &tmp_index[1], &tmp_index[ns+1], eindex );
	eindex[ns] = ms;

	// Set edges
	for( VID vs=0; vs < ns; ++vs ) {
	    VID v = pset.at( vs );
	    EID e = sindex[vs];
	    const VID * gedges = G.get_neighbours( v );
	    VID deg = G.getDegree( v );
	    VID ge = 0, gee = deg;

	    for( VID us=0; us < ns; ++us ) {
		VID u = pset.at( us );
		while( ge != gee && gedges[ge] < u )
		    ++ge;
		assert( ge == gee || gedges[ge] >= u );
		if( ( ge == gee || u != gedges[ge] )
		    && us != vs ) // no self-edges
		    edges[e++] = us;
	    }
	    assert( ge == gee || ge == gee-1 );
	    assert( e == sindex[vs+1] );
	    assert( e == eindex[vs] );
	}
    }

    auto & get_graph() { return S; }

private:
    graptor::graph::GraphDoubleIndexCSx<VID,EID> S;
};

template<typename lVID, typename lEID>
class GraphBuilderInducedComplement<
    graptor::graph::GraphCSxDepth<lVID,lEID>> {
public:
    template<typename HGraph, typename DualSet>
    GraphBuilderInducedComplement(
	const HGraph & G,
	const DualSet & pset ) {
	using gVID = typename HGraph::VID;
	using gEID = typename HGraph::EID;
	
	gVID n = G.numVertices();
	lVID ns = pset.size();

	// Count induced edges
	std::vector<lEID> tmp_index( ns+1 );
	for( lVID p=0; p < ns; ++p ) {
	    VID v = pset.at( p );

	    // Degree of vertex
	    // lVID deg = pset.intersect_size( G.get_neighbours_set( v ) );
	    lVID deg = graptor::set_operations<graptor::MC_intersect>
		::intersect_size_ds( pset, G.get_neighbours_set( v ) );

	    tmp_index[p] = ns - 1 - deg;
	}
	std::exclusive_scan( &tmp_index[0], &tmp_index[ns+1],
			     &tmp_index[0], 0 );

	// Construct selected graph
	// Edges: complement, not including diagonal
	lEID ms = tmp_index[ns];
	new ( &S ) graptor::graph::GraphCSxDepth(
	    ns, ms /*, numa_allocation_unbound()*/ );
	lEID * index = S.getIndex();
	lVID * edges = S.getEdges();
	lVID * depth = S.getDepth();
	lVID * degree = S.getDegree();

	// Set up index array
	std::copy( &tmp_index[0], &tmp_index[ns+1], index );
	index[ns] = ms;

	// Set edges
	for( lVID vs=0; vs < ns; ++vs ) {
	    gVID v = pset.at( vs );
	    lEID e = index[vs];
	    const gVID * gedges = G.get_neighbours( v );
	    gVID deg = G.getDegree( v );

	    gVID ge = 0, gee = deg;
	    for( lVID us=0; us < ns; ++us ) {
		gVID u = pset.at( us );
		while( ge != gee && gedges[ge] < u )
		    ++ge;
		assert( ge == gee || gedges[ge] >= u );
		if( ( ge == gee || u != gedges[ge] )
		    && us != vs ) // no self-edges
		    edges[e++] = us;
	    }
	    // assert( ge == gee || ge == gee-1 );
	    assert( e == index[vs+1] );

	    degree[vs] = index[vs+1] - index[vs];
	    depth[vs] = S.initial_depth;
	}

	S.complete_init();
    }

    auto & get_graph() { return S; }

private:
    graptor::graph::GraphCSxDepth<lVID,lEID> S;
};

template<typename lVID, typename lEID>
class GraphBuilderInducedComplement<
    graptor::graph::GraphCSx<lVID,lEID>> {
public:
    template<typename HGraph, typename DualSet>
    GraphBuilderInducedComplement(
	const HGraph & G,
	const DualSet & pset ) {
	using gVID = typename HGraph::VID;
	using gEID = typename HGraph::EID;
	
	gVID n = G.numVertices();
	lVID ns = pset.size();

	// Count induced edges
	std::vector<lEID> tmp_index( ns+1 );
	for( lVID p=0; p < ns; ++p ) {
	    VID v = pset.at( p );

	    // Degree of vertex
	    // lVID deg = pset.intersect_size( G.get_neighbours_set( v ) );
	    lVID deg = graptor::set_operations<graptor::MC_intersect>
		::intersect_size_ds( pset, G.get_neighbours_set( v ) );

	    tmp_index[p] = ns - 1 - deg;
	}
	std::exclusive_scan( &tmp_index[0], &tmp_index[ns+1],
			     &tmp_index[0], 0 );

	// Construct selected graph
	// Edges: complement, not including diagonal
	lEID ms = tmp_index[ns];
	new ( &S ) graptor::graph::GraphCSx(
	    ns, ms /*, numa_allocation_unbound()*/ );
	lEID * index = S.getIndex();
	lVID * edges = S.getEdges();

	// Set up index array
	std::copy( &tmp_index[0], &tmp_index[ns+1], index );
	index[ns] = ms;

	// Set edges
	for( lVID vs=0; vs < ns; ++vs ) {
	    gVID v = pset.at( vs );
	    lEID e = index[vs];
	    const gVID * gedges = G.get_neighbours( v );
	    gVID deg = G.getDegree( v );
	    gVID ge = 0, gee = deg;

	    for( lVID us=0; us < ns; ++us ) {
		gVID u = pset.at( us );
		while( ge != gee && gedges[ge] < u )
		    ++ge;
		assert( ge == gee || gedges[ge] >= u );
		if( ( ge == gee || u != gedges[ge] )
		    && us != vs ) // no self-edges
		    edges[e++] = us;
	    }
	    // assert( ge == gee || ge == gee-1 );
	    assert( e == index[vs+1] );
	}
    }

    auto & get_graph() { return S; }

private:
    graptor::graph::GraphCSx<lVID,lEID> S;
};


/*======================================================================*
 * graph matching
 *======================================================================*/

/**
 * eligible: constrain matching to edges incident to eligible vertices.
 *           A vertex v is eligible when eligible[v] == 0.
 */
template<typename GraphType>
std::vector<uint8_t>
graph_matching_outsiders( GraphType & G ) {
    // Bool array giving state for each vertex, initially 0
    VID n = G.numVertices();
    std::vector<uint8_t> state( n, 0 );

    for( VID u=0; u < n; ++u ) {
	if( state[u] != 0 )
	    continue;
	for( auto I=G.nbegin(u), E=G.nend(u); I != E; ++I ) {
	    VID v = *I;
	    if( state[v] == 0 ) {
		// Match edge
		state[u] = 1;
		state[v] = 1;
		break;
	    }
	}
    }

    return state;
}

template<typename GraphType>
std::vector<std::pair<VID,VID>>
auxiliary_graph_matching( GraphType & G,
			  const std::vector<uint8_t> & eligible,
			  std::vector<uint8_t> & state ) {
    // Bool array giving state for each vertex, initially 0
    VID n = G.numVertices();
    std::fill( state.begin(), state.end(), uint8_t(0) );
    std::vector<std::pair<VID,VID>> match;

    for( VID u=0; u < n; ++u ) {
	if( eligible[u] != 0 )
	    continue;
	if( state[u] != 0 )
	    continue;
	for( auto I=G.nbegin(u), E=G.nend(u); I != E; ++I ) {
	    VID v = *I;
	    if( state[v] == 0 ) {
		// Match edge
		match.push_back( { u, v } );
		state[u] = 1;
		state[v] = 1;
		break;
	    }
	}
    }

    return match;
}

// TODO: as we iterate over M2 frequently, would be useful to explicitly
// list those edges as they are few compared to all edges in graph.
template<typename GraphType>
std::vector<uint8_t>
crown_kernel( GraphType & G ) {
    VID n = G.numVertices();

    // Primary and auxiliary matchings
    // Let O = { M1 == 0 }
    std::vector<uint8_t> M1 = graph_matching_outsiders( G );
    std::vector<uint8_t> M2( n );
    std::vector<std::pair<VID,VID>> Ma = auxiliary_graph_matching( G, M1, M2 );

    std::vector<uint8_t> crown( n, 0 );

    // Check if every vertex in N(O) is matched by M2
    bool all_ngh_match_M2 = true;
    for( VID u=0; u < n && all_ngh_match_M2; ++u ) {
	if( M1[u] != 0 ) // membership O
	    continue;
	if( G.getDegree( u ) > 0 ) // ignore isolated vertices
	    crown[u] = 1; // I
	for( auto I=G.nbegin(u), E=G.nend(u); I != E; ++I ) {
	    VID v = *I;
	    if( M2[v] == 0 ) { // neighbour not matched
		all_ngh_match_M2 = false;
		break;
	    }
	    crown[v] = 2; // H
	}
    }

    if( all_ngh_match_M2 )
	return crown;

    // Initial I: I0 = { M2 == 0 }
    // Ignore isolated vertices
    for( VID v=0; v < n; ++v )
	crown[v] = M1[v] == 0 && M2[v] == 0 && G.getDegree( v ) > 0 ? 1 : 0;
    bool change = true;
    while( change ) {
	// Reset H
	for( VID u=0; u < n; ++u ) {
	    if( crown[u] == 2 )
		crown[u] = 0;
	}
	
	// Hn = N(In)
	for( VID u=0; u < n; ++u ) {
	    if( crown[u] != 1 ) // membership In
		continue;
	    for( auto I=G.nbegin(u), E=G.nend(u); I != E; ++I ) {
		VID v = *I;
		// crown[v] == 0 should hold trivially as In is independent set
		if( crown[v] == 0 )
		    crown[v] = 2;
	    }
	}

	// In+1 = In union N_M2(Hn)
	change = false;
	for( auto I=Ma.begin(), E=Ma.end(); I != E; ++I ) {
	    auto [ u, v ] = *I;
	    if( crown[u] == 2 && crown[v] != 1 ) { // u in Hn
		change = true;
		crown[v] = 1;
	    } else if( crown[v] == 2 && crown[u] != 1 ) { // v in Hn
		change = true;
		crown[u] = 1;
	    }
	}
    }

    return crown;
}

/*======================================================================*
 * Degeneracy (sequential, optimised for small graphs)
 *======================================================================*/
template<typename GraphType>
std::vector<typename GraphType::VID>
compute_coreness( GraphType & G ) {
    using VID = typename GraphType::VID;
    VID n = G.numVertices();
    std::vector<VID> degree( n );
    std::vector<VID> coreness( n, 0 );
    std::vector<VID> pos( n );
    std::vector<VID> queue( n );

    // Get degrees and construct index of all vertices
    // Assumes vertices are numbered [0..n)
    for( auto I=G.vbegin(), E=G.vend(); I != E; ++I ) {
	VID v = *I;
	degree[v] = G.getDegree( v );
	queue[v] = v;
    }

    // Sort vertices by increasing degree
    std::sort( queue.begin(), queue.end(),
	       [&]( VID u, VID v ) { return degree[u] < degree[v]; } );

    // Construct index
    for( VID p=0; p < n; ++p )
	pos[queue[p]] = p;

    VID K = 0;
    for( VID p=0; p < n; ++p ) {
	VID v = queue[p];
	if( K < degree[v] )
	    K = degree[v];
	coreness[v] = K;
	for( auto I=G.nbegin(v), E=G.nend(v); I != E; ++I ) {
	    VID u = *I;
	    --degree[u];
	    VID pp = pos[u];
	    while( pp > p+1 && degree[queue[pp-1]] > degree[u] ) {
		VID w = queue[pp-1];
		queue[pp] = w;
		pos[w] = pp;
		--pp;
	    }
	    pos[u] = pp;
	    queue[pp] = u;
	}
    }

    // Sanity check (have all vertices been visited)
    for( VID v=0; v < n; ++v ) {
	assert( coreness[v] != 0 || G.getDegree( v ) == 0 );
    }

    return coreness;
}

/*======================================================================*
 * vertex cover
 *======================================================================*/
template<bool exists, typename GraphType, typename lVID>
bool
vertex_cover_vc3( GraphType & G,
		  lVID k,
		  lVID c,
		  lVID & best_size,
		  lVID * best_cover );

// For path or cycle
template<typename GraphType, typename lVID>
void
trace_path( const GraphType & G,
	    bool * visited,
	    lVID & best_size,
	    lVID * best_cover,
	    lVID cur,
	    lVID nxt,
	    bool incl ) {
    if( visited[nxt] )
	return;

    visited[nxt] = true;

    if( incl )
	best_cover[best_size++] = nxt; // mark

    // const EID * const index = G.getBeginIndex();
    // const VID * const edges = G.getEdges();

    // Done if nxt is degree-1 vertex
    if( G.get_remaining_degree( nxt ) == 2 ) {
	auto ni = G.nbegin( nxt );
	lVID ngh1 = *ni; // edges[index[nxt]];
	lVID ngh2 = *++ni; // edges[index[nxt]+1];

	lVID ngh = ngh1 == cur ? ngh2 : ngh1;

	trace_path( G, visited, best_size, best_cover, nxt, ngh, !incl );
    }
}

template<typename GraphType, typename lVID>
bool
vertex_cover_poly( const GraphType & G,
		   lVID k,
		   lVID & best_size,
		   lVID * best_cover ) {
    lVID n = G.numVertices();

    std::vector<char> visited( n, false );

    VID old_best_size = best_size;

    // Find paths
    for( lVID v=0; v < n; ++v ) {
	lVID deg = G.get_remaining_degree( v );
	assert( deg <= 2 );
	if( deg == 1 && !visited[v] ) {
	    visited[v] = true;
	    trace_path( G, (bool*)&visited[0], best_size, best_cover,
			v, *G.nbegin( v ), true );
	}
    }
    
    // Find cycles (uses same auxiliary as paths)
    for( lVID v=0; v < n; ++v ) {
	lVID deg = G.get_remaining_degree( v );
	assert( deg <= 2 );
	if( deg == 2 && !visited[v] ) {
	    visited[v] = true;
	    best_cover[best_size] = v; // mark
	    ++best_size;
	    trace_path( G, (bool*)&visited[0], best_size, best_cover,
			v, *G.nbegin( v ), false );
	}
    }

    return best_size - old_best_size <= k;
}

template<bool exists, typename GraphType, typename lVID>
bool
vertex_cover_vc3_buss( GraphType & G,
		       lVID k,
		       lVID c,
		       lVID & best_size,
		       lVID * best_cover ) {
    using lEID = typename GraphType::EID;
    
    // Set U of vertices of degree higher than k
    lVID u_size = std::count_if( G.dbegin(), G.dend(),
				 [&]( lVID deg ) { return deg > k; } );

    // If |U| > k, then there exists no cover of size k
    if( u_size > k )
	return false;

    assert( u_size > 0 );
    
    // Construct G'
    lVID n = G.numVertices();
    auto chkpt = G.checkpoint();
    std::vector<lVID> ugt( u_size );
    lVID pos = 0;
    for( lVID v=0; v < n; ++v )
	if( G.get_remaining_degree( v ) > k )
	    ugt[pos++] = v;
    G.disable_incident_edges( ugt.begin(), ugt.end() );
    // G.disable_incident_edges( [&]( lVID v ) {
	// return chkpt.get_degree( v ) > k;
    // } );
    auto m = G.numEdges();
    
    // If G' has more than k(k-|U|) edges, reject
    if( m/2 > lEID(k) * lEID( k - u_size ) ) {
	G.restore_checkpoint( chkpt );
	return false;
    }

    // Find a cover for the remaining vertices
    lVID gp_best_size = 0;
    bool rec = vertex_cover_vc3<exists>(
	G, lVID( k - u_size ), c, gp_best_size, &best_cover[best_size] );

    if( rec ) {
	// Debug
	// check_cover( G, gp_best_size, &best_cover[best_size] );

	// for( lVID i=0; i < gp_best_size; ++i )
	// best_cover[best_size+i] = gp_xlat[best_cover[best_size+i]];
	best_size += gp_best_size;

	// All vertices with degree > k must be included in the cover
	for( lVID v=0; v < n; ++v )
	    if( chkpt.get_degree( v ) > k ) {
		best_cover[best_size] = v;
		++best_size;
	    }

	// check_cover( G, best_size, best_cover );
    }

    G.restore_checkpoint( chkpt );

    return rec;
}

#if 0
template<bool exists, typename GraphType, typename lVID>
int
vertex_cover_vc3_crown( GraphType & G,
			lVID k,
			lVID c,
			lVID & best_size,
			lVID * best_cover ) {
    lVID n = G.numVertices();

    // Compute crown kernel: (I=1,H=2)
    std::vector<uint8_t> crown = crown_kernel( G );

    // All vertices in H are included in cover
    lVID tmp_best_size = best_size;
    lVID h_size = 0, i_size = 0;
    for( lVID v=0; v < n; ++v ) {
	if( crown[v] == 2 ) {
	    ++h_size;
	    best_cover[tmp_best_size++] = v;
	} else if( crown[v] == 1 ) {
	    if( G.getDegree( v ) == 0 ) // ignore isolated vertices
		crown[v] = 0;
	    else
		++i_size;
	}
    }

    // Failure to identify crown
    if( i_size == 0 || h_size == 0 )
	return 2;

    if( h_size > k )
	return false;

    // Construct G' by removing all of I and H
    auto chkpt = G.checkpoint();
    G.disable_incident_edges( [&]( lVID v ) {
	return crown[v] != 0;
    } );
    typename Graphtype::EID lEID m = G.numEdges();

    // Find a cover for the remaining vertices
    lVID gp_best_size = 0;
    bool rec = vertex_cover_vc3<exists>(
	G, k - h_size, c,
	gp_best_size, &best_cover[tmp_best_size] );

    if( rec )
	best_size = tmp_best_size + gp_best_size;

    G.restore_checkpoint( chkpt );

    return rec;
}
#endif

template<unsigned Bits, typename lVID, typename lEID, typename GraphType>
bool leaf_vertex_cover(
    const GraphType & H,
    lVID n_remain,
    lVID k,
    lVID & best_size,
    lVID * best_cover ) {

    variant_statistics & stats
	= mc_stats.get_statistics().get_leaf( av_bk, ilog2( Bits ) );
    timer tm;
    tm.start();

    std::vector<lVID> cutout;
    cutout.reserve( n_remain );
    lVID nh = H.get_num_vertices();
    // std::cout << "nh=" << nh << " dp=" << H.get_cur_depth() << "\n";
    for( lVID v=0; v < nh; ++v )
	if( H.get_remaining_degree( v ) > 0 ) {
	    cutout.push_back( v );
	    // std::cout << "cut v=" << v << " deg=" << H.get_degree(v)
	    // << " depth=" << H.get_depth(v) << "\n";
	}

    // std::cout << "nh=" << nh << " nr=" << n_remain << " Bits=" << Bits
    // << " cut=" << cutout.size() << "\n";
    assert( cutout.size() == H.get_num_remaining_vertices() );
    assert( cutout.size() <= Bits );

    DenseMatrix<Bits,lVID,lEID> D( H, &cutout[0], cutout.size() );

    lVID n = D.numVertices();
    lEID m = D.get_num_edges();
    float d = (float)m / ( (float)n * (float)(n-1) );
    stats.record_build( tm.next() );

    assert( m == H.get_num_remaining_edges() );

    if( verbose > 3 )
	std::cout << "VC cutout: nrem=" << n_remain << " n=" << n
		  << " m=" << m << " d=" << d << "\n";

    auto [ bs, sz ] = D.template vertex_cover_kernelised<false>( k );
    bool ret;
    if( ~sz == (lVID)0 || sz > k )
	ret = false;
    else {
	for( auto v : bs )
	    best_cover[best_size++] = cutout[v];
	ret = true;
    }

    float tbk = tm.next();
    stats.record( tbk );

    return ret;
}

template<bool exists, typename GraphType, typename lVID>
bool
vertex_cover_vc3( GraphType & G,
		  lVID k,
		  lVID c,
		  lVID & best_size,
		  lVID * best_cover ) {
    using lEID = typename GraphType::EID;
    
    lVID n = G.numVertices();
    lEID m = G.numEdges();

    if( k == 0 )
	return m == 0;

    // TODO: track number of non-zero-degree vertices and if
    //       less than k, declare success (not searching best option...).
    //       actually, look at vertices with degree > 1 (paths/cycles)?
    // if( get_num_remaining_vertices() <= k ) -> success, but how?

    // Apply reduction rules for degree-1 vertices.
    lVID min_v, min_deg;
    std::tie( min_v, min_deg ) = G.min_degree();
    if( min_deg == 1 ) {
	lVID ngh = *G.nbegin( min_v );
	// lVID rm[2] = { min_v, ngh };
	// auto chkptv = G.disable_incident_edges( &rm[0], &rm[2] );
	// should suffice to remove just ngh as the 1-degree neighbour min_v
	// will be removed/disabled with it
	auto chkptv = G.disable_incident_edges_for( ngh );
	lVID m_best_size = 0;
	bool ok = vertex_cover_vc3<exists>(
	    G, (lVID)(k-(lVID)1), c, m_best_size, &best_cover[best_size+1] );
	if( ok ) {
	    best_cover[best_size] = ngh;
	    best_size += m_best_size + 1;
	}
	G.restore_checkpoint( chkptv );
	return ok;
    }

    lVID max_v, max_deg;
    std::tie( max_v, max_deg ) = G.max_degree();
    if( max_deg <= 2 )
	return vertex_cover_poly( G, k, best_size, best_cover );

/* in-effective
    int ret
	= vertex_cover_vc3_crown<exists>( G, k, c, best_size, best_cover );
    if( ret != 2 )
	return (bool)ret;
*/

    if( m/2 > c * k * k && max_deg > k ) {
	// replace by Buss kernel
	return vertex_cover_vc3_buss<exists>( G, k, c, best_size, best_cover );
    }

/*
    std::cout << "nrem=" << G.get_num_remaining_vertices()
	      << " k=" << k
	      << " max_v=" << max_v
	      << " max_deg=" << max_deg
	      << "\n";
*/

    // Consider if it is worthwhile to transform to a dense problem.
    // Do this after the buss kernel, as the buss kernel may give a reasonable
    // reduction graph size.
#if !ABLATION_DISABLE_LEAF
    lVID n_remain = G.get_num_remaining_vertices();
    if( n_remain <= (VID(1)<<N_MAX_SIZE) ) {
	typedef bool (*fptr_t)( const GraphType &, lVID, lVID, lVID &, lVID * );
	static fptr_t fptr[N_DIM+1] = {
	    &leaf_vertex_cover<32,lVID,lEID,GraphType>,
	    &leaf_vertex_cover<64,lVID,lEID,GraphType>,
	    &leaf_vertex_cover<128,lVID,lEID,GraphType>,
	    &leaf_vertex_cover<256,lVID,lEID,GraphType>,
	    &leaf_vertex_cover<512,lVID,lEID,GraphType> };
	    
	VID nlg = get_size_class( n_remain );
	if( nlg < N_MIN_SIZE )
	    nlg = N_MIN_SIZE;
	if( nlg <= N_MAX_SIZE ) {
	    lVID d_best_size = 0;
	    bool ok = fptr[nlg-N_MIN_SIZE](
		G, n_remain, k, d_best_size, &best_cover[best_size] );
	    if( ok )
	        best_size += d_best_size;
	    return ok;
	}
    }
#endif // ABLATION_DISABLE_LEAF

    // Must have a vertex with degree >= 3 (trivial given check for poly)
    // assert( max_deg >= 3 );

    // Create two subproblems ; branch on max_v.
    // Try the first subproblem (include vertex in cover). As we pick high-
    // degree vertices first, it makes sense to try the included case first
    // as including high-degree vertices helps to minimise the cover size.

    // Erase the selected vertex
    auto chkptv = G.disable_incident_edges_for( max_v );

    // The first subproblem is stored direct in the best_cover storage space.
    // Tentatively add vertex to cover
    best_cover[best_size] = max_v;

    // First subproblem includes max_v, reduces target k by one
    lVID i_k = k-1;
    lVID i_best_size = 0;
    bool i_ok = vertex_cover_vc3<exists>(
	G, i_k, c, i_best_size, &best_cover[best_size+1] ); // +1 for max_v

    if constexpr ( exists ) {
	if( i_ok ) {
	    G.restore_checkpoint( chkptv );
	    // assert( i_best_size <= k );
	    best_size += i_best_size + 1;
	    return true;
	}
    }
    
    // The first subproblem did not provide a solution (exists=true) or
    // we are looking for the best possible solution (exists=false).
    // Store the second subproblem in best_cover (exists=true) or create
    // new storage space (exists=false).
    lVID x_best_size = 0;
    lVID * x_best_cover = nullptr;
    if constexpr ( exists ) {
	// The first subproblem has failed, overwrite space. Discard max_v.
	x_best_cover = &best_cover[best_size];
    } else {
	if( i_ok ) {
	    // Retain the first subproblem, we will pick the best one later.
	    x_best_cover = new lVID[n-1-max_deg];
	} else {
	    // Overwrite the solution to the first subproblem as it is invalid.
	    x_best_cover = &best_cover[best_size];
	}
    }

    // In case v is excluded, erase v (previous step) and all its neighbours.
    // Create a checkpoint. Note that max_v has been removed and remains
    // removed at this stage.
    auto NI = G.nbegin( max_v );
    auto NE = G.nend( max_v );
    auto chkptn = G.disable_incident_edges( NI, NE );

    // The k value to aim for.
    // Note: if !exists, then additionally take x_k = std::min( x_k, i_k ).
    lVID x_k = std::min( n-1-max_deg, k-max_deg );
    // If the first subproblem succeeded, tighten k to find only a better
    // solution. We have a solution of size i_best_size+1 (including max_v);
    // look for at most k = i_best_size+1-1.
    if( !exists && i_ok )
	x_k = std::min( x_k, i_best_size );

    bool x_ok = false;
    if( k >= max_deg )
	x_ok = vertex_cover_vc3<exists>( G, x_k, c, x_best_size, x_best_cover );

    // Restore checkpoints on the graph, putting back neighbours
    // as well as max_v
    G.restore_checkpoint( chkptn );
    G.restore_checkpoint( chkptv );

    bool ret = false;
    
    // Check if a solution was found
    if constexpr ( exists ) {
	// Inclusive subproblem had failed. Overwritten existing space in
	// best_cover.
	if( x_ok ) {
	    // Add neighbours of excluded vertex (max_v) to the cover.
	    best_size += x_best_size;
	    for( auto I=NI; I != NE; ++I ) {
		best_cover[best_size] = *I;
		++best_size;
	    }

	    ret = true;
	} else
	    ret = false;
    } else {
	if( !i_ok ) {
	    if( x_ok ) {
		assert( std::distance( NI, NE ) == max_deg );
		
		// Inclusive subproblem had failed. Overwritten existing
		// space in best_cover.
		// Add neighbours of excluded vertex (max_v) to the cover.
		best_size += x_best_size;
		std::copy( NI, NE, &best_cover[best_size] );
		best_size += max_deg;

		ret = true;
	    } else {
		// Both subproblems failed.
		ret = false;
	    }
	} else { // i_ok
	    // For memory allocated
	    assert( x_best_size <= n - 1 - max_deg );

	    // Take best of two solutions,
	    // or first subproblem succeeded while second did not improve.
	    if( !x_ok || i_best_size+1 <= x_best_size+max_deg ) {
		// Retain first solution
		delete[] x_best_cover;
	    } else {
		// Retain second solution.
		std::copy( NI, NE, &best_cover[best_size] );
		std::copy( &x_best_cover[0], &x_best_cover[x_best_size],
			   &best_cover[best_size+max_deg] );
		delete[] x_best_cover;
	    }
	    best_size += x_best_size + max_deg;
	    ret = true;
	}
    }

    return ret;
}

template<typename lVID>
lVID cc_find( lVID v, lVID * label ) {
    lVID u = label[v];
    while( v != u ) {
	label[v] = label[u];
	v = u;
	u = label[v];
    }
    return u;
}

template<typename lVID, typename Fn>
void cc_union( lVID u, lVID v, Fn && fn, lVID * label ) {
    lVID x = cc_find( u, label );
    lVID y = cc_find( v, label );

    // Already in same tree?
    if( x == y )
	return;

    // Vertex with highest degree goes on top, tie-breaker is lexicographic
    // ordering. Swap such that x goes on top.
    if( fn( x, y ) ) // does y go on top of x?
	label[x] = y;
    else
	label[y] = x;
}

/*! Find connected components
 *
 * This is a sequential implementation using union-find
 */
template<typename lVID, typename lEID, typename GraphTy>
lVID
connected_components( const GraphTy & G, std::vector<lVID> & label ) {
    lVID n = G.numVertices();
    
    // Initialise single-node trees
    std::iota( label.begin(), label.end(), 0 );

    // Make pass over all edges, merging trees for end-points of each edge
    // Ignore any erased edges; we assume none have been erased yet.
    const lEID * const index = G.getIndex();
    const lVID * const edges = G.getEdges();
    lEID e = 0;
    for( lVID v=0; v < n; ++v ) {
	lEID ee = index[v+1];
	for( ; e < ee; ++e ) {
	    // Use symmetry of graph to process each edge once.
	    if( v <= edges[e] ) {
		e = ee;
		break; // break assuming vertices sorted in increasing manner
	    }
	    
	    // Perform union operation
	    cc_union( v, edges[e], [=]( lVID x, lVID y ) {
		lVID deg_x = index[x+1] - index[x];
		lVID deg_y = index[y+1] - index[y];
		return deg_x < deg_y || ( deg_x == deg_y && x < y );
	    }, &label[0] );
	}
    }

    lVID nwcc = 0;
    for( lVID v=0; v < n; ++v )
	if( cc_find( v, &label[0] ) == v )
	    ++nwcc;

    return nwcc;
}

template<typename lVID, typename lEID>
bool
find_min_vertex_cover( graptor::graph::GraphCSxDepth<lVID,lEID> & G,
		       lVID & csize,
		       lVID * cover ) {
    lVID cn = G.numVertices();
    lEID cm = G.numEdges();

    // If no edges remain after pruning, then cover has size 0 and
    // clique has size cn.
    if( cm == 0 )
	return true;

    // If one edge remains, cover has size 1
    if( cm == 1 ) {
	cover[csize++] = 0; // any vertex
	return true;
    }

    constexpr lVID c = 1;
    return vertex_cover_vc3<false>( G, cn, c, csize, cover );
}

template<typename lVID, typename lEID>
bool
find_min_vertex_cover_existential( graptor::graph::GraphCSxDepth<lVID,lEID> & G,
				   lVID k_max, // we know a cover with size k_max exists
				   lVID & csize,
				   lVID * cover ) {
    lVID cn = G.numVertices();
    lEID cm = G.numEdges();

    // If no edges remain after pruning, then cover has size 0 and
    // clique has size cn.
    if( cm == 0 )
	return true;

    // If one edge remains, cover has size 1
    if( cm == 1 ) {
	cover[csize++] = 0; // any vertex
	return true;
    }

    constexpr lVID c = 1;
    std::vector<lVID> cur_cover( cn );
    lVID cur_size = 0;

    timer tm;
    tm.start();

    const lVID k_prior = k_max;
    lVID k_up = k_prior - 1;
    /* TODO: the minimum size of a vertex cover is at least half the number
     *       of vertices with non-zero degree (?) 
     * Assumption mentioned by Chen and Kanj in "On Approximating Minimum
     * Vertex Cover for Graphs with Perfect Matching"
     * Assuming graph G is a connected component, then there are no zero-degree
     * vertices (as previously checked this is not a singleton component).
     */
    lVID k_lo = 1; // cn / 2; // lower bound, round down: only after kernelisation?
    lVID k_best_size = k_prior;
    lVID k = k_up;
    bool first_attempt = true;
    while( true ) {
	constexpr lVID c = 1;
	cur_size = 0;
	bool any = vertex_cover_vc3<true>( G, k, c, cur_size, &cur_cover[0] );
	if( verbose > 1 ) {
	    std::cout << " vc3: cn=" << cn << " k=[" << k_lo << ','
		      << k << ',' << k_up << "] bs=" << cur_size
		      << " ok=" << any
		      << ' ' << tm.next()
		      << "\n";
	}
	if( any ) {
	    k_best_size = cur_size;
	    // in case we find a better cover than requested
	    if( k > k_best_size )
		k = k_best_size;
	    std::copy( &cur_cover[0], &cur_cover[cur_size], cover );
	    csize = cur_size;
	}

	// Reduce range
	if( any ) // k too high
	    k_up = k;
	else
	    k_lo = k;
	if( k_up <= k_lo+1 )
	    break;

	// Determine next k
	// On the first attempt, we prefer to drop k just by one as it often
	// fails at -1. After that, we perform binary search.
	if( first_attempt ) {
	    first_attempt = false;
	    k = k_up - 1;
	} else
	    k = ( k_up + k_lo ) / 2;
    }

    return k_best_size < k_max;
}

template<typename T, typename Fn>
T complement_set( T n, const T * b, const T * e, T * x, Fn && fn ) {
    size_t k = 0;
    for( T i=0; i < n; ++i ) {
	if( b != e && *b == i ) {
	    ++b;
	} else {
	    *x++ = fn( i );
	    ++k;
	}
    }
    return k;
}

template<typename lVID, typename lEID, typename Enumerator>
lVID
clique_via_vc3_cc( graptor::graph::GraphCSx<lVID,lEID> & CG,
		   lVID v,
		   lVID degeneracy,
		   Enumerator & E,
		   int depth ) {
    // PSet<VID> pset = PSet<VID>::create_full_set( G );
    // lVID ce = pset.size();
    
    // TODO: potentially apply more filtering using up-to-date best
    //       might do conditionally on improvement of best since
    //       previous cut-out
    // Note: when called from top-level, the pset contains all vertices and
    //       no further filtering is applied.
    // assert( ce == pset.get_fill() );
    // GraphBuilderInducedComplement<graptor::graph::GraphCSx<lVID,lEID>>
	// cbuilder( G, pset );
    // auto & CG = cbuilder.get_graph();
    lVID cn = CG.numVertices();
    lEID cm = CG.numEdges();

/*
    {
	std::ofstream f( "./cutout.hgr" );
	f << "p edge " << cn << ' ' << cm << "\n";
	for( VID v=0; v < cn; ++v ) {
	    const lVID * ngh = CG.get_neighbours( v );
	    VID deg = CG.getDegree( v );
	    for( VID i=0; i < deg; ++i )
		f << "e " << (v+1) << ' ' << (ngh[i]+1) << "\n";
	}
	f.close();
    }
*/

    // If no edges remain after pruning, then cover has size 0 and
    // clique has size cn.
    if( cm == 0 ) {
	std::cout << "clique_via_vc3: no edges in complement graph\n";
	PSet<VID> pset = PSet<VID>::create_full_set( CG );
	E.record( depth+cn, pset.begin(), pset.end() );
	return depth+cn; // return true;
    }

    // Check if improving the best known clique is impossible
    lVID bc = E.get_max_clique_size();
    if( bc >= cn + depth ) {
	std::cout << "[unexpected] clique_via_vc3: insufficient vertices: "
		  << " cn=" << cn << " depth=" << depth << " bc=" << bc << "\n";
	return 0; // return false;
    }

    // Calculate weakly connected components
    std::vector<lVID> wcc_label( cn );
    lVID nwcc = connected_components<lVID,lEID>( CG, wcc_label );
    std::cout << "number of components: " << nwcc << "\n";

    std::vector<lVID> wcc_root( nwcc ), wcc_size( nwcc ), wcc_id( cn );
    lVID cur = 0;
    for( lVID v=0; v < cn; ++v ) {
	lVID root = cc_find( v, &wcc_label[0] );
	if( root == v ) {
	    wcc_root[cur] = v;
	    wcc_size[cur] = 1;
	    wcc_id[v] = cur++;
	}
    }
    for( lVID v=0; v < cn; ++v ) {
	lVID root = cc_find( v, &wcc_label[0] );
	if( root != v )
	    ++wcc_size[wcc_id[root]];
    }

    assert( cur == nwcc );

    lVID tot = 0;
    for( lVID cc=0; cc < nwcc; ++cc )
	tot += wcc_size[cc];

    assert( tot == cn );

    // Sort components by size
    paired_sort( wcc_size.begin(), wcc_size.end(), wcc_root.begin() );
    for( lVID cc=0; cc < nwcc; ++cc )
	wcc_id[wcc_root[cc]] = cc;

/*
    for( lVID cc=0; cc < nwcc; ++cc ) {
	std::cout << "wcc " << cc << ": root=" << wcc_root[cc]
		  << " size=" << wcc_size[cc]
		  << " root_deg=" << CG.getDegree( wcc_root[cc] )
		  << "\n";
    }
*/

    std::vector<lVID> best_clique( cn );
    std::vector<lVID> cover( cn );
    lVID best_size = 0;

    // Solve components one by one
    for( lVID cc=0; cc < nwcc; ++cc ) {
	if( wcc_size[cc] == 1 ) {
	    // An isolated vertex - nothing to add to vertex cover.
	    // Vertex goes into clique
	    best_clique[best_size++] = wcc_root[cc];
	} else if( wcc_size[cc] == 2 ) {
	    // A graph with 2 vertices which are known to be connected
	    // has precisely one vertex in the minimum vertex cover.
	    // Pick either. The other becomes part of the clique.
	    best_clique[best_size++] = wcc_root[cc];
	} else {
	    /*
	    std::cout << "processing wcc " << cc << ": root=" << wcc_root[cc]
		      << " size=" << wcc_size[cc]
		      << " depth=" << depth
		      << " cn=" << cn
		      << " bc=" << bc
		      << " best_size=" << best_size
		      << "\n";
	    */
	    // Create cutout of graph representing this component
	    std::vector<lVID> cut;
	    cut.reserve( wcc_size[cc] );
	    for( lVID v=0; v < cn; ++v )
		if( cc_find( v, &wcc_label[0] ) == wcc_root[cc] )
		    cut.push_back( v );

	    assert( cut.size() == wcc_size[cc] );
	    
	    GraphBuilderInduced<graptor::graph::GraphCSxDepth<lVID,lEID>>
		ccbuilder( CG, &cut[0], cut.size() );
	    graptor::graph::GraphCSxDepth<lVID,lEID> & CCG =
		ccbuilder.get_graph();

	    // Find minimum vertex cover
	    lVID csize = 0;
	    
	    // We know of a clique of size bc. We currently have assembled
	    // a clique of size depth + best_size. We can at best make a clique
	    // of size depth + best_size + wcc_size[cc]. Check if we can
	    // already improve upon the best known clique with the current
	    // component.
	    bool fnd;
	    if( cc == nwcc-1
		&& depth + best_size <= bc
		&& depth + best_size + wcc_size[cc] > bc ) {
		// This component may result in increased clique size.
		// Solve using existential queries as we assume improvement
		// won't be possible.
		lVID oc = depth + best_size + wcc_size[cc];
		lVID k_known = std::min( wcc_size[cc], (lVID)( oc - bc ) );
		// std::cout << "existential solve k_known=" << k_known << "\n";
		fnd = find_min_vertex_cover_existential<lVID,lEID>(
		    CCG, k_known,
		    csize, &cover[0] );
	    } else {
		// Solving this component won't affect best clique size yet.
		// std::cout << "absolute solve\n";
		fnd = find_min_vertex_cover<lVID,lEID>(
		    CCG, csize, &cover[0] );
	    }
	    // std::cout << "existential solve fnd=" << fnd << " csize=" << csize
	    // << " bs=" << best_size << " depth=" << depth << "\n";
	    // If we cannot identify any minimum vertex cover, then we
	    // fail overall.
	    if( !fnd )
		return 0;

	    // Translate IDs
	    std::sort( &cover[0], &cover[csize] );
	    best_size += complement_set( CCG.get_num_vertices(),
					 &cover[0], &cover[csize],
					 &best_clique[best_size],
					 [&]( lVID v ) { return cut[v]; } );
	    // for( lVID i=0; i < csize; ++i )
	    // best_cover[best_size+i] = cut[best_cover[best_size+i]];
	    // best_size += csize;
	}
    }
	    
    if( E.is_feasible( depth + best_size, fr_cover ) ) {
	if( verbose > 3 )
	    std::cout << "clique_via_vc3: max_clique: "
		      << ( depth + best_size )
		      << " E.best: " << bc << "\n";
	E.record( depth + best_size, best_clique.begin(),
		  std::next( best_clique.begin(), best_size ) );
    }

    return depth + cn - best_size;
}

template<typename lVID, typename lEID>
void
validate_vertex_cover( graptor::graph::GraphCSxDepth<lVID,lEID> & CG, 
		       const lVID * CI, const lVID * CE ) {
    lVID n = CG.get_num_vertices();
    auto cover = graptor::make_array_slice( CI, CE );

    assert( CG.get_cur_depth() == 0 );
    for( lVID v=0; v < n; ++v )
	assert( ~CG.get_depth( v ) == 0 );
    
    for( lVID v=0; v < n; ++v ) {
	if( std::binary_search( CI, CE, v ) ) {
	    // all incident edges covered
	} else {
	    // all neighbours must be in cover
	    lVID size = graptor::set_operations<graptor::MC_intersect>::
		intersect_size_ds( cover, CG.get_neighbours_set( v ) );
	    if( size < CG.get_degree( v ) ) {
		std::cout << "ERROR: vertex " << v
			  << " has not all neighbours covered. size="
			  << size << "\n";
	    }
	}
    }
}
    
template<typename lVID, typename lEID, bool use_exist, typename Enumerator>
lVID
clique_via_vc3_mono( graptor::graph::GraphCSxDepth<lVID,lEID> & CG,
		     lVID v,
		     lVID degeneracy,
		     Enumerator & E,
		     int depth ) {
    // PSet<VID> pset = PSet<VID>::create_full_set( G );
    // lVID ce = pset.size();
    
    // TODO: potentially apply more filtering using up-to-date best
    //       might do conditionally on improvement of best since
    //       previous cut-out
    // Note: when called from top-level, the pset contains all vertices and
    //       no further filtering is applied.
    // assert( ce == pset.get_fill() );
    // GraphBuilderInducedComplement<graptor::graph::GraphCSxDepth<lVID,lEID>>
    // cbuilder( G, pset );
    // auto & CG = cbuilder.get_graph();
    lVID cn = CG.numVertices();
    lEID cm = CG.numEdges();

    // If no edges remain after pruning, then cover has size 0 and
    // clique has size cn.
    if( cm == 0 ) {
	// PSet<VID> pset = PSet<VID>::create_full_set( CG );
	std::vector<VID> pset( cn );
	std::iota( pset.begin(), pset.end(), 0 );
	E.record( depth+cn, pset.begin(), pset.end() );
	return depth+cn; // return true;
    }

    // Check if improving the best known clique is impossible
    lVID bc = E.get_max_clique_size();
    if( bc >= cn + depth )
	return 0; // return false;

    std::vector<lVID> best_cover( cn );
    lVID best_size = 0;

    // Set initial k on the basis of best known clique by E
    // May need two tries: once with pessimistic but restrictive k asking
    // the question if we can improve over the best known clique.
    // If yes, then second time trying to find the minimum cover with k=1
    // -- but best_cover/best_size carried over as found by the first attempt.
    // Problem: shouldn't register those VC/cliques as the VC aren't
    //          minimum...
    // The idea is that once the optimal clique has been found, setting a
    // more challenging constraint will fail faster than setting a constraint
    // of k=1; in fact should fail faster in general?
    // Don't run search for a 2-clique at the very start - doesn't help as it
    // won't return a no answer for sure.
    // looking for a better clique/cover than what we know
    timer tm;
    tm.start();
    const lVID k_prior = cn + depth - bc;
    lVID k_best_size = k_prior;

    if constexpr ( use_exist ) {
	std::vector<lVID> cover( cn );
	lVID k_up = k_prior - 1;
	// TODO: the minimum size of a vertex cover is at least half the number
	//       of vertices with non-zero degree (?) 
	// Assumption mentioned by Chen and Kanj in "On Approximating Minimum
	// Vertex Cover for Graphs with Perfect Matching"
	lVID k_lo = 1;
	lVID k = k_up;
	bool first_attempt = true;
	while( true ) {
	    constexpr lVID c = 1;
	    best_size = 0;
	    bool any = vertex_cover_vc3<true>( CG, k, c, best_size, &cover[0] );
	    if( verbose > 1 ) {
		std::cout << " vc3: cn=" << cn << " k=[" << k_lo << ','
			  << k << ',' << k_up << "] bs=" << best_size
			  << " ok=" << any
			  << ' ' << tm.next()
			  << "\n";
	    }
	    if( any ) {
		k_best_size = best_size;
		// in case we find a better cover than requested
		if( k > k_best_size )
		    k = k_best_size;
		std::copy( &cover[0], &cover[best_size], best_cover.begin() );
	    }

	    // Reduce range
	    if( any ) // k too high
		k_up = k;
	    else
		k_lo = k;
	    if( k_up <= k_lo+1 )
		break;

	    // Determine next k
	    // On the first attempt, we prefer to drop k just by one as it often
	    // fails at -1. After that, we perform binary search.
	    if( first_attempt ) {
		first_attempt = false;
		k = k_up - 1;
	    } else {
		k = ( k_up + k_lo ) / 2;
	    }
	}
    } else {
	constexpr lVID c = 1;
	lVID k = k_prior - 1;
	bool any = vertex_cover_vc3<false>(
	    CG, k, c, best_size, &best_cover[0] );
	if( verbose > 1 ) {
	    std::cout << " vc3 (abs): cn=" << cn << " k=" << k
		      << " ok=" << any << ' ' << tm.next() << "\n";
	}
	if( any ) {
	    // in case we find a better cover than requested
	    k_best_size = best_size;
	}
    }

    // Record clique
    if( k_best_size < k_prior ) {
	if( E.is_feasible( depth + cn - k_best_size, fr_cover ) ) {
	    if( verbose > 3 )
		std::cout << "clique_via_vc3: max_clique: "
			  << ( depth + cn - k_best_size )
			  << " E.best: " << bc << "\n";
	    // Create complement set. Store at full width (VID)
	    std::vector<VID> clique( cn - k_best_size );
	    std::sort( &best_cover[0], &best_cover[k_best_size] );
	    for( lVID i=0, j=0, k=0; i < cn; ++i ) {
		if( best_cover[j] == i ) {
		    if( j < k_best_size-1 )
			++j;
		} else
		    clique[k++] = i;
	    }
	    E.record( depth + cn - k_best_size,
		      clique.begin(), clique.end() ); // size of complement
	}
    }

    return depth + cn - k_best_size;
}

/*======================================================================*
 * recursively parallel version of Bron-Kerbosch w/ pivoting
 *======================================================================*/

#if MEMOIZE_MC_PER_VERTEX
static VID * max_clique_per_vertex = nullptr;
#endif

template<typename VID, typename EID>
bool mc_leaf(
    const HGraphTy & H,
    MC_CutOutEnumerator & E,
    const clique_set<VID> * R,
    const PSet<VID> & xp_set,
    size_t depth );

template<bool allow_dense>
void
mc_bron_kerbosch_recpar_xps(
    const HGraphTy & G,
    VID degeneracy,
    MC_CutOutEnumerator & E,
    const clique_set<VID> * R,
    PSet<VID> & xp,
    int depth );

template<typename HGraph>
std::pair<VID,VID>
__attribute__((noinline))
mc_get_pivot(
    const HGraph & G,
    const PSet<VID> & pset ) {

    const VID * const XP = pset.get_set();
    const VID ce = pset.get_fill();

    // Tunable (|P| and selecting vertex from X or P)
    if( ce <= 3 )
	return std::make_pair( XP[0], 0 );

    VID v_max = ~VID(0);
    VID tv_max = std::numeric_limits<VID>::min();

    for( VID i=0; i < ce; ++i ) {
	VID v = XP[i];
	auto & hadj = G.get_adjacency( v );
	VID deg = hadj.size();
	if( deg <= tv_max )
	    continue;

	// Abort during intersection_size if size will be less than tv_max
	// Note: hash_vector is much slower in this instance
	size_t tv = pset.intersect_size_gt_val(
	    G.get_neighbours_set( v ), tv_max );
	if( tv > tv_max ) {
	    tv_max = tv;
	    v_max = v;
	}
    }

    // return first element of P if nothing good found
    return std::make_pair( ~v_max == 0 ? XP[0] : v_max, tv_max );
}

template<typename HGraph>
std::pair<VID,VID>
__attribute__((noinline))
get_max_degree_vertex( const HGraph & G ) {

    const VID n = G.get_num_vertices();

    if( n <= 3 )
	return std::make_pair( 0, 0 );

    VID v_max = 0;
    VID tv_max = std::numeric_limits<VID>::min();

    for( VID v=0; v < n; ++v ) {
	auto & hadj = G.get_adjacency( v );
	VID deg = hadj.size();
	if( deg > tv_max ) {
	    tv_max = deg;
	    v_max = v;
	}
    }

    // return first element of P if nothing good found
    return std::make_pair( v_max, tv_max );
}


template<bool allow_dense>
void
bk_recursive_call(
    const HGraphTy & G,
    VID degeneracy,
    MC_CutOutEnumerator & E,
    const clique_set<VID> * R,
    PSet<VID> & xp_new,
    int depth ) {
    VID ce_new = xp_new.size();
    
    // Check if the best possible clique we can construct would improve
    // over the current best known clique.
    if( !E.is_feasible( depth + ce_new, fr_pset ) )
	return;

    // Reached leaf of search tree
    if( ce_new == 0 ) {
	E.record( depth, R->begin(), R->end() );
	return;
    }

    if constexpr ( allow_dense ) {
	if( mc_leaf<VID,EID>( G, E, R, xp_new, depth ) )
	    return;
    }

    // Large sub-problem; search recursively
    // Tuning point: do we cut out a subgraph or not?
    // Tuning point: do we proceed with MC or switch to VC?
    mc_bron_kerbosch_recpar_xps<allow_dense>(
	G, degeneracy, E, R, xp_new, depth );
}

template<typename VID>
std::pair<VID,VID>
count_colours_ub( const HGraphTy & G, const PSet<VID> & xp ) {
    // Upper bound, loose?
    // Example: if we have i neighbours in the PSet, we would deduce the need
    // for colour i, however, if some of those neighbours can have the same
    // colour, then we don't need colour i.
    VID n = xp.size();
    VID c = 1; // number of colours in use
    VID max_rdeg = 0;
    for( VID j=1; j < n; ++j ) {
	VID i = n-1 - j;
	VID v = xp.at( i );
	const auto & adj = G.get_neighbours_set( v );
	VID isz = xp.intersect_size_from( adj, i );
	// c = std::max( isz, c );
	// Rationale: if we have as many neighbours as colours already handed
	// out, we will need an additional colour. If, for instance, we have
	// 10 neighbours but only 5 colours are used for them, we just need
	// a 6-th colour, we don't need 10.
	// What we are not checking is the situation where those 10 neighbours
	// actually do use all 5 colours. They may be using only 3 among them,
	// which would not require an additional colour.
	if( isz >= c )
	    ++c;
	if( isz > max_rdeg )
	    max_rdeg = isz;
    }
    return { c, max_rdeg };
}

// Also clears array
template<unsigned short VL>
uint8_t *
find_first_zero_vectorized( uint8_t * b, uint8_t * e ) {
    using tr = vector_type_traits_vl<uint8_t,VL>;
    using type = typename tr::type;
    using mtype = typename tr::mask_type;

    type z = tr::setzero();
    for( ; b+VL <= e; b += VL ) {
	mtype m = tr::cmpeq( tr::loadu( b ), z, target::mt_mask() );
	tr::storeu( b, z );
	if( m != 0 )
	    return b + _tzcnt_u64( m );
    }

    return b;
}

// Also clears array
VID find_first_zero( uint8_t * b, uint8_t * e ) {
    uint8_t * const ob = b;

    // Use a vectorized approach
#if __AVX512F__
    if( b+64 <= e )
	b = find_first_zero_vectorized<64>( b, e );
    if( b+32 <= e && *b != 0 ) 
	b = find_first_zero_vectorized<32>( b, e );
#elif __AVX2__
    if( b+32 <= e )
	b = find_first_zero_vectorized<32>( b, e );
#endif

    for( ; b != e && *b != 0; ++b ) {
	*b = 0;
    }

    return b - ob;
}

template<typename VID>
std::tuple<VID,std::vector<VID>>
count_colours_greedy( const HGraphTy & G, const PSet<VID> & xp,
		      VID exceed ) {
    // Upper bound, loose?
    // Example: if we have i neighbours in the PSet, we would deduce the need
    // for colour i, however, if some of those neighbours can have the same
    // colour, then we don't need colour i.
    VID n = G.numVertices();
    VID s = xp.size();
    std::vector<VID> colour( n );
    std::vector<uint8_t> histo( s, (uint8_t)0 );
    std::vector<VID> retain;
    retain.reserve( s/2 );
    VID c = 1; // number of colours in use
    VID max_col = 0;
    // for( VID j=0; j < s; ++j ) {
	// VID i = s-1 - j;
    for( VID i=0; i < s; ++i ) {
	VID v = xp.at( i );

	// Left neighbours only
	const auto & adj = G.get_left_neighbours_set( v );
	auto nb = adj.begin();
	auto ne = adj.end();

	// Intersect and check colours.
	// Filter left neighbours of v from [pb,pe)
	const VID * pb = xp.get_set();
	const VID * pe = xp.get_set() + i;
	if( ne != nb ) {
	    // Trimming
	    pb = std::lower_bound( pb, pe, *nb );
	    pe = std::upper_bound( pb, pe, *(ne-1) );
	    // Iterate
	    for( ; pb != pe; ++pb ) {
		if( adj.contains( *pb ) )
		    histo[colour[*pb]] = (uint8_t)1;
	    }
	}

	VID c = find_first_zero( &*histo.begin(), &*histo.end() );
	colour[v] = c;
	if( c >= exceed )
	    retain.push_back( v );
	if( c > max_col )
	    max_col = c;

	// We have already cleared array elements up to c
	std::fill( histo.begin() + c, histo.begin() + max_col + 1, (uint8_t)0 );
    }

    // Add one to the maximum colour in use as colours are numbered [0,max_col]
    // and thus the number of colours is max_col+1
    return { max_col+1, retain };
}


// XP may be modified by the method. It is not required to be in sort order.
template<bool allow_dense>
void
mc_bron_kerbosch_recpar_xps(
    const HGraphTy & G,
    VID degeneracy,
    MC_CutOutEnumerator & E,
    const clique_set<VID> * R,
    PSet<VID> & xp,
    int depth ) {

    // This code is expected to be called from bk_recursive_call().
    // Hence, it is expcted that ce != 0 and that depth + ce is a feasible
    // solution.
    VID ce = xp.size();
    const VID n = G.numVertices();

    if constexpr ( io_trace ) {
	std::lock_guard<std::mutex> guard( io_mux );
	std::cout << "XPS loop: ce=" << ce << " depth=" << depth << "\n";
    }

    // Instruct count_colours to abort procedure if a sufficient
    // number of colours is reached to make the set feasible.
    // Note trickiness with feasibility check as the max clique may have
    // improved since initiating the call to the colouring routine, which means
    // it may have stopped early and found a size that is insufficient.
    VID target = E.get_max_clique_size();
    // auto [ num_colours, max_rdeg ] = count_colours_ub( G, xp );
    auto [ num_colours, retain ] = count_colours_greedy( G, xp, target - depth );
    if( !E.is_feasible_bool( depth + num_colours > target, fr_colour_greedy ) )
	return;
    // if( !E.is_feasible( depth + 1 + max_rdeg, fr_rdeg ) )
    // return;

#if PIVOT_COLOUR
    VID skip_colours = target - depth;
#else
    VID pivot = mc_get_pivot( G, xp ).first;
    const auto & p_adj = G.get_neighbours_set( pivot );
#endif

#if PIVOT_COLOUR
    if( retain.size() == 1 ) {
	VID v = retain.front();

	// Add vertex v to running clique
	clique_set<VID> R_new( v, R );

	// Intersect candidates with neighbours of v and proceed
	PSet<VID> xp_new = xp.intersect( n, G.get_neighbours_set( v ) );
	bk_recursive_call<allow_dense>(
	    G, degeneracy, E, &R_new, xp_new, depth+1 );

	return;
    }

    for( VID i=0; i < retain.size(); ++i ) {
	VID v = retain[i];
#else
    for( VID i=0; i < xp.size(); ++i ) {
	VID v = xp.at( i );
#endif

#if PIVOT_COLOUR
	// Skip first few colour classes, as on their own they cannot
	// lead to an improvement of the clique. The lowest-numbered colour
	// classes should have the most vertices and the highest-degree
	// vertices.
	// if( colours[v] < skip_colours )
	// continue;
#else
	// Skip neighbours of pivot.
	// Could remove them explicitly, however, not needed in sequential
	// execution.
	if( p_adj.contains( v ) )
	    continue;
#endif

#if CLIQUER_PRUNE
	// Based on Cliquer.
	// Only works really when vertices in P have already been visited
	// at top level.
	if( max_clique_per_vertex[v] != 0
	    && !E.is_feasible( depth + max_clique_per_vertex[v], fr_cliquer ) )
	    break;
#endif
	
	// Add vertex v to running clique
	clique_set<VID> R_new( v, R );

	// Get neighbours of v
	const auto adj = G.get_neighbours_set( v ); 

	if constexpr ( io_trace ) {
	    std::lock_guard<std::mutex> guard( io_mux );
	    std::cout << "XP2: X=" << i << " P=" << (ce - (i+1)) << " adj="
		      << adj.size() << " depth=" << depth << "\n";
	}

	// Some complexity:
	// + Need to consider all vertices prior to v in XP are now
	//   in the X set. Could set ne to i, however:
	// + Vertices that are filtered due to pivoting,
	//   i.e., neighbours of pivot, are still in P.
	// + In sequential execution, we can update XP incrementally,
	//   however in parallel execution we cannot.
	// TODO: streamline with dual_set
	PSet<VID> xp_new = xp.intersect_validate( n, adj );
	bk_recursive_call<allow_dense>(
	    G, degeneracy, E, &R_new, xp_new, depth+1 );

	xp.invalidate( v );
    }
}

void
mc_bron_kerbosch_recpar_top_xps(
    const HGraphTy & G,
    VID degeneracy,
    MC_CutOutEnumerator & E,
    const clique_set<VID> * root,
    VID depth ) {
    const VID n = G.numVertices();

    // 1. find pivot, e.g., highest degree
    // VID pivot = 0; // presumed highest degree vertex
    VID pivot = get_max_degree_vertex( G ).first;
    const auto & p_adj = G.get_neighbours_set( pivot );

    // 2. create iteration set it = all vertices \ ngh(P)
    auto it = PSet<VID>::create_complement( n, p_adj );
    
    // 3. loop over elements it ; initial P is all vertices
    VID it_size = it.size();
    const VID * it_elm = it.get_set();
    for( VID i=0; i < it_size; ++i ) {
	VID v = it_elm[i];
	VID deg = G.getDegree( v ); 
	clique_set<VID> R_new( v, root );

	if( deg == 0 ) {
	    // avoid overheads of copying and cutout
	    // TODO: assume this never happens at top level due to
	    //       filtering before creating cutout.
	    E.record( 1, R_new.begin(), R_new.end() );
	} else {
	    auto adj = G.get_neighbours_set( v ); 
	    PSet<VID> xp_new = PSet<VID>::left_union_right( n, v, p_adj, adj );
	    bk_recursive_call<true>( G, degeneracy, E, &R_new, xp_new, depth+1 );
	}
    }
}

template<unsigned Bits, typename HGraph, typename Enumerator>
std::pair<VID,EID> mc_dense_fn(
    const HGraph & H,
    Enumerator & E,
    VID v,
    const graptor::graph::NeighbourCutOutDegeneracyOrderFiltered<VID,EID> & cut,
    const clique_set<VID> * root,
    VID depth,
    all_variant_statistics & stats ) {

    timer tm;
    tm.start();

    VID num = cut.get_num_vertices();
    size_t cl = get_size_class( num );

    // Build induced graph
    DenseMatrix<Bits,VID,EID>
	IG( H, H, cut.get_vertices(), 0, cut.get_num_vertices() );

    VID n = IG.numVertices();
    VID m = IG.get_num_edges();
    float d = (float)m / ( (float)n * (float)(n-1) );

    double tc = tm.next();
    algo_variant av = av_bk;
    bool do_vc = false, do_bk = false;
    double t_vc = 0, t_bk = 0;

    VID bc = E.get_max_clique_size();
    VID k_max = n < bc ? 0 : n - bc + 1;

#if TOP_DENSE_SELECT == 0
#if !ABLATION_DISABLE_VC && !ABLATION_DISABLE_BK
    if( d > density_threshold ) {
	do_vc = true;
	av = av_vc;
    } else {
	do_bk = true;
	av = av_bk;
    }
#elif !ABLATION_DISABLE_BK
    do_bk = true;
    av = av_bk;
#else
    do_vc = true;
    av = av_vc;
#endif
#else
    // by predictor
    float iv_n = float(n) / 512.0f;
    float iv_k = float(k_max) / float(n);
    float tp_bk = algo_predictor[0].predict( iv_n, iv_k, d );
    float tp_vc = algo_predictor[1].predict( iv_n, iv_k, d );
    if( tp_bk < tp_vc ) {
	av = av_bk;
	do_bk = true;
    } else {
	av = av_vc;
	do_vc = true;
    }
#if TOP_DENSE_SELECT == 1
    // profile both
    do_vc = do_bk = true;
#endif
#endif

    if( do_vc ) {
#if TOP_DENSE_SELECT == 0
	if( verbose > 2 )
	    std::cout << "top-level dense VC: v=" << v
		      << " cut=" << n << " density=" << d
		      << " k_max=" << k_max << "\n";
#endif
    
	MC_DenseEnumerator DE( E, root );
	auto bs = IG.clique_via_vertex_cover( k_max );
	DE.record( depth + bs.size(), bs.begin(), bs.end() );
	t_vc = tm.next();
    }

    if( do_bk ) {
#if TOP_DENSE_SELECT == 0
	if( verbose > 2 )
	    std::cout << "top-level dense BK: v=" << v << " cut=" << n
		      << " density=" << d << "\n";
#endif
    
	MC_DenseEnumerator DE( E, root );
	IG.mc_search( DE, depth );
	t_bk = tm.next();
    }

#if TOP_DENSE_SELECT > 0
    if( verbose > 2 ) {
	    std::cout << "top-level dense: v=" << v
		      << " cut=" << n << " density=" << d
		      << " k_max=" << k_max
		      << " VC=" << t_vc
		      << " BK=" << t_bk
		      << " pred=" << ( av == av_bk ? "BK" : "VC" )
		      << " pred_VC=" << tp_vc
		      << " pred_BK=" << tp_bk
		      << "\n";
    }
    // Update predictor
    if( do_vc )
	algo_predictor[1].update( t_vc, iv_n, iv_k, d );
    if( do_bk )
	algo_predictor[0].update( t_bk, iv_n, iv_k, d );
#endif

    variant_statistics & s = stats.get( av, cl );
    s.record_build( tc );
    s.record( av == av_bk ? t_bk : t_vc );

    return std::make_pair( n, m );
}

typedef std::pair<VID,EID> (*mc_func)(
    const HFGraphTy &,
    MC_CutOutEnumerator &,
    VID,
    const graptor::graph::NeighbourCutOutDegeneracyOrderFiltered<VID,EID> & cut,
    const clique_set<VID> *,
    VID,
    all_variant_statistics & );
    
static mc_func mc_dense_func[N_DIM+1] = {
    &mc_dense_fn<32,HFGraphTy,MC_CutOutEnumerator>,  // N=32
    &mc_dense_fn<64,HFGraphTy,MC_CutOutEnumerator>,  // N=64
    &mc_dense_fn<128,HFGraphTy,MC_CutOutEnumerator>, // N=128
    &mc_dense_fn<256,HFGraphTy,MC_CutOutEnumerator>, // N=256
    &mc_dense_fn<512,HFGraphTy,MC_CutOutEnumerator>  // N=512
};

void mc_top_level_bk(
    const HFGraphTy & H,
    MC_Enumerator & E,
    VID v,
    VID degeneracy,
    float density,
    const VID * const remap_coreness,
    graptor::graph::NeighbourCutOutDegeneracyOrderFiltered<VID,EID> & cut,
    const clique_set<VID> * root,
    VID depth ) {

    all_variant_statistics & stats = mc_stats.get_statistics();

    if( verbose > 2 )
	std::cout << "top-level generic BK: v=" << v
		  << " cut=" << cut.get_num_vertices()
		  << " density=" << density << "\n";

    timer tm;
    tm.start();

    GraphBuilderInduced<HGraphTy> ibuilder( H, v, cut );
    const auto & HG = ibuilder.get_graph();

    stats.record_genbuild( av_bk, tm.next() );

    MC_CutOutEnumerator CE( E, v, cut.get_vertices() );
    mc_bron_kerbosch_recpar_top_xps( HG, degeneracy, CE, root, depth );

#if MEMOIZE_MC_PER_VERTEX
    max_clique_per_vertex[v] = CE.get_max_clique_size();
#endif

    stats.record_gen( av_bk, tm.next() );
}

template<typename GraphType, typename SeqType>
void validate_clique( const GraphType & G,
		      const SeqType & mc ) {
    VID sz = mc.size();
    bool ok = true;
    if( sz > 1 ) {
	VID v0 = *mc.begin();
	auto adj0 = G.get_neighbours_set( v0 );
	std::vector<VID> ins( adj0.size()+1 );
	std::copy( adj0.begin(), adj0.end(), ins.begin() );
	ins[adj0.size()] = v0;
	std::sort( ins.begin(), ins.end() );
	std::vector<VID> reconstruct( ins.size() );
	
	for( VID v : mc | std::views::drop(1) ) {
	    auto ins_slice = graptor::make_array_slice( ins );
	    VID * start = &*reconstruct.begin();
	    VID * end = graptor::set_operations<graptor::MC_intersect>::
		intersect_ds( ins_slice, G.get_neighbours_set( v ), start );
	    size_t insz = end - start;
	    if( insz+1 < sz ) { // add 1 as v is not in its neighbour list
		std::cout << "Validation of clique failed: vertex "
			  << v << " has " << insz
			  << " common neighbours with the clique\n";
		ok = false;
	    }
	    *end++ = v;
	    std::sort( start, end );
	    reconstruct.resize( end - start );
	    std::swap( ins, reconstruct );
	}
	if( ins.size() > sz ) {
	    std::cout << "Clique is not maximal. Can be expanded to "
		      << ins.size() << " vertices:";
	    for( VID v : ins )
		std::cout << ' ' << v;
	    std::cout << "\n";
	    ok = false;
	}
    }
    if( ok )
	std::cout << "Validation of maximal clique: OK\n";
    else
	std::cout << "Validation of maximal clique: FAIL\n";
}


void mc_top_level_vc(
    const HFGraphTy & H,
    MC_Enumerator & E,
    VID v,
    VID degeneracy,
    float density,
    const VID * const remap_coreness,
    graptor::graph::NeighbourCutOutDegeneracyOrderFiltered<VID,EID> & cut,
    const clique_set<VID> * root,
    VID depth ) {

    all_variant_statistics & stats = mc_stats.get_statistics();

    if( verbose > 2 )
	std::cout << "top-level generic VC: v=" << v
		  << " cut=" << cut.get_num_vertices()
		  << " density=" << density << "\n";
    
    timer tm;
    tm.start();

    stats.record_genbuild( av_vc, tm.next() );

    // Enumeration: vertices on root chain are also translated.
    MC_CutOutEnumerator PE( E, v, cut.get_vertices() );
    MC_DenseEnumerator CE( PE, root, nullptr );
#if VERTEX_COVER_COMPONENTS
    if( cut.get_num_vertices() < (VID(1) << 16) ) {
	GraphBuilderInducedComplement<graptor::graph::GraphCSx<uint16_t,uint32_t>>
	    cbuilder( H, cut.get_slice() );
	auto & CG = cbuilder.get_graph();
	clique_via_vc3_cc<uint16_t,uint32_t>( CG, v, degeneracy, CE, depth );
    } else {
	GraphBuilderInducedComplement<graptor::graph::GraphCSx<uint32_t,uint64_t>>
	    cbuilder( H, cut.get_slice() );
	auto & CG = cbuilder.get_graph();
	clique_via_vc3_cc<VID,EID>( CG, v, degeneracy, CE, depth );
    }
#else
#if VERTEX_COVER_ABSOLUTE
    constexpr bool use_exist = false;
#else
    constexpr bool use_exist = true;
#endif
    if( cut.get_num_vertices() < (VID(1) << 16) ) {
	GraphBuilderInducedComplement<graptor::graph::GraphCSxDepth<uint16_t,uint32_t>>
	    cbuilder( H, cut.get_slice() );
	auto & CG = cbuilder.get_graph();
	clique_via_vc3_mono<uint16_t,uint32_t,use_exist>( CG, v, degeneracy, CE, depth );
    } else {
	GraphBuilderInducedComplement<graptor::graph::GraphCSxDepth<uint32_t,uint64_t>>
	    cbuilder( H, cut.get_slice() );
	auto & CG = cbuilder.get_graph();
	clique_via_vc3_mono<VID,EID,use_exist>( CG, v, degeneracy, CE, depth );
    }
#endif

#if MEMOIZE_MC_PER_VERTEX
    max_clique_per_vertex[v] = CE.get_max_clique_size();
#endif

    stats.record_gen( av_vc, tm.next() );
}

std::pair<float,EID> induced_density(
    const HFGraphTy & H,
    graptor::array_slice<VID,VID> cut
    ) {
    VID n = cut.size();
    EID m = 0;
    for( VID u : cut ) {
	VID d = graptor::set_operations<graptor::MC_intersect>
	    ::intersect_size_ds( cut, H.get_neighbours_set( u ) );
	m += d;
    }

    return std::make_pair( float(m) / float(n) / float(n-1), m );
}
    

//! Largest and average right-hand neighbour list size observed
//  by mc_top_level_select
std::atomic<VID> g_largest_rhs, g_sum_rhs, g_count_rhs;

#if PROFILE_DENSITY
std::mutex g_density_mux;

graptor::descriptive_statistics<float,size_t> g_size_rhs_all,
    g_size_rhs, g_size_filtered;

graptor::descriptive_statistics<float,size_t> g_density_rhs_all,
    g_density_rhs, g_density_filtered;
#endif

template<int select>
std::pair<VID,EID> mc_top_level_select(
    const HFGraphTy & H,
    MC_Enumerator & E,
    VID v,
    VID degeneracy,
    const VID * const remap_coreness ) {

    all_variant_statistics & stats = mc_stats.get_statistics();

    timer tm;
    tm.start();

    VID best = E.get_max_clique_size();

#if ABLATION_FILTER_STEPS >= 0
    // No point analysing a vertex of too low degree
    if( remap_coreness[v] < best ) {
	stats.record_filter0( tm.next() );
	return std::make_pair( VID(0), EID(1) );
    }
#endif

    // Profiling: check largest right-hand neighbour list encountered.
    // Good performance is expected when this does not exceed the degeneracy
    // by much.
    auto v_radj = H.get_right_neighbours_set( v );
    g_sum_rhs.fetch_add( v_radj.size() );
    g_count_rhs.fetch_add( 1 );
    // g_largest_rhs.fetch_max( v_radj.size() ); -- c++26
    {
	size_t v_radj_sz = v_radj.size();
	VID lg = g_largest_rhs.load();
	if( v_radj_sz > lg ) {
	    while( !g_largest_rhs.compare_exchange_weak(
		       lg, v_radj_sz,
		       std::memory_order_release,
		       std::memory_order_relaxed ) ) { }
	}
    }

#if PROFILE_DENSITY
    float d_rhs = induced_density( H, v_radj.get_seq() ).first;
    if( !std::isnan( d_rhs ) ) {
	std::lock_guard<std::mutex> guard( g_density_mux );
	g_density_rhs_all.update( d_rhs );
	g_size_rhs_all.update( v_radj.size() );
    }
#endif

#if ABLATION_FILTER_STEPS >= 1
    // Filter out vertices where coreness in main graph < best.
    // With coreness == best, we can make a clique of size best+1 at best.
    // Cut-out constructed filters out left-neighbours.
    graptor::graph::NeighbourCutOutDegeneracyOrderFiltered<VID,EID>
	cut( v_radj, [&]( VID u ) { return remap_coreness[u] >= best; } );
#else
    // For purposes of analysis: take full neighbourhood.
    graptor::graph::NeighbourCutOutDegeneracyOrderFiltered<VID,EID>
	cut( v_radj, [&]( VID u ) { return true; } );
#endif

    stats.record_filter0( tm.next() );

    VID hn1 = cut.get_num_vertices();
    if( hn1 < best || hn1 == 0 )
	return std::make_pair( VID(0), EID(2) );

#if ABLATION_FILTER_STEPS >= 2
    // Make a second pass over the vertices in the cut-out and check
    // that their common neighbours with the cut-out is at least best
    // (using intersect-size-exceeds). If not, throw them out also.
    // Could adapt to abort intersection also if known that intersection
    // is larger than target (vertex is eligible), however, we use the
    // intersection size now to estimate the density of the cutout graph.
    // The estimate may be wrong (over-estimated) if some vertices are
    // removed during this filtering step. It is expected that the error
    // will be small as the removed vertices have a relatively low degree
    // and we expect that not many vertices will be removed.
    //
    // Based on the observation that most often the filtering proves that
    // there is no need to perform any detailed search, it would be better
    // to introduce a boolean gt primitive to reduce the time spent in
    // intersections than the exceed primitive does. The boolean gt primitive
    // can stop earlier than exceed as soon as it is established that the size
    // of the intersection is greater than the threshold. The price is that a
    // second set of intersections may need to be performed to determine the
    // density of the induced graph, however, this would only be necessary
    // for non-dense cases. Moreover, as the overall result of filtering is
    // that the threshold is not sufficiently met, the benefits may be
    // restricted to a small subset of the performed intersections.
    //
    // This filtering step is covered by Chang KDD'19, although it is performed
    // after creating a cutout in that paper.
    cut.filter( [&,best]( VID u ) {
	// Note: intersection size >= best-1 as the current vertex may be part
	// of the clique but is not a neighbour of itself.
	bool ge = graptor::set_operations<graptor::MC_intersect>
	    ::intersect_size_ge_ds(
		cut.get_slice(),
		H.get_lazy_neighbours_set( u ),
		best-1 ); // keep if intersection size >= best-1
	return ge;
    }, best );
    stats.record_filter1( tm.next() );

    // If size of cut-out graph is less than best, then there is no point
    // in analysing it, nor constructing cut-out. A cut-out of size S can
    // lead to a S+1 clique when including the top-level vertex.
    // Check for empty cut-out just in case best is zero. Strictly speaking,
    // should log 1-clique in case num == 0.
    VID hn2 = cut.get_num_vertices();
    if( hn2 < best || hn2 == 0 )
	return std::make_pair( VID(0), EID(3) );
#endif

    clique_set<VID> * root = nullptr;
    VID depth = 1;
    EID m_est = 0;
#if ABLATION_FILTER_STEPS >= 3
    // The previous filtering loop uses the size_ge primitive to maximise
    // performance under the expectation that the filtering loop is most
    // often successful at proving that the subproblem does not need to be
    // analysed. The next loop re-calculates the intersections in such a way
    // that we can estimate the density of the subproblem without cutting it
    // out. This loop additionally filters as well (only if the previous
    // filtering loop managed to filter out vertices), which is doubly
    // useful.
#if !ABLATION_DISABLE_CONNECTED_FILTERING
    // Try to decide upon vertices that must be included.
    std::vector<VID> selected;
    VID hn3;
    VID num = hn2;
    VID threshold = best;
    //  do
    {
	hn3 = cut.get_num_vertices();
	cut.filter( [&]( VID u ) {
	    // d > best-2 is impossible to meet when best == 0 or best == 1
	    // due to wrap-around of unsigned numbers.
	    // Adjust the threshold for it to make sense.
	    VID d = graptor::set_operations<graptor::MC_intersect>
		::intersect_size_gt_val_ds(
		    cut.get_slice(),
		    H.get_neighbours_set( u ),
		    threshold <= 2 ? VID(0)
		    : threshold-2 ); // exceed checks >, we need >= best-1

	    // Reduction rules based on degree
	    // Chang KDD'19
	    if( d == cut.get_num_vertices()-1 ) {
		// Is this vertex connected to all other vertices?
		// If so, adopt this vertex and reduce search problem
		selected.push_back( u );
		if( threshold > 0 )
		    --threshold;
		return false;
	    } /* else if( d == cut.get_num_vertices()-2 ) {
		// Either u or the vertex it is not connected to make up the
		// maximum clique for this subgraph. Can include either this
		// one or the other one (but the other one is unknown).
		// The other one *must* be removed too, otherwise we could
		// decide to include it too in case of a truss.
How to know the odd one out?
		selected.push_back( u );
		return false;
		} */

	    if( d+1 >= threshold )
		m_est += d;
	    return d+1 >= threshold;
	}, threshold );

	num = cut.get_num_vertices() + selected.size();
    } //  while( cut.get_num_vertices() < hn3 && num != 0 && num >= best );

    // if( !selected.empty() )
    // std::cout << "selected size: " << selected.size() << "\n";

    std::vector<clique_set<VID>> r_nodes;
    r_nodes.reserve( selected.size() );
    VID off = cut.get_num_vertices();
    for( VID v : selected ) {
	r_nodes.emplace_back( off, root ); // will be translated ...
	const_cast<VID *>( cut.get_vertices() )[off] = v; // off is valid index because v was removed
	root = &r_nodes.back();
	++depth;
	++off;
    }
#else
    cut.filter( [&,best]( VID u ) {
	// d > best-2 is impossible to meet when best == 0 or best == 1
	// due to wrap-around of unsigned numbers.
	// Adjust the threshold for it to make sense.
	VID d = graptor::set_operations<graptor::MC_intersect>
	    ::intersect_size_gt_val_ds(
		cut.get_slice(),
		H.get_neighbours_set( u ),
		best <= 2 ? VID(0)
		: best-2 ); // exceed checks >, we need >= best-1
	if( d+1 >= best )
	    m_est += d;
	return d+1 >= best;
    }, best );

    VID num = cut.get_num_vertices();
#endif
    stats.record_filter2( tm.next() );

    if( num < best || num == 0 )
	return std::make_pair( VID(0), EID(4) );
#else
    VID num = cut.get_num_vertices();
#endif

// #if FILTER_NEIGHBOUR_CLIQUE
    // Check for each vertex in the cutout if an MC is known for them.
    // If an MC is known for all neighbours, it provides us with an upper bound
    // on the maximum clique in the current cutout (see Cliquer).
    // If any vertex has an unknown maximum clique, then we cannot deduce
    // anything for the current cutout.
    // Although we could deduce that any vertex v has an MC in its RHS
    // neighbourhood of at most u-v+MC(u) for any of its RHS neighbours u.
    // This could be an improvement over the coreness as an upper bound.
    // However, if u and v have different coreness, then u-v will by far
    // exceed the degeneracy of the graph and the upper bound u-v+MC(u) will
    // not be useful. This trick seems to only really apply in sequential
    // execution where the MC for all RHS neighbours is known.
    //
    // Parallelism more generally causes also issues with initiating the
    // evaluation of certain vertices than can be provably if we wait for
    // other vertices to be evaluated.
    //
    // Consider a DAG such that any vertex can be evaluated only when its
    // RHS neighbours have been evaluated, limiting both the unnecessary
    // evaluation of cliques and the availability of RHS neighbours' MC UB.
// #endif 

    float d_ret = 0;
    
#if PROFILE_DENSITY
    float d_filtered = d_filtered = induced_density( H, cut.get_slice() ).first;
    d_ret = d_filtered;
    if( !std::isnan( d_rhs ) ) {
	std::lock_guard<std::mutex> guard( g_density_mux );
	g_density_rhs.update( d_rhs );
	g_density_filtered.update( d_filtered );

	g_size_rhs.update( v_radj.size() );
	g_size_filtered.update( cut.get_num_vertices() );
    }
#endif

    // Note prefix heuristic mentioned in Tomita & Kameda, J Glob Optim 2007,
    // allowing to deduce a clique in the lowest-degree vertices, under
    // certain circumstances. Only likely to improve the maximum clique known
    // in what appear to be rare circumstances, i.e., the cutout must be
    // substantially larger than the incumbent clique size, or, all vertices
    // in cutout have the same degree.
    // We could check if this heuristic helps here.
    // If it does help to improve incumbent clique size, we could then consider
    // to redo the filtering from the top (checking coreness).

#if !ABLATION_DISABLE_TOP_DENSE
    // Needs to be determined if it is useful to engage the dense cutout
    // sooner when not all filtering is done. This would be useful when the
    // last filtering step does not reduce the cutout size any further.
    VID nlg = get_size_class( num );

    if( nlg <= N_MAX_SIZE ) {
	MC_CutOutEnumerator CE( E, v, cut.get_vertices() );
	return mc_dense_func[nlg-N_MIN_SIZE]( H, CE, v, cut, root, depth, stats );
    }
#endif

    float d = float(m_est) / ( float(num) * float(num) );
    if( d_ret == 0 )
	d_ret = d;
    if constexpr ( select == 0 ) {
	if( d > density_threshold ) {
	    mc_top_level_vc( H, E, v, degeneracy, d, remap_coreness, cut,
			     root, depth );
	} else {
	    mc_top_level_bk( H, E, v, degeneracy, d, remap_coreness, cut,
			     root, depth );
	}
    } else if constexpr ( select == 1 ) {
	mc_top_level_bk( H, E, v, degeneracy, d, remap_coreness, cut,
			 root, depth );
    } else {
	mc_top_level_vc( H, E, v, degeneracy, d, remap_coreness, cut,
			 root, depth );
    }

    return std::make_pair( num, m_est );
}

#if PROFILE_INCUMBENT_SIZE != 0
static double incumbent_profiling_limit = 0.0;
static VID incumbent_profiling_degree = 0;
static VID incumbent_profiling_vertex = ~(VID)0;
#endif

void mc_top_level(
    const HFGraphTy & H,
    const GraphCSx & G,
    MC_Enumerator & E,
    VID v,
    VID degeneracy,
    const VID * const remap_coreness,
    const VID * const rev_order ) {
#if PROFILE_INCUMBENT_SIZE != 0
    if( H.get_right_neighbours_set( v ).size() < incumbent_profiling_degree )
	return;

    if( incumbent_profiling_vertex != ~(VID)0
	&& incumbent_profiling_vertex != v )
	return;
    
    VID deg = H.getDegree( v );
    std::vector<std::pair<double,std::pair<VID,EID>>> timings;
    timings.reserve( deg );
    std::vector<VID> empty;
    VID mc = 0;
    VID cnt = deg;
    for( VID is=0; is <= cnt; ++is ) {
	// Clear the enumerator (should execute sequentially for this to work)
	E.reset();

	// Register a clique of size is
	E.record( is, v, empty.begin(), empty.end() );

	assert( E.get_max_clique_size() == std::max( is, VID(1) ) );

	// Execute top-level code
	{
	    timer tm;
	    tm.start();
	    std::pair<VID,EID> nm = mc_top_level_select<PROFILE_INCUMBENT_SIZE>(
		H, E, v, degeneracy, remap_coreness );
	    timings.push_back( std::make_pair( tm.next(), nm ) );
	}

#if 0
	// Report maximum clique found
	E.report( std::cout );

	// Get clique this after reporting as the top-level
	// will now get sorted in line with the rest of the clique
	auto mcs = E.sort_and_get_max_clique();

	// Report on coreness of clique members
	std::cout << "clique coreness:";
	for( VID u : mcs )
	    std::cout << ' ' << remap_coreness[rev_order[u]];
	std::cout << "\n";

	validate_clique( G, mcs );
#endif

	// The actual maximum clique size for this vertex's subgraph
	if( is == 0 ) {
	    mc = E.get_max_clique_size();
	    cnt = std::min( cnt, 2*mc );
	} else {
	    if( E.get_max_clique_size() != std::max( mc, is ) )
		std::cout << v << ' ' << E.get_max_clique_size() << ' ' << mc << ' ' << is << "\n";
	    assert( E.get_max_clique_size() == std::max( mc, is ) );
	}
    }

    auto mx = std::max_element( timings.begin(), timings.end() )->first;
    if( mx >= incumbent_profiling_limit ) {
	std::cout << v << " deg=" << deg << " mc=" << mc;
	for( auto t : timings )
	    std::cout << ' ' << t.first;
	std::cout << "\n";
	std::cout << v << " cutout vertices";
	for( auto t : timings )
	    std::cout << ' ' << t.second.first;
	std::cout << "\n";
	std::cout << v << " cutout edges";
	for( auto t : timings )
	    std::cout << ' ' << t.second.second;
	std::cout << "\n";
	std::cout << v << " cutout density";
	for( auto t : timings ) {
	    double d = double(t.second.second) / double(t.second.first)
		/ double(t.second.first-1);
	    std::cout << ' ' << d;
	}
	std::cout << "\n";
    }

    // Make sure future filtering in outer loop is disabled
    E.reset();

#else
#if !ABLATION_DISABLE_VC && !ABLATION_DISABLE_BK
    mc_top_level_select<0>( H, E, v, degeneracy, remap_coreness );
#elif !ABLATION_DISABLE_BK
    mc_top_level_select<1>( H, E, v, degeneracy, remap_coreness );
#else
    mc_top_level_select<2>( H, E, v, degeneracy, remap_coreness );
#endif
#endif // PROFILE_INCUMBENT_SIZE
}

template<unsigned Bits, typename VID, typename EID>
void leaf_dense_fn(
    const HGraphTy & H,
    MC_CutOutEnumerator & E,
    const clique_set<VID> * R,
    const PSet<VID> & xp_set,
    size_t depth ) {
    timer tm;
    tm.start();

/*
    // Do a bit of extra filtering
    VID req_deg = E.get_max_clique_size() - depth;
    graptor::graph::NeighbourCutOutDegeneracyOrderFiltered<VID,EID> cut(
	xp_set.get_set(), xp_set.get_fill(),
	[&]( VID u ) {
	    VID d = graptor::set_operations<graptor::MC_intersect>
		::intersect_size_gt_val_ds(
		    xp_set.hash_set(),
		    H.get_neighbours_set( u ),
		    req_deg );
	    return d >= req_deg;
	} );

    if( cut.get_num_vertices() != xp_set.get_fill() ) {
	cut.filter( [&]( VID u ) {
	    VID d = graptor::set_operations<graptor::MC_intersect>
		::intersect_size_gt_val_ds(
		    cut.get_slice(),
		    H.get_neighbours_set( u ),
		    req_deg );
	    return d >= req_deg;
	}, req_deg );
    }
    DenseMatrix<Bits,VID,VID> D( H, H, cut.get_vertices(), 0,
				 cut.get_num_vertices() );
*/

    DenseMatrix<Bits,VID,VID> D( H, H, xp_set.get_set(), 0, xp_set.get_fill() );
    VID n = D.numVertices();
    VID m = D.get_num_edges();
    float d = (float)m / ( (float)n * (float)(n-1) );
    double tm_build = tm.next();

    // Maximum clique size is depth (size of R), maximum degree in D
    // (max number of plausible neighbours), +1 for the vertex whose neighbours
    // we are checking.
    if( !E.is_feasible( depth + D.get_max_degree() + 1, fr_maxdeg ) ) {
	return;
    }

    MC_DenseEnumerator DE( E, R, xp_set.get_set() ); // cut.get_vertices() );
    algo_variant av = av_bk;
#if !ABLATION_DISABLE_VC && !ABLATION_DISABLE_BK
    if( d > density_threshold ) {
	VID init_k = n - ( E.get_max_clique_size() - depth );
	auto bs = D.clique_via_vertex_cover( init_k );
	DE.record( depth + bs.size(), bs.begin(), bs.end() );
	av = av_vc;
    } else {
	D.mc_search( DE, depth );
    }
#elif !ABLATION_DISABLE_BK
    D.mc_search( DE, depth );
#else
    VID init_k = n - ( E.get_max_clique_size() - depth );
    auto bs = D.clique_via_vertex_cover( init_k );
    DE.record( depth + bs.size(), bs.begin(), bs.end() );
    av = av_vc;
#endif

    float tbk = tm.next();
    variant_statistics & stats
	= mc_stats.get_statistics().get_leaf( av, ilog2( Bits ) );
    stats.record_build( tm_build );
    stats.record( tbk );

    // std::cout << "leaf-dense<" << Bits << "> n_req=" << xp_set.get_fill()
    // << " n=" << n << " m=" << m << " d=" << d
    // << " delay=" << tbk << "\n";
}

typedef void (*mc_leaf_func)(
    const HGraphTy &,
    MC_CutOutEnumerator &,
    const clique_set<VID> *,
    const PSet<VID> &,
    size_t );
    
static mc_leaf_func leaf_dense_func[N_DIM+1] = {
    &leaf_dense_fn<32,VID,EID>,  // N=32
    &leaf_dense_fn<64,VID,EID>,  // N=64
    &leaf_dense_fn<128,VID,EID>, // N=128
    &leaf_dense_fn<256,VID,EID>, // N=256
    &leaf_dense_fn<512,VID,EID>  // N=512
};

template<typename VID, typename EID>
bool mc_leaf(
    const HGraphTy & H,
    MC_CutOutEnumerator & E,
    const clique_set<VID> * R,
    const PSet<VID> & xp_set,
    size_t depth ) {
#if ABLATION_DISABLE_LEAF
    return false;
#else
    VID ce = xp_set.get_fill();
    VID num = ce;
    VID * XP = xp_set.get_set();

    if( ce < BK_MIN_LEAF )
	return false;

    VID nlg = get_size_class( num );
    if( nlg < N_MIN_SIZE )
	nlg = N_MIN_SIZE;

    if( nlg <= N_MAX_SIZE ) {
	leaf_dense_func[nlg-N_MIN_SIZE]( H, E, R, xp_set, depth );
	return true;
    }

    return false;
#endif // ABLATION_DISABLE_LEAF
}

// Suffix heuristic mentioned in Chang KDD'19. Does not bring much joy here.
// Prefix heuristic mentioned in Tomita & Kameda, J Glob Optim 2007:
// If R_min is the set of vertices with lowest degree (and thus first vertices
// in degeneracy order), and for all p in R_min: deg(p)=|R_min|-1, then these
// vertices much form a clique. This is only useful really in the absence of
// low-degree vertices, i.e., after significant filtering...
template<typename GraphType>
void heuristic_suffix(
    const GraphType & H,
    MC_Enumerator & E,
    const VID * const coreness ) {

    timer tm;
    tm.start();

    all_variant_statistics & stats = mc_stats.get_statistics();

    VID n = H.get_num_vertices();

    const auto & adj = H.get_neighbours_set( n-1 );
    if( adj.size() == 0 )
	return;

    std::vector<VID> pset;
    pset.reserve( adj.size() );

    // Only interested in vertices that can make a clique larger than th
    // th is likely 0 or 1
    VID th = E.get_max_clique_size();
    for( auto u : adj )
	if( coreness[u] >= th ) 
	    pset.push_back( u );

    std:vector<VID> clique;
    clique.push_back( n-1 );
    
    for( VID v=n-2; v > 0; --v ) {
	const auto & adj = H.get_neighbours_set( v );

	std::vector<VID> ins( std::min( pset.size(), adj.size() ) );

	graptor::array_slice<VID,size_t> s( &pset[0], &pset[pset.size()-1] );
	VID * out = &ins[0];
	size_t sz =
	    graptor::set_operations<graptor::MC_intersect>::intersect_ds(
		s, adj, out ) - out;
	ins.resize( sz );

	std::swap( pset, ins );

	if( pset.size() > 0 )
	    clique.push_back( v );
	else
	    break;
    }

    E.record( clique.size(), n-1, clique.begin(), clique.end() );

    stats.record_heuristic( tm.stop() );
}


template<typename GraphType>
VID heuristic_expand(
    const GraphType & H,
    const clique_set<VID> * R,
    VID top_v,
    const VID * P_begin,
    const VID * P_end,
    MC_Enumerator & E,
    size_t depth ) {

    if( P_begin == P_end ) {
	E.record( depth, top_v, R->begin(), R->end() );
	return depth;
    }

    // Assume highest-core vertices have higher vertex ID, at end of list
    VID v = *(P_end-1);
    const auto & v_adj = H.get_neighbours_set( v );
    clique_set<VID> R_new( v, R );
    graptor::array_slice<VID,size_t> s( P_begin, P_end-1 );

    // Perform intersection in such a way that we abort the intersection
    // operation if it will not lead to an improvement in maximum clique size.
    std::vector<VID> ins( s.size() );
    VID * out = &ins[0];
    size_t cur_size = E.get_max_clique_size();
    size_t sz;
    if( cur_size <= depth )
	sz = graptor::set_operations<graptor::MC_intersect>::intersect_ds(
	    s, v_adj, out ) - out;
    else
	sz = graptor::set_operations<graptor::MC_intersect>::intersect_gt_ds(
	    s, v_adj, cur_size - depth, out ) - out;

    return heuristic_expand( H, &R_new, top_v, out, out+sz, E, depth+1 );
}

/*! Heuristic search method for maximum clique
 */
template<typename GraphType>
VID heuristic_search(
    const GraphType & H,
    MC_Enumerator & E,
    VID v,
    const VID * const coreness ) {

    timer tm;
    tm.start();

    all_variant_statistics & stats = mc_stats.get_statistics();

    VID deg = H.get_degree( v );
    if( deg == 0 )
	return 1;

    const VID * b = H.get_neighbours( v ); // force creation of sequential set
    const VID * e = b + deg;

    std::vector<VID> pset;
    pset.reserve( deg );

    // Only interested in vertices that can make a clique larger than th
    VID th = E.get_max_clique_size();
    for( ; b != e; ++b )
	if( coreness[*b] >= th ) 
	    pset.push_back( *b );

    // Heuristically expand a clique.
    VID mc =
	heuristic_expand( H, nullptr, v, &pset[0], &pset[pset.size()], E, 1 );

    stats.record_heuristic( tm.stop() );

    return mc;
}

/*! Main method for maximum clique search
 */
int main( int argc, char *argv[] ) {
    CommandLine P(
	argc, argv,
	"\t-s\t\tinput graph is symmetric\n"
	"\t-p\t\tapply pruning before reordering\n"
	"\t-H [012]\theuristic search (default 2)\n"
	"\t--suffix\theuristic search for suffix clique\n"
	"\t-v {level}\tverbosity level (default 0)\n"
	"\t-pre {vertex}\tpre-trial vertex heuristic search\n"
	"\t-what-if {size}\tassuming clique of given size exists\n"
	"\t--incumbent-limit {lim}\treport only timings larger than lim\n"
	"\t--incumbent-degree {deg}\tonly analyse vertices with degree deg or above\n"
	"\t--hash-threshold {threshold}\tthreshold for pre-constructing hashed neighbour sets\n"
	"\t-d {threshold}\tdensity threshold for applying vertex cover\n"
	"\t-i {file}\tinput file containing graph\n"
	"\t-h, --help\tprint help message and exit\n"
	);
    const bool symmetric = P.get_bool_option( "-s" );
    const bool early_pruning = P.get_bool_option( "-p" );
    const bool suffix_clique = P.get_bool_option( "--suffix" );
    const int heuristic = P.get_long_option( "-H", 2 );
    const VID pre = P.get_long_option( "-pre", -1 );
    const VID what_if = P.get_long_option( "-what-if", -1 );
    verbose = P.get_long_option( "-v", 0 );
    density_threshold = P.get_double_option( "-d", 0.5 );
    const char * ifile = P.get_string_option( "-i" );
    const VID hash_threshold = P.get_long_option( "--hash-threshold", 16 );

#if PROFILE_INCUMBENT_SIZE != 0
    incumbent_profiling_limit = P.get_double_option( "--incumbent-limit", 0.0 );
    incumbent_profiling_degree = P.get_long_option( "--incumbent-degree", 0 );
    incumbent_profiling_vertex = P.get_long_option( "--incumbent-vertex", ~(VID)0 );
#endif

    std::cout << "Options:"
	      << "\n\tABLATION_PDEG=" << ABLATION_PDEG
	      << "\n\tABLATION_DISABLE_LEAF=" << ABLATION_DISABLE_LEAF
	      << "\n\tABLATION_DISABLE_TOP_DENSE=" << ABLATION_DISABLE_TOP_DENSE
	      << "\n\tABLATION_HADJPA_DISABLE_XP_HASH="
	      << ABLATION_HADJPA_DISABLE_XP_HASH
	      << "\n\tABLATION_DENSE_DISABLE_XP_HASH="
	      << ABLATION_DENSE_DISABLE_XP_HASH
	      << "\n\tDENSE_THRESHOLD_SEQUENTIAL_BITS="
	      << DENSE_THRESHOLD_SEQUENTIAL_BITS
	      << "\n\tDENSE_THRESHOLD_SEQUENTIAL="
	      << DENSE_THRESHOLD_SEQUENTIAL
	      << "\n\tDENSE_THRESHOLD_DENSITY="
	      << DENSE_THRESHOLD_DENSITY
	      << "\n\tABLATION_DENSE_NO_PIVOT_TOP="
	      << ABLATION_DENSE_NO_PIVOT_TOP
	      << "\n\tABLATION_DENSE_PIVOT_FILTER="
	      <<  ABLATION_DENSE_PIVOT_FILTER
	      << "\n\tABLATION_DISABLE_VC=" << ABLATION_DISABLE_VC
	      << "\n\tABLATION_DISABLE_BK=" << ABLATION_DISABLE_BK
	      << "\n\tABLATION_FILTER_STEPS=" << ABLATION_FILTER_STEPS
	      << "\n\tABLATION_DISABLE_CONNECTED_FILTERING="
	      << ABLATION_DISABLE_CONNECTED_FILTERING
	      << "\n\tUSE_512_VECTOR=" <<  USE_512_VECTOR
	      << "\n\tINTERSECTION_TRIM=" << INTERSECTION_TRIM
	      << "\n\tINTERSECTION_ALGORITHM=" << INTERSECTION_ALGORITHM
	      << "\n\tMC_INTERSECTION_ALGORITHM=" << MC_INTERSECTION_ALGORITHM
	      << "\n\tABLATION_DISABLE_ADV_INTERSECT=" << ABLATION_DISABLE_ADV_INTERSECT
	      << "\n\tINTERSECT_ONE_SIDED=" << INTERSECT_ONE_SIDED
	      << "\n\tBK_MIN_LEAF=" << BK_MIN_LEAF
	      << "\n\tCLIQUER_PRUNE=" << CLIQUER_PRUNE
	      // << "\n\tFILTER_NEIGHBOUR_CLIQUE=" << FILTER_NEIGHBOUR_CLIQUE
	      << "\n\tPIVOT_COLOUR=" << PIVOT_COLOUR
	      << "\n\tPIVOT_COLOUR_DENSE=" << PIVOT_COLOUR_DENSE
	      << "\n\tVERTEX_COVER_COMPONENTS=" << VERTEX_COVER_COMPONENTS
	      << "\n\tSORT_ORDER=" << SORT_ORDER
	      << "\n\tTRAVERSAL_ORDER=" << TRAVERSAL_ORDER
	      << "\n\tPROFILE_INCUMBENT_SIZE=" << PROFILE_INCUMBENT_SIZE
	      << "\n\tPROFILE_DENSITY=" << PROFILE_DENSITY
	      << "\n\tVERTEX_COVER_ABSOLUTE=" << VERTEX_COVER_ABSOLUTE
	      << "\n\tTOP_DENSE_SELECT=" << TOP_DENSE_SELECT
	      << "\n\tHOPSCOTCH_HASHING=" << HOPSCOTCH_HASHING
	      << "\n\tABLATION_DISABLE_LAZY_HASHING="
	      << ABLATION_DISABLE_LAZY_HASHING
#ifdef LOAD_FACTOR
	      << "\n\tLOAD_FACTOR=" << LOAD_FACTOR
#else
	      << "\n\tLOAD_FACTOR=undef"
#endif
	      << "\n\tdensity_threshold=" << density_threshold
	      << '\n';
    
    system( "hostname" );
    system( "date" );

    timer tm;
    tm.start();

    GraphCSx G0( ifile, -1, symmetric );

    std::cout << "Reading graph: " << tm.next() << "\n";

    GraphCSx G = graptor::graph::remove_self_edges( G0, true );
    G0.del();
    std::cout << "Removed self-edges: " << tm.next() << "\n";

    // Reset timer as graph ingress has high variability, and comparator
    // frameworks don't measure ingress or tend to perform poorly on ingress.
    // Also exclude time removing self-edges. This is part of data set
    // preparation. Others may do this during graph ingress.
    tm = timer();
    tm.start();

    MC_Enumerator E( G );

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

    VID pn = n;
    EID pm = m;
    VID prune_th = ~(VID)0;

    if( early_pruning ) {
	// First explore highest-degree vertex, then visit all others.
	// This aims to increase the amount of pruning done.
	heuristic_search( G, E, dmax_v, G.getDegree() );
	
	// Explore all vertices in a greedy manner, finding one
	// clique per vertex. Traverse in natural order; we haven't
	// obtained any sort order yet.
	for( VID v=0; v < n; ++v ) {
	    if( v != dmax_v )
		heuristic_search( G, E, v, G.getDegree() );
	}
	std::cout << "early pruning: " << tm.next() << "\n";
    }

    // TODO: after coreness computation, could further trim down
    //       the set of vertices to retain based on their coreness
    //       and how it compares to the best known clique.
    //       However, coreness computation remains fairly expensives for
    //       some graphs.

    mm::buffer<VID> order( n, numa_allocation_interleaved() );
    mm::buffer<VID> rev_order( n, numa_allocation_interleaved() );
    mm::buffer<VID> remap_coreness( pn, numa_allocation_interleaved() );

    VID degeneracy;
    std::vector<VID> histo;

#if ABLATION_DISABLE_LAZY_REMAPPING
    GraphCSx R;
    std::tie( R, degeneracy, histo )
	= reorder_kcore<SORT_ORDER,true>(
	    G, order, rev_order, remap_coreness, P, prune_th, dmax_v );
    std::cout << "Constructing remap info and remap graph: "
	      << tm.next() << "\n";

    VID lazy_threshold = std::max( VID(64), degeneracy );
    HFGraphTy H( R, numa_allocation_interleaved(), lazy_threshold );
    std::cout << "Building hashed graph: " << tm.next() << "\n";
#else
    std::tie( degeneracy, histo )
	= reorder_kcore<SORT_ORDER,false>(
	    G, order, rev_order, remap_coreness, P, prune_th, dmax_v );
    std::cout << "Constructing remap info: " << tm.next() << "\n";

    const VID lazy_threshold = std::max( VID(64), degeneracy );
    HFGraphTy H( G, order.get(), rev_order.get(),
		 numa_allocation_interleaved(),
		 hash_threshold,
		 [&,lazy_threshold]( VID v ) {
		     return remap_coreness[v] >= lazy_threshold;
		 } );
    std::cout << "Building hashed graph: " << tm.next() << "\n";
#endif

    // Cleanup remapped graph now; we won't need it any more.
    // Note: keep graph G around for validation
    // R required for GraphLazyHashedAdj
    // R.del();

#if MEMOIZE_MC_PER_VERTEX
    max_clique_per_vertex = new VID[pn];
    std::fill( max_clique_per_vertex, max_clique_per_vertex+pn, VID(0) );  
#endif

    if( histo.size() != 0 )
	std::cout << "Sort order check:\n\thisto[0]=" << histo[0]
		  << " histo[1]=" << histo[1]
		  << " histo[degeneracy]=" << histo[degeneracy]
		  << "\n\tcoreness[0]=" << remap_coreness[0]
		  << " coreness[pn-1]=" << remap_coreness[pn-1]
		  << " coreness[histo[1]]=" << remap_coreness[histo[1]]
		  << " coreness[histo[degeneracy]]="
		  << remap_coreness[histo[degeneracy]]
		  << "\n\tdegree[0]=" << H.getDegree(0)
		  << " degree[pn-1]=" << H.getDegree(pn-1)
		  << " degree[histo[1]]=" << H.getDegree(histo[1])
		  << " degree[histo[degeneracy]]="
		  << H.getDegree(histo[degeneracy])
		  << "\n";

#if TOP_DENSE_SELECT > 0
    graptor::hoeffding_tree_config cfg( 100, 200, 100, 0.99, 20 );
    algo_predictor.reserve( 2 );
    algo_predictor.emplace_back( cfg );
    algo_predictor.emplace_back( cfg );
#endif

#if PAPI_REGION == 1 
    map_workers( [&]( uint32_t t ) {
	int ret = PAPI_hl_region_begin( "MC" );
	if( ret != PAPI_OK ) {
	    {
		static std::mutex mux;
		std::lock_guard<std::mutex> guard( mux );
		static bool once = true;
		if( once ) {
		    once = false;
		    std::cerr << "PAPI_ENOTRUN: " << PAPI_ENOTRUN << "\n";
		    std::cerr << "PAPI_ESYS: " << PAPI_ESYS << "\n";
		    std::cerr << "PAPI_EMISC: " << PAPI_EMISC << "\n";
		    std::cerr << "PAPI_ENOMEM: " << PAPI_ENOMEM << "\n";
		}
		std::cerr << "Error " << ret << " initialising PAPI on worker "
			  << t << "\n";
	    }
	    // exit(1);
	}
    } );
#endif

    E.rebase( degeneracy, order.get(), remap_coreness.get() );

    std::cout << "setup: " << tm.next() << std::endl;
    std::cout << "Start enumeration at " << tm.total() << std::endl;

    timer tm_search;
    tm_search.start();

    /*! Traversal orders
     * 1. SOTA: sort by decreasing degree, visit low to high degree.
     * 2. SOTA: sort by decreasing degeneracy, visit low to high degeneracy.
     * 3. sort by decreasing degeneracy, visit high to low degeneracy.
     *  + 1 and 2 tie in sort order with left/right order which determines
     *    the maximum size of the candidate set, which effects efficiency of
     *    search.
     *  + The best left/right order, assuming candidate sets are drawn from
     *    "the right", i.e., higher-numbered vertices, assumes vertices are
     *    sorted in order of decreasing degeneracy. We maintain this, but
     *    decouple the outer-loop (but not inner loop) traversal order.
     *  + We enhance this by skipping in the outer loop entire batches of
     *    of vertices with the same degeneracy if that degeneracy is
     *    insufficient to improve the maximum clique
     * 4. For each degeneracy level, from high to low, pick the vertex with
     *    the highest degree and evaluate this. Following on, perform 3.
     */

    if( pre != ~(VID)0 ) {
	VID v = pre;
	std::cout << "pre-trial vertex v=" << v
		  << " deg=" << H.getDegree( v )
		  << " rho=" << remap_coreness[v]
		  << "\n";
	mc_top_level( H, G, E, v, degeneracy, remap_coreness.get(),
		      rev_order.get());
	std::cout << "pre-trial: " << tm.next() << "\n";
    }

    if( what_if != ~(VID)0 ) {
	// For highest-degree vertex with maximum degeneracy, try setting
	// a strong precedent for the clique size
	std::vector<VID> empty;
	E.record( what_if, what_if, empty.begin(), empty.end() );
	VID v = 0;
	std::cout << "what-if clique=" << what_if << " vertex v=" << v
		  << " deg=" << H.getDegree( v )
		  << " rho=" << remap_coreness[v]
		  << "\n";
	mc_top_level( H, G, E, v, degeneracy, remap_coreness.get(),
		      rev_order.get() );

	// Successfully found a clique larger than postulated size?
	// If not, erase result.
	if( E.get_max_clique_size() <= what_if )
	    E.reset();

	std::cout << "what-if: " << tm.next() << "\n";
    }

    if( suffix_clique ) {
	heuristic_suffix( H, E, remap_coreness.get() );
	std::cout << "heuristic suffix: " << tm.next() << "\n";
    }
    
    if( heuristic == 1 ) {
	// Heuristic 1: explore all vertices in a greedy manner, finding one
	// clique per vertex. Traverse from high to low vertex number.
	// for( VID w=0; w < n; ++w ) {
	parallel_loop( (VID)0, (VID)n, (VID)1, [&]( VID w ) {
	    VID v = n - 1 - w;
	    if( E.is_feasible( remap_coreness[v], fr_outer ) )
		heuristic_search( H, E, v, remap_coreness.get() );
	} );
	std::cout << "heuristic 1: " << tm.next() << "\n";
    } else if( heuristic == 2 ) {
#if SORT_ORDER == 2 || SORT_ORDER >= 4
	// Heuristic 2: explore selected vertices, one per core number.
	// for( VID cc=0; cc <= degeneracy; ++cc ) {
	parallel_loop( (VID)0, degeneracy+1, (VID)1, [&]( VID cc ) {
	    VID c = degeneracy - cc;
	    VID c_up = histo[c+1];
	    VID c_lo = histo[c];
#if SORT_ORDER >= 6
	    std::swap( c_up, c_lo );
#endif
	    if( c_up != c_lo ) {
		VID v = c_lo;
		if( E.is_feasible( c+1, fr_outer ) )
		    heuristic_search( H, E, v, remap_coreness.get() );
	    }
	} );
	std::cout << "heuristic 2: " << tm.next() << "\n";
#else
	std::cerr << "Heuristic 2 not supported if sort order does not "
		  << "partition vertex range by equal degeneracy\n";
	return -1;
#endif
    }

#if PROFILE_INCUMBENT_SIZE != 0
    // Disable filtering 
    E.reset();
#endif

#if SORT_ORDER < 4
    /* 0. low to high degree order */
    /* 1. high to low degree order */
    /* 2. low to high degeneracy order */
    /* 3. high to low degeneracy order */
    // for( VID w=0; w < n; ++w ) {
    parallel_loop( (VID)0, (VID)n, (VID)1, [&]( VID w ) {
#if ( TRAVERSAL_ORDER & 1 ) == 0
	VID v = w;
#else
	VID v = n-1-w;
#endif
	if( E.is_feasible( remap_coreness[v]+1, fr_outer ) )
	    mc_top_level( H, G, E, v, degeneracy, remap_coreness.get(),
			  rev_order.get()  );
    } );

#elif ( SORT_ORDER == 2 || SORT_ORDER == 4 ) && TRAVERSAL_ORDER == 3
    // cleaned up version with a small improvement
    // degeneracy+1 iterations to deal with 1 vertex/degeneracy,
    // additional iteration allows execution of all vertices in parallel
    parallel_loop( (VID)0, (VID)degeneracy+2, (VID)1, [&]( VID cc ) {
	if( cc <= degeneracy ) {
	    VID c = cc;
	    VID c_up = histo[c+1];
	    VID c_lo = histo[c];
	    if( c_up != c_lo ) {
		VID v = c_lo;
		if( E.is_feasible( c+1, fr_outer ) )
		    mc_top_level( H, G, E, v, degeneracy, remap_coreness.get(),
				  rev_order.get()  );
	    }
	} else {
	    // Downwards traversal over all coreness levels
	    for( VID cc_i=0; cc_i < degeneracy+1; ++cc_i ) {
		VID c = degeneracy - cc_i;

		VID c_up = histo[c+1];
		VID c_lo = histo[c];

		if( c_up <= c_lo+1 )
		    continue; // 0 or 1 vertices; go to next c value
		++c_lo; // already did c_lo in preamble
		if( !E.is_feasible( c+1, fr_outer ) )
		    break; // decreasing degeneracy, no hope for better

		parallel_loop( c_lo, c_up, (VID)1, [&,c,degeneracy](
				   VID w ) {
		    VID v = c_up - ( w - c_lo ) - 1;

		    if( E.is_feasible( c+1, fr_outer ) )
			mc_top_level( H, G, E, v, degeneracy,
				      remap_coreness.get(), rev_order.get() );
		} );
	    }
	}
    } );

#elif SORT_ORDER == 2 || ( SORT_ORDER >= 4 && SORT_ORDER <= 7 )
    /* 4. first evaluate highest-degree vertex per degeneracy level, then
     *    iterate by decreasing coreness, increasing degree. */
    /* 5. first evaluate highest-degree vertex per degeneracy level, then
     *    iterate by decreasing coreness, decreasing degree. */
    // for( VID cc=0; cc <= degeneracy; ++cc ) {
    parallel_loop( (VID)0, (VID)degeneracy+1, (VID)1, [&]( VID cc ) {
#if ( TRAVERSAL_ORDER & 1 ) == 0
	VID c = cc;
#else
	VID c = degeneracy - cc;
#endif
	VID c_up = histo[c+1];
	VID c_lo = histo[c];
#if SORT_ORDER >= 6
	std::swap( c_up, c_lo );
#endif
	if( c_up != c_lo ) {
	    VID v = c_lo;
	    if( E.is_feasible( c+1, fr_outer ) )
		mc_top_level( H, G, E, v, degeneracy, remap_coreness.get(),
			      rev_order.get()  );
	}
    } );
    std::cout << "phase 1 (one vertex per degeneracy): " << tm.next() << "\n";
	    
    // Use a parallel loop to reduce code duplication.
    // If TRAVERSAL_ORDER & 8 == 0 then only one iteration of the loop
    // performs actual work.
    parallel_loop( (VID)0, (VID)2, (VID)1, [&,degeneracy]( VID cc_half ) {
#if ( TRAVERSAL_ORDER & 8 ) == 8
	VID cc_mid = ( degeneracy + 1 ) / 2; // both up and downwards
#elif ( TRAVERSAL_ORDER & 1 ) == 1
	VID cc_mid = 0; // only downwards direction
#else
	VID cc_mid = degeneracy + 1; // only upwards direction
#endif
	VID cc_b = cc_half == 0 ? 0 : cc_mid;
	VID cc_e = cc_half == 0 ? cc_mid : degeneracy+1;
	
	for( VID cc_i=cc_b; cc_i < cc_e; ++cc_i ) {
	    VID c = cc_half == 0 ? cc_i : degeneracy - ( cc_i - cc_mid );

	    VID c_up = histo[c+1];
	    VID c_lo = histo[c];
#if SORT_ORDER >= 6
	    std::swap( c_up, c_lo );
#endif

	    if( c_up == c_lo || c_up == c_lo+1 )
		continue; // go to next c value
	    ++c_lo; // already did c_lo in preamble
	    if( !E.is_feasible( c+1, fr_outer ) ) {
		if( cc_half == 0 )
		    continue; // increasing degeneracy, more to come
		else
		    break; // decreasing degeneracy, no hope for better
	    }

	    parallel_loop( (VID)0, (VID)2, (VID)1, [&,c_lo,c_up,c,degeneracy](
			       VID c_half ) {
#if ( TRAVERSAL_ORDER & 4 ) == 4
		VID c_mid = ( c_lo + c_up ) / 2; // both up and downwards
#elif ( TRAVERSAL_ORDER & 2 ) == 2
		VID c_mid = c_lo; // only downwards direction
#else
		VID c_mid = c_up; // only upwards direction
#endif
		VID c_b = c_half == 0 ? c_lo : c_mid;
		VID c_e = c_half == 0 ? c_mid : c_up;

		parallel_loop( c_b, c_e, (VID)1, [&,c,degeneracy,c_mid](
				   VID w ) {
		    VID v = c_half == 0 ? w : ( c_up - ( w - c_mid ) - 1 );

		    if( E.is_feasible( c+1, fr_outer ) )
			mc_top_level( H, G, E, v, degeneracy,
				      remap_coreness.get(), rev_order.get() );
		} );
	    } );
	}
    } );
#else
#error "SORT_ORDER must be in range [0,7]"
#endif

#if PAPI_REGION == 1
    map_workers( [&]( uint32_t t ) {
	if( PAPI_OK != PAPI_hl_region_end( "MC" ) ) {
	    std::cerr << "Error ending PAPI\n";
	    exit(1);
	}
    } );
#endif

    std::cout << "Enumeration: " << tm.next() << " seconds\n";
    std::cout << "Completed search in " << tm_search.next() << " seconds\n";
    std::cout << "Completed MC in " << tm.total() << " seconds\n";

    // std::string what = "numastat -vmp ";
    // what += argv[0];
    // system( what.c_str() );

    all_variant_statistics stats = mc_stats.sum();

    for( size_t n=N_MIN_SIZE; n <= N_MAX_SIZE; ++n ) {
	std::cout << (1<<n) << "-bit dense BK: ";
	stats.get( av_bk, n ).print( std::cout ); 
	std::cout << (1<<n) << "-bit dense VC: ";
	stats.get( av_vc, n ).print( std::cout ); 
    }
    for( size_t n=N_MIN_SIZE; n <= N_MAX_SIZE; ++n ) {
	std::cout << (1<<n) << "-bit dense leaf BK: ";
	stats.get_leaf( av_bk, n ).print( std::cout ); 
	std::cout << (1<<n) << "-bit dense leaf VC: ";
	stats.get_leaf( av_vc, n ).print( std::cout ); 
    }
    std::cout << "generic BK: ";
    stats.m_gen[av_bk].print( std::cout );
    std::cout << "generic VC: ";
    stats.m_gen[av_vc].print( std::cout );
    std::cout << "filter0: ";
    stats.m_filter0.print( std::cout );
    std::cout << "filter1: ";
    stats.m_filter1.print( std::cout );
    std::cout << "filter2: ";
    stats.m_filter2.print( std::cout );
    std::cout << "heuristic: ";
    stats.m_heuristic.print( std::cout );

    std::cout << "RHS neighbour list statistics:"
	      << "\nRHS largest: " << g_largest_rhs.load()
	      << "\nRHS sum: " << g_sum_rhs.load() 
	      << "\nRHS count: " << g_count_rhs.load() 
	      << "\nRHS average: "
	      << ( (float)g_sum_rhs.load() / (float)g_count_rhs.load() )
	      << "\n";

#if PROFILE_DENSITY
    std::cout << "Size of top-level cutouts:\nAll cutouts (RHS): ";
    g_size_rhs_all.show( std::cout );
    std::cout << "\nAll retained cutouts before filtering: ";
    g_size_rhs.show( std::cout );
    std::cout << "\nAll retained cutouts after filtering: ";
    g_size_filtered.show( std::cout );
    std::cout << "\nDensity of top-level cutouts:\nAll cutouts (RHS): ";
    g_density_rhs_all.show( std::cout );
    std::cout << "\nAll retained cutouts before filtering: ";
    g_density_rhs.show( std::cout );
    std::cout << "\nAll retained cutouts after filtering: ";
    g_density_filtered.show( std::cout );
    std::cout << "\n";
#endif

    // Report maximum clique found
    E.report( std::cout );

    // Get clique this after reporting as the top-level
    // will now get sorted in line with the rest of the clique
    auto mc = E.sort_and_get_max_clique();

    // Report on coreness of clique members
    std::cout << "clique coreness:";
    for( VID v : mc )
	std::cout << ' ' << remap_coreness[rev_order[v]];
    std::cout << "\n";

    // Validate clique
    validate_clique( G, mc );

#if !ABLATION_DISABLE_LAZY_REMAPPING
    if constexpr ( lazy_hashing ) {
	VID num_init_h = 0;
	VID num_init_s = 0;
	for( VID v=0; v < n; ++v ) {
	    if( H.is_hash_set_initialised( v ) )
		++num_init_h;
	    if( H.is_seq_initialised( v ) )
		++num_init_s;
	}
	std::cerr << "Lazy initialisation: " << num_init_h << " / "
		  << n << " hash sets initialised\n";
	std::cerr << "Lazy initialisation: " << num_init_s << " / "
		  << n << " sequential sets initialised\n";
    }
#endif

#if ABLATION_DISABLE_LAZY_REMAPPING
    R.del();
#endif
    G.del();
    remap_coreness.del();
    rev_order.del();
    order.del();

#if MEMOIZE_MC_PER_VERTEX
    delete[] max_clique_per_vertex;
    max_clique_per_vertex = nullptr;
#endif

    return 0;
}
