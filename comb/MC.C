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

#ifndef ABLATION_DENSE_DISABLE_XP_HASH
#define ABLATION_DENSE_DISABLE_XP_HASH 0
#endif

#ifndef ABLATION_BITCONSTRUCT_XP_VEC
#define ABLATION_BITCONSTRUCT_XP_VEC 0
#endif

#ifndef ABLATION_DISABLE_LEAF
#define ABLATION_DISABLE_LEAF 0
#endif

#ifndef ABLATION_DISABLE_TOP_TINY
#define ABLATION_DISABLE_TOP_TINY 1
#endif

#ifndef ABLATION_DISABLE_TOP_DENSE
#define ABLATION_DISABLE_TOP_DENSE 0
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

#ifndef USE_512_VECTOR
#if __AVX512F__
#define USE_512_VECTOR 1
#else
// #define USE_512_VECTOR 0
#define USE_512_VECTOR 1
#endif
#endif

#ifndef OUTER_ORDER
#define OUTER_ORDER 4
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
#include "graptor/graph/simple/xp_set.h"
#include "graptor/graph/transform/rmself.h"

#include "graptor/container/bitset.h"
#include "graptor/container/hash_fn.h"
#include "graptor/container/intersect.h"
#include "graptor/container/transform_iterator.h"
#include "graptor/container/concatenate_iterator.h"

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

#if ABLATION_HADJPA_DISABLE_XP_HASH
using HGraphTy = graptor::graph::GraphHAdjPA<VID,EID,false,hash_fn>;
#else
using HGraphTy = graptor::graph::GraphHAdjPA<VID,EID,true,hash_fn>;
#endif
using HFGraphTy = graptor::graph::GraphHAdjPA<VID,EID,true,hash_fn>;

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

static bool verbose = false;

static std::mutex io_mux;
static constexpr bool io_trace = false;

enum filter_reason {
    fr_pset = 0,
    fr_colour_ub = 1,
    fr_colour_greedy = 2,
    fr_rdeg = 3,
    fr_maxdeg = 4,
    fr_unknown = 5,
    filter_reason_num = 6
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
    MC_Enumerator( size_t degen, const VID * const order )
	: m_degeneracy( degen ),
	  m_best( degen > 0 ? 1 : 0 ),
	  m_order( order ) {
	m_timer.start();
    }

    void reset() {
	m_best = m_degeneracy > 0 ? 1 : 0;
	m_max_clique.clear();
    }

    // Record solution
    template<typename It>
    void record( size_t s, VID top_v, It && begin, It && end ) {
	assert( s <= m_degeneracy+1 );
	if( s > m_best )
	    update_best( s, top_v, begin, end );
    }

    // Feasability check
    bool is_feasible( size_t upper_bound, filter_reason r = fr_unknown ) {
	if( upper_bound > m_best.load( std::memory_order_relaxed ) )
	    return true;
	else {
	    ++m_reason[(int)r];
	    return false;
	}
    }

    size_t get_max_clique_size() const {
	return m_best.load( std::memory_order_relaxed );
    }

    // Modifies m_max_clique to adjust sort order
    std::ostream & report( std::ostream & os ) {
	os << "Maximum clique size: " << m_best.load()
	   << " from top-vertex " << m_max_clique[0]
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

	return os;
    }

private:
    template<typename It>
    void update_best( size_t s, VID top_v, It && begin, It && end ) {
	size_t prior = m_best.load( std::memory_order_relaxed );
	while( s > prior )  {
	    if( m_best.compare_exchange_weak(
		    prior, s, 
		    std::memory_order_release,
		    std::memory_order_relaxed ) ) {
		std::cout << "max_clique: " << s << " at "
			  << m_timer.elapsed()
			  << " top-level vertex: " << top_v
			  << '\n';

		// TODO: concurrency concerns!
		m_max_clique.clear();
		m_max_clique.push_back( m_order[top_v] );
		for( It b=begin; b != end; ++b )
		    m_max_clique.push_back( m_order[*b] );

		break;
	    }
	    prior = m_best.load( std::memory_order_relaxed );
	}
    }

private:
    size_t m_degeneracy;
    std::atomic<size_t> m_best;
    std::array<std::atomic<size_t>,filter_reason_num> m_reason;
    timer m_timer;
    const VID * const m_order; //!< to translate IDs back to input file IDs
    std::vector<VID> m_max_clique; //!< max clique contents, not thread-safe
};

class MC_CutOutEnumerator {
public:
    MC_CutOutEnumerator( MC_Enumerator & E, VID top_v, const VID * const order )
	: m_E( E ), m_top_vertex( top_v ), m_order( order ) { }

    // Record solution
    template<typename It>
    void record( size_t s, It && begin, It && end ) {
	auto fn = [&]( VID v ) { return m_order[v]; };
	m_E.record( s, m_top_vertex,
		    graptor::make_transform_iterator( begin, fn ),
		    graptor::make_transform_iterator( end, fn ) );
    }

    // Feasability check
    bool is_feasible( size_t upper_bound, filter_reason r = fr_unknown ) {
	return m_E.is_feasible( upper_bound, r );
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
    }

    // Feasability check
    bool is_feasible( size_t upper_bound, filter_reason r = fr_unknown ) {
	return m_E.is_feasible( upper_bound, r );
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
	sum.m_tiny = m_tiny + s.m_tiny;
	return sum;
    }

    void record_tiny( double atm ) { m_tiny.record( atm ); }
    void record_gen( algo_variant var, double atm ) {
	m_gen[(size_t)var].record( atm );
    }
    void record_genbuild( algo_variant var, double atm ) {
	m_gen[(size_t)var].record_build( atm );
    }

    variant_statistics & get( algo_variant var, size_t n ) {
	return m_dense[(size_t)var][n-N_MIN_SIZE];
    }
    variant_statistics & get_leaf( algo_variant var, size_t n ) {
	return m_leaf_dense[(size_t)var][n-N_MIN_SIZE];
    }
    
    variant_statistics m_dense[N_VARIANTS][N_DIM];
    variant_statistics m_leaf_dense[N_VARIANTS][N_DIM];
    variant_statistics m_tiny, m_gen[N_VARIANTS];

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
#if 0
template<typename HGraph>
void
mc_tiny(
    const HGraph & H,
    const VID * const ngh,
    const VID start_pos,
    const VID num,
    const VID v,
    MC_Enumerator & E ) {
    if( num == 0 ) {
	E.record( 0, v );
    } else if( num == 1 ) {
	if( start_pos == 0 )
	    E.record( 2, v ); // v, 0
    } else if( num == 2 ) {
	bool n01 = H.get_adjacency( ngh[0] ).contains( ngh[1] );
	if( start_pos == 0 ) {
	    if( n01 ) // Two neighbours of v are neighbours themselves
		E.record( 3, v ); // v, 0, 1
	    else {
		E.record( 2, v ); // v, 0
		E.record( 2, v ); // v, 1
	    }
	} else if( start_pos == 1 ) {
	    // No maximal clique in case start_pos == 1
	    if( !n01 ) // triangle v, 0, 1 does not exist
		E.record( 2, v ); // v, 1
	}
    } else if( num == 3 ) {
	int n01 = H.get_adjacency( ngh[0] ).contains( ngh[1] ) ? 1 : 0;
	int n02 = H.get_adjacency( ngh[0] ).contains( ngh[2] ) ? 1 : 0;
	int n12 = H.get_adjacency( ngh[1] ).contains( ngh[2] ) ? 1 : 0;
	if( start_pos == 0 ) {
	    // v < ngh[0] < ngh[1] < ngh[2]
	    if( n01 + n02 + n12 == 3 )
		E.record( 4, v );
	    else if( n01 + n02 + n12 == 2 ) {
		E.record( 3, v );
		E.record( 3, v );
	    } else if( n01 + n02 + n12 == 1 ) {
		E.record( 3, v );
		E.record( 2, v );
	    } else if( n01 + n02 + n12 == 0 ) {
		E.record( 2, v );
		E.record( 2, v );
		E.record( 2, v );
	    }
	} else if( start_pos == 1 ) {
	    // ngh[0] < v < ngh[1] < ngh[2]
	    if( n01 + n02 + n12 == 3 )
		; // duplicate
	    else if( n01 + n02 + n12 == 2 ) { // wedge
		if( n12 == 1 )
		    E.record( 3, v );
	    } else if( n01 + n02 + n12 == 1 ) {
		if( n12 == 1 )
		    E.record( 3, v ); // v, 1, 2
		else if( n01 == 1 )
		    E.record( 2, v ); // v, 2
		else if( n02 == 1 )
		    E.record( 2, v ); // v, 1
	    } else if( n01 + n02 + n12 == 0 ) {
		E.record( 2, v ); // v, 1
		E.record( 2, v ); // v, 2
	    }
	} else if( start_pos == 2 ) {
	    // ngh[0] < ngh[1] < v < ngh[2]
	    if( n02 + n12 == 0 ) // not part of a triangle or more
		E.record( 2, v );
	}
    }
}    
#endif

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
	    if( verbose )
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
    template<typename HGraph>
    GraphBuilderInduced(
	const GraphCSx & G,
	const HGraph & H,
	VID v,
	const graptor::graph::NeighbourCutOutDegeneracyOrderFiltered<VID,EID> & cut )
	: S( G, H, cut.get_vertices(), cut.get_num_vertices(),
	     numa_allocation_interleaved() ),
	  start_pos( 0 ) { }

    const auto & get_graph() const { return S; }
    VID get_start_pos() const { return start_pos; }

private:
    graptor::graph::GraphHAdjPA<VID,EID,dual_rep,Hash> S;
    VID start_pos;
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
template<bool exists>
bool
vertex_cover_vc3( graptor::graph::GraphDoubleIndexCSx<VID,EID> & G,
		  VID k,
		  VID c,
		  VID & best_size,
		  VID * best_cover );

void
mark( VID & best_size, VID * best_cover, VID v ) {
    best_cover[best_size++] = v;
}

// For path or cycle
void
trace_path( const graptor::graph::GraphDoubleIndexCSx<VID,EID> & G,
	    bool * visited,
	    VID & best_size,
	    VID * best_cover,
	    VID cur,
	    VID nxt,
	    bool incl ) {
    if( visited[nxt] )
	return;

    visited[nxt] = true;

    if( incl )
	mark( best_size, best_cover, nxt );

    const EID * const index = G.getBeginIndex();
    const VID * const edges = G.getEdges();

    // Done if nxt is degree-1 vertex
    if( G.getDegree( nxt ) == 2 ) {
	VID ngh1 = edges[index[nxt]];
	VID ngh2 = edges[index[nxt]+1];

	VID ngh = ngh1 == cur ? ngh2 : ngh1;

	trace_path( G, visited, best_size, best_cover, nxt, ngh, !incl );
    }
}

bool
vertex_cover_poly( const graptor::graph::GraphDoubleIndexCSx<VID,EID> & G,
		   VID k,
		   VID & best_size,
		   VID * best_cover ) {
    VID n = G.numVertices();

    std::vector<char> visited( n, false );

    VID old_best_size = best_size;

    // Find paths
    for( VID v=0; v < n; ++v ) {
	VID deg = G.getDegree( v );
	assert( deg <= 2 );
	if( deg == 1 && !visited[v] ) {
	    visited[v] = true;
	    trace_path( G, (bool*)&visited[0], best_size, best_cover,
			v, *G.nbegin( v ), true );
	}
    }
    
    // Find cycles (uses same auxiliary as paths)
    for( VID v=0; v < n; ++v ) {
	VID deg = G.getDegree( v );
	assert( deg <= 2 );
	if( deg == 2 && !visited[v] ) {
	    visited[v] = true;
	    mark( best_size, best_cover, v );
	    trace_path( G, (bool*)&visited[0], best_size, best_cover,
			v, *G.nbegin( v ), false );
	}
    }

    return best_size - old_best_size <= k;
}

template<bool exists>
bool
vertex_cover_vc3_buss( graptor::graph::GraphDoubleIndexCSx<VID,EID> & G,
		       VID k,
		       VID c,
		       VID & best_size,
		       VID * best_cover ) {
    // Set U of vertices of degree higher than k
    VID u_size = std::count_if( G.dbegin(), G.dend(),
				[&]( VID deg ) { return deg > k; } );

    // If |U| > k, then there exists no cover of size k
    if( u_size > k )
	return false;

    assert( u_size > 0 );
    
    // Construct G'
    VID n = G.numVertices();
    auto chkpt = G.checkpoint();
    G.disable_incident_edges( [&]( VID v ) {
	return chkpt.get_degree( v ) > k;
    } );
    EID m = G.numEdges();
    
    // If G' has more than k(k-|U|) edges, reject
    if( m/2 > k * ( k - u_size ) ) {
	G.restore_checkpoint( chkpt );
	return false;
    }

    // Find a cover for the remaining vertices
    VID gp_best_size = 0;
    bool rec = vertex_cover_vc3<exists>(
	G, k - u_size, c,
	gp_best_size, &best_cover[best_size] );

    if( rec ) {
	// Debug
	// check_cover( G, gp_best_size, &best_cover[best_size] );

	// for( VID i=0; i < gp_best_size; ++i )
	// best_cover[best_size+i] = gp_xlat[best_cover[best_size+i]];
	best_size += gp_best_size;

	// All vertices with degree > k must be included in the cover
	for( VID v=0; v < n; ++v )
	    if( chkpt.get_degree( v ) > k )
		best_cover[best_size++] = v;

	// check_cover( G, best_size, best_cover );
    }

    G.restore_checkpoint( chkpt );

    return rec;
}

template<bool exists>
int
vertex_cover_vc3_crown( graptor::graph::GraphDoubleIndexCSx<VID,EID> & G,
			VID k,
			VID c,
			VID & best_size,
			VID * best_cover ) {
    VID n = G.numVertices();

    // Compute crown kernel: (I=1,H=2)
    std::vector<uint8_t> crown = crown_kernel( G );

    // All vertices in H are included in cover
    VID tmp_best_size = best_size;
    VID h_size = 0, i_size = 0;
    for( VID v=0; v < n; ++v ) {
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
    G.disable_incident_edges( [&]( VID v ) {
	return crown[v] != 0;
    } );
    EID m = G.numEdges();

    // Find a cover for the remaining vertices
    VID gp_best_size = 0;
    bool rec = vertex_cover_vc3<exists>(
	G, k - h_size, c,
	gp_best_size, &best_cover[tmp_best_size] );

    if( rec )
	best_size = tmp_best_size + gp_best_size;

    G.restore_checkpoint( chkpt );

    return rec;
}

template<bool exists>
bool
vertex_cover_vc3( graptor::graph::GraphDoubleIndexCSx<VID,EID> & G,
		  VID k,
		  VID c,
		  VID & best_size,
		  VID * best_cover ) {
    VID n = G.numVertices();
    EID m = G.numEdges();

    if( k == 0 )
	return m == 0;

    VID max_v, max_deg;
    std::tie( max_v, max_deg ) = G.max_degree();
    if( max_deg <= 2 ) {
	bool ret = vertex_cover_poly( G, k, best_size, best_cover );
	// if( ret )
	// check_cover( G, best_size, best_cover );
	return ret;
    }

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

    // Must have a vertex with degree >= 3
    assert( max_deg >= 3 );

    // Create two subproblems ; branch on max_v
    VID i_best_size = 0;
    VID * i_best_cover = new VID[n-1];
    VID x_best_size = 0;
    VID * x_best_cover = new VID[n-1-max_deg];

    // Neighbour list of max_v, retained by disable_incident_edges
    G.sort_neighbours( max_v );
    auto NI = G.nbegin( max_v );
    auto NE = G.nend( max_v );

    // In case v is included, erase only v
    auto chkpt = G.checkpoint();
    G.disable_incident_edges( [=]( VID v ) { return v == max_v; } );

    // In case v is excluded, erase v (previous step) and all its neighbours
    // Make sure our neighbours are sorted. Iterators remain valid after
    // erasing incident edges.
    auto chkpti = G.checkpoint();
    G.disable_incident_edges( [=]( VID v ) {
	auto pos = std::lower_bound( NI, NE, v );
	return pos != NE && *pos == v;
    } );

    VID x_k = std::min( n-1-max_deg, k-max_deg );
    bool x_ok = false;
    if( k >= max_deg ) {
	if( verbose )
	    std::cerr << "vc3: n=" << n << " m=" << m
		      << " vertex " << max_v << " deg " << max_deg
		      << " excluded k=" << k << "\n";
	x_ok = vertex_cover_vc3<exists>( G, x_k, c, x_best_size, x_best_cover );
    }

    if constexpr ( exists ) {
	if( x_ok ) {
	    G.restore_checkpoint( chkpt );
	    assert( x_best_size <= k );

	    for( auto I=NI; I != NE; ++I )
		best_cover[best_size++] = *I;
	    for( VID i=0; i < x_best_size; ++i )
		best_cover[best_size++] = x_best_cover[i];
	    
	    delete[] i_best_cover;
	    delete[] x_best_cover;

	    return true;
	}
    }

    G.restore_checkpoint( chkpti );
    VID i_k = x_ok ? std::min( max_deg+x_best_size, k-1 ) : k-1;
    if( verbose )
	std::cerr << "vc3: n=" << n << " m=" << m
		  << " vertex " << max_v << " deg " << max_deg
		  << " included k=" << k << "\n";
    bool i_ok = vertex_cover_vc3<exists>(
	G, i_k, c, i_best_size, i_best_cover );

    if( i_ok && ( !x_ok || i_best_size+1 < x_best_size+max_deg ) ) {
	best_cover[best_size++] = max_v;
	for( VID i=0; i < i_best_size; ++i )
	    best_cover[best_size++] = i_best_cover[i];
    } else if( x_ok ) {
	for( auto I=NI; I != NE; ++I )
	    best_cover[best_size++] = *I;
	for( VID i=0; i < x_best_size; ++i )
	    best_cover[best_size++] = x_best_cover[i];
    }

    G.restore_checkpoint( chkpt );

    // if( i_ok || x_ok )
    // check_cover( G, best_size, best_cover );

    delete[] i_best_cover;
    delete[] x_best_cover;

    return i_ok || x_ok;
}

VID
clique_via_vc3( const HGraphTy & G,
		VID v,
		VID degeneracy,
		MC_CutOutEnumerator & E,
		PSet<VID> & pset,
		VID ce,
		int depth ) {
    // TODO: potentially apply more filtering using up-to-date best
    //       might do conditionally on improvement of best since
    //       previous cut-out
    // Note: when called from top-level, the pset contains all vertices and
    //       no further filtering is applied.
    assert( ce == pset.get_fill() );
    GraphBuilderInducedComplement<graptor::graph::GraphDoubleIndexCSx<VID,EID>>
	cbuilder( G, pset );
    auto & CG = cbuilder.get_graph();
    VID cn = CG.numVertices();
    EID cm = CG.numEdges();

    for( VID v=0; v < cn; ++v )
	assert( CG.getDegree(v) + G.getDegree(v) + 1 == cn );

    // If no edges remain after pruning, then cover has size 0 and
    // clique has size cn.
    if( cm == 0 ) {
	E.record( depth+cn, pset.begin(), pset.end() );
	return depth+cn; // return true;
    }

    VID best_size = 0;
    std::vector<VID> best_cover( cn );

    VID bc = E.get_max_clique_size();
    if( bc >= cn + depth )
	return 0; // return false;

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

    const VID k_prior = cn + depth - bc;
    VID k_up = k_prior - 1;
    VID k_lo = 1;
    VID k_best_size = k_prior;
    VID k = k_up;
    bool first_attempt = true;
    while( true ) {
	best_size = 0;
	bool any = vertex_cover_vc3<true>( CG, k, 1, best_size, &best_cover[0] );
	if( verbose ) {
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

    // Record clique
    if( k_best_size < k_prior ) {
	if( E.is_feasible( depth + cn - k_best_size ) ) {
	    if( verbose )
		std::cout << "clique_via_vc3: max_clique: "
			  << ( depth + cn - k_best_size )
			  << " E.best: " << bc << "\n";
	    // Create complement set
	    std::vector<VID> clique( depth + cn - k_best_size );
	    std::sort( best_cover.begin(), best_cover.end() );
	    for( VID i=0, j=0; i < cn; ++i ) {
		if( best_cover[j] == i )
		    ++j;
		else
		    clique.push_back( i );
	    }
	    E.record( depth + cn - k_best_size,
		      clique.begin(), clique.end() ); // size of complement
	}
    }

    // return true;
    return depth + cn - k_best_size;
}

/*======================================================================*
 * recursively parallel version of Bron-Kerbosch w/ pivoting
 *======================================================================*/
template<typename VID, typename EID>
bool mc_leaf(
    const HGraphTy & H,
    MC_CutOutEnumerator & E,
    const clique_set<VID> * R,
    const PSet<VID> & xp_set,
    size_t depth );

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
	size_t tv = pset.intersect_size_exceed(
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

#if TUNABLE_SMALL_AVOID_CUTOUT_LEAF != 0
    if( ce_new - ne_new >= TUNABLE_SMALL_AVOID_CUTOUT_LEAF )
#endif
    {
	if( mc_leaf<VID,EID>( G, E, R, xp_new, depth ) )
	    return;
    }

    // Large sub-problem; search recursively
    // Tuning point: do we cut out a subgraph or not?
    // Tuning point: do we proceed with MC or switch to VC?
    mc_bron_kerbosch_recpar_xps( G, degeneracy, E, R, xp_new, depth );
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

template<typename VID>
std::pair<VID,VID>
count_colours_greedy( const HGraphTy & G, const PSet<VID> & xp ) {
    // Upper bound, loose?
    // Example: if we have i neighbours in the PSet, we would deduce the need
    // for colour i, however, if some of those neighbours can have the same
    // colour, then we don't need colour i.
    VID n = G.numVertices();
    VID s = xp.size();
    std::vector<VID> colour( n );
    std::vector<VID> histo( s );
    VID c = 1; // number of colours in use
    VID max_col = 0;
    VID max_rdeg = 0;
    for( VID j=0; j < s; ++j ) {
	VID i = s-1 - j;
	VID v = xp.at( i );
	const auto & adj = G.get_neighbours_set( v );
	std::fill( histo.begin(), histo.end(), 0 );

	// Right-degree
	auto nb = adj.begin();
	auto ne = adj.end();
	nb = std::upper_bound( nb, ne, i );
	VID rdeg = std::distance( nb, ne );
	if( rdeg > max_rdeg )
	    max_rdeg = rdeg;

	// Intersect and check colours
	const VID * pb = xp.get_set() + i + 1;
	const VID * pe = xp.get_set() + s;
	if( ne != nb )
	    pe = std::upper_bound( pb, pe, *(ne-1) );
	for( ; pb != pe; ++pb ) {
	    if( adj.contains( *pb ) )
		histo[colour[*nb]] = 1;
	}
		
	for( VID c=0; c < n; ++c ) {
	    if( histo[c] == 0 ) {
		colour[i] = c;
		if( c > max_col )
		    max_col = c;
		break;
	    }
	}
    }
    return { max_col, max_rdeg };
}

// XP may be modified by the method. It is not required to be in sort order.
void
mc_bron_kerbosch_recpar_xps(
    const HGraphTy & G,
    VID degeneracy,
    MC_CutOutEnumerator & E,
    const clique_set<VID> * R,
    PSet<VID> & xp,
    int depth ) {
    VID ce = xp.size();
    
    // Termination condition
    if( 0 == ce ) {
	E.record( depth, R->begin(), R->end() );
	return;
    }
    const VID n = G.numVertices();

    if constexpr ( io_trace ) {
	std::lock_guard<std::mutex> guard( io_mux );
	std::cout << "XPS loop: ce=" << ce << " depth=" << depth << "\n";
    }

    if( !E.is_feasible( depth + xp.size(), fr_pset ) )
	return;

    // auto [ num_colours, max_rdeg ] = count_colours_ub( G, xp );
    auto [ num_colours, max_rdeg ] = count_colours_greedy( G, xp );
    if( !E.is_feasible( depth + num_colours, fr_colour_greedy ) )
	return;
    if( !E.is_feasible( depth + 1 + max_rdeg, fr_rdeg ) )
	return;

    // VID pivot = xp.at( 0 );
    VID pivot = mc_get_pivot( G, xp ).first;
    const auto & p_adj = G.get_neighbours_set( pivot );

    for( VID i=0; i < xp.size(); ++i ) {
	VID v = xp.at( i );

	// Skip neighbours of pivot.
	// Could remove them explicitly, however, not needed in sequential
	// execution.
	if( p_adj.contains( v ) )
	    continue;
	
	const auto adj = G.get_neighbours_set( v ); 
	VID deg = adj.size();

	clique_set<VID> R_new( v, R );

	if constexpr ( io_trace ) {
	    std::lock_guard<std::mutex> guard( io_mux );
	    std::cout << "XP2: X=" << i << " P=" << (ce - (i+1)) << " adj="
		      << adj.size() << " depth=" << depth << "\n";
	}

	if( deg == 0 ) [[unlikely]] { // implies ne == ce == 0
	    // avoid overheads of copying and cutout
	    E.record( depth+1, R_new.begin(), R_new.end() );
	} else {
	    // Some complexity:
	    // + Need to consider all vertices prior to v in XP are now
	    //   in the X set. Could set ne to i, however:
	    // + Vertices that are filtered due to pivoting,
	    //   i.e., neighbours of pivot, are still in P.
	    // + In sequential execution, we can update XP incrementally,
	    //   however in parallel execution we cannot.
	    // TODO: streamline with dual_set
	    PSet<VID> xp_new = xp.intersect_validate( n, adj );
	    bk_recursive_call( G, degeneracy, E, &R_new, xp_new, depth+1 );
	}

	xp.invalidate( v );
    }
}

void
mc_bron_kerbosch_recpar_top_xps(
    const HGraphTy & G,
    VID degeneracy,
    MC_CutOutEnumerator & E,
    const clique_set<VID> * R ) {
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
	clique_set<VID> R_new( v, R );

	if( deg == 0 ) {
	    // avoid overheads of copying and cutout
	    // TODO: assume this never happens at top level due to
	    //       filtering before creating cutout.
	    E.record( 1, R_new.begin(), R_new.end() );
	} else {
	    auto adj = G.get_neighbours_set( v ); 
	    PSet<VID> xp_new = PSet<VID>::left_union_right( n, v, p_adj, adj );
	    bk_recursive_call( G, degeneracy, E, &R_new, xp_new, 2 );
	}
    }
}

template<unsigned Bits, typename HGraph, typename Enumerator>
void mc_dense_fn(
    const GraphCSx & G,
    const HGraph & H,
    Enumerator & E,
    VID v,
    const graptor::graph::NeighbourCutOutDegeneracyOrderFiltered<VID,EID> & cut,
    all_variant_statistics & stats ) {

    timer tm;
    tm.start();

    VID num = cut.get_num_vertices();
    size_t cl = get_size_class( num );

    // Build induced graph
    DenseMatrix<Bits,VID,EID>
	IG( G, H, cut.get_vertices(), 0, cut.get_num_vertices() );

    VID n = IG.numVertices();
    VID m = IG.calculate_num_edges(); // considers inverted graph
    float d = 1.0f - ( (float)m / ( (float)n * (float)(n-1) ) );

    double tc = tm.next();

    algo_variant av = av_bk;
    if( d > 0.9f ) {
	VID bc = E.get_max_clique_size();
	VID k_max = n < bc ? 0 : n - bc + 1;

	auto bs = IG.vertex_cover_kernelised( k_max );
	E.record( 1 + bs.size(), bs.begin(), bs.end() );
	av = av_vc;
    } else {
	clique_set<VID> R( v );
	MC_DenseEnumerator DE( E, &R );
	IG.mc_search( DE, 1 );
    }

    double t = tm.next();

    variant_statistics & s = stats.get( av, cl );
    s.record_build( tc );
    s.record( t );
}

typedef void (*mc_func)(
    const GraphCSx &, 
    const HFGraphTy &,
    MC_CutOutEnumerator &,
    VID,
    const graptor::graph::NeighbourCutOutDegeneracyOrderFiltered<VID,EID> & cut,
    all_variant_statistics & );
    
static mc_func mc_dense_func[N_DIM+1] = {
    &mc_dense_fn<32,HFGraphTy,MC_CutOutEnumerator>,  // N=32
    &mc_dense_fn<64,HFGraphTy,MC_CutOutEnumerator>,  // N=64
    &mc_dense_fn<128,HFGraphTy,MC_CutOutEnumerator>, // N=128
    &mc_dense_fn<256,HFGraphTy,MC_CutOutEnumerator>, // N=256
    &mc_dense_fn<512,HFGraphTy,MC_CutOutEnumerator>  // N=512
};

void mc_top_level_bk(
    const GraphCSx & G,
    const HFGraphTy & H,
    MC_Enumerator & E,
    VID v,
    VID degeneracy,
    const VID * const remap_coreness,
    graptor::graph::NeighbourCutOutDegeneracyOrderFiltered<VID,EID> & cut ) {

    all_variant_statistics & stats = mc_stats.get_statistics();

    timer tm;
    tm.start();

    GraphBuilderInduced<HGraphTy> ibuilder( G, H, v, cut );
    const auto & HG = ibuilder.get_graph();

    stats.record_genbuild( av_bk, tm.next() );

    clique_set<VID> R( v );
    MC_CutOutEnumerator CE( E, v, cut.get_vertices() );
    mc_bron_kerbosch_recpar_top_xps( HG, degeneracy, CE, &R );

    stats.record_gen( av_bk, tm.next() );
}

void mc_top_level_vc(
    const GraphCSx & G,
    const HFGraphTy & H,
    MC_Enumerator & E,
    VID v,
    VID degeneracy,
    const VID * const remap_coreness,
    graptor::graph::NeighbourCutOutDegeneracyOrderFiltered<VID,EID> & cut ) {

    all_variant_statistics & stats = mc_stats.get_statistics();

    timer tm;
    tm.start();

    // TODO: cut out just once, not twice
    GraphBuilderInduced<HGraphTy> ibuilder( G, H, v, cut );
    const auto & HG = ibuilder.get_graph();

    PSet<VID> pset = PSet<VID>::create_full_set( HG );

    stats.record_genbuild( av_vc, tm.next() );

    MC_CutOutEnumerator CE( E, v, cut.get_vertices() );
    clique_via_vc3( HG, v, degeneracy, CE, pset, pset.size(), 1 );

    stats.record_gen( av_vc, tm.next() );
}

void mc_top_level(
    const GraphCSx & G,
    const HFGraphTy & H,
    MC_Enumerator & E,
    VID v,
    VID degeneracy,
    const VID * const remap_coreness ) {

    all_variant_statistics & stats = mc_stats.get_statistics();

    timer tm;
    tm.start();

    VID best = E.get_max_clique_size();

    // No point analysing a vertex of too low degree
    if( remap_coreness[v] < best )
	return;

    // Filter out vertices where degree in main graph < best.
    // With degree == best, we can make a clique of size best+1 at best.
    // Cut-out constructed filters out left-neighbours.
    graptor::graph::NeighbourCutOutDegeneracyOrderFiltered<VID,EID>
	cut( G, v, [&]( VID u ) { return remap_coreness[u] > best; } );

    VID hn1 = cut.get_num_vertices();

    if( hn1 < best )
	return;

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
    EID m_est = 0;
    cut.filter( [&]( VID u ) {
	VID d = graptor::set_operations<graptor::hash_vector>
	    ::intersect_size_exceed_ds(
		cut.get_slice(),
		H.get_neighbours_set( u ),
		best-1 ); // exceed checks >, we need >= best
	m_est += d;
	return d >= best;
    }, best );

    // If size of cut-out graph is less than best, then there is no point
    // in analysing it, nor constructing cut-out.
    if( cut.get_num_vertices() < best )
	return;

    VID num = cut.get_num_vertices();

    double tf = tm.next();

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

    float d = float(m_est) / ( float(num) * float(num) );

#if !ABLATION_DISABLE_TOP_DENSE
    VID nlg = get_size_class( num );

    if( nlg <= N_MAX_SIZE ) {
	MC_CutOutEnumerator CE( E, v, cut.get_vertices() );
	return mc_dense_func[nlg-N_MIN_SIZE]( G, H, CE, v, cut, stats );
    }
#endif

    if( d > .9f ) {
	mc_top_level_vc( G, H, E, v, degeneracy, remap_coreness, cut );
    } else {
	mc_top_level_bk( G, H, E, v, degeneracy, remap_coreness, cut );
    }

#if 0
    if constexpr ( true )
    {
	VID hn = HG.numVertices();
	EID hm = 0;
	for( auto I=HG.vbegin(), E=HG.vend(); I != E; ++I )
	    hm += HG.getDegree( *I );
	float hd = float(hm) / ( (float)(hn) * (float)(hn-1) );
	float avg = float(hm) / float(hn);

	VID * deg = new VID[hn];
	for( VID i=0; i < hn; ++i )
	    deg[i] = HG.getDegree( i );
	std::sort( deg, deg+hn );

	float med = *( deg + hn/2 );
	if( ( hn & 1 ) == 0 )
	    med = float( *( deg+hn/2-1 ) + *( deg+hn/2 ) ) / 2.0f;

	std::lock_guard<std::mutex> guard( io_mux );
	std::cout << "v=" << v
		  << " n1=" << hn1
		  << " n=" << hn
		  << " m=" << hm
		  << " d=" << hd
		  << " davg=" << avg
		  << " dmed=" << med
		  << " tbk=" << t1
		  << " tvc=" << t0
		  << " tdns=" << t2
		  << " tf=" << tf
		  << " tc=" << tc
		  << " deg: {";
	for( VID i=0; i < hn; ++i )
	    std::cout << ' ' << deg[hn-i-1];
	std::cout << " } core: {";
	for( VID i=0; i < hn; ++i )
	    std::cout << ' ' << HG_coreness[i];
	std::cout << " } " << ( t1<t0 ? "BK" : "VC" ) << "\n";

	if( t1 < t0 )
	    ++num_top_bk;
	else
	    ++num_top_vc;

	delete[] deg;
    }
#endif
}

template<unsigned Bits, typename VID, typename EID>
void leaf_dense_fn(
    const HGraphTy & H,
    MC_CutOutEnumerator & E,
    const clique_set<VID> * R,
    const PSet<VID> & xp_set,
    size_t depth ) {
    variant_statistics & stats
	= mc_stats.get_statistics().get_leaf( av_bk, ilog2( Bits ) );
    timer tm;
    tm.start();
// TODO: Integrate colouring heuristic into cutout (?)
// or at least track max degree and use this for early filtering
    DenseMatrix<Bits,VID,VID> D( H, H, xp_set.get_set(), 0, xp_set.get_fill() );
    // VID n = D.numVertices();
    // VID m = D.calculate_num_edges();
    // VID m = D.get_num_edges();
    // float d = (float)m / ( (float)n * (float)(n-1) );
    stats.record_build( tm.next() );

/*
    if( d < 0.5 )
	D.mc_search( E, depth );
    else {
	auto bs = D.vertex_cover_kernelised();
	E.record( depth + bs.size() );
    }
*/
    
    // Maximum clique size is depth (size of R), maximum degree in D
    // (max number of plausible neighbours), +1 for the vertex whose neighbours
    // we are checking.
    if( !E.is_feasible( depth + D.get_max_degree() + 1, fr_maxdeg ) ) {
/*
	std::cout << "Leaf task: n=" << D.numVertices()
		  << " m=" << m << " d=" << d
		  << " pset=" << xp_set.get_fill()
		  << " depth=" << depth
		  << " d_max=" << D.get_max_degree()
		  << " not feasible\n";
*/
	return;
    }

/*
    VID init_k = n - ( E.get_max_clique_size() - depth );
    auto bs = D.vertex_cover_kernelised( init_k );
    float tvc = tm.next();
*/

    MC_DenseEnumerator DE( E, R, xp_set.get_set() );
    D.mc_search( DE, depth );
    float tbk = tm.next();

    stats.record( tbk );

/*
    std::cout << "Leaf task: n=" << D.numVertices()
	      << " m=" << m << " d=" << d
	      << " d_max=" << D.get_max_degree()
	      << " pset=" << xp_set.get_fill()
	      << " mc=" << bs.size() << " depth=" << depth
	      << " tbk=" << tbk
	      << " tvc=" << tvc
	      << " bk/vc=" << (tbk/tvc)
	      << "\n";
*/

    // E.record( depth + bs.size() );
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
    // VID pnum = ce - ne;
    VID * XP = xp_set.get_set();

/*
    if( false && ce <= 3 ) { // TODO
	mc_tiny( H, XP, 0, ce, E );
	return true;
    }
*/

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

int main( int argc, char *argv[] ) {
    commandLine P( argc, argv, " help" );
    bool symmetric = P.getOptionValue("-s");
    VID npart = P.getOptionLongValue( "-c", 256 );
    VID pre = P.getOptionLongValue( "-pre", -1 );
    VID what_if = P.getOptionLongValue( "-what-if", -1 );
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

    // Number of partitions is tunable. A fairly large number is helpful
    // to help load balancing.
    GraphCSRAdaptor GA( G, npart );
    KCv<GraphCSRAdaptor> kcore( GA, P );
    kcore.run();
    auto & coreness = kcore.getCoreness();
    std::cout << "Calculating coreness: " << tm.next() << "\n";
    std::cout << "coreness=" << kcore.getLargestCore() << "\n";

    mm::buffer<VID> order( n, numa_allocation_interleaved() );
    mm::buffer<VID> rev_order( n, numa_allocation_interleaved() );
    VID K = kcore.getLargestCore();
    std::vector<VID> histo; histo.resize( K+1 );
    sort_order_ties( order.get(), rev_order.get(),
		     coreness, n, K, &histo[0], G.getDegree(),
		     true ); // reverse sort
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
	      << "\n\tABLATION_DISABLE_LEAF=" << ABLATION_DISABLE_LEAF
	      << "\n\tABLATION_DISABLE_TOP_TINY=" << ABLATION_DISABLE_TOP_TINY
	      << "\n\tABLATION_DISABLE_TOP_DENSE=" << ABLATION_DISABLE_TOP_DENSE
	      << "\n\tABLATION_HADJPA_DISABLE_XP_HASH="
	      << ABLATION_HADJPA_DISABLE_XP_HASH
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
	      << "\n\tABLATION_DENSE_NO_PIVOT_TOP="
	      << ABLATION_DENSE_NO_PIVOT_TOP
	      << "\n\tABLATION_DENSE_PIVOT_FILTER="
	      <<  ABLATION_DENSE_PIVOT_FILTER
	      << "\n\tUSE_512_VECTOR=" <<  USE_512_VECTOR
	      << "\n\tOUTER_ORDER=" <<  OUTER_ORDER
	      << '\n';
    
    system( "hostname" );
    system( "date" );

    std::cout << "Start enumeration: " << tm.next() << std::endl;


#if PAPI_REGION == 1 
    map_workers( [&]( uint32_t t ) {
	if( PAPI_OK != PAPI_hl_region_begin( "MC" ) ) {
	    std::cerr << "Error initialising PAPI\n";
	    exit(1);
	}
    } );
#endif

    VID degeneracy = kcore.getLargestCore();
    MC_Enumerator E( degeneracy, order.get() );

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
	mc_top_level( R, H, E, v, degeneracy, remap_coreness.get() );
    }

    if( what_if != ~(VID)0 ) {
	// For highest-degree vertex with maximum degeneracy, try setting
	// a strong precedent for the clique size
	std::vector<VID> empty;
	E.record( what_if, what_if, empty.begin(), empty.end() );
	VID v = histo[1];
	std::cout << "what-if clique=" << what_if << " vertex v=" << v
		  << " deg=" << H.getDegree( v )
		  << " rho=" << remap_coreness[v]
		  << "\n";
	mc_top_level( R, H, E, v, degeneracy, remap_coreness.get() );

	// Successfully found a clique larger than postulated size?
	// If not, erase result.
	if( E.get_max_clique_size() <= what_if )
	    E.reset();
    }

#if OUTER_ORDER == 1
    /* 1. low to high degree order */

#elif OUTER_ORDER == 2
    /* 2. low to high degeneracy order */
    for( VID w=0; w < n; ++w ) {
	VID v = w;
	// std::cout << "w=" << w << " v=" << v
	// << " deg=" << H.getDegree( v )
	// << " rho=" << remap_coreness[v]
	// << "\n";
	mc_top_level( R, H, E, v, degeneracy, remap_coreness.get() );
    }

#elif OUTER_ORDER == 3
    /* 3. high to low degeneracy order */
    for( VID w=0; w < n; ++w ) {
	VID v = n - 1 - w;
	// std::cout << "w=" << w << " v=" << v
	// << " deg=" << H.getDegree( v )
	// << " rho=" << remap_coreness[v]
	// << "\n";
	mc_top_level( R, H, E, v, degeneracy, remap_coreness.get() );
    }

#elif OUTER_ORDER == 4
    /* 4. first evaluate highest-degree vertex per degeneracy level, then 3. */
    for( VID c=0; c <= K; ++c ) {
	VID c_up = histo[c];
	VID c_lo = c == K ? 0 : histo[c+1];
	if( c_up == c_lo )
	    continue;

	VID v = c_lo;
	// std::cout << "c=" << c << " v=" << v
	// << " deg=" << H.getDegree( v )
	// << " rho=" << remap_coreness[v]
	// << "\n";
	mc_top_level( R, H, E, v, degeneracy, remap_coreness.get() );

	if( !E.is_feasible( K-c+1 ) )
	    break;
    }
	    
    for( VID cc=0; cc <= K; ++cc ) {
	VID c = K - cc;
	VID c_up = histo[c];
	VID c_lo = c == K ? 0 : histo[c+1];
	++c_lo; // already did c_lo in preamble
	if( !E.is_feasible( K-c+1 ) )
	    continue;
	for( VID v=c_lo; v < c_up; ++v ) {
	    // std::cout << "c=" << c << " v=" << v
	    // << " deg=" << H.getDegree( v )
	    // << " rho=" << remap_coreness[v]
	    // << "\n";
	    mc_top_level( R, H, E, v, degeneracy, remap_coreness.get() );
	    if( !E.is_feasible( K-c+1 ) )
		break;
	}
    }
#elif OUTER_ORDER == 5
    /* 5. Preamble as with 4, then process in order of decreasing degeneracy. */
    for( VID c=0; c <= K; ++c ) {
	VID c_up = histo[c];
	VID c_lo = c == K ? 0 : histo[c+1];
	if( c_up == c_lo )
	    continue;

	VID v = c_lo;
	// std::cout << "c=" << c << " v=" << v
	// << " deg=" << H.getDegree( v )
	// << " rho=" << remap_coreness[v]
	// << "\n";
	mc_top_level( R, H, E, v, degeneracy, remap_coreness.get() );

	if( !E.is_feasible( K-c+1 ) )
	    break;
    }
	    
    for( VID c=0; c <= K; ++c ) {
	VID c_up = histo[c];
	VID c_lo = c == K ? 0 : histo[c+1];
	++c_lo; // already did c_lo in preamble
	if( !E.is_feasible( K-c+1 ) ) // decreasing degeneracy -> done
	    break;
	for( VID v=c_lo; v < c_up; ++v ) {
	    // std::cout << "c=" << c << " v=" << v
	    // << " deg=" << H.getDegree( v )
	    // << " rho=" << remap_coreness[v]
	    // << "\n";
	    mc_top_level( R, H, E, v, degeneracy, remap_coreness.get() );
	    if( !E.is_feasible( K-c+1 ) )
		break;
	}
    }
#endif

#if PAPI_REGION == 1
    map_workers( [&]( uint32_t t ) {
	if( PAPI_OK != PAPI_hl_region_end( "MC" ) ) {
	    std::cerr << "Error initialising PAPI\n";
	    exit(1);
	}
    } );
#endif

    std::cout << "Enumeration: " << tm.next() << "\n";

    all_variant_statistics stats = mc_stats.sum();

    double duration = tm.total();
    std::cout << "Completed MC in " << duration << " seconds\n";
    for( size_t n=N_MIN_SIZE; n <= N_MAX_SIZE; ++n ) {
	std::cout << (1<<n) << "-bit dense BK: ";
	stats.get( av_bk, n ).print( std::cout ); 
	std::cout << (1<<n) << "-bit dense VC: ";
	stats.get( av_vc, n ).print( std::cout ); 
    }
    std::cout << "generic BK: ";
    stats.m_gen[av_bk].print( std::cout );
    std::cout << "generic VC: ";
    stats.m_gen[av_vc].print( std::cout );

    // Note: reported top-level vertex not translated back after reordering
    E.report( std::cout );

    remap_coreness.del();
    rev_order.del();
    order.del();
    G.del();

    return 0;
}
