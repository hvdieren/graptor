#include <cmath>

#include "graptor/graptor.h"
#include "graptor/api.h"
#include "path.h"

using expr::_0;
using expr::_1;
using expr::_c;

// By default set options for highest performance
#ifndef DEFERRED_UPDATE
#define DEFERRED_UPDATE 1
#endif

#ifndef LEVEL_ASYNC
#define LEVEL_ASYNC 1
#endif

#ifndef UNCOND_EXEC
#define UNCOND_EXEC 1
#endif

#ifdef CONVERGENCE
#undef CONVERGENCE
#endif
#define CONVERGENCE 0

#ifdef MEMO
#undef MEMO
#endif
#define MEMO 0

#ifndef VARIANT
#define VARIANT 0
#endif

#ifndef SP_THRESHOLD
#define SP_THRESHOLD -1
#endif

#ifndef FUSION
#define FUSION 1
#endif

#ifndef AVG_DELTA
#define AVG_DELTA 0
#endif

struct BF_customfp_config {
    static constexpr size_t bit_size = 16;
};

using cfp_cfg = variable_customfp_config<BF_customfp_config>;

using SlotID = int32_t;
using USlotID = uint32_t;

#if VARIANT == 0
using FloatTy = float;
using SFloatTy = float;
using EncCur = array_encoding<FloatTy>;
using EncNew = array_encoding<FloatTy>;
using EncEdge = array_encoding<FloatTy>;
#elif VARIANT == 1
using FloatTy = float;
// using SFloatTy = scustomfp<false,7,9,true,6>;
using SFloatTy = scustomfp<false,5,11,true,15>;
using EncCur = array_encoding_wide<SFloatTy>;
using EncNew = EncCur;
using EncEdge = array_encoding<FloatTy>;
#elif VARIANT == 2
using FloatTy = float;
using SFloatTy = vcustomfp<cfp_cfg>;
using EncCur = array_encoding<SFloatTy>;
using EncNew = EncCur;
using EncEdge = array_encoding<FloatTy>;
#elif VARIANT == 3
using FloatTy = float;
using SFloatTy = scustomfp<false,5,11,true,21>;
using EncCur = array_encoding_wide<SFloatTy>;
using EncNew = EncCur;
using EncEdge = array_encoding<FloatTy>;
#elif VARIANT == 4
using FloatTy = float;
using SFloatTy = scustomfp<false,5,11,true,30>;
using EncCur = array_encoding_wide<SFloatTy>;
using EncNew = EncCur;
using EncEdge = array_encoding<FloatTy>;
#else
#error "Bad value for VARIANT"
#endif

static constexpr float weight_range = float(1<<7);

using CurTy = typename EncCur::stored_type;
using NewTy = typename EncNew::stored_type;
using EdgeTy = typename EncEdge::stored_type;

enum variable_name {
    var_cur = 0,
    var_new = 1,
    var_min = 3,
    var_max = 4,
    var_remap = 5,
    var_cur2 = 6,
    var_new2 = 7,
    var_d = 8,
    var_neg = 10,
    var_zero = 11,
    var_ref = 12
};

#if AVG_DELTA
namespace {
    static api::vertexprop<FloatTy,VID,var_ref,EncNew> * ref_sol = nullptr;
    static bool have_ref = false;
}
#endif

template <class GraphType>
class BF {
public:
    BF( GraphType & _GA, commandLine & P )
	: GA( _GA ),
	  info_buf( 60 ),
#if !LEVEL_ASYNC || DEFERRED_UPDATE
	  cur_dist( GA.get_partitioner(), "current distance" ),
#endif
	  new_dist( GA.get_partitioner(), "new distance" ),
	  edge_weight( GA.get_partitioner(), "edge weight" ) {
	itimes = P.getOption( "-itimes" );
	debug = P.getOption( "-debug" );
	calculate_active = P.getOption( "-cactive" );
	est_diam = P.getOptionLongValue( "-sssp:diam", 20 );
	start = GA.remapID( P.getOptionLongValue( "-start", 0 ) );
	lo_frac = P.getOptionDoubleValue( "-lo:frac", 0.05 );
	hi_frac = P.getOptionDoubleValue( "-hi:frac", 0.50 );
	tgt_frac = P.getOptionDoubleValue( "-tgt:frac", 0.025 );

	if( GA.getWeights() == nullptr ) {
	    std::cerr << "BF requires edge weights. Terminating.\n";
	    exit( 1 );
	}

	FloatTy scale = P.getOptionDoubleValue( "-scale", 1.0 );
	if( scale != 1.0 ) {
	    std::cerr << "Scaling weights by " << scale << "\n";
	    api::edgemap(
		GA,
		api::relax( [&]( auto s, auto d, auto e ) {
		    return edge_weight[e] *= _c( scale );
		} ) )
		.materialize();
	}

#if AVG_DELTA
	if( ref_sol == nullptr ) {
	    ref_sol = new api::vertexprop<FloatTy,VID,var_ref,EncNew>(
		GA.get_partitioner(), "reference solution" );
	    have_ref = false;
	}
#endif

	static bool once = false;
	if( !once ) {
	    once = true;
	    std::cerr << "DEFERRED_UPDATE=" << DEFERRED_UPDATE << "\n";
	    std::cerr << "UNCOND_EXEC=" << UNCOND_EXEC << "\n";
	    std::cerr << "LEVEL_ASYNC=" << LEVEL_ASYNC << "\n";
	    std::cerr << "CONVERGENCE=" << CONVERGENCE << "\n";
	    std::cerr << "MEMO=" << MEMO << "\n";
	    std::cerr << "VARIANT=" << VARIANT << "\n";
	    std::cerr << "SP_THRESHOLD=" << SP_THRESHOLD << "\n";
	    std::cerr << "FUSION=" << FUSION << "\n";
	    std::cerr << "sizeof(FloatTy)=" << sizeof(FloatTy) << "\n";
	    std::cerr << "sizeof(SFloatTy)=" << sizeof(SFloatTy) << "\n";
	    std::cerr << "lo_frac=" << lo_frac << "\n";
	    std::cerr << "hi_frac=" << hi_frac << "\n";
	    std::cerr << "tgt_frac=" << tgt_frac << "\n";
#if VARIANT != 2
	    using ft = fp_traits<SFloatTy>;
	    std::cerr << "FP: sign bit=" << ft::sign_bit << "\n";
	    std::cerr << "FP: exponent bits=" << ft::exponent_bits << "\n";
	    std::cerr << "FP: mantissa bits=" << ft::mantissa_bits << "\n";
	    std::cerr << "FP: exponent truncated=" << ft::exponent_truncated << "\n";
	    std::cerr << "FP: can hold zero=" << ft::maybe_zero << "\n";
	    std::cerr << "FP: exponent bias=" << ft::exponent_bias << "\n";
#endif
	}
    }
    ~BF() {
#if !LEVEL_ASYNC || DEFERRED_UPDATE
	cur_dist.del();
#endif
	new_dist.del();
	edge_weight.del();
    }

    struct info {
	double delay;
	float density;
	VID nactv;
	EID nacte;
	float meps;
#if AVG_DELTA
	float avg_delta;
#endif

	void dump( int i ) {
	    std::cerr << "Iteration " << i << ": " << delay
		      << " density: " << density
		      << " (E:" << nacte
		      << ", V:" << nactv
		      << ") M-edges/s: " << meps
#if AVG_DELTA
		      << " avg-delta: " << avg_delta
#endif
		      << "\n";
	}
    };

    struct stat {
	FloatTy shortest, longest;
    };

    void run() {
	const partitioner &part = GA.get_partitioner();
	VID n = GA.numVertices();
	EID m = GA.numEdges();

	timer tm_iter;
	tm_iter.start();

	iter = 0;
	active = 1;

#if !LEVEL_ASYNC || DEFERRED_UPDATE
	assert( cur_dist.get_ptr() );
#endif

#if VARIANT == 2
	// Analyse exponent range of edge weights, and infer floating-point
	// format.
	FloatTy f_min = std::numeric_limits<FloatTy>::max();
	FloatTy f_max = std::numeric_limits<FloatTy>::lowest();
	bool f_neg = false;
	bool f_zero = false;
	expr::array_ro<FloatTy,VID,var_min> a_min( &f_min );
	expr::array_ro<FloatTy,VID,var_max> a_max( &f_max );
	expr::array_ro<bool,VID,var_neg> a_neg( &f_neg );
	expr::array_ro<bool,VID,var_zero> a_zero( &f_zero );
#if 0
	// TODO: How to filter out padding edges? Need reference to graph.
	//       It would be faster to perform a flat edge map, however, to do
	//       so correctly requires code specific to the graph encoding
	//       to recognise padding edges. In the absence of such code, this
	//       version is disabled and we use full-blown edge map instead.
	make_lazy_executor( part )
	    .flat_edge_map( [&]( auto e ) {
		auto z =
		    expr::value<simd::ty<VID,decltype(e)::VL>,expr::vk_zero>();
		return make_seq(
		    a_neg[z] |= edge_weight[e] < _0,
		    a_zero[z] |= edge_weight[e] == _0,
		    a_min[z].min(
			expr::add_mask( expr::abs( edge_weight[e] ),
					edge_weight[e] != _0 ) ),
		    a_max[z].max( edge_weight[e] ) );
	    } )
	    .materialize();
#else
	api::edgemap(
	    GA,
	    api::relax( [&]( auto s, auto d, auto e ) {
		auto z =
		    expr::value<simd::ty<VID,decltype(e)::VL>,expr::vk_zero>();
		return make_seq(
		    a_neg[z] |= expr::cast<bool>( expr::make_unop_switch_to_vector( edge_weight[e] < _0 ) ),
		    a_zero[z] |= expr::cast<bool>( expr::make_unop_switch_to_vector( edge_weight[e] == _0 ) ),
		    a_min[z].min(
			expr::add_predicate( expr::abs( edge_weight[e] ),
					     edge_weight[e] != _0 ) ),
		    a_max[z].max( edge_weight[e] ) );
	    } ) )
	    .materialize();
#endif

	std::cout << "Estimated diameter: " << est_diam << "\n";
	std::cout << "Smallest edge_weight: " << f_min << "\n";
	std::cout << "Largest edge_weight: " << f_max << "\n";
	std::cout << "Any negative edge_weight: " << f_neg << "\n";
	std::cout << "Any zero edge_weight: " << f_zero << "\n";
	std::cout << "Longest path: " << f_max*(FloatTy)est_diam << "\n";

	// Determine configuration of custom floating-point format, assuming
	// maximum path length is est_diam hubs at most.
	// The shortest path length is 0 (from vertex to itself), however,
	// this is an exceptional case and we may detect and handle it, or
	// assume that the smallest path length can be 0.
	// Here, we require that 0 and infinity are correctly represented.
#if VARIANT != 0 && VARIANT != 2
	std::cout << "Smallest float: " << SFloatTy::min().get()
		  << " = " << (float)SFloatTy::min() << "\n";
	std::cout << "Largest float: " << SFloatTy::max().get()
		  << " = " << (float)SFloatTy::max() << "\n";
	assert( f_min >= (float)SFloatTy::min()
		&& f_max <= (float)SFloatTy::max()
		&& "edge weights out of range for data type" );
#elif VARIANT == 2
	// Need to add up to this many, worst case
	f_max *= (FloatTy)est_diam;

	// Ensure we can at least encode the value 1.0
	f_max = std::max( f_max, 1.0f );

	cfp_cfg::set_param( f_min, f_max, f_neg, /*f_zero*/true, false );
	auto vz = typename EncCur::stored_type(0.0f);
	std::cout << "Configuration (0.0f value): "
		  << vz << " -> " << (float)vz << "\n";
	auto vo = typename EncCur::stored_type(1.0f);
	std::cout << "Configuration (1.0f value): "
		  << vo << " -> " << (float)vo << "\n";
	auto vh = typename EncCur::stored_type(0.5f);
	std::cout << "Configuration (0.5f value): "
		  << vh << " -> " << (float)vh << "\n";
#endif

	static_assert( std::is_same_v<EdgeTy,FloatTy>,
		       "requires re-coding of edges" );
#endif // VARIANT != 0

	// Assign initial path lengths
	make_lazy_executor( part )
	    .vertex_map( [&]( auto v ) {
#if VARIANT == 0
		auto inf = expr::constant_val2<SFloatTy>(
		    v, std::numeric_limits<SFloatTy>::infinity() );
#elif VARIANT != 2
		auto inf = expr::constant_val2<SFloatTy>(
		    v, SFloatTy::max() );
#else
		auto inf = expr::constant_val2<SFloatTy>(
		    v, cfp_cfg::infinity() );
#endif
		 return expr::make_seq(
#if !LEVEL_ASYNC || DEFERRED_UPDATE
		     cur_dist[v] = inf,
#endif
		     new_dist[v] = inf );
	    } )
	    .materialize();
#if !LEVEL_ASYNC || DEFERRED_UPDATE
	cur_dist.get_ptr()[start] = 0.0f;
#endif
	new_dist.get_ptr()[start] = 0.0f;

	frontier F = frontier::create( n, start, GA.getOutDegree(start) );

	if( itimes ) {
	    info_buf.resize( iter+1 );
	    info_buf[iter].density = 0;
	    info_buf[iter].nactv = 0;
	    info_buf[iter].nacte = 0;
	    info_buf[iter].delay = tm_iter.next();
	    info_buf[iter].meps = 
		float(info_buf[iter].nacte) / info_buf[iter].delay / 1e6f;
#if AVG_DELTA
	    info_buf[iter].avg_delta = avg_distance();
#endif
	    if( debug )
		info_buf[iter].dump( iter );
	}

	++iter;

	// bool once = false; // true;
	// bool require_dense = false;
	EID postponed = 0;

	while( !F.isEmpty() /* || require_dense */ ) {  // iterate until all vertices visited

/*
use delta filtering on output of dense step (delta compared to smallest?)
in order to avoid redundant work.
Remember filtered out vertices for the final dense step
Somehow need a rule to decide when to pick those up...
if waiting until F is empty, we still need 3-4 dense steps

	       ... raise sp-threshold, would reduce number of dense phases?
	       but still take time, perhaps save two half dense phases' time.
	       */
    
#if 0
	    float density = F.density( m );
	    if( once && density > lo_frac && density < hi_frac
		&& F.nActiveVertices() > VID(float(n)*tgt_frac)
		&& postponed < n/2 ) {
		// Retain only the shortest vertices
		float threshold = determine_threshold( F, n*tgt_frac );

		// TODO: don't really need filter as the threshold method
		// has partitioned the data in larger than and less than
		// We can simply truncate the array
		frontier filtered;
		make_lazy_executor( part )
		    .vertex_filter(
			GA,
			F,
			filtered,
			[&]( auto v ) {
			    auto cT
				= expr::constant_val( new_dist[v], threshold );
			    return new_dist[v] <= cT;
			} )
		    .materialize();
		std::cerr << "F: v=" << F.nActiveVertices()
			  <<  " e=" << F.nActiveEdges()
			  << " d=" << F.density( m )
			  << "\n";
		std::cerr << "filtered: v=" << filtered.nActiveVertices()
			  <<  " e=" << filtered.nActiveEdges()
			  << " d=" << filtered.density( m )
			  << "\n";
		postponed += F.nActiveVertices() - filtered.nActiveVertices();
		F.del();
		F = filtered;
		// Enforce sparse traversal regardless of degrees
		F.setActiveCounts( F.nActiveVertices(), F.nActiveVertices() );
		require_dense = true;
	    } else if( density >= hi_frac ) {
		once = false;
	    }
#endif

	    frontier output;
	    // Traverse edges, remove duplicate live destinations.
/*
	    if( F.density( m ) >= 0.50 )
		require_dense = false;
	    else if( F.isEmpty() ) {
		assert( require_dense );
		F.del();
		F = frontier::all_true( n, m );
	    }
*/
#if UNCOND_EXEC
	    auto filter_strength = api::weak;
#else
	    auto filter_strength = api::strong;
#endif
	    // TODO: config to not calculate nactv, nacte?
	    api::edgemap(
		GA, 
#if SP_THRESHOLD >= 0
		api::config( api::frac_threshold( SP_THRESHOLD ) ),
#endif
#if DEFERRED_UPDATE
		// Allow for (sparse) push to use reduction update to inform
		// frontier rather than using the method.
		api::record( output,
			     // api::reduction_or_method,
			     [&] ( auto d ) {
				 return cur_dist[d] != new_dist[d]; },
			     api::strong ),
#else
		// Do not allow unbacked frontiers
		api::record( output, api::reduction, api::strong ),
#endif
#if FUSION
		// TODO: parameter 0.1 to be tuned similarly to delta in DSSSP
		api::fusion( [&]( auto v ) {
		    auto inf = expr::constant_val2<FloatTy>(
			v, std::numeric_limits<FloatTy>::infinity() );
		    return expr::iif( new_dist[v] < cur_dist[v] * expr::constant_val( cur_dist[v], 0.1 ) && cur_dist[v] != inf, _0, _1 );
		} ),
#endif
		api::filter( filter_strength, api::src, F ),
		api::relax( [&]( auto s, auto d, auto e ) {
#if LEVEL_ASYNC
		    return new_dist[d].min( new_dist[s] + edge_weight[e] );
#else
		    return new_dist[d].min( cur_dist[s] + edge_weight[e] );
#endif
		} )
		).materialize();


#if !LEVEL_ASYNC || DEFERRED_UPDATE
	    if( output.getType() == frontier_type::ft_unbacked ) {
		make_lazy_executor( part )
		    .vertex_map( [&]( auto v ) {
			return cur_dist[v] = new_dist[v];
		    } )
		    .materialize();
	    } else {
		make_lazy_executor( part )
		    .vertex_map( output, [&]( auto v ) {
			return cur_dist[v] = new_dist[v];
		    } )
		    .materialize();
	    }
#endif

	    active += output.nActiveVertices();

	    if( itimes ) {
		info_buf.resize( iter+1 );
		info_buf[iter].density = F.density( m );
		info_buf[iter].nactv = F.nActiveVertices();
		info_buf[iter].nacte = F.nActiveEdges();
		info_buf[iter].delay = tm_iter.next();
		info_buf[iter].meps = 
		    float(info_buf[iter].nacte) / info_buf[iter].delay / 1e6f;
#if AVG_DELTA
		info_buf[iter].avg_delta = avg_distance();
#endif
		if( debug )
		    info_buf[iter].dump( iter );
	    }

	    // Cleanup frontier
	    F.del();
	    F  = output;

	    ++iter;
	}

	F.del();
    }

private:
#if AVG_DELTA
    float avg_distance() {
	if( !have_ref )
	    return 0;
	
	double d = 0;
	expr::array_ro<double,VID,var_d> a_delta( &d );
	make_lazy_executor( GA.get_partitioner() )
	    .vertex_scan( [&]( auto v ) {
		auto z = expr::zero_val( v );
		auto inf = expr::constant_val2<FloatTy>(
		    v, std::numeric_limits<FloatTy>::infinity() );
		return a_delta[z] += expr::cast<double>(
		    expr::iif( new_dist[v] == inf,
			       expr::abs( new_dist[v] - (*ref_sol)[v] ),
			       expr::iif( (*ref_sol)[v] == inf,
					  (*ref_sol)[v],
					  _0 ) ) );
	    } )
	    .materialize();
	return d / double( GA.numVertices() );
    }
#endif
    
    FloatTy determine_threshold( frontier & F, VID target_vertices ) {
	F.toSparse( GA.get_partitioner() );
	VID n = F.nActiveVertices();
	VID * s = F.getSparse();
	return randomized_select( s, 0, n, target_vertices );
    }

    VID partition( VID * s, VID p, VID r ) {
	FloatTy x = new_dist[s[p]];
	VID i = p-1;
	VID j = r;
	while( true ) {
	    do {
		--j;
	    } while( new_dist[s[j]] > x );
	    do {
		++i;
	    } while( new_dist[s[i]] <= x );
	    if( i < j )
		std::swap( s[i], s[j] );
	    else
		return j;
	}
    }

    VID randomized_partition( VID * s, VID p, VID r ) {
	VID i = p + ( rand() % (r - p) );
	std::swap( s[i], s[p] );
	return partition( s, p, r );
    }

    // i: select i-th smallest value 
    // p: start index of array
    // r: end index of array
    FloatTy randomized_select( VID * s, VID p, VID r, VID i ) {
	if( p == r ) // only one element
	    return new_dist[s[p]];

	VID q = randomized_partition( s, p, r );
	VID k = q - p + 1;
	if( i <= k )
	    return randomized_select( s, p, q, i );
	else
	    return randomized_select( s, q+1, r, i-k );
    }

public:
    void post_process( stat & stat_buf ) {
	if( itimes ) {
	    for( int i=0; i < iter; ++i )
		info_buf[i].dump( i );
	}
#if AVG_DELTA
	make_lazy_executor( GA.get_partitioner() )
	    .vertex_map( [&]( auto v ) {
		return (*ref_sol)[v] = new_dist[v];
	    } )
	    .materialize();
	have_ref = true;
#endif
    }

    static void report( const std::vector<stat> & stat_buf ) {
	bool fail = false;
	for( int i=1; i < stat_buf.size(); ++i ) {
	    if( stat_buf[i].shortest != stat_buf[0].shortest ) {
		std::cerr << "ERROR: round " << i << " has shortest path="
			  << stat_buf[i].shortest
			  << " while round 0 has shortest path="
			  << stat_buf[0].shortest << "\n";
		fail = true;
	    }
	    if( stat_buf[i].longest != stat_buf[0].longest ) {
		std::cerr << "ERROR: round " << i << " has longest path="
			  << stat_buf[i].longest
			  << " while round 0 has longest path="
			  << stat_buf[0].longest << "\n";
		fail = true;
	    }
	}
	if( fail )
	    abort();
    }

    void validate( stat & stat_buf ) {
	VID n = GA.numVertices();

	FloatTy shortest, longest;
	std::tie( shortest, longest ) = find_min_max( GA, new_dist );

	stat_buf.shortest = shortest;
	stat_buf.longest = longest;

	std::cout << "Shortest path from " << start
		  << " (original: " << GA.originalID( start ) << ") : "
		  << shortest << "\n";
	std::cout << "Longest path from " << start
		  << " (original: " << GA.originalID( start ) << ") : "
		  << longest << "\n";

	std::cout << "Number of activated vertices: " << active << "\n";
	std::cout << "Number of vertices: " << n << "\n";

#if BF_DEBUG
	all.calculateActiveCounts( GA.getCSR() );
	std::cout << "all vertices: " << all.nActiveVertices() << "\n";
#endif
    }

private:
    const GraphType & GA;
    bool itimes, debug, calculate_active;
    int iter;
    VID start, active;
    VID est_diam;
#if !LEVEL_ASYNC || DEFERRED_UPDATE
    api::vertexprop<FloatTy,VID,var_cur,EncCur> cur_dist;
#endif
    api::vertexprop<FloatTy,VID,var_new,EncNew> new_dist;
    api::edgeprop<FloatTy,EID,expr::vk_eweight,EncEdge> edge_weight;
    float lo_frac, hi_frac, tgt_frac;
    std::vector<info> info_buf;
};

template <class GraphType>
using Benchmark = BF<GraphType>;

#include "driver.C"
