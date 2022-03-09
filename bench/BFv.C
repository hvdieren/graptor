#include <cmath>

#include "graptor/graptor.h"
#include "graptor/api.h"
#include "path.h"

using expr::_0;
using expr::_1;

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
    var_zero = 11
};

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

	if( GA.getWeights() == nullptr ) {
	    std::cerr << "BF requires edge weights. Terminating.\n";
	    exit( 1 );
	}

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

	void dump( int i ) {
	    std::cerr << "Iteration " << i << ": " << delay
		      << " density: " << density
		      << " (E:" << nacte
		      << ", V:" << nactv
		      << ") M-edges/s: " << meps
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
	    if( debug )
		info_buf[iter].dump( iter );
	}

	++iter;

	while( !F.isEmpty() ) {  // iterate until all vertices visited
	    // Traverse edges, remove duplicate live destinations.
	    frontier output;
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
			     api::reduction_or_method,
			     [&] ( auto d ) {
			     return cur_dist[d] != new_dist[d]; },
			     api::strong ),
#else
		// Do not allow unbacked frontiers
		api::record( output, api::reduction, api::strong ),
#endif
#if FUSION
		api::fusion( [&]( auto v ) {
		    return expr::true_val( v );
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
		info_buf[iter].density = F.density( GA.numEdges() );
		info_buf[iter].nactv = F.nActiveVertices();
		info_buf[iter].nacte = F.nActiveEdges();
		info_buf[iter].delay = tm_iter.next();
		info_buf[iter].meps = 
		    float(info_buf[iter].nacte) / info_buf[iter].delay / 1e6f;
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

    void post_process( stat & stat_buf ) {
	if( itimes ) {
	    for( int i=0; i < iter; ++i )
		info_buf[i].dump( i );
	}
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
    std::vector<info> info_buf;
};

template <class GraphType>
using Benchmark = BF<GraphType>;

#include "driver.C"
