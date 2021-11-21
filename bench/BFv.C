#include <cmath>

#include "graptor/graptor.h"
#include "graptor/api.h"
#include "path.h"

using expr::_0;

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

#if VARIANT == 0
using FloatTy = float;
using SFloatTy = float;
using EncCur = array_encoding<FloatTy>;
using EncNew = array_encoding<FloatTy>;
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
    var_weight = 2,
    var_min = 3,
    var_max = 4,
    var_remap = 5,
    var_cur2 = 6,
    var_new2 = 7
};

template <class GraphType>
class BFv {
public:
    BFv( GraphType & _GA, commandLine & P )
	: GA( _GA ), info_buf( 60 ),
#if !LEVEL_ASYNC || DEFERRED_UPDATE
	  cur_dist( GA.get_partitioner(), "current distance" ),
#endif
	  new_dist( GA.get_partitioner(), "new distance" ),
	  edge_weight( GA.get_partitioner(), "edge weight" ) {
	itimes = P.getOption( "-itimes" );
	debug = P.getOption( "-debug" );
	calculate_active = P.getOption( "-cactive" );
	est_diam = P.getOptionLongValue( "-bf:diam", 20 );
	start = GA.remapID( P.getOptionLongValue( "-start", 0 ) );

	// Initialise (random) edge weights
	VID n = GA.numVertices();
	auto remap = GA.get_remapper();
	if constexpr ( remap.is_idempotent() ) {
	    api::edgemap(
		GA,
		api::relax( [&]( auto s, auto d, auto e ) {
		    auto cR = expr::constant_val2<VID>( d, 127 );
		    auto cO = expr::constant_val2<FloatTy>( d, 64 );
		    auto cW = expr::constant_val2<FloatTy>( d, 64 * 16 );
		    auto val = ( s + d ) & cR;
		    return expr::make_seq(
			edge_weight[e] =
			expr::constant_val2<FloatTy>( d, 0.125 )
			+ ( expr::cast<FloatTy>( val ) - cO ) / cW,
			expr::true_val( d ) );
		} )
		).materialize();
	} else {
	    expr::array_ro<VID,VID,var_remap> a_remap( remap.getOrigIDPtr() );
	    api::edgemap(
		GA,
		api::relax( [&]( auto s, auto d, auto e ) {
		    auto ss = a_remap[s];
		    auto dd = a_remap[d];
		    auto cR = expr::constant_val2<VID>( d, 127 );
		    auto cO = expr::constant_val2<FloatTy>( d, 64 );
		    auto cW = expr::constant_val2<FloatTy>( d, 64 * 16 );
		    auto val = ( ss + dd ) & cR;
		    return expr::make_seq(
			edge_weight[e] =
			expr::constant_val2<FloatTy>( d, 0.125 )
			+ ( expr::cast<FloatTy>( val ) - cO ) / cW,
			expr::true_val( d ) );
		} )
		).materialize();
	}

	static bool once = false;
	if( !once ) {
	    once = true;
	    std::cerr << "DEFERRED_UPDATE=" << DEFERRED_UPDATE << "\n";
	    std::cerr << "UNCOND_EXEC=" << UNCOND_EXEC << "\n";
	    std::cerr << "LEVEL_ASYNC=" << LEVEL_ASYNC << "\n";
	    std::cerr << "CONVERGENCE=" << CONVERGENCE << "\n";
	    std::cerr << "MEMO=" << MEMO << "\n";
	    std::cerr << "sizeof(FloatTy)=" << sizeof(FloatTy) << "\n";
	    std::cerr << "sizeof(SFloatTy)=" << sizeof(SFloatTy) << "\n";
	}
    }
    ~BFv() {
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

	void dump( int i ) {
	    std::cerr << "Iteration " << i << ": " << delay
		      << " density: " << density
		      << " (E:" << nacte
		      << ", V:" << nactv
		      << ")\n";
	}
    };

    struct stat { };

    void run() {
	const partitioner &part = GA.get_partitioner();
	VID n = GA.numVertices();
	EID m = GA.numEdges();

#if !LEVEL_ASYNC || DEFERRED_UPDATE
	assert( cur_dist.get_ptr() );
#endif

#if VARIANT != 0
	// Analyse exponent range of edge weights, and infer floating-point
	// format.
	FloatTy f_min = std::numeric_limits<FloatTy>::max();
	FloatTy f_max = -std::numeric_limits<FloatTy>::max();
	expr::array_ro<FloatTy,VID,var_min> a_min( &f_min );
	expr::array_ro<FloatTy,VID,var_max> a_max( &f_max );
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
		return make_seq( a_min[z].min( edge_weight[e] ),
				 a_max[z].max( edge_weight[e] ) );
	    } )
	    .materialize();
#else
	api::edgemap(
	    GA,
	    api::relax( [&]( auto s, auto d, auto e ) {
		auto z =
		    expr::value<simd::ty<VID,decltype(e)::VL>,expr::vk_zero>();
		return make_seq( a_min[z].min( edge_weight[e] ),
				 a_max[z].max( edge_weight[e] ) );
	    } ) )
	    .materialize();
#endif

	std::cout << "Estimated diameter: " << est_diam << "\n";
	std::cout << "Smallest edge_weight: " << f_min << "\n";
	std::cout << "Largest edge_weight: " << f_max << "\n";
	std::cout << "Longest path: " << f_max*(FloatTy)est_diam << "\n";

	static_assert( std::is_same_v<EdgeTy,FloatTy>,
		       "requires re-coding of edges" );
#endif // VARIANT != 0

	// Assign initial path lengths
	make_lazy_executor( part )
	    .vertex_map( [&]( auto v ) {
#if VARIANT == 0
		auto inf = expr::constant_val2<SFloatTy>(
		    v, std::numeric_limits<SFloatTy>::infinity() );
#elif VARIANT == 1 || VARIANT == 3
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
	    }
		)
	    .materialize();
#if !LEVEL_ASYNC || DEFERRED_UPDATE
	cur_dist.get_ptr()[start] = 0.0f;
#endif
	new_dist.get_ptr()[start] = 0.0f;

	// Create initial frontier
	frontier F = frontier::create( n, start, GA.getOutDegree(start) );

	iter = 0;
	active = 1;

	while( !F.isEmpty() ) {  // iterate until all vertices visited
	    timer tm_iter;
	    tm_iter.start();

#if VARIANT == 3
#if 1
	    f_max = -std::numeric_limits<FloatTy>::max();
	    make_lazy_executor( part )
		.vertex_scan( [&]( auto v ) {
		    auto z =
			expr::value<simd::ty<VID,decltype(v)::VL>,expr::vk_zero>();
		    auto inf = expr::constant_val2<FloatTy>(
			v, (FloatTy)SFloatTy::max() );
		    return expr::set_mask( cur_dist[v] != inf,
					   a_max[z].max( cur_dist[v] ) );
		} )
		.materialize();
	    // std::cout << "Shortest path: " << f_min << "\n";
	    // std::cout << "Longest path: " << f_max << "\n";
	    if( f_max >= f_min * 256 )
		break;
#endif
#endif

	    // If we have a sparse frontier that is unbacked, we cannot
	    // do sparse processing. Hence, perform another dense edgemap
	    // but one which produces a backed frontier.
	    
	    // Traverse edges, remove duplicate live destinations.
	    frontier output;
#if UNCOND_EXEC
	    auto filter_strength = api::weak;
#else
	    auto filter_strength = api::strong;
#endif

	    api::edgemap(
		GA, 
#if DEFERRED_UPDATE
		api::record( output,
			     [&] ( auto d ) {
				 return cur_dist[d] != new_dist[d]; },
			     filter_strength ),
#else
		api::record( output, api::reduction, filter_strength ),
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

#if 0
	    map_vertexL( part,
			 [&]( VID i ) {
			     if( cur_dist[i] > new_dist[i]
				 && cur_dist[i] != (float)SFloatTy::max() ) {
				 std::cout << "       decreasing path: "
					   << " i=" << i
					   << " cur=" << cur_dist[i]
					   << ", " << cur_dist.get_ptr()[i]
					   << " new=" << new_dist[i]
					   << ", " << new_dist.get_ptr()[i]
					   << "\n";
			     } } );
#endif

#if 0
	    std::cout << "F     : " << F << "\n";
	    std::cout << "output: " << output << "\n";
#if !LEVEL_ASYNC || DEFERRED_UPDATE
	    print( std::cout, GA.get_remapper(), part, cur_dist );
#endif
	    print( std::cout, GA.get_remapper(), part, new_dist );

	    std::cout << "F[29] " << F.is_set( 29 ) << "\n";
	    std::cout << "output[29] " << output.is_set( 29 ) << "\n";
#if !LEVEL_ASYNC || DEFERRED_UPDATE
	    std::cout << "cur_dist[29] " << cur_dist[29] << "\n";
#endif
	    std::cout << "new_dist[29] " << new_dist[29] << "\n";

#if !LEVEL_ASYNC || DEFERRED_UPDATE
	    if( output.getType() == frontier_type::ft_logical4 ) {
		logical<4> * f
		    = output.template getDense<frontier_type::ft_logical4>();
		VID k = 0;
		for( VID v=0; v < n; ++v ) {
		    if( cur_dist[v] != new_dist[v] ) {
/*
			std::cout << "diff: " << cur_dist[v] << ' '
				  << new_dist[v] << " k=" << k
				  << " v=" << v << "\n";
*/
			assert( f[v] != 0 );
			++k;
		    } else
			assert( f[v] == 0 );
		}
		assert( k == output.nActiveVertices() );
	    }
#endif
#endif

#if !LEVEL_ASYNC || DEFERRED_UPDATE
	    // A sparse frontier that cannot be automatically converted due to
	    // being unbacked.
	    if( F.getType() == frontier_type::ft_unbacked
		&& api::default_threshold().is_sparse( output, m ) ) {
		frontier ftrue = frontier::all_true( n, m );
		frontier output2;
		make_lazy_executor( part )
		    .vertex_filter( GA, ftrue, output2,
				    [&]( auto v ) {
					return cur_dist[v] != new_dist[v];
				    } )
		    .materialize();

		assert( output.nActiveVertices() == output2.nActiveVertices()
			&& "check number of vertices is the same" );
		assert( output.nActiveEdges() == output2.nActiveEdges()
			&& "check number of edges is the same" );

		output.del();
		output = output2;
	    }
#endif

#if !LEVEL_ASYNC || DEFERRED_UPDATE
	    if( output.getType() == frontier_type::ft_unbacked /*|| 1*/ ) {
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
	    // maintain_copies( part, output, cur_dist.get_ptr(),
	    // new_dist.get_ptr() );
#endif

	    active += output.nActiveVertices();

	    if( itimes ) {
		info_buf.resize( iter+1 );
		info_buf[iter].density = F.density( GA.numEdges() );
		info_buf[iter].nactv = F.nActiveVertices();
		info_buf[iter].nacte = F.nActiveEdges();
		info_buf[iter].delay = tm_iter.next();
		if( debug )
		    info_buf[iter].dump( iter );
	    }

	    // Cleanup old frontier
	    F.del();
	    F = output;

	    ++iter;
	}

#if VARIANT == 3
	if( !F.isEmpty() ) {  // need a second phase
#if !LEVEL_ASYNC || DEFERRED_UPDATE
	    api::vertexprop<FloatTy,VID,var_cur2> cur_dist2(
		GA.get_partitioner(), "current distance (2)" );
#endif
	    api::vertexprop<FloatTy,VID,var_new2> new_dist2(
		GA.get_partitioner(), "new distance (2)" );

	    make_lazy_executor( part )
		.vertex_map( [&]( auto v ) {
		    return expr::make_seq(
			cur_dist2[v] = cur_dist[v],
			new_dist2[v] = new_dist[v] );
		} )
		.materialize();

	    while( !F.isEmpty() ) {  // iterate until all vertices visited
		timer tm_iter;
		tm_iter.start();

		// Traverse edges, remove duplicate live destinations.
		frontier output;
#if UNCOND_EXEC
		auto filter_strength = api::weak;
#else
		auto filter_strength = api::strong;
#endif
		api::edgemap(
		    GA, 
#if DEFERRED_UPDATE
		    api::record( output,
				 [&] ( auto d ) {
				     return cur_dist2[d] != new_dist2[d]; },
				 filter_strength ),
#else
		    api::record( output, api::reduction, filter_strength ),
#endif
		    api::filter( filter_strength, api::src, F ),
		    api::relax( [&]( auto s, auto d, auto e ) {
#if LEVEL_ASYNC
			return new_dist2[d].min( new_dist2[s] + edge_weight[e] );
#else
			return new_dist2[d].min( cur_dist2[s] + edge_weight[e] );
#endif
				} )
		    ).materialize();

#if !LEVEL_ASYNC || DEFERRED_UPDATE
		if( output.getType() == frontier_type::ft_unbacked || 1 ) {
		    make_lazy_executor( part )
			.vertex_map( [&]( auto v ) {
			    return cur_dist2[v] = new_dist2[v];
			} )
			.materialize();
		} else {
		    make_lazy_executor( part )
			.vertex_map( output, [&]( auto v ) {
			    return cur_dist2[v] = new_dist2[v];
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
		    if( debug )
			info_buf[iter].dump( iter );
		}

		// Cleanup old frontier
		F.del();
		F = output;

		++iter;
	    }

	    make_lazy_executor( part )
		.vertex_map( [&]( auto v ) {
		    return expr::make_seq(
			cur_dist[v] = cur_dist2[v],
			new_dist[v] = new_dist2[v] );
		} )
		.materialize();

	    cur_dist2.del();
	    new_dist2.del();
	}
#endif // VARIANT == 3

	F.del();
    }

    void post_process( stat & stat_buf ) {
	if( itimes ) {
	    for( int i=0; i < iter; ++i )
		info_buf[i].dump( i );
	}
    }

    static void report( const std::vector<stat> & stat_buf ) { }

    void validate( stat & stat_buf ) {
	VID n = GA.numVertices();
	const partitioner &part = GA.get_partitioner();

	FloatTy shortest, longest;
	std::tie( shortest, longest ) = find_min_max( GA, new_dist );

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
    api::edgeprop<FloatTy,EID,var_weight,EncEdge> edge_weight;
    std::vector<info> info_buf;
};

template <class GraphType>
using Benchmark = BFv<GraphType>;

#include "driver.C"
