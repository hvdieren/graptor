/*
 * TODO:
 * ? after a fusion iteration, bucket should be empty?
 *   if not empty, then it should contain only duplicate vertices (already done)
 */

#include <cmath>

#include "graptor/graptor.h"
#include "graptor/api.h"
#include "path.h"
#include "buckets.h"

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
#elif VARIANT == 1 || VARIANT == 11
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
#elif VARIANT == 3 || VARIANT == 13
using FloatTy = float;
using SFloatTy = scustomfp<false,5,11,true,21>;
using EncCur = array_encoding_wide<SFloatTy>;
using EncNew = EncCur;
using EncEdge = array_encoding<FloatTy>;
#elif VARIANT == 4 || VARIANT == 14
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
    var_dstep_delta = 8,
    var_neg = 10,
    var_zero = 11
};

struct bucket_fn {
    using ID = VID;
    using BID = std::make_unsigned_t<ID>;
    
    bucket_fn( SFloatTy * dist, FloatTy delta )
	: m_dist( dist ), m_delta( delta ) { }

    BID operator() ( VID v, BID cur, BID oflow ) const {
	if( v == std::numeric_limits<VID>::max() )
	    return std::numeric_limits<BID>::max();
	BID bkt = ((FloatTy)m_dist[v]) / m_delta;
	if( bkt < cur )
	    return std::numeric_limits<BID>::max();
	else
	    return bkt;
    }

/*
    template<VID Scale>
    FloatTy get_scaled( VID v ) const {
	return (VID)( ((FloatTy)m_dist[v]) / ( m_delta / (FloatTy)Scale ) );
    }

    FloatTy get( VID v ) const {
	return m_dist[v];
    }
*/
    
private:
    SFloatTy * m_dist;
    FloatTy m_delta;
};

template <class GraphType>
class DSSSP {
public:
    DSSSP( GraphType & _GA, commandLine & P )
	: GA( _GA ),
	  num_buckets( P.getOptionLongValue( "-sssp:buckets", 127 ) ),
	  info_buf( 60 ),
#if !LEVEL_ASYNC || DEFERRED_UPDATE
	  cur_dist( GA.get_partitioner(), "current distance" ),
	  cur_dist_final( GA.get_partitioner(), "current distance (final)" ),
#endif
	  new_dist( GA.get_partitioner(), "new distance" ),
	  new_dist_final( GA.get_partitioner(), "new distance (final)" ),
	  edge_weight( GA.get_partitioner(), "edge weight" ) {
	itimes = P.getOption( "-itimes" );
	debug = P.getOption( "-debug" );
	calculate_active = P.getOption( "-cactive" );
	est_diam = P.getOptionLongValue( "-sssp:diam", 20 );
	start = GA.remapID( P.getOptionLongValue( "-start", 0 ) );
	delta = P.getOptionDoubleValue( "-sssp:delta", 1.0 );

	if( GA.getWeights() == nullptr ) {
	    std::cerr << "DSSSP requires edge weights. Terminating.\n";
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
	    std::cerr << "num_buckets=" << num_buckets << "\n";
	    std::cerr << "delta=" << delta << "\n";
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
    ~DSSSP() {
#if !LEVEL_ASYNC || DEFERRED_UPDATE
	cur_dist.del();
	cur_dist_final.del();
#endif
	new_dist.del();
	new_dist_final.del();
	edge_weight.del();
    }

    struct info {
	double delay;
	float density;
	VID nactv;
	EID nacte;
	VID cur_bkt;
	float meps;

	void dump( int i ) {
	    std::cerr << "Iteration " << i << ": " << delay
		      << " density: " << density
		      << " (E:" << nacte
		      << ", V:" << nactv
		      << ") bkt: " << cur_bkt
		      << " M-edges/s: " << meps
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

	expr::array_ro<FloatTy,VID,var_dstep_delta> a_delta( &delta );

#if !LEVEL_ASYNC || DEFERRED_UPDATE
	assert( cur_dist.get_ptr() );
	assert( cur_dist_final.get_ptr() );
#endif

#if VARIANT == 2 || VARIANT >= 10
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

#if VARIANT >= 10
	FloatTy e_min = f_min;
	FloatTy e_max = f_max;
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

	// Create bucket structure
	buckets<VID,bucket_fn>
	    bkts( n, num_buckets, bucket_fn( new_dist.get_ptr(), delta ) );

	// Place start vertex in first bucket
	bkts.insert( start, 0 );

	if( itimes ) {
	    info_buf.resize( iter+1 );
	    info_buf[iter].density = 0;
	    info_buf[iter].nactv = 0;
	    info_buf[iter].nacte = 0;
	    info_buf[iter].delay = tm_iter.next();
	    info_buf[iter].cur_bkt = 0;
	    info_buf[iter].meps = 
		float(info_buf[iter].nacte) / info_buf[iter].delay / 1e6f;
	    if( debug )
		info_buf[iter].dump( iter );
	}

#if VARIANT >= 10
	FloatTy p_min = std::numeric_limits<FloatTy>::max();
	FloatTy p_max = std::numeric_limits<FloatTy>::lowest();
#endif

	++iter;

	while( !bkts.empty() ) {  // iterate until all vertices visited
	    frontier F = bkts.next_bucket();
	    VID cur_bkt = bkts.get_current_bucket();

	    // There will be duplicate vertices in the list and also vertices
	    // whose distance from the source is less than the current bucket.
	    // Filter out the vertices that already completed.
	    // As per GAPBS, we tolerate duplicates of non-completed vertices.
	    // As we don't have a filter-source-by-method ability in edgemap,
	    // adjust the frontier here.
	    frontier unique;
	    make_lazy_executor( part )
		.vertex_filter(
		    GA, F, unique,
		    [&]( auto v ) {
			auto threshold = expr::constant_val2<FloatTy>(
			    v, delta * (FloatTy)cur_bkt );
			return new_dist[v] >= threshold;
		    } )
		.materialize();
	    F.del();
	    F = unique;

#if 0
	    std::cerr << "threshold: " << delta * (FloatTy)cur_bkt << "\n";
	    F.toSparse( part );
	    for( VID i=0, e=F.nActiveVertices(); i < e; ++i ) {
		VID v = F.getSparse()[i];
		std::cerr << ' ' << v << '(' << new_dist[v] << ')';
	    }
	    std::cerr << "\n";
#endif

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
		api::filter( filter_strength, api::src, F ),
#if FUSION
		api::config( api::always_sparse ),
		api::fusion( [&]( auto d ) {
		    // We are not using the feature to avoid inserting vertices 
		    // multiple times, especially in the overflow bucket (i.e.,
		    // return -1 from api::fusion). This is because vertices
		    // are inserted in the buckets only when they are woken up
		    // and we cannot tell easily whether a vertex is already
		    // present in the overflow bucket or not.
// TODO: GAPBS runs up to 1000 buckets ahead + sorts those buckets locally in the edgemap
		    auto threshold = expr::constant_val2<FloatTy>(
			d, delta * (FloatTy)(1+cur_bkt) );
		    return expr::iif( new_dist[d] <= threshold,
				      _0, _1 ); // int
		} ),
#else // no FUSION
#if SP_THRESHOLD >= 0
		api::config( api::frac_threshold( SP_THRESHOLD ) ),
#endif
#endif
		api::relax( [&]( auto s, auto d, auto e ) {
		    auto threshold = expr::constant_val2<FloatTy>(
			d, delta * (FloatTy)(1+cur_bkt) );
		    return expr::iif(
			new_dist[d].min( new_dist[s] + edge_weight[e] ),
			expr::_1s,
			expr::iif( new_dist[d] <= threshold,
				   _0, _1 ) ); // int
/*
#if LEVEL_ASYNC
		    return new_dist[d].min( new_dist[s] + edge_weight[e] );
#else
		    return new_dist[d].min( cur_dist[s] + edge_weight[e] );
#endif
*/
		} )
		).materialize();
			   

#if ( !LEVEL_ASYNC || DEFERRED_UPDATE ) && 0
	    // A sparse frontier that cannot be automatically converted due to
	    // being unbacked.
	    if( F.getType() == frontier_type::ft_unbacked
		&& api::default_threshold().is_sparse( output, m ) ) {
		assert( 0 && "Should be handled by edgemap" );
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

#if VARIANT >= 10
	    bool needs_break = false;
	    {
		// Note that p_max should be over-estimated because a long
		// path may shorten over time
		expr::array_ro<FloatTy,VID,var_min> a_min( &p_min );
		expr::array_ro<FloatTy,VID,var_max> a_max( &p_max );
		make_lazy_executor( part )
		    .vertex_scan( output, [&]( auto v ) {
			auto z =
			    expr::value<simd::ty<VID,decltype(v)::VL>,expr::vk_zero>();
			auto inf = expr::constant_val2<FloatTy>(
			    v, (FloatTy)SFloatTy::max() );
			auto w = expr::cast<FloatTy>( cur_dist[v] );
			return expr::set_mask(
			    w != inf && w != _0,
			    expr::make_seq(
				a_min[z].min( w ),
				a_max[z].max( w ) ) );
		} )
		.materialize();
		using ft = fp_traits<SFloatTy>;
		FloatTy ep_max = p_max + e_max;
		if( ep_max >= e_min * (FloatTy)(1<<(SFloatTy::mantissa_bits-2))
		    ) {
		    std::cout << "Break out of low-precision loop because:\n";
		    std::cout << "   shortest path: " << p_min << "\n";
		    std::cout << "   longest path: " << p_max << "\n";
		    std::cout << "   largest weight: " << e_min << "\n";
		    std::cout << "   smallest weight: " << e_max << "\n";
		    std::cout << "   worst-case path: " << ep_max << "\n";
		    std::cout << "   mantissa bits: " << ft::mantissa_bits
			      << "\n";
		    needs_break = true;
		}
	    }
#endif

/*
	    for( VID v=0; v < 128; ++v ) {
		std::cerr << ' ' << (float)new_dist[v];
	    }
	    std::cerr << '\n';
*/

	    bkts.update_buckets( part, output );

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
		info_buf[iter].cur_bkt = cur_bkt;
		info_buf[iter].meps = 
		    float(info_buf[iter].nacte) / info_buf[iter].delay / 1e6f;
		if( debug )
		    info_buf[iter].dump( iter );
	    }

	    // Cleanup frontier
	    F.del();
	    output.del();

	    ++iter;

#if VARIANT >= 10
	    if( needs_break )
		break;
#endif
	}
	
	// Copy to wider weights.
	make_lazy_executor( part )
	    .vertex_map( [&]( auto v ) {
		return expr::make_seq(
#if !LEVEL_ASYNC || DEFERRED_UPDATE
		    cur_dist_final[v] = cur_dist[v],
#endif
		    new_dist_final[v] = new_dist[v] );
	    } )
	    .materialize();

#if VARIANT >= 10
	// need a second phase
	while( !bkts.empty() ) {  // iterate until all vertices visited
	    frontier F = bkts.next_bucket();
	    F.calculateActiveCounts( GA.getCSR(), part, F.nActiveVertices() );

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
#if DEFERRED_UPDATE
		// TODO: Do we apply the deferred update function in sparse
		//       edgemap?
		api::record( output,
			     [&] ( auto d ) {
			     return cur_dist_final[d] != new_dist_final[d]; },
			     api::strong ),
#else
		// Do not allow unbacked frontiers
		api::record( output, api::reduction, api::strong ),
#endif
		api::filter( filter_strength, api::src, F ),
		api::relax( [&]( auto s, auto d, auto e ) {
#if LEVEL_ASYNC
		    return new_dist_final[d].min( new_dist_final[s] + edge_weight[e] );
#else
		    return new_dist_final[d].min( cur_dist_final[s] + edge_weight[e] );
#endif
		} )
		).materialize();

	    bkts.update_buckets( part, output );

#if !LEVEL_ASYNC || DEFERRED_UPDATE
	    if( output.getType() == frontier_type::ft_unbacked ) {
		make_lazy_executor( part )
		    .vertex_map( [&]( auto v ) {
			return cur_dist_final[v] = new_dist_final[v];
		    } )
		    .materialize();
	    } else {
		make_lazy_executor( part )
		    .vertex_map( output, [&]( auto v ) {
			return cur_dist_final[v] = new_dist_final[v];
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
	    output.del();

	    ++iter;
	}
#endif // VARIANT >= 10
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
	std::tie( shortest, longest ) = find_min_max( GA, new_dist_final );

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
    int num_buckets;
    bool itimes, debug, calculate_active;
    int iter;
    VID start, active;
    VID est_diam;
    FloatTy delta;
#if !LEVEL_ASYNC || DEFERRED_UPDATE
    api::vertexprop<FloatTy,VID,var_cur,EncCur> cur_dist;
    api::vertexprop<FloatTy,VID,var_cur2,EncCur> cur_dist_final;
#endif
    api::vertexprop<FloatTy,VID,var_new,EncNew> new_dist;
    api::vertexprop<FloatTy,VID,var_new2,EncNew> new_dist_final;
    api::edgeprop<FloatTy,EID,expr::aid_eweight,EncEdge> edge_weight;
    std::vector<info> info_buf;
};

template <class GraphType>
using Benchmark = DSSSP<GraphType>;

#include "driver.C"
