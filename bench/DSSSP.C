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
using SFloatTy = scustomfp<false,7,9,true,6>;
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
using SFloatTy = scustomfp<false,7,9,true,11>;
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
    var_weight = 2,
    var_min = 3,
    var_max = 4,
    var_remap = 5,
    var_cur2 = 6,
    var_new2 = 7,
    var_dstep_delta = 8,
    var_bkt_cur = 9,
    var_slot = 10,
    var_small = 11
};

struct bucket_fn {
    using ID = VID;
    
    bucket_fn( SFloatTy * dist, FloatTy delta )
	: m_dist( dist ), m_delta( delta ) { }

    VID operator() ( VID v ) const {
	return (VID)( ((FloatTy)m_dist[v]) / m_delta );
    }

    template<VID Scale>
    FloatTy get_scaled( VID v ) const {
	return (VID)( ((FloatTy)m_dist[v]) / ( m_delta / (FloatTy)Scale ) );
    }

    FloatTy get( VID v ) const {
	return m_dist[v];
    }
    
private:
    SFloatTy * m_dist;
    FloatTy m_delta;
};

struct bucket_small_fn {
    using ID = VID;
    
    bucket_small_fn( SFloatTy * dist, SFloatTy * small, FloatTy delta )
	: m_dist( dist ), m_small( small ), m_delta( delta ) { }

    VID operator() ( VID v ) const {
	FloatTy d = m_dist[v];
	FloatTy s = m_small[v];
	return (VID)( ( d + s ) / m_delta );
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
    SFloatTy * m_small;
    FloatTy m_delta;
};

struct bucket_uniq_fn {
    using ID = VID;
    
    bucket_uniq_fn( FloatTy * dist, FloatTy delta, SlotID * slot )
	: m_dist( dist ), m_delta( delta ), m_slot( slot ) { }

    VID operator() ( VID v ) const {
	return (VID)( m_dist[v] / m_delta );
    }

    template<VID Scale>
    FloatTy get_scaled( VID v ) const {
	return (VID)( m_dist[v] / ( m_delta / (FloatTy)Scale ) );
    }

    bool set_slot( VID v, SlotID b ) {
	// Could also use atomic_min...
	if( m_slot[v] <= b )
	    return false;
	else {
	    m_slot[v] = b;
	    return true;
	}
    }
    
private:
    FloatTy * m_dist;
    FloatTy m_delta;
    SlotID * m_slot;
};

template <class GraphType>
class DSSSP {
public:
    DSSSP( GraphType & _GA, commandLine & P )
	: GA( _GA ), info_buf( 60 ),
#if !LEVEL_ASYNC || DEFERRED_UPDATE
	  cur_dist( GA.get_partitioner(), "current distance" ),
#endif
	  new_dist( GA.get_partitioner(), "new distance" ),
	  // slot( GA.get_partitioner(), "bucket slot" ),
	  edge_weight( GA.get_partitioner(), "edge weight" ) {
	itimes = P.getOption( "-itimes" );
	debug = P.getOption( "-debug" );
	calculate_active = P.getOption( "-cactive" );
	est_diam = P.getOptionLongValue( "-sssp:diam", 20 );
	start = GA.remapID( P.getOptionLongValue( "-start", 0 ) );
	delta = P.getOptionDoubleValue( "-sssp:delta", 1.0 );
	const char * dump_ligra_file = P.getOptionValue( "-dump:ligra" );

#if 0
	{
	    const partitioner &part = GA.get_partitioner();
	    map_partitionL( part, [&]( unsigned p ) {
		EID s = part.edge_start_of( p );
		auto csc = GA.getCSC( p );
		float * w = csc.getWeights();
		std::copy( &w[0], &w[csc.numEdges()],
			   &edge_weight.get_ptr()[s] );
	    } );
	}

	// Initialise (random) edge weights
	VID n = GA.numVertices();
	auto remap = GA.get_remapper();
#if 0
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
#if 0
	    api::edgemap(
		GA,
		api::relax( [&]( auto s, auto d, auto e ) {
		    auto ss = a_remap[s];
		    auto dd = a_remap[d];
		    auto h = expr::iif( ss > dd, ss, dd );
		    auto l = expr::iif( ss > dd, dd, ss );
		    auto cR = expr::constant_val2<VID>( d, 127 );
		    auto cO = expr::constant_val2<FloatTy>( d, 64 );
		    auto cW = expr::constant_val2<FloatTy>( d, 64 * 16 );
		    auto val = ( ((h<<_1)<<_1) + l ) & cR;
		    // auto val = ( ss + dd ) & cR;
		    return expr::make_seq(
			edge_weight[e] =
			expr::constant_val2<FloatTy>( d, 0.125 )
			+ ( expr::cast<FloatTy>( val ) - cO ) / cW,
			expr::true_val( d ) );
/*
		    return edge_weight[e] = expr::iif(
			( ( ss + dd ) & _1 ) != _0,
			expr::constant_val2<FloatTy>( d, 0.125 ),
			expr::constant_val2<FloatTy>( d, 0.25 ) );
*/
		} ) )
#else
		api::edgemap(
		    GA,
		    api::relax( [&]( auto s, auto d, auto e ) {
			using traits = typename decltype(e)::data_type;
			constexpr unsigned Bits = traits::B;
			auto magic
			    = expr::constant_val2( e, (EID)0xcdef893021b83a );
			auto shm = expr::constant_val2( e, (EID)(Bits-17) );
			auto sh = expr::constant_val2( e, (EID)(17) );
			auto mask = expr::allones_val( e ) >> shm;
			auto scale = expr::constant_val2( edge_weight[e],
							  (FloatTy)(1<<14) );
			auto slice0 = e ^ magic;
			auto slice1 = slice0 >> sh;
			auto slice2 = slice1 >> sh;
			auto slices = slice0 ^ slice1 ^ slice2;
			auto val = ( slices & mask ) + _1;
			return edge_weight[e]
			    = expr::cast<FloatTy>( val ) / scale;
		    } ) )
#endif
		.materialize();
	}
#endif

#if 0
	if( dump_ligra_file ) {
	    std::ofstream os( dump_ligra_file );
	    os << std::setprecision(10);
	    const GraphCSx & G = GA.getCSR();
	    const partitioner &part = GA.get_partitioner();

	    VID n = part.get_num_vertices();

	    os << "WeightedAdjacencyGraph\n"
	       << n << "\n" << G.numEdges() << "\n";

	    EID idx = 0;
	    for( VID v=0; v < n; ++v ) {
		VID vv = remap.remapID( v );
		VID d = G.getDegree( vv );
		os << idx << "\n";
		idx += d;
	    }

	    auto * edges = new std::pair<VID,FloatTy>[G.numEdges()];
	    auto eid_retriever = GA.get_eid_retriever();
	    EID off = 0;
	    for( VID v=0; v < n; ++v ) {
		VID vv = remap.remapID( v );
		VID d = G.getDegree( vv );
		for( VID j=0; j < d; ++j ) {
		    VID uu = G.getNeighbor( vv, j );
		    VID ngh = remap.origID( uu );
		    EID eid = eid_retriever.get_edge_eid( vv, j );
		    edges[off+j].first = ngh;
		    edges[off+j].second = edge_weight[eid];

/*
		    VID cR = 127;
		    FloatTy cO = 64;
		    FloatTy cW = 64 * 16;
		    VID val = ( v + ngh ) & cR;
		    FloatTy w = 0.125 + ( ((FloatTy)val) - cO ) / cW;
		    assert( w == edge_weight[eid] );
*/
		}
		std::sort( &edges[off], &edges[off+d] );
		for( VID j=0; j < d; ++j )
		    os << edges[off+j].first << "\n";
		off += d;
	    }

	    off = 0;
	    for( VID v=0; v < n; ++v ) {
		VID vv = remap.remapID( v );
		VID d = G.getDegree( vv );
		for( VID j=0; j < d; ++j ) {
		    os << edges[off+j].second << "\n";
		}
		off += d;
	    }

	    delete[] edges;
	    os.close();
	}
#endif
#endif

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
	    std::cerr << "delta=" << delta << "\n";
	}
    }
    ~DSSSP() {
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

	expr::array_ro<FloatTy,VID,var_dstep_delta> a_delta( &delta );

#if !LEVEL_ASYNC || DEFERRED_UPDATE
	assert( cur_dist.get_ptr() );
#endif

#if VARIANT == 3
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

	// Determine configuration of custom floating-point format, assuming
	// maximum path length is est_diam hubs at most.
	// The shortest path length is 0 (from vertex to itself), however,
	// this is an exceptional case and we may detect and handle it, or
	// assume that the smallest path length can be 0.
	// Here, we require that 0 and infinity are correctly represented.
#if VARIANT == 1 || VARIANT == 3
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

	cfp_cfg::set_param_for_range( std::min( (FloatTy)0, f_min ),
				      f_max, false );
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
	    } )
/*
	    .vertex_map( [&]( auto v ) {
		auto ones
		    = expr::value<simd::ty<USlotID,decltype(v)::VL>,
				  expr::vk_truemask>();
		return slot[v] = ones >> _1;
	    } )
*/
	    .materialize();
#if !LEVEL_ASYNC || DEFERRED_UPDATE
	cur_dist.get_ptr()[start] = 0.0f;
#endif
	new_dist.get_ptr()[start] = 0.0f;

	// Calculate lowest weight among out-going edges
#if 0
	api::vertexprop<FloatTy,VID,var_small,EncNew>
	    small_out( part, "smallest out-edge" );
	make_lazy_executor( part )
	    .vertex_map( [&]( auto v ) {
		auto inf = expr::constant_val2<SFloatTy>(
		    v, std::numeric_limits<SFloatTy>::infinity() );
		return small_out[v] = inf;
	    } )
	    .materialize();
	api::edgemap(
	    GA, 
	    api::relax( [&]( auto s, auto d, auto e ) {
		return small_out[d].min(
		    expr::cast<FloatTy>( edge_weight[e] ) );
	    } ) )
	    .vertex_map( [&]( auto v ) {
		auto inf = expr::constant_val2<SFloatTy>(
		    v, std::numeric_limits<SFloatTy>::infinity() );
		return small_out[v] = expr::iif( small_out[v] == inf,
						 _0, small_out[v] );
	    } )
	    .materialize();
#endif


	// Create bucket structure
/*
	buckets<VID,bucket_uniq_fn>
	    bkts( n, 127,
		  bucket_uniq_fn( new_dist.get_ptr(), delta, slot.get_ptr() ) );
*/
	buckets<VID,bucket_fn>
	    bkts( n, 127, bucket_fn( new_dist.get_ptr(), delta ) );
/*
	buckets<VID,bucket_small_fn>
	    bkts( n, 127, bucket_small_fn( new_dist.get_ptr(),
					   small_out.get_ptr(), delta ) );
*/

	// Place start vertex in first bucket
	bkts.insert( start, 0 );

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

	while( !bkts.empty() ) {  // iterate until all vertices visited
	    // timer tm;
	    // tm.start();

	    frontier F = bkts.next_bucket();
	    F.calculateActiveCounts( GA.getCSR(), part, F.nActiveVertices() );

	    // Record that elements removed from bucketing structure by
	    // clearing slot
/*
	    make_lazy_executor( part )
		.vertex_map( F, [&]( auto v ) {
		    auto ones
			= expr::value<simd::ty<USlotID,decltype(v)::VL>,
				      expr::vk_truemask>();
		    return slot[v] = ones >> _1;
		} )
		.materialize();
*/

	    // std::cerr << "get bucket: " << tm.next() << "\n";

#if VARIANT == 3
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

	    // Traverse edges, remove duplicate live destinations.
	    frontier output;
#if UNCOND_EXEC
	    auto filter_strength = api::weak;
#else
	    auto filter_strength = api::strong;
#endif

	    auto bkt_fn = [&]( auto d ) {
		auto z = 
		    expr::value<simd::ty<VID,decltype(d)::VL>,expr::vk_zero>();
		return expr::cast<VID>( d / a_delta[z] );
	    };

	    VID f_cur = bkts.get_current_bucket();
	    expr::array_ro<VID,VID,var_bkt_cur> a_cur( &f_cur );

	    // TODO: config to not calculate nactv, nacte?
	    api::edgemap(
		GA, 
		// api::config( api::always_dense ),
		// api::config( api::always_sparse ),
#if DEFERRED_UPDATE
		// TODO: Do we apply the deferred update function in sparse
		//       edgemap?
		api::record( output,
			     [&] ( auto d ) {
				 auto z =
				     expr::value<simd::ty<VID,decltype(d)::VL>,expr::vk_zero>();
/* TODO: bkts differ, or bkt remains BUT distances differ
				 auto cur_bkt = bkt_fn( cur_dist[d] );
				 auto new_bkt = bkt_fn( new_dist[d] );
				 return cur_bkt != new_bkt || new_bkt < a_cur[z]; },
*/
				 return cur_dist[d] != new_dist[d]; },
			     api::strong ),
#else
		// api::record( output, api::reduction, filter_strength ),
		// Do not allow unbacked frontiers
		api::record( output, api::reduction, strong ),
#endif
		api::filter( filter_strength, api::src, F ),
		api::relax( [&]( auto s, auto d, auto e ) {
#if LEVEL_ASYNC
		    return new_dist[d].min( new_dist[s] + edge_weight[e] );
/*
---> calculate bucket, if changed, make active
need to keep b[] up to date when taking out a frontier (next_frontier -> vertex_map w/ frontier to set to inf)
		    return expr::make_seq(
			new_dist[d].min( new_dist[s] + edge_weight[e] ),
			b[d].min( bkt_fn( new_dist[d] ) ) );
*/
#else
		    return new_dist[d].min( cur_dist[s] + edge_weight[e] );
#endif
		} )
		).materialize();

/*
	    std::cerr << "P:";
	    for( VID v=0; v < n; ++v ) {
		std::cerr << ' ' << new_dist[v];
	    }
	    std::cerr << '\n';
*/

#if !LEVEL_ASYNC || DEFERRED_UPDATE
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

	    bkts.update_buckets( part, output );

	    // std::cerr << "unbacked: " << tm.next() << "\n";

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

	    // std::cerr << "copy: " << tm.next() << "\n";

	    // std::cerr << "output: " << output.nActiveVertices() << "\n";
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

	// small_out.del();
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
    FloatTy delta;
#if !LEVEL_ASYNC || DEFERRED_UPDATE
    api::vertexprop<FloatTy,VID,var_cur,EncCur> cur_dist;
#endif
    api::vertexprop<FloatTy,VID,var_new,EncNew> new_dist;
    // api::vertexprop<VID,VID,var_slot,array_encoding<SlotID>> slot;
    api::edgeprop<FloatTy,EID,expr::vk_eweight,EncEdge> edge_weight;
    // api::edgeprop<FloatTy,EID,var_weight,EncEdge> edge_weight;
    std::vector<info> info_buf;
};

template <class GraphType>
using Benchmark = DSSSP<GraphType>;

#include "driver.C"
