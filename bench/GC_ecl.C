#include "graptor/graptor.h"
#include "graptor/api.h"
#include "unique.h"
#include "check.h"

using expr::_0;
using expr::_1;
using expr::_1s;
using expr::_true;
using expr::_c;
using expr::_p;

#define WITH_DEPTH 0

#define PEELING 1

enum gc_variable_name {
    var_color = 0,
    var_depth = 1,
    var_posscol = 2,
    var_funion = 3,
    var_priority = 4,
    var_predecessors = 5,
    var_tie = 6,
    var_dag = 7,
    var_dep = 8,
    var_no_overlap = 9,
    var_src_fixed = 10,
    var_posscol_s = 11,
    var_degrees_ro = expr::aid_graph_degree
};

enum iteration_kind {
    ik_prio,
    ik_depth,
    ik_color,
    ik_post
};

std::ostream & operator << ( ostream & os, iteration_kind k ) {
    switch( k ) {
    case ik_prio: os << "priority"; break;
    case ik_depth: os << "DAG-depth"; break;
    case ik_color: os << "coloring"; break;
    case ik_post: os << "post-process"; break;
    default: os << "<unknown>"; break;
    }
    return os;
}

using BitMaskTy = uint64_t;
using DepTy = logical<4>;

template <class GraphType>
class GC_ECL {
    static constexpr float mu = 0.0025f;

public:
    GC_ECL( GraphType & _GA, commandLine & P )
	: GA( _GA ),
	  color( GA.get_partitioner(), "color" ),
	  posscol( GA.get_partitioner(), "possible colors" ),
	  funion( GA.get_partitioner(), "funion" ),
	  priority( GA.get_partitioner(), "priority" ),
	  predecessors( GA.get_partitioner(), "predecessors" ),
	  tie_breaker( GA.get_partitioner(), "tie_breaker" ),
#if WITH_DEPTH
	  depth( GA.get_partitioner(), "depth" ),
#endif
	  dag( GA.get_partitioner(), "DAG" ),
	  prng( GA.get_partitioner(),
		static_cast<VID>( static_cast<float>( GA.getCSR().max_degree() )
				  * ( 1.0 + mu ) ) ),
	  info_buf( 60 ) {
	itimes = P.getOption( "-itimes" );
	debug = P.getOption( "-debug" );
	outfile = P.getOptionValue( "-gc:outfile" );

	static bool once = false;
	if( !once ) {
	    once = true;
	    std::cerr << "WITH_DEPTH=" << WITH_DEPTH << "\n";
	    std::cerr << "PEELING=" << PEELING << "\n";
	}
    }
    ~GC_ECL() {
	color.del();
	posscol.del();
	funion.del();
	priority.del();
	predecessors.del();
	tie_breaker.del();
#if WITH_DEPTH
	depth.del();
#endif
	dag.del();
    }

    struct info {
	double delay;
	float density;
	VID nactv;
	EID nacte;
	iteration_kind iknd;

	void dump( int i ) {
	    std::cerr << "Iteration " << i << ": " << delay
		      << " density: " << density
		      << " nactv: " << nactv
		      << " nacte: " << nacte
		      << " " << iknd
		      << "\n";
	}
    };

    void log( int & iter, timer & tm_iter, frontier & F, iteration_kind iknd ) {
	if( itimes ) {
	    info_buf.resize( iter+1 );
	    info_buf[iter].density = F.density( GA.numEdges() );
	    info_buf[iter].delay = tm_iter.next();
	    info_buf[iter].nactv = F.nActiveVertices();
	    info_buf[iter].nacte = F.nActiveEdges();
	    info_buf[iter].iknd = iknd;
	    if( debug )
		info_buf[iter].dump( iter );
	    ++iter;
	}
    }

    struct stat {
	double delay;
	int iter;
	EID edges;
	VID colours;
    };

    void run_ecl() {
	const partitioner &part = GA.get_partitioner();
	VID n = GA.numVertices();
	EID m = GA.numEdges();

	timer tm_iter;
	tm_iter.start();
	iter = 0;

	expr::array_ro<VID, VID, var_degrees_ro> degree(
	    const_cast<VID *>( GA.getCSR().getDegree() ) );

	frontier ftrue = frontier::all_true( n, m );

	make_lazy_executor( part )
	    .vertex_map( [&]( auto v ) { return priority[v] = _0; } )
	    .materialize();

/*
first set color[v] = v
	expr::rnd::random_shuffle(
	    color.get_ptr(),
	    tie_breaker.get_ptr(),
	    n,
	    expr::rnd::simple_rng(1) );
*/

	// constexpr VID threshold = 2;
	constexpr VID delta = 2;
	VID cur_depth = delta;

	// frontier roots;
	frontier badv;
	api::edgemap(
	    GA,
	    api::config( api::always_dense ), // because of edge numbering
	    api::relax( [&]( auto s, auto d, auto e ) {
		auto fn = [&]( auto v ) {
		    // return v & expr::slli<8>( expr::srli<27>( _1s(v) ) );
		    // return expr::srli<8>( v );
		    return v;
		};
		return expr::let<var_dep>(
#if LLF
		    expr::make_unop_lzcnt<VID>( degree[s] )
		    > expr::make_unop_lzcnt<VID>( degree[d] )
		    || ( expr::make_unop_lzcnt<VID>( degree[s] )
			 == expr::make_unop_lzcnt<VID>( degree[d] )
			 && s < d )
#else
		    degree[s] > degree[d]
		    || ( degree[s] == degree[d]
			 && fn(s) > fn(d) )
// -> fn(s) > fn(d) || ( fn(s) == fn(d) && s < d )
		    // || ( degree[s] == degree[d] && tie_breaker[s] < tie_breaker[d] )
#endif
		    , [&]( auto dep ) {
			return expr::make_seq(
			    dag[e] = dep,
			    priority[d] += _p( _1(priority[d]), dep )
			    );
		    } );
	    } )
/*
	    api::record( roots,
			 [&]( auto d ) {
			     return priority[d] < _c(threshold); // == _0;
			 }, api::strong )
*/
	    )
	    .vertex_map( [&]( auto v ) {
		constexpr size_t W = sizeof(BitMaskTy)*8;
		auto w = _c( W );
		auto msb = expr::slli<W-1>( _1s(posscol[v]) );
		return posscol[v]
		    = expr::iif( priority[v] < w, _0,
				 expr::sra( msb,
					    expr::cast<BitMaskTy>( priority[v] ) ) );
	    } )
	    .vertex_map( [&]( auto v ) { return funion[v] = _0; } )
	    .vertex_filter( GA, ftrue, badv,
			    [&]( auto v ) {
				return posscol[v] == _0 && degree[v] != _0;
			    } )
	    .materialize();

	make_lazy_executor( part )
	    .vertex_map( [&]( auto v ) {
		return predecessors[v] = priority[v];
	    } )
	    .materialize();

	assert( badv.nActiveVertices() == 0
		&& "cannot encode posscol in available bits" );
	badv.del();

	log( iter, tm_iter, ftrue, ik_prio );

#if WITH_DEPTH
	{
	    frontier todo;
	    make_lazy_executor( part )
		.vertex_map( [&]( auto v ) { return depth[v] = _0; } )
		.vertex_filter(
		    GA, ftrue, todo,
		    [&]( auto v ) {
			return priority[v] == _0;
		    } )
		.materialize();
	    while( !todo.isEmpty() ) {
		frontier new_todo;
		api::edgemap(
		    GA,
		    api::relax( [&]( auto s, auto d, auto e ) {
			return expr::make_seq(
			    depth[d].max( depth[s] + _1, predecessors[d] > _0 ),
			    predecessors[d].count_down( _0(d) ) );
		    } ),
		    api::filter( api::src, api::strong, todo ),
		    api::record( new_todo, api::reduction, api::strong )
		    ).materialize();

		log( iter, tm_iter, todo, ik_depth );

		todo.del();
		todo = new_todo;
	    }
	    todo.del();
	    if( debug ) {
		print( std::cerr, part, depth );
	    }

	    make_lazy_executor( part )
		.vertex_map( [&]( auto v ) {
		    return predecessors[v] = priority[v];
		} )
		.materialize();

	}
#endif

#if 0
	frontier roots;
	make_lazy_executor( part )
	    .vertex_filter(
		GA, ftrue, roots,
		[&]( auto v ) {
		    return depth[v] < _c( cur_depth );
		} )
	    .materialize();

	log( iter, tm_iter, ftrue, ik_depth );
#else
	frontier roots = ftrue;
#endif

	while( !roots.isEmpty() ) {

	    std::cerr << "cur_depth: " << cur_depth << "\n";

	    make_lazy_executor( part )
		.vertex_map( roots, [&]( auto v ) { return funion[v] = _0; } )
		.materialize();

	    api::edgemap(
		GA,
		api::config( api::always_dense ), // because of edge numbering
		api::filter( api::dst, api::strong, roots ),
		api::relax( [&]( auto s, auto d, auto e ) {
		    // It is expected that posscol[v] != _0; otherwise no
		    // available colours.
		    using MTr = typename decltype(posscol[d])::data_type
			::prefmask_traits;
		    return expr::let<var_dep>( dag[e], [&]( auto dep ) {
		    return expr::let<var_posscol_s>( posscol[s],
		    [&]( auto pcs ) {
		    return expr::let<var_no_overlap>( ( posscol[d] & pcs ) == _0,
		    [&]( auto overlap_v ) {
		    return expr::let<var_src_fixed>( ( pcs & ( pcs - _1 ) ) == _0,
		    [&]( auto src_fixed_v ) {
		    auto no_overlap
			= expr::make_unop_cvt_to_mask<MTr>( overlap_v );
		    // = ( posscol[d] & pcs ) == _0;
		    auto src_fixed
			= expr::make_unop_cvt_to_mask<MTr>( src_fixed_v );
		    // = ( pcs & ( pcs - _1 ) ) == _0;
		    return expr::make_seq(
			funion[d] |= _p( pcs, dep ),
#if 0
			predecessors[d] += _p(
			    _1s(predecessors[d]),
			    // is-a predecessor and is resolved
			    ( no_overlap || src_fixed ) && dep ),
#endif
			posscol[d] = _p(
			    expr::iif(
				no_overlap,
				expr::iif( src_fixed,
					   posscol[d],
					   posscol[d] ^ pcs ),
				posscol[d] & ( posscol[d] - _1 )
				),
			    dep
			    ),
			dag[e] &= expr::cast<DepTy>(
			    !( no_overlap || src_fixed ) )
			);
		    } );
		    } );
		    } );
		    } );
		} )
		)
		.materialize();

	    make_lazy_executor( part )
		.vertex_map( roots, [&]( auto d ) {
		    constexpr size_t W = sizeof(BitMaskTy)*8 - 1;
		    auto msb = expr::slli<W>( _1s(posscol[d]) );
		    auto best = expr::lzcnt<BitMaskTy>( posscol[d] );
		    auto done = ( ( msb >> best ) & ~funion[d] ) != _0;
		    // Returns true if predicate is true, false otherwise
		    return posscol[d] = _p( msb >> best, done );
		} )
		.materialize();

	    if( debug ) {
		roots.toSparse( part );

		std::cerr << "roots: " << roots << "\n";
		print( std::cerr, part, priority );
		print( std::cerr, part, predecessors );
#if WITH_DEPTH
		print( std::cerr, part, depth );
#endif
		std::cerr << std::hex;
		print( std::cerr, part, posscol );
		print( std::cerr, part, funion );
		std::cerr << std::dec;
	    }
	    
	    // what are the new roots/active set?
	    frontier new_roots;
#if 0
	    make_lazy_executor( part )
		.vertex_filter(
		    GA,
		    ftrue, // roots,
		    new_roots,
		    [&]( auto d ) {
			constexpr size_t W = sizeof(VID)*8;
			return ( depth[d] < _c( cur_depth + delta )
			    )
				 // || predecessors[d] == _0 ) // -> done => P==0
			    // && expr::srli<W-1>( color[d] ) != _0; // -> !done
			    && ( posscol[d] & ( posscol[d] - _1 ) ) != _0; // !done
		    } )
		.materialize();
#else
	    make_lazy_executor( part )
		.vertex_filter(
		    GA,
		    roots,
		    new_roots,
		    [&]( auto d ) {
			return ( posscol[d] & ( posscol[d] - _1 ) ) != _0;
		    } )
		.materialize();
#endif

	    if( debug ) {
		new_roots.toSparse( part );
		std::cerr << "new_roots   : " << new_roots << "\n";
	    }

	    log( iter, tm_iter, roots, ik_color );

	    roots.del();
	    roots = new_roots;

	    cur_depth += delta;
	}

	roots.del();

	make_lazy_executor( part )
	    .vertex_map(
		[&]( auto d ) {
		    auto w = _c( sizeof(BitMaskTy)*8 - 1 );
		    auto msb = expr::slli<sizeof(BitMaskTy)*8-1>( _1s(posscol[d]) );
		    auto best = expr::lzcnt<BitMaskTy>( posscol[d] );
		    return color[d] = best;
		} )
	    .materialize();

	log( iter, tm_iter, ftrue, ik_post );

	ftrue.del();
    }
    
    void run() {
	run_ecl();
    }

    frontier list_conflicts() {
	frontier c;
	api::edgemap(
	    GA,
	    api::record( c, api::reduction, api::strong ),
	    api::relax( [&]( auto s, auto d, auto e ) {
		return color[s] == color[d] && color[d] > _0;
	    } )
	    ).materialize();
	return c;
    }

    void post_process( stat & stat_buf ) {
	if( itimes ) {
	    for( int i=0; i < iter; ++i )
		info_buf[i].dump( i );
	}

	if( outfile )
	    writefile( GA, outfile, color.get_ptr() );
    }

    void validate( stat & stat_buf ) {
	const partitioner &part = GA.get_partitioner();

	frontier output;
	api::edgemap(
	    GA,
	    api::record( output, api::reduction, api::strong ),
	    api::relax( [&]( auto s, auto d, auto e ) {
		return color[s] == color[d];
	    } )
	    ).materialize();

	if( output.isEmpty() ) {
	    std::cerr << "Validation successfull\n";
	} else {
	    std::cerr << "Validation failed on " << output.nActiveVertices()
		      << " vertices (FAIL)\n";
	    output.toSparse( part );
	    if( output.getType() == frontier_type::ft_sparse ) {
		std::cerr << "Conflicts:";
		VID * cc = output.getSparse();
		VID k = std::min( (VID)100, output.nActiveVertices() );
		for( VID i=0; i < k; ++i )
		    std::cerr << ' ' << cc[i] << '#' << color[cc[i]];
		if( k < output.nActiveVertices() )
		    std::cerr << " ...";
		std::cerr << "\n";
	    }
	    
	    abort();
	}

	output.del();

	VID ncol = count_unique<VID>( GA, color.get_ptr(), std::cerr );

	if( itimes ) {
	    double total = 0.0;
	    EID edges = 0;
	    for( int i=0; i < info_buf.size(); ++i ) {
		info_buf[i].dump( i );
		total += info_buf[i].delay;
		edges += info_buf[i].nacte;
	    }

	    stat_buf.delay = total;
	    stat_buf.iter = iter;
	    stat_buf.edges = edges;
	    stat_buf.colours = ncol;
	}
    }

    static void report( const std::vector<stat> & stat_buf ) {
	size_t repeat = stat_buf.size();
	for( size_t i=0; i < repeat; ++i )
	    std::cerr << "round " << i << ": delay: " << stat_buf[i].delay
		      << " iterations: " << stat_buf[i].iter
		      << " total-edges: " << stat_buf[i].edges
		      << " colours: " << stat_buf[i].colours
		      << '\n';
    }

private:
    const GraphType & GA;
    bool itimes, debug;
    const char * outfile;
    int iter;
    api::vertexprop<VID,VID,var_color> color;
    api::vertexprop<BitMaskTy,VID,var_posscol> posscol;
    api::vertexprop<BitMaskTy,VID,var_funion> funion;
    api::vertexprop<VID,VID,var_priority> priority;
    api::vertexprop<VID,VID,var_predecessors> predecessors;
    api::vertexprop<VID,VID,var_tie> tie_breaker;
#if WITH_DEPTH
    api::vertexprop<VID,VID,var_depth> depth;
#endif
    // api::edgeprop<bitfield<1>,EID,var_dag,array_encoding_bit<1>> dag;
    api::edgeprop<DepTy,EID,var_dag> dag;
    expr::rnd::prng<VID> prng;
    std::vector<info> info_buf;
};

template <class GraphType>
using Benchmark = GC_ECL<GraphType>;

#include "driver.C"
