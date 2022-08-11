#include "graptor/graptor.h"
#include "graptor/api.h"
#include "unique.h"
#include "check.h"

using expr::_0;
using expr::_1;
using expr::_1s;
using expr::_true;
using expr::_c;

enum gc_variable_name {
    var_color = 0,
    var_depth = 1,
    var_posscol = 2,
    var_funion = 3,
    var_priority = 4,
    var_predecessors = 5,
    var_tie = 6,
    var_degrees_ro = expr::aid_graph_degree
};

using BitMaskTy = uint64_t;

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
	  depth( GA.get_partitioner(), "depth" ),
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
	    // std::cerr << "FUSION=" << FUSION << "\n";
	}
    }
    ~GC_ECL() {
	color.del();
	posscol.del();
	funion.del();
	priority.del();
	predecessors.del();
	tie_breaker.del();
	depth.del();
    }

    struct info {
	double delay;
	float density;
	VID nactv;
	EID nacte;

	void dump( int i ) {
	    std::cerr << "Iteration " << i << ": " << delay
		      << " density: " << density
		      << " nactv: " << nactv
		      << " nacte: " << nacte
		      << "\n";
	}
    };

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
	    .vertex_map( [&]( auto v ) {
		constexpr size_t W = sizeof(VID)*8;
		auto msb = expr::slli<W-1>( _1s(color[v]) );
		return expr::make_seq(
		    priority[v] = _0,
		    // color[v] = v );
		    // color[v] = _0 );
		    // color[v] = _1s );
		    color[v] = msb | v ); // all unique - for debugging
	    } )
	    .materialize();

/*
	expr::rnd::random_shuffle(
	    color.get_ptr(),
	    tie_breaker.get_ptr(),
	    n,
	    expr::rnd::simple_rng(1) );

	make_lazy_executor( part )
	    .vertex_map( [&]( auto v ) { return color[v] = _0; } )
	    .materialize();
*/

	// constexpr VID threshold = 2;
	constexpr VID delta = 1;
	VID cur_depth = delta;

	// frontier roots;
	frontier badv;
	api::edgemap(
	    GA,
	    api::relax( [&]( auto s, auto d, auto e ) {
		auto fn = [&]( auto v ) {
		    // return v & expr::slli<8>( expr::srli<27>( _1s(v) ) );
		    // return expr::srli<8>( v );
		    return v;
		};
		return priority[d] +=
		    expr::add_predicate(
			_1(priority[d]),
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
			);
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

#if 1
	{
	    frontier todo;
	    make_lazy_executor( part )
		.vertex_map( [&]( auto v ) {
		    return depth[v] = _0;
		} )
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

		if( itimes ) {
		    info_buf.resize( iter+1 );
		    info_buf[iter].density = todo.density( m );
		    info_buf[iter].delay = tm_iter.next();
		    info_buf[iter].nactv = todo.nActiveVertices();
		    info_buf[iter].nacte = todo.nActiveEdges();
		    if( debug )
			info_buf[iter].dump( iter );
		    ++iter;
		}

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

	frontier roots;
	make_lazy_executor( part )
	    .vertex_filter(
		GA, ftrue, roots,
		[&]( auto v ) {
		    return depth[v] < _c( cur_depth );
		} )
	    .materialize();

	if( itimes ) {
	    info_buf.resize( iter+1 );
	    info_buf[iter].density = -1.0;
	    info_buf[iter].delay = tm_iter.next();
	    info_buf[iter].nactv = n;
	    info_buf[iter].nacte = m;
	    if( debug )
		info_buf[iter].dump( iter );
	    ++iter;
	}

	while( !roots.isEmpty() ) {

	    std::cerr << "cur_depth: " << cur_depth << "\n";

	    /* Idea:
	     * If a neighbour has a color assigned (permanently) that is larger
	     * then our degree, then that neighbour no longer has priority
	     * over us (dependency removed).
	     */
	    
	    /* Idea:
	     * - process vertices with prio <= 1 at the same time
	     * - if conflict -> resolve in favour of prio == 0
	     * - use short-cutting between vertices at prio 0 and prio 1
	     *   i.e.:
	     *   1. calculate forbidden color mask
	     *   2. calculate intersection of all forbidden
	     *   3. if best color in forbidden is not blocked by intersection
	     *      then color is permanent and priority of dependents can be
	     *      be dropped to zero
	     *      so frontier out would be: is color permanent
	     *      / non-conflicting?
	     * Alt:
	     * - forbidden-prio0 ; forbidden-prio>0 (tentative)
	         possibly including all vertices in same vector as those prio 0
	     * - ;
	     */

/*
	    make_lazy_executor( part )
		..vertex_map( roots, [&]( auto v ) { return forbidden[v] = _0; } )
		.vertex_map( roots, [&]( auto v ) { return funion[v] = _0; } )
		.vertex_map( roots, [&]( auto v ) { return color[v] = _1; } )
	    .materialize();
*/

	    // TODO: specialise first iteration: priority == _0 -> color = _0

	    make_lazy_executor( part )
		.vertex_map( /*roots,*/ [&]( auto v ) { return funion[v] = _0; } )
		.materialize();

	// Iterate one partition at a time
	    frontier vdone; // roots, or non-roots that are correct
	    api::edgemap(
		GA,
		api::config( api::always_dense ),
		api::filter( api::dst, api::strong, roots ),
		api::relax( [&]( auto s, auto d, auto e ) {
		    // It is expected that posscol[v] != _0; otherwise no
		    // available colours.
		    constexpr size_t W = sizeof(VID)*8;
		    // auto w = _c( sizeof(BitMaskTy)*8 - 1 );
		    // auto msb = expr::slli<sizeof(BitMaskTy)*8-1>( _1s(posscol[d]) );
		    return expr::make_seq(
			// funion[d] |= expr::add_predicate(
			// posscol[s], priority[s] == _0 ), // < _c(threshold) ),
			funion[d] |= expr::add_predicate(
			    posscol[s], 
			    depth[s] < depth[d] // s is predecessor of d -- can use formula used for priority (LF/LLF)?
			    // expr::srli<W-1>( color[s] ) == _0
			    ),
			predecessors[d] += expr::add_predicate(
			    _1s(predecessors[d]),
			    // expr::srli<W-1>( color[s] ) == _0
			    // is-a predecessor and is resolved
			    ( ( posscol[d] & posscol[s] ) == _0
			      || ( posscol[s] & ( posscol[s] - _1 ) ) == _0 )
			    && depth[s] < depth[d] ),
			posscol[d] = expr::add_predicate(
			    expr::iif(
				( posscol[d] & posscol[s] ) == _0,
				expr::iif( ( posscol[s] & ( posscol[s] - _1 ) ) == _0,
					   posscol[d],
					   posscol[d] ^ posscol[s] ),
				posscol[d] & ( posscol[d] - _1 )
				),
			    // priority[s] == _0 // < _c(threshold)
			    depth[s] < depth[d]
			    // expr::srli<W-1>( color[s] ) == _0
			    ) );
		} )
/*
		api::record( vdone, [&]( auto d ) {
		    return priority[d] == _0
			|| ( priority[d] < _c(threshold)
			     && ( ( funion[d] >> expr::cast<BitMaskTy>( color[d] ) )
				  & _1 ) == _0 );
		}, api::strong )
*/
		)
		.materialize();

	    make_lazy_executor( part )
		.vertex_filter(
		    GA,
		    roots,
		    vdone,
		    [&]( auto d ) {
			auto w = _c( sizeof(BitMaskTy)*8 - 1 );
			auto msb = expr::slli<sizeof(BitMaskTy)*8-1>( _1s(posscol[d]) );
			auto best = expr::lzcnt<BitMaskTy>( posscol[d] );
			auto done = ( ( msb >> best ) & ~funion[d] ) != _0
			    || predecessors[d] == _0;
			// Returns true if predicate is true, false otherwise
			return color[d] = expr::add_predicate( best, done );
		    } )
		.materialize();

	    std::cerr << "vdone: " << vdone.nActiveVertices()
		      << " roots: " << roots.nActiveVertices() << "\n";

	    if( debug ) {
		frontier x = list_conflicts();

		roots.toSparse( part );
		vdone.toSparse( part );
		x.toSparse( part );

		std::cerr << "roots: " << roots << "\n";
		std::cerr << "vdone: " << vdone << "\n";
		std::cerr << "cnflt: " << x << "\n";
		print( std::cerr, part, color );
		print( std::cerr, part, priority );
		print( std::cerr, part, predecessors );
		print( std::cerr, part, depth );
		std::cerr << std::hex;
		print( std::cerr, part, posscol );
		print( std::cerr, part, funion );
		std::cerr << std::dec;
		x.del();
	    }
	    
	    frontier new_roots;
#if 0
	    api::edgemap(
		GA,
		// api::config( api::always_sparse ),
		api::relax( [&]( auto s, auto d, auto e ) {
		    constexpr size_t W = sizeof(VID)*8;
/*
		    return priority[d].count_down_value( _0(d) ) <= _c(threshold)
			// && expr::cast<int>( color[d] ) < _0;
			&& expr::srli<W-1>( color[d] ) != _0;
		    // OR: source, if source has not a final colour yet
		    */
		    return expr::make_seq(
			priority[d].count_down( _0(d) ),
			depth[d] < _c( cur_depth + delta )
			&& expr::srli<W-1>( color[d] ) != _0 );
		} ),
		api::filter( api::src, api::strong, vdone ), // weak?
		// api::filter( api::src, api::strong, roots ),
		api::record( new_roots, api::reduction, api::strong )
		)
		.vertex_map( roots,
			     [&]( auto v ) {
				 constexpr size_t W = sizeof(VID)*8;
				 return priority[v]
				     += expr::add_predicate(
					 _1s(priority[v]),
					 expr::srli<W-1>( color[v] ) == _0 ); // predicate should always be true for roots (but applying to broader active set)
			     } )
		.materialize();

	    if( debug ) {
		new_roots.toSparse( part );
		std::cerr << "new_roots: " << new_roots << "\n";
	    }

#if 1
	    // new_roots.del();

	    // To include also vertices previously prio < threshold, but not
	    // come to conclusion
	    frontier revive;
	    make_lazy_executor( part )
		.vertex_filter(
		    GA,
		    roots,
		    revive,
		    [&]( auto d ) {
			constexpr size_t W = sizeof(VID)*8;
			// return priority[d] < _c(threshold)
			// && expr::srli<W-1>( color[d] ) != _0;
			return depth[d] < _c( cur_depth + delta )
			    && expr::srli<W-1>( color[d] ) != _0;
		    } )
		.materialize();

	    std::cerr << "revive   : " << revive << "\n";
	    new_roots.merge_or( GA, revive );
	    std::cerr << "merged   : " << new_roots << "\n";
	    revive.del();
#endif
#else
	    // what are the new roots/active set?
	    make_lazy_executor( part )
		.vertex_filter(
		    GA,
		    ftrue, // roots,
		    new_roots,
		    [&]( auto d ) {
			constexpr size_t W = sizeof(VID)*8;
			return ( depth[d] < _c( cur_depth + delta )
				 || predecessors[d] == _0 )
			    && expr::srli<W-1>( color[d] ) != _0;
		    } )
		.materialize();

	    std::cerr << "new_roots   : " << new_roots << "\n";
#endif

	    if( itimes ) {
		info_buf.resize( iter+1 );
		info_buf[iter].density = roots.density( m );
		info_buf[iter].delay = tm_iter.next();
		info_buf[iter].nactv = roots.nActiveVertices();
		info_buf[iter].nacte = roots.nActiveEdges();
		if( debug )
		    info_buf[iter].dump( iter );
		++iter;
	    }

	    vdone.del();
	    roots.del();
	    roots = new_roots;

	    cur_depth += delta;
	}

	roots.del();
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
    api::vertexprop<VID,VID,var_depth> depth;
    expr::rnd::prng<VID> prng;
    std::vector<info> info_buf;
};

template <class GraphType>
using Benchmark = GC_ECL<GraphType>;

#include "driver.C"
