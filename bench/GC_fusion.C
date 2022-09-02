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

#ifndef FUSION
#define FUSION 1
#endif

enum gc_variable_name {
    var_color = 0,
    var_posscol = 1,
    var_priority = 2,
    var_dep = 3,
    var_new = 4,
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
class GC_JP_Fusion {
    static constexpr float mu = 0.0025f;

public:
    GC_JP_Fusion( GraphType & _GA, commandLine & P )
	: GA( _GA ),
	  color( GA.get_partitioner(), "color" ),
	  posscol( GA.get_partitioner(), "possible colors" ),
	  priority( GA.get_partitioner(), "priority" ),
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
	    std::cerr << "FUSION=" << FUSION << "\n";
	}
    }
    ~GC_JP_Fusion() {
	color.del();
	posscol.del();
	priority.del();
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

    void run() {
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

	// Evaluates to true if s has priority over d
	auto prio_fn = [&]( auto s, auto d ) {
#if LLF
	    return expr::make_unop_lzcnt<VID>( degree[s] )
		> expr::make_unop_lzcnt<VID>( degree[d] )
		|| ( expr::make_unop_lzcnt<VID>( degree[s] )
		     == expr::make_unop_lzcnt<VID>( degree[d] )
		     && s < d );
#else
	    return degree[s] > degree[d]
		|| ( degree[s] == degree[d] && s < d );
#endif
	};

	frontier roots, badv;
	api::edgemap(
	    GA,
	    api::config( api::always_dense ), // because of edge numbering
	    api::relax( [&]( auto s, auto d, auto e ) {
		return priority[d] += _p( _1(priority[d]), prio_fn( s, d ) );
	    } ),
	    api::record( roots,
			 [&]( auto d ) {
			     return priority[d] == _0;
			 }, api::strong )
	    )
	    .vertex_map( [&]( auto v ) {
		constexpr size_t W = sizeof(BitMaskTy)*8;
		auto w = _c( W );
		auto lsb = _1(posscol[v]);
		return posscol[v]
		    = expr::iif( priority[v] < w, _0,
				 ( lsb << expr::cast<BitMaskTy>( priority[v] + _1 ) ) - _1 );
	    } )
	    .vertex_filter( GA, ftrue, badv,
			    [&]( auto v ) {
				return posscol[v] == _0 && degree[v] != _0;
			    } )
	    .materialize();

	assert( badv.nActiveVertices() == 0
		&& "cannot encode posscol in available bits" );
	badv.del();

	log( iter, tm_iter, ftrue, ik_prio );

	while( !roots.isEmpty() ) {
	    frontier new_roots;
	    api::edgemap(
		GA,
		api::config( api::always_sparse ),
		api::filter( api::src, api::strong, roots ),
#if FUSION
		api::fusion( [&]( auto v ) {
		    // If priority drops to zero, or if only one color choice
		    // remains, then process immediately, else keep in overflow
		    // bucket
/*
		    return expr::iif(
			priority[d] == _0
			|| ( posscol[d] & ( posscol[d] - _1 ) ) == _0,
			_1s(v), // stay inactive
			_1(v) ); // process
*/
		    return expr::make_seq(
			posscol[v] &= ~( posscol[v] - _1 ), // reduce to best color (attention: race conditions with relax in case progressed on single color condition; should be benign)
			_1(v) // process immediately
			);
		} ),
#endif
		// Remove color of source from possible colors of d
		// Destination is enabled if all predecessors removed,
		// or (optimisation) only one color remains
		// The race between atomic posscol &= and check of single bit
		// set in posscol is benign
		api::relax( [&]( auto s, auto d, auto e ) {
		    return expr::make_seq(
			// Can we do ECL optimisations?
			posscol[d] &= ~posscol[s],
			priority[d].count_down(
			    _p( _0(priority[d]), prio_fn( s, d ) ) )
			// || ( posscol[d] & ( posscol[d] - _1 ) ) == _0
			);
		} ),
		api::record( new_roots, api::reduction, api::strong )
		)
		.vertex_map(
		    new_roots,
		    [&]( auto v ) {
			return posscol[v] &= ~( posscol[v] - _1 );
		    } )
		.materialize();

	    if( debug ) {
		roots.toSparse( part );
		new_roots.toSparse( part );

		frontier pz;
		make_lazy_executor( part )
		    .vertex_filter(
			GA,
			ftrue,
			pz,
			[&]( auto v ) {
			    return posscol[v] == _0;
			} )
		    .materialize();
		pz.toSparse( part );

		std::cerr << "roots    : " << roots << "\n";
		std::cerr << "new_roots: " << new_roots << "\n";
		std::cerr << "no colors: " << pz << "\n";
		print( std::cerr, part, priority );
		std::cerr << std::hex;
		print( std::cerr, part, posscol );
		std::cerr << std::dec;

		assert( pz.nActiveVertices() == 0 );
		pz.del();
	    }

	    log( iter, tm_iter, roots, ik_color );

	    roots.del();
	    roots = new_roots;
	}

	roots.del();

	make_lazy_executor( part )
	    .vertex_map(
		[&]( auto d ) {
		    return color[d] = expr::tzcnt<BitMaskTy>( posscol[d] );
		} )
	    .materialize();

	log( iter, tm_iter, ftrue, ik_post );

	ftrue.del();
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
    api::vertexprop<VID,VID,var_priority> priority;
    expr::rnd::prng<VID> prng;
    std::vector<info> info_buf;
};

template <class GraphType>
using Benchmark = GC_JP_Fusion<GraphType>;

#include "driver.C"
