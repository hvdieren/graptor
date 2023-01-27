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

// LLF: log-largest-first (1) or largest-first (0)
// Use largest-first by default for backward compatibility
#ifndef LLF
#define LLF 0
#endif

// FUSION: enable fusion operator
#ifndef FUSION
#define FUSION 1
#endif

// VECTORIZE indicates vector length of fusion operation with
// scalar destination. 1 means no vectorization (scalar baseline),
// 8 is typical vector length for AVX2 with 4-byte flags, 16 is typical
// for AVX512 with 4-byte flags, 32 is useful either way for flags stored
// as bits (FLAG_STORAGE 2)
#ifndef VECTORIZE
#define VECTORIZE 1
#endif

// Vectorization applies to the fusion operation. The same operation in
// vertex_map is vectorized differently.
// #if FUSION == 0
// #undef VECTORIZE
// #define VECTORIZE 1

#if VECTORIZE == 0
#error VECTORIZE may not be zero
#endif

#ifndef FLAG_TYPE
#define FLAG_TYPE 0
#endif

#ifndef FLAG_STORAGE
#define FLAG_STORAGE 0
#endif

enum gc_variable_name {
    var_color = 0,
    var_i = 1,
    var_priority = 2,
    var_dep = 3,
    var_new = 4,
    var_usedcol = 5,
    var_index = 6,
    var_tmp = 7,
    var_sel = 8,
    var_nghcol = 9,
    var_edge = 10,
    var_degrees_ro = expr::aid_graph_degree
};

enum iteration_kind {
    ik_prio,
    ik_depth,
    ik_color,
    ik_post,
    ik_step1
};

std::ostream & operator << ( ostream & os, iteration_kind k ) {
    switch( k ) {
    case ik_prio: os << "priority"; break;
    case ik_depth: os << "DAG-depth"; break;
    case ik_color: os << "coloring"; break;
    case ik_post: os << "post-process"; break;
    case ik_step1: os << "step-1"; break;
    default: os << "<unknown>"; break;
    }
    return os;
}

#if FLAG_TYPE == 0
using FlagTy = int32_t; // signed integer; 32-bit wide for scatter
#elif FLAG_TYPE == 1
using FlagTy = int8_t; // signed integer; 32-bit wide for scatter
#elif FLAG_TYPE == 2
using FlagTy = bitfield<1>; // single bit
#elif FLAG_TYPE == 3
using FlagTy = logical<4>; // natural width for scatter, convert from bit
#else
#error "illegal FLAG_TYPE"
#endif

#if FLAG_STORAGE == 0
using FlagEnc = array_encoding<int32_t>;
#elif FLAG_STORAGE == 1
using FlagEnc = array_encoding<int8_t>;
#elif FLAG_STORAGE == 2
using FlagEnc = array_encoding_bit<1>;
#else
#error "illegal FLAG_STORAGE"
#endif

#if FLAG_STORAGE == 3 && FLAG_TYPE != 2 && FLAG_TYPE != 3
#warn "this combination of FLAG_STORAGE and FLAG_TYPE may not work"
#endif

template <class GraphType>
class GC_JP_Fusion {
    static constexpr float mu = 0.0025f;

public:
    GC_JP_Fusion( GraphType & _GA, commandLine & P )
	: GA( _GA ),
	  color( GA.get_partitioner(), "color" ),
	  priority( GA.get_partitioner(), "priority" ),
	  usedcol( GA.get_partitioner(), "used colors" ),
	  // prng( GA.get_partitioner(),
		// static_cast<VID>( static_cast<float>( GA.getCSR().max_degree() )
				  // * ( 1.0 + mu ) ) ),
	  info_buf( 60 ) {
	itimes = P.getOption( "-itimes" );
	debug = P.getOption( "-debug" );
	outfile = P.getOptionValue( "-gc:outfile" );

	static bool once = false;
	if( !once ) {
	    once = true;
	    std::cerr << "LLF=" << LLF << "\n";
	    std::cerr << "FUSION=" << FUSION << "\n";
	    std::cerr << "VECTORIZE=" << VECTORIZE << "\n";
	    std::cerr << "FLAG_TYPE=" << FLAG_TYPE
		      << " width=" << sizeof(FlagTy) << "\n";
	    std::cerr << "FLAG_STORAGE=" << FLAG_STORAGE
		      << " width=" << sizeof(typename FlagEnc::stored_type)
		      << "\n";
	}
    }
    ~GC_JP_Fusion() {
	color.del();
	priority.del();
	usedcol.del();
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
		      << ' ' << (1e-6*(double(nacte)/delay)) << " meps"
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

	constexpr bool debug_verbose = false;

	timer tm_iter;
	tm_iter.start();
	iter = 0;

	expr::array_ro<VID, VID, var_degrees_ro> degree(
	    const_cast<VID *>( GA.getCSR().getDegree() ) );
	expr::array_ro<EID, VID, var_index> index(
	    const_cast<EID *>( GA.getCSR().getIndex() ) );
	expr::array_ro<VID, EID, var_edge> edge(
	    const_cast<VID *>( GA.getCSR().getEdges() ) );

	frontier ftrue = frontier::all_true( n, m );

	make_lazy_executor( part )
	    .vertex_map( [&]( auto v ) { return priority[v] = _0; } )
	    .vertex_map( [&]( auto v ) { return color[v] = _1s; } )
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
		< expr::make_unop_lzcnt<VID>( degree[d] )
		|| ( expr::make_unop_lzcnt<VID>( degree[s] )
		     == expr::make_unop_lzcnt<VID>( degree[d] )
		     && s < d );
#else
	    return degree[s] > degree[d]
		|| ( degree[s] == degree[d] && s < d );
#endif
	};

#if FLAG_STORAGE == 2
	parallel_loop( (EID)0, (m+7)/8, [&]( EID e ) { usedcol.get_ptr()[e] = 0; } );
#else
	parallel_loop( (EID)0, m, [&]( EID e ) { usedcol.get_ptr()[e] = 0; } );
#endif

	frontier roots;
	api::edgemap(
	    GA,
	    api::relax( [&]( auto s, auto d, auto e ) {
		return priority[d] += _p( _1(priority[d]), prio_fn( s, d ) );
	    } ),
	    api::record( roots, [&]( auto d ) { return priority[d] == _0; },
			 api::strong )
	    )
	    // Set color of roots (initial color = _1s to indicate undefined)
	    .vertex_map( roots, [&]( auto v ) { return color[v] = _0; } )
	    .materialize();

	log( iter, tm_iter, ftrue, ik_prio );

	auto check_color = [&]( auto s, auto d, auto color_d ) {
	    return expr::make_seq(
		// Mark color as used
		usedcol[index[d]+expr::cast<EID>(color[s])]
		= _p( _1s(usedcol[expr::cast<EID>(d)]), // set flag: colour used
		      color[s] < degree[d] // array bounds check
		      // Could check color[s] > color_d over and above s != d
		      // such that we don't store values for colors that have
		      // already been discarded (not color_d incremented if
		      // conflict without checking usedcol[color_d]).
		    ), // && s != d ), // not a self-edge
		// Scan forward if conflict
		expr::set_mask(
		    // Only perform loop if conflict
		    color[s] == color_d && s != d,
		    expr::make_seq(
			color_d += _1,
			expr::make_loop(
			    // condition: color in use and possible
			    // to increase
			    usedcol[index[d]+expr::cast<EID>(color_d)] < _0
			    && color_d < degree[d],
			    // loop body and increment
			    color_d += _1,
			    // return value - anything
			    _0(d) )
			) )
		);
	};

	while( !roots.isEmpty() ) {
	    frontier new_roots;
#if FUSION
	    auto select_col_pull = [&]( auto v ) {
		auto idx = expr::make_scalar<var_i,VID,decltype(v)::VL>();
		auto col = expr::make_scalar<var_tmp,VID,decltype(v)::VL>();
		return expr::make_seq(
		    col = _0,
		    // color[v] = _0,
		    idx = _0,
		    expr::make_loop(
			idx < degree[v], // condition
			// loop body and increment
			expr::make_seq(
			    expr::let<var_nghcol>(
				edge[index[v]+expr::cast<EID>(idx)],
				[&]( auto ngh ) {
				    return check_color( ngh, v, col );
				} ),
			    idx += _1 ), // increment
			// process immediately -- TODO: drop if degree 0
			expr::make_seq(
			    color[v] = col,
			    _1(v) // final value
			    )
			)
		    );
	    };
	    api::edgemap(
		GA,
		api::config( api::always_sparse ),
		api::filter( api::src, api::strong, roots ),
		api::fusion( [&]( auto v ) {
		    return select_col_pull( v );
		} ),
		api::relax( [&]( auto s, auto d, auto e ) {
		    return priority[d].count_down( _0(priority[d]) );
		} ),
		api::record( new_roots, api::reduction, api::strong )
		)
		.materialize();

#else // non-fusion case below
	    // Push edgemap: count-down predecessors of our neighbours
	    api::edgemap(
		GA,
		api::config( api::always_sparse ),
		api::filter( api::src, api::strong, roots ),
		api::relax( [&]( auto s, auto d, auto e ) {
		    return priority[d].count_down( _0(priority[d]) );
		} ),
		api::record( new_roots, api::reduction, api::strong )
		)
		// Set color of new roots to 0
		.vertex_map( new_roots, [&]( auto v ) {
		    return color[v] = _0;
		} )
		.materialize();
	    // Pull edgemap: gather colours of neighbours of activated vertices
	    api::edgemap(
		GA,
		api::config( api::always_sparse ),
		api::filter( api::dst, api::strong, new_roots ),
		api::relax( [&]( auto s, auto d, auto e ) {
		    return check_color( s, d, color[d] );
		} )
		)
		.materialize();
#endif // fusion/no-fusion

	    if( debug && debug_verbose ) {
		roots.toSparse( part );
		new_roots.toSparse( part );

		std::cerr << "roots    : " << roots << "\n";
		std::cerr << "new_roots: " << new_roots << "\n";
		print( std::cerr, part, priority );
		print( std::cerr, part, color );
		std::cerr << std::hex;
		print( std::cerr, part, usedcol );
		std::cerr << std::dec;
	    }

	    log( iter, tm_iter, roots, ik_color );

	    roots.del();
	    roots = new_roots;
	}

	roots.del();
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
		return color[s] == color[d] && s != d;
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
    api::vertexprop<VID,VID,var_priority> priority;
    api::edgeprop<FlagTy,EID,var_usedcol,FlagEnc> usedcol;
    // expr::rnd::prng<VID> prng;
    std::vector<info> info_buf;
};

template <class GraphType>
using Benchmark = GC_JP_Fusion<GraphType>;

#include "driver.C"
