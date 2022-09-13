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

#ifndef USE_MSB
#define USE_MSB 1
#endif

#ifndef USE_BM64
#define USE_BM64 1
#endif

enum gc_variable_name {
    var_color = 0,
    var_mupd = 1,
    var_sh = 2,
    var_smod = 3,
    var_unavail = 4,
    var_bound = 5,
    var_used = 6,
    var_diff = 7,
    var_dfsh = 8,
    var_degrees_ro = expr::aid_graph_degree
};

enum iteration_kind {
    ik_speculative,
    ik_serial,
    ik_conflicts
};

std::ostream & operator << ( ostream & os, iteration_kind k ) {
    switch( k ) {
    case ik_speculative: os << "speculative"; break;
    case ik_serial: os << "serial"; break;
    case ik_conflicts: os << "conflicts"; break;
    default: os << "<unknown>"; break;
    }
    return os;
}

#if USE_BM64
using BitMaskTy = uint64_t;
#else
using BitMaskTy = uint32_t;
#endif

using ColTy = std::make_signed_t<VID>;

#if USE_MSB
#define SHIFT_OUT <<
#define SHIFT_IN >>
template<typename AST>
auto base_mask( AST mask ) {
    return expr::slli<sizeof(BitMaskTy)*8-1>( _1s( mask ) );
}
template<typename T, typename AST>
auto zero_cnt( AST mask ) {
    return expr::lzcnt<T>( mask );
}
#else
#define SHIFT_OUT >>
#define SHIFT_IN <<
template<typename AST>
auto base_mask( AST mask ) {
    return expr::srli<sizeof(BitMaskTy)*8-1>( _1s( mask ) );
}
template<typename T, typename AST>
auto zero_cnt( AST mask ) {
    return expr::tzcnt<T>( mask );
}
#endif

template <class GraphType>
class GC_GM3P {
    static constexpr float mu = 0.0025f;

public:
    GC_GM3P( GraphType & _GA, commandLine & P )
	: GA( _GA ),
	  spec_iter( P.getOptionLongValue( "-gc:iter", 2 ) ),
	  color( GA.get_partitioner(), "color" ),
	  bound( GA.get_partitioner(), "bound on seen color" ),
	  used( GA.get_partitioner(), "used colors" ),
	  unavail( GA.get_partitioner(), "unavailable coors" ),
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
	    std::cerr << "USE_MSB=" << USE_MSB << "\n";
	    std::cerr << "USE_BM64=" << USE_BM64 << "\n";
	}
    }
    ~GC_GM3P() {
	color.del();
	bound.del();
	used.del();
	unavail.del();
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

	frontier ftrue = frontier::all_true( n, m );

	frontier active = ftrue;
	for( int k=0; k < spec_iter && !active.isEmpty(); ++k ) {
	    speculative_coloring( active );

	    if( debug && false )
		print( std::cerr, part, color );

	    log( iter, tm_iter, active, ik_speculative );

	    frontier f = list_conflicts_and_clear( active );

	    if( debug && false ) {
		f.toSparse( part );
		active.toSparse( part );
		std::cerr << "active: " << f << "\n";
		std::cerr << "conflicts: " << f << "\n";
		print( std::cerr, part, color );
	    }

	    log( iter, tm_iter, active, ik_conflicts );

	    active.del();
	    active = f;
	}

	serial_coloring( active );

	if( debug && false )
	    print( std::cerr, part, color );

	log( iter, tm_iter, active, ik_serial );

	active.del();
	ftrue.del();
    }

    void speculative_coloring( frontier & active ) {
	const partitioner &part = GA.get_partitioner();
	VID n = GA.numVertices();
	EID m = GA.numEdges();

	expr::array_ro<VID, VID, var_degrees_ro> degree(
	    const_cast<VID *>( GA.getCSR().getDegree() ) );

	make_lazy_executor( part )
	    .vertex_map( active, [&]( auto v ) { return unavail[v] = _0; } )
	    .vertex_map( active, [&]( auto v ) { return bound[v] = _1s; } )
	    .vertex_map( active, [&]( auto v ) {
		return color[v]
		    = expr::iif( degree[v] == _0, _1s, _0(color[v]) );
	    } )
	    .materialize();

	auto greedy = [&]( auto s, auto dd ) {
	    auto & col = color;
	    
	    auto d = expr::set_mask( dd != s, dd ); // avoid self-loops
	    // auto d = dd;
	    auto zero = _0( col[d] );
	    auto undef = _1s( col[d]);
	    auto one = _1( col[d] );

	    auto Mone = base_mask( unavail[d] );
	    auto Mzero = _0( unavail[d] );

	    constexpr ColTy cw = sizeof(BitMaskTy)*8;
	    constexpr ColTy log_w = ilog2( cw );

	    auto w = expr::slli<log_w>( _1(bound[d]) ); // expr::constant_val( bound[d], cw );
	    auto Mw = expr::slli<log_w>( _1(unavail[d]) ); // expr::constant_val( unavail[d], cw );
	    // auto Mwm = expr::slli<log_w>( _1(unavail[d]) ) - _1;

	    // using MTr = simd::detail::mask_logical_traits<sizeof(ColTy),decltype(dd)::VL>;

	    // Conditional set of bound value. The bound is the smallest colour
	    // of a neighbour that is larger than the current colour and
	    // not captured by the mask. If we eventually assign the colour
	    // bound[d], then we know we have assigned the same colour as
	    // a previously seen neighbour.
	    // If the colour is less then the bound and not present in the mask,
	    // then the colour has not been seen on a neighbour.
	    return expr::let<var_diff>(
		color[s] - color[d],
		[&]( auto diff ) {
	    return expr::template let<var_mupd>( // updated mask (add neighbour)
#if 0
		expr::iif( ( diff | col[s] ) < _0, // need a positive shift value
			   unavail[d] | ( Mone SHIFT_IN expr::cast<BitMaskTy>( diff ) ), // set bit
			   unavail[d] ),         // nothing to set
#else
		// Assumptions: BitMaskTy = 32 bits ; MSB
		// bit pattern with top bit 1 if condition met, else zeros
		( expr::cast<BitMaskTy>( ~( diff | col[s] ) & ( expr::slli<31>( _1s(col[s]) ) ) ) SHIFT_IN expr::cast<BitMaskTy>( diff ) ) | unavail[d],
#endif
		[&]( auto Mupd ) {
		    return expr::template let<var_sh>( // positions to shift mask
			expr::iif( diff == _0 && !( col[s] < _0 ), // colour conflict?
				   Mzero,             // no: 0 shift; yes: find next
				   zero_cnt<BitMaskTy>( ~Mupd ) ),
			[&]( auto sh ) {
			    return expr::make_seq(
/*
				bound[d] = expr::add_predicate(
				    smod,
				    smod < bound[d] && col[d]+w < smod ),
*/
				// x86 scalar >> (shrx) masks sh by w
				// bits ((1<<w)-1) shifting by zero if
				// sh == w, while SSE/AVX will
				// set to zero if sh >= w
				// Note: sh can be at most #bits in mask
				//  due to computed by zero count.
				//  Thus, we only need to consider the
				//  case where sh == w.
				unavail[d] = expr::iif( sh >= Mw,
							Mupd SHIFT_OUT sh,
							Mzero ),
				col[d] = expr::iif(
				    col[d] < _0,
				    col[d] + expr::cast<ColTy>( sh ),          // advance
				    zero )               // first colour
				);
			} );
		} );
		} );
	};

	auto simple = [&]( auto s, auto d ) {
	    auto Mone = base_mask( unavail[d] );
	    return expr::make_seq(
		unavail[d] |= _p(
		    Mone SHIFT_IN expr::cast<BitMaskTy>( color[s] ),
		    color[s] != _1s ),
		color[d] = zero_cnt<VID>( ~unavail[d] )
		);
		
	};

	api::edgemap(
	    GA,
	    api::filter( api::dst, api::strong, active ),
	    api::relax( [&]( auto s, auto d, auto e ) {
		return greedy( s, d );
		// return simple( s, d );
	    } )
	    )
	    .materialize();
    }

    void serial_coloring( frontier & f ) {
	const partitioner &part = GA.get_partitioner();

	// Nothing to do if frontier is empty
	if( f.isEmpty() )
	    return;
	
	// Current implementation
	f.toSparse( part );
	assert( f.isSparse() && "NYI - different frontier types" );

	const VID n = GA.numVertices();
	const VID * fv = f.getSparse();
	const VID fn = f.nActiveVertices();
	const EID * idx = GA.getCSR().getIndex();
	const VID * edge = GA.getCSR().getEdges();

	make_lazy_executor( part )
	    .vertex_map( [&]( auto v ) { return used[v] = _1s; } )
	    .materialize();
	
	VID * const used_p = reinterpret_cast<VID *>( used.get_ptr() );

	for( VID k=0; k < fn; ++k ) {
	    const VID v = fv[k];
	    const EID ee = idx[v+1];
	    for( VID e=idx[v]; e < ee; ++e ) {
		const VID u = edge[e];
		const VID c = color[u];
		if( c != ~(VID)0 )
		    used_p[c] = v;
	    }
	    VID c = 0;
	    for( ; c < n; ++c ) {
		if( used_p[c] != v )
		    break;
	    }
	    color.get_ptr()[v] = c;
	}
    }
    
    frontier list_conflicts_and_clear( frontier & active ) {
	frontier c;
	api::edgemap(
	    GA,
	    api::record( c, api::reduction, api::strong ),
	    api::filter( api::dst, api::strong, active ),
	    api::filter( api::dst,
			 [&]( auto d ) {
			     return color[d] != _1s;
			 }, api::weak ), 
	    api::relax( [&]( auto s, auto d, auto e ) {
		// Returns true if color reset
		return color[d] = _p( _1s(color[d]),
				      color[s] == color[d]
				      // && color[s] != _1s
				      // && s < d
		    );
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
		// In case there are self-loops
		// return color[s] == color[d] && s != d;
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
    int spec_iter;
    api::vertexprop<ColTy,VID,var_color> color;
    api::vertexprop<ColTy,VID,var_bound> bound;
    api::vertexprop<ColTy,VID,var_bound> used;
    api::vertexprop<BitMaskTy,VID,var_unavail> unavail;
    // expr::rnd::prng<VID> prng;
    std::vector<info> info_buf;
};

template <class GraphType>
using Benchmark = GC_GM3P<GraphType>;

#include "driver.C"
