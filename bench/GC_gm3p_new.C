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

enum gc_variable_name {
    var_color = 0,
    var_mupd = 1,
    var_sh = 2,
    var_smod = 3,
    var_unavail = 4,
    var_used = 5,
    var_diff = 6,
    var_wdiff = 7,
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

/**
 * ColTy Type representing a vertex color.
 * Graptor makes the assumption that the number of vertices in the graph is
 * less than 2**31 when VID is a 32-bit quantity. This way, the top bit is
 * available for metadata information in the Graptor data structure.
 * The number of colours needed to colour a graph is not more than the number
 * of vertices. As such, a color type of the same width as a vertex ID will
 * also have a spare bit. We make this a sign bit and adopt the convention
 * that negative colours are not allowed, in particular, they indicate
 * undefined colours.
 * Making the colour type signed makes it more efficient to test for undefined
 * colours.
 */
using ColTy = std::make_signed_t<VID>;

/**
 * BitMaskTy Type representing a bitmask of (un-)available colours.
 * A width different from ColTy is inconvenient in vectorization, however,
 * the algorithm suffers from too short bitmasks as they cannot store
 * accurate information (it collects information only about as many colours
 * as there are bits in the mask).
 */
using BitMaskTy = uint64_t;

template <class GraphType>
class GC_GM3P {
    static constexpr float mu = 0.0025f;

public:
    GC_GM3P( GraphType & _GA, commandLine & P )
	: GA( _GA ),
	  spec_iter( P.getOptionLongValue( "-gc:iter", 2 ) ),
	  color( GA.get_partitioner(), "color" ),
	  used( GA.get_partitioner(), "used colors" ),
	  unavail( GA.get_partitioner(), "unavailable colors" ),
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
	    // std::cerr << "WITH_DEPTH=" << WITH_DEPTH << "\n";
	}
    }
    ~GC_GM3P() {
	color.del();
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
	constexpr bool debug_info = false;
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

	    if( debug && debug_info ) {
		print( std::cerr, part, color );
		std::cerr << std::hex;
		print( std::cerr, part, unavail );
		std::cerr << std::dec;
	    }

	    log( iter, tm_iter, active, ik_speculative );

	    frontier f = list_conflicts_and_clear( active );

	    if( debug && debug_info ) {
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

	// Fix up colors in singleton components
	expr::array_ro<VID, VID, var_degrees_ro> degree(
	    const_cast<VID *>( GA.getCSR().getDegree() ) );
	make_lazy_executor( part )
	    .vertex_map(
		[&]( auto v ) {
		    return color[v] = _p( _0(color[v]), color[v] < _0 );
		} )
	    .materialize();

	serial_coloring( active );

	if( debug && debug_info )
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
	    .vertex_map( active, [&]( auto v ) { return color[v] = _1s; } )
	    .materialize();

	auto greedy = [&]( auto s, auto dd ) {
	    auto & col = color;
	    constexpr ColTy w = sizeof(BitMaskTy)*8;
	    constexpr ColTy log_w = ilog2( w );
	    
	    auto d = expr::set_mask( dd != s, dd ); // avoid self-loops

	    auto Mone = expr::slli<w-1>( _1s(unavail[d]) );
	    auto wmask = expr::slli<log_w>( _1s(unavail[d]) );

	    return expr::let<var_diff>(
		expr::cast<BitMaskTy>( color[s] - color[d] - _1 ),
		[&]( auto wdiffm1 ) {
		    auto match = wdiffm1 == _1s;
		    auto inmsk = ( wdiffm1 & wmask ) == _0;
		    return expr::let<var_sh>(
			expr::iforz( match,
				     expr::lzcnt<BitMaskTy>( ~unavail[d] ) + _1 ),
			[&]( auto shift ) {
			    auto setmsk = expr::iforz( inmsk, Mone >> wdiffm1 );

			    return expr::make_seq(
				unavail[d] = ( unavail[d] << shift ) | setmsk,
				color[d] += expr::iif(
				    color[d] < _0,
				    expr::cast<ColTy>( shift ),
				    _1 )
				);
			} );
		} );
	};

	auto simple = [&]( auto s, auto d ) {
	    return expr::make_seq(
		unavail[d] |= _p(
		    _1(unavail[d]) << expr::cast<BitMaskTy>( color[s] ),
		    color[s] != _1s ),
		color[d] = expr::tzcnt<VID>( ~unavail[d] )
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
	
	VID * const used_p = used.get_ptr();

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
		// Returns true if color reset.
		// Check for self-loops so we don't need to redo vertices
		// in serial fashion when it's not necessary.
		// If the frontier is all-true, we might also require s < d,
		// but for other frontiers this may leave conflicts undetected.
		return color[d] = _p( _1s(color[d]),
				      color[s] == color[d] && s != d );
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
		return color[s] == color[d] && s != d;
		// return color[s] == color[d];
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
    api::vertexprop<VID,VID,var_used> used;
    api::vertexprop<BitMaskTy,VID,var_unavail> unavail;
    // expr::rnd::prng<VID> prng;
    std::vector<info> info_buf;
};

template <class GraphType>
using Benchmark = GC_GM3P<GraphType>;

#include "driver.C"
