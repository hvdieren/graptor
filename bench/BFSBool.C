#include "graptor/graptor.h"
#include "graptor/api.h"

// TODO: consider adding narrow width level

using expr::_0;
using expr::_1;
using expr::_1s;
using expr::_c;
using expr::_true;
using expr::_false;

// By default set options for highest performance
#ifndef DEFERRED_UPDATE
#define DEFERRED_UPDATE 0
#endif

#ifndef UNCOND_EXEC
#define UNCOND_EXEC 0
#endif

#ifndef CONVERGENCE
#define CONVERGENCE 1
#endif

#ifdef LEVEL_ASYNC
#undef LEVEL_ASYNC
#endif
#define LEVEL_ASYNC 0

#ifdef MEMO
#undef MEMO
#endif
#define MEMO 0

#ifndef FUSION
#define FUSION 0
#endif

#ifndef UNVISITED_BIT
#define UNVISITED_BIT 1
#endif

enum var {
    var_level = 0,
    var_unvisited = 1
};

// We compare against level-1 in validation, hence require signed levels
using sVID = std::make_signed_t<VID>;

template <class GraphType>
class BFSBool {
public:
    BFSBool( GraphType & _GA, commandLine & P )
	: GA( _GA ),
	  level( GA.get_partitioner(), "bfs level" ),
	  unvisited( GA.get_partitioner(), "unvisited at any level" ),
	  info_buf( 60 ) {
	itimes = P.getOption( "-itimes" );
	debug = P.getOption( "-debug" );
	calculate_active = P.getOption( "-cactive" );
	start = GA.remapID( P.getOptionLongValue( "-start", 0 ) );

	static bool once = false;
	if( !once ) {
	    once = true;
	    std::cerr << "DEFERRED_UPDATE=" << DEFERRED_UPDATE << "\n";
	    std::cerr << "UNCOND_EXEC=" << UNCOND_EXEC << "\n";
	    std::cerr << "LEVEL_ASYNC=" << LEVEL_ASYNC << "\n";
	    std::cerr << "CONVERGENCE=" << CONVERGENCE << "\n";
	    std::cerr << "MEMO=" << MEMO << "\n";
	    std::cerr << "FUSION=" << FUSION << "\n";
	    std::cerr << "UNVISITED_BIT=" << UNVISITED_BIT << "\n";
	}
    }
    ~BFSBool() {
	level.del();
	unvisited.del();
    }

    struct info {
	double delay;
	float density;
	float active;
	EID nacte;
	VID nactv;
	frontier_type ftype;

	void dump( int i ) {
	    std::cerr << "Iteration " << i << ": " << delay
		      << ' ' << ftype
		      << " density: " << density
		      << " nacte: " << nacte
		      << " nactv: " << nactv
		      << " active: " << active << "\n";
	}
    };

    struct stat { };

    void run() {
	const partitioner &part = GA.get_partitioner();
	sVID n = GA.numVertices();
	EID m = GA.numEdges();

	// Initialise arrays
	// TODO: level.fill() and unvisited.fill() may be faster
	level.fill( part, ~(VID)0 );
	unvisited.fill( part, ~(VID)0 );
/*
	make_lazy_executor( part )
	    .vertex_map( [&]( auto v ) { return level[v] = _1s; } )
	    .vertex_map( [&]( auto v ) { return unvisited[v] = _1s; } )
	    .materialize();
*/

	level.set( start, 0 );
	unvisited.set( start, 0 );

	// Create initial frontier
	frontier F = frontier::create( n, start, GA.getOutDegree(start) );

	iter = 0;
	active = 1;

	// Enable fusion immediately if the graph has relatively low
	// degrees (e.g., road network). Otherwise, wait until a dense
	// iteration has occured (in which case further dense iterations
	// will still take precedence over sparse/fusion iterations).
#if FUSION
	bool enable_fusion = isLowDegreeGraph( GA );
#endif

	while( !F.isEmpty() ) {  // iterate until all vertices visited
	    timer tm_iter;
	    tm_iter.start();

	    // Traverse edges, remove duplicate live destinations.
#if UNCOND_EXEC
	    auto filter_strength = api::weak;
#else
	    auto filter_strength = api::strong;
#endif
	    frontier output;
	    api::edgemap(
		GA,
		// api::config( api::frac_threshold( 10 ) ),
		api::record( output, api::reduction, api::strong ),
		api::filter( filter_strength, api::src, F ),
#if CONVERGENCE
		api::filter( api::weak, api::dst, [&]( auto d ) {
		    // Making a guess on the type of the frontier
#if FRONTIER_BITMASK
		    expr::bitarray_intl<unsigned char,VID,expr::aid_frontier_new> curfr;
#else
		    expr::array_intl<logical<4>,VID,expr::aid_frontier_new> curfr;
#endif
		    return !curfr[d] && unvisited[d] != _0;
		    // return unvisited[d] != _0;
		} ),
#endif
		// unvisited is effectively the amalgamation of prior
		// frontiers, so simplify code by:
		//   relax: return true (conditional on strong src F)
		//   convergence: current frontier false AND prior F false
		//   prior if is dense, but seq access; updated together with
		//   level
		// This avoids duplication in code handling unvisited and
		// code handling new frontier (effectively same operation)
		api::relax( [&]( auto s, auto d, auto e ) {
		    return unvisited[d] != _0;
		    // return unvisited[d] &= _0;
		},
		    // vertex-op / conditional on frontier
		    [&]( auto d ) { return expr::make_seq(
			    level[d] = _c( iter+1 ),
			    unvisited[d] = _0
			    ); }
		    )
		)
		.materialize();

	    // print( std::cerr, part, level );
	    
	    if( itimes ) {
		VID active = 0;
		if( calculate_active ) { // Warning: expensive, included in time
		    for( VID v=0; v < n; ++v )
			if( level[v] == n+1 )
			    active++;
		}
		info_buf.resize( iter+1 );
		info_buf[iter].density = F.density( GA.numEdges() );
		info_buf[iter].delay = tm_iter.next();
		info_buf[iter].active = float(active)/float(n);
		info_buf[iter].nacte = F.nActiveEdges();
		info_buf[iter].nactv = F.nActiveVertices();
		info_buf[iter].ftype = F.getType();
		if( debug )
		    info_buf[iter].dump( iter );
	    }

	    // Cleanup old frontier
	    F.del();
	    F = output;

	    ++iter;

#if FUSION
	    if( !api::default_threshold().is_sparse( F, m ) )
		enable_fusion = true;
#endif
	}

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
	// VID longest = iter - 1;
	VID longest = 0;
	VID longest_v = start;
	for( VID v=0; v < n; ++v )
	    if( longest < level[v] && level[v] < n ) {
		longest = level[v];
		longest_v = v;
	    }

	std::cout << "Longest path from " << start
		  << " (original: " << GA.originalID( start ) << ") : "
		  << longest
		  << " at: " << longest_v
		  << " (" << GA.originalID( longest_v ) << ")\n";

	frontier wrong;
	// Assuming symmetric graph, showing inverse properties:
	// Any neigbours must be immediately before us, together with
	// us or at most immediately after us.
	// Note: condition for wrong values
	if( GA.isSymmetric() ) {
	    api::edgemap(
		GA,
		api::record( wrong, api::reduction, api::strong ),
		api::relax( [&]( auto s, auto d, auto e ) {
		    return level[s] > level[d] + _1
			|| level[s] < level[d] - _1;
		} )
		)
		.materialize();
	}  else {
	    // Directed graph: some paths may take longer to reach us and that
	    // is fine. But no-one that can reach us can do it more than
	    // one step ahead of us.
	    api::edgemap(
		GA,
		api::record( wrong, api::reduction, api::strong ),
		api::relax( [&]( auto s, auto d, auto e ) {
		    return level[s] < level[d] - _1;
		} )
		)
		.materialize();
	}
	std::cout << "Number of vertices with nghbs not activated in time: "
		  << wrong.nActiveVertices()
		  << ( wrong.nActiveVertices() == 0 ? " PASS" : " FAIL" )
		  << "\n";

	std::cout << "Number of activated vertices: " << active << "\n";
	std::cout << "Number of vertices: " << n << "\n";
#if !UNCOND_EXEC
	std::cout << "Every vertex activated at most once: "
		  << ( n >= active ? "PASS" : "FAIL" ) << "\n";
#endif

#if BFS_DEBUG && 0
	all.calculateActiveCounts( GA.getCSR() );
	std::cout << "all vertices: " << all.nActiveVertices() << "\n";
#endif
    }

private:
    const GraphType & GA;
    bool itimes, debug, calculate_active;
    sVID iter;
    VID start, active;
#if UNVISITED_BIT
    api::vertexprop<bitfield<1>,VID,var_unvisited,
		    array_encoding_bit<1>> unvisited;
#else
    api::vertexprop<VID,VID,var_unvisited> unvisited;
#endif
    api::vertexprop<sVID,VID,var_level> level;
    std::vector<info> info_buf;
};

#ifndef NOBENCH
template <class GraphType>
using Benchmark = BFSBool<GraphType>;

#include "driver.C"
#endif // NOBENCH
