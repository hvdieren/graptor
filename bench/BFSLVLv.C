#include "graptor/graptor.h"
#include "graptor/api.h"

using expr::_0;
using expr::_1;
using expr::_1s;
using expr::_c;

// By default set options for highest performance
#ifndef DEFERRED_UPDATE
#define DEFERRED_UPDATE 1
#endif

#ifndef UNCOND_EXEC
#define UNCOND_EXEC 1
#endif

#ifndef CONVERGENCE
#define CONVERGENCE 1
#endif

#ifndef LEVEL_ASYNC
#define LEVEL_ASYNC 1
#endif

#ifdef MEMO
#undef MEMO
#endif
#define MEMO 0

// TODO: FUSION works well for road networks: estimate network type and select at runtime
#ifndef FUSION
#define FUSION 0
#endif

enum var {
    var_current = 0,
    var_previous = 1
};

// Ensure signed comparisons between levels as ordered comparators in AVX2
// are more efficient on signed than unsigned.
using sVID = std::make_signed_t<VID>;

using bVID = sVID;
using Enc = array_encoding<sVID>;

// using bVID = int8_t;
// using Enc = array_encoding_wide<bVID>;

template <class GraphType>
class BFSLVLv {
public:
    BFSLVLv( GraphType & _GA, commandLine & P )
	: GA( _GA ),
	  a_level( GA.get_partitioner(), "current level" ),
#if !LEVEL_ASYNC || DEFERRED_UPDATE
	  a_prev_level( GA.get_partitioner(), "previous level" ),
#endif
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
	}
    }
    ~BFSLVLv() {
	a_level.del();
#if !LEVEL_ASYNC || DEFERRED_UPDATE
	a_prev_level.del();
#endif
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

	sVID ilvl = n+1; // std::numeric_limits<bVID>::max(); // can only store bVID
	// Assign initial labels
	make_lazy_executor( part )
	    .vertex_map( [&]( auto v ) { return a_level[v] = _c( ilvl ); } )
#if DEFERRED_UPDATE || !LEVEL_ASYNC
	    .vertex_map( [&]( auto v ) { return a_prev_level[v] = _c( ilvl ); } )
#endif
	    .materialize();

	a_level.get_ptr()[start] = 0;
#if DEFERRED_UPDATE || !LEVEL_ASYNC
	a_prev_level.get_ptr()[start] = 0;
#endif

	// Create initial frontier
	frontier F = frontier::create( n, start, GA.getOutDegree(start) );

	iter = 0;
	active = 1;

#if BFS_DEBUG
	frontier all = frontier::dense( n, part );
	all.toDense(part);
	all.getDenseB()[start] = true;
#endif

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
#if DEFERRED_UPDATE
		// TODO: if not FUSION, compare a_level[d] to iters,
		//       obviating the need for prev_level
		// TODO: try reduction_or_method
		api::record( output,
			     [&]( auto d ) {
				 return a_level[d] != a_prev_level[d]; 
			     }, api::strong ),
#else
		api::record( output, api::reduction, api::strong ),
/*
 * This frontier rule is correct, and does not require a_prev_level, however
 * it is slower due to enabling too many vertices, in particular those whose
 * level has been updated in a chain due to asynchronous and unconditional
 * execution, but whose level is still too large to be final. It is more
 * efficient to wake those up later. Selecting a_level[d] == _c( iter+1 )
 * proves insufficient.
		api::record( output,
			     [&]( auto d ) {
				 // A vertex is active if its level has
				 // grown to at least the current iteration:
				 // a_level[d] >= iter+1
				 return a_level[d] > _c( iter );
			     }, api::strong ),
*/
#endif
		api::filter( filter_strength, api::src, F ),
#if CONVERGENCE
		api::filter( api::weak, api::dst, [&]( auto d ) {
		    // Use > comparison as it is faster than >= on AVX2
		    return a_level[d] > _c( iter+1 );
		} ),
#endif
#if FUSION
		api::config( api::fusion_select( enable_fusion ) ),
		api::fusion( [&]( auto s, auto d, auto e ) {
		    return expr::cast<int>(
			expr::iif( a_level[d].min( a_level[s]+_1 ),
				   _1s, // no change, ignore d
				   _1 // changed, propagate change
			    ) );
		} ),
#endif
		api::relax( [&]( auto s, auto d, auto e ) {
#if LEVEL_ASYNC
		    return a_level[d].min( a_level[s] + _1 );
#else
		    return a_level[d].min( a_prev_level[s] + _1 );
#endif
		} )
		)
		.materialize();
#if DEFERRED_UPDATE || !LEVEL_ASYNC
	    maintain_copies( part, /*output,*/ prev_level, level );
#endif

	    // print( std::cerr, part, a_level );
	    
	    if( itimes ) {
		VID active = 0;
		if( calculate_active ) { // Warning: expensive, included in time
		    for( VID v=0; v < n; ++v )
			if( a_level[v] == n+1 )
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
	sVID longest = 0;
	VID longest_v = start;
	for( VID v=0; v < n; ++v )
	    if( longest < a_level[v] && a_level[v] < n ) {
		longest = a_level[v];
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
		    return a_level[s] > a_level[d] + _1
			|| a_level[s] < a_level[d] - _1;
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
		    return a_level[s] < a_level[d] - _1;
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

    // const sVID * get_level() const { return a_level.get_ptr(); }

private:
    const GraphType & GA;
    bool itimes, debug, calculate_active;
    sVID iter;
    VID start, active;
    api::vertexprop<sVID,VID,var_current> a_level;
#if !LEVEL_ASYNC || DEFERRED_UPDATE
    api::vertexprop<sVID,VID,var_previous> a_prev_level;
#endif
    std::vector<info> info_buf;
};

#ifndef NOBENCH
template <class GraphType>
using Benchmark = BFSLVLv<GraphType>;

#include "driver.C"
#endif // NOBENCH
