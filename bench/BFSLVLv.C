#include "graptor/graptor.h"
#include "graptor/api.h"

using expr::_1;

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

template <class GraphType>
class BFSv {
public:
    BFSv( GraphType & _GA, commandLine & P )
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
    ~BFSv() {
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

	void dump( int i ) {
	    std::cerr << "Iteration " << i << ": " << delay
		      << " density: " << density
		      << " nacte: " << nacte
		      << " nactv: " << nactv
		      << " active: " << active << "\n";
	}
    };

    struct stat { };

    void run() {
	const partitioner &part = GA.get_partitioner();
	VID n = GA.numVertices();
	EID m = GA.numEdges();

	// Assign initial labels
	make_lazy_executor( part )
	    .vertex_map( [&]( auto v ) {
		return a_level[v]
		    = expr::constant_val( a_level[v], n+1 ); } )
	    .materialize();
	a_level.get_ptr()[start] = 0;

#if DEFERRED_UPDATE || !LEVEL_ASYNC
	mmap_ptr<VID> prev_level;
	prev_level.allocate( numa_allocation_partitioned( part ) );

	expr::array_ro<VID, VID, 1> a_prev_level( prev_level );

	make_lazy_executor( part )
	    .vertex_map( [&]( auto v ) {
		return a_prev_level[v]
		    = expr::constant_val( a_prev_level[v], n+1 ); } )
	    .materialize();
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
#endif
		api::filter( filter_strength, api::src, F ),
		api::fusion( [&]( auto v ) {
		    return expr::true_val( v );
		} ),
#if CONVERGENCE
		api::filter( api::weak, api::dst,
			     [&]( auto d ) {
				 auto cIt = expr::constant_val( d, iter+1 );
				 return a_level[d] > cIt;
			     } ),
#endif
#if FUSION
		api::fusion( [&]( auto v ) {
		    // return expr::true_val( v );
		    auto cTh = expr::constant_val( v, 3*iter+3 );
		    return a_level[v] <= cTh;
		} ),
#endif
		api::relax( [&]( auto s, auto d, auto e ) {
		    // TODO: alternative formulation
		    // Will avoid an add in critical path
		    // (no fusion only)
		    // doesn't help performance due to broadcast...
		    // The check a_level[s] == cur_level amounts to
		    // strong filtering on api::src.
		    /*
		    auto cur_level = expr::constant_val( a_level[s], iter );
		    auto high_level = expr::constant_val( a_level[s], iter+1 );
		    return expr::let<3>(
			a_level[d],
			[&]( auto old ) {
			    return a_level[d] = expr::add_predicate(
				high_level,
				a_level[s] == cur_level && old > high_level );
			} );
		    */
#if LEVEL_ASYNC
		    return a_level[d].min( a_level[s]+expr::constant_val_one(s) );
#else
		    return a_level[d].min( a_prev_level[s]+expr::constant_val_one(s) );
#endif
		} )
		)
		.materialize();
#if DEFERRED_UPDATE || !LEVEL_ASYNC
	    maintain_copies( part, /*output,*/ prev_level, level );
#endif

#if BFS_DEBUG
	    // Correctness check
	    // output.toDense<logical<4>>(part);
	    output.template toDense<frontier_type::ft_bool>(part);
	    std::cout << "output: ";
	    output.dump( std::cout );
	    std::cout << "all   : ";
	    all.dump( std::cout );
	    {
		bool *nf=output.template getDense<frontier_type::ft_bool>();
		bool *af = all.getDenseB();
		for( VID v=0; v < n; ++v ) {
		    if( af[v] && nf[v] )
			std::cerr << v << " activated again\n";
		    if( nf[v] )
			af[v] = true;
		    if( af[v] && level[v] == n+1 )
			std::cerr << v << " active but no level recorded\n";
		}
	    }
#endif

	    active += output.nActiveVertices();

	    if( itimes ) {
		VID active = 0;
		if( calculate_active ) { // Warning: expensive, included in time
		    for( VID v=0; v < n; ++v )
			if( a_level[v] == ~VID(0) )
			    active++;
		}
		info_buf.resize( iter+1 );
		info_buf[iter].density = F.density( GA.numEdges() );
		info_buf[iter].delay = tm_iter.next();
		info_buf[iter].active = float(active)/float(n);
		info_buf[iter].nacte = F.nActiveEdges();
		info_buf[iter].nactv = F.nActiveVertices();
		if( debug )
		    info_buf[iter].dump( iter );
	    }

	    // Cleanup old frontier
	    F.del();
	    F = output;

	    ++iter;
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
	for( VID v=0; v < n; ++v )
	    if( longest < a_level[v] && a_level[v] < n ) 
		longest = a_level[v];

	std::cout << "Longest path from " << start
		  << " (original: " << GA.originalID( start ) << ") : "
		  << longest << "\n";

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
    int iter;
    VID start, active;
    api::vertexprop<VID,VID,var_current> a_level;
#if !LEVEL_ASYNC || DEFERRED_UPDATE
    api::vertexprop<VID,VID,var_previous> a_prev_level;
#endif
    std::vector<info> info_buf;
};

template <class GraphType>
using Benchmark = BFSv<GraphType>;

#include "driver.C"
