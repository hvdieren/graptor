#include "graptor/graptor.h"
#include "graptor/api.h"

// By default set options for highest performance
#ifndef DEFERRED_UPDATE
#define DEFERRED_UPDATE 1
#endif

#ifndef CONVERGENCE
#define CONVERGENCE 1
#endif

#ifndef UNCOND_EXEC
#define UNCOND_EXEC 1
#endif

#ifdef LEVEL_ASYNC
#undef LEVEL_ASYNC
#endif
#define LEVEL_ASYNC 0

#ifdef MEMO
#undef MEMO
#endif
#define MEMO 0

template <class GraphType>
class BFSv {
public:
    BFSv( GraphType & _GA, commandLine & P ) : GA( _GA ), info_buf( 60 ) {
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
	}
    }
    ~BFSv() {
	parent.del();
    }

    struct info {
	double delay;
	float density;
	float active;

	void dump( int i ) {
	    std::cerr << "Iteration " << i << ": " << delay
		      << " density: " << density
		      << " active: " << active << "\n";
	}
    };

    struct stat { };

    void run() {
	const partitioner &part = GA.get_partitioner();
	VID n = GA.numVertices();
	EID m = GA.numEdges();

	parent.allocate( numa_allocation_partitioned( part ) );

	expr::array_ro/*update*/<VID, VID, 0> a_parent( parent );

	// Assign initial labels
	// vertexMap( part, BFS_Init( parent ) );
	make_lazy_executor( part )
	    .vertex_map( [&]( auto v ) {
			     return a_parent[v] = expr::allones_val( v ); } )
	    .materialize();
	
	parent[start] = start;

#if DEFERRED_UPDATE
	mmap_ptr<VID> prev_parent;
	prev_parent.allocate( numa_allocation_partitioned( part ) );

	expr::array_ro/*update*/<VID, VID, 1> a_prev_parent( prev_parent );

	// vertexMap( part, BFS_Init( prev_parent ) );
	make_lazy_executor( part )
	    .vertex_map( [&]( auto v ) {
			     return a_prev_parent[v] = expr::allones_val( v ); } )
	    .materialize();
	prev_parent[start] = start;
#endif

	// Create initial frontier
	frontier F = frontier::create( n, start, GA.getOutDegree(start) );

	iter = 0;
	active = 1;

#if BFS_DEBUG && 0
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
		api::record( output,
			     [&] ( auto d ) {
				 return a_parent[d] != a_prev_parent[d]; },
			     api::strong ),
#else
		api::record( output, api::reduction, api::strong ),
#endif
		api::filter( filter_strength, api::src, F ),
#if CONVERGENCE
		api::filter( api::weak, api::dst,
			     [&] ( auto d ) {
				 return a_parent[d] == expr::allones_val(d);
			     } ),
#endif
		api::relax( [&]( auto s, auto d, auto e ) {
#if UNCOND_EXEC
			// If we call relax while frontier says not included,
			// then we need to check here, but luckily we know the
			// frontier from parent[]
			return a_parent[d].setif(
			    expr::add_predicate( s,
						 a_parent[s] != expr::allones_val(d) )
			    );
#else
			return a_parent[d].setif( s );
#endif
		    } )
		)
		.materialize();
#if DEFERRED_UPDATE
	    maintain_copies( part, output, prev_parent, parent );
/*
	    make_lazy_executor( part )
		.vertex_map( output,
			     [&](auto vid) { return a_prev_parent[vid] = a_parent[vid]; } )
		.materialize();
*/
#endif


#if BFS_DEBUG && 0
	    // Correctness check
	    output.toDense<logical<8>>(part);
	    {
		logical<8> *nf=output.template getDenseL<8>();
		bool *af = all.getDenseB();
		for( VID v=0; v < n; ++v ) {
		    if( af[v] && nf[v] )
			std::cerr << v << " activated again\n";
		    if( nf[v] )
			af[v] = true;
		    if( af[v] && parent[v] == ~VID(0) )
			std::cerr << v << " active but no parent recorded\n";
		}
	    }
	    output.dump( std::cout );
#endif

	    active += output.nActiveVertices();

	    if( itimes ) {
		VID active = 0;
		if( calculate_active ) { // Warning: expensive, included in time
		    for( VID v=0; v < n; ++v )
			if( parent[v] == ~VID(0) )
			    active++;
		}
		info_buf.resize( iter+1 );
		info_buf[iter].density = F.density( GA.numEdges() );
		info_buf[iter].delay = tm_iter.next();
		info_buf[iter].active = float(active)/float(n);
		if( debug )
		    info_buf[iter].dump( iter );
	    }

	    // Cleanup old frontier
	    F.del();
	    F = output;

	    ++iter;
	}

	F.del();
#if DEFERRED_UPDATE
	prev_parent.del();
#endif
    }

    void post_process( stat & stat_buf ) {
	if( itimes ) {
	    for( int i=0; i < iter; ++i )
		info_buf[i].dump( i );
	}
    }

    static void report( const std::vector<stat> & stat_buf ) { }

    void validate( stat & stat_buf ) {
	VID longest = iter - 1;
	VID n = GA.numVertices();
	std::cout << "Longest path from " << start
		  << " (original: " << GA.originalID( start ) << ") : "
		  << longest << "\n";

	std::cout << "Number of activated vertices: " << active << "\n";
	std::cout << "Number of vertices: " << n << "\n";
	std::cout << "Every vertex activated at most once: "
		  << ( n >= active ? "PASS" : "FAIL" ) << "\n";

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
    mmap_ptr<VID> parent;
    std::vector<info> info_buf;
};

template <class GraphType>
using Benchmark = BFSv<GraphType>;

#include "driver.C"
