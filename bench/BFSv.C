#include "graptor/graptor.h"
#include "graptor/api.h"

// By default set options for highest performance
#ifndef DEFERRED_UPDATE
#if __AVX512__
#define DEFERRED_UPDATE 1
#else
#define DEFERRED_UPDATE 0
#endif
#endif

#ifndef CONVERGENCE
#define CONVERGENCE 1
#endif

#ifndef UNCOND_EXEC
#define UNCOND_EXEC 0
#endif

#ifdef LEVEL_ASYNC
#undef LEVEL_ASYNC
#endif
#define LEVEL_ASYNC 0

#ifdef MEMO
#undef MEMO
#endif
#define MEMO 0

using expr::_0;
using expr::_1;
using expr::_1s;

template <class GraphType>
class BFSv {
public:
    BFSv( GraphType & _GA, commandLine & P )
	: GA( _GA ),
	  parent( GA.get_partitioner(), "parent" ),
#if DEFERRED_UPDATE
	  prev_parent( GA.get_partitioner(), "previous parent" ),
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
	}
    }
    ~BFSv() {
	parent.del();
#if DEFERRED_UPDATE
	prev_parent.del();
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
	    .vertex_map( [&]( auto v ) { return parent[v] = _1s; } )
#if DEFERRED_UPDATE
	    .vertex_map( [&]( auto v ) { return prev_parent[v] = _1s; } )
#endif
	    .materialize();
	
	parent.get_ptr()[start] = start;
#if DEFERRED_UPDATE
	prev_parent.get_ptr()[start] = start;
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
		api::record( output,
			     [&] ( auto d ) {
				 return parent[d] != prev_parent[d]; },
			     api::strong ),
#else
		api::record( output, api::reduction, api::strong ),
#endif
		api::filter( filter_strength, api::src, F ),
#if CONVERGENCE
		api::filter( api::weak, api::dst,
			     [&] ( auto d ) { return parent[d] == _1s; }
		    ),
#endif
		api::relax( [&]( auto s, auto d, auto e ) {
#if UNCOND_EXEC
			// If we call relax while frontier says not included,
			// then we need to check here, but luckily we know the
			// frontier from parent[]
			// Actually, this is WRONG as parent gets overwritten
		        // during the edgemap and does not accurately reflect
		        // the previous frontier.
			return parent[d].setif(
			    expr::add_predicate( s,
						 parent[s] != expr::allones_val(d) )
			    );
#else
			return parent[d].setif( s );
#endif
		    } )
		)
		.materialize();
#if DEFERRED_UPDATE
	    maintain_copies( part, output, prev_parent, parent );
#endif


#if BFS_DEBUG
	    // Correctness check
	    output.template toDense<frontier_type::ft_bool>(part);
	    {
		bool *nf=output.template getDense<frontier_type::ft_bool>();
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
	    // output.dump( std::cout );
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
	VID longest = iter - 1;
	VID n = GA.numVertices();
	std::cout << "Longest path from " << start
		  << " (original: " << GA.originalID( start ) << ") : "
		  << longest << "\n";

	VID longer = 0;
	map_vertexL( GA.get_partitioner(),
		     [&]( VID v ) {
			 VID p = v;
			 VID steps = 0;
			 while( ~parent[p] != 0 && parent[p] != p ) {
			     p = parent[p];
			     ++steps;
			 }
			 if( steps > longest )
			     __sync_fetch_and_add( &longer, 1 );
		     } );
	std::cout << "Number of vertices with longer chains: "
		  << longer
		  << ( longer == 0 ? " PASS" : " FAIL" )
		  << "\n";

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
    api::vertexprop<VID,VID,0> parent;
#if DEFERRED_UPDATE
    api::vertexprop<VID,VID,1> prev_parent;
#endif
    std::vector<info> info_buf;
};

#ifndef NOBENCH
template <class GraphType>
using Benchmark = BFSv<GraphType>;

#include "driver.C"
#endif // NOBENCH
