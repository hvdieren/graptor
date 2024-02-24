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

#ifdef UNCOND_EXEC
#undef UNCOND_EXEC
#endif
#define UNCOND_EXEC 0

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
	VID n = GA.numVertices();
	EID m = GA.numEdges();

	constexpr VID max_vertices = VID(1) << (8*sizeof(VID)-1);
	if( n > max_vertices ) {
	    std::cerr << "ERROR: maximum of " << max_vertices
		      << " vertices supported; " << n << " present\n";
	    return;
	}

	// Assign initial labels
	parent.fill( part, ~(VID)0 );
#if DEFERRED_UPDATE
	prev_parent.fill( part, ~(VID)0 );
#endif
	
	parent.set( start, start );
#if DEFERRED_UPDATE
	prev_parent.set( start, start );
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
		api::filter( api::strong, api::src, F ),
#if CONVERGENCE
		api::filter( api::weak, api::dst,
			     // [&] ( auto d ) { return !expr::msbset( parent[d] ); }
			     [&] ( auto d ) { return parent[d] != _1s; }
		    ),
#endif
		api::relax( [&]( auto s, auto d, auto e ) {
			return parent[d].setif( s );
		    } )
		)
		.materialize();
#if DEFERRED_UPDATE
	    maintain_copies( part, output, prev_parent, parent );
#endif

	    // print( std::cout, part, parent );

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
		info_buf.resize( iter+1 );
		info_buf[iter].density = F.density( GA.numEdges() );
		info_buf[iter].delay = tm_iter.next();
		info_buf[iter].active = float(n - active)/float(n);
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
