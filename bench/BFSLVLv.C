#include "graptor/graptor.h"
#include "graptor/api.h"

// By default set options for highest performance
#ifndef DEFERRED_UPDATE
#define DEFERRED_UPDATE 1
#endif

#ifndef UNCOND_EXEC
#define UNCOND_EXEC 1
#endif

#ifndef CONVERGENCE
#define CONVERGENCE 0
#endif

#ifndef LEVEL_ASYNC
#define LEVEL_ASYNC 1
#endif

#ifdef MEMO
#undef MEMO
#endif
#define MEMO 0

// Main edge-map operation for BFS using the tropical semi-ring
struct BFS_Level_F
{
    expr::array_ro/*update*/<VID, VID, 0> level;
#if DEFERRED_UPDATE || !LEVEL_ASYNC
    expr::array_ro<VID, VID, 1> prev_level;
#endif
    VID infinity;

#if DEFERRED_UPDATE || !LEVEL_ASYNC
    BFS_Level_F(VID* _level, VID* _prev_level, VID _infinity)
	: level(_level), prev_level(_prev_level), infinity(_infinity) {}
#else
    BFS_Level_F(VID* _level, VID _infinity)
	: level(_level), infinity(_infinity) {}
#endif

    // Calculate the new frontier through reduction
    // (almost uniquely sparse CSR and dense CSC cases)
#if DEFERRED_UPDATE
    static constexpr frontier_mode new_frontier = fm_calculate;
#else
    static constexpr frontier_mode new_frontier = fm_reduction;
#endif
    static constexpr bool is_scan = false;
    static constexpr bool is_idempotent = true;
    static constexpr bool new_frontier_dense = false;

#if UNCOND_EXEC
    static constexpr bool may_omit_frontier_rd = true;
#else
    static constexpr bool may_omit_frontier_rd = false;
#endif
    static constexpr bool may_omit_frontier_wr = true;

    template<typename VIDSrc, typename VIDDst>
    auto relax( VIDSrc s, VIDDst d ) {
#if LEVEL_ASYNC
	return level[d].min( level[s]+expr::constant_val_one(s) );
#else
	return level[d].min( prev_level[s]+expr::constant_val_one(s) );
#endif
    }

    template<typename VIDDst>
    auto update( VIDDst d ) {
#if DEFERRED_UPDATE
	return level[d] != prev_level[d]; 
#else
	return expr::true_val(d);
#endif
    }

    template<typename VIDDst>
    auto different( VIDDst d ) {
	// TODO: level >= current level
	return expr::true_val(d);
    }

    template<typename VIDDst>
    auto active( VIDDst d ) {
#if CONVERGENCE
	// return level[d] == expr::constant_val(d, infinity); -- erroneous when LEVEL_ASYNC + UNCOND_EXEC
	return level[d] != expr::zero_val(d);
#else
	return expr::true_val(d);
#endif
    }

    template<typename VIDDst>
    auto vertexop( VIDDst d ) {
	// TODO: copy new level to prev_level !?
	return expr::make_noop();
    }
};

// Initialise level array
struct BFS_Init
{
    VID* level;
    VID val;
    BFS_Init( VID* _level, VID _val ) : level( _level ), val( _val ) { }
    inline void operator () ( VID i ) {
	level[i] = val; // ~VID(0);
    }
};

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
	level.del();
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

	level.allocate( numa_allocation_partitioned( part ) );
	expr::array_ro/*update*/<VID, VID, 0> a_level( level );

	// Assign initial labels
	// vertexMap( part, BFS_Init( level, n+1 ) );
	make_lazy_executor( part )
	    .vertex_map( [&]( auto v ) {
		 return a_level[v]
		     = expr::constant_val( a_level[v], n+1 ); } )
	    .materialize();
	level[start] = 0;

#if DEFERRED_UPDATE || !LEVEL_ASYNC
	mmap_ptr<VID> prev_level;
	prev_level.allocate( numa_allocation_partitioned( part ) );

	expr::array_ro<VID, VID, 1> a_prev_level( prev_level );

	// vertexMap( part, BFS_Init( prev_level, n+1 ) );
	make_lazy_executor( part )
	    .vertex_map( [&]( auto v ) {
		 return a_prev_level[v]
		     = expr::constant_val( a_prev_level[v], n+1 ); } )
	    .materialize();
	prev_level[start] = 0;
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
#if 0
#if DEFERRED_UPDATE || !LEVEL_ASYNC
	    frontier output;
	    vEdgeMap( GA, F,
		      output, BFS_Level_F( level, prev_level, n+1 ) )
		.materialize();
	    maintain_copies( part, /*output,*/ prev_level, level );
#else
	    frontier output;
	    // TODO: removeDups seems to be a big performance cost
	    vEdgeMap( GA, F, output, BFS_Level_F( level, n+1 ) )
	    .materialize();
#endif
#else

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
			     [&]( auto d ) {
				 return a_level[d] != a_prev_level[d]; 
			     }, api::strong ),
#else
		api::record( output, api::reduction, api::strong ),
#endif
		api::filter( filter_strength, api::src, F ),
#if CONVERGENCE
		api::filter( api::weak, api::dst,
			     [&]( auto d ) {
				 return a_level[d] != expr::zero_val(d); } ),
#endif
		api::relax( [&]( auto s, auto d, auto e ) {
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
#endif

#if 0
	    {
		output.toDense<logical<4>>(part);
		logical<4> *nf=output.template getDenseL<4>();
		std::cerr << "output:";
		for( VID v=0; v < n; ++v )
		    std::cerr << ' ' << nf[v];
		std::cerr << "\n";

		std::cerr << "defer match:";
		for( VID v=0; v < n; ++v )
		    if( ( level[v] != prev_level[v] ) == ( nf[v] != 0 ) )
			std::cerr << '.';
		    else
			std::cerr << 'X';
		std::cerr << "\n";

		std::cerr << "pvlvl:";
		for( VID v=0; v < n; ++v )
		    std::cerr << ' ' << prev_level[v];
		std::cerr << "\n";

		std::cerr << "level:";
		for( VID v=0; v < n; ++v )
		    std::cerr << ' ' << level[v];
		std::cerr << "\n";
	    }
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
			if( level[v] == ~VID(0) )
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
#if DEFERRED_UPDATE || !LEVEL_ASYNC
	prev_level.del();
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
	VID n = GA.numVertices();
	// VID longest = iter - 1;
	VID longest = 0;
	for( VID v=0; v < n; ++v )
	    if( longest < level[v] && level[v] < n ) 
		longest = level[v];

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
    mmap_ptr<VID> level;
    std::vector<info> info_buf;
};

template <class GraphType>
using Benchmark = BFSv<GraphType>;

#include "driver.C"
