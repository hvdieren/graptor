// extern char * gets(); // work-around clang++ 3.9.0

#include <math.h>
#include <fstream>
#include <limits>

#include "graptor/graptor.h"
#include "graptor/api.h"

// Select fastest configuration by default
#ifndef MEMO
#define MEMO 1
#endif

#ifndef DEFERRED_UPDATE
#define DEFERRED_UPDATE 1
#endif

#ifndef UNCOND_EXEC
#define UNCOND_EXEC 1
#endif

#ifdef LEVEL_ASYNC
#undef LEVEL_ASYNC
#endif
#define LEVEL_ASYNC 0

#ifdef CONVERGENCE
#undef CONVERGENCE
#endif
#define CONVERGENCE 0

using FloatTy = float; // double;

enum variable_name {
    var_pr = 0,
    var_contrib = 1,
    var_oldpr = 2,
    var_outdeg = 3,
    var_s = 4,
    var_delta = 5,
    var_degree = expr::aid_graph_degree
};

// Main edge-map operation for PageRank using power iteration method
struct PR_F
{
    // expr::array_ro/*update*/<FloatTy, VID, var_pr> pr;
    expr::array_ro/*update*/<double, VID, var_pr, array_encoding<FloatTy>> pr;
    expr::array_ro<FloatTy,VID, var_contrib> contrib;

    PR_F(FloatTy *_pr, FloatTy *_contrib) : pr(_pr), contrib(_contrib) {}

    // Do we need to calculate the new frontier, or is it all ones?
#if DEFERRED_UPDATE
    static constexpr frontier_mode new_frontier = fm_all_true;
#else
    static constexpr frontier_mode new_frontier = fm_reduction;
#endif
    static constexpr bool is_scan = false;
    static constexpr bool is_idempotent = false;
#if UNCOND_EXEC
    static constexpr bool may_omit_frontier_rd = true;
#else
    static constexpr bool may_omit_frontier_rd = false;
#endif
    static constexpr bool may_omit_frontier_wr = true;
    static constexpr bool new_frontier_dense = false;

    template<typename VIDSrc, typename VIDDst>
    auto relax( VIDSrc s, VIDDst d ) {
	// return pr[d] += contrib[s];
	return pr[d] += expr::cast<double>( contrib[s] );
    }

    template<typename VIDDst>
    auto active( VIDDst d ) {
	return expr::true_val(d);
    }

    template<typename VIDDst>
    auto different( VIDDst d ) {
	return expr::true_val(d);
    }

    template<typename VIDDst>
    auto update( VIDDst d ) {
	// Frontier always full; won't be used if new_frontier == fm_all_true
	return expr::true_val(d);
    }

    template<typename VIDDst>
    auto vertexop( VIDDst d ) {
	return expr::make_noop();
    }
};

// Inefficient edge-map operation for PageRank using power iteration method
struct PR_Base_F
{
    expr::array_ro/*update*/<FloatTy, VID, var_pr> pr;
    expr::array_ro<FloatTy,VID, var_oldpr> oldpr;
    expr::array_ro<FloatTy,VID, var_outdeg> outdeg;
    FloatTy dampen;

    PR_Base_F(FloatTy *_pr, FloatTy _d, FloatTy *_oldpr, FloatTy *_outdeg)
	: pr( _pr ), dampen( _d ), oldpr( _oldpr ), outdeg( _outdeg ) { }

    // Do we need to calculate the new frontier, or is it all ones?
#if DEFERRED_UPDATE
    static constexpr frontier_mode new_frontier = fm_all_true;
#else
    static constexpr frontier_mode new_frontier = fm_reduction;
#endif
    static constexpr bool is_scan = false;
    static constexpr bool is_idempotent = false;
#if UNCOND_EXEC
    static constexpr bool may_omit_frontier_rd = true;
#else
    static constexpr bool may_omit_frontier_rd = false;
#endif
    static constexpr bool may_omit_frontier_wr = true;
    static constexpr bool new_frontier_dense = false;

    template<typename VIDSrc, typename VIDDst>
    auto relax( VIDSrc s, VIDDst d ) {
	return pr[d] += ( expr::constant_val( oldpr[s], dampen ) * oldpr[s] ) / outdeg[s];
    }

    template<typename VIDDst>
    auto active( VIDDst d ) {
	return expr::true_val(d);
    }

    template<typename VIDDst>
    auto different( VIDDst d ) {
	return expr::true_val(d);
    }

    template<typename VIDDst>
    auto update( VIDDst d ) {
	// Frontier always full; won't be used if new_frontier == fm_all_true
	return expr::true_val(d);
    }

    template<typename VIDDst>
    auto vertexop( VIDDst d ) {
	return expr::make_noop();
    }
};

template<typename GraphType, typename floatty>
void readfile( const GraphType & GA, const char *fname, floatty *pr, VID n ) {
    std::ifstream ifs( fname, std::ifstream::in );
    using flim = std::numeric_limits<floatty>;
    ifs.precision( flim::max_digits10 ); // full precisison
    for( VID v=0; v < n; ++v ) {
	ifs >> pr[GA.remapID(v)];
    }
}

template<typename GraphType, typename floatty>
void writefile( const GraphType & GA, const char *fname, floatty *pr, VID n ) {
    std::ofstream ofs( fname, std::ofstream::out );
    using flim = std::numeric_limits<floatty>;
    ofs.precision( flim::max_digits10 ); // full precisison
    for( VID v=0; v < n; ++v ) {
	ofs << pr[GA.remapID(v)] << '\n';
    }
}

template <class GraphType>
class PRv {
public:
    PRv( GraphType & _GA, commandLine & P ) : GA( _GA ) {
	itimes = P.getOption( "-itimes" );
	debug = P.getOption( "-debug" );
	infile = P.getOptionValue( "-pr:infile" );
	outfile = P.getOptionValue( "-pr:outfile" );
	max_iter = P.getOptionLongValue( "-pr:maxiter", 100 );
	info_buf = itimes ? new info[max_iter] : nullptr;

	x_mem.allocate( numa_allocation_partitioned( GA.get_partitioner() ) );
	if( infile )
	    readfile( GA, infile, &x_mem[0], GA.numVertices() );

	static bool once = false;
	if( !once ) {
	    once = true;
	    std::cerr << "DEFERRED_UPDATE=" << DEFERRED_UPDATE << "\n";
	    std::cerr << "UNCOND_EXEC=" << UNCOND_EXEC << "\n";
	    std::cerr << "LEVEL_ASYNC=" << LEVEL_ASYNC << "\n";
	    std::cerr << "CONVERGENCE=" << CONVERGENCE << "\n";
	    std::cerr << "MEMO=" << MEMO << "\n";
	    std::cerr << "sizeof(FloatTy)=" << sizeof(FloatTy) << "\n";
	}
    }
    ~PRv() {
	x_mem.del();
	if( info_buf )
	    delete[] info_buf;
    }

    struct info {
	double delay;
	float density;
	FloatTy delta;

	void dump( int i ) {
	    std::cerr << "Iteration " << i << ": " << delay
		      << " density: " << density
		      << " delta: " << delta << "\n";
	}
    };

    struct stat { };

    void run() {
	const partitioner &part = GA.get_partitioner();
	VID n = GA.numVertices();
	EID m = GA.numEdges();

	// Accumulator of PR values: interface type of double such that register
	// caches are created at double precision, while the memory copy only
	// has FloatTy precision.
	// y and y_acc have the same AID, so they are known to alias
	mmap_ptr<FloatTy> y_mem;
	y_mem.allocate( numa_allocation_partitioned( part ) );
	expr::array_ro<FloatTy,VID,var_pr> y( y_mem.get() );
	expr::array_ro<double,VID,var_pr,array_encoding<FloatTy>> y_acc( y_mem.get() );

#if MEMO
	mmap_ptr<FloatTy> contrib_mem;
	contrib_mem.allocate( numa_allocation_partitioned( part ) );
	expr::array_ro<FloatTy,VID,var_contrib> contrib( contrib_mem.get() );
#else
	mmap_ptr<FloatTy> outdeg_mem;
	outdeg_mem.allocate( numa_allocation_partitioned( part ) );
	expr::array_ro<FloatTy,VID,var_outdeg> outdeg( outdeg_mem.get() );
#endif

	expr::array_ro<FloatTy,VID,var_oldpr> x( x_mem.get() );

	double s;
	expr::array_ro<double,VID,var_s> accum( &s );

	expr::array_ro<VID,VID,var_degree> degree(
	    const_cast<VID *>( GA.getCSR().getDegree() ) );

	double sdelta = 2;
	expr::array_ro<double,VID,var_delta> delta( &sdelta );

	// Configuration
	FloatTy dampen = 0.85;
	double tol = 1e-7;

	// Initialise arrays
	if( infile ) {
	    make_lazy_executor( part )
		.vertex_map( [&]( auto vid ) {
			return y[vid] = expr::zero_val( y[vid] ); } )
		.materialize();
	} else {
	    make_lazy_executor( part )
		.vertex_map( [&]( auto vid ) {
			return y[vid] = expr::zero_val( y[vid] ); } )
		.vertex_map( [&]( auto vid ) {
			return x[vid] = expr::constant_val( x[vid], FloatTy(1) )
			    / expr::constant_val( x[vid], FloatTy(n) );
		    } )
		.materialize();
	}

#if !MEMO
	map_vertexL( part, [&]( VID j ) { outdeg_mem[j] = GA.getOutDegree(j); } );
#else
	// TODO: create a vertex_map variation that skips over zero-out-degree
	//       vertices to save time. Do this through extending the
	//       partitioner class with info on zero-degree chunks (will be in!)
	make_lazy_executor( part )
	    .vertex_map( [&]( auto vid ) {
		    return contrib[vid] = expr::constant_val( x[vid], dampen )
			* x[vid] / expr::cast<FloatTy>( degree[vid] );
		} )
	    .materialize();
#endif


	// Control
	iter = 0;

#if UNCOND_EXEC
	frontier ftrue = frontier::all_true( n, m );
#else
	// frontier ftrue = frontier::dense( part, n );
	// map_vertexL( part, [&]( VID v ) { ftrue.getDenseB()[v] = true; } );
	// ftrue.setActiveCounts( n, m );
	using traits = gtraits<GraphType>;
	frontier ftrue = traits::template createValidFrontier<sizeof(VID)>( GA );
#endif

	timer tm_iter;
	tm_iter.start();

	while( iter < max_iter && sdelta > tol ) {
	    auto lazy = api::edgemap(
		GA,
		api::relax( [&]( auto s, auto d, auto e ) {
#if MEMO
			return y_acc[d] += expr::cast<double>( contrib[s] );
#else
			return y_acc[d] += expr::cast<double>(
			    ( expr::constant_val( x[s], dampen ) * x[s] )
			    / outdeg[s] );
#endif
		    } )
		);

	    s = 0;
	    lazy
/*
		.materialize();
	    std::cerr << "y:";
	    for( VID v=0; v < n; ++v ) {
		assert( y_mem[v] < FloatTy(1) );
		std::cerr << ' ' << y_mem[GA.remapID(v)];
	    }
	    std::cerr << "\n";
	    make_lazy_executor( part )
*/

		.vertex_scan( [&]( auto vid ) {
			return accum[expr::zero_val(vid)]
			    += expr::cast<double>( y[vid] );
		    } )
		.materialize();
	    assert( s > 0 );
	    double w = double(1) - s;
	    assert( w >= FloatTy(0) );
	    // output.del(); // TODO: this could also be enqueued on the lazy exec

	    s = 0;
	    make_lazy_executor( part )
		.vertex_map( [&]( auto vid ) {
			return y[vid] += expr::constant_val( y[vid], w/double(n) );
		    } )
		.vertex_scan( [&]( auto vid ) {
			return accum[expr::zero_val(vid)]
			    += expr::cast<double>( y[vid] );
		    } )
		.materialize();

	    w = FloatTy(1) / FloatTy(s);
	    sdelta = 0;

	    make_lazy_executor( part )
		.vertex_map( [&]( auto vid ) {
			return y[vid] *= expr::constant_val( y[vid], w );
		    } )
		.vertex_scan( [&]( auto vid ) {
			return delta[expr::zero_val( vid )]
			    += expr::cast<double>(
				expr::make_unop_abs( x[vid] - y[vid] ) );
		    } )
		// .materialize();
	    // make_lazy_executor( part )
		.vertex_map( [&]( auto vid ) { return x[vid] = y[vid]; } )
		.vertex_map( [&]( auto vid ) {
			return y[vid] = expr::zero_val( y[vid] ); } )
		// .materialize();
	    // make_lazy_executor( part )
#if MEMO
		.vertex_map( [&]( auto vid ) {
			return contrib[vid] = expr::constant_val( x[vid], dampen )
			    * x[vid] / expr::cast<FloatTy>(
				degree[vid] );
		    } )
#endif
		.materialize();

	    if( itimes ) {
		info_buf[iter].density = ftrue.density( GA.numEdges() );
		info_buf[iter].delta = sdelta;
		info_buf[iter].delay = tm_iter.next();
		if( debug )
		    info_buf[iter].dump( iter );
	    }
	    // std::cerr << "tm_e: " << tm_e.total() << "\n";

	    iter++;
	}

	ftrue.del();
	y_mem.del();
#if MEMO
	contrib_mem.del();
#else
	outdeg_mem.del();
#endif
    }

    void post_process( stat & stat_buf ) {
	if( iter == max_iter )
	    std::cerr << "WARNING: took all allowed iterations\n";

	if( itimes ) {
	    for( int i=0; i < iter; ++i )
		info_buf[i].dump( i );
	}

	if( outfile )
	    writefile( GA, outfile, &x_mem[0], GA.numVertices() );
    }

    static void report( const std::vector<stat> & stat_buf ) { }

    void validate( stat & stat_buf ) {
	for( VID i=0; i < 10; ++i )
	    std::cout << "PR[" << i << "]: " << x_mem[GA.remapID(i)] << "\n";
    }
    
private:
    const GraphType & GA;
    bool itimes, debug;
    int iter, max_iter;
    const char * infile, * outfile;
    mmap_ptr<FloatTy> x_mem;
    info * info_buf;
};

template <class GraphType>
using Benchmark = PRv<GraphType>;

#include "driver.C"
