#include <math.h>
#include <fstream>
#include <limits>

#include "graptor/graptor.h"
#include "graptor/api.h"

// Select fastest configuration by default
#ifndef MEMO
#define MEMO 1
#endif

#ifdef DEFERRED_UPDATE
#undef DEFERRED_UPDATE
#endif
#define DEFERRED_UPDATE 0

#ifdef UNCOND_EXEC
#undef UNCOND_EXEC
#endif
#define UNCOND_EXEC 0

#ifdef LEVEL_ASYNC
#undef LEVEL_ASYNC
#endif
#define LEVEL_ASYNC 0

#ifdef CONVERGENCE
#undef CONVERGENCE
#endif
#define CONVERGENCE 0

#ifndef SWITCH_DYNAMIC
#define SWITCH_DYNAMIC 0
#endif

#ifndef VARIANT
#define VARIANT 4620
#endif

#if 0
/* 
 * PageRank analysis (FP32/FP64 only):
 * 1.   FP32/FP64 throughout (push/pull) (limits on convergence, perf diff)
 *      ContribTy = float/double
 *      AccumTy = float/double
 *      FloatTy = float/double
 */
#if VARIANT == 10
using ContribTy = float;
using AccumTy = float;
using FloatTy = float;
using ContribEnc = array_encoding<FloatTy>;
using AccumEnc = array_encoding<float>;
using FloatEnc = array_encoding<FloatTy>;
#endif
#if VARIANT == 11
using ContribTy = double;
using AccumTy = double;
using FloatTy = double;
using ContribEnc = array_encoding<FloatTy>;
using AccumEnc = array_encoding<double>;
using FloatEnc = array_encoding<FloatTy>;
#endif
/* 2.a. FP32 contribut with FP64 newpr (push/pull)
 *      so FP32 gather -> convert FP64 -> add up FP64
 *      ContribTy = double with array_encoding<float>
 *      AccumTy = double
 *      FloatTy = float
 */
#if VARIANT == 20
using ContribTy = double;
using AccumTy = double;
using FloatTy = float;
using ContribEnc = array_encoding<FloatTy>;
using AccumEnc = array_encoding<double>;
using FloatEnc = array_encoding<FloatTy>;
#endif
/* 2.b. where a does normal convert for FP32->FP64,
 *      b uses wide gather + integer-based convert
 *      ContribTy = double with array_encoding_wide<float>
 *      AccumTy = double
 *      FloatTy = double
 */
#if VARIANT == 21
using ContribTy = double;
using AccumTy = double;
using FloatTy = float;
using ContribEnc = array_encoding_wide<FloatTy>;
using AccumEnc = array_encoding<double>;
using FloatEnc = array_encoding<FloatTy>;
#endif
/* 3.a. FP32 with FP64 accumulator, converting back to FP32 (pull only)
 *      with widening convert cvtps_pd on contrib and newpr
 *      ContribTy = double with array_encoding<float>
 *      AccumTy = double with array_encoding<float>
 *      FloatTy = float
 */
#if VARIANT == 30
using ContribTy = float;
using AccumTy = double;
using FloatTy = float;
using ContribEnc = array_encoding<FloatTy>;
using AccumEnc = array_encoding<FloatTy>;
using FloatEnc = array_encoding<FloatTy>;
#endif
/* 3.b. 
 *      newpr: wide gather + integer-based convert
 *      contrib: wide gather + integer-based convert
 *      ContribTy = double with array_encoding_wide<float>
 *      AccumTy = double with array_encoding_wide<float>
 *      FloatTy = float
 */
#if VARIANT == 31
using ContribTy = float;
using AccumTy = double;
using FloatTy = float;
using ContribEnc = array_encoding_wide<FloatTy>;
using AccumEnc = array_encoding_wide<FloatTy>;
using FloatEnc = array_encoding<FloatTy>;
#endif
/* 3.e. data layout modification and widening load/store/gather
 *      ContribTy = double, permuted
 *      AccumTy = double with array_encoding_permute<float>
 *      FloatTy = float, no permute
 */
#if VARIANT == 34
using ContribTy = double; // as we need to convert for addition
using AccumTy = double;
using FloatTy = float;
using ContribEnc = array_encoding_wide<FloatTy>;
using AccumEnc = array_encoding_permute<FloatTy,MAX_VL>;
using FloatEnc = array_encoding<FloatTy>;
#endif
/* 4.a. as in 3.a., but contributions are compact 16 bit
 */
#if VARIANT == 40
using ContribTy = double;
using AccumTy = double;
using FloatTy = float;
using ContribEnc = array_encoding<customfp<6,10>>;
using AccumEnc = array_encoding<FloatTy>;
using FloatEnc = array_encoding<FloatTy>;
#endif
/* 4.a. as in 3.a., but contributions are compact 16 bit
 */
#if VARIANT == 41
using ContribTy = double;
using AccumTy = double;
using FloatTy = float;
using ContribEnc = array_encoding_wide<customfp<6,10>>;
using AccumEnc = array_encoding_wide<FloatTy>;
using FloatEnc = array_encoding<FloatTy>;
#endif
#endif // 0


enum variable_name {
    var_pr = 0,
    var_contrib = 1,
    var_oldpr = 2,
    var_outdeg = 3,
    var_s = 4,
    var_delta = 5,
    var_degree = expr::aid_graph_degree,
    var_extpr = 6,
};

template<typename GraphType, typename floatty>
void readfile( const GraphType & GA, const char *fname, floatty *pr, VID n ) {
    std::ifstream ifs( fname, std::ifstream::in );
    using flim = std::numeric_limits<floatty>;
    ifs.precision( flim::max_digits10 ); // full precision
    for( VID v=0; v < n; ++v ) {
	ifs >> pr[GA.remapID(v)];
    }
}

template<typename GraphType, typename floatty>
void writefile( const GraphType & GA, const char *fname, floatty *pr, VID n ) {
    std::ofstream ofs( fname, std::ofstream::out );
    using flim = std::numeric_limits<floatty>;
    ofs.precision( flim::max_digits10 ); // full precision
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

	xd_mem.allocate( numa_allocation_partitioned( GA.get_partitioner() ) );
	if( infile )
	    readfile( GA, infile, &xd_mem[0], GA.numVertices() );

	static bool once = false;
	if( !once ) {
	    once = true;
	    std::cerr << "DEFERRED_UPDATE=" << DEFERRED_UPDATE << "\n";
	    std::cerr << "UNCOND_EXEC=" << UNCOND_EXEC << "\n";
	    std::cerr << "LEVEL_ASYNC=" << LEVEL_ASYNC << "\n";
	    std::cerr << "CONVERGENCE=" << CONVERGENCE << "\n";
	    std::cerr << "MEMO=" << MEMO << "\n";
	    std::cerr << "WIDEN_APPROACH=" << WIDEN_APPROACH << "\n";
	    std::cerr << "VARIANT=" << VARIANT << "\n";
	    std::cerr << "SWITCH_DYNAMIC=" << SWITCH_DYNAMIC << "\n";
	}
    }
    ~PRv() {
	xd_mem.del();
	if( info_buf )
	    delete[] info_buf;
    }

    struct info {
	double delay;
	float density;
	double delta;
	const char * variant;

	void dump( int i ) {
	    std::cerr << "Iteration " << i << ": " << delay
		      << " density: " << density
		      << " delta: " << delta
		      << ' ' << variant
		      << "\n";
	}
    };

    struct stat {
	double delay;
	int iter;
    };

    template<typename ContribTy, typename AccumTy, typename FloatTy,
	     typename ContribEnc, typename AccumEnc, typename FloatEnc,
	     typename InitType>
    mmap_ptr<FloatTy>
    run_prec( mmap_ptr<InitType> x_init,
	      int & iter,
	      double tol, const char * variant ) {
	timer tm_iter;
	tm_iter.start();

	const partitioner &part = GA.get_partitioner();
	VID n = GA.numVertices();
	EID m = GA.numEdges();

	mmap_ptr<FloatTy> x_mem;
	x_mem.allocate( numa_allocation_partitioned( part ) );

	// Accumulator of PR values: interface type of double such that register
	// caches are created at double precision, while the memory copy only
	// has FloatTy precision.
	// y and y_acc have the same AID, so they are known to alias
	mmap_ptr<typename AccumEnc::stored_type> y_mem;
	y_mem.allocate( numa_allocation_partitioned( part ) );
	expr::array_ro<FloatTy,VID,var_pr,AccumEnc> y( y_mem.get() );
	expr::array_ro<AccumTy,VID,var_pr,AccumEnc> y_acc( y_mem.get() );

#if MEMO
	mmap_ptr<typename ContribEnc::stored_type> contrib_mem;
	contrib_mem.allocate( numa_allocation_partitioned( part ) );
	expr::array_ro<FloatTy,VID,var_contrib,ContribEnc>
	    contrib_seq( contrib_mem.get() );
	expr::array_ro<ContribTy,VID,var_contrib,ContribEnc>
	    contrib_rnd( contrib_mem.get() );

#else // MEMO
	// Note: degree stored as floating-point value
	mmap_ptr<typename ContribEnc::stored_type> outdeg_mem;
	outdeg_mem.allocate( numa_allocation_partitioned( part ) );
	expr::array_ro<ContribTy,VID,var_outdeg,ContribEnc> outdeg( outdeg_mem.get() );
#endif // MEMO

	expr::array_ro<FloatTy,VID,var_oldpr,FloatEnc> x( x_mem.get() );
	expr::array_ro<FloatTy,VID,var_extpr,array_encoding<InitType>> xi( x_init.get() );

	double s;
	expr::array_ro<double,VID,var_s> accum( &s );

	expr::array_ro<VID,VID,var_degree> degree_rd(
	    const_cast<VID *>( GA.getCSR().getDegree() ) );

	double sdelta = 2;
	expr::array_ro<double,VID,var_delta> delta( &sdelta );

	// Configuration
	FloatTy dampen = 0.85;
	// double tol = 1e-7;

	// Initialise arrays
	make_lazy_executor( part )
	    .vertex_map( [&]( auto vid ) { return x[vid] = xi[vid]; } )
	    .vertex_map( [&]( auto vid ) {
		 return y[vid] = expr::zero_val( y[vid] ); } )
	    .materialize();

#if !MEMO
	map_vertexL( part, [&]( VID j ) { outdeg_mem[j] = GA.getOutDegree(j); } );
#else
	// TODO: create a vertex_map variation that skips over zero-out-degree
	//       vertices to save time. Do this through extending the
	//       partitioner class with info on zero-degree chunks (will be in!)
	make_lazy_executor( part )
	    .vertex_map( [&]( auto vid ) {
		    return contrib_seq[vid] =
			/*expr::cast<ContribTy>*/(
			    expr::constant_val( x[vid], dampen )
			    * x[vid] / expr::cast<FloatTy>( degree_rd[vid] ) );
		} )
	    .materialize();
#endif

	// Control
	// iter = 0;

	frontier ftrue = frontier::all_true( n, m );

#if SWITCH_DYNAMIC
	double delta_1 = sdelta;
	double delta_rate = 1, delta_rate_1 = 1;
	bool dyn_enabled = false;
#endif

	while( iter < max_iter && sdelta > tol
#if SWITCH_DYNAMIC
	       && ( !dyn_enabled || delta_rate / delta_rate_1 <= 1.01 )
#endif
	    ) {
	    // Power iteration step

	    auto lazy = api::edgemap(
		GA,
		api::relax( [&]( auto s, auto d, auto e ) {
#if MEMO
			return y_acc[d] += expr::cast<AccumTy>( contrib_rnd[s] );
#else
			return y_acc[d] += expr::cast<AccumTy>(
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
		        // TODO: should we read y as double (convert-load)?
			return accum[expr::zero_val(vid)]
			    += expr::cast<double>( y_acc[vid] );
		    } )
		.materialize();
	    assert( s > 0 );
	    double w = double(1) - s;
	    assert( w >= double(0) );

	    s = 0;
	    make_lazy_executor( part )
		.vertex_map( [&]( auto vid ) {
			return y[vid] += expr::constant_val( y[vid], w/double(n) );
		    } )
		.vertex_scan( [&]( auto vid ) {
		        // TODO: should we read y as double (convert-load)?
			return accum[expr::zero_val(vid)]
			    += expr::cast<double>( y[vid] );
		    } )
		.materialize();

	    w = double(1) / double(s);
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
		.vertex_map( [&]( auto vid ) { return x[vid] = y[vid]; } )
		.vertex_map( [&]( auto vid ) {
			return y[vid] = expr::zero_val( y[vid] ); } )
#if MEMO
		.vertex_map( [&]( auto vid ) {
			return contrib_seq[vid] =
			    /*expr::cast<ContribTy>*/(
				expr::constant_val( x[vid], dampen )
				* x[vid] / expr::cast<FloatTy>( degree_rd[vid] ) );
		    } )
#endif
		.materialize();

	    if( itimes ) {
		info_buf[iter].density = ftrue.density( GA.numEdges() );
		info_buf[iter].delta = sdelta;
		info_buf[iter].delay = tm_iter.next();
		info_buf[iter].variant = variant;
		if( debug )
		    info_buf[iter].dump( iter );
	    }
	    // std::cerr << "tm_e: " << tm_e.total() << "\n";

#if SWITCH_DYNAMIC
	    delta_rate_1 = delta_rate;
	    delta_rate = sdelta / delta_1;
	    delta_1 = sdelta;
	    dyn_enabled = dyn_enabled || delta_rate / delta_rate_1 <= 1.01;
#endif

	    iter++;
	}

	ftrue.del();
	y_mem.del();
#if MEMO
	contrib_mem.del();
#else
	outdeg_mem.del();
#endif

	return x_mem;
    }

    template<typename InitType>
    mmap_ptr<float>
    run_prec_40( mmap_ptr<InitType> x_init, int & iter, double tol ) {
	using ContribTy = double;
	using AccumTy = double;
	using FloatTy = float;
	using ContribEnc = array_encoding<customfp<6,10>>;
	using AccumEnc = array_encoding<FloatTy>;
	using FloatEnc = array_encoding<FloatTy>;

	return run_prec<ContribTy,AccumTy,FloatTy,
			ContribEnc,AccumEnc,FloatEnc>(
			    x_init, iter, tol, "40" );
    }

    template<typename InitType>
    mmap_ptr<float>
    run_prec_41( mmap_ptr<InitType> x_init, int & iter, double tol ) {
	using ContribTy = double;
	using AccumTy = double;
	using FloatTy = float;
	using ContribEnc = array_encoding_wide<customfp<6,10>>;
	using AccumEnc = array_encoding_wide<FloatTy>;
	using FloatEnc = array_encoding<FloatTy>;

	return run_prec<ContribTy,AccumTy,FloatTy,
			ContribEnc,AccumEnc,FloatEnc>(
			    x_init, iter, tol, "41" );
    }

    template<typename InitType>
    mmap_ptr<float>
    run_prec_42( mmap_ptr<InitType> x_init, int & iter, double tol ) {
	using ContribTy = float;
	using AccumTy = double;
	using FloatTy = float;
	using ContribEnc = array_encoding_wide<customfp<6,10>>;
	using AccumEnc = array_encoding_wide<FloatTy>;
	using FloatEnc = array_encoding<FloatTy>;

	return run_prec<ContribTy,AccumTy,FloatTy,
			ContribEnc,AccumEnc,FloatEnc>(
			    x_init, iter, tol, "42" );
    }

    template<typename InitType>
    mmap_ptr<float>
    run_prec_45( mmap_ptr<InitType> x_init, int & iter, double tol ) {
	using ContribTy = float;
	using AccumTy = float;
	using FloatTy = float;
	using ContribEnc = array_encoding<customfp<6,10>>;
	using AccumEnc = array_encoding<FloatTy>;
	using FloatEnc = array_encoding<FloatTy>;

	return run_prec<ContribTy,AccumTy,FloatTy,
			ContribEnc,AccumEnc,FloatEnc>(
			    x_init, iter, tol, "45" );
    }

    template<typename InitType>
    mmap_ptr<float>
    run_prec_46( mmap_ptr<InitType> x_init, int & iter, double tol ) {
	using ContribTy = float;
	using AccumTy = float;
	using FloatTy = float;
	using ContribEnc = array_encoding_wide<customfp<6,10>>;
	using AccumEnc = array_encoding/*_wide*/<FloatTy>;
	using FloatEnc = array_encoding<FloatTy>;

	return run_prec<ContribTy,AccumTy,FloatTy,
			ContribEnc,AccumEnc,FloatEnc>(
			    x_init, iter, tol, "46" );
    }

    template<typename InitType>
    mmap_ptr<float>
    run_prec_36( mmap_ptr<InitType> x_init, int & iter, double tol ) {
	using ContribTy = float;
	using AccumTy = float;
	using FloatTy = float;
	using ContribEnc = array_encoding<customfp<6,10>>;
	using AccumEnc = array_encoding/*_wide*/<FloatTy>;
	using FloatEnc = array_encoding<FloatTy>;

	return run_prec<ContribTy,AccumTy,FloatTy,
			ContribEnc,AccumEnc,FloatEnc>(
			    x_init, iter, tol, "36" );
    }

    template<typename InitType>
    mmap_ptr<float>
    run_prec_20( mmap_ptr<InitType> x_init, int & iter, double tol ) {
	using ContribTy = double;
	using AccumTy = double;
	using FloatTy = float;
	using ContribEnc = array_encoding<FloatTy>;
	using AccumEnc = array_encoding<double>;
	using FloatEnc = array_encoding<FloatTy>;

	return run_prec<ContribTy,AccumTy,FloatTy,
			ContribEnc,AccumEnc,FloatEnc>(
			    x_init, iter, tol, "20" );
    }

    template<typename InitType>
    mmap_ptr<float>
    run_prec_99( mmap_ptr<InitType> x_init, int & iter, double tol ) {
	using ContribTy = float;
	using AccumTy = float;
	using FloatTy = float;
	// The top half of a float only
	using ContribEnc = array_encoding_wide<scustomfp<true,8,7,false,0>>;
	using AccumEnc = array_encoding/*_wide*/<FloatTy>;
	using FloatEnc = array_encoding<FloatTy>;

	return run_prec<ContribTy,AccumTy,FloatTy,
			ContribEnc,AccumEnc,FloatEnc>(
			    x_init, iter, tol, "99" );
    }

    template<typename InitType>
    mmap_ptr<float>
    run_prec_98( mmap_ptr<InitType> x_init, int & iter, double tol ) {
	using ContribTy = double;
	using AccumTy = double;
	using FloatTy = float;
	// The top half of a float only
	using ContribEnc = array_encoding_wide<scustomfp<true,11,4,false,0>>;
	using AccumEnc = array_encoding/*_wide*/<FloatTy>;
	using FloatEnc = array_encoding<FloatTy>;

	return run_prec<ContribTy,AccumTy,FloatTy,
			ContribEnc,AccumEnc,FloatEnc>(
			    x_init, iter, tol, "98" );
    }

    template<typename InitType>
    mmap_ptr<float>
    run_prec_97( mmap_ptr<InitType> x_init, int & iter, double tol ) {
	using ContribTy = float;
	using AccumTy = float;
	using FloatTy = float;
	// The top half of a float only
	using ContribEnc = array_encoding_wide<scustomfp<false,8,8,false,0>>;
	using AccumEnc = array_encoding/*_wide*/<FloatTy>;
	using FloatEnc = array_encoding<FloatTy>;

	return run_prec<ContribTy,AccumTy,FloatTy,
			ContribEnc,AccumEnc,FloatEnc>(
			    x_init, iter, tol, "97" );
    }

    template<typename InitType>
    mmap_ptr<float>
    run_prec_96( mmap_ptr<InitType> x_init, int & iter, double tol ) {
	using ContribTy = double;
	using AccumTy = double;
	using FloatTy = float;
	// The top half of a float only
	using ContribEnc = array_encoding_wide<scustomfp<false,11,5,false,0>>;
	using AccumEnc = array_encoding/*_wide*/<FloatTy>;
	using FloatEnc = array_encoding<FloatTy>;

	return run_prec<ContribTy,AccumTy,FloatTy,
			ContribEnc,AccumEnc,FloatEnc>(
			    x_init, iter, tol, "96" );
    }

    template<typename InitType>
    mmap_ptr<float>
    run_prec_10( mmap_ptr<InitType> x_init, int & iter, double tol ) {
	using ContribTy = float;
	using AccumTy = float;
	using FloatTy = float;
	using ContribEnc = array_encoding<FloatTy>;
	using AccumEnc = array_encoding<float>;
	using FloatEnc = array_encoding<FloatTy>;

	return run_prec<ContribTy,AccumTy,FloatTy,
			ContribEnc,AccumEnc,FloatEnc>(
			    x_init, iter, tol, "10" );
    }



    template<typename InitType>
    mmap_ptr<double>
    run_prec_11( mmap_ptr<InitType> x_init, int & iter, double tol ) {
	using ContribTy = double;
	using AccumTy = double;
	using FloatTy = double;
	using ContribEnc = array_encoding<FloatTy>;
	using AccumEnc = array_encoding<double>;
	using FloatEnc = array_encoding<FloatTy>;

	return run_prec<ContribTy,AccumTy,FloatTy,
			ContribEnc,AccumEnc,FloatEnc>(
			    x_init, iter, tol, "11" );
    }



    void run() {
	const partitioner &part = GA.get_partitioner();
	VID n = GA.numVertices();
	EID m = GA.numEdges();

	expr::array_ro<double,VID,var_oldpr> x( xd_mem.get() );

	// Initialise arrays
	if( !infile ) {
	    make_lazy_executor( part )
		.vertex_map( [&]( auto vid ) {
				 return x[vid] = expr::constant_val(
				     x[vid], double(1)/double(n) );
		    } )
		.materialize();
	}

	// Control
	iter = 0;

	timer tm_iter;
	tm_iter.start();

#if VARIANT == 99
	auto x1_mem = run_prec_99( xd_mem, iter, 1e-7 );
#elif VARIANT == 98
	auto x1_mem = run_prec_98( xd_mem, iter, 1e-7 );
#elif VARIANT == 10
	auto x1_mem = run_prec_10( xd_mem, iter, 1e-7 );
#elif VARIANT == 11
	auto x1_mem = run_prec_11( xd_mem, iter, 1e-7 );
#elif VARIANT == 20
	auto x1_mem = run_prec_20( xd_mem, iter, 1e-7 );
#elif VARIANT == 21
	auto x1_mem = run_prec_21( xd_mem, iter, 1e-7 );
#elif VARIANT == 40
	auto x1_mem = run_prec_40( xd_mem, iter, 1e-7 );
#elif VARIANT == 41
	auto x1_mem = run_prec_41( xd_mem, iter, 1e-7 );
#elif VARIANT == 42
	auto x1_mem = run_prec_42( xd_mem, iter, 1e-7 );
#elif VARIANT == 45
	auto x1_mem = run_prec_45( xd_mem, iter, 1e-7 );
#elif VARIANT == 46
	auto x1_mem = run_prec_46( xd_mem, iter, 1e-7 );
#elif VARIANT == 4020
	auto x0_mem = run_prec_40( xd_mem, iter, 1e-3 );
	auto x1_mem = run_prec_20( x0_mem, iter, 1e-7 );
	x0_mem.del();
#elif VARIANT == 4120
	auto x0_mem = run_prec_41( xd_mem, iter, 1e-3 );
	auto x1_mem = run_prec_20( x0_mem, iter, 1e-7 );
	x0_mem.del();
#elif VARIANT == 4220
	auto x0_mem = run_prec_42( xd_mem, iter, 1e-3 );
	auto x1_mem = run_prec_20( x0_mem, iter, 1e-7 );
	x0_mem.del();
#elif VARIANT == 4520
	auto x0_mem = run_prec_45( xd_mem, iter, 1e-3 );
	auto x1_mem = run_prec_20( x0_mem, iter, 1e-7 );
	x0_mem.del();
#elif VARIANT == 4620
	auto x0_mem = run_prec_46( xd_mem, iter, 1e-3 );
	auto x1_mem = run_prec_20( x0_mem, iter, 1e-7 );
	x0_mem.del();
#elif VARIANT == 9620
	auto x0_mem = run_prec_96( xd_mem, iter, 1e-1 );
	auto x1_mem = run_prec_20( x0_mem, iter, 1e-7 );
	x0_mem.del();
#elif VARIANT == 9720
	auto x0_mem = run_prec_97( xd_mem, iter, 1e-2 );
	auto x1_mem = run_prec_20( x0_mem, iter, 1e-7 );
	x0_mem.del();
#elif VARIANT == 9820
	auto x0_mem = run_prec_98( xd_mem, iter, 1e-1 );
	auto x1_mem = run_prec_20( x0_mem, iter, 1e-7 );
	x0_mem.del();
#elif VARIANT == 9920
	auto x0_mem = run_prec_99( xd_mem, iter, 1e-2 );
	auto x1_mem = run_prec_20( x0_mem, iter, 1e-7 );
	x0_mem.del();
#elif VARIANT == 994620
	auto x0_mem = run_prec_99( xd_mem, iter, 1e-2 );
	auto x2_mem = run_prec_46( x0_mem, iter, 1e-3 );
	x0_mem.del();
	auto x1_mem = run_prec_20( x2_mem, iter, 1e-7 );
	x2_mem.del();
#elif VARIANT == 3620
	auto x0_mem = run_prec_36( xd_mem, iter, 1e-3 );
	auto x1_mem = run_prec_20( x0_mem, iter, 1e-7 );
	x0_mem.del();
#else
	assert( 0 && "misconfigured" );
#endif

	// Copy back to xd_mem
	using x1_ty = std::decay_t<decltype(x1_mem)>;
	using type = typename x1_ty::type;
	expr::array_ro<double,VID,var_extpr,array_encoding<type>>
	    x1a( x1_mem.get() );
	make_lazy_executor( part )
	    .vertex_map( [&]( auto vid ) { return x[vid] = x1a[vid]; } )
	    .materialize();

	x1_mem.del();
    }

    void post_process( stat & stat_buf ) {
	if( iter == max_iter )
	    std::cerr << "WARNING: took all allowed iterations\n";

	if( itimes ) {
	    double total = 0.0;
	    for( int i=0; i < iter; ++i ) {
		info_buf[i].dump( i );
		total += info_buf[i].delay;
	    }

	    stat_buf.delay = total;
	    stat_buf.iter = iter;
	}

	if( outfile )
	    writefile( GA, outfile, &xd_mem[0], GA.numVertices() );
    }

    static void report( const std::vector<stat> & stat_buf ) {
	size_t repeat = stat_buf.size();
	double total = 0.0;
	for( size_t i=0; i < repeat; ++i ) {
	    double avg = double(stat_buf[i].delay)/double(stat_buf[i].iter);
	    total += avg;
	    std::cerr << "round " << i << ": delay: " << stat_buf[i].delay
		      << " iterations: " << stat_buf[i].iter
		      << " average delay: " << avg << '\n';
	}
	std::cerr << "Average power iteration delay: "
		  << total/double(repeat) << '\n';
    }

    void validate( const stat & stat_buf ) {
	for( VID i=0; i < 10; ++i )
	    std::cout << "PR[" << i << "]: " << xd_mem[GA.remapID(i)] << "\n";
    }
    
private:
    const GraphType & GA;
    bool itimes, debug;
    int iter, max_iter;
    const char * infile, * outfile;
    mmap_ptr<double> xd_mem;
    info * info_buf;
};


template <class GraphType>
using Benchmark = PRv<GraphType>;

#include "driver.C"
