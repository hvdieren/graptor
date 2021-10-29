#include <math.h>
#include <fstream>
#include <limits>

#include "graptor/graptor.h"
#include "graptor/api.h"
#include "graptor/dsl/ast/constant.h"

using expr::_0;

#ifndef DEFERRED_UPDATE
#define DEFERRED_UPDATE 1
#endif

#ifndef MEMO
#define MEMO 1
#endif

#ifndef UNCOND_EXEC
#define UNCOND_EXEC 1
#endif 
#ifndef UNCOND_EXEC_THRESHOLD
#define UNCOND_EXEC_THRESHOLD 0.6
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
    var_z = 2,
    var_outdeg = 3,
    var_s = 4,
    var_delta = 5,
    var_degree = expr::aid_graph_degree,
    var_frontier = 7,
    var_x = 8
};

double seqsum( double *a, VID n ) {
    double d = 0.;
    double err = 0.;
    for( VID i=0; i < n; ++i ) {
	// The code below achieves
	// d += a[i];
	// but does so with high accuracy
	double tmp = d;
	double y = fabs(a[i]) + err;
	d = tmp + y;
	err = tmp - d;
	err += y;
    }
    return d;
}

double seqsum( float *a, VID n ) {
// TODO: either use float inside, or do double w/o compensated sum
    double d = 0.;
    double err = 0.;
    for( VID i=0; i < n; ++i ) {
	// The code below achieves
	// d += a[i];
	// but does so with high accuracy
	double tmp = d;
	double y = double(fabs(a[i])) + err;
	d = tmp + y;
	err = tmp - d;
	err += y;
    }
    return d;
}


double sum( const partitioner &part, double *a, VID n ) {
    int p = part.get_num_partitions();
    double psum[p];

    map_partitionL( part, [&]( unsigned int k ) {
	    VID s = part.start_of(k);
	    VID e = part.start_of(k+1);
	    psum[k] = seqsum( &a[s], e-s );
	} );

    double d = 0.;
    for( int i=0; i < p; ++i )
	d += psum[i];

    return d;
}

float sum( const partitioner &part, float *a, int n ) {
    int p = part.get_num_partitions();
    double psum[p];

    map_partitionL( part, [&]( unsigned int k ) {
	    VID s = part.start_of(k);
	    VID e = part.start_of(k+1);
	    psum[k] = seqsum( &a[s], e-s );
	} );

    double d = 0.;
    for( int i=0; i < p; ++i )
	d += psum[i];

    return float(d);
}

double seqavg_progress( float *a, VID n, float r, const VID * degree ) {
    double d = 0.;
    for( VID i=0; i < n; ++i ) {
	if( degree[i] != 0 ) {
	    double ai = fabs( r * a[i] ) / double(degree[i]);;
	    d += ai;
	}
    }
    return d;
}

float avg_progress( const partitioner &part, float *a, int n, float d, const VID * degree ) {
    int p = part.get_num_partitions();
    float r = float(1) - d;
    double psum[p];

    map_partitionL( part, [&]( unsigned int k ) {
	    VID s = part.start_of(k);
	    VID e = part.start_of(k+1);
	    psum[k] = seqavg_progress( &a[s], e-s, r, &degree[s] );
	} );

    double ts = 0.;
    for( int i=0; i < p; ++i )
	ts += psum[i];

    return ts / double(n); // average value
}



double seqnormdiff( double *a, double *b, VID n ) {
    double d = 0.;
    double err = 0.;
    for( VID i=0; i < n; ++i ) {
	// The code below achieves
	// d += a[i];
	// but does so with high accuracy
	double tmp = d;
	double y = fabs( a[i] - b[i] ) + err;
	d = tmp + y;
	err = tmp - d;
	err += y;
    }
    return d;
}

double seqnormdiff( float *a, float *b, VID n ) {
    double d = 0.;
    double err = 0.;
    for( VID i=0; i < n; ++i ) {
	// The code below achieves
	// d += a[i];
	// but does so with high accuracy
	double tmp = d;
	double y = fabs( double(a[i]) - double(b[i]) ) + err;
	d = tmp + y;
	err = tmp - d;
	err += y;
    }
    return d;
}


double normdiff( const partitioner &part, double *a, double *b, VID n ) {
    int p = part.get_num_partitions();
    double psum[p];

    map_partitionL( part, [&]( unsigned int k ) {
	    VID s = part.start_of(k);
	    VID e = part.start_of(k+1);
	    psum[k] = seqnormdiff( &a[s], &b[s], e-s );
	} );

    double d = 0.;
    for( int i=0; i < p; ++i )
	d += psum[i];

    return d;
}

float normdiff( const partitioner &part, float *a, float *b, int n ) {
    int p = part.get_num_partitions();
    double psum[p];

    map_partitionL( part, [&]( unsigned int k ) {
	    VID s = part.start_of(k);
	    VID e = part.start_of(k+1);
	    psum[k] = seqnormdiff( &a[s], &b[s], e-s );
	} );

    double d = 0.;
    for( int i=0; i < p; ++i )
	d += psum[i];

    return float(d);
}


// VertexMap functor: rescale PageRank vector
struct PR_Rescale_F
{
    FloatTy w;
    FloatTy *x;
    
    PR_Rescale_F( FloatTy _w, FloatTy *_x ) : w( _w ), x( _x ) { }
    
    inline void operator () ( VID v ) { x[v] *= w; }
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

// Only for GraphVEBOPartCCSR
#define AVOID_ERROR 1

template <class GraphType>
class APRv {
public:
    APRv( GraphType & _GA, commandLine & P ) : GA( _GA ) {
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
	    std::cerr << "UNCOND_EXEC_THRESHOLD=" << UNCOND_EXEC_THRESHOLD << "\n";
	    std::cerr << "LEVEL_ASYNC=" << LEVEL_ASYNC << "\n";
	    std::cerr << "CONVERGENCE=" << CONVERGENCE << "\n";
	    std::cerr << "MEMO=" << MEMO << "\n";
	    std::cerr << "sizeof(FloatTy)=" << sizeof(FloatTy) << "\n";
	}
    }
    ~APRv() {
	x_mem.del();
	if( info_buf )
	    delete[] info_buf;
    }

    struct info {
	double delay;
	float density;
	FloatTy error;

	void dump( int i ) {
	    std::cerr << "Iteration " << i << ": " << delay
		      << " density: " << density
		      << " error: " << error << "\n";
	}
    };

    struct stat { };

    void run() {
	const partitioner &part = GA.get_partitioner();
	VID n = GA.numVertices();
	EID m = GA.numEdges();

	mmap_ptr<FloatTy> delta_mem;
	delta_mem.allocate( numa_allocation_partitioned( part ) );
	expr::array_ro<FloatTy,VID,var_delta> delta( delta_mem.get() );

	mmap_ptr<FloatTy> z_mem;
	z_mem.allocate( numa_allocation_partitioned( part ) );
	expr::array_ro<FloatTy,VID,var_z> z( z_mem.get() );

	expr::array_ro<FloatTy,VID,var_x> x( x_mem.get() );

#if MEMO
	mmap_ptr<FloatTy> contrib_mem;
	contrib_mem.allocate( numa_allocation_partitioned( part ) );
	expr::array_ro<FloatTy,VID,var_contrib> contrib( contrib_mem.get() );
#else
#endif
	mmap_ptr<FloatTy> outdeg_mem;
	outdeg_mem.allocate( numa_allocation_partitioned( part ) );
	expr::array_ro<FloatTy,VID,var_outdeg> outdeg( outdeg_mem.get() );

	expr::array_ro<VID,VID,var_degree> degree(
	    const_cast<VID *>( GA.getCSR().getDegree() ) );

	// Configuration
	FloatTy dampen = 0.85;
	const FloatTy eps  = 1e-7;
	const FloatTy eps2 = 1e-2;

	FloatTy ninv = FloatTy(1)/FloatTy(n);

	// Initialise arrays
	if( infile )
	    map_vertexL( part, [&]( VID v ){ delta_mem[v] = ninv; } );
	else
	    map_vertexL( part, [&]( VID v ){ x_mem[v] = 0; delta_mem[v] = ninv; } );

#if !MEMO
#endif
	map_vertexL( part, [&]( VID j ){ outdeg_mem[j] = GA.getOutDegree(j); } );

	// Control
	iter = 0;

#if DEFERRED_UPDATE
	frontier ftrue = frontier::all_true( n, m );
	frontier F = frontier::all_true( n, m );
#else
	frontier ftrue = frontier::dense( part, n );
	frontier F = frontier::dense( part, n );
	map_vertexL( part, [&]( VID v ) {
		ftrue.getDenseB()[v] = F.getDenseB()[v] = true;
	    } );
	ftrue.setActiveCounts( n, m );
	F.setActiveCounts( n, m );
#endif

	timer tm_iter;
	tm_iter.start();

	while( iter < max_iter ) {
	    // frontier output; // unused, initialized to all_true
	    frontier active = frontier::dense<sizeof(VID)>( part );
	    using FrTy = typename add_logical<VID>::type;
	    expr::array_ro<FrTy,VID,var_frontier> active_arr( active.getDense<FrTy>() );
	    FloatTy cst = (FloatTy(1)-dampen) * ninv;
	    if( iter == 0 ) {
		make_lazy_executor( part )
		    .vertex_map( [&]( auto v ) { return z[v] = _0; } )
#if MEMO
		    // Don't specify frontier, as all vertices initially active
		    .vertex_map( [&]( auto vid ) {
			    return contrib[vid] = /* expr::constant_val( delta[vid], dampen )
						   * */ delta[vid] / outdeg[vid];
				// / expr::make_unop_cvt_type<FloatTy>( degree[vid] );
			} )
#endif
		    .materialize();

	    api::edgemap(
		GA,
		api::relax( [&]( auto s, auto d, auto e ) {
#if MEMO
			return z[d] += contrib[s];
#else
			return z[d] += delta[s] / outdeg[s];
#endif
		    } )
		)
#if !MEMO || AVOID_ERROR
		    // Need to break here because next steps overwrite delta,
		    // which would result in data races - xAW ineffective?
		    .materialize();
		make_lazy_executor( part )
#endif
		    // TODO: Could do this conditionally on output frontier?
		    //       Need to check the math...
		    .vertex_map( [&]( auto v ) {
			    // Corrected as per Ligra 4/10/17
			    return delta[v] = expr::constant_val( z[v], dampen )
				* z[v]
				+ expr::constant_val( z[v], cst );
			} )
		    .vertex_map( [&]( auto v ) { return x[v] += delta[v]; } )
		    .vertex_map( [&]( auto v ) {
			    return delta[v] = delta[v]
				- expr::constant_val( delta[v], ninv ); } )
		    .vertex_map( [&]( auto v ) {
			    return active_arr[v] =
				expr::make_unop_cvt_to_vector<FrTy>(
				    expr::make_unop_abs(delta[v])
				    > expr::constant_val( x[v], eps2 ) * x[v] );
			} )
		    .materialize();
		// TODO: fuse with above... (scan)
		active.calculateActiveCounts( GA );
#if UNCOND_EXEC
	    } else if( F.density( m ) > UNCOND_EXEC_THRESHOLD ) { // use ftrue, not F
		make_lazy_executor( part )
		    .vertex_map( [&]( auto v ) { return z[v] = _0; } )
#if MEMO
		    .vertex_map( [&]( auto vid ) {
			    return contrib[vid] = /* expr::constant_val( delta[vid], dampen )
						   * */ delta[vid] / outdeg[vid];
				// / expr::make_unop_cvt_type<FloatTy>( degree[vid] );
			} )
#endif
		    .materialize();

		api::edgemap(
		    GA,
		    api::relax( [&]( auto s, auto d, auto e ) {
#if MEMO
			    return z[d] += contrib[s];
#else
			    return z[d] += delta[s] / outdeg[s];
#endif
			} )
		    )
#if !MEMO || AVOID_ERROR
		    // Need to break here because next steps overwrite delta,
		    // which would result in data races - xAW ineffective?
		    .materialize();
		make_lazy_executor( part )
#endif
		    .vertex_map( [&]( auto v ) {
			    return delta[v] =
				z[v] * expr::constant_val( z[v], dampen ); } )
		    .vertex_map( [&]( auto v ) {
			    return active_arr[v] =
				expr::make_unop_cvt_to_vector<FrTy>(
				    expr::make_unop_abs(delta[v])
				    > expr::constant_val( x[v], eps2 ) * x[v] ); } )
		    .vertex_map( [&]( auto v ) {
			    return x[v] +=
				expr::add_predicate( delta[v],
						     active_arr[v] != _0 );
		    } )
		    .materialize();
		// TODO: fuse with above... (scan)
		active.calculateActiveCounts( GA );
#endif
	    } else { // ! UNCOND_EXEC
		// assert( 0 && "ERROR in vmap with frontier - not executed correctly" );
		make_lazy_executor( part )
		    .vertex_map( [&]( auto v ) { return z[v] = _0; } )
#if MEMO
		    .vertex_map( F, [&]( auto vid ) {
			    return contrib[vid] = /* expr::constant_val( delta[vid], dampen )
						   * */ delta[vid] / outdeg[vid];
				// / expr::cast<FloatTy>( degree[vid] );
			} )
#endif
		    .materialize();

		api::edgemap(
		    GA,
		    api::filter( F, api::strong, api::src ),
		    api::relax( [&]( auto s, auto d, auto e ) {
#if MEMO
			    return z[d] += contrib[s];
#else
			    return z[d] += delta[s] / outdeg[s];
#endif
			} )
		    )
#if !MEMO || 1
		    // Need to break here because next steps overwrite delta,
		    // which would result in data races - xAW ineffective?
		    .materialize();
		make_lazy_executor( part )
#endif
		    .vertex_map( [&]( auto v ) {
			    return delta[v] =
				z[v] * expr::constant_val( z[v], dampen ); } )
		    .vertex_map( [&]( auto v ) {
			    return active_arr[v] =
				expr::make_unop_cvt_to_vector<FrTy>(
				    expr::make_unop_abs(delta[v])
				    > expr::constant_val( x[v], eps2 ) * x[v] ); } )
		    .vertex_map( [&]( auto v ) {
			    return x[v] +=
				expr::add_predicate( delta[v],
						     active_arr[v] != _0 );
		    } )
		    .materialize();
		// TODO: fuse with above... (scan)
		active.calculateActiveCounts( GA );
	    }

	    FloatTy L1norm = sum( part, delta_mem, n );

	    if( itimes ) {
		info_buf[iter].density = F.density( GA.numEdges() );
		info_buf[iter].error = L1norm;
		info_buf[iter].delay = tm_iter.next();
		if( debug )
		    info_buf[iter].dump( iter );
	    }
	    iter++;

	    F.del();
	    F = active;

	    if( L1norm < eps )
		break;
	}

	FloatTy s = FloatTy(1) / sum( part, x_mem, n );
	vertexMap( part, ftrue, PR_Rescale_F( s, x_mem ) );

	F.del();
	ftrue.del();
	delta_mem.del();
	z_mem.del();
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
    info * info_buf;
    mmap_ptr<FloatTy> x_mem;
};

template <class GraphType>
using Benchmark = APRv<GraphType>;

#include "driver.C"
