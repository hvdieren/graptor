#include <math.h>
#include <fstream>
#include <limits>

#include "graptor/graptor.h"
#include "graptor/api.h"

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

#ifndef SCALE_CONTRIB
#define SCALE_CONTRIB 1
#endif

#ifndef VARIANT
#define VARIANT 2046
#endif

enum variable_name {
    var_pr = 0,
    var_contrib = 1,
    var_z = 2,
    var_outdeg = 3,
    var_s = 4,
    var_delta = 5,
    var_degree = expr::aid_graph_degree,
    var_frontier = 7,
    var_x = 8,
    var_extpr = 9,
    var_extdl = 10
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

template<unsigned short base_AID, typename ftype>
void analyse_range( const partitioner & part, const ftype * vals ) {
    constexpr unsigned short var_data = base_AID + 0;
    constexpr unsigned short var_min = base_AID + 1;
    constexpr unsigned short var_max = base_AID + 2;
    constexpr unsigned short var_avg = base_AID + 3;
    constexpr unsigned short var_sq  = base_AID + 4;
    constexpr unsigned short var_neg = base_AID + 5;
    constexpr unsigned short var_zero = base_AID + 6;
    constexpr unsigned short var_minz = base_AID + 7;
    constexpr unsigned short var_lg1 = base_AID + 8;
    
    ftype f_min = std::numeric_limits<ftype>::max();
    ftype f_minz = std::numeric_limits<ftype>::max();
    ftype f_max = std::numeric_limits<ftype>::min();
    ftype f_avg = 0;
    ftype f_sq = 0;
    VID   f_neg = 0;
    VID   f_zero = 0;
    VID   f_lg1 = 0;

    expr::array_ro<ftype,VID,var_data> a_data( const_cast<ftype *>( vals ) );
    expr::array_ro<ftype,VID,var_min>  a_min( &f_min );
    expr::array_ro<ftype,VID,var_minz> a_minz( &f_minz );
    expr::array_ro<ftype,VID,var_max>  a_max( &f_max );
    expr::array_ro<ftype,VID,var_avg>  a_avg( &f_avg );
    expr::array_ro<ftype,VID,var_sq>   a_sq(  &f_sq );
    expr::array_ro<VID,VID,var_neg>    a_neg( &f_neg );
    expr::array_ro<VID,VID,var_zero>   a_zero( &f_zero );
    expr::array_ro<VID,VID,var_lg1>    a_lg1( &f_lg1 );

    make_lazy_executor( part )
	.vertex_scan( [&]( auto v ) {
	     auto one = expr::constant_val_one( a_neg[expr::zero_val(v)] );
	     auto zero = expr::zero_val( a_data[v] );
	     return expr::make_seq(
		 a_min[expr::zero_val(v)].min( a_data[v] ),
		 a_minz[expr::zero_val(v)].min( expr::add_predicate( a_data[v], a_data[v] != expr::zero_val( a_data[v] ) ) ),
		 a_max[expr::zero_val(v)].max( a_data[v] ),
		 a_avg[expr::zero_val(v)] += a_data[v],
		 a_sq[expr::zero_val(v)] += a_data[v] * a_data[v],
		 a_neg[expr::zero_val(v)]
		     += expr::add_predicate(
			 expr::constant_val_one( a_neg[expr::zero_val(v)] ),
			 a_data[v] < zero ),
		 a_zero[expr::zero_val(v)]
		     += expr::add_predicate(
			 expr::constant_val_one( a_neg[expr::zero_val(v)] ),
			 a_data[v] == zero ),
		 a_lg1[expr::zero_val(v)]
		     += expr::add_predicate(
			 expr::constant_val_one( a_lg1[expr::zero_val(v)] ),
			 a_data[v] > expr::constant_val_one( a_data[v] ) )
		 );
		      } )
	.materialize();
	     

    VID n = part.get_num_vertices();
    ftype sdev = ( f_avg * f_avg - f_sq ) / ftype(n-1);

    std::cerr << "min=" << f_min
	      << " minz=" << f_minz
	      << " max=" << f_max
	      << " avg=" << ( f_avg / ftype(n) )
	      << " sq=" << f_sq
	      << " sdev=" << sdev
	      << " neg=" << f_neg
	      << " zero=" << f_zero
	      << " lg1=" << f_lg1
	      << " n=" << n
	      << "\n";
}

template<unsigned short base_AID, typename ftype>
void analyse_range( const partitioner & part, const ftype * vals,
		    const VID * deg ) {
    constexpr unsigned short var_data = base_AID + 0;
    constexpr unsigned short var_min = base_AID + 1;
    constexpr unsigned short var_max = base_AID + 2;
    constexpr unsigned short var_avg = base_AID + 3;
    constexpr unsigned short var_sq  = base_AID + 4;
    constexpr unsigned short var_neg = base_AID + 5;
    constexpr unsigned short var_zero = base_AID + 6;
    constexpr unsigned short var_deg = base_AID + 7;
    constexpr unsigned short var_minz = base_AID + 8;
    constexpr unsigned short var_lg1 = base_AID + 9;
    
    ftype f_min = std::numeric_limits<ftype>::max();
    ftype f_minz = std::numeric_limits<ftype>::max();
    ftype f_max = std::numeric_limits<ftype>::min();
    ftype f_avg = 0;
    ftype f_sq = 0;
    VID   f_neg = 0;
    VID   f_zero = 0;
    VID   f_lg1 = 0;

    expr::array_ro<ftype,VID,var_data> a_data( const_cast<ftype *>( vals ) );
    expr::array_ro<ftype,VID,var_min> a_min( &f_min );
    expr::array_ro<ftype,VID,var_minz> a_minz( &f_minz );
    expr::array_ro<ftype,VID,var_max> a_max( &f_max );
    expr::array_ro<ftype,VID,var_avg> a_avg( &f_avg );
    expr::array_ro<ftype,VID,var_sq>  a_sq(  &f_sq );
    expr::array_ro<VID,VID,var_neg>   a_neg( &f_neg );
    expr::array_ro<VID,VID,var_zero>  a_zero( &f_zero );
    expr::array_ro<VID,VID,var_deg>   a_deg( const_cast<VID *>( deg ) );
    expr::array_ro<VID,VID,var_lg1>    a_lg1( &f_lg1 );

    make_lazy_executor( part )
	.vertex_scan( [&]( auto v ) {
	     auto one = expr::constant_val_one( a_neg[expr::zero_val(v)] );
	     auto zero = expr::zero_val( a_data[v] );
	     auto dz = expr::zero_val( a_deg[v] );
	     // auto d = expr::add_mask( a_data[v], a_deg[v] != dz );
	     auto d = a_data[v];
	     auto c = a_deg[v] != dz;
	     return expr::make_seq(
		 a_min[expr::zero_val(v)].min( expr::add_predicate( expr::abs( d ), c ) ),
		 a_minz[expr::zero_val(v)].min( expr::add_predicate( expr::abs( d ), a_data[v] != expr::zero_val( a_data[v] ) && c ) ),
		 a_max[expr::zero_val(v)].max( expr::add_predicate( expr::abs( d ), c ) ),
		 a_avg[expr::zero_val(v)] += expr::add_predicate( d, c ),
		 a_sq[expr::zero_val(v)] += expr::add_predicate( d * d, c ),
		 a_neg[expr::zero_val(v)]
		     += expr::add_predicate(
			 expr::constant_val_one( a_neg[expr::zero_val(v)] ),
			 d < zero && a_deg[v] != dz ),
		 a_zero[expr::zero_val(v)]
		     += expr::add_predicate(
			 expr::constant_val_one( a_neg[expr::zero_val(v)] ),
			 d == zero && a_deg[v] != dz ),
		 a_lg1[expr::zero_val(v)]
		     += expr::add_predicate(
			 expr::constant_val_one( a_lg1[expr::zero_val(v)] ),
			 expr::abs( d ) > expr::constant_val_one( a_data[v] )
			 && c )
		 );
		     } )
	.materialize();
	     

    VID n = part.get_num_vertices();
    ftype sdev = ( f_avg * f_avg - f_sq ) / ftype(n-1);

    std::cerr << "min=" << f_min
	      << " minz=" << f_minz
	      << " max=" << f_max
	      << " avg=" << ( f_avg / ftype(n) )
	      << " sq=" << f_sq
	      << " sdev=" << sdev
	      << " neg=" << f_neg
	      << " zero=" << f_zero
	      << " lg1=" << f_lg1
	      << " n=" << n
	      << "\n";
}

template<unsigned short base_AID, typename ftype>
void analyse_range( const partitioner & part, const ftype * vals,
		    const VID * deg, const frontier & F ) {
    constexpr unsigned short var_data = base_AID + 0;
    constexpr unsigned short var_min = base_AID + 1;
    constexpr unsigned short var_max = base_AID + 2;
    constexpr unsigned short var_avg = base_AID + 3;
    constexpr unsigned short var_sq  = base_AID + 4;
    constexpr unsigned short var_neg = base_AID + 5;
    constexpr unsigned short var_zero = base_AID + 6;
    constexpr unsigned short var_deg = base_AID + 7;
    constexpr unsigned short var_minz = base_AID + 8;
    constexpr unsigned short var_lg1 = base_AID + 9;
    
    ftype f_min = std::numeric_limits<ftype>::max();
    ftype f_minz = std::numeric_limits<ftype>::max();
    ftype f_max = std::numeric_limits<ftype>::min();
    ftype f_avg = 0;
    ftype f_sq = 0;
    VID   f_neg = 0;
    VID   f_zero = 0;
    VID   f_lg1 = 0;

    expr::array_ro<ftype,VID,var_data> a_data( const_cast<ftype *>( vals ) );
    expr::array_ro<ftype,VID,var_min> a_min( &f_min );
    expr::array_ro<ftype,VID,var_minz> a_minz( &f_minz );
    expr::array_ro<ftype,VID,var_max> a_max( &f_max );
    expr::array_ro<ftype,VID,var_avg> a_avg( &f_avg );
    expr::array_ro<ftype,VID,var_sq>  a_sq(  &f_sq );
    expr::array_ro<VID,VID,var_neg>   a_neg( &f_neg );
    expr::array_ro<VID,VID,var_zero>  a_zero( &f_zero );
    expr::array_ro<VID,VID,var_deg>   a_deg( const_cast<VID *>( deg ) );
    expr::array_ro<VID,VID,var_lg1>    a_lg1( &f_lg1 );

    make_lazy_executor( part )
	.vertex_scan( F, [&]( auto v ) {
	     auto one = expr::constant_val_one( a_neg[expr::zero_val(v)] );
	     auto zero = expr::zero_val( a_data[v] );
	     auto dz = expr::zero_val( a_deg[v] );
	     // auto d = expr::add_mask( a_data[v], a_deg[v] != dz );
	     auto d = a_data[v];
	     auto c = a_deg[v] != dz;
	     return expr::make_seq(
		 a_min[expr::zero_val(v)].min( expr::add_predicate( expr::abs( d ), c ) ),
		 a_minz[expr::zero_val(v)].min( expr::add_predicate( expr::abs( d ), a_data[v] != expr::zero_val( a_data[v] ) && c ) ),
		 a_max[expr::zero_val(v)].max( expr::add_predicate( expr::abs( d ), c ) ),
		 a_avg[expr::zero_val(v)] += expr::add_predicate( d, c ),
		 a_sq[expr::zero_val(v)] += expr::add_predicate( d * d, c ),
		 a_neg[expr::zero_val(v)]
		     += expr::add_predicate(
			 expr::constant_val_one( a_neg[expr::zero_val(v)] ),
			 d < zero && a_deg[v] != dz ),
		 a_zero[expr::zero_val(v)]
		     += expr::add_predicate(
			 expr::constant_val_one( a_neg[expr::zero_val(v)] ),
			 d == zero && a_deg[v] != dz ),
		 a_lg1[expr::zero_val(v)]
		     += expr::add_predicate(
			 expr::constant_val_one( a_lg1[expr::zero_val(v)] ),
			 expr::abs( d ) > expr::constant_val_one( a_data[v] )
			 && c )
		 );
		     } )
	.materialize();
	     

    VID n = part.get_num_vertices();
    ftype sdev = ( f_avg * f_avg - f_sq ) / ftype(n-1);

    std::cerr << "min=" << f_min
	      << " minz=" << f_minz
	      << " max=" << f_max
	      << " avg=" << ( f_avg / ftype(n) )
	      << " sq=" << f_sq
	      << " sdev=" << sdev
	      << " neg=" << f_neg
	      << " zero=" << f_zero
	      << " lg1=" << f_lg1
	      << " n=" << n
	      << "\n";
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
	    std::cerr << "WIDEN_APPROACH=" << WIDEN_APPROACH << "\n";
	    std::cerr << "VARIANT=" << VARIANT << "\n";
	    std::cerr << "SCALE_CONTRIB=" << SCALE_CONTRIB << "\n";
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
	VID nactv;
	EID nacte;
	double error;
	const char * variant;

	void dump( int i ) {
	    std::cerr << "Iteration " << i << ": " << delay
		      << " density: " << density
		      << " (E:" << nacte
		      << ", V:" << nactv
		      << ") error: " << error
		      << " variant: " << variant
		      << "\n";
	}
    };

    struct stat { };

    template<typename ContribTy, typename AccumTy, typename FloatTy,
	     typename ContribEnc, typename AccumEnc, typename FloatEnc,
	     typename InitType>
    std::pair<mmap_ptr<FloatTy>,mmap_ptr<FloatTy>>
    run_prec( mmap_ptr<InitType> x_init,
	      mmap_ptr<InitType> d_init,
	      frontier & F,
	      int & iter,
	      double eps, double eps2,
	      const char * variant ) {
	timer tm_iter;
	tm_iter.start();

	const partitioner &part = GA.get_partitioner();
	VID n = GA.numVertices();
	EID m = GA.numEdges();

	mmap_ptr<FloatTy> delta_mem;
	delta_mem.allocate( numa_allocation_partitioned( part ) );
	expr::array_ro<FloatTy,VID,var_delta,FloatEnc> delta( delta_mem.get() );
	expr::array_ro<FloatTy,VID,var_extdl,array_encoding<InitType>> di( d_init.get() );

	mmap_ptr<typename AccumEnc::stored_type> z_mem;
	z_mem.allocate( numa_allocation_partitioned( part ) );
	expr::array_ro<FloatTy,VID,var_z,AccumEnc> z( z_mem.get() );
	expr::array_ro<AccumTy,VID,var_z,AccumEnc> z_acc( z_mem.get() );

	mmap_ptr<FloatTy> x_mem;
	x_mem.allocate( numa_allocation_partitioned( part ) );
	expr::array_ro<FloatTy,VID,var_x,FloatEnc> x( x_mem.get() );
	expr::array_ro<FloatTy,VID,var_extpr,array_encoding<InitType>> xi( x_init.get() );

#if MEMO
	mmap_ptr<typename ContribEnc::stored_type> contrib_mem;
	contrib_mem.allocate( numa_allocation_partitioned( part ) );
	expr::array_ro<FloatTy,VID,var_contrib,ContribEnc>
	    contrib_seq( contrib_mem.get() );
	expr::array_ro<ContribTy,VID,var_contrib,ContribEnc>
	    contrib_rnd( contrib_mem.get() );
#else
#endif
	mmap_ptr<FloatTy> outdeg_mem;
	outdeg_mem.allocate( numa_allocation_partitioned( part ) );
	expr::array_ro<FloatTy,VID,var_outdeg,FloatEnc> outdeg( outdeg_mem.get() );

	expr::array_ro<VID,VID,var_degree> degree(
	    const_cast<VID *>( GA.getCSR().getDegree() ) );

	// Configuration
	FloatTy dampen = 0.85;
	// const FloatTy eps  = 1e-7;
	// const FloatTy eps2 = 1e-2;

	// Initialise arrays
	FloatTy ninv = FloatTy(1)/FloatTy(n);
	make_lazy_executor( part )
	    .vertex_map( [&]( auto vid ) { return x[vid] = xi[vid]; } )
	    .vertex_map( [&]( auto vid ) { return delta[vid] = di[vid]; } )
	    .materialize();

#if !MEMO
#endif

#if SCALE_CONTRIB
	FloatTy factor = 1.0;
	VID nnn = n;
	while( nnn > 1 ) {
	    factor /= 10.0;
	    nnn /= 10;
	}
	// factor = 0.01; // override
	// factor = 1; // override
	map_vertexL( part, [&]( VID j ){ outdeg_mem[j] = ((FloatTy)1)/(factor * (FloatTy)GA.getOutDegree(j)); } );
#else
	map_vertexL( part, [&]( VID j ){ outdeg_mem[j] = ((FloatTy)1)/((FloatTy)GA.getOutDegree(j)); } );
#endif // SCALE_CONTRIB

	// Control
	// iter = 0;

	while( iter < max_iter ) {
	    // frontier output; // unused, initialized to all_true
	    frontier active = frontier::dense<sizeof(VID)>( part );
	    using FrTy = typename add_logical<VID>::type;
	    expr::array_ro<FrTy,VID,var_frontier> active_arr( active.getDense<FrTy>() );
	    FloatTy cst = (FloatTy(1)-dampen) * ninv;
	    if( iter == 0 ) {
		make_lazy_executor( part )
		    .vertex_map( [&]( auto v ) {
			    return z[v] = expr::zero_val( z[v] );
			} )
#if MEMO
		    // Don't specify frontier, as all vertices initially active
		    .vertex_map( [&]( auto vid ) {
			    return contrib_seq[vid] = delta[vid] * outdeg[vid];
			} )
#endif
		    .materialize();

	    api::edgemap(
		GA,
		api::relax( [&]( auto s, auto d, auto e ) {
#if MEMO
#if SCALE_CONTRIB
				return z_acc[d] += expr::cast<AccumTy>( contrib_rnd[s] ) * expr::constant_val( z_acc[d], factor );
#else
				return z_acc[d] += expr::cast<AccumTy>( contrib_rnd[s] );
#endif
#else
			return z_acc[d] += expr::cast<AccumTy>(
			    delta[s] * outdeg[s] );
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
		    .vertex_map( [&]( auto v ) {
			    return z[v] = expr::zero_val( z[v] );
			} )
#if MEMO
		    .vertex_map( [&]( auto vid ) {
			    return contrib_seq[vid] = delta[vid] * outdeg[vid];
			} )
#endif
		    .materialize();

		api::edgemap(
		    GA,
		    api::relax( [&]( auto s, auto d, auto e ) {
#if MEMO
#if SCALE_CONTRIB
			    return z_acc[d] += expr::cast<AccumTy>( contrib_rnd[s] ) * expr::constant_val( z_acc[d], factor );
#else
			    return z_acc[d] += expr::cast<AccumTy>( contrib_rnd[s] );
#endif
#else
			    return z_acc[d] += delta[s] * outdeg[s];
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
						     active_arr[v] != expr::zero_val( active_arr[v] ) ); } )
		    .materialize();
		// TODO: fuse with above... (scan)
		active.calculateActiveCounts( GA );
#endif
	    } else { // ! UNCOND_EXEC
		// TODO: vertex_map's take much more time than edgemap
		//       when F is sparse. Need to rewrite with a second
		//       frontier that shows what cells in z may be non-zero.
		//       This second frontier is determined by edgemap.
		//       Most of the vmap time is spent after the edgemap
		//       (setting of delta), then calculation of frontier
		make_lazy_executor( part )
		    .vertex_map( [&]( auto v ) {
			    return z[v] = expr::zero_val( z[v] );
			} )
#if MEMO
		    .vertex_map( F, [&]( auto vid ) {
			    return contrib_seq[vid] = delta[vid] * outdeg[vid];
			} )
#endif
		    .materialize();

		api::edgemap(
		    GA,
		    api::config( api::frac_threshold(1) ),
		    api::filter( F, api::strong, api::src ),
		    api::relax( [&]( auto s, auto d, auto e ) {
#if MEMO
#if SCALE_CONTRIB
			    return z_acc[d] += expr::cast<AccumTy>( contrib_rnd[s] ) * expr::constant_val( z_acc[d], factor );
#else
			    return z_acc[d] += expr::cast<AccumTy>( contrib_rnd[s] );
#endif
#else
			    return z_acc[d] += delta[s] * outdeg[s];
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
						     active_arr[v] != expr::zero_val( active_arr[v] ) ); } )
		    .materialize();
		// TODO: fuse with above... (scan)
		active.calculateActiveCounts( GA );
	    }

/*
	    std::cerr << "z:";
	    for( VID v=0; v < 16; ++v ) {
		assert( z_mem[v] < FloatTy(1) );
		std::cerr << ' ' << z_mem[GA.remapID(v)];
	    }
	    std::cerr << "\n";

	    std::cerr << "contrib:";
	    for( VID v=0; v < 16; ++v ) {
		std::cerr << ' ' << (float)contrib_mem[GA.remapID(v)];
	    }
	    std::cerr << "\n";
*/

	    FloatTy L1norm = 0;
	    expr::array_ro<FloatTy,VID,var_s> accum( &L1norm );
	    make_lazy_executor( part )
		.vertex_scan( [&]( auto vid ) {
		      return accum[expr::zero_val(vid)]
			      += expr::abs( expr::cast<FloatTy>( delta[vid] ) ); } )
		.materialize();

	    if( itimes ) {
		info_buf[iter].density = F.density( GA.numEdges() );
		info_buf[iter].nactv = F.nActiveVertices();
		info_buf[iter].nacte = F.nActiveEdges();
		info_buf[iter].error = L1norm;
		info_buf[iter].delay = tm_iter.next();
		info_buf[iter].variant = variant;
		if( debug ) {
		    info_buf[iter].dump( iter );
		}
#if ANALYSE_VALUES
		analyse_range<100>( part, contrib_seq.ptr(), degree.ptr(), F );
#endif
	    }
	    iter++;

	    F.del();
	    F = active;

	    if( L1norm < eps )
		break;
	    // if( L1norm > 2.0 ) // Addition: early
	    // break;
	}

	z_mem.del();
#if MEMO
	contrib_mem.del();
#else
	outdeg_mem.del();
#endif

	return std::make_pair( x_mem, delta_mem );
    }

    template<typename InitType>
    std::pair<mmap_ptr<float>,mmap_ptr<float>>
    run_prec_40( mmap_ptr<InitType> x_init, mmap_ptr<InitType> d_init,
		 frontier & F, int & iter, double eps, double eps2 ) {
	using ContribTy = double;
	using AccumTy = double;
	using FloatTy = float;
	using ContribEnc = array_encoding_wide<scustomfp<true,6,9,true,0>>;
	using AccumEnc = array_encoding<FloatTy>;
	using FloatEnc = array_encoding<FloatTy>;

	return run_prec<ContribTy,AccumTy,FloatTy,
			ContribEnc,AccumEnc,FloatEnc>(
			    x_init, d_init, F, iter, eps, eps2, "40" );
    }

    template<typename InitType>
    std::pair<mmap_ptr<float>,mmap_ptr<float>>
    run_prec_41( mmap_ptr<InitType> x_init, mmap_ptr<InitType> d_init,
		 frontier & F, int & iter, double eps, double eps2 ) {
	using ContribTy = double;
	using AccumTy = double;
	using FloatTy = float;
	using ContribEnc = array_encoding_wide<scustomfp<true,6,9,true,0>>;
	using AccumEnc = array_encoding_wide<FloatTy>;
	using FloatEnc = array_encoding<FloatTy>;

	return run_prec<ContribTy,AccumTy,FloatTy,
			ContribEnc,AccumEnc,FloatEnc>(
			    x_init, d_init, F, iter, eps, eps2, "41" );
    }

    template<typename InitType>
    std::pair<mmap_ptr<float>,mmap_ptr<float>>
    run_prec_42( mmap_ptr<InitType> x_init, mmap_ptr<InitType> d_init,
		 frontier & F, int & iter, double eps, double eps2 ) {
	using ContribTy = float;
	using AccumTy = double;
	using FloatTy = float;
	using ContribEnc = array_encoding_wide<scustomfp<true,6,9,true,0>>;
	using AccumEnc = array_encoding_wide<FloatTy>;
	using FloatEnc = array_encoding<FloatTy>;

	return run_prec<ContribTy,AccumTy,FloatTy,
			ContribEnc,AccumEnc,FloatEnc>(
			    x_init, d_init, F, iter, eps, eps2, "42" );
    }

    template<typename InitType>
    std::pair<mmap_ptr<float>,mmap_ptr<float>>
    run_prec_45( mmap_ptr<InitType> x_init, mmap_ptr<InitType> d_init,
		 frontier & F, int & iter, double eps, double eps2 ) {
	using ContribTy = float;
	using AccumTy = float;
	using FloatTy = float;
	using ContribEnc = array_encoding_wide<scustomfp<true,6,9,true,0>>;
	using AccumEnc = array_encoding<FloatTy>;
	using FloatEnc = array_encoding<FloatTy>;

	return run_prec<ContribTy,AccumTy,FloatTy,
			ContribEnc,AccumEnc,FloatEnc>(
			    x_init, d_init, F, iter, eps, eps2, "45" );
    }

    template<typename InitType>
    std::pair<mmap_ptr<float>,mmap_ptr<float>>
    run_prec_46( mmap_ptr<InitType> x_init, mmap_ptr<InitType> d_init,
		 frontier & F, int & iter, double eps, double eps2 ) {
	using ContribTy = float;
	using AccumTy = float;
	using FloatTy = float;
	using ContribEnc = array_encoding_wide<scustomfp<true,6,9,true,0>>;
	using AccumEnc = array_encoding/*_wide*/<FloatTy>;
	using FloatEnc = array_encoding<FloatTy>;

	return run_prec<ContribTy,AccumTy,FloatTy,
			ContribEnc,AccumEnc,FloatEnc>(
			    x_init, d_init, F, iter, eps, eps2, "46" );
    }

    template<typename InitType>
    std::pair<mmap_ptr<float>,mmap_ptr<float>>
    run_prec_56( mmap_ptr<InitType> x_init, mmap_ptr<InitType> d_init,
		 frontier & F, int & iter, double eps, double eps2 ) {
	using ContribTy = float;
	using AccumTy = float;
	using FloatTy = float;
	using ContribEnc = array_encoding_wide<scustomfp<true,6,9,true,-20>>;
	using AccumEnc = array_encoding/*_wide*/<FloatTy>;
	using FloatEnc = array_encoding<FloatTy>;

	return run_prec<ContribTy,AccumTy,FloatTy,
			ContribEnc,AccumEnc,FloatEnc>(
			    x_init, d_init, F, iter, eps, eps2, "56" );
    }

    template<typename InitType>
    std::pair<mmap_ptr<float>,mmap_ptr<float>>
    run_prec_36( mmap_ptr<InitType> x_init, mmap_ptr<InitType> d_init,
		 frontier & F, int & iter, double eps, double eps2 ) {
	using ContribTy = float;
	using AccumTy = float;
	using FloatTy = float;
	using ContribEnc = array_encoding<scustomfp<true,6,9,true,0>>;
	using AccumEnc = array_encoding<FloatTy>;
	using FloatEnc = array_encoding<FloatTy>;

	return run_prec<ContribTy,AccumTy,FloatTy,
			ContribEnc,AccumEnc,FloatEnc>(
			    x_init, d_init, F, iter, eps, eps2, "36" );
    }

    template<typename InitType>
    std::pair<mmap_ptr<float>,mmap_ptr<float>>
    run_prec_47( mmap_ptr<InitType> x_init, mmap_ptr<InitType> d_init,
		 frontier & F, int & iter, double eps, double eps2 ) {
	using ContribTy = float;
	using AccumTy = float;
	using FloatTy = float;
	using ContribEnc = array_encoding_wide<scustomfp<false,8,8,false,0>>;
	using AccumEnc = array_encoding/*_wide*/<FloatTy>;
	using FloatEnc = array_encoding<FloatTy>;

	return run_prec<ContribTy,AccumTy,FloatTy,
			ContribEnc,AccumEnc,FloatEnc>(
			    x_init, d_init, F, iter, eps, eps2, "47" );
    }

    template<typename InitType>
    std::pair<mmap_ptr<float>,mmap_ptr<float>>
    run_prec_48( mmap_ptr<InitType> x_init, mmap_ptr<InitType> d_init,
		 frontier & F, int & iter, double eps, double eps2 ) {
	using ContribTy = float;
	using AccumTy = float;
	using FloatTy = float;
	using ContribEnc = array_encoding_wide<scustomfp<true,5,10,true,0>>;
	using AccumEnc = array_encoding/*_wide*/<FloatTy>;
	using FloatEnc = array_encoding<FloatTy>;

	return run_prec<ContribTy,AccumTy,FloatTy,
			ContribEnc,AccumEnc,FloatEnc>(
			    x_init, d_init, F, iter, eps, eps2, "48" );
    }

    template<typename InitType>
    std::pair<mmap_ptr<float>,mmap_ptr<float>>
    run_prec_49( mmap_ptr<InitType> x_init, mmap_ptr<InitType> d_init,
		 frontier & F, int & iter, double eps, double eps2 ) {
	using ContribTy = float;
	using AccumTy = float;
	using FloatTy = float;
	using ContribEnc = array_encoding_wide<scustomfp<true,4,11,true,0>>;
	using AccumEnc = array_encoding/*_wide*/<FloatTy>;
	using FloatEnc = array_encoding<FloatTy>;

	return run_prec<ContribTy,AccumTy,FloatTy,
			ContribEnc,AccumEnc,FloatEnc>(
			    x_init, d_init, F, iter, eps, eps2, "49" );
    }

    template<typename InitType>
    std::pair<mmap_ptr<float>,mmap_ptr<float>>
    run_prec_50( mmap_ptr<InitType> x_init, mmap_ptr<InitType> d_init,
		 frontier & F, int & iter, double eps, double eps2 ) {
	using ContribTy = float;
	using AccumTy = float;
	using FloatTy = float;
	using ContribEnc = array_encoding_wide<scustomfp<true,4,3,true,0>>;
	using AccumEnc = array_encoding/*_wide*/<FloatTy>;
	using FloatEnc = array_encoding<FloatTy>;

	return run_prec<ContribTy,AccumTy,FloatTy,
			ContribEnc,AccumEnc,FloatEnc>(
			    x_init, d_init, F, iter, eps, eps2, "50" );
    }

    template<typename InitType>
    std::pair<mmap_ptr<float>,mmap_ptr<float>>
    run_prec_20( mmap_ptr<InitType> x_init, mmap_ptr<InitType> d_init,
		 frontier & F, int & iter, double eps, double eps2 ) {
	using ContribTy = double;
	using AccumTy = double;
	using FloatTy = float;
	using ContribEnc = array_encoding<FloatTy>;
	using AccumEnc = array_encoding<double>;
	using FloatEnc = array_encoding<FloatTy>;

	return run_prec<ContribTy,AccumTy,FloatTy,
			ContribEnc,AccumEnc,FloatEnc>(
			    x_init, d_init, F, iter, eps, eps2, "20" );
    }

    template<typename InitType>
    std::pair<mmap_ptr<float>,mmap_ptr<float>>
    run_prec_10( mmap_ptr<InitType> x_init, mmap_ptr<InitType> d_init,
		 frontier & F, int & iter, double eps, double eps2 ) {
	using ContribTy = float;
	using AccumTy = float;
	using FloatTy = float;
	using ContribEnc = array_encoding<FloatTy>;
	using AccumEnc = array_encoding<float>;
	using FloatEnc = array_encoding<FloatTy>;

	return run_prec<ContribTy,AccumTy,FloatTy,
			ContribEnc,AccumEnc,FloatEnc>(
			    x_init, d_init, F, iter, eps, eps2, "10" );
    }

    template<typename InitType>
    std::pair<mmap_ptr<double>,mmap_ptr<double>>
    run_prec_11( mmap_ptr<InitType> x_init, mmap_ptr<InitType> d_init,
		 frontier & F, int & iter, double eps, double eps2 ) {
	using ContribTy = double;
	using AccumTy = double;
	using FloatTy = double;
	using ContribEnc = array_encoding<FloatTy>;
	using AccumEnc = array_encoding<double>;
	using FloatEnc = array_encoding<FloatTy>;

	return run_prec<ContribTy,AccumTy,FloatTy,
			ContribEnc,AccumEnc,FloatEnc>(
			    x_init, d_init, F, iter, eps, eps2, "11" );
    }

    void run() {
	const partitioner &part = GA.get_partitioner();
	VID n = GA.numVertices();
	EID m = GA.numEdges();

	mmap_ptr<double> delta_mem;
	delta_mem.allocate( numa_allocation_partitioned( part ) );
	expr::array_ro<double,VID,var_delta> delta( delta_mem.get() );

	expr::array_ro<double,VID,var_x> x( x_mem.get() );

	// Configuration
	double dampen = 0.85;
	const double eps  = 1e-7;
	const double eps2 = 1e-2;

	double ninv = double(1)/double(n);

	// Initialise arrays
	if( infile )
	    map_vertexL( part, [&]( VID v ){ delta_mem[v] = ninv; } );
	else
	    map_vertexL( part, [&]( VID v ){ x_mem[v] = 0; delta_mem[v] = ninv; } );

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


	// Control
	iter = 0;

	timer tm_iter;
	tm_iter.start();

#if VARIANT == 10
	auto xd0 = run_prec_10( x_mem, delta_mem, F, iter, eps, eps2 );
#elif VARIANT == 11
	auto xd0 = run_prec_11( x_mem, delta_mem, F, iter, eps, eps2 );
#elif VARIANT == 20
	auto xd0 = run_prec_20( x_mem, delta_mem, F, iter, eps, eps2 );
#elif VARIANT == 40
	auto xd0 = run_prec_40( x_mem, delta_mem, F, iter, eps, eps2 );
#elif VARIANT == 41
	auto xd0 = run_prec_41( x_mem, delta_mem, F, iter, eps, eps2 );
#elif VARIANT == 42
	auto xd0 = run_prec_42( x_mem, delta_mem, F, iter, eps, eps2 );
#elif VARIANT == 45
	auto xd0 = run_prec_45( x_mem, delta_mem, F, iter, eps, eps2 );
#elif VARIANT == 46
	auto xd0 = run_prec_46( x_mem, delta_mem, F, iter, eps, eps2 );
#elif VARIANT == 4620
	auto x0 = run_prec_46( x_mem, delta_mem, F, iter, 1e-1, eps2 );
	auto xd0 = run_prec_20( x0.first, x0.second, F, iter, eps, eps2 );
	x0.first.del();
	x0.second.del();
#elif VARIANT == 2042
	auto x0 = run_prec_20( x_mem, delta_mem, F, iter, 1e-1, eps2 );
	auto xd0 = run_prec_42( x0.first, x0.second, F, iter, eps, eps2 );
	x0.first.del();
	x0.second.del();
#elif VARIANT == 1046
	auto x0 = run_prec_10( x_mem, delta_mem, F, iter, 1e-1, eps2 );
	auto xd0 = run_prec_46( x0.first, x0.second, F, iter, eps, eps2 );
	x0.first.del();
	x0.second.del();
#elif VARIANT == 2046
	auto x0 = run_prec_20( x_mem, delta_mem, F, iter, 1e-1, eps2 );
	auto xd0 = run_prec_46( x0.first, x0.second, F, iter, eps, eps2 );
	x0.first.del();
	x0.second.del();
#elif VARIANT == 2056
	auto x0 = run_prec_20( x_mem, delta_mem, F, iter, 1e-1, eps2 );
	auto xd0 = run_prec_56( x0.first, x0.second, F, iter, eps, eps2 );
	x0.first.del();
	x0.second.del();
#elif VARIANT == 2036
	auto x0 = run_prec_20( x_mem, delta_mem, F, iter, 1e-1, eps2 );
	auto xd0 = run_prec_36( x0.first, x0.second, F, iter, eps, eps2 );
	x0.first.del();
	x0.second.del();
#elif VARIANT == 4720
	auto x0 = run_prec_47( x_mem, delta_mem, F, iter, 1e-1, eps2 );
	auto xd0 = run_prec_20( x0.first, x0.second, F, iter, eps, eps2 );
	x0.first.del();
	x0.second.del();
#elif VARIANT == 2048
	auto x0 = run_prec_20( x_mem, delta_mem, F, iter, 1e-1, eps2 );
	auto xd0 = run_prec_48( x0.first, x0.second, F, iter, eps, eps2 );
	x0.first.del();
	x0.second.del();
#elif VARIANT == 2049
	auto x0 = run_prec_20( x_mem, delta_mem, F, iter, 1e-1, eps2 );
	auto xd0 = run_prec_49( x0.first, x0.second, F, iter, eps, eps2 );
	x0.first.del();
	x0.second.del();
#elif VARIANT == 204950
	auto x0 = run_prec_20( x_mem, delta_mem, F, iter, 1e-1, eps2 );
	auto xd1 = run_prec_49( x0.first, x0.second, F, iter, 1e-2, eps2 );
	x0.first.del();
	x0.second.del();
	auto xd0 = run_prec_49( xd1.first, xd1.second, F, iter, eps, eps2 );
	xd1.first.del();
	xd1.second.del();
#endif
	delta_mem.del();
	F.del();

	// Copy back to xd_mem
	using x1_ty = std::decay_t<decltype(xd0.first)>;
	using type = typename x1_ty::type;
	expr::array_ro<double,VID,var_extpr,array_encoding<type>>
	    x1a( xd0.first.get() );
	double ss = 0;
	expr::array_ro<double,VID,var_s> accum( &ss );
	make_lazy_executor( part )
	    .vertex_map( [&]( auto vid ) { return x[vid] = x1a[vid]; } )
	    .vertex_scan( [&]( auto vid ) {
		      return accum[expr::zero_val(vid)] += expr::abs( x[vid] ); } )
	    .materialize();

	double s = double(1) / ss;
	make_lazy_executor( part )
	    .vertex_map( [&]( auto vid ) {
			     return x[vid] *= expr::constant_val( x[vid], s ); } )
	    .materialize();
	ftrue.del();

	xd0.first.del();
	xd0.second.del();
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
    mmap_ptr<double> x_mem;
};

template <class GraphType>
using Benchmark = APRv<GraphType>;

#include "driver.C"
