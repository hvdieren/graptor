// -*- c++ -*-
/*!=====================================================================*
 * \brief A micro-benchmark for intersection algorithms.
 *
 * Tested algorithms:
 * - merge-based intersection using scalar operations
 * - merge-based intersection using vector operations
 * - merge-based intersection using scalar operations and jumping ahead
 *   in the stream using std::lower_bound()
 * - hash-based intersection using scalar operations
 * - hash-based intersection using vector operations
 * Three operation types are tested:
 * - Performing the intersection, producing a list
 * - Counting intersection size
 * - Counting intersection size only when it exceeds a preset threshold.
 *   When it becomes clear that the preset threshold cannot be reached,
 *   the intersection is aborted.
 * The algorithms are tested on sets drawn from the adjacency lists of
 * a graph dataset.
 *======================================================================*/

/*!=====================================================================*
 * TODO:
 * Further variations and tweaks to consider
 * + all variants: trim lists at the start and end similar to jumping
 *======================================================================*/
#include <mutex>
#include <numeric>
#include <thread>

#include <time.h>

#include "graptor/graptor.h"
#include "graptor/api.h"

#include "graptor/container/hash_set.h"
#include "graptor/graph/simple/hadjt.h"
#include "graptor/container/intersect.h"
#include "graptor/container/hash_fn.h"
#include "graptor/stat/timing.h"

using hash_fn = graptor::rand_hash<uint32_t>;

/*! Enumeration of algorithmic variants
 */
enum ins_variant {
    var_merge_scalar = 0,	/**< merge-based, scalar */
    var_merge_vector = 1,	/**< merge-based, vector */
    var_merge_jump = 2,		/**< merge-based, scalar and jumping ahead */
    var_hash_scalar = 3,	/**< hash-based, scalar */
    var_hash_vector = 4,	/**< hash-based, vector */
    var_merge_partitioned_scalar = 5,	/**< partition pre-study, scalar */
    var_hash_partitioned_vector = 6,	/**< partition pre-study, vector hash */
    var_hash_wide = 7, 	 	/**< hash-based, fetch multiple slots */
    var_N = 8 	 		/**< number of options */
};

/*! Enumeration of operation types
 */
enum operation {
    op_intersect = 0,		/**< intersection producing list */
    op_intersect_size = 1,	/**< intersection size */
    op_intersect_size_gt_val = 2,	/**< intersection size or abort */
    op_N = 3			/**< number of options */
};

/*! Print an algorithmic variant in human-readable form
 */
std::ostream & operator << ( ostream & os, ins_variant v ) {
    switch( v ) {
    case var_merge_scalar: return os << "merge_scalar";
    case var_merge_vector: return os << "merge_vector";
    case var_merge_jump: return os << "merge_jump";
    case var_hash_scalar: return os << "hash_scalar";
    case var_hash_vector: return os << "hash_vector";
    case var_merge_partitioned_scalar: return os << "merge_partitioned_scalar";
    case var_hash_partitioned_vector: return os << "hash_partitioned_vector";
    case var_hash_wide: return os << "hash_wide";
    default:
	return os << "illegal-variant";
    }
}

/*! Print an operation type in human-readable form
 */
std::ostream & operator << ( ostream & os, operation o ) {
    switch( o ) {
    case op_intersect: return os << "intersect";
    case op_intersect_size: return os << "intersect_size";
    case op_intersect_size_gt_val: return os << "intersect_size_gt_val";
    default:
	return os << "illegal-operation";
    }
}

/*! Determine the threshold to reach for @op_intersect_size_gt_val operations
 *
 * The chosen threshold is around three quarters of the shortest length
 * of the two sets, and at least 1. This ensures at least some operation
 * is performed and allows scope to terminate early. It has room for
 * speedup compared to @op_intersect_size, but also may incur slowdown if
 * the overhead for checking against the threshold is too large.
 */
size_t
exceed_threshold( const VID * lb, const VID * le,
		  const VID * rb, const VID * re ) {
    size_t l = std::distance( lb, le );
    size_t r = std::distance( rb, re );
    size_t d = std::min( l, r );
    return std::max( (3*d)/4, (size_t)1 );
}

template<ins_variant>
struct intersect_traits;

template<>
struct intersect_traits<var_merge_scalar> : public graptor::merge_scalar {
    static constexpr bool uses_hash = false;
    static constexpr bool uses_prestudy = false;
};

template<>
struct intersect_traits<var_merge_vector> : public graptor::merge_vector {
    static constexpr bool uses_hash = false;
    static constexpr bool uses_prestudy = false;
};

template<>
struct intersect_traits<var_merge_jump> : public graptor::merge_jump {
    static constexpr bool uses_hash = false;
    static constexpr bool uses_prestudy = false;
};

template<>
struct intersect_traits<var_hash_scalar> : public graptor::hash_scalar {
    static constexpr bool uses_hash = true;
    static constexpr bool uses_prestudy = false;
};

template<>
struct intersect_traits<var_hash_vector> : public graptor::hash_vector {
    static constexpr bool uses_hash = true;
    static constexpr bool uses_prestudy = false;
};

template<>
struct intersect_traits<var_merge_partitioned_scalar>
    : public graptor::merge_partitioned<graptor::merge_scalar> {
    static constexpr bool uses_hash = false;
    static constexpr bool uses_prestudy = true;
};

template<>
struct intersect_traits<var_hash_partitioned_vector>
    : public graptor::merge_partitioned<graptor::hash_vector> {
    static constexpr bool uses_hash = false;
    static constexpr bool uses_prestudy = true;
};

template<>
struct intersect_traits<var_hash_wide> : public graptor::hash_wide {
    static constexpr bool uses_hash = true;
    static constexpr bool uses_prestudy = false;
};


/*! Prestudy data
 */
static size_t levels;
static mmap_ptr<VID> prestudy;

/*! Generic driver method for the various variants and operations.
 *
 * The main purpose of this method and its specialisations is to provide
 * a common interface to call all methods.
 */
template<ins_variant var, operation op>
auto
bench( const VID * lb, const VID * le,
       const VID * rb, const VID * re,
       const graptor::hash_set<VID,hash_fn> & ht,
       const VID * lidx, const VID * ridx,
       VID * out,
       VID x ) {
    using traits = intersect_traits<var>;

    if constexpr ( op == op_intersect ) {
	if constexpr ( traits::uses_hash )
	    return traits::intersect( lb, le, ht, out ) - out;
	else if constexpr ( traits::uses_prestudy )
	    return traits::intersect(
		lb, le, rb, re, ht,
		levels, 0, 1<<levels, lidx, ridx,
		out ) - out;
	else
	    return traits::intersect( lb, le, rb, re, out ) - out;
    } else if constexpr ( op == op_intersect_size ) {
	if constexpr ( traits::uses_hash )
	    return traits::intersect_size( lb, le, ht );
	else if constexpr ( traits::uses_prestudy )
	    return traits::intersect_size(
		lb, le, rb, re, ht,
		levels, 0, 1<<levels, lidx, ridx );
	else
	    return traits::intersect_size( lb, le, rb, re );
    } else if constexpr ( op == op_intersect_size_gt_val ) {
	if constexpr ( traits::uses_hash )
	    return traits::intersect_size_gt_val( lb, le, ht, x );
	else if constexpr ( traits::uses_prestudy )
	    return traits::intersect_size_gt_val(
		lb, le, rb, re, ht,
		levels, 0, 1<<levels, lidx, ridx, x );
	else
	    return traits::intersect_size_gt_val( lb, le, rb, re, x );
    } else {
	assert( 0 && "Invalid operation" );
	return 0;
    }
}

template<ins_variant var, operation op>
double
bench( VID lv, VID rv,
       const VID * lb, const VID * le,
       const VID * rb, const VID * re,
       const graptor::hash_set<VID,hash_fn> & ht,
       const VID * lidx, const VID * ridx,
       VID * out,
       int repeat,
       size_t ref ) {
    size_t x = exceed_threshold( lb, le, rb, re );
    timer tm;
    tm.start();
    for( int r=0; r < repeat; ++r ) {
	size_t sz = bench<var,op>( lb, le, rb, re, ht, lidx, ridx, out, x );
	if( ( sz != ref && op != op_intersect_size_gt_val )
	    || ( !( ( sz <= x && ref <= x ) || ( sz == ref ) )
		 && op == op_intersect_size_gt_val )
	    ) {
	    std::cerr << "FAIL: " << var
		      << ", " << op << ": lv=" << lv << " rv=" << rv
		      << " result: " << sz
		      << " reference: " << ref << "\n";
	    assert( 0 && "WRONG" );
	}
    }
    return tm.stop() / double(repeat);
}

static constexpr size_t MAX_CLASS = 10;

//! Mutually exclusive access to timings
static std::mutex timings_mux;

static graptor::distribution_timing timings[var_N][MAX_CLASS][MAX_CLASS];

void
record( size_t l, size_t r, double tm, ins_variant var, operation op ) {
    size_t cl = ilog2( l ) / 2;
    if( cl >= MAX_CLASS )
	cl = MAX_CLASS-1;
    size_t cr = ilog2( r ) / 2;
    if( cr >= MAX_CLASS )
	cr = MAX_CLASS-1;

    std::lock_guard<std::mutex> g( timings_mux );
    timings[(int)var][cl][cr].add_sample( tm );
}

template<operation op>
void
bench( VID lv, VID rv,
       const VID * lb, const VID * le,
       const VID * rb, const VID * re,
       graptor::hash_set<VID,hash_fn> & ht,
       const VID * lidx, const VID * ridx,
       int repeat ) {
    size_t len = std::min( le - lb, re - rb );
    VID * out = new VID[len];
    
    size_t sz_ref = bench<var_merge_scalar,op_intersect_size>(
	lb, le, rb, re, ht, lidx, ridx, out, 0 );

    double tm = bench<var_merge_scalar,op>(
	lv, rv, lb, le, rb, re, ht, lidx, ridx, out, repeat, sz_ref );
    record( le-lb, re-rb, tm, var_merge_scalar, op );

    tm = bench<var_merge_vector,op>(
	lv, rv, lb, le, rb, re, ht, lidx, ridx, out, repeat, sz_ref );
    record( le-lb, re-rb, tm, var_merge_vector, op );

    tm = bench<var_merge_jump,op>(
	lv, rv, lb, le, rb, re, ht, lidx, ridx, out, repeat, sz_ref );
    record( le-lb, re-rb, tm, var_merge_jump, op );

    tm = bench<var_hash_scalar,op>(
	lv, rv, lb, le, rb, re, ht, lidx, ridx, out, repeat, sz_ref );
    record( le-lb, re-rb, tm, var_hash_scalar, op );

    tm = bench<var_hash_vector,op>(
	lv, rv, lb, le, rb, re, ht, lidx, ridx, out, repeat, sz_ref );
    record( le-lb, re-rb, tm, var_hash_vector, op );

    tm = bench<var_merge_partitioned_scalar,op>(
	lv, rv, lb, le, rb, re, ht, lidx, ridx, out, repeat, sz_ref );
    record( le-lb, re-rb, tm, var_merge_partitioned_scalar, op );

    tm = bench<var_hash_partitioned_vector,op>(
	lv, rv, lb, le, rb, re, ht, lidx, ridx, out, repeat, sz_ref );
    record( le-lb, re-rb, tm, var_hash_partitioned_vector, op );

    tm = bench<var_hash_wide,op>(
	lv, rv, lb, le, rb, re, ht, lidx, ridx, out, repeat, sz_ref );
    record( le-lb, re-rb, tm, var_hash_wide, op );

    delete[] out;
}

void report_difference( int compare0, int compare1 ) {
    if( compare0 != compare1 ) {
	assert( 0 <= compare0 && compare0 < var_N );
	assert( 0 <= compare1 && compare1 < var_N );
	
	std::cerr << "mean difference " << (ins_variant)compare0
		  << " and " << (ins_variant)compare1 << ":\n";
	for( size_t l=0; l < MAX_CLASS; ++l )
	    for( size_t r=0; r < MAX_CLASS; ++r )
		std::cerr
		    << l << "," << r << ": "
		    << graptor::characterize_mean_difference<double>(
			timings[compare0][l][r].samples,
			timings[compare1][l][r].samples,
			0.95, 1000, 100 )
		    << '\n';
    }
}

int main( int argc, char *argv[] ) {
    commandLine P( argc, argv, " help" );
    bool symmetric = P.getOptionValue("-s");
    int repetitions = P.getOptionLongValue("-r", 3);
    VID u_min_size = P.getOptionLongValue("--u-min-size", 0);
    VID v_min_size = P.getOptionLongValue("--v-min-size", 0);
    VID min_size = P.getOptionLongValue("--min-size", 0);
    levels = P.getOptionLongValue("--levels", 3);
    const char * ifile = P.getOptionValue( "-i" );

    if( min_size > 0 )
	u_min_size = v_min_size = min_size;

    timer tm;
    tm.start();

    GraphCSx G( ifile, -1, symmetric );

    std::cerr << "Reading graph: " << tm.next() << "\n";

    VID n = G.numVertices();
    EID m = G.numEdges();
    const EID * const index = G.getIndex();
    const VID * const edges = G.getEdges();

    assert( G.isSymmetric() );
    std::cerr << "Undirected graph: n=" << n << " m=" << m << std::endl;

    graptor::graph::GraphHAdjTable<VID,EID,hash_fn> H( n );
    
    parallel_loop( (VID)0, n, [&]( VID v ) {
	auto & adj = H.get_adjacency( v );
	EID ee = index[v+1];
	for( EID e=index[v]; e != ee; ++e ) {
	    VID u = edges[e];
	    adj.insert( u );
	}
    } );

    std::cerr << "Building hashed graph: " << tm.next() << "\n";

    prestudy.allocate( n * ( 1 + ( 1 << levels ) ),
		       numa_allocation_interleaved() );
    parallel_loop( VID(0), n, [&]( VID v ) {
	graptor::merge_partitioned<void>::prestudy(
	    &edges[index[v]], &edges[index[v+1]],
	    n, levels, &prestudy[v*(1+(1<<levels))] );
    } );
    std::cerr << "Building prestudy: " << tm.next() << "\n";

    std::cerr << "Configuration:"
	      << "\n  repetitions: " << repetitions
	      << "\n  operation: " << (operation)OPERATION
	      << "\n  v_min_size: " << v_min_size
	      << "\n  u_min_size: " << u_min_size
	      << "\n  min_size: " << min_size
	      << "\n  levels: " << levels
	      << "\n";

    parallel_loop( VID(0), n, [&]( VID v ) {
	// Filter size of adjacency lists
	if( index[v+1] - index[v] >= v_min_size ) {
	    for( EID e=index[v], ee=index[v+1]; e != ee; ++e ) {
		VID u = edges[e];

		// Filter size of adjacency lists
		if( index[u+1] - index[u] < u_min_size )
		    continue;

		// Benchmark intersection operations between neighbour lists of
		// u and v
#if OPERATION == 0
		bench<op_intersect>(
		    v, u,
		    &edges[index[v]], &edges[index[v+1]],
		    &edges[index[u]], &edges[index[u+1]],
		    H.get_adjacency( u ),
		    &prestudy[v*(1+(1<<levels))],
		    &prestudy[u*(1+(1<<levels))],
		    repetitions );
#elif OPERATION == 1
		bench<op_intersect_size>(
		    v, u,
		    &edges[index[v]], &edges[index[v+1]],
		    &edges[index[u]], &edges[index[u+1]],
		    H.get_adjacency( u ),
		    &prestudy[v*(1+(1<<levels))],
		    &prestudy[u*(1+(1<<levels))],
		    repetitions );
#elif OPERATION == 2
		bench<op_intersect_size_gt_val>(
		    v, u,
		    &edges[index[v]], &edges[index[v+1]],
		    &edges[index[u]], &edges[index[u+1]],
		    H.get_adjacency( u ),
		    &prestudy[v*(1+(1<<levels))],
		    &prestudy[u*(1+(1<<levels))],
		    repetitions );
#else
#error "invalid value for OPERATION (0,1,2)"
#endif
	    }
	}
    } );

    std::cerr << "Binning: bin(list) is ilog2(list length)/2\n";
    std::cerr << "         E.g.: bin(4)=" << ilog2(4)/2 << "\n";
    std::cerr << "         E.g.: bin(8)=" << ilog2(8)/2 << "\n";
    std::cerr << "         E.g.: bin(15)=" << ilog2(15)/2 << "\n";
    std::cerr << "         E.g.: bin(16)=" << ilog2(16)/2 << "\n";
    std::cerr << "Results:\n";

    for( int var=0; var < var_N; ++var ) {
	std::cerr << "Results for variant " << (ins_variant)var << ":\n";
	for( size_t l=0; l < MAX_CLASS; ++l )
	    for( size_t r=0; r < MAX_CLASS; ++r )
		std::cerr
		    << l << "," << r << ": "
		    << timings[var][l][r].characterize( .95, 1000, 100 )
		    << '\n';
    }

    report_difference( var_hash_vector, var_merge_vector );
    report_difference( var_hash_vector, var_hash_wide );

    G.del();

    return 0;
}
