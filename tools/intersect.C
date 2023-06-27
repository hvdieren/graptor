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
#include <random>
#include <thread>

#include <time.h>

#include "graptor/graptor.h"
#include "graptor/api.h"

#include "graptor/graph/simple/hash_table.h"
#include "graptor/graph/simple/hadjt.h"
#include "graptor/container/intersect.h"

template<typename T>
struct java_hash;

template<>
struct java_hash<uint32_t> {
    using type = uint32_t;

    explicit java_hash( uint32_t log_size ) { }

    void resize( uint32_t ) { }

    type operator() ( uint32_t h ) const {
	h ^= (h >> 20) ^ (h >> 12);
	return h ^ (h >> 7) ^ (h >> 4);
    }

    template<unsigned short VL>
    typename vector_type_traits_vl<uint32_t,VL>::type
    vectorized( typename vector_type_traits_vl<uint32_t,VL>::type h ) const {
	using tr = vector_type_traits_vl<type,VL>;
	using vtype = typename tr::type;

	vtype h20 = tr::srli( h, 20 );
	vtype h12 = tr::srli( h, 12 );
	h = tr::bitwise_xor( h20, tr::bitwise_xor( h, h12 ) );
	vtype h7 = tr::srli( h, 7 );
	vtype h4 = tr::srli( h, 4 );
	h = tr::bitwise_xor( h7, tr::bitwise_xor( h, h4 ) );
	return h;
    }
};

template<typename T>
struct rand_hash;

template<>
struct rand_hash<uint32_t> {
    // Same RNG as Blanusa's code.
    using type = uint32_t;

    explicit rand_hash( uint32_t log_size ) {
	resize( log_size );
    }

    void resize( uint32_t log_size ) {
	m_shift = 32 - log_size - 1;
	m_a = rand() | 1;
	m_b = rand() & ((uint32_t(1) << m_shift) - 1);
    }

    type operator() ( uint32_t h ) const {
	h = h * m_a + m_b;
	return h >> m_shift;
    }

    template<unsigned short VL>
    typename vector_type_traits_vl<uint32_t,VL>::type
    vectorized( typename vector_type_traits_vl<uint32_t,VL>::type h ) const {
	using tr = vector_type_traits_vl<type,VL>;
	using vtype = typename tr::type;

	vtype a_vec = tr::set1( m_a );
	vtype b_vec = tr::set1( m_b );
	vtype c = tr::mul( a_vec, h );
	vtype d = tr::add( c, b_vec );
	vtype e = tr::srli( d, m_shift );
	return e;
    }
    
private:
    uint32_t rand() {
	static thread_local mt19937* generator = nullptr;
	if( !generator ) {
	    pthread_t self = pthread_self();
	    generator = new mt19937( clock() + self );
	}
	uniform_int_distribution<int> distribution;
	return distribution( *generator );
    }

private:
    uint32_t m_a, m_b;
    uint32_t m_shift;
};

using hash_fn = rand_hash<uint32_t>;

/*! Enumeration of algorithmic variants
 */
enum variant {
    var_merge_scalar = 0,	/**< merge-based, scalar */
    var_merge_vector = 1,	/**< merge-based, vector */
    var_merge_jump = 2,	/**< merge-based, scalar and jumping ahead */
    var_hash_scalar = 3,	/**< hash-based, scalar */
    var_hash_vector = 4,	/**< hash-based, vector */
    var_N = 5 	 	/**< number of options */
};

/*! Enumeration of operation types
 */
enum operation {
    op_intersect = 0,		/**< intersection producing list */
    op_intersect_size = 1,	/**< intersection size */
    op_intersect_size_exceed = 2,	/**< intersection size or abort */
    op_N = 3			/**< number of options */
};

/*! Print an algorithmic variant in human-readable form
 */
std::ostream & operator << ( ostream & os, variant v ) {
    switch( v ) {
    case var_merge_scalar: return os << "merge_scalar";
    case var_merge_vector: return os << "merge_vector";
    case var_merge_jump: return os << "merge_jump";
    case var_hash_scalar: return os << "hash_scalar";
    case var_hash_vector: return os << "hash_vector";
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
    case op_intersect_size_exceed: return os << "intersect_size_exceed";
    default:
	return os << "illegal-operation";
    }
}

/*! Determine the threshold to reach for @op_intersect_size_exceed operations
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
    size_t d = std::min( l ,r );
    return std::max( (3*d)/4, (size_t)1 );
}

/*! Generic driver method for the various variants and operations.
 *
 * The main purpose of this method and its specialisations is to provide
 * a common interface to call all methods.
 */
template<variant, operation>
size_t
bench( const VID * lb, const VID * le,
       const VID * rb, const VID * re,
       const graptor::graph::hash_table<VID,hash_fn> & ht,
       VID * out );

template<>
size_t
bench<var_merge_scalar,op_intersect>(
    const VID * lb, const VID * le,
    const VID * rb, const VID * re,
    const graptor::graph::hash_table<VID,hash_fn> & ht,
    VID * out ) {
    return graptor::merge_scalar::intersect( lb, le, rb, re, out ) - out;
}

template<>
size_t
bench<var_merge_scalar,op_intersect_size>(
    const VID * lb, const VID * le,
    const VID * rb, const VID * re,
    const graptor::graph::hash_table<VID,hash_fn> & ht,
    VID * out ) {
    return graptor::merge_scalar::intersect_size( lb, le, rb, re );
}

template<>
size_t
bench<var_merge_scalar,op_intersect_size_exceed>(
    const VID * lb, const VID * le,
    const VID * rb, const VID * re,
    const graptor::graph::hash_table<VID,hash_fn> & ht,
    VID * out ) {
    size_t x = exceed_threshold( lb, le, rb, re );
    return graptor::merge_scalar::intersect_size_exceed( lb, le, rb, re, x );
}

template<>
size_t
bench<var_merge_vector,op_intersect>(
    const VID * lb, const VID * le,
    const VID * rb, const VID * re,
    const graptor::graph::hash_table<VID,hash_fn> & ht,
    VID * out ) {
    return graptor::merge_vector::intersect( lb, le, rb, re, out ) - out;
}

template<>
size_t
bench<var_merge_vector,op_intersect_size>(
    const VID * lb, const VID * le,
    const VID * rb, const VID * re,
    const graptor::graph::hash_table<VID,hash_fn> & ht,
    VID * out ) {
    return graptor::merge_vector::intersect_size( lb, le, rb, re );
}

template<>
size_t
bench<var_merge_vector,op_intersect_size_exceed>(
    const VID * lb, const VID * le,
    const VID * rb, const VID * re,
    const graptor::graph::hash_table<VID,hash_fn> & ht,
    VID * out ) {
    size_t x = exceed_threshold( lb, le, rb, re );
    return graptor::merge_vector::intersect_size_exceed( lb, le, rb, re, x );
}

template<>
size_t
bench<var_merge_jump,op_intersect>(
    const VID * lb, const VID * le,
    const VID * rb, const VID * re,
    const graptor::graph::hash_table<VID,hash_fn> & ht,
    VID * out ) {
    return graptor::merge_jump::intersect( lb, le, rb, re, out ) - out;
}

template<>
size_t
bench<var_merge_jump,op_intersect_size>(
    const VID * lb, const VID * le,
    const VID * rb, const VID * re,
    const graptor::graph::hash_table<VID,hash_fn> & ht,
    VID * out ) {
    return graptor::merge_jump::intersect_size( lb, le, rb, re );
}

template<>
size_t
bench<var_merge_jump,op_intersect_size_exceed>(
    const VID * lb, const VID * le,
    const VID * rb, const VID * re,
    const graptor::graph::hash_table<VID,hash_fn> & ht,
    VID * out ) {
    size_t x = exceed_threshold( lb, le, rb, re );
    return graptor::merge_jump::intersect_size_exceed( lb, le, rb, re, x );
}

template<>
size_t
bench<var_hash_scalar,op_intersect>(
    const VID * lb, const VID * le,
    const VID * rb, const VID * re,
    const graptor::graph::hash_table<VID,hash_fn> & ht,
    VID * out ) {
    return graptor::hash_scalar::intersect( lb, le, ht, out ) - out;
}

template<>
size_t
bench<var_hash_scalar,op_intersect_size>(
    const VID * lb, const VID * le,
    const VID * rb, const VID * re,
    const graptor::graph::hash_table<VID,hash_fn> & ht,
    VID * out ) {
    return graptor::hash_scalar::intersect_size( lb, le, ht );
}

template<>
size_t
bench<var_hash_scalar,op_intersect_size_exceed>(
    const VID * lb, const VID * le,
    const VID * rb, const VID * re,
    const graptor::graph::hash_table<VID,hash_fn> & ht,
    VID * out ) {
    size_t x = exceed_threshold( lb, le, rb, re );
    return graptor::hash_scalar::intersect_size_exceed( lb, le, ht, x );
}

template<>
size_t
bench<var_hash_vector,op_intersect>(
    const VID * lb, const VID * le,
    const VID * rb, const VID * re,
    const graptor::graph::hash_table<VID,hash_fn> & ht,
    VID * out ) {
    return graptor::hash_vector::intersect( lb, le, ht, out ) - out;
}

template<>
size_t
bench<var_hash_vector,op_intersect_size>(
    const VID * lb, const VID * le,
    const VID * rb, const VID * re,
    const graptor::graph::hash_table<VID,hash_fn> & ht,
    VID * out ) {
    return graptor::hash_vector::intersect_size( lb, le, ht );
}

template<>
size_t
bench<var_hash_vector,op_intersect_size_exceed>(
    const VID * lb, const VID * le,
    const VID * rb, const VID * re,
    const graptor::graph::hash_table<VID,hash_fn> & ht,
    VID * out ) {
    size_t x = exceed_threshold( lb, le, rb, re );
    return graptor::hash_vector::intersect_size_exceed( lb, le, ht, x );
}

template<variant var, operation op>
double
bench( const VID * lb, const VID * le,
       const VID * rb, const VID * re,
       const graptor::graph::hash_table<VID,hash_fn> & ht,
       VID * out,
       int repeat,
       size_t ref ) {
    timer tm;
    tm.start();
    for( int r=0; r < repeat; ++r ) {
	size_t sz = bench<var,op>( lb, le, rb, re, ht, out );
	if( ( sz != ref && op != op_intersect_size_exceed )
	    || ( sz != ref && sz != 0 && op == op_intersect_size_exceed )
	    ) {
	    std::cerr << "FAIL: " << var
		      << ", " << op << ": result: " << sz
		      << " reference: " << ref << "\n";
	    assert( 0 && "WRONG" );
	}
    }
    return tm.stop() / double(repeat);
}

static constexpr size_t MAX_CLASS = 10;

struct timing {
    size_t cnt;
    double tm;
};

timing timings[var_N][MAX_CLASS][MAX_CLASS];

void
record( size_t l, size_t r, double tm, variant var, operation op ) {
    size_t cl = ilog2( l ) / 2;
    if( cl >= MAX_CLASS )
	cl = MAX_CLASS-1;
    size_t cr = ilog2( r ) / 2;
    if( cr >= MAX_CLASS )
	cr = MAX_CLASS-1;

    timings[(int)var][cl][cr].tm += tm;
    timings[(int)var][cl][cr].cnt++;
}

template<operation op>
void
bench( const VID * lb, const VID * le,
       const VID * rb, const VID * re,
       graptor::graph::hash_table<VID,hash_fn> & ht,
       int repeat ) {
    size_t len = std::min( le - lb, re - rb );
    VID * out = new VID[len];
    
    size_t sz_ref = bench<var_merge_scalar,op_intersect_size>(
	lb, le, rb, re, ht, out );

    double tm = bench<var_merge_scalar,op>(
	lb, le, rb, re, ht, out, repeat, sz_ref );
    record( le-lb, re-rb, tm, var_merge_scalar, op );

    tm = bench<var_merge_vector,op>(
	lb, le, rb, re, ht, out, repeat, sz_ref );
    record( le-lb, re-rb, tm, var_merge_vector, op );

    tm = bench<var_merge_jump,op>(
	lb, le, rb, re, ht, out, repeat, sz_ref );
    record( le-lb, re-rb, tm, var_merge_jump, op );

    tm = bench<var_hash_scalar,op>(
	lb, le, rb, re, ht, out, repeat, sz_ref );
    record( le-lb, re-rb, tm, var_hash_scalar, op );

    tm = bench<var_hash_vector,op>(
	lb, le, rb, re, ht, out, repeat, sz_ref );
    record( le-lb, re-rb, tm, var_hash_vector, op );

    delete[] out;
}

int main( int argc, char *argv[] ) {
    commandLine P( argc, argv, " help" );
    bool symmetric = P.getOptionValue("-s");
    int repetitions = P.getOptionLongValue("-r", 3);
    VID u_min_length = P.getOptionLongValue("--u-min-length", 0);
    VID v_min_length = P.getOptionLongValue("--v-min-length", 0);
    VID min_length = P.getOptionLongValue("--min-length", 0);
    const char * ifile = P.getOptionValue( "-i" );

    if( min_length > 0 )
	u_min_length = v_min_length = min_length;

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

    for( int var=0; var < var_N; ++var ) {
	for( size_t l=0; l < MAX_CLASS; ++l )
	    for( size_t r=0; r < MAX_CLASS; ++r ) {
		timings[(int)var][l][r].cnt = 0;
		timings[(int)var][l][r].tm = 0;
	    }
    }

    std::cerr << "Configuration:"
	      << "\n  repetitions: " << repetitions
	      << "\n  operation: " << (operation)OPERATION
	      << "\n  v_min_length: " << v_min_length
	      << "\n  u_min_length: " << u_min_length
	      << "\n  min_length: " << min_length
	      << "\n";

    parallel_loop( VID(0), n, [&]( VID v ) {
	// Filter length of adjacency lists
	if( index[v+1] - index[v] >= v_min_length ) {
	    for( EID e=index[v], ee=index[v+1]; e != ee; ++e ) {
		VID u = edges[e];

		// Filter length of adjacency lists
		if( index[u+1] - index[u] < u_min_length )
		    continue;

		// Benchmark intersection operations between neighbour lists of
		// u and v
#if OPERATION == 0
		bench<op_intersect>(
		    &edges[index[v]], &edges[index[v+1]],
		    &edges[index[u]], &edges[index[u+1]],
		    H.get_adjacency( u ), repetitions );
#elif OPERATION == 1
		bench<op_intersect_size>(
		    &edges[index[v]], &edges[index[v+1]],
		    &edges[index[u]], &edges[index[u+1]],
		    H.get_adjacency( u ), repetitions );
#elif OPERATION == 2
		bench<op_intersect_size_exceed>(
		    &edges[index[v]], &edges[index[v+1]],
		    &edges[index[u]], &edges[index[u+1]],
		    H.get_adjacency( u ), repetitions );
#else
#error "invalid value for OPERATION (0,1,2)"
#endif
	    }
	}
    } );

    std::cerr << "Results:\n";

    for( int var=0; var < var_N; ++var ) {
	std::cerr << "Results for variant " << (variant)var << ":\n";
	for( size_t l=0; l < MAX_CLASS; ++l )
	    for( size_t r=0; r < MAX_CLASS; ++r )
		std::cerr << l << "," << r << ": "
			  << timings[var][l][r].cnt << ' '
			  << timings[var][l][r].tm << ' '
			  << (timings[var][l][r].tm
			      / double(timings[var][l][r].cnt))
			  << '\n';
    }

    G.del();

    return 0;
}
