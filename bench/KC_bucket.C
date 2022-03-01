#include <math.h>
#include <fstream>
#include <limits>

#include "graptor/graptor.h"
#include "graptor/api.h"
#include "buckets.h"
#include "check.h"

using expr::_0;
using expr::_1;
using expr::_1s;

// By default set options for highest performance
#ifndef FUSION
#define FUSION 1
#endif

enum variable_name {
    var_coreness = 0,
    var_let = 1,
    var_ngh = 2,
    var_degrees_ro = expr::aid_graph_degree
};

struct bucket_fn {
    using ID = VID;
    
    bucket_fn( VID * degree, VID * K )
	: m_degree( degree ), m_K( K ) { }

    VID operator() ( VID v ) const {
	return v == ~(VID)0 || m_degree[v] < *m_K ? ~(VID)0 : m_degree[v];
    }
    
private:
    VID * m_degree;
    VID * m_K;
};

template <class GraphType>
class KCv {
public:
    KCv( GraphType & _GA, commandLine & P )
	: GA( _GA ),
	  num_buckets( P.getOptionLongValue( "-kc:buckets", 127 ) ),
	  coreness( GA.get_partitioner(), "count-down degrees / coreness" ),
	  info_buf( 60 ) {
	itimes = P.getOption( "-itimes" );
	debug = P.getOption( "-debug" );
	outfile = P.getOptionValue( "-kc:outfile" );

	static bool once = false;
	if( !once ) {
	    once = true;
	    std::cerr << "FUSION=" << FUSION << "\n";
	    std::cerr << "num_buckets=" << num_buckets << "\n";
	}
    }
    ~KCv() {
	coreness.del();
    }

    struct info {
	double delay;
	float density;
	VID F_act, rm_act, wakeup_act;
	VID K;

	void dump( int i ) {
	    std::cerr << "Iteration " << i << ": " << delay
		      << " density: " << density
		      << " F_act: " << F_act
		      << " rm_act: " << rm_act
		      << " wakeup_act: " << wakeup_act
		      << " K: " << K << "\n";
	}
    };

    struct stat { };

    void run() {
	const partitioner &part = GA.get_partitioner();
	VID n = GA.numVertices();
	EID m = GA.numEdges();

	expr::array_ro<VID, VID, var_degrees_ro> a_degrees_ro(
	    const_cast<VID *>( GA.getCSR().getDegree() ) );

	// Initialise arrays
	frontier ftrue = frontier::all_true( n, m );
	frontier nonzero;
	make_lazy_executor( part )
	    .vertex_filter(
		GA, 	 	 	// graph
		ftrue,
		nonzero,  	// record new frontier
		[&]( auto v ) {
		    return expr::make_seq(
			coreness[v] = a_degrees_ro[v],
			a_degrees_ro[v] > _0 );
		} )
	    .materialize();

	VID K = 0;

	// Create bucket structure
	buckets<VID,bucket_fn>
	    bkts( n, num_buckets, bucket_fn( coreness.get_ptr(), &K ) );

	// Place each vertex in the bucket corresponding with its degree
	// ... in parallel
	bkts.update_buckets( part, nonzero );
	VID todo = nonzero.nActiveVertices();
	nonzero.del();

	largestCore = 0;
	iter = 0;
#if FUSION
	while( !bkts.empty() ) // cannot count todo as needed with fusion
#else
	while( todo > 0 ) // iterate until all vertices visited
#endif
	{

#if !FUSION
	    assert( !bkts.empty() );
#endif
	
	    timer tm_iter;
	    tm_iter.start();

	    // next_bucket updates current bucket, so do this first...
	    frontier F = bkts.next_bucket();
	    K = bkts.get_current_bucket();
	    VID overflow_bkt = bkts.get_overflow_bucket();
	    assert( K >= largestCore );

	    // We don't need to know how many edges there are on F since we
	    // only use it in a vertex-filter operation. Simply record
	    // non-zero.
	    F.setActiveCounts( F.nActiveVertices(), F.nActiveVertices() );

	    // std::cerr << "get buckets: " << tm_iter.next() << "\n";

	    // All vertices in bucket are removed, have coreness K
	    // Watch out for duplicates in the buckets, as moved vertices
	    // are not removed from their previous bucket.
	    frontier unique;
	    make_lazy_executor( part )
		.vertex_filter(
		    GA, F, unique,
		    [&]( auto v ) {
			auto cK = expr::constant_val( coreness[v], K );
			return coreness[v] >= cK;
		    } )
		.materialize();

	    // Remove duplicate edges. Edges may be multiply represented
	    // as they may be inserted once per edge in the worst case.
	    // This is a consequence of not removing vertices from a bucket
	    // when they move to a different bucket.
	    // In principle each vertex should occur in each bucket once,
	    // however, they may appear multiple times in the overflow bucket.
	    // Note: we have avoided the need to remove duplicates *by design*.
	    if( 0 && !unique.isEmpty() ) {
		frontier Fnew
		    = frontier::sparse( GA.numVertices(), unique.nActiveVertices() );
		removeDuplicates( unique.getSparse(), unique.nActiveVertices(), part );
		VID nv_new = sequence::filter(
		    unique.getSparse(), Fnew.getSparse(), unique.nActiveVertices(),
		    ValidVID() );
		Fnew.setActiveCounts( nv_new, nv_new );
		VID diff = unique.nActiveVertices() - nv_new;
		std::cerr << "removed " << diff << " duplicates\n";
		assert( diff == 0 );
		unique.del();
		unique = Fnew;
	    }

	    // std::cerr << "remove duplicates: " << tm_iter.next() << "\n";

	    // std::cerr << "F filtered: " << F.nActiveVertices() << "\n";
	    // std::cerr << "F: " << F << "\n";

	    if( !unique.isEmpty() )
		largestCore = K;

	    // std::cerr << "filter completed vertices: " << tm_iter.next() << "\n";

	    // std::cerr << "K: " << K << "\n";
	    // std::cerr << "overflow_bkt: " << overflow_bkt << "\n";
	    // std::cerr << "F     : " << F << "\n";
	    // std::cerr << "unique: " << unique << "\n";
	    // print( std::cerr, part, coreness );

#if FUSION
	    frontier output;
	    api::edgemap(
		GA,
		api::config( api::always_sparse ),
		api::filter( api::src, api::strong, unique ),
		api::record( output, api::reduction, api::strong ),
		api::fusion( [&]( auto v ) {
		    // Requires that RHS of && is evaluated after LHS.
		    auto cK = expr::constant_val( coreness[v], K );
		    auto cO = expr::constant_val( coreness[v], overflow_bkt );
		    // return coreness[v].count_down( cK );
		    return expr::let<var_let>(
			coreness[v].count_down_value( cK ),
			[&]( auto old ) {
			    return expr::cast<int>(
				expr::iif(
				    old == cK + _1,
				    expr::iif( old > cO,
					       _0, // degree below overflow, move
					       _1s ), // in overflow bucket, stay
				    _1 ) ); // degree dropped to K
			} );
		} ),
		api::relax( [&]( auto s, auto d, auto e ) {
		    auto cK = expr::constant_val( coreness[s], K );
		    return coreness[d] > cK;
		} )
		)
		.materialize();
#else
	    frontier output;
	    api::edgemap(
		GA,
		api::filter( api::src, api::strong, unique ),
		api::record( output, api::reduction, api::strong ),
		api::relax( [&]( auto s, auto d, auto e ) {
		    // Note: constant_val copies over the mask of s
		    auto cK = expr::constant_val( s, K );
		    // We use count_down to K (and never lower) because
		    // there may be situations where a vertex is sitting at
		    // degree K+1 and has two or more neighbours whose degree
		    // is K. Processing those neighbours concurrently may
		    // push down the degree of the vertex to K-1, however, the
		    // coreness should still be K. Hence, we use the count_down
		    // primitive. This is the only way to ensure atomicity
		    // of the check-larger-than-K-and-subtract operation.
		    return coreness[d].count_down( cK );
		} )
		)
		.materialize();
#endif

	    // std::cerr << "unique: " << unique.nActiveVertices() << "\n";

	    todo -= unique.nActiveVertices();
	    unique.del();

	    // std::cerr << "edgemap: " << tm_iter.next() << "\n";

	    // std::cerr << "output: " << output << "\n";
	    // print( std::cerr, part, coreness );
	    // std::cerr << "todo: " << todo << "\n";

	    bkts.update_buckets( part, output );

	    // std::cerr << "update buckets: " << tm_iter.next() << "\n";

	    if( itimes ) {
		info_buf.resize( iter+1 );
		info_buf[iter].density = unique.density( GA.numEdges() );
		info_buf[iter].delay = tm_iter.total();
		info_buf[iter].F_act = F.nActiveVertices();
		info_buf[iter].rm_act = unique.nActiveVertices();
		info_buf[iter].wakeup_act = output.nActiveVertices();
		info_buf[iter].K = K;
		if( debug )
		    info_buf[iter].dump( iter );
		++iter;
	    }

	    output.del();
	    F.del();
	}
    }

    void post_process( stat & stat_buf ) {
	std::cout << "Largest core: " << largestCore << "\n";

	if( itimes ) {
	    for( int i=0; i < iter; ++i )
		info_buf[i].dump( i );
	}

	if( outfile )
	    writefile( GA, outfile, coreness.get_ptr() );
    }

    void validate( stat & ) {
	const partitioner &part = GA.get_partitioner();
	VID n = GA.numVertices();
	EID m = GA.numEdges();

	api::vertexprop<VID,VID,var_ngh>
	    neighbours( part, "coreness neighbour count" );

	expr::array_ro<VID, VID, var_degrees_ro> a_degrees(
	    const_cast<VID *>( GA.getCSR().getDegree() ) );

	make_lazy_executor( part )
	    .vertex_map(
		[&]( auto v ) { return neighbours[v] = _0; }
		).materialize();
	
	frontier ftrue = frontier::all_true( n, m );
	frontier F;

	make_lazy_executor( part )
	    .vertex_filter(
		GA, ftrue, F,
		[&]( auto v ) { return coreness[v] > a_degrees[v]; }
		).materialize();
	if( F.nActiveVertices() != 0 ) {
	    std::cerr << "Validation failed: " << F.nActiveVertices()
		      << " vertices with coreness higher than degree\n";
	    // std::cerr << F << "\n";
	    // print( std::cerr, part, neighbours );
	    // print( std::cerr, part, coreness );

	    if( F.getType() == frontier_type::ft_logical4 ) {
		logical<4> *p = F.getDense<frontier_type::ft_logical4>();
		for( VID v=0; v < n; ++v ) {
		    if( p[v] )
			std::cerr << "v=" << v << " d[v]=" << a_degrees[v]
				  << " c[v]=" << coreness[v] << "\n";
		}
	    }
	    
	    abort();
	}
	F.del();

	make_lazy_executor( part )
	    .vertex_filter(
		GA, ftrue, F,
		[&]( auto v ) {
		    return coreness[v] == _0 && a_degrees[v] != _0;
		} ).materialize();
	if( F.nActiveVertices() != 0 ) {
	    std::cerr << "Validation failed: " << F.nActiveVertices()
		      << " vertices with zero coreness and non-zero degree, "
		      << " or not initialised\n";
	    // std::cerr << F << "\n";
	    // print( std::cerr, part, neighbours );
	    // print( std::cerr, part, coreness );

	    if( F.getType() == frontier_type::ft_logical4 ) {
		logical<4> *p = F.getDense<frontier_type::ft_logical4>();
		for( VID v=0; v < n; ++v ) {
		    if( p[v] )
			std::cerr << "v=" << v << " d[v]=" << a_degrees[v]
				  << " c[v]=" << coreness[v] << "\n";
		}
	    }
	    
	    abort();
	}
	F.del();
	
	api::edgemap( GA,
		      api::relax( [&]( auto s, auto d, auto e ) {
			  return neighbours[d]
			      += expr::iif( coreness[s] >= coreness[d], _0, _1 );
		      } )
	    )
	    .vertex_filter( GA, ftrue, F,
			    [&]( auto v ) {
				return neighbours[v] + _1 < coreness[v];
			    } )
	    .materialize();

	if( F.nActiveVertices() != 0 ) {
	    std::cerr << "Validation failed: " << F.nActiveVertices()
		      << " vertices with too few neighbours\n";
	    // std::cerr << F << "\n";
	    // print( std::cerr, part, neighbours );
	    // print( std::cerr, part, coreness );

	    if( F.getType() == frontier_type::ft_logical4 ) {
		logical<4> *p = F.getDense<frontier_type::ft_logical4>();
		for( VID v=0; v < n; ++v ) {
		    if( p[v] ) {
			std::cerr << "v=" << v << " n[v]=" << neighbours[v]
				  << " c[v]=" << coreness[v];
			VID d = GA.getCSR().getDegree( v );
			for( VID i=0; i < d; ++i ) {
			    VID u = GA.getCSR().getNeighbor( v, i );
			    std::cerr << ' ' << u << '=' << coreness[u];
			}
			std::cerr << "\n";
		    }
		}
	    }
	    
	    abort();
	}
	F.del();

	make_lazy_executor( part )
	    .vertex_map( [&]( auto v ) { return neighbours[v] = _0; } )
	    .materialize();

	api::edgemap( GA,
		      api::relax( [&]( auto s, auto d, auto e ) {
			  return neighbours[d]
			      += expr::iif( coreness[s] > coreness[d], _0, _1 );
		      } )
	    )
	    .vertex_filter( GA, ftrue, F,
			    [&]( auto v ) {
				return neighbours[v] > coreness[v];
			    } )
	    .materialize();

	if( F.nActiveVertices() != 0 ) {
	    std::cerr << "Validation failed: " << F.nActiveVertices()
		      << " vertices with too low k-core number\n";

	    if( F.getType() == frontier_type::ft_logical4 ) {
		logical<4> *p = F.getDense<frontier_type::ft_logical4>();
		for( VID v=0; v < n; ++v ) {
		    if( p[v] ) {
			std::cerr << "v=" << v << " n[v]=" << neighbours[v]
				  << " c[v]=" << coreness[v];
			VID d = GA.getCSR().getDegree( v );
			for( VID i=0; i < d; ++i ) {
			    VID u = GA.getCSR().getNeighbor( v, i );
			    std::cerr << ' ' << u << '=' << coreness[u];
			}
			std::cerr << "\n";
		    }
		}
	    }

	    abort();
	} else {
	    std::cerr << "Validation successful\n";
	}
	
	F.del();
	neighbours.del();
    }

    static void report( const std::vector<stat> & stat_buf ) { }

private:
    const GraphType & GA;
    bool itimes, debug;
    int num_buckets;
    const char * outfile;
    int iter;
    VID largestCore;
    api::vertexprop<VID,VID,var_coreness> coreness;
    std::vector<info> info_buf;
};

template <class GraphType>
using Benchmark = KCv<GraphType>;

#include "driver.C"
