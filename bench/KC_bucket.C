#include <math.h>
#include <fstream>
#include <limits>

#include "graptor/graptor.h"
#include "graptor/api.h"
#include "buckets.h"
#include "check.h"

using expr::_0;
using expr::_1;

// By default set options for highest performance
#ifndef FUSION
#define FUSION 1
#endif

enum variable_name {
    var_degrees = 0,
    var_coreness = 1,
    var_degrees_ro = expr::aid_graph_degree,
    var_let = 2,
    var_ngh = 3
};

struct bucket_fn {
    using ID = VID;
    
    bucket_fn( VID * degree, VID * coreness )
	: m_degree( degree ), m_coreness( coreness ) { }

    VID operator() ( VID v ) const {
	if( m_coreness[v] != 0 )
	    return ~(VID)0;
	else
	    return m_degree[v];
    }

    VID get( VID v ) const {
	return m_degree[v];
    }
    
private:
    VID * m_degree, * m_coreness;
};

template <class GraphType>
class KCv {
public:
    KCv( GraphType & _GA, commandLine & P )
	: GA( _GA ),
	  num_buckets( P.getOptionLongValue( "-kc:buckets", 127 ) ),
	  coreness( GA.get_partitioner(), "coreness" ),
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
	VID rm_act;
	VID K;

	void dump( int i ) {
	    std::cerr << "Iteration " << i << ": " << delay
		      << " density: " << density
		      << " rm_act: " << rm_act << " K: " << K << "\n";
	}
    };

    struct stat { };

    void run() {
	const partitioner &part = GA.get_partitioner();
	VID n = GA.numVertices();
	EID m = GA.numEdges();

	api::vertexprop<VID,VID,var_degrees>
	    degrees( part, "count-down degrees" );

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
			coreness[v] = _0,
			degrees[v] = a_degrees_ro[v],
			degrees[v] > _0 );
		} )
	    .materialize();

	// Create bucket structure
	buckets<VID,bucket_fn>
	    bkts( n, num_buckets,
		  bucket_fn( degrees.get_ptr(), coreness.get_ptr() ) );

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
	    VID K = bkts.get_current_bucket();
	    assert( K >= largestCore );

	    // We don't need to know how many edges there are on F since we
	    // only use it in a vertex-filter operation. Simply record
	    // non-zero.
	    F.setActiveCounts( F.nActiveVertices(), F.nActiveVertices() );

	    // Remove duplicate edges. Edges may be multiply represented
	    // as they may be inserted once per edge in the worst case.
	    // This is a consequence of not removing vertices from a bucket
	    // when they move to a different bucket.
	    if( !F.isEmpty() ) {
		frontier Fnew
		    = frontier::sparse( GA.numVertices(), F.nActiveVertices() );
		removeDuplicates( F.getSparse(), F.nActiveVertices(), part );
		VID nv_new = sequence::filter(
		    F.getSparse(), Fnew.getSparse(), F.nActiveVertices(),
		    ValidVID() );
		Fnew.setActiveCounts( nv_new, nv_new );
		F.del();
		F = Fnew;
	    }

	    // std::cerr << "F filtered: " << F.nActiveVertices() << "\n";
	    // std::cerr << "F: " << F << "\n";

	    // All vertices in bucket are removed, have coreness K
	    // Watch out for duplicates in the buckets, as moved vertices
	    // are not removed from their previous bucket.
	    frontier unique;
	    // If vertices are repeated in a sparse frontier, there
	    // will be a race condition if two threads process the same
	    // vertex concurrently. Hence, we remove duplicates before.
	    make_lazy_executor( part )
		.vertex_filter(
		    GA, F, unique,
		    [&]( auto v ) {
			auto cK = expr::constant_val( coreness[v], K );
			return expr::let<var_let>(
			    coreness[v],
			    [&]( auto k ) {
				return expr::make_seq(
				    coreness[v] = expr::add_predicate(
					cK, k == _0 ),
				    k == _0 );
			    } );
		    } )
		.materialize();

	    if( !unique.isEmpty() )
		largestCore = K;

	    // std::cerr << "K: " << K << "\n";
	    // std::cerr << "F     : " << F << "\n";
	    // std::cerr << "unique: " << unique << "\n";
	    // print( std::cerr, part, degrees );
	    // print( std::cerr, part, coreness );

	    frontier output;
	    api::edgemap(
		GA,
		api::filter( api::src, api::strong, unique ),
		api::record( output, api::reduction, api::strong ),
#if FUSION
		api::fusion( [&]( auto v ) {
		    auto cK = expr::constant_val( coreness[v], K );
		    return expr::let<var_let>(
			coreness[v],
			[&]( auto k ) {
			    return expr::make_seq(
				coreness[v] = expr::add_predicate(
				    cK, k == _0 && degrees[v] <= cK ),
				k == _0 && degrees[v] <= cK );
			} );
		} ),
#endif
		api::relax( [&]( auto s, auto d, auto e ) {
		    // Note: constant_val copies over the mask of s
		    auto cK = expr::constant_val( degrees[d], K );
		    return degrees[d] +=
			expr::add_predicate(
			    expr::constant_val( s, -1 ),
			    coreness[d] == _0 );
		} )
		)
		.materialize();

	    // std::cerr << "unique: " << unique.nActiveVertices() << "\n";

	    todo -= unique.nActiveVertices();
	    unique.del();

	    // std::cerr << "output: " << output << "\n";
	    // print( std::cerr, part, degrees );
	    // print( std::cerr, part, coreness );
	    // std::cerr << "todo: " << todo << "\n";

	    bkts.update_buckets( part, output );

	    if( itimes ) {
		info_buf.resize( iter+1 );
		info_buf[iter].density = unique.density( GA.numEdges() );
		info_buf[iter].delay = tm_iter.next();
		info_buf[iter].rm_act = unique.nActiveVertices();
		info_buf[iter].K = K;
		if( debug )
		    info_buf[iter].dump( iter );
		++iter;
	    }

	    output.del();
	    F.del();
	}
	degrees.del();
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
		      << " vertices with zero coreness and non-zero degree\n";
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
				return expr::let<var_let>(
				    neighbours[v],
				    [&]( auto n ) {
					return expr::make_seq(
					    neighbours[v] = _0, // reset
					    n + _1 < coreness[v] );
				    } );
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

	api::edgemap( GA,
		      api::relax( [&]( auto s, auto d, auto e ) {
			  return neighbours[d]
			      += expr::iif( coreness[s] > coreness[d], _0, _1 );
		      } )
	    )
	    .vertex_filter( GA, ftrue, F,
			    [&]( auto v ) {
				return neighbours[v] > coreness[v];
				// && coreness[v] < a_degrees[v];
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
