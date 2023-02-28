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
using expr::_c;

// By default set options for highest performance
#ifndef FUSION
#define FUSION 1
#endif

#ifndef MD_ORDERING
#define MD_ORDERING 0
#endif

enum kc_variable_name {
    var_coreness = 0,
    var_let = 1,
    var_ngh = 2,
    var_mdorder = 3,
    var_kc_num = 4,
    var_degrees_ro = expr::aid_graph_degree
};

struct bucket_fn {
    using ID = VID;
    using BID = std::make_unsigned_t<ID>;
#if OPTIONAL
    using SID = std::make_signed_t<ID>;
#endif
    
    bucket_fn( VID * degree )
	: m_degree( degree ) { }

    BID operator() ( VID v, BID current, BID overflow ) const {
	return v == std::numeric_limits<VID>::max() // illegal vertex
	    || BID(m_degree[v]) < current	// or processing completed
	    || BID(m_degree[v]) >= overflow	// or already in overflow bucket
#if OPTIONAL
	    || SID(m_degree[v]) < 0             // pre-calculated
#endif
	    ? std::numeric_limits<BID>::max() // ... then drop vertex
	    : BID(m_degree[v]);		// ... else this is the bucket
    }
    
private:
    VID * m_degree;
};

template <class GraphType>
class KCv {
public:
    KCv( const GraphType & _GA, commandLine & P )
	: GA( _GA ),
	  num_buckets( P.getOptionLongValue( "-kc:buckets", 127 ) ),
	  coreness( GA.get_partitioner(), "count-down degrees / coreness" ),
#if MD_ORDERING
	  md_order( GA.get_partitioner(), "md-order" ),
#endif
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

#if OPTIONAL
	// Pre-determine the coreness of some vertices where we can
	// TODO: merge emap and vfilter?
	api::vertexprop<VID,VID,var_ngh>
	    n_ngh( part, "coreness neighbour count" );
	frontier predefined;
	make_lazy_executor( part )
	    .vertex_map(
		[&]( auto v ) { return n_ngh[v] = _0; }
		).materialize();
	api::edgemap(
	    GA,
	    api::relax( [&]( auto s, auto d, auto e ) {
		return n_ngh[d] += expr::iif( coreness[s] <= _1, _0, _1 );
	    } )
	    ).materialize();

	make_lazy_executor( part )
	    .vertex_filter(
		GA,
		ftrue,
		predefined,
		[&]( auto v ) {
		    constexpr size_t W = sizeof(VID) * 8 - 1;
		    auto cW = expr::constant_val( coreness[v], W );
		    auto val = ( _1 << cW ) + _1;
		    return expr::make_seq(
			coreness[v] = expr::set_mask(
			    n_ngh[v] >= a_degrees_ro[v] - _1, val ),
			n_ngh[v] >= a_degrees_ro[v] - _1 );
		} )
	    .materialize();
	n_ngh.del();

	// For each pre-calculated coreness, subtract one from its neighbours'
	// coreness, provided the neighbour has not been precalculated
	// TODO: check if more efficient to filter with frontier...
	api::edgemap(
	    GA,
	    api::filter( api::src, api::strong, predefined ),
	    api::filter( api::dst, api::strong,
			 [&]( auto d ) {
			     using SID = std::make_signed_t<VID>;
			     return expr::cast<SID>( coreness[d] ) >= _0; } ),
	    api::relax( [&]( auto s, auto d, auto e ) {
		// using SID = std::make_signed_t<VID>;
		return coreness[d] += _1s;
		// return coreness[d] += expr::iif(
		// expr::cast<SID>( coreness[s] ) < _0, _0, _1s ); // -1
/*
		return coreness[d] += expr::set_mask(
		    expr::cast<SID>( coreness[s] ) >= _0,
		    _1s ); // conditionally subtract 1
*/
	    } )
	    ).materialize();
#endif

	VID K = 0;
	VID L = 0;

	// Create bucket structure
	buckets<VID,bucket_fn>
	    bkts( n, num_buckets, bucket_fn( coreness.get_ptr() ) );

	// Place each vertex in the bucket corresponding with its degree
	// ... in parallel
	bkts.initialize_buckets( part, nonzero );
#if !FUSION
	VID todo = nonzero.nActiveVertices();
#if OPTIONAL
	todo -= predefined.nActiveVertices();
#endif
	// std::cerr << "todo: " << todo << "\n";
#endif

#if MD_ORDERING
	VID md_index = 0;

	frontier zero;
	make_lazy_executor( part )
	    .vertex_filter(
		GA, 	 	 	// graph
		ftrue,
		zero,  	// record new frontier
		[&]( auto v ) {
		    return a_degrees_ro[v] == _0;
		} )
	    .materialize();

	zero.toSparse( part );
	const VID * s = zero.getSparse();
	VID l = zero.nActiveVertices();
	std::copy( &s[0], &s[l], md_order.get_ptr() + md_index );
	md_index += l;
	std::cerr << "md_index: " << md_index << "\n";
	zero.del();
#endif

	VID n_nonzero = nonzero.nActiveVertices();
	nonzero.del();

	largestCore = 0;
	iter = 0;

	while(
#if FUSION
	    !bkts.empty() // cannot count todo as needed with fusion
#else
	    todo > 0 // iterate until all vertices visited
#endif
	    ) {

#if !FUSION
	    assert( !bkts.empty() );
#endif

	    timer tm_iter;
	    tm_iter.start();

	    // TODO: once all m_degrees[] are less than/equal to largestCore,
	    //       computation is complete. Could check for this if we have
	    //       seen a few (effectively) empty buckets

	    // next_bucket updates current bucket, so do this first...
	    frontier F = bkts.next_bucket();
	    K = bkts.get_current_bucket();
	    VID overflow_bkt = bkts.get_overflow_bucket();
	    assert( K >= largestCore );

	    // All vertices in bucket are removed, have coreness K
	    // Watch out for duplicates in the buckets, as moved vertices
	    // are not removed from their previous bucket.
	    frontier unique;
	    make_lazy_executor( part )
		.vertex_filter(
		    GA, F, unique,
		    [&]( auto v ) {
			// TODO: AVX2 >= is more expensive than > and comparison
			//       on unsigned is more expensive than on signed.
			//       Replace by == cK which should be sufficient
			auto cK = expr::constant_val( coreness[v], K );
			return coreness[v] >= cK;
		    } )
		.materialize();

#if 0
	    unique.toSparse( part );
	    if( unique.getType() == frontier_type::ft_sparse ) {
		VID * s = unique.getSparse();
		VID k = unique.nActiveVertices();

		for( VID i=0; i < k; ++i ) {
		    if( s[i] == ~(VID)0 )
			std::cerr << "unique[" << i << "] is _1s\n";
		}
	    }
#endif

	    // Remove duplicate edges. Edges may be multiply represented
	    // as they may be inserted once per edge in the worst case.
	    // This is a consequence of not removing vertices from a bucket
	    // when they move to a different bucket.
	    // In principle each vertex should occur in each bucket once,
	    // however, they may appear multiple times in the overflow bucket.
	    // Note: we have avoided the need to remove duplicates *by design*.
	    // There may remain ~0 duplicates, which are harmless.
	    if( false && !unique.isEmpty() ) {
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

	    // std::cerr << "F: " << F.nActiveVertices() << "\n";
	    // std::cerr << "F: " << F << "\n";

	    if( !unique.isEmpty() )
		largestCore = K;

#if !FUSION
	    if( L == 0 && largestCore >= n_nonzero - todo )
		L = largestCore + 1;
#endif

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
		api::fusion( [&]( auto s, auto d, auto e ) {
		    // Requires that RHS of && is evaluated after LHS.
		    auto cK = expr::constant_val( coreness[d], K );
		    auto cO = expr::constant_val( coreness[d], overflow_bkt );
		    // return coreness[v].count_down( cK );
#if OPTIONAL
		    using SID = std::make_signed_t<VID>;
		    return expr::let<var_let>(
			coreness[v].count_down_value( cK ),
			[&]( auto old ) {
			    return expr::cast<int>(
				expr::iif(
				    expr::cast<SID>( coreness[v] ) >= _0,
				    _1s, // inactive, stay
				    expr::iif(
					old == cK + _1,
					expr::iif( old > cO,
						   _0, // degree below overflow, move
						   _1s ), // in overflow bucket, stay
					_1 ) ) ); // degree dropped to K
			} );
#else
		    return expr::let<var_let>(
			coreness[d].count_down_value( cK ),
			[&]( auto old ) {
			    return
				expr::cast<int>(
				expr::iif(
				    old <= cK || old > cO,
				    expr::iif( old == cK + _1,
					       _0, // degree below overflow, move
					       _1 ), // degree dropped to K
				    _1s ) ); // in overflow bucket, or done, don't move
			} );
#endif
		},
		    api::no_reporting_processed | api::no_duplicate_reporting ),
		api::relax( [&]( auto s, auto d, auto e ) {
		    auto cK = expr::constant_val( coreness[s], K );
#if OPTIONAL
		    using SID = std::make_signed_t<VID>;
		    return coreness[d].count_down_value(
			expr::set_mask( expr::cast<SID>( coreness[d] ) >= _0,
					cK )
			) > cK;
#else
		    return coreness[d].count_down_value( cK ) > cK;
/*
		    using SID = std::make_signed_t<VID>;
		    return coreness[d].count_down_value(
			expr::set_mask( expr::cast<SID>( coreness[d] ) >= _0,
					cK )
			) > cK;
*/
#endif
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
		    return coreness[d].count_down_value( cK ) > cK;
		} )
		)
		.materialize();
#endif

	    // std::cerr << "unique: " << unique.nActiveVertices() << "\n";

#if 0
	    output.toSparse( part );
	    if( output.getType() == frontier_type::ft_sparse ) {
		VID * s = output.getSparse();
		VID k = output.nActiveVertices();

		for( VID i=0; i < k; ++i ) {
		    if( s[i] == ~(VID)0 )
			std::cerr << "unique[" << i << "] is _1s\n";
		}
	    }
#endif

#if MD_ORDERING
	    {
		unique.toSparse( part );
		const VID * s = unique.getSparse();
		VID l = unique.nActiveVertices();
		std::copy( &s[0], &s[l], md_order.get_ptr() + md_index );
		md_index += l;
		std::cerr << "md_index: " << md_index << "\n";
	    }
#endif

#if !FUSION
	    todo -= unique.nActiveVertices();
#endif
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

	std::cerr << "Estimated lower bound on maximum clique size: "
		  << L << "\n";

#if OPTIONAL
	make_lazy_executor( part )
	    .vertex_map(
		predefined,
		[&]( auto v ) {
		    using DT = simd::ty<VID, decltype(v)::VL>;
		    auto mask = expr::_xp<DT>( _1s ) >> expr::_xp<DT>( _1 );
		    return coreness[v] &= mask;
		}
		).materialize();

	// std::cerr << "predefined: " << predefined << "\n";

	predefined.del();
#endif

#if MD_ORDERING
	std::cerr << "md_index: " << md_index << "\n";
	std::cerr << "n: " << n << "\n";
	std::cerr << "todo: " << todo << "\n";
	// assert( md_index == n && "all vertices accounted for" );
#endif
    }

#if MD_ORDERING && 0
    void check_md_ordering() {
	for( VID i=0; i < n; ++i ) {
	    VID v = md_order[i];

	    // Count number of neighbours of all vertices behind v
	    // that are also behind v.
	    expr::scalar<VID,var_mindeg> min_deg;
	    api::edgemap(
		GA,
		api::filter( api::dst, api::strong,
			     [&]( auto v ) {
				 return md_rev_order[v] > _c( i-1 );
			     } ),
		api::relax( [&]( auto s, auto d, auto e ) {
		    return rdeg[d] += _p( _1(rdeg[d]),
					  md_rev_order[s] > _c( i-1 ) );
		} )
		)
		.vertex_map( part,
			     [&]( auto v ) {
				 return min_deg.min(
				     rdeg[v],
				     md_rev_order[v] > _c( i-1 ) );
			     } )
		.materialize();
	    assert( rdeg[v] <= min_deg );
	}
    }
#endif

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
			VID deg = GA.getCSR().getDegree( v );
			VID d = std::min( deg, (VID)100 );
			for( VID i=0; i < d; ++i ) {
			    VID u = GA.getCSR().getNeighbor( v, i );
			    std::cerr << ' ' << u << '=' << coreness[u];
			}
			if( d != deg )
			    std::cerr << " [...]";
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
			VID deg = GA.getCSR().getDegree( v );
			VID d = std::min( deg, (VID)100 );
			for( VID i=0; i < d; ++i ) {
			    VID u = GA.getCSR().getNeighbor( v, i );
			    std::cerr << ' ' << u << '=' << coreness[u];
			}
			if( d != deg )
			    std::cerr << " [...]";
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

    VID getLargestCore() const { return largestCore; }
    const auto & getCoreness() const { return coreness; }

#if MD_ORDERING
    const auto & get_md_ordering() const { return md_order; }
#endif

private:
    const GraphType & GA;
    bool itimes, debug;
    int num_buckets;
    const char * outfile;
    int iter;
    VID largestCore;
    api::vertexprop<VID,VID,var_coreness> coreness;
#if MD_ORDERING
    api::vertexprop<VID,VID,var_mdorder> md_order;
#endif
    std::vector<info> info_buf;
};

#ifndef NOBENCH
template <class GraphType>
using Benchmark = KCv<GraphType>;

#include "driver.C"
#endif
