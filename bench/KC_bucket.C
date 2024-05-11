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
    
    bucket_fn( VID * degree )
	: m_degree( degree ) { }

    BID operator() ( VID v, BID current, BID overflow ) const {
	BID deg = m_degree[v];
	return // v == std::numeric_limits<VID>::max() // illegal vertex
	    // ||
	    deg < current	// or processing completed
	    || deg >= overflow	// or already in overflow bucket
	    ? std::numeric_limits<BID>::max() // ... then drop vertex
	    : deg;		// ... else this is the bucket
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
#if MD_ORDERING
	md_order.del();
#endif
    }

    struct info {
	double delay;
	float density;
	VID F_act, rm_act, wakeup_act;
	VID K;
	frontier_type ftype;

	void dump( int i ) {
	    std::cerr << "Iteration " << i << ": " << delay
		      << ' ' << ftype
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
	VID L = 0;

	// Create bucket structure
	buckets<VID,bucket_fn>
	    bkts( n, num_buckets, bucket_fn( coreness.get_ptr() ) );

	// Place each vertex in the bucket corresponding with its degree
	// ... in parallel
	bkts.initialize_buckets( part, nonzero );
#if !FUSION
	VID todo = nonzero.nActiveVertices();
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

	VID prev_iter_K = K;

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

	    // TODO: once all coreness[] are less than/equal to largestCore,
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
	    // Duplicates only appear in the buckets, not in the output frontier
	    VID F_nactv = F.nActiveVertices();
	    if( prev_iter_K != K ) {
		frontier unique;
		make_lazy_executor( part )
		    .vertex_filter(
			GA, F, unique,
			[&]( auto v ) {
			    // AVX2 >= is more expensive than > and comparison
			    // on unsigned is more expensive than on signed.
			    // Replace by == cK which should be sufficient
			    // auto cK = expr::constant_val( coreness[v], K );
			    return coreness[v] == _c( K );
			} )
		    .materialize();
		F.del();
		F = unique;
	    }

	    prev_iter_K = K;

	    if( !F.isEmpty() )
		largestCore = K;

#if !FUSION
	    if( L == 0 && largestCore >= n_nonzero - todo )
		L = largestCore + 1;
#endif

	    // std::cerr << "filter completed vertices: " << tm_iter.next() << "\n";

	    // std::cerr << "K: " << K << "\n";
	    // std::cerr << "overflow_bkt: " << overflow_bkt << "\n";
	    // std::cerr << "F     : " << F << "\n";
	    // print( std::cerr, part, coreness );

	    frontier output = frontier::empty();
	    
	    if( !F.isEmpty() ) {
#if FUSION
	    api::edgemap(
		GA,
		api::config(
		    // api::always_sparse ),
		    api::fusion_select(
			F.nActiveEdges() >= EMAP_BLOCK_SIZE * 8 ) ),
		api::filter( api::src, api::strong, F ),
		api::record( output, api::reduction, api::strong ),
		api::fusion( [&]( auto s, auto d, auto e ) {
		    // Requires that RHS of && is evaluated after LHS.
		    auto cK = expr::constant_val( coreness[d], K );
		    auto cO = expr::constant_val( coreness[d], overflow_bkt );
		    return expr::let<var_let>(
			coreness[d].count_down_value( cK ),
			[&]( auto old ) {
			    return expr::cast<int>(
				expr::iif(
				    old == cK + _1,
				    expr::iif(
					old <= cK || old > cO,
					_0(d), // degree below overflow, move
					_1s(d) ), // in oflow or done, don't move
				    _1(d) ) ); // degree dropped to K
			} );
		},
		    api::no_reporting_processed | api::no_duplicate_reporting ),
		api::relax( [&]( auto s, auto d, auto e ) {
		    auto cK = expr::constant_val( coreness[s], K );
		    return coreness[d].count_down_value( cK ) > cK;
		} )
		)
		.materialize();
#else
	    api::edgemap(
		GA,
		api::config( api::always_sparse ),
		api::filter( api::src, api::strong, F ),
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

	    } // !F.isEmpty()

#if MD_ORDERING
	    {
		F.toSparse( part );
		const VID * s = F.getSparse();
		VID l = F.nActiveVertices();
		std::copy( &s[0], &s[l], md_order.get_ptr() + md_index );
		md_index += l;
		std::cerr << "md_index: " << md_index << "\n";
	    }
#endif

#if !FUSION
	    todo -= F.nActiveVertices();
#endif

	    // std::cerr << "edgemap: " << tm_iter.next() << "\n";

	    // std::cerr << "output: " << output << "\n";
	    // print( std::cerr, part, coreness );
	    // std::cerr << "todo: " << todo << "\n";

	    bkts.update_buckets( part, output );

	    // std::cerr << "update buckets: " << tm_iter.next() << "\n";

	    if( itimes ) {
		info_buf.resize( iter+1 );
		info_buf[iter].density = F.density( GA.numEdges() );
		info_buf[iter].delay = tm_iter.total();
		info_buf[iter].F_act = F_nactv;
		info_buf[iter].rm_act = F.nActiveVertices();
		info_buf[iter].wakeup_act = output.nActiveVertices();
		info_buf[iter].K = K;
		info_buf[iter].ftype = output.getType();
		if( debug )
		    info_buf[iter].dump( iter );
		++iter;
	    }

	    output.del();
	    F.del();
	}

	std::cerr << "Estimated lower bound on maximum clique size: "
		  << L << "\n";

#if MD_ORDERING
	std::cerr << "md_index: " << md_index << "\n";
	std::cerr << "n: " << n << "\n";
	// std::cerr << "todo: " << todo << "\n";
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

//! Auxiliary method for downstream tasks
void
sort_order( VID n,
	    VID K,
	    const VID * const coreness,
	    VID * order, VID * rev_order,
	    bool reverse = false ) {
    VID * histo = new VID[K+1];
    std::fill( &histo[0], &histo[K+1], 0 );

    // Histogram
    for( VID v=0; v < n; ++v ) {
	VID c = coreness[v];
	assert( c <= K );
	histo[K-c]++;
    }

    // Prefix sum
    // Note: require int variables as the code checks >= 0 which is futile
    //       with unsigned int
    VID sum = sequence::scan( histo, (int)0, (int)K+1, addF<VID>(),
			      sequence::getA<VID,int>( histo ),
			      (VID)0, false, reverse );

    // Place in order
    for( VID v=0; v < n; ++v ) {
	VID c = coreness[v];
	VID pos = histo[K-c]++;
	order[pos] = v;
	rev_order[v] = pos;
    }

    delete[] histo;
}

//! Auxiliary method for downstream tasks
template<typename T, typename U, short AID, typename Encoding, bool NT>
void
sort_order( VID n,
	    VID K,
	    const api::vertexprop<T,U,AID,Encoding,NT> & coreness,
	    VID * order, VID * rev_order,
	    bool reverse = false ) {
    return sort_order( n, K, coreness.get_ptr(), order, rev_order, reverse );
}

//! \brief Sort vertices by coreness and by degree for equal coreness
//
// Auxiliary method for downstream tasks.
// Given a graph with \p n vertices in the range [0,\p n), and their degrees
// given in \p degree and coreness in \p coreness, sort the vertices in
// ascending (\p reverse == false) or descending order (\p reverse == true).
// Vertices are primarily sorted by coreness, and those with equal coreness
// are sorted by degree.
// The array \p order lists the vertex IDs in sort order; \p rev_order makes
// the reverse mapping. \p order is indexed by the remapped IDs whereas
// \p rev_order is ordered by original vertex IDs
// \p histo contains the partitions of the sort order such that vertices
// with coreness \c k are found in the range \c [histo[k],histo[k+1]) in the
// \p order array if \p reverse is false. When \p reverse is true, then the
// vertices with degeneracy \c k are ound in the range \c [histo[k+1],histo[k]).
// \p histo must be of length at least \p K+2.
//
// \param[in] n Number of vertices
// \param[in] K maximum coreness (\p K inclusive)
// \param[in] coreness Coreness of the vertices
// \param[in] degree Degree of vertices
// \param[out] order Order in which vertices are processed. Members of this
//                   array are VIDs in the original graph
// \param[out] rev_order Reverse order. Members of this array are remapped VIDs
// \param[out] histo Partition of order array by vertices with equal coreness
// \param[in] reverse_core Sort ascending (false) or descending (true) coreness
// \param[in] reverse_deg Sort ascending (false) or descending (true) degree
//
// \post
//   0 <= order[r] < n for any r in [0,n).
//   0 <= rev_order[v] < n for any v in [0,n).
//   rev_order[order[r]] == r.
//   order[rev_order[v]] == v.
//   coreness[order[histo[k]]] > coreness[order[histo[l]]] if k > l
//       and reverse is false.
//   degree[order[r]] > degree[order[s]] if r > s
//       and coreness[order[r]] == coreness[order[s]] and reverse is false.
//   coreness[order[histo[k]]] < coreness[order[histo[l]]] if k > l
//       and reverse is true.
//   degree[order[r]] < degree[order[s]] if r > s
//       and coreness[order[r]] == coreness[order[s]] and reverse is true.
void
sort_order_ties( VID n,
		 VID K,
		 const VID * const degree,
		 const VID * const coreness,
		 VID * order,
		 VID * rev_order,
		 VID * histo, // array of length K+2
		 bool reverse_core = false,
		 bool reverse_deg = false ) {
    std::fill( &histo[0], &histo[K+1], 0 );

    // Histogram
    for( VID v=0; v < n; ++v ) {
	VID c = coreness[v];
	assert( c <= K );
	histo[c]++;
    }

    // Prefix sum
    // Note: require int variables as the code checks >= 0 which is futile
    //       with unsigned int
    VID sum = sequence::scan( histo, (int)0, (int)K+1, addF<VID>(),
			      sequence::getA<VID,int>( histo ),
			      (VID)0, false, reverse_core );
    
    // Place in order
    for( VID v=0; v < n; ++v ) {
	VID c = coreness[v];
	VID pos = histo[c]++;
	order[pos] = v;
    }

    // Restore histo
    if( reverse_core ) {
	for( VID c=1; c <= K; ++c )
	    histo[c] = histo[c+1];
	histo[0] = n;
	histo[K+1] = 0;
    } else {
	for( VID c=0; c <= K; ++c )
	    histo[K-c+1] = histo[K-c];
	histo[0] = 0;
    }

    // Degree sorting (crude), decreasing order
    // Check reverse inside loop to reduce code size.
    parallel_loop( (VID)0, K+1, [&]( VID c ) {
	VID c_lo = histo[c];
	VID c_up = histo[c+1];
	if( reverse_core )
	    std::swap( c_lo, c_up );
	if( reverse_deg ) {
	    std::sort( &order[c_lo], &order[c_up],
		       [&]( VID a, VID b ) { return degree[a] > degree[b]; } );
	} else {
	    std::sort( &order[c_lo], &order[c_up],
		       [&]( VID a, VID b ) { return degree[a] < degree[b]; } );
	}
    } );

    // Construct reverse order
    parallel_loop( (VID)0, n, [&]( VID pos ) {
	rev_order[order[pos]] = pos;
    } );
}

//! \brief Sort vertices by coreness and by degree for equal coreness
//
//  Auxiliary method for downstream tasks
template<typename T, typename U, short AID, typename Encoding, bool NT>
void
sort_order_ties( VID n,
		 VID K,
		 const VID * const degree,
		 const api::vertexprop<T,U,AID,Encoding,NT> & coreness,
		 VID * order,
		 VID * rev_order,
		 VID * histo, // array of length K+1
		 bool reverse_core = false,
		 bool reverse_deg = false ) {
    return sort_order_ties( n, K, degree, coreness.get_ptr(),
			    order, rev_order, histo,
			    reverse_core, reverse_deg );
}

#ifndef NOBENCH
template <class GraphType>
using Benchmark = KCv<GraphType>;

#include "driver.C"
#endif
