/*
 * TODO: after each edgemap, run a vertexmap with a while loop that
 *       copies through labels (chain shortening):
 *       while( label[v] != label[label[v]] )
 *              label[v] = label[label[v]];
 *       To obey convergence, perhaps make it while larger than
 */

#include "graptor/graptor.h"
#include "graptor/dsl/vertexmap.h"
#include "graptor/api.h"
#include "check.h"
#include "unique.h"

// By default set options for highest performance
#ifndef LEVEL_ASYNC
#define LEVEL_ASYNC 1
#endif

#ifndef DEFERRED_UPDATE
#define DEFERRED_UPDATE 1
#endif

#ifndef UNCOND_EXEC
#define UNCOND_EXEC 1
#endif

#ifndef CONVERGENCE
#define CONVERGENCE 1
#endif

#ifdef MEMO
#undef MEMO
#endif
#define MEMO 0

#ifndef INITID
#define INITID 0
#endif

#ifndef FUSION
#define FUSION 1
#endif

using LabelTy = VID;

enum variable_name {
    var_ids = 0,
    var_previds = 1
};

// To construct a summary histogram of component sizes
struct CC_Vertex_Count
{
    const LabelTy* IDs;
    size_t* count;
    CC_Vertex_Count( const LabelTy* _IDs, size_t* _count ) :
        IDs(_IDs), count(_count) {}
    inline bool operator () ( VID i ) {
	__sync_fetch_and_add( &count[IDs[i]], 1 );
        return IDs[i] == i;
    }
};

template <class GraphType>
struct DegreeCmp {
    DegreeCmp( const GraphType & G_ ) : G( G_ ) { }

    bool operator () ( VID x, VID y ) const {
	// Sort in decreasing order, highest-degree vertex first.
	return G.getOutDegree(x) > G.getOutDegree(y);
    }
private:
    const GraphType & G;
};

template<typename GraphType>
void recode( const GraphType & GA, const VID *labels, VID *orig_labels ) {
    const partitioner &part = GA.get_partitioner();
    VID n = GA.numVertices();
    EID m = GA.numEdges();

    // 0. translate labels
    for( VID v=0; v < n; ++v )
	orig_labels[v] = GA.originalID( labels[ GA.remapID(v) ] );

    // 1. initialise data
    mmap_ptr<size_t> count;
    count.allocate( numa_allocation_partitioned( part ) );

    map_vertexL( part, [&]( VID j ){ count[j]=0; } );

    // 2. count number of vertices per partition
    //    set only vertex donating label as active
    frontier ftrue = frontier::all_true( n, m ); // all active
    frontier components
	= vertexFilter( GA, ftrue, CC_Vertex_Count( orig_labels, count ) );

    // 3. get number of components
    VID ncomponents = components.nActiveVertices();
    components.toSparse( part );
    const VID * s = components.getSparse();

    // 4. Re-use count array as a map to current smallest
    for( VID i=0; i < ncomponents; ++i )
	count[s[i]] = s[i];

    // 5. Determine mininum label for each component
    for( VID v=0; v < n; ++v ) {
	VID cur_label = orig_labels[v];
	if( count[cur_label] > v )
	    count[cur_label] = v;
    }
    
    // 6. Apply new labels
    for( VID v=0; v < n; ++v ) {
	VID cur_label = orig_labels[v];
	orig_labels[v] = count[cur_label];
    }

    // clean up
    count.del();
    ftrue.del();
    components.del();
}

template <class GraphType>
class CCv {
public:
    CCv( GraphType & _GA, commandLine & P ) : GA( _GA ), info_buf( 60 ) {
	itimes = P.getOption( "-itimes" );
	debug = P.getOption( "-debug" );
	calculate_active = P.getOption( "-cactive" );
	outfile = P.getOptionValue( "-cc:outfile" );

	static bool once = false;
	if( !once ) {
	    once = true;
	    std::cerr << "DEFERRED_UPDATE=" << DEFERRED_UPDATE << "\n";
	    std::cerr << "UNCOND_EXEC=" << UNCOND_EXEC << "\n";
	    std::cerr << "LEVEL_ASYNC=" << LEVEL_ASYNC << "\n";
	    std::cerr << "CONVERGENCE=" << CONVERGENCE << "\n";
	    std::cerr << "MEMO=" << MEMO << "\n";
	    std::cerr << "INITID=" << INITID << "\n";
	    std::cerr << "FUSION=" << FUSION << "\n";
	}
    }
    ~CCv() {
	m_IDs.del();
	m_prevIDs.del();
    }

    struct info {
	double delay;
	float density;
	float active;
	EID nacte;
	VID nactv;

	void dump( int i ) {
	    std::cerr << "Iteration " << i << ": " << delay
		      << " density: " << density
		      << " nacte: " << nacte
		      << " nactv: " << nactv
		      << " active: " << active << "\n";
	}
    };

    struct stat { };

    void run() {
	const partitioner &part = GA.get_partitioner();
	VID n = GA.numVertices();
	EID m = GA.numEdges();

	m_IDs.allocate( numa_allocation_partitioned( part ) );
	m_prevIDs.allocate( numa_allocation_partitioned( part ) );

	expr::array_ro<LabelTy,VID,var_previds> prevIDs( m_prevIDs );
	expr::array_ro<LabelTy,VID,var_ids> IDs( m_IDs );

	// Assign initial labels
#if INITID
	// std::sort( &IDs[0], &IDs[n], DegreeCmp<GraphType>( GA ) );
	map_vertexL( part, [&]( VID j ){ m_IDs[j]=GA.originalID(j); } );
#if INITID == 2
	{
	    VID rmaphi = GA.remapID(0);
	    VID orighi = GA.originalID(0);
	    assert( m_IDs[rmaphi] == 0 );
	    m_IDs[0] = 0;
	    m_IDs[rmaphi] = orighi;
	}
#endif
#else
	make_lazy_executor( part )
	    .vertex_map( [&]( auto v ) { return IDs[v] = v; } )
	    .materialize();
#endif

	// Create initial frontier
	frontier F = frontier::all_true( n, m );

	make_lazy_executor( part )
	    .vertex_map( [&]( auto vid ) { return prevIDs[vid] = IDs[vid]; } )
	    .materialize();

	iter = 0;

	timer tm_iter;
	tm_iter.start();

	while( !F.isEmpty() ) {  // iterate until IDs converge
	    // Propagate labels
	    frontier output;

#if UNCOND_EXEC
	    auto filter_strength = api::weak;
#else
	    auto filter_strength = api::strong;
#endif
	    api::edgemap(
		GA,
#if DEFERRED_UPDATE
		api::record( output,
			     api::reduction_or_method, 
			     [&] ( auto d ) { return IDs[d] != prevIDs[d]; },
			     api::strong ),
#else
		api::record( output, api::reduction, api::strong ),
#endif
		api::filter( filter_strength, api::src, F ),
#if CONVERGENCE
		api::filter( api::weak, api::dst,
			     [&] ( auto d ) {
				 return IDs[d] != expr::zero_val(d);
			     } ),
#endif
#if FUSION
		api::fusion( [&]( auto d ) { return expr::true_val( d ); } ),
#endif
		api::relax( [&]( auto s, auto d, auto e ) {
#if LEVEL_ASYNC
			return IDs[d].min( IDs[s] );
#else
			return IDs[d].min( prevIDs[s] );
#endif
		    } )
		)
#if DEFERRED_UPDATE || !LEVEL_ASYNC
		.vertex_map( [&](auto vid) { return prevIDs[vid] = IDs[vid]; } )
#endif
		.materialize();

#if 0
	    std::cerr << "IDs:";
	    for( VID v=0; v < GA.numVertices(); ++v )
		std::cerr << ' ' << m_IDs[v];
	    std::cerr << "\n";

	    if( output.getType() == frontier_type::ft_logical4 && output.getDenseL<4>() ) {
		VID na = 0;
		EID va = 0;
		std::cerr << "output:";
		for( VID v=0; v < GA.numVertices(); ++v ) {
		    std::cerr << ( output.getDenseL<4>()[v] ? 1 : 0 );
		    if( output.getDenseL<4>()[v] ) {
			++na;
			va += GA.getOutDegree(v);
		    }
		}
		std::cerr << "\n";
		std::cerr << " nset=" << na << " nactv=" << output.nActiveVertices() << "\n";
		std::cerr << " eset=" << va << " nacte=" << output.nActiveEdges() << "\n";
		// std::cerr << "differ:";
		// for( VID v=0; v < GA.numVertices(); ++v )
		// std::cerr << ( m_IDs[v] != m_prevIDs[v] ? 1 : 0 );
		// std::cerr << "\n";
		assert( na == output.nActiveVertices() );
		assert( va == output.nActiveEdges() );
	    } else if( output.getType() == frontier_type::ft_bool && output.getDense<frontier_type::ft_bool>() ) {
		VID na = 0;
		EID va = 0;
		std::cerr << "output:";
		for( VID v=0; v < GA.numVertices(); ++v ) {
		    std::cerr << ( output.getDense<frontier_type::ft_bool>()[v] ? 1 : 0 );
		    if( output.getDense<frontier_type::ft_bool>()[v] ) {
			++na;
			va += GA.getOutDegree(v);
		    }
		}
		std::cerr << "\n";
		std::cerr << " nset=" << na << " nactv=" << output.nActiveVertices() << "\n";
		std::cerr << " eset=" << va << " nacte=" << output.nActiveEdges() << "\n";
		// std::cerr << "differ:";
		// for( VID v=0; v < GA.numVertices(); ++v )
		// std::cerr << ( m_IDs[v] != m_prevIDs[v] ? 1 : 0 );
		// std::cerr << "\n";
		assert( na == output.nActiveVertices() );
		assert( va == output.nActiveEdges() );
	    }

#endif

	    // make_lazy_executor( part )
	    // .vertex_map( output,
	    // [&](auto vid) { return prevIDs[vid] = IDs[vid]; } )
	    // .materialize();

	    if( itimes ) {
		VID active = 0;
		if( calculate_active ) { // Warning: expensive, included in time
		    for( VID v=0; v < n; ++v )
			if( m_prevIDs[v] != 0 )
			    active++;
		}
		info_buf.resize( iter+1 );
		info_buf[iter].density = F.density( GA.numEdges() );
		info_buf[iter].nacte = F.nActiveEdges();
		info_buf[iter].nactv = F.nActiveVertices();
		info_buf[iter].active = float(active)/float(n);
		info_buf[iter].delay = tm_iter.next();
		if( debug ) {
		    info_buf[iter].dump( iter );
/*
		    if( F.nActiveVertices() < 10 ) {
			F.toSparse( part );
			const VID * f = F.getSparse();
			std::cerr << "Frontier:";
			for( long i=0; i < F.nActiveVertices(); ++i )
			    std::cerr << " " << GA.originalID( f[i] );
			std::cerr << "\n";
		    }
*/
		}
	    }

	    // Cleanup old frontier
	    F.del();
	    F = output;

	    iter++;
	}
    }

    void post_process( stat & stat_buf ) {
	if( itimes ) {
	    for( int i=0; i < iter; ++i )
		info_buf[i].dump( i );
	}

	if( outfile ) {
	    recode( GA, m_IDs.get(), m_prevIDs.get() );
	    writefile( GA, outfile, m_prevIDs, false ); // do not remap vertex IDs
	}
    }

    static void report( const std::vector<stat> & stat_buf ) { }

    void validate( stat & stat_buf ) {
	count_unique<VID>( GA, m_IDs.get(), std::cout );

	expr::array_ro<LabelTy,VID,var_ids> IDs( m_IDs );
	frontier F;
	api::edgemap(
	    GA,
	    api::record( F, api::reduction, api::strong ),
	    api::relax( [&]( auto s, auto d, auto e ) {
		return expr::make_unop_switch_to_vector( IDs[d] != IDs[s] ); } )
	    )
	    .materialize();

	if( F.isEmpty() ) {
	    std::cerr << "Neighbour check: PASS\n";
	} else {
	    std::cerr << "Neighbour check: FAIL; #vertices: "
		      << F.nActiveVertices() << "\n";
	    std::cerr << "failures:";
	    VID n = GA.numVertices();
	    VID k = std::min( n, (VID)10 ); 
	    VID l = 0;
	    for( VID v=0; v < n; ++v )
		if( F.is_set( v ) ) {
		    std::cerr << "\n\t" << v << '(' << m_IDs[v]
			      << ',' << m_prevIDs[v]
			      << ')';

#if 0
		    auto ns = GA.getCSC().neighbour_begin( v );
		    auto ne = GA.getCSC().neighbour_end( v );
		    std::cerr << '#' << (ne-ns) << ' ';
		    VID ll=0;
		    for( auto ngh=ns; ngh != ne; ++ngh ) {
			if( m_IDs[*ngh] != m_IDs[v] ) {
			    std::cerr << *ngh
				      << '#' << GA.getOutDegree( *ngh )
				      << '[' << m_IDs[*ngh]
			      << ',' << m_prevIDs[*ngh]
			      << ']';
			    if( ++ll > 10 )
				break;
			}
		    }
#endif
		    
		    ++l;
		    if( l > k )
			break;
		}
	    std::cerr << '\n';
	    assert( 0 && "Validation failed" );
	}

	F.del();
    }

#if 0
    void validate( frontier & in, frontier & out ) {
	expr::array_ro<LabelTy,VID,var_ids> IDs( m_IDs );
	frontier F;
	api::edgemap(
	    GA,
	    api::record( F, api::reduction, api::strong ),
	    api::relax( [&]( auto s, auto d, auto e ) {
		return expr::make_unop_switch_to_vector( IDs[d] != IDs[s] ); } )
	    )
	    .materialize();

	if( F.isEmpty() ) {
	    std::cerr << "Neighbour check: PASS\n";
	} else {
	    std::cerr << "Neighbour check: FAIL; #vertices: "
		      << F.nActiveVertices() << "\n";
	    std::cerr << "failures:";
	    VID n = GA.numVertices();
	    VID k = std::min( n, (VID)10 ); 
	    VID l = 0;
	    for( VID v=0; v < n; ++v )
		if( F.is_set( v ) ) {
		    std::cerr << "\n\t" << v << '(' << m_IDs[v]
			      << ',' << m_prevIDs[v]
			      << ( in.is_set(v) ? ",in" : ",not-in" )
			      << ( out.is_set(v) ? ",out" : ",not-out" )
			      << ')';

		    auto ns = GA.getCSC().neighbour_begin( v );
		    auto ne = GA.getCSC().neighbour_end( v );
		    std::cerr << '#' << (ne-ns) << ' ';
		    VID ll=0;
		    for( auto ngh=ns; ngh != ne; ++ngh ) {
			if( m_IDs[*ngh] != m_IDs[v] ) {
			    std::cerr << *ngh
				      << '[' << m_IDs[*ngh]
			      << ',' << m_prevIDs[*ngh]
			      << ( in.is_set(*ngh) ? ",in" : ",not-in" )
			      << ( out.is_set(*ngh) ? ",out" : ",not-out" )
			      << ']';
			    if( ++ll > 10 )
				break;
			}
		    }
		    
		    ++l;
		    if( l > k )
			break;
		}
	    std::cerr << '\n';

	    {
		VID v = 641592;

		std::cerr << "extra:";
		std::cerr << "\n\t" << v << '(' << m_IDs[v]
			  << ',' << m_prevIDs[v]
			  << ( F.is_set(v) ? ",in" : ",not-in" )
			  << ( in.is_set(v) ? ",in" : ",not-in" )
			  << ( out.is_set(v) ? ",out" : ",not-out" )
			  << ')';

		auto ns = GA.getCSC().neighbour_begin( v );
		auto ne = GA.getCSC().neighbour_end( v );
		std::cerr << '#' << (ne-ns) << ' ';
		VID ll=0;
		for( auto ngh=ns; ngh != ne; ++ngh ) {
		    if( m_IDs[*ngh] != m_IDs[v] ) {
			std::cerr << *ngh
				  << '[' << m_IDs[*ngh]
				  << ',' << m_prevIDs[*ngh]
				  << ( in.is_set(*ngh) ? ",in" : ",not-in" )
				  << ( out.is_set(*ngh) ? ",out" : ",not-out" )
				  << ']';
			if( ++ll > 10 )
			    break;
		    }
		}
	    }
	    std::cerr << '\n';
	}

	F.del();
    }
#endif


private:
    const GraphType & GA;
    bool itimes, debug, calculate_active;
    int iter;
    const char * outfile;
    mmap_ptr<LabelTy> m_IDs, m_prevIDs;
    std::vector<info> info_buf;
};

template <class GraphType>
using Benchmark = CCv<GraphType>;

#include "driver.C"
