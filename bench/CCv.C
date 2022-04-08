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

using expr::_0;

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

#ifndef PUSH_ZERO
#undef FUSION
#define PUSH_ZERO 1
#define FUSION 1
#endif

#ifndef PRESET_ZERO
#define PRESET_ZERO 1
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
	    std::cerr << "PUSH_ZERO=" << PUSH_ZERO << "\n";
	    std::cerr << "PRESET_ZERO=" << PRESET_ZERO << "\n";
	}

	// VID max_deg = GA.getCSR().findHighestDegreeVertex();
	// GA.getCSR().setMaxDegreeVertex( max_deg );
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

#if PUSH_ZERO
	// VID max_v = GA.getCSR().findHighestDegreeVertex();
	// VID max_deg = GA.getCSR().getDegree( max_v );
	frontier zeros;
#endif

	iter = 0;

	timer tm_iter;
	tm_iter.start();

#if PRESET_ZERO
	// Create frontier
	frontier G = frontier::sparse(
	    n, GA.getCSR().getDegree(0),
	    const_cast<VID *>(
		&(GA.getCSR().getEdges())[GA.getCSR().getIndex()[0]] ) );

	assert( IDs[0] == 0 && "Assumption about initialisation" );
	make_lazy_executor( part )
	    .vertex_map( G, [&]( auto vid ) {
		return IDs[vid] = expr::zero_val( vid );
	    } )
	    .materialize();
	// Do not delete frontier G, it does not own data

	if( itimes ) {
	    info_buf.resize( iter+1 );
	    info_buf[iter].density = G.density( GA.numEdges() );
	    info_buf[iter].nacte = G.nActiveEdges();
	    info_buf[iter].nactv = G.nActiveVertices();
	    info_buf[iter].active = float(1)/float(n);
	    info_buf[iter].delay = tm_iter.next();
	    if( debug )
		info_buf[iter].dump( iter );
	}
	++iter;
#endif

#if PUSH_ZERO
	// Create bucket structure
	bool switched_to_fusion = false;
	bool have_checked = false;
#endif


	make_lazy_executor( part )
	    .vertex_map( [&]( auto vid ) { return prevIDs[vid] = IDs[vid]; } )
	    .materialize();

	// Create frontier
	frontier F = frontier::all_true( n, m );

	while( !F.isEmpty() ) {  // iterate until IDs converge
	    // Propagate labels
	    frontier output = emap_step( F, 5 );

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
		if( debug )
		    info_buf[iter].dump( iter );
	    }

#if PUSH_ZERO
	    // If the number of vertices with a zero label is low after one
	    // iteration of the dense edgemap, then likely we are working
	    // on a road-network-like graph. In this case, enable bucketing
	    // where there are two buckets: the bucket with active vertices
	    // with a zero label, and those vertices with a non-zero label.
	    // The inactive vertices with a zero label can be ignored (although
	    // they may not be known in case of an unbacked frontier).
	    // In combination with fusion, this allows us to first propagate
	    // the zero label to as many vertices as we can reach, without
	    // synchronisation in a push-style traversal. Only after this is
	    // complete will we consider the remaining vertices, i.e., those
	    // in different clusters.
	    // if( max_deg < n / (128*1024) ) { // assumes a non-power-law graph
	    if( !have_checked ) { // check at most once
		zeros = find_zeros();
		if( zeros.nActiveVertices() * 10 < n ) {
/*
  if(
  #if PRESET_ZERO
  1 +
  #endif
  0 == iter && F.density( GA.numEdges() ) < 0.8 ) {
*/
		    // Cleanup old frontier
		    F.del();
		    F = output;
		    iter++;

		    switched_to_fusion = true;
		    break;
		}
		zeros.del();
		have_checked = true;
	    }
#endif

	    // Cleanup old frontier
	    F.del();
	    F = output;
	    iter++;
	}
	F.del();

#if PUSH_ZERO
	if( switched_to_fusion ) {
	    frontier ftrue = frontier::all_true( n, m ); // all active

	    // If high_penetration is true, the algorithm completed using
	    // the loop above. Nothing further to do.

	    // Loop in case fusion not applied
	    while( !zeros.isEmpty() ) {  // push through zero labels
		frontier F = zeros;
		frontier output = emap_step( F, 200 ); // always sparse

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
		    if( debug )
			info_buf[iter].dump( iter );
		}

		++iter;

		// Cleanup old frontier
		F.del();
		zeros = find_zeros( output );
		output.del();
	    }
	    zeros.del();

	    frontier nonzeros = find_nonzeros();
	    // Loop in case fusion not applied
	    while( !nonzeros.isEmpty() ) {  // push through non-zero labels
		frontier F = nonzeros;
		frontier output = emap_step( F, 200 ); // always sparse

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
		    if( debug )
			info_buf[iter].dump( iter );
		}

		++iter;

		// Cleanup old frontier
		F.del();
		nonzeros = find_nonzeros( output );
		output.del();
	    }
	    nonzeros.del();
	}
#endif
    }

private:
    __attribute__((noinline)) // no-inline to save compilation time
    frontier find_nonzeros( frontier & F ) {
	expr::array_ro<LabelTy,VID,var_ids> IDs( m_IDs );
	frontier nonzeros;
	make_lazy_executor( GA.get_partitioner() )
	    .vertex_filter(
		GA, 	 	 	// graph
		F,			// check these vertices only
		nonzeros,  		// record new frontier
		[&]( auto v ) { return IDs[v] != _0; } )
	    .materialize();
	return nonzeros;
    }

    frontier find_nonzeros() {
	VID n = GA.numVertices();
	EID m = GA.numEdges();
	frontier ftrue = frontier::all_true( n, m ); // all active
	frontier z = find_nonzeros( ftrue );
	ftrue.del();
	return z;
    }

    __attribute__((noinline)) // no-inline to save compilation time
    frontier find_zeros( frontier & F ) {
	expr::array_ro<LabelTy,VID,var_ids> IDs( m_IDs );
	frontier zeros;
	make_lazy_executor( GA.get_partitioner() )
	    .vertex_filter(
		GA, 	 	 	// graph
		F,			// check these vertices only
		zeros,  		// record new frontier
		[&]( auto v ) { return IDs[v] == _0; } )
	    .materialize();
	return zeros;
    }

    frontier find_zeros() {
	VID n = GA.numVertices();
	EID m = GA.numEdges();
	frontier ftrue = frontier::all_true( n, m ); // all active
	frontier z = find_zeros( ftrue );
	ftrue.del();
	return z;
    }

    __attribute__((noinline)) // no-inline to save compilation time
    frontier emap_step( frontier & F, int threshold ) {
	// Propagate labels
	frontier output;

	expr::array_ro<LabelTy,VID,var_previds> prevIDs( m_prevIDs );
	expr::array_ro<LabelTy,VID,var_ids> IDs( m_IDs );

#if UNCOND_EXEC
	auto filter_strength = api::weak;
#else
	auto filter_strength = api::strong;
#endif
	api::edgemap(
	    GA,
	    api::config( api::frac_threshold( threshold ) ),
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
	    api::fusion( [&]( auto d ) {
		return expr::constant_val_one( d );
	    } ),
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

#if DEFERRED_UPDATE || !LEVEL_ASYNC
	// The use of a separate copy pass is less efficient than
	// copying all data immediately when generated
	// maintain_copies( GA.get_partitioner(), output, prevIDs, IDs );
#endif

	return output;
    }

public:
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
		return IDs[d] != IDs[s]; } )
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
