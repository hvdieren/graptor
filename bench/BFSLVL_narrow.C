#include "graptor/graptor.h"
#include "graptor/api.h"

using expr::_0;
using expr::_1;
using expr::_1s;
using expr::_c;

// By default set options for highest performance
#ifndef DEFERRED_UPDATE
#define DEFERRED_UPDATE 1
#endif

#ifndef UNCOND_EXEC
#define UNCOND_EXEC 1
#endif

#ifndef CONVERGENCE
#define CONVERGENCE 1
#endif

#ifndef LEVEL_ASYNC
#define LEVEL_ASYNC 1
#endif

#ifdef MEMO
#undef MEMO
#endif
#define MEMO 0

// TODO: FUSION works well for road networks: estimate network type and select at runtime
#ifndef FUSION
#define FUSION 0
#endif

enum var {
    var_current8 = 0,
    var_previous8 = 1,
    var_current16 = 2,
    var_previous16 = 3,
    var_current = 4,
    var_previous = 5
};

enum dvariant {
    spmv8,
    spmv16,
    spmv,
    terminate_step
};

// Ensure signed comparisons between levels as ordered comparators in AVX2
// are more efficient on signed than unsigned.
using sVID = std::make_signed_t<VID>;

// TODO: make previous level always at width of sVID, OR expose raw interface
//       during vertex map
using Enc8 = array_encoding_wide<uint8_t,true>;
using Enc16 = array_encoding_wide<uint16_t,true>;
using Enc = array_encoding<sVID>; // natural encoding

template <class GraphType>
class BFSLVLNarrow {
public:
    BFSLVLNarrow( GraphType & _GA, commandLine & P )
	: GA( _GA ),
	  a_level8( GA.get_partitioner(), "current level/8" ),
	  a_level16( GA.get_partitioner(), "current level/16" ),
	  a_level( GA.get_partitioner(), "current level" ),
#if !LEVEL_ASYNC || DEFERRED_UPDATE
	  a_prev_level8( GA.get_partitioner(), "previous level/8" ),
	  a_prev_level16( GA.get_partitioner(), "previous level/16" ),
	  a_prev_level( GA.get_partitioner(), "previous level" ),
#endif
	  info_buf( 60 ) {
	itimes = P.getOption( "-itimes" );
	debug = P.getOption( "-debug" );
	calculate_active = P.getOption( "-cactive" );
	start = GA.remapID( P.getOptionLongValue( "-start", 0 ) );

	static bool once = false;
	if( !once ) {
	    once = true;
	    std::cerr << "DEFERRED_UPDATE=" << DEFERRED_UPDATE << "\n";
	    std::cerr << "UNCOND_EXEC=" << UNCOND_EXEC << "\n";
	    std::cerr << "LEVEL_ASYNC=" << LEVEL_ASYNC << "\n";
	    std::cerr << "CONVERGENCE=" << CONVERGENCE << "\n";
	    std::cerr << "MEMO=" << MEMO << "\n";
	    std::cerr << "FUSION=" << FUSION << "\n";
	}
    }
    ~BFSLVLNarrow() {
	a_level8.del();
	a_level16.del();
	a_level.del();
#if !LEVEL_ASYNC || DEFERRED_UPDATE
	a_prev_level8.del();
	a_prev_level16.del();
	a_prev_level.del();
#endif
    }

    struct info {
	double delay;
	float density;
	float active;
	EID nacte;
	VID nactv;
	frontier_type ftype;
	int width;

	void dump( int i ) {
	    std::cerr << "Iteration " << i << ": " << delay
		      << ' ' << ftype
		      << ' ' << width
		      << " density: " << density
		      << " nacte: " << nacte
		      << " nactv: " << nactv
		      << " active: " << active << "\n";
	}
    };

    struct stat { };

    void run() {
	const partitioner &part = GA.get_partitioner();
	sVID n = GA.numVertices();
	EID m = GA.numEdges();

	// Create initial frontier
	frontier F = frontier::create( n, start, GA.getOutDegree(start) );

	iter = 0;
	active = 1;

	// Enable fusion immediately if the graph has relatively low
	// degrees (e.g., road network). Otherwise, wait until a dense
	// iteration has occured (in which case further dense iterations
	// will still take precedence over sparse/fusion iterations).
#if FUSION
	bool enable_fusion = isLowDegreeGraph( GA );
#else
	bool enable_fusion = false;
#endif

	constexpr sVID upper8 = ( sVID(1) << 8 ) - 4;
	constexpr sVID upper16 = ( sVID(1) << 16 ) - 4;
	const sVID ilvl = n+1;

	// Assign initial labels at most narrow width
	if( enable_fusion ) {
	    make_lazy_executor( part )
		.vertex_map( [&]( auto v ) {
		    return a_level[v] = _c( ilvl ); } )
#if DEFERRED_UPDATE || !LEVEL_ASYNC
		.vertex_map( [&]( auto v ) {
		    return a_prev_level[v] = _c( ilvl ); } )
#endif
		.materialize();

	    a_level.get_ptr()[start] = 0;
#if DEFERRED_UPDATE || !LEVEL_ASYNC
	    a_prev_level.get_ptr()[start] = 0;
#endif
	} else {
	    const sVID ilvl8 = std::numeric_limits<uint8_t>::max();
	    make_lazy_executor( part )
		.vertex_map( [&]( auto v ) {
		    return a_level8[v] = _c( ilvl8 ); } )
#if DEFERRED_UPDATE || !LEVEL_ASYNC
		.vertex_map( [&]( auto v ) {
		    return a_prev_level8[v] = _c( ilvl8 ); } )
#endif
		.materialize();

	    a_level8.get_ptr()[start] = 0;
#if DEFERRED_UPDATE || !LEVEL_ASYNC
	    a_prev_level8.get_ptr()[start] = 0;
#endif
	}

	bool done = F.isEmpty();
	dvariant state = enable_fusion ? spmv : spmv8, prev_state;
	
	while( !done ) {
	    prev_state = state;
	    switch( state ) {
	    case spmv8:
		done = iterate( iter, upper8, enable_fusion, a_level8,
#if !LEVEL_ASYNC || DEFERRED_UPDATE
				a_prev_level8,
#endif
				F );
		state = done ? terminate_step : enable_fusion ? spmv : spmv16;
		break;
	    case spmv16:
		done = iterate( iter, upper16, enable_fusion, a_level16,
#if !LEVEL_ASYNC || DEFERRED_UPDATE
				a_prev_level16,
#endif
				F );
		state = done ? terminate_step : spmv;
		break;
	    case spmv:
		done = iterate( iter, n+1, enable_fusion, a_level,
#if !LEVEL_ASYNC || DEFERRED_UPDATE
				a_prev_level,
#endif
				F );
		assert( done );
		state = terminate_step;
		break;
	    case terminate_step:
		assert( 0 && "should not get here" );
		break;
	    }

	    // Transition
	    switch( state ) {
	    case spmv8:
		assert( prev_state == spmv8 && "only initial state" );
		break;
	    case spmv16:
		switch( prev_state ) {
		case spmv8: transition( spmv8, spmv16 ); break;
		default:
		    assert( 0 && "not valid transition" ); break;
		}
		break;
	    case spmv:
	    case terminate_step:
		switch( prev_state ) {
		case spmv8: transition( spmv8, spmv ); break;
		case spmv16: transition( spmv16, spmv ); break;
		default: break;
		}
		break;
	    }

	    active += F.nActiveVertices();
	}

#if 0
	// Iterate at width 8
	bool done = iterate( iter, upper8, a_level8,
#if !LEVEL_ASYNC || DEFERRED_UPDATE
			     a_prev_level8,
#endif
			     F );

	if( done ) {
	    // Convert from 8 bits to whatever sVID is
	    make_lazy_executor( part )
		.vertex_map( [&]( auto v ) {
		    return a_level[v] =
			expr::iif( a_level8[v] > _c( upper8 ),
				   a_level8[v], _c( ilvl ) );
		} )
		.materialize();
	}

	if( !done ) {
	    // Convert from 8 to 16 bits
	    // TODO: check if any vertices that exceeded upper8 need to
	    //       be reactivated in frontier
	    const sVID ilvl16 = std::numeric_limits<uint16_t>::max();
	    make_lazy_executor( part )
		.vertex_map( [&]( auto v ) {
		    return a_level16[v] =
			expr::iif( a_level8[v] > _c( upper8 ),
				   a_level8[v], _c( ilvl16 ) );
		} )
#if DEFERRED_UPDATE || !LEVEL_ASYNC
		.vertex_map( [&]( auto v ) {
		    return a_prev_level16[v] = a_level16[v]; } )
#endif
		.materialize();

	    done = iterate( iter, upper16, a_level16,
#if !LEVEL_ASYNC || DEFERRED_UPDATE
			    a_prev_level16,
#endif
			    F );

	    if( done ) {
		// Convert from 16 bits to whatever sVID is
		make_lazy_executor( part )
		    .vertex_map( [&]( auto v ) {
			return a_level[v] =
			    expr::iif( a_level16[v] > _c( upper16 ),
				       a_level16[v], _c( ilvl ) );
		    } )
		    .materialize();
	    }
	}

	if( !done ) {
	    // Convert from 16 bits to whatever sVID is
	    make_lazy_executor( part )
		.vertex_map( [&]( auto v ) {
		    return a_level[v] =
			expr::iif( a_level16[v] > _c( upper16 ),
				   a_level16[v], _c( ilvl ) );
		} )
#if DEFERRED_UPDATE || !LEVEL_ASYNC
		.vertex_map( [&]( auto v ) {
		    return a_prev_level[v] = a_level[v]; } )
#endif
		.materialize();

	    sVID upper = ilvl;
	    done = iterate( iter, upper, a_level,
#if !LEVEL_ASYNC || DEFERRED_UPDATE
			    a_prev_level,
#endif
			    F );
	}

	assert( done );
#endif

	F.del();
    }

    void post_process( stat & stat_buf ) {
	if( itimes ) {
	    for( int i=0; i < iter; ++i )
		info_buf[i].dump( i );
	}
    }

    static void report( const std::vector<stat> & stat_buf ) { }

    void validate( stat & stat_buf ) {
	VID n = GA.numVertices();
	// VID longest = iter - 1;
	sVID longest = 0;
	VID longest_v = start;
	for( VID v=0; v < n; ++v )
	    if( longest < a_level[v] && a_level[v] < n+1 ) {
		longest = a_level[v];
		longest_v = v;
	    }

	std::cout << "Longest path from " << start
		  << " (original: " << GA.originalID( start ) << ") : "
		  << longest
		  << " at: " << longest_v
		  << " (" << GA.originalID( longest_v ) << ")\n";

	frontier wrong;
	// Assuming symmetric graph, showing inverse properties:
	// Any neigbours must be immediately before us, together with
	// us or at most immediately after us.
	// Note: condition for wrong values
	if( GA.isSymmetric() ) {
	    api::edgemap(
		GA,
		api::record( wrong, api::reduction, api::strong ),
		api::relax( [&]( auto s, auto d, auto e ) {
		    return a_level[s] > a_level[d] + _1
			|| a_level[s] < a_level[d] - _1;
		} )
		)
		.materialize();
	}  else {
	    // Directed graph: some paths may take longer to reach us and that
	    // is fine. But no-one that can reach us can do it more than
	    // one step ahead of us.
	    api::edgemap(
		GA,
		api::record( wrong, api::reduction, api::strong ),
		api::relax( [&]( auto s, auto d, auto e ) {
		    return a_level[s] < a_level[d] - _1;
		} )
		)
		.materialize();
	}
	std::cout << "Number of vertices with nghbs not activated in time: "
		  << wrong.nActiveVertices()
		  << ( wrong.nActiveVertices() == 0 ? " PASS" : " FAIL" )
		  << "\n";

	std::cout << "Number of activated vertices: " << active << "\n";
	std::cout << "Number of vertices: " << n << "\n";
#if !UNCOND_EXEC
	std::cout << "Every vertex activated at most once: "
		  << ( n >= active ? "PASS" : "FAIL" ) << "\n";
#endif
    }

    const sVID * get_level() const { return a_level.get_ptr(); }

private:
    void transition( dvariant from, dvariant to ) {
	const sVID upper8 = ( sVID(1) << 8 ) - 4;
	const sVID upper16 = ( sVID(1) << 16 ) - 4;
	const sVID ilvl = GA.numVertices()+1;
	const sVID ilvl16 = std::numeric_limits<uint16_t>::max();
	const partitioner & part = GA.get_partitioner();

	if( from == spmv8 && ( to == spmv || to == terminate_step ) ) {
	    // Convert from 8 bits to whatever sVID is
	    make_lazy_executor( part )
		.vertex_map( [&]( auto v ) {
		    return a_level[v] =
			expr::iif( a_level8[v] > _c( upper8 ),
				   a_level8[v], _c( ilvl ) );
		} )
#if DEFERRED_UPDATE || !LEVEL_ASYNC
		.vertex_map( [&]( auto v ) {
		    return a_prev_level[v] = a_level[v]; } )
#endif
		.materialize();
	} else if( from == spmv8 && to == spmv16 ) {
	    make_lazy_executor( part )
		.vertex_map( [&]( auto v ) {
		    return a_level16[v] =
			expr::iif( a_level8[v] > _c( upper8 ),
				   a_level8[v], _c( ilvl16 ) );
		} )
#if DEFERRED_UPDATE || !LEVEL_ASYNC
		.vertex_map( [&]( auto v ) {
		    return a_prev_level16[v] = a_level16[v]; } )
#endif
		.materialize();
	} else if( from == spmv16 && ( to == spmv || to == terminate_step ) ) {
	    // Convert from 16 bits to whatever sVID is
	    make_lazy_executor( part )
		.vertex_map( [&]( auto v ) {
		    return a_level[v] =
			expr::iif( a_level16[v] > _c( upper16 ),
				   a_level16[v], _c( ilvl ) );
		} )
#if DEFERRED_UPDATE || !LEVEL_ASYNC
		.vertex_map( [&]( auto v ) {
		    return a_prev_level[v] = a_level[v]; } )
#endif
		.materialize();
	}
    }

    template<typename LvlEnc, short aid_cur, short aid_prev>
    bool
    iterate( sVID & iter,
	     sVID max_iter,
	     bool & enable_fusion,
	     api::vertexprop<sVID,VID,aid_cur,LvlEnc> & a_level,
#if !LEVEL_ASYNC || DEFERRED_UPDATE
	     api::vertexprop<sVID,VID,aid_prev,LvlEnc> & a_prev_level,
#endif
	     frontier & F ) {
	const sVID n = GA.numVertices();
	const EID m = GA.numEdges();

	// iterate until all vertices visited, or stop when time to grow
	// the width of the levels
	while( !F.isEmpty() && iter < max_iter ) {
	    timer tm_iter;
	    tm_iter.start();

	    frontier output = edgemap_step( iter, enable_fusion, a_level,
#if !LEVEL_ASYNC || DEFERRED_UPDATE
					    a_prev_level,
#endif
					    F );

	    if( itimes ) {
		VID active = 0;
		if( calculate_active ) { // Warning: expensive, included in time
		    for( VID v=0; v < n; ++v )
			if( a_level[v] == n+1 )
			    active++;
		}
		info_buf.resize( iter+1 );
		info_buf[iter].density = F.density( GA.numEdges() );
		info_buf[iter].delay = tm_iter.next();
		info_buf[iter].active = float(active)/float(n);
		info_buf[iter].nacte = F.nActiveEdges();
		info_buf[iter].nactv = F.nActiveVertices();
		info_buf[iter].ftype = F.getType();
		info_buf[iter].width = sizeof(typename LvlEnc::stored_type) * 8;
		if( debug )
		    info_buf[iter].dump( iter );
	    }

	    // Cleanup old frontier
	    F.del();
	    F = output;

	    ++iter;

#if FUSION
	    if( !api::default_threshold().is_sparse( F, m ) ) {
		enable_fusion = true;
		if constexpr (
		    sizeof(typename LvlEnc::stored_type) != sizeof(VID) )
		    break; // transition to wider levels
	    }
#endif
	}

	// Are we done?
	return F.isEmpty();
    }
    
    template<typename LvlEnc, short aid_cur, short aid_prev>
    frontier
    edgemap_step( sVID iter,
		  bool & enable_fusion,
		  api::vertexprop<sVID,VID,aid_cur,LvlEnc> & a_level,
#if !LEVEL_ASYNC || DEFERRED_UPDATE
		  api::vertexprop<sVID,VID,aid_prev,LvlEnc> & a_prev_level,
#endif
		  frontier & F ) const {
	// Traverse edges, remove duplicate live destinations.
#if UNCOND_EXEC
	auto filter_strength = api::weak;
#else
	auto filter_strength = api::strong;
#endif
	frontier output;
	api::edgemap(
	    GA,
#if DEFERRED_UPDATE
	    // TODO: if not FUSION, compare a_level[d] to iters,
	    //       obviating the need for prev_level
	    // TODO: try reduction_or_method
	    api::record( output,
			 [&]( auto d ) {
			     return a_level[d] != a_prev_level[d]; 
			 }, api::strong ),
#else
	    api::record( output, api::reduction, api::strong ),
/*
* This frontier rule is correct, and does not require a_prev_level, however
* it is slower due to enabling too many vertices, in particular those whose
* level has been updated in a chain due to asynchronous and unconditional
* execution, but whose level is still too large to be final. It is more
* efficient to wake those up later. Selecting a_level[d] == _c( iter+1 )
* proves insufficient.
	    api::record( output,
			 [&]( auto d ) {
			     // A vertex is active if its level has
			     // grown to at least the current iteration:
			     // a_level[d] >= iter+1
			     return a_level[d] > _c( iter );
			 }, api::strong ),
*/
#endif
	    api::filter( filter_strength, api::src, F ),
#if CONVERGENCE
	    api::filter( api::weak, api::dst, [&]( auto d ) {
		// Use > comparison as it is faster than >= on AVX2
		return a_level[d] > _c( iter+1 );
	    } ),
#endif
#if FUSION
	    api::config( api::fusion_select( enable_fusion ) ),
	    api::fusion( [&]( auto s, auto d, auto e ) {
		return expr::cast<int>(
		    expr::iif( a_level[d].min( a_level[s]+_1 ),
			       _1s, // no change, ignore d
			       _1 // changed, propagate change
			) );
	    } ),
#endif
	    api::relax( [&]( auto s, auto d, auto e ) {
#if LEVEL_ASYNC
		return a_level[d].min( a_level[s] + _1 );
#else
		return a_level[d].min( a_prev_level[s] + _1 );
#endif
	    } )
	    )
		.materialize();
#if DEFERRED_UPDATE || !LEVEL_ASYNC
	const partitioner &part = GA.get_partitioner();
	maintain_copies( part, output, a_prev_level, a_level );
#endif
	return output;
    }
	    

private:
    const GraphType & GA;
    bool itimes, debug, calculate_active;
    sVID iter;
    VID start, active;
    api::vertexprop<sVID,VID,var_current8,Enc8> a_level8;
    api::vertexprop<sVID,VID,var_current16,Enc16> a_level16;
    api::vertexprop<sVID,VID,var_current,Enc> a_level;
#if !LEVEL_ASYNC || DEFERRED_UPDATE
    api::vertexprop<sVID,VID,var_previous8,Enc8> a_prev_level8;
    api::vertexprop<sVID,VID,var_previous16,Enc16> a_prev_level16;
    api::vertexprop<sVID,VID,var_previous,Enc> a_prev_level;
#endif
    std::vector<info> info_buf;
};

#ifndef NOBENCH
template <class GraphType>
using Benchmark = BFSLVLNarrow<GraphType>;

#include "driver.C"
#endif // NOBENCH
