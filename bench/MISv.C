#include "graptor/graptor.h"
#include "graptor/api.h"
#include "check.h"

// By default set options for highest performance
#ifdef LEVEL_ASYNC
#undef LEVEL_ASYNC
#endif
#define LEVEL_ASYNC 0

#ifdef DEFERRED_UPDATE
#undef DEFERRED_UPDATE
#endif
#define DEFERRED_UPDATE 0 // not applicable - edgemap calculates no frontier

#ifndef UNCOND_EXEC
#define UNCOND_EXEC 1
#endif

#ifdef CONVERGENCE
#undef CONVERGENCE
#endif
#define CONVERGENCE 0

#ifdef MEMO
#undef MEMO
#endif
#define MEMO 0

#ifndef MIS_DEBUG
#define MIS_DEBUG 0
#endif

#ifndef VARIANT
#define VARIANT 13
#endif

// mis_state_t (int) storage
#if VARIANT == 10
using StateTy = uint32_t;
using StoredTy = uint32_t;
using Enc = array_encoding<StoredTy>;
#endif

#if VARIANT == 11
using StateTy = uint8_t;
using StoredTy = uint8_t;
using Enc = array_encoding<StoredTy>;
#endif

#if VARIANT == 12
using StateTy = uint32_t;
using StoredTy = uint8_t;
using Enc = array_encoding_bit<2>;
#endif

#if VARIANT == 13
using StateTy = uint8_t;
using StoredTy = uint8_t;
using Enc = array_encoding_bit<2>;
#endif

#if VARIANT == 14
using StateTy = bitfield<2>;
using StoredTy = uint8_t;
using Enc = array_encoding_bit<2>;
#endif

#if VARIANT == 15
using StateTy = uint32_t;
using StoredTy = uint8_t;
using Enc = array_encoding_wide<StoredTy>;
#endif


template<unsigned short MaxVLCSC, typename StateTy, typename Enable = void>
struct MIS_frontier_spec {
    static constexpr frontier_type ftype = frontier_type::ft_logical4;
    using ftype_t = logical<4>;
};

template<typename StateTy>
struct MIS_frontier_spec<1,StateTy> {
    static constexpr frontier_type ftype = frontier_type::ft_bool;
    using ftype_t = bool;
};

template<unsigned short MaxVLCSC>
struct MIS_frontier_spec<MaxVLCSC,bitfield<2>,std::enable_if_t<MaxVLCSC!=1>> {
    static constexpr frontier_type ftype = frontier_type::ft_bit2;
    using ftype_t = bitfield<2>;
};

// This algorithm is adapted from ligra: https://github.com/jshun/ligra,
// which states:
//  "This is an implementation of the MIS algorithm from "Greedy
//   Sequential Maximal Independent Set and Matching are Parallel on
//   Average", Proceedings of the ACM Symposium on Parallelism in
//   Algorithms and Architectures (SPAA), 2012 by Guy Blelloch, Jeremy
//   Fineman and Julian Shun.  Note: this is not the most efficient
//   implementation. For a more efficient implementation, see
//   http://www.cs.cmu.edu/~pbbs/benchmarks.html."

/*
enum mis_state_t {
    mis_undecided = 0,
    mis_conditionally_in = 1,
    mis_out = 2,
    mis_in = 3
};
*/
using mis_state_t = expr::dfsaop_MIS::mis_state_t;

enum variable_name {
    var_flags = 0,
    var_frontier = 1,
    var_tmp
};

template <class GraphType>
class MISv {
public:
    __attribute__((noinline)) 
    MISv( GraphType & _GA, commandLine & P ) : GA( _GA ), info_buf( 60 ) {
	itimes = P.getOption( "-itimes" );
	debug = P.getOption( "-debug" );
	calculate_active = P.getOption( "-cactive" );

	static bool once = false;
	if( !once ) {
	    once = true;
	    std::cerr << "DEFERRED_UPDATE=" << DEFERRED_UPDATE << "\n";
	    std::cerr << "UNCOND_EXEC=" << UNCOND_EXEC << "\n";
	    std::cerr << "LEVEL_ASYNC=" << LEVEL_ASYNC << "\n";
	    std::cerr << "CONVERGENCE=" << CONVERGENCE << "\n";
	    std::cerr << "MEMO=" << MEMO << "\n";
	}
    }
    ~MISv() {
	m_flags.del();
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
		      << " (E:" << nacte
		      << ", V:" << nactv
		      << ") active: " << active << "\n";
	}
    };

    struct stat { };

    void run() {
	const partitioner &part = GA.get_partitioner();
	VID n = GA.numVertices();
	EID m = GA.numEdges();

	m_flags = Enc::allocate( numa_allocation_partitioned( part ) );
	expr::array_ro<StateTy,VID,var_flags,Enc> flags( m_flags.get() );
	auto vflags = flags.observe_stored();

	// Assign initial state
	make_lazy_executor( part )
	    .vertex_map( [&]( auto vid ) { return vflags[vid]
			= expr::constant_val2( vid,
					       (StoredTy)mis_state_t::mis_conditionally_in );
		} )
	    .materialize();

	iter = 0;

	using spec = MIS_frontier_spec<GraphType::getPullVLBound(),StateTy>;
	using ftype_t = typename spec::ftype_t;
	constexpr frontier_type ftype = spec::ftype;

	timer tm_iter;
	tm_iter.start();

	// Create initial frontier
	frontier F = frontier::all_true( n, m );

	while( !F.isEmpty() ) {  // iterate until IDs converge
#if MIS_DEBUG
	    if( F.getDenseL<4>() ) {
		VID na = 0;
		EID va = 0;
		std::cerr << "F     :";
		for( VID v=0; v < GA.numVertices(); ++v ) {
		    std::cerr << ( F.getDenseL<4>()[v] ? 1 : 0 );
		    if( F.getDenseL<4>()[v] ) {
			++na;
			va += GA.getOutDegree(v);
		    }
		}
		std::cerr << "\n";
	    }
#endif

	    // State array
	    expr::array_ro<StateTy, VID, var_flags, Enc> flags( m_flags );

	    // Infect states
	    frontier output;
	    api::edgemap(
		GA,
		api::relax( [&]( auto s, auto d, auto e_unused ) {
		    using Tr = simd::detail::mask_preferred_traits_type<
			StateTy, decltype(d)::VL>;
		    return expr::set_mask(
			s != d,
			expr::make_dfsaop_MIS(
			    flags[d], flags[s],
			    expr::make_unop_cvt_to_mask<Tr>( s < d ) ) );
			    } ),
		api::record( output,
			     [&]( auto d ) {
		    static_assert( (int)mis_state_t::mis_undecided  == 0,
				   "assumption" );
		    auto s_undecided = expr::value<
			simd::ty<StateTy,decltype(d)::VL>,expr::vk_zero>();
		    return flags[d] == s_undecided;
			     }, api::strong ),
		api::filter( F, api::dst, api::strong )
		)
		.materialize();

#if MIS_DEBUG
	    if( output.getType() != ftype ) // F is always dense for sure.
		output.toDense<ftype>( part );
	    assert( output.getType() == ftype );
	    // expr::array_ro<StateTy, VID, var_flags, Enc> flags( m_flags );
	    std::cerr << "edgemp:";
	    for( VID v=0; v < GA.numVertices(); ++v ) {
		switch( (mis_state_t)(int)flags[v].get() ) {
		case mis_state_t::mis_undecided:
		    std::cerr << '.';
		    break;
		case mis_state_t::mis_conditionally_in:
		    std::cerr << 'c';
		    break;
		case mis_state_t::mis_out:
		    std::cerr << 'o';
		    break;
		case mis_state_t::mis_in:
		    std::cerr << 'i';
		    break;
		}
	    }
	    std::cerr << "\n";
#endif

	    // And post-process state updates
	    make_lazy_executor( part )
		.vertex_map(
		    // GA, // Need graph for per-vertex degrees
		    F, // Only for those vertices in the frontier
		    // output, // Filtered frontier (sparse vs dense)
		    [&]( auto vid ) {
			using Ty = typename Enc::stored_type;
			auto vidx = expr::remove_mask( vid );
			auto s_undecided = expr::constant_val2(
			    vidx, (Ty)mis_state_t::mis_undecided );
			auto s_conditionally_in = expr::constant_val2(
			    vidx, (Ty)mis_state_t::mis_conditionally_in );
			auto s_out = expr::constant_val2(
			    vidx, (Ty)mis_state_t::mis_out );
			auto s_in = expr::constant_val2(
			    vidx, (Ty)mis_state_t::mis_in );

			return vflags[vid] = iif(
			    vflags[vid] == s_conditionally_in,
			    // false case
			    iif( vflags[vid] == s_out,
				 // false case
				 s_conditionally_in,
				 // true case
				 s_out ),
			    // true case
			    s_in );
		    } )
		.materialize();

#if MIS_DEBUG
	    std::cerr << "States:";
	    VID nin = 0;
	    for( VID v=0; v < GA.numVertices(); ++v ) {
		switch( (mis_state_t)(int)flags[v].get() ) {
		case mis_state_t::mis_undecided:
		    std::cerr << '.';
		    break;
		case mis_state_t::mis_conditionally_in:
		    std::cerr << 'c';
		    break;
		case mis_state_t::mis_out:
		    std::cerr << 'o';
		    break;
		case mis_state_t::mis_in:
		    ++nin;
		    std::cerr << 'i';
		    break;
		}
	    }
	    std::cerr << "\n";
	    std::cerr << "Number in: " << nin << "\n";

	    if( output.getDenseL<4>() ) {
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
	    }
	    if( output.getDense<frontier_type::ft_bit2>() ) {
		VID na = 0;
		EID va = 0;
		std::cerr << "output:";
		auto ptr = output.getDense<frontier_type::ft_bit2>();
		expr::array_ro<bitfield<2>,VID,var_flags,array_encoding_bit<2>> fr( ptr );
		for( VID v=0; v < GA.numVertices(); ++v ) {
		    std::cerr << ( ( fr[v].get() & 2 ) ? 1 : 0 );
		    if( fr[v].get() & 2 ) {
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

	    if( itimes ) {
		VID active = 0;
		if( calculate_active ) { // Warning: expensive, included in time
		    // for( VID v=0; v < n; ++v )
		    // if( condition ??? )
		    // // active++;
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

	    // Cleanup old frontier
	    F.del();
	    F = output;

	    iter++;
	}
	F.del();
    }

    void post_process( stat & stat_buf ) {
	if( itimes ) {
	    for( int i=0; i < iter; ++i )
		info_buf[i].dump( i );
	}

	if( outfile ) {
	    // recode( GA, m_IDs.get(), m_prevIDs.get() );
	    // writefile( GA, outfile, m_prevIDs, false ); // do not remap vertex IDs
	}
    }

    static void report( const std::vector<stat> & stat_buf ) { }

    void validate( stat & stat_buf, bool fail=true ) {
	// Validate sequentially, not using edgemap, because we need to
	// validate correctness of edgemap.
	const auto G = GA.getCSR();
	VID n = G.numVertices();
	EID m = G.numEdges();

	expr::array_ro<StateTy, VID, var_flags, Enc> flags( m_flags );

	VID ne_conflict = 0;
	VID ne_in = 0;
	VID ne_val = 0;
	VID n_in = 0;
	VID ne_selfl = 0;

	map_vertexL( GA.get_partitioner(), [&]( VID v ) {
		VID deg = G.getDegree( v );
		VID nconflict = 0, nin = 0, nval = 0, selfl = 0;
		for( VID i=0; i < deg; ++i ) {
		    VID w = G.getNeighbor( v, i );
		    int f_v = flags[v].get();
		    int f_w = flags[w].get();
		    if( v == w )
			++selfl;
		    else if( f_v == mis_state_t::mis_in && f_w == mis_state_t::mis_in )
			++nconflict;
		    else if( f_w == mis_state_t::mis_in )
			++nin;
		    if( f_v != mis_state_t::mis_in && f_v != mis_state_t::mis_out )
			++nval;
		}

		if( nconflict > 0 )
		    __sync_fetch_and_add( &ne_conflict, 1);
		if( (int)flags[v].get() != mis_state_t::mis_in && nin == 0 )
		    __sync_fetch_and_add( &ne_in, 1);
		if( (int)flags[v].get() == mis_state_t::mis_in )
		    __sync_fetch_and_add( &n_in, 1);
		if( nval > 0 )
		    __sync_fetch_and_add( &ne_val, 1);
		if( selfl > 0 )
		    __sync_fetch_and_add( &ne_selfl, 1);
	    } );

	std::cerr << "Validation:\n\tconflicts (IN-IN neighbours): "
		  << ne_conflict << "\n\tmissing (no IN neighbours): "
		  << ne_in << "\n\tnon-final values (conditional,undecided): "
		  << ne_val << "\n\tMIS size (IN values): "
		  << n_in << "\n\tself loops: "
		  << ne_selfl << "\n\tresult: "
		  << ( ne_conflict == 0 && ne_in == 0 && ne_val == 0
		       ? "PASS\n" : "FAIL\n" );

	if( fail ) {
	    assert( ne_conflict == 0 );
	    assert( ne_in == 0 );
	    assert( ne_val == 0 );
	}
    }

private:
    const GraphType & GA;
    bool itimes, debug, calculate_active;
    int iter;
    const char * outfile;
    mmap_ptr<StoredTy> m_flags;
    std::vector<info> info_buf;
};

template <class GraphType>
using Benchmark = MISv<GraphType>;

#include "driver.C"
