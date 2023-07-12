// -*- c++ -*-
/*!=====================================================================*
 * \brief A micro-benchmark for MCE on dense binary matrices.
 *
 * Test maximal clique enumeration (MCE) on cut-outs of the neighbourhoods
 * of vertices and compares algorithms at various sizes.
 *======================================================================*/

#include <mutex>
#include <numeric>
#include <thread>

#include <time.h>

#include "graptor/graptor.h"
#include "graptor/api.h"

#include "graptor/graph/simple/cutout.h"
#include "graptor/graph/simple/dense.h"
#include "graptor/container/hash_table.h"
#include "graptor/graph/simple/hadjt.h"
#include "graptor/container/hash_fn.h"
#include "graptor/stat/timing.h"

using hash_fn = graptor::rand_hash<uint32_t>;

#define NOBENCH
#define MD_ORDERING 0
#define VARIANT 11
#define FUSION 1
#include "../bench/KC_bucket.C"
#undef FUSION
#undef VARIANT
#undef MD_ORDERING
#undef NOBENCH

//! Number of size classes distinguished (32, 64, 128, 256, 512)
static constexpr size_t MAX_CLASS = 5;

//! Mutually exclusive access to timings
static std::mutex timings_mux;

//! Variables holding timing information
static graptor::distribution_timing timings[MAX_CLASS];

/*! Prestudy data
 */
static size_t levels;
static mmap_ptr<VID> prestudy;

/*! True if graph is remapped in non-increasing degeneracy order at start of
 *  program.
 */
static bool remap_graph = false;

/*! Generic driver method for MCE. Graph not a priory remapped.
 */
template<unsigned Bits, typename HGraph>
size_t
bench( const GraphCSx & G,
       const HGraph & H,
       VID v,
       const graptor::graph::NeighbourCutOutXP<VID,EID> & cut,
       const VID * const core_order ) {
#if USE_PRESTUDY
    graptor::graph::DenseMatrix<Bits,VID,EID>
	IG( G, H, v, cut.get_num_vertices(), cut.get_vertices(),
	    cut.get_s2g(), cut.get_n2s(), cut.get_start_pos(),
	    core_order, levels, prestudy,
	    graptor:: UNDERLYING () );
#else
    graptor::graph::DenseMatrix<Bits,VID,EID>
	IG( G, v, cut.get_num_vertices(), cut.get_vertices(),
	    cut.get_s2g(), cut.get_n2s(), cut.get_start_pos(),
	    core_order );
#endif
    size_t num_cliques = 0;
#if ONLY_CUTOUT
    __asm__ __volatile__ ( "" : : : "memory" );
#else
    IG.mce_bron_kerbosch( [&]( const bitset<Bits> &, size_t ) {
	++num_cliques;
    } );
#endif
    return num_cliques;
}

/*! Generic driver method for MCE. Graph a priory remapped.
 */
template<unsigned Bits, typename HGraph>
size_t
bench( const GraphCSx & G,
       const HGraph & H,
       VID v,
       const graptor::graph::NeighbourCutOutDegeneracyOrder<VID,EID> & cut,
       const VID * const core_order ) {
#if USE_PRESTUDY
    graptor::graph::DenseMatrix<Bits,VID,EID>
	IG( G, H, v, cut, levels, prestudy, graptor:: UNDERLYING () );
#else
    graptor::graph::DenseMatrix<Bits,VID,EID> IG(
	G, H, v, cut, graptor:: UNDERLYING () );
#endif
    size_t num_cliques = 0;
#if ONLY_CUTOUT
    __asm__ __volatile__ ( "" : : : "memory" );
#else
    IG.mce_bron_kerbosch( [&]( const bitset<Bits> &, size_t ) {
	++num_cliques;
    } );
#endif
    return num_cliques;
}


template<unsigned Bits, typename HGraph, typename CutOut>
void
mce_search( const GraphCSx & G,
	    const HGraph & H,
	    VID v,
	    const CutOut & cut,
	    const VID * const core_order,
	    size_t repeat,
	    size_t ref ) {
    timer tm;
    tm.start();
    for( int r=0; r < repeat; ++r ) {
	size_t num_cliques = bench<Bits>( G, H, v, cut, core_order );
	assert( ref == num_cliques && "ERROR" );
    }

    double t = tm.stop() / (double) repeat;

    constexpr unsigned b = ilog2( Bits ) - 5;
    assert( 0 <= b && b < 5 );

    std::lock_guard<std::mutex> g( timings_mux );
    timings[b].add_sample( t );
}

int main( int argc, char *argv[] ) {
    commandLine P( argc, argv, " help" );
    bool symmetric = P.getOptionValue("-s");
    remap_graph = P.getOptionValue("--remap");
    int repetitions = P.getOptionLongValue("-r", 3);
    VID min_size = P.getOptionLongValue("--min-size", 32);
    VID max_size = P.getOptionLongValue("--max-size", 512);
    levels = P.getOptionLongValue("--levels", 3);
    const char * ifile = P.getOptionValue( "-i" );

    if( max_size > 512 )
	max_size = 512;
    if( min_size > max_size )
	min_size = max_size;

    timer tm;
    tm.start();

    GraphCSx G( ifile, -1, symmetric );

    std::cerr << "Reading graph: " << tm.next() << "\n";

    VID n = G.numVertices();
    EID m = G.numEdges();

    assert( G.isSymmetric() );
    std::cerr << "Undirected graph: n=" << n << " m=" << m << std::endl;

    GraphCSRAdaptor GA( G, 256 );
    KCv<GraphCSRAdaptor> kcore( GA, P );
    kcore.run();
    auto & coreness = kcore.getCoreness();
    std::cerr << "coreness=" << kcore.getLargestCore() << "\n";
    std::cerr << "Calculating coreness: " << tm.next() << "\n";

    mm::buffer<VID> order( n, numa_allocation_interleaved() );
    mm::buffer<VID> rev_order( n, numa_allocation_interleaved() );
    sort_order( order.get(), rev_order.get(),
		coreness.get_ptr(), n, kcore.getLargestCore() );
    std::cerr << "Calculating degeneracy order: " << tm.next() << "\n";

    GraphCSx R( G, std::make_pair( order.get(), rev_order.get() ) );
    std::cerr << "Remapping graph: " << tm.next() << "\n";

    graptor::graph::GraphHAdjTable<VID,EID,hash_fn> H( n );
    const EID * const index = remap_graph ? R.getIndex() : G.getIndex();
    const VID * const edges = remap_graph ? R.getEdges() : G.getEdges();
    
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
	      << "\n  min_size: " << min_size
	      << "\n  max_size: " << max_size
	      << "\n  levels: " << levels
	      << "\n  remap: " << ( remap_graph ? "yes" : "no" )
	      << "\n";

#if ONLY_CUTOUT
    std::cerr << "  Only timing cutout operation\n";
#endif

    if( remap_graph ) {
	parallel_loop( VID(0), n, [&]( VID v ) {
	    graptor::graph::NeighbourCutOutDegeneracyOrder<VID,EID> cut( R, v );
	    size_t num = cut.get_num_vertices();
	    size_t rep = repetitions;
	    if( min_size < num && num <= max_size ) {
		size_t ref = bench<512>( R, H, v, cut, nullptr );
		if( num <= 32 )
		    mce_search<32>( R, H, v, cut, nullptr, rep, ref );
		if( num <= 64 )
		    mce_search<64>( R, H, v, cut, nullptr, rep, ref );
		if( num <= 128 )
		    mce_search<128>( R, H, v, cut, nullptr, rep, ref );
		if( num <= 256 )
		    mce_search<256>( R, H, v, cut, nullptr, rep, ref );
		if( num <= 512 )
		    mce_search<512>( R, H, v, cut, nullptr, rep, ref );
	    }
	} );
    } else {
	parallel_loop( VID(0), n, [&]( VID i ) {
	    VID v = order[i];

	    graptor::graph::NeighbourCutOutXP<VID,EID> cut( G, v, rev_order.get() );
	    size_t num = cut.get_num_vertices();
	    size_t rep = repetitions;
	    if( min_size < num && num <= max_size ) {
		size_t ref = bench<512>( G, H, v, cut, rev_order.get() );
		if( num <= 32 )
		    mce_search<32>( G, H, v, cut, rev_order.get(), rep, ref );
		if( num <= 64 )
		    mce_search<64>( G, H, v, cut, rev_order.get(), rep, ref );
		if( num <= 128 )
		    mce_search<128>( G, H, v, cut, rev_order.get(), rep, ref );
		if( num <= 256 )
		    mce_search<256>( G, H, v, cut, rev_order.get(), rep, ref );
		if( num <= 512 )
		    mce_search<512>( G, H, v, cut, rev_order.get(), rep, ref );
	    }
	} );
	}

    std::cerr << "Results:\n";

    for( size_t r=0; r < MAX_CLASS; ++r ) {
	std::cerr << (size_t(1)<<(r+5)) << ' '
		  << timings[r].characterize( .95, 1000, 100 )
		  << '\n';
    }

    rev_order.del();
    order.del();
    R.del();
    G.del();

    return 0;
}
