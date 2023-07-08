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
#include "graptor/stat/timing.h"

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

/*! Generic driver method for MCE.
 */
template<unsigned Bits>
size_t
bench( const GraphCSx & G,
       VID v,
       const graptor::graph::NeighbourCutOutXP<VID,EID> & cut,
       const VID * const core_order ) {
    graptor::graph::DenseMatrix<Bits,VID,EID>
	IG( G, v, cut.get_num_vertices(), cut.get_vertices(),
	    cut.get_s2g(), cut.get_n2s(), cut.get_start_pos(),
	    core_order );
    size_t num_cliques = 0;
#if ONLY_CUTOUT
    volatile __asm__( "" : : : "memory" );
#else
    IG.mce_bron_kerbosch( [&]( const bitset<Bits> & c ) { ++num_cliques; } );
#endif
    return num_cliques;
}

template<unsigned Bits>
void
mce_search( const GraphCSx & G,
	    VID v,
	    const graptor::graph::NeighbourCutOutXP<VID,EID> & cut,
	    const VID * const core_order,
	    size_t repeat,
	    size_t ref ) {
    timer tm;
    tm.start();
    for( int r=0; r < repeat; ++r ) {
	size_t num_cliques = bench<Bits>( G, v, cut, core_order );
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
    int repetitions = P.getOptionLongValue("-r", 3);
    VID min_size = P.getOptionLongValue("--min-size", 32);
    VID max_size = P.getOptionLongValue("--max-size", 512);
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
    const EID * const index = G.getIndex();
    const VID * const edges = G.getEdges();

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

    std::cerr << "Configuration:"
	      << "\n  repetitions: " << repetitions
	      << "\n  min_size: " << min_size
	      << "\n  max_size: " << max_size
	      << "\n";

#if ONLY_CUTOUT
    std::cerr << "  Only timing cutout operation\n";
#endif

    parallel_loop( VID(0), n, [&]( VID i ) {
	VID v = order[i];
	
	graptor::graph::NeighbourCutOutXP<VID,EID> cut( G, v, rev_order.get() );
	size_t num = cut.get_num_vertices();
	if( min_size < num && num <= max_size ) {
	    size_t ref = bench<512>( G, v, cut, rev_order.get() );
	    if( num <= 32 )
		mce_search<32>( G, v, cut, rev_order.get(), repetitions, ref );
	    if( num <= 64 )
		mce_search<64>( G, v, cut, rev_order.get(), repetitions, ref );
	    if( num <= 128 )
		mce_search<128>( G, v, cut, rev_order.get(), repetitions, ref );
	    if( num <= 256 )
		mce_search<256>( G, v, cut, rev_order.get(), repetitions, ref );
	    if( num <= 512 )
		mce_search<512>( G, v, cut, rev_order.get(), repetitions, ref );
	}
    } );

    std::cerr << "Results:\n";

    for( size_t r=0; r < MAX_CLASS; ++r ) {
	std::cerr << (size_t(1)<<(r+5)) << ' '
		  << timings[r].characterize( .95, 1000, 100 )
		  << '\n';
    }

    rev_order.del();
    order.del();
    G.del();

    return 0;
}
