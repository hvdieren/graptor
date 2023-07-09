// -*- c++ -*-
/*!=====================================================================*
 * \brief A micro-benchmark for MCE on blocked binary matrices.
 *
 * Test maximal clique enumeration (MCE) on cut-outs of the neighbourhoods
 * of vertices and compares algorithms at various sizes.
 *======================================================================*/

/*!=====================================================================*
 * TODO:
 * + Create a variant with building matrix only to split out time building
 *   from time searching. Building time is relatively high. Might be faster
 *   for very small problems to directly work on the main representation.
 * + Construct an optimised variant, picking the fastest variations as
 *   applicable.
 *======================================================================*/

#include <mutex>
#include <numeric>
#include <thread>

#include <time.h>

#include "graptor/graptor.h"
#include "graptor/api.h"

#include "graptor/graph/simple/cutout.h"
#include "graptor/graph/simple/blocked.h"
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

//! Variables holding timing information, X and P parts
static graptor::distribution_timing timings[MAX_CLASS][MAX_CLASS];

/*! Generic driver method for MCE.
 */
template<unsigned XBits, unsigned PBits>
size_t
bench( const GraphCSx & G,
       VID v,
       const graptor::graph::NeighbourCutOutXP<VID,EID> & cut,
       const VID * const core_order ) {
    graptor::graph::BlockedBinaryMatrix<XBits,PBits,VID,EID>
	IG( G, v, cut.get_num_vertices(), cut.get_vertices(),
	    cut.get_s2g(), cut.get_n2s(), cut.get_start_pos(),
	    core_order );
    size_t num_cliques = 0;
#if ONLY_CUTOUT
    __asm__ __volatile__ ( "" : : : "memory" );
#else
    mce_bron_kerbosch( IG, [&]( const bitset<PBits> & c ) { ++num_cliques; } );
#endif
    return num_cliques;
}

template<unsigned XBits, unsigned PBits>
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
	size_t num_cliques = bench<XBits,PBits>( G, v, cut, core_order );
	assert( ref == num_cliques && "ERROR" );
    }

    double t = tm.stop() / (double) repeat;

    constexpr unsigned bx = ilog2( XBits ) - 5;
    assert( 0 <= bx && bx < 5 );
    constexpr unsigned bp = ilog2( PBits ) - 5;
    assert( 0 <= bp && bp < 5 );

    std::lock_guard<std::mutex> g( timings_mux );
    timings[bx][bp].add_sample( t );
}

int main( int argc, char *argv[] ) {
    commandLine P( argc, argv, " help" );
    bool symmetric = P.getOptionValue("-s");
    int repetitions = P.getOptionLongValue("-r", 3);
    VID xmin_size = P.getOptionLongValue("--xmin-size", 32);
    VID xmax_size = P.getOptionLongValue("--xmax-size", 512);
    VID pmin_size = P.getOptionLongValue("--pmin-size", 32);
    VID pmax_size = P.getOptionLongValue("--pmax-size", 512);
    const char * ifile = P.getOptionValue( "-i" );

    if( xmax_size > 512 )
	xmax_size = 512;
    if( xmin_size > xmax_size )
	xmin_size = xmax_size;
    if( pmax_size > 512 )
	pmax_size = 512;
    if( pmin_size > pmax_size )
	pmin_size = pmax_size;

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
	      << "\n  xmin_size: " << xmin_size
	      << "\n  xmax_size: " << xmax_size
	      << "\n  pmin_size: " << pmin_size
	      << "\n  pmax_size: " << pmax_size
	      << "\n";

#if ONLY_CUTOUT
    std::cerr << "  Only timing cutout operation\n";
#endif

    parallel_loop( VID(0), n, [&]( VID i ) {
	VID v = order[i];
	
	graptor::graph::NeighbourCutOutXP<VID,EID> cut( G, v, rev_order.get() );
	size_t num = cut.get_num_vertices();
	size_t xnum = cut.get_start_pos();
	size_t pnum = num - xnum;
	if( xmin_size < xnum && xnum <= xmax_size
	    && pmin_size < pnum && pnum <= pmax_size ) {
	    size_t ref = bench<512,512>( G, v, cut, rev_order.get() );
	    size_t rep = repetitions;
	    if( xnum <= 32 && pnum <= 32 )
		mce_search<32,32>( G, v, cut, rev_order.get(), rep, ref );
	    if( xnum <= 32 && pnum <= 64 )
		mce_search<32,64>( G, v, cut, rev_order.get(), rep, ref );
	    if( xnum <= 32 && pnum <= 128 )
		mce_search<32,128>( G, v, cut, rev_order.get(), rep, ref );
	    if( xnum <= 32 && pnum <= 256 )
		mce_search<32,256>( G, v, cut, rev_order.get(), rep, ref );
	    if( xnum <= 32 && pnum <= 512 )
		mce_search<32,512>( G, v, cut, rev_order.get(), rep, ref );
	    if( xnum <= 64 && pnum <= 32 )
		mce_search<64,32>( G, v, cut, rev_order.get(), rep, ref );
	    if( xnum <= 64 && pnum <= 64 )
		mce_search<64,64>( G, v, cut, rev_order.get(), rep, ref );
	    if( xnum <= 64 && pnum <= 128 )
		mce_search<64,128>( G, v, cut, rev_order.get(), rep, ref );
	    if( xnum <= 64 && pnum <= 256 )
		mce_search<64,256>( G, v, cut, rev_order.get(), rep, ref );
	    if( xnum <= 64 && pnum <= 512 )
		mce_search<64,512>( G, v, cut, rev_order.get(), rep, ref );
	    if( xnum <= 128 && pnum <= 32 )
		mce_search<128,32>( G, v, cut, rev_order.get(), rep, ref );
	    if( xnum <= 128 && pnum <= 64 )
		mce_search<128,64>( G, v, cut, rev_order.get(), rep, ref );
	    if( xnum <= 128 && pnum <= 128 )
		mce_search<128,128>( G, v, cut, rev_order.get(), rep, ref );
	    if( xnum <= 128 && pnum <= 256 )
		mce_search<128,256>( G, v, cut, rev_order.get(), rep, ref );
	    if( xnum <= 128 && pnum <= 512 )
		mce_search<128,512>( G, v, cut, rev_order.get(), rep, ref );
	    if( xnum <= 256 && pnum <= 32 )
		mce_search<256,32>( G, v, cut, rev_order.get(), rep, ref );
	    if( xnum <= 256 && pnum <= 64 )
		mce_search<256,64>( G, v, cut, rev_order.get(), rep, ref );
	    if( xnum <= 256 && pnum <= 128 )
		mce_search<256,128>( G, v, cut, rev_order.get(), rep, ref );
	    if( xnum <= 256 && pnum <= 256 )
		mce_search<256,256>( G, v, cut, rev_order.get(), rep, ref );
	    if( xnum <= 256 && pnum <= 512 )
		mce_search<256,512>( G, v, cut, rev_order.get(), rep, ref );
	    if( xnum <= 512 && pnum <= 32 )
		mce_search<512,32>( G, v, cut, rev_order.get(), rep, ref );
	    if( xnum <= 512 && pnum <= 64 )
		mce_search<512,64>( G, v, cut, rev_order.get(), rep, ref );
	    if( xnum <= 512 && pnum <= 128 )
		mce_search<512,128>( G, v, cut, rev_order.get(), rep, ref );
	    if( xnum <= 512 && pnum <= 256 )
		mce_search<512,256>( G, v, cut, rev_order.get(), rep, ref );
	    if( xnum <= 512 && pnum <= 512 )
		mce_search<512,512>( G, v, cut, rev_order.get(), rep, ref );
	}
    } );

    std::cerr << "Results:\n";

    for( size_t l=0; l < MAX_CLASS; ++l ) {
	for( size_t r=0; r < MAX_CLASS; ++r ) {
	    std::cerr << "x=" << (size_t(1)<<(l+5)) << ','
		      << "p=" << (size_t(1)<<(r+5)) << ' '
		      << timings[l][r].characterize( .95, 1000, 100 )
		      << '\n';
	}
    }

    rev_order.del();
    order.del();
    G.del();

    return 0;
}
