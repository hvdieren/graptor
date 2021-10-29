#include <math.h>
#include <random>

#include "graptor/graptor.h"
#include "graptor/api.h"
#include "check.h"

// By default set options for highest performance
#ifndef UNCOND_EXEC
#define UNCOND_EXEC 1
#endif

#ifndef DEFERRED_UPDATE
#define DEFERRED_UPDATE 1
#endif

#ifndef CONVERGENCE
#define CONVERGENCE 1
#endif

// Level-asynchrony is not allowed because we need to be able
// to measure how the estimated set sizes vary when increasing the number
// of hops
#ifdef LEVEL_ASYNC
#undef LEVEL_ASYNC
#endif
#define LEVEL_ASYNC 0

#ifdef MEMO
#undef MEMO
#endif
#define MEMO 0

#include "graptor/longint.h"
#include "graptor/target/vt_longint.h"

// Code is defunct with K>1 because of gcc 7.2.0 error at -O2 and above
// that removes live code for calculating the new frontier. Code correct
// however when DEFERRED_UPDATES is on.
// -- not sure how relevant this still is.
static const VID K = 1; // 32;

// We create individual bit masks of size of VID, as this allows to
// count all elements in the set.
using BitMaskTy = VID;
static const size_t L = sizeof(BitMaskTy)*8;

// The actual strings used internally during the graph operations are
// K times longer.
using BitStringTy = BitMaskTy; // longint<sizeof(BitMaskTy)*K>;

// Bit masks are laid out in memory with K consecutive bit masks for each
// vertex. This maximizes memory locality.

// Find lowest-order bit set to 1 in a bit mask.
static BitMaskTy rho( BitMaskTy x ) {
    BitMaskTy mask = 1;
    for( size_t i=0; i < L; ++i ) { // how about if x has L low zero bits?
	if( x & mask )
	    return mask;
	mask <<= 1;
    }
    assert( mask == 0 && "L bits not set" );
    return mask;
}

// Estimate the size of a vertex's neighbourhood. Access all K bitmaps for
// this vertex, placed consecutively in memory.
// To vectorize, need to replace by trailing zero count
template<unsigned int NumMasks>
static float estimate_size( const BitMaskTy * bitmap ) {
    BitMaskTy S = 0;
    for( size_t k=0; k < NumMasks; ++k ) {
	BitMaskTy mask = BitMaskTy(1);
	if( bitmap[k] == 0 ) // unlikely
	    S += ~BitMaskTy(0);
	else {
	    for( size_t R=0; R < L; ++R ) {
		if( ( bitmap[k] & mask ) == 0 ) {
		    S += mask;
		    break;
		}
		mask <<= 1;
	    }
	}
    }
    const double phi = 0.77351;
    float sz = (double(1)/phi) * double(S)/double(NumMasks);
    return sz;
}

static std::mt19937_64 generator;

// From https://oeis.org/A000043
template<unsigned short Bits>
struct MersenneP;

template<>
struct MersenneP<16> {
    static constexpr BitMaskTy value = 13;
};

template<>
struct MersenneP<32> {
    static constexpr BitMaskTy value = 31;
};

template<>
struct MersenneP<64> {
    static constexpr BitMaskTy value = 61;
};

template<>
struct MersenneP<128> {
    static constexpr BitMaskTy value = 127;
};

static BitMaskTy FMHashMult( VID v ) {
    // Multiply vertex index, offset by one, by a large prime.
    // This counteracts the tendency of std:: hash function to map
    // small numbers onto themselves.
    constexpr BitMaskTy Pn = MersenneP<sizeof(BitMaskTy)*8>::value;
    constexpr BitMaskTy P = (BitMaskTy(1)<<Pn)-1;
    BitMaskTy vv = P * (BitMaskTy(v)+1);
    BitMaskTy h = std::hash<BitMaskTy>{}( vv );
    BitMaskTy mask = rho(h);
    return BitMaskTy(mask);
}

static BitMaskTy FMHashRandom( VID v ) {
    static_assert( sizeof(std::mt19937_64::result_type) >= sizeof(BitMaskTy),
		   "Need sufficient number of bits in the random number "
		   "generator" );
    std::uniform_int_distribution<std::mt19937_64::result_type>
	dist(0, ~BitMaskTy(0) );
    BitMaskTy h = dist( generator );
    BitMaskTy r = BitMaskTy(1) << (L-1);
    BitMaskTy ar = r >> 1;
    // Assuming random numbers are uniformly distributed over the range
    // of bitmask values, return a bit set in position 0 with 50% probability,
    // a bit in position 1 with 25% probability, etc.
    for( BitMaskTy i=0; i < L; ++i ) {
	if( h < r )
	    return BitMaskTy(1) << i;
	r += ar;
	ar >>= 1;
    }
    return 0; // extremely unlikely but possible in theory
}

static BitMaskTy FMHash( VID v ) {
    // Choose one.
    return FMHashRandom( v );
}

// Main edge-map operation: bitwise-OR bit masks
// Based on:
// Flajolet, P. and Martin, G. N. 1985. Probabilistic counting algorithms
// for data base applications. Journal of Computer and System Sciences.
// Kang, Tsourakakis, Appel, Faloutsos and Leskovec. 2011. HADI: Mining Radii
// of Lare Graphs. ACM TKD.
struct FM_F
{
    expr::array_ro<BitStringTy, VID, 0> new_mask;
    expr::array_ro<BitStringTy,VID, 1> prev_mask;

    FM_F(BitMaskTy* _new_mask, BitMaskTy* _prev_mask)
	: new_mask( reinterpret_cast<BitStringTy *>(_new_mask) ),
	  prev_mask( reinterpret_cast<BitStringTy *>(_prev_mask) ) {}

    // How do we calculate the new frontier?
#if DEFERRED_UPDATE
    static constexpr frontier_mode new_frontier = fm_calculate;
#else
    static constexpr frontier_mode new_frontier = fm_reduction;
#endif

    static constexpr bool is_scan = false;
    static constexpr bool is_idempotent = true;
    static constexpr bool new_frontier_dense = false;

#if UNCOND_EXEC
    static constexpr bool may_omit_frontier_rd = true;
#else
    static constexpr bool may_omit_frontier_rd = false;
#endif
    static constexpr bool may_omit_frontier_wr = true;

    template<typename VIDSrc, typename VIDDst>
    auto relax( VIDSrc s, VIDDst d ) {
	return new_mask[d] |= prev_mask[s];
    }

    template<typename VIDDst>
    auto different( VIDDst d ) {
	return expr::true_val(d);
    }

    template<typename VIDDst>
    auto update( VIDDst d ) {
	return new_mask[d] != prev_mask[d];
    }

    template<typename VIDDst>
    auto active( VIDDst d ) {
#if CONVERGENCE
	return ( new_mask[d] != expr::constant_val2(d, ~BitStringTy(0)) );
#else
	return expr::true_val(d);
#endif
    }

    template<typename VIDDst>
    auto vertexop( VIDDst d ) {
	return expr::make_noop();
    }
};

class EstSize {
    float * estsz;
    const BitMaskTy * new_mask;
    BitMaskTy * prev_mask;

public:
    EstSize( float * estsz_,
	     const BitMaskTy * new_mask_, BitMaskTy * prev_mask_ )
	: estsz( estsz_ ), new_mask( new_mask_ ), prev_mask( prev_mask_ ) { }

    bool operator()( VID v ) {
	for( int k=0; k < K; ++k )
	    prev_mask[v*K+k] = new_mask[v*K+k];
	estsz[v] = estimate_size<K>( &new_mask[v*K] );
	return true;
    }
};

template <class GraphType>
class FMv {
public:
    FMv( GraphType & _GA, commandLine & P ) : GA( _GA ), info_buf( 60 ) {
	itimes = P.getOption( "-itimes" );
	debug = P.getOption( "-debug" );
	calculate_active = P.getOption( "-cactive" );

	generator.seed( generator.default_seed );

	static bool once = false;
	if( !once ) {
	    once = true;
	    std::cerr << "DEFERRED_UPDATE=" << DEFERRED_UPDATE << "\n";
	    std::cerr << "UNCOND_EXEC=" << UNCOND_EXEC << "\n";
	    std::cerr << "LEVEL_ASYNC=" << LEVEL_ASYNC << " (n/a)\n";
	    std::cerr << "CONVERGENCE=" << CONVERGENCE << "\n";
	    std::cerr << "MEMO=" << MEMO << "\n";
	    std::cerr << "K=" << K << "\n";
	}
    }
    ~FMv() {
	new_mask.del();
	prev_mask.del();
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
	const partitioner Kpart = part.scale( K );
	VID n = GA.numVertices();
	EID m = GA.numEdges();

	new_mask.allocate( numa_allocation_partitioned( Kpart ) );
	prev_mask.allocate( numa_allocation_partitioned( Kpart ) );

	expr::array_ro<BitStringTy, VID, 0> a_new_mask( new_mask );
	expr::array_ro<BitStringTy,VID, 1> a_prev_mask( prev_mask );

	mmap_ptr<float> estsz;
	estsz.allocate( numa_allocation_partitioned( part ) );

	// Assign initial labels -- in serial for repeatability
	map_vertex_serialL( part, [&]( VID v ) {
		for( int k=0; k < K; ++k ) {
		    BitMaskTy m = FMHash( v * K + k );
		    new_mask[v*K+k] = prev_mask[v*K+k] = m;
		}
		estsz[v] = 0;
	    } );

	// Create initial frontier
	frontier F = frontier::all_true( n, m ); // all active

	iter = 0;

	timer tm_iter;
	tm_iter.start();

	std::vector<float> Nhm1i( 32 );

	while( !F.isEmpty() ) {  // iterate until bit strings stabilise
#if 0
	    std::cerr << "masks:";
	    for( VID v=0; v < n*K; ++v )
		std::cerr << ' ' << new_mask.get()[v];
	    std::cerr << "\n";
	    vertexMap( part, F,
		       EstSize( estsz.get(),
				new_mask.get(), prev_mask.get() )
		);
	    std::cerr << "sizes:";
	    for( VID v=0; v < n; ++v )
		std::cerr << ' ' << estsz.get()[v];
	    std::cerr << "\n";
#endif

	    // Propagate bit strings
	    frontier output;
#if 0
	    vEdgeMap( GA, F, output, FM_F( new_mask, prev_mask ) )
		.materialize();
#else
#if UNCOND_EXEC
	    auto filter_strength = api::weak;
#else
	    auto filter_strength = api::strong;
#endif
	    api::edgemap(
		GA,
#if DEFERRED_UPDATE
		api::record( output,
			     [&] ( auto d ) {
				 return a_new_mask[d] != a_prev_mask[d]; },
			     api::strong ),
#else
		api::record( output, api::reduction, api::strong ),
#endif
		api::filter( filter_strength, api::src, F ),
#if CONVERGENCE
		api::filter( api::weak, api::dst,
			     [&] ( auto d ) {
				 return ( a_new_mask[d] != expr::constant_val2(d, ~BitStringTy(0)) );
			     } ),
#endif
		api::relax( [&]( auto s, auto d, auto e ) {
				return a_new_mask[d] |= a_prev_mask[s];
		    } )
		)
		.materialize();
#endif

#if 0
	    VID nv = 0;
	    EID ne = 0;
	    VID nvf = 0;
	    EID nef = 0;
	    output.toDense<ft_logical4>( part );
	    logical<4> * f = output.getDense<ft_logical4>();
	    for( VID v=0; v < n; ++v ) {
		if( new_mask.get()[v] != prev_mask.get()[v] ) {
		    nv++;
		    ne += GA.getOutDegree(v);
		}
		if( f && f[v] != 0 ) {
		    nvf++;
		    nef += GA.getOutDegree(v);
		}
	    }
	    std::cerr << "F: v=" << output.nActiveVertices() << " e=" << output.nActiveEdges() << "\n";
	    std::cerr << "f: v=" << nvf << " e=" << nef << "\n";
	    std::cerr << "M: v=" << nv << " e=" << ne << "\n";
#endif

	    vertexMap( part, // output,
		       EstSize( estsz.get(),
				new_mask.get(), prev_mask.get() )
		);
	    Nhm1i.resize( iter+1 );
	    Nhm1i[iter] = sequence::plusReduce( estsz.get(), n );

	    if( itimes ) {
		VID active = 0;
		if( calculate_active ) { // Warning: expensive, included in time
		    for( VID v=0; v < n; ++v )
			if( prev_mask[v] != ~BitMaskTy(0) )
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

	    // Cleanup old frontier
	    F.del();
	    F = output;

	    iter++;
	}

	// Need to consider float rounding
	float Nmax = Nhm1i[iter-1];
	float Nihi = Nmax;
	float Nh = Nmax;
	float Nhm1;
	int i;
	for( i=iter-2; i >= 0; --i ) {
	    Nhm1 = Nhm1i[i];
	    if( Nhm1 < 0.9 * Nmax )
		break;
	    Nh = Nhm1;
	    std::cerr << "i=" << i << " Nh=" << Nh << "\n";
	}

	if( debug ) {
	    std::cerr << "Nmax: " << Nmax << "\nNh: " << Nh
		      << "\nNhm1: " << Nhm1 << "\ni: " << i
		      << "\n";
	}

	if( Nh == Nhm1 )
	    deffG = 1;
	else
	    deffG = i + (0.9 * Nmax - Nhm1) / (Nh - Nhm1);

	std::cerr << "Estimates for vertex v=0:";
	for( unsigned i=0; i < K; ++i )
	    std::cerr << ' ' << estimate_size<1>( new_mask.get() );
	std::cerr << '\n';

	estsz.del();
    }

    void post_process( stat & stat_buf ) {
	if( itimes ) {
	    for( int i=0; i < iter; ++i )
		info_buf[i].dump( i );
	}

	std::cerr << "deffG: " << deffG << "\n";
    }

    static void report( const std::vector<stat> & stat_buf ) { }

    void validate( stat & stat_buf ) {
    }

private:
    const GraphType & GA;
    bool itimes, debug, calculate_active;
    int iter;
    mmap_ptr<BitMaskTy> new_mask, prev_mask;
    std::vector<info> info_buf;
    float deffG;
};

template <class GraphType>
using Benchmark = FMv<GraphType>;

#include "driver.C"
