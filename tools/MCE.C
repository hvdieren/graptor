// -*- c++ -*-
// Specialised to MCE

// TODO:
// * online machine learning
// * MCE_Enumerator thread-local (no sync-fetch-and-add)

// Novelties:
// + find pivot -> abort intersection if seen to be too small
// + small sub-problems -> dense matrix; O(1) operations

// Experiments:
// + Check that 32-bit is faster than 64-bit for same-sized problems;
//   same for SSE vs AVX

// A Pattern Decomposed Graph
#include <signal.h>
#include <sys/time.h>

#include <thread>
#include <mutex>
#include <map>
#include <exception>
#include <numeric>

#include <pthread.h>

#include <cilk/cilk.h>

#include "graptor/graptor.h"
#include "graptor/api.h"

#include "graptor/graph/contract/vertex_set.h"
#include "graptor/graph/GraphCSx.h"
#include "graptor/graph/GraphPDG.h"
#include "graptor/graph/partitioning.h"

#include "graptor/graph/simple/csx.h"
#include "graptor/graph/simple/dicsx.h"
#include "graptor/graph/simple/hadj.h"
#include "graptor/graph/simple/hadjt.h"

#include "graptor/container/bitset.h"

#define NOBENCH
#define MD_ORDERING 0
#define VARIANT 11
#define FUSION 1
#include "../bench/KC_bucket.C"
#undef FUSION
#undef VARIANT
#undef MD_ORDERING
#undef NOBENCH

enum cvt_pdg_variable_name {
    var_dcount = var_kc_num + 0,
    var_max = var_kc_num + 1,
    var_min = var_kc_num + 2,
    var_priority = var_kc_num + 3
};

static bool verbose = false;

template<unsigned Bits, typename sVID, typename sEID>
class DenseMatrix;

template<typename T>
struct murmur_hash;

template<>
struct murmur_hash<uint64_t> {
    using type = uint64_t;
    
    type operator()( type h ) const {
	h ^= h >> 33;
	h *= 0xff51afd7ed558ccdL;
	h ^= h >> 33;
	h *= 0xc4ceb9fe1a85ec53L;
	h ^= h >> 33;
	return h;
    }
};

template<>
struct murmur_hash<uint32_t> {
    using type = uint32_t;
    
    type operator()( type h ) const {
	h ^= h >> 16;
	h *= 0x85ebca6b;
	h ^= h >> 13;
	h *= 0xc2b2ae35;
	h ^= h >> 16;

	return h;
    }

    template<unsigned short VL>
    typename vector_type_traits_vl<uint32_t,VL>::type
    vectorized( typename vector_type_traits_vl<uint32_t,VL>::type h ) const {
	using tr = vector_type_traits_vl<type,VL>;
	using vtype = typename tr::type;
	const vtype c1 = tr::set1( 0x85ebca6b );
	const vtype c2 = tr::set1( 0xc2b2ae35 );
	h = tr::bitwise_xor( h, tr::srli( h, 16 ) );
	h = tr::mul( h, c1 );
	h = tr::bitwise_xor( h, tr::srli( h, 13 ) );
	h = tr::mul( h, c2 );
	h = tr::bitwise_xor( h, tr::srli( h, 16 ) );
	return h;
    }
};

template<typename T>
struct java_hash;

template<>
struct java_hash<uint32_t> {
    using type = uint32_t;

    type operator() ( uint32_t h ) const {
	h ^= (h >> 20) ^ (h >> 12);
	return h ^ (h >> 7) ^ (h >> 4);
    }

    template<unsigned short VL>
    typename vector_type_traits_vl<uint32_t,VL>::type
    vectorized( typename vector_type_traits_vl<uint32_t,VL>::type h ) const {
	using tr = vector_type_traits_vl<type,VL>;
	using vtype = typename tr::type;

	vtype h20 = tr::srli( h, 20 );
	vtype h12 = tr::srli( h, 12 );
	h = tr::bitwise_xor( h20, tr::bitwise_xor( h, h12 ) );
	vtype h7 = tr::srli( h, 7 );
	vtype h4 = tr::srli( h, 4 );
	h = tr::bitwise_xor( h4, tr::bitwise_xor( h, h4 ) );
	return h;
    }
};

void
sort_order( VID * order, VID * rev_order,
	    const VID * const coreness,
	    VID n,
	    VID K ) {
    VID * histo = new VID[K+1];
    std::fill( &histo[0], &histo[K+1], 0 );

    // Histogram
    for( VID v=0; v < n; ++v ) {
	VID c = coreness[v];
	assert( c <= K );
	histo[K-c]++;
    }

    // Prefix sum
    VID sum = sequence::plusScan( histo, histo, K+1 );

    // Place in order
    for( VID v=0; v < n; ++v ) {
	VID c = coreness[v];
	VID pos = histo[K-c]++;
	order[pos] = v;
	rev_order[v] = pos;
    }

    delete[] histo;
}

class timeout_exception : public std::exception {
public:
    explicit timeout_exception( uint64_t usec = 0, int idx = -1 )
	: m_usec( usec ), m_idx( idx ) { }
    timeout_exception( const timeout_exception & e )
	: m_usec( e.m_usec ), m_idx( e.m_idx ) { }
    timeout_exception & operator = ( const timeout_exception & e ) {
	m_idx = e.m_idx;
	m_usec = e.m_usec;
	return *this;
    }

    uint64_t usec() const noexcept { return m_usec; }
    int idx() const noexcept { return m_idx; }

    const char * what() const noexcept {
	return "timeout exception";
    }

private:
    uint64_t m_usec;
    int m_idx;
};

bool is_member( VID v, VID C_size, const VID * C_set ) {
    const VID * const pos = std::lower_bound( C_set, C_set+C_size, v );
    if( pos == C_set+C_size || *pos != v )
	return false;
    return true;
}

template<typename VID, typename EID, typename Hash>
VID mc_get_pivot_xp(
    const graptor::graph::GraphHAdjTable<VID,EID,Hash> & G,
    const VID * XP,
    VID ne,
    VID ce ) {

    assert( ce - ne != 0 );

    // Tunable (|P| and selecting vertex from X or P)
    if( ce - ne <= 3 )
	return XP[ne];

    VID v_max = ~VID(0);
    VID tv_max = std::numeric_limits<VID>::min();

    for( VID i=0; i < ce; ++i ) {
	VID v = XP[i];
	// VID v = XP[ce-1-i]; -- slower
	// VID v = XP[(i+ne)%ce]; -- makes no difference
	auto & hadj = G.get_adjacency( v );
	VID deg = hadj.size();
	if( deg <= tv_max )
	    continue;

	// Abort during intersection_size if size will be less than tv_max
	size_t tv = hadj.intersect_size( XP+ne, XP+ce, tv_max );
	if( tv > tv_max ) {
	    tv_max = tv;
	    v_max = v;
	}
    }

    // return first element of P if nothing good found
    return ~v_max == 0 ? XP[ne] : v_max;
}

#if 0
class StackLikeAllocator {
public:
    StackLikeAllocator( size_t min_chunk_size = 0 ) { }

    template<typename T>
    T * allocate( size_t n_elements ) {
	return new T[n_elements];
    }
    template<typename T>
    void deallocate_to( T * p ) {
	delete[] p;
    }
};
#else
class StackLikeAllocator {
    static constexpr size_t PAGE_SIZE = size_t(1) << 20; // 1MiB
    struct chunk_t {
	static constexpr size_t MAX_BYTES =
	    ( size_t(1) << (8*sizeof(uint32_t)) ) - 2 * sizeof( uint32_t );

	chunk_t( size_t sz ) : m_size( sz ), m_end( 0 ) {
	    assert( sz <= MAX_BYTES );
	    assert( sz == (size_t)m_size );
	    assert( (size_t)m_size >= PAGE_SIZE );
	}

	char * allocate( size_t sz ) {
	    assert( (size_t)m_size >= PAGE_SIZE );
	    assert( sz <= MAX_BYTES );
	    assert( m_end + sz <= m_size );
	    char * p = get_ptr() + m_end;
	    m_end += sz;
	    return p;
	}

	bool has_available_space( size_t sz ) const {
	    assert( (size_t)m_size >= PAGE_SIZE );
	    assert( sz <= MAX_BYTES );
	    uint32_t new_end = m_end + sz;
	    return new_end - m_end == sz && new_end <= m_size;
	    
	}

	char * get_ptr() const {
	    char * me = const_cast<char *>(
		reinterpret_cast<const char *>( this ) );
	    me += sizeof( m_size );
	    me += sizeof( m_end );
	    return me;
	}

	bool release_to( char * p ) {
	    assert( (size_t)m_size >= PAGE_SIZE );
	    char * q = get_ptr();
	    if( q <= p && p < q+m_end ) {
		m_end = p - q;
		return true;
	    } else {
		m_end = 0;
		return false;
	    }
	}

    private:
	uint32_t m_size;
	uint32_t m_end;
    };

public:
    StackLikeAllocator( size_t min_chunk_size = PAGE_SIZE )
	: m_min_chunk_size( min_chunk_size ), m_current( 0 ) {
	if( verbose )
	    std::cerr << "sla " << this << ": constructor\n";
    }
    ~StackLikeAllocator() {
	for( chunk_t * c : m_chunks ) {
	    if( verbose )
		std::cerr << "sla " << this << ": delete chunk "
			  << c << "\n";
	    delete[] reinterpret_cast<char *>( c );
	}
	if( verbose )
	    std::cerr << "sla " << this << ": destructor done\n";
    }

    template<typename T>
    T * allocate( size_t n_elements ) {
	size_t sz = n_elements * sizeof(T);
	sz = ( sz + 3 ) & ~size_t(3); // multiple of 4 bytes
	T * p = reinterpret_cast<T*>( allocate_private( sz ) );
	// if( verbose )
	// std::cerr << "sla " << this << ": allocate " << n_elements
	// << ' ' << (void *)p << "\n";
	return p;
    }
    template<typename T>
    void deallocate_to( T * p ) {
	// if( verbose )
	// std::cerr << "sla " << this << ": deallocate-to "
	// << (void *)p << "\n";
	release_chunks( reinterpret_cast<char *>( p ) );
    }

private:
    char * allocate_private( size_t nbytes ) {
	// Do we have any available chunks?
	if( m_chunks.empty() )
	    return allocate_from_new_chunk( nbytes );
	
	// Check if any free chunk has sufficient space
	// Might be better to insert a larger chunk in the sequence if
	// the current cannot hold it, as future calls will require smaller
	// allocations which may be served from the available chunks
	do {
	    chunk_t * c = m_chunks[m_current];
	    if( c->has_available_space( nbytes ) )
		return c->allocate( nbytes );
	} while( ++m_current < m_chunks.size() );

	// No chunk can hold this
	return allocate_from_new_chunk( nbytes );
    }

    char * allocate_from_new_chunk( size_t nbytes ) {
	size_t sz = std::max( nbytes, m_min_chunk_size );
	sz = ( sz + PAGE_SIZE - 1 ) & ~( PAGE_SIZE - 1 );
	assert( sz >= nbytes );
	char * cc = new char[sz];
	chunk_t * c = new ( cc ) chunk_t( sz );
	m_chunks.push_back( c );
	m_current = m_chunks.size() - 1;
	if( verbose )
	    std::cerr << "sla " << this << ": new chunk " << c << "\n";
	return c->allocate( nbytes );
    }

    void release_chunks( char * p ) {
	for( size_t i=0; i <= m_current; ++i ) {
	    size_t j = m_current - i;
	    chunk_t * const c = m_chunks[j];
	    if( c->release_to( p ) ) {
		m_current = j;
		return;
	    }
	}
	assert( false && "deallocation error - should not reach here" );
    }
    
private:
    std::vector<chunk_t *> m_chunks;
    size_t m_min_chunk_size;
    size_t m_current;
};
#endif

template<typename T, typename S>
S insert_sorted( T * p, S sz, T u ) {
    T * q = std::lower_bound( p, p+sz, u );
    if( q == p+sz || *q != u ) { // not already present
	std::copy_backward( q, p+sz, p+sz+1 );
	*q = u;
	return sz+1;
    }
    return sz;
}

template<typename T, typename S>
S remove_sorted( T * p, S sz, T u ) {
    T * q = std::lower_bound( p, p+sz, u );
    if( q != p+sz && *q == u ) {
	std::copy( q+1, p+sz, q );
	return sz-1;
    }
    return sz;
}

void
check_clique( const graptor::graph::GraphDoubleIndexCSx<VID,EID> & G,
	      VID size,
	      VID * clique ) {
    std::sort( clique, clique+size );
    VID n = G.numVertices();
    const EID * const bindex = G.getBeginIndex();
    const EID * const eindex = G.getEndIndex();
    const VID * const edges = G.getEdges();

    for( VID i=0; i < size; ++i ) {
	VID v = clique[i];
	for( VID j=0; j < size; ++j ) {
	    if( j == i )
		continue;
	    VID u = clique[j];
	    const VID * const pos
		= std::lower_bound( &edges[bindex[v]], &edges[eindex[v]], u );
	    if( pos == &edges[eindex[v]] || *pos != u )
		abort();
	}
    }
}

void
check_clique( const GraphCSx & G,
	      VID size,
	      VID * clique ) {
    std::sort( clique, clique+size );
    VID n = G.numVertices();
    const EID * const index = G.getIndex();
    const VID * const edges = G.getEdges();

    VID v0 = clique[0];
    contract::vertex_set<VID> ins;
    ins.push( &edges[index[v0]], &edges[index[v0+1]] );

    for( VID i=1; i < size; ++i ) {
	VID v = clique[i];
	for( VID j=0; j < size; ++j ) {
	    if( j == i )
		continue;
	    VID u = clique[j];
	    const VID * const pos
		= std::lower_bound( &edges[index[v]], &edges[index[v+1]], u );
	    if( pos == &edges[index[v+1]] || *pos != u )
		abort();
	}
	ins = ins.intersect( &edges[index[v]], index[v+1] - index[v] );
    }
    assert( ins.size() == 0 ); // check if maximal
}

bool
is_maximal_clique(
    const graptor::graph::GraphCSx<VID,EID> & G,
    VID size,
    VID * clique ) {
    // std::sort( clique, clique+size );
    VID n = G.numVertices();
    const EID * const index = G.getIndex();
    const VID * const edges = G.getEdges();

    VID vs = clique[size-1];
    contract::vertex_set<VID> ins;
    ins.push( &edges[index[vs]], &edges[index[vs+1]] );

    for( VID i=0; i < size-1; ++i ) {
	VID v = clique[size-2-i];
	ins = ins.intersect( &edges[index[v]], index[v+1] - index[v] );
	if( ins.size() == 0 )
	    break;
    }
    return ins.size() == 0;
}

bool
is_maximal_clique(
    const graptor::graph::GraphCSx<VID,EID> & G,
    contract::vertex_set<VID> & R ) {
    return is_maximal_clique( G, R.size(), &*R.begin() );
}

bool is_subset( VID S_size, const VID * S_set,
		VID C_size, const VID * C_set ) {
    for( VID i=0; i < S_size; ++i ) {
	VID v = S_set[i];
	if( !is_member( v, C_size, C_set ) )
	    return false;
    }
    return true;
}

std::atomic<size_t> pruning;

template<typename Enumerate, typename VID, typename EID, typename Hash>
void
mce_iterate_xp(
    const graptor::graph::GraphHAdjTable<VID,EID,Hash> & G,
    Enumerate && Ee,
    StackLikeAllocator & alloc,
    contract::vertex_set<VID> & R,
    VID * XP,
    VID ne, // not edges
    VID ce, // candidate edges
    int depth ) {
    if( ce == ne ) {
	if( ne == 0 )
	    Ee( R );
	return;
    }

    // pivot...
    VID pivot = mc_get_pivot_xp( G, XP, ne, ce );
    auto & p_ngh = G.get_adjacency( pivot );

    // If space is an issue, could put alloc/dealloc inside loop and tune
    // space depending on neighbour list length (each of X and P can not
    // be longer than number of neighbours of u, nor longer than their
    // current size, so allocate std::min( ce, degree(u) ).
    VID * XP_new = alloc.template allocate<VID>( ce );
    VID * prev_tgt = XP;

    for( VID i=ne; i < ce; ++i ) {
	VID u = XP[i];

	// Is u in the neighbour list of the pivot? If so, skip
	if( p_ngh.contains( u ) )
	    continue;

	auto & adj = G.get_adjacency( u );
	VID deg = adj.size();
	VID ne_new = adj.intersect( XP, XP+ne, XP_new ) - XP_new;
	VID ce_new = adj.intersect( XP+ne, XP+ce, XP_new+ne_new ) - XP_new;
	assert( ce_new <= ce );
	R.push( u );

	// Tunable
	if( ce_new - ne_new < 16 || ce_new > 256 ) {
	    mce_iterate_xp( G, Ee, alloc, R, XP_new, ne_new, ce_new, depth+1 );
	} else if( ce_new <= 32 ) {
	    DenseMatrix<32,VID,EID> D( G, XP_new, ne_new, ce_new );
	    D.mce_bron_kerbosch( [&]( const bitset<32> & c ) {
		Ee( R, c.size() );
	    } );
	} else if( ce_new <= 64 ) {
	    DenseMatrix<64,VID,EID> D( G, XP_new, ne_new, ce_new );
	    D.mce_bron_kerbosch( [&]( const bitset<64> & c ) {
		Ee( R, c.size() );
	    } );
	} else if( ce_new <= 128 ) {
	    DenseMatrix<128,VID,EID> D( G, XP_new, ne_new, ce_new );
	    D.mce_bron_kerbosch( [&]( const bitset<128> & c ) {
		Ee( R, c.size() );
	    } );
	} else if( ce_new <= 256 ) {
	    DenseMatrix<256,VID,EID> D( G, XP_new, ne_new, ce_new );
	    D.mce_bron_kerbosch( [&]( const bitset<256> & c ) {
		Ee( R, c.size() );
	    } );
	} else {
	    assert( 0 && "Should not get here" );
	}

	R.pop();

	// Move candidate (u) from original position to appropriate
	// place in X part (maintaining sort order).
	// Cache tgt for next iteration as next iteration's u
	// will be strictly larger.
	VID * tgt = std::upper_bound( prev_tgt, XP+ne, u );
	if( tgt != &XP[i] ) { // equality when u moves to tgt == XP+ne
	    std::copy_backward( tgt, &XP[i], &XP[i+1] );
	    *tgt = u;
	}
	prev_tgt = tgt+1;
	++ne;
    }

    alloc.deallocate_to( XP_new );
}

template<typename Enumerate, typename VID, typename EID, typename Hash>
void
mce_iterate_xp_iterative(
    const graptor::graph::GraphHAdjTable<VID,EID,Hash> & G,
    Enumerate && Ee,
    StackLikeAllocator & alloc,
    contract::vertex_set<VID> & R,
    VID degeneracy,
    VID * XP,
    VID ne, // not edges
    VID ce ) { // candidate edges

    struct frame_t {
	VID * XP;
	VID ne;
	VID ce;
	VID pivot;
	VID * XP_new;
	VID * prev_tgt;
	VID i;
    };

    // Base case - trivial problem
    if( ce == ne ) {
	if( ne == 0 )
	    Ee( R );
	return;
    }

    frame_t * frame = new frame_t[degeneracy+1];
    int depth = 1;

    // If space is an issue, could put alloc/dealloc inside loop and tune
    // space depending on neighbour list length (each of X and P can not
    // be longer than number of neighbours of u, nor longer than their
    // current size, so allocate std::min( ce, degree(u) ).

    new ( &frame[0] ) frame_t { XP, ne, ce, 0, nullptr, XP, ne };
    frame[0].pivot = mc_get_pivot_xp( G, XP, ne, ce );
    frame[0].XP_new = alloc.template allocate<VID>( ce );

    while( depth >= 1 ) {
	frame_t & fr = frame[depth-1];
	assert( fr.ce >= fr.ne );

	// Loop iteration control
	if( fr.i >= fr.ce ) {
	    // Pop frame
	    assert( depth > 0 );
	    assert( R.size() == depth );
	    --depth;
	    alloc.deallocate_to( fr.XP_new );
	    fr.XP_new = 0;

	    // Finish off vertex in higher-level frame
	    if( depth > 0 ) {
		frame_t & fr = frame[depth-1];
		VID u = fr.XP[fr.i];
		
		R.pop();

		// Move candidate (u) from original position to appropriate
		// place in X part (maintaining sort order).
		// Cache tgt for next iteration as next iteration's u
		// will be strictly larger.
		VID * tgt = std::upper_bound( fr.prev_tgt, fr.XP+fr.ne, u );
		if( tgt != &fr.XP[fr.i] ) { // equality when u moves to tgt == XP+ne
		    std::copy_backward( tgt, &fr.XP[fr.i], &fr.XP[fr.i+1] );
		    *tgt = u;
		}
		fr.prev_tgt = tgt+1;
		++fr.ne;
		++fr.i;
	    }
	    continue;
	}

	// Next step on frame
	VID u = fr.XP[fr.i];

	// Is u in the neighbour list of the pivot? If so, skip
	auto & p_ngh = G.get_adjacency( fr.pivot );
	if( !p_ngh.contains( u ) ) {
	    auto & adj = G.get_adjacency( u );
	    VID deg = adj.size();
	    VID * XP = fr.XP;
	    VID ne = fr.ne;
	    VID ce = fr.ce;
	    VID * XP_new = fr.XP_new;
	    VID ne_new = adj.intersect( XP, XP+ne, XP_new ) - XP_new;
	    VID ce_new = adj.intersect( XP+ne, XP+ce, XP_new+ne_new ) - XP_new;
	    assert( ce_new <= ce );
	    R.push( u );
	    assert( R.size() == depth+1 );

	    // Recursion, check base case (avoid pushing on stack)
	    if( ce_new == ne_new ) {
		if( ne_new == 0 )
		    Ee( R );
		// done
	    // Tunable
	    } else if( ce_new - ne_new < 16 || ce_new > 256 ) {
		// mce_iterate_xp( G, Ee, alloc, R, XP_new, ne_new, ce_new, depth+1 );
		// Recursion - push new frame
		assert( depth+1 < degeneracy+1 );
		frame_t & nfr = frame[depth++];
		new ( &nfr ) frame_t {
		    fr.XP_new, ne_new, ce_new, 0, nullptr, nullptr, ne_new };

		nfr.pivot = mc_get_pivot_xp( G, XP_new, ne_new, ce_new );
		nfr.XP_new = alloc.template allocate<VID>( ce_new );
		nfr.prev_tgt = XP_new;
		assert( R.size() == depth );
		// Go to handle top frame
		continue;
	    } else if( ce_new <= 32 ) {
		DenseMatrix<32,VID,EID> D( G, XP_new, ne_new, ce_new );
		D.mce_bron_kerbosch( [&]( const bitset<32> & c ) {
		    Ee( R, c.size() );
		} );
		// done
	    } else if( ce_new <= 64 ) {
		DenseMatrix<64,VID,EID> D( G, XP_new, ne_new, ce_new );
		D.mce_bron_kerbosch( [&]( const bitset<64> & c ) {
		    Ee( R, c.size() );
		} );
		// done
	    } else if( ce_new <= 128 ) {
		DenseMatrix<128,VID,EID> D( G, XP_new, ne_new, ce_new );
		D.mce_bron_kerbosch( [&]( const bitset<128> & c ) {
		    Ee( R, c.size() );
		} );
		// done
	    } else if( ce_new <= 256 ) {
		DenseMatrix<256,VID,EID> D( G, XP_new, ne_new, ce_new );
		D.mce_bron_kerbosch( [&]( const bitset<256> & c ) {
		    Ee( R, c.size() );
		} );
		// done
	    } else {
		assert( 0 && "Should not get here" );
	    }

	    R.pop();

	    // Move candidate (u) from original position to appropriate
	    // place in X part (maintaining sort order).
	    // Cache tgt for next iteration as next iteration's u
	    // will be strictly larger.
	    VID * tgt = std::upper_bound( fr.prev_tgt, XP+ne, u );
	    if( tgt != &XP[fr.i] ) { // equality when u moves to tgt == XP+ne
		std::copy_backward( tgt, &XP[fr.i], &XP[fr.i+1] );
		*tgt = u;
	    }
	    fr.prev_tgt = tgt+1;
	    ++fr.ne;
	}

	++fr.i;
    }

    // alloc.deallocate_to( frame[0].XP_new );
    assert( frame[0].XP_new == 0 );
    delete[] frame;
}


template<typename lVID, typename lEID>
class NeighbourCutOut {
public:
    using VID = lVID;
    using EID = lEID;

public:
    // For maximal clique enumeration: all vertices regardless of coreness
    // Sort neighbour list in increasing order
    NeighbourCutOut( const GraphCSx & G,
		     VID v )
	: NeighbourCutOut( G, v, G.getIndex()[v+1] - G.getIndex()[v] ) { }
    NeighbourCutOut( const GraphCSx & G,
		     VID v,
		     VID deg )
	: m_iset( deg ), m_totdeg( 0 ), m_maxdeg( 0 ), m_num_iset( 0 ) {
	const EID * const index = G.getIndex();
	const VID * const edges = G.getEdges();

	for( EID e=index[v], ee=index[v+1]; e != ee; ++e ) {
	    VID u = edges[e];

	    if( u == v ) // self-edge
		continue;

	    VID udeg = 0;
// TODO: redundant loop??? (prunes #vertices - successful?)
	    for( EID f=index[u], ff=index[u+1]; f != ff; ++f ) {
		VID w = edges[f];
		// Filter vertices (duplicates)
		// if( coreness[w] > coreness[v] )
		// continue;
	    
		if( w == v ) // v is not included in cutout
		    continue;
		const VID * pos
		    = std::lower_bound( &edges[index[v]], &edges[index[v+1]],
					w );
		if( pos != &edges[index[v+1]] && *pos == w )
		    ++udeg;
	    }

	    // The calculated degrees are not precise as we may consider
	    // edges to neighbours that are later discarded because
	    // those neighbours later turn out to have insufficient
	    // degree themselves. As such, udeg is an upper bound to the
	    // actual degree of the vertex.
	    // m_maxdeg, m_totdeg and m_degrees are therefore also upper
	    // bounds.
	    m_iset[m_num_iset] = u;
	    m_totdeg += udeg;
	    ++m_num_iset;
	    if( udeg > m_maxdeg )
		m_maxdeg = udeg;
	}

	assert( m_num_iset <= deg );
    }

    VID get_max_degree() const { return m_maxdeg; }
    VID get_num_vertices() const { return m_num_iset; }
    const VID * get_vertices() const { return &m_iset[0]; }

private:
    std::vector<VID> m_iset;
    EID m_totdeg;
    VID m_maxdeg;
    VID m_num_iset;
};

template<typename lVID, typename lEID>
class NeighbourCutOutAll {
public:
    using VID = lVID;
    using EID = lEID;

public:
    // For maximal clique enumeration: all vertices regardless of coreness
    // Sort neighbour list in increasing order
    NeighbourCutOutAll( const GraphCSx & G, VID v )
	: NeighbourCutOutAll( G, v, G.getIndex()[v+1] - G.getIndex()[v] ) { }
    NeighbourCutOutAll( const GraphCSx & G, VID v, VID deg )
	: m_iset( &G.getEdges()[G.getIndex()[v]] ),
	  m_num_iset( deg ) { }

    VID get_num_vertices() const { return m_num_iset; }
    const VID * get_vertices() const { return m_iset; }

private:
    const VID * m_iset;
    VID m_num_iset;
};


template<typename GraphType>
class GraphBuilderInduced;

// For maximal clique enumeration - all vertices regardless of coreness
// Sort/relabel vertices by decreasing coreness
// TODO: make hybrid between hash table / adj list for lowest degrees?
template<typename VID, typename EID, typename Hash>
class GraphBuilderInduced<graptor::graph::GraphHAdjTable<VID,EID,Hash>> {
public:
    GraphBuilderInduced( const GraphCSx & G,
			 VID v,
			 const NeighbourCutOut<VID,EID> & cut,
			 const VID * const core_order )
	: GraphBuilderInduced( G, v, cut.get_num_vertices(), cut.get_vertices(),
			       core_order ) { }
    GraphBuilderInduced( const GraphCSx & G,
			 VID v,
			 const NeighbourCutOutAll<VID,EID> & cut,
			 const VID * const core_order )
	: GraphBuilderInduced( G, v, cut.get_num_vertices(), cut.get_vertices(),
			       core_order ) { }
    GraphBuilderInduced( const GraphCSx & G,
			 VID v,
			 VID num_neighbours,
			 const VID * neighbours,
			 const VID * const core_order )
	: s2g( &neighbours[0], &neighbours[num_neighbours] ),
	  S( num_neighbours ) {
	VID n = G.numVertices();
	EID m = G.numEdges();
	const EID * const gindex = G.getIndex();
	const VID * const gedges = G.getEdges();

	// Indices:
	// global (g): original graph's vertex IDs
	// short (s): relabeled vertex IDs in induced graph S
	// neighbours (n): position of vertex in neighbour list, which is
	//                 sorted by global IDs, facilitating lookup
	VID ns = num_neighbours;
	VID * n2s = new VID[ns];
	std::iota( &s2g[0], &s2g[ns], 0 );

	// Sort by increasing core_order
	std::sort( &s2g[0], &s2g[ns],
		   [&]( VID u, VID v ) {
		       return core_order[neighbours[u]]
			   < core_order[neighbours[v]];
		   } );
	// Invert permutation into n2s and create mapping for m_s2g
	for( VID su=0; su < ns; ++su ) {
	    VID x = s2g[su];
	    s2g[su] = neighbours[x]; // create mapping
	    n2s[x] = su; // invert permutation
	}

	// Find first vertex ordered after the reference vertex v.
	// All preceeding vertices have already been tried and can be skipped.
	// We include them in the cutout however in order to construct the
	// excluded set (X) in Bron-Kerbosch. Remains a question if we need
	// to build the neighbour list for the excluded vertices (probably
	// only for the purpose of finding a pivot vertex).
	VID * sp2_pos = std::upper_bound(
	    &s2g[0], &s2g[ns], v,
	    [&]( VID a, VID b ) {
		return core_order[a] < core_order[b];
	    } );
	start_pos = sp2_pos - &s2g[0];

	for( VID su=0; su < ns; ++su ) {
	    VID u = s2g[su];
	    VID k = 0;
	    auto & adj = S.get_adjacency( su );
	    for( EID e=gindex[u], ee=gindex[u+1]; e != ee && k != ns; ++e ) {
		VID w = gedges[e];
		while( k != ns && neighbours[k] < w )
		    ++k;
		if( k == ns )
		    break;
		// If neighbour is selected in cut-out, add to induced graph.
		// Skip self-edges.
		// Skip edges between vertices in X.
		if( neighbours[k] == w && w != u
		    && ( su >= start_pos || n2s[k] >= start_pos ) )
		    adj.insert( n2s[k] );
	    }
	}

	S.sum_up_edges();

	delete[] n2s;
    }

    const VID * get_s2g() const { return &s2g[0]; }
    const auto & get_graph() const { return S; }
    VID get_start_pos() const { return start_pos; }

private:
    graptor::graph::GraphHAdjTable<VID,EID,Hash> S;
    std::vector<VID> s2g;
    VID start_pos;
};

template<typename Enumerate, typename VID, typename EID, typename Hash>
void
mce_bron_kerbosch_seq_xp(
    const graptor::graph::GraphHAdjTable<VID,EID,Hash> & G,
    VID start_pos,
    VID degeneracy,
    Enumerate && E ) {
    VID n = G.numVertices();

    StackLikeAllocator alloc;

    // start_pos calculate to avoid revisiting vertices ordered before the
    // reference vertex of this cut-out
    for( VID v=start_pos; v < n; ++v ) {
	contract::vertex_set<VID> R;

	R.push( v );

	// Consider as candidates only those neighbours of v that are larger
	// than u to avoid revisiting the vertices unnecessarily.
	auto & adj = G.get_adjacency( v ); 

	VID deg = adj.size();
	VID * XP = alloc.template allocate<VID>( deg );
	auto end = std::copy_if(
	    adj.begin(), adj.end(), XP,
	    [&]( VID v ) { return v != adj.invalid_element; } );
	assert( end - XP == deg );
	std::sort( XP, XP+deg );
	const VID * const start = std::upper_bound( XP, XP+deg, v );
	VID ne = start - XP;
	VID ce = deg;

	// mce_iterate_xp( G, E, alloc, R, XP, ne, ce, 1 );
	mce_iterate_xp_iterative( G, E, alloc, R, degeneracy, XP, ne, ce );

	if( ce > 0 )
	    alloc.deallocate_to( XP );
    }
}

template<typename Enumerate, typename VID, typename EID, typename Hash>
void
mce_bron_kerbosch_par_xp(
    const graptor::graph::GraphHAdjTable<VID,EID,Hash> & G,
    VID start_pos,
    VID degeneracy,
    Enumerate && E ) {
    VID n = G.numVertices();

    // start_pos calculate to avoid revisiting vertices ordered before the
    // reference vertex of this cut-out
    parallel_loop( start_pos, n, 1, [&]( VID v ) {
	StackLikeAllocator alloc;
	contract::vertex_set<VID> R;

	R.push( v );

	// Consider as candidates only those neighbours of v that are larger
	// than u to avoid revisiting the vertices unnecessarily.
	auto & adj = G.get_adjacency( v ); 

	VID deg = adj.size();
	VID * XP = alloc.template allocate<VID>( deg );
	auto end = std::copy_if(
	    adj.begin(), adj.end(), XP,
	    [&]( VID v ) { return v != adj.invalid_element; } );
	assert( end - XP == deg );
	std::sort( XP, XP+deg );
	const VID * const start = std::upper_bound( XP, XP+deg, v );
	VID ne = start - XP;
	VID ce = deg;

	// mce_iterate_xp( G, E, alloc, R, XP, ne, ce, 1 );
	mce_iterate_xp_iterative( G, E, alloc, R, degeneracy, XP, ne, ce );
    } );
}

void check_clique_edges( EID m, const VID * assigned_clique, EID ce ) {
    EID cce = 0;
    for( EID e=0; e != m; ++e )
	if( ~assigned_clique[e] != 0 )
	    ++cce;
    assert( cce == ce );
}

class TimeLimitedExecution {
    struct thread_info {
	timeval m_expired_time;
	volatile bool m_termination_flag;
	bool m_active;
	std::mutex m_lock;
    };

public:
    static TimeLimitedExecution & getInstance() {
	// Guaranteed to be destroyed and instantiated on first use.
	static TimeLimitedExecution instance;
	return instance;
    }
private:
    TimeLimitedExecution() : m_terminated( false ), m_thread( guard_thread ) {
	// install_signal_handler();
	// set_timer();
    }
    ~TimeLimitedExecution() {
	m_terminated = true; // causes guard_thread to terminate
	m_thread.join(); // wait until it has terminated
	// clear_timer();
	// remove_signal_handler();
    }

public:
    TimeLimitedExecution( TimeLimitedExecution const& ) = delete;
    void operator = ( TimeLimitedExecution const& )  = delete;

public:
    template<typename Fn, typename... Args>
    static auto execute_time_limited( uint64_t usec, Fn && fn, Args && ... args ) {
	// The singleton object
	TimeLimitedExecution & tlexec = getInstance();
	
	// Who am I?
	pthread_t self = pthread_self();

	// Look up my record
	thread_info & ti = tlexec.m_thread_info[self];

	// Check current time and calculate expiry time
	if( gettimeofday( &ti.m_expired_time, NULL ) < 0 ) {
	    std::cerr << "Error getting current time: "
		      << strerror( errno ) << "\n";
	    exit( 1 );
	}

	// Lock the record
	{
	    std::lock_guard<std::mutex> g( ti.m_lock );
	    uint64_t mln = 1000000ull;
	    ti.m_expired_time.tv_sec += usec / mln;
	    ti.m_expired_time.tv_usec += usec % mln;
	    if( ti.m_expired_time.tv_usec >= mln ) {
		ti.m_expired_time.tv_sec
		    += ti.m_expired_time.tv_usec / mln;
		ti.m_expired_time.tv_usec
		    = ti.m_expired_time.tv_usec % mln;
	    }

	    // Set active
	    ti.m_termination_flag = false;
	    ti.m_active = true;
	} // releases lock

	// std::cerr << "set to expire at sec=" << ti.m_expired_time.tv_sec
	// << " usec=" << ti.m_expired_time.tv_usec << "\n";

	decltype( fn( &ti.m_termination_flag, args... ) ) ret;
	try {
	    ret = fn( &ti.m_termination_flag, args... );
	} catch( const timeout_exception & e ) {
	    // std::cerr << "reached timeout; invalid result\n";

	    // Disable - no need to lock
	    ti.m_active = false;

	    // Rethrow exception
	    throw timeout_exception( usec );
	}
	
	// Disable - no need to lock
	ti.m_active = false;

	return ret;
    }

private:
    static void guard_thread() {
	getInstance().process_loop();
    }
    static void alarm_signal_handler( int ) {
	getInstance().process_periodically();
    }
    
    void process_loop() {
	while( !m_terminated ) {
	    std::this_thread::sleep_for( 10us );
	    process_periodically();
	}
    }
    
    void process_periodically() {
	// Lock map
	std::lock_guard<std::mutex> g( m_lock );
	
	// Get gurrent time
	timeval now;
	if( gettimeofday( &now, NULL ) < 0 ) {
	    std::cerr << "Error getting current time: "
		      << strerror( errno ) << "\n";
	    exit( 1 );
	}
	
	for( auto & tip : m_thread_info ) {
	    thread_info & ti = tip.second;

	    // Avoid deadlock in case we are manipulating the record in the
	    // same thread that executes the signal handler. If the record
	    // is being manipulated, then the computation is not in progress
	    // and need not be interrupted.
	    if( ti.m_lock.try_lock() ) {
		std::lock_guard<std::mutex> g( ti.m_lock, std::adopt_lock );
		if( !ti.m_active )
		    continue;
		if( ti.m_expired_time.tv_sec < now.tv_sec
		    || ( ti.m_expired_time.tv_sec == now.tv_sec
			 && ti.m_expired_time.tv_usec < now.tv_usec ) ) {
		    ti.m_termination_flag = true;

		    /*
		    std::cerr << "set to expire at sec=" << ti.m_expired_time.tv_sec
			      << " usec=" << ti.m_expired_time.tv_usec << "\n";
		    std::cerr << "triggering at sec=" << now.tv_sec
			      << " usec=" << now.tv_usec << "\n";
		    */
		}
	    }
	}
    }

    void install_signal_handler() {
	struct sigaction act;

	act.sa_handler = alarm_signal_handler;
	act.sa_flags = 0;
	
	int ret = sigaction( SIGALRM, &act, NULL );
	if( ret < 0 ) {
	    std::cerr << "Error setting signal handler: "
		      << strerror( errno ) << "\n";
	    exit( 1 );
	}
    }

    void remove_signal_handler() {
	struct sigaction act;

	act.sa_handler = SIG_DFL;
	act.sa_flags = 0;
	
	int ret = sigaction( SIGALRM, &act, NULL );
	if( ret < 0 ) {
	    std::cerr << "Error removing signal handler: "
		      << strerror( errno ) << "\n";
	    exit( 1 );
	}
    }

    void set_timer() {
	struct itimerval when;

	when.it_interval.tv_sec = 0;
	when.it_interval.tv_usec = 100000;
	when.it_value.tv_sec = 0;
	when.it_value.tv_usec = 100000;

	int ret = setitimer( ITIMER_REAL, &when, NULL );
	if( ret < 0 ) {
	    std::cerr << "Error setting timer: "
		      << strerror( errno ) << "\n";
	    exit( 1 );
	}

	ret = getitimer( ITIMER_REAL, &when );
    }

    void clear_timer() {
	struct itimerval when;

	when.it_interval.tv_sec = 0;
	when.it_interval.tv_usec = 0;
	when.it_value.tv_sec = 0;
	when.it_value.tv_usec = 0;

	int ret = setitimer( ITIMER_REAL, &when, NULL );
	if( ret < 0 ) {
	    std::cerr << "Error clearing timer: "
		      << strerror( errno ) << "\n";
	    exit( 1 );
	}
    }

private:
    volatile bool m_terminated;
    std::mutex m_lock;
    std::thread m_thread;
    std::map<pthread_t,thread_info> m_thread_info;
};

template<typename Fn, typename... Args>
auto execute_time_limited( uint64_t usec, Fn && fn, Args && ... args ) {
    return TimeLimitedExecution::execute_time_limited( usec, fn, args... );
}

template<typename... Fn>
class AlternativeSelector {
    static constexpr size_t num_fns = sizeof...( Fn );
    
public:
    AlternativeSelector( Fn && ... fn )
	: m_fn( std::forward<Fn>( fn )... ) {
	std::fill( &m_success[0], &m_success[num_fns], 0 );
	std::fill( &m_fail[0], &m_fail[num_fns], 0 );
	std::fill( &m_best[0], &m_best[num_fns], 0 );
	std::fill( &m_success_time_total[0], &m_success_time_total[num_fns], 0 );
	std::fill( &m_success_time_max[0], &m_success_time_max[num_fns], 0 );
	std::fill( &m_best_time_total[0], &m_best_time_total[num_fns], 0 );
    }
    ~AlternativeSelector() {
	report( std::cerr );
    }

    template<typename... Args>
    auto execute( uint64_t base_usec, Args && ... args ) {
#if 1
	using return_type = decltype( std::get<0>( m_fn )( 0ull, args... ) );
	return_type ret;

	for( uint64_t rep=1; rep <= 24; ++rep ) {
	    uint64_t usec = base_usec << rep;
	    try {
		return attempt_fn<0>( usec, std::forward<Args>( args )... );
	    } catch( timeout_exception & e ) {
	    }
	}

	// None of the alternatives completed in time limit
	abort();
#else
	try {
	    // uint64_t usec = 800000000ull; // 800sec
	    uint64_t usec = 50000000ull << 13; // 50sec
	    return attempt_all_fn( usec, std::forward<Args>( args )... );
	} catch( timeout_exception & e ) {
	    std::cerr << "timeout: usec=" << e.usec() << " idx=" << e.idx() << "\n";
	    throw;
	}
#endif
    }

    std::ostream & report( std::ostream & os ) {
	os << "Success of alternatives (#=" << num_fns << "):\n";
	for( size_t i=0; i < num_fns; ++i ) {
	    os << "alternative " << i
	       << ": success=" << m_success[i]
	       << " avg-success-tm="
	       << ( m_success_time_total[i] / double(m_success[i]) )
	       << " max-success-tm=" << m_success_time_max[i] 
	       << " fail=" << m_fail[i]
	       << " best=" << m_best[i]
	       << " avg-best-time="
	       << ( m_best_time_total[i] / double(m_best[i]) )
	       << "\n";
	}
	return os;
    }

private:
    template<size_t idx, typename... Args>
    auto attempt_fn( uint64_t usec, Args && ... args ) {
	using return_type = decltype( std::get<0>( m_fn )( 0ull, args... ) );
	return_type ret;

	timer tm;
	tm.start();

	try {
	    ret = execute_time_limited(
		usec, std::get<idx>( m_fn ), std::forward<Args>( args )... );
	    auto dly = tm.stop();
	    std::cerr << "   alt #" << idx << " succeeded after "
		      << dly << "\n";
	    m_success_time_total[idx] += dly;
	    m_success_time_max[idx] = std::max( m_success_time_max[idx], dly );
	    ++m_success[idx];
	    return ret;
	} catch( timeout_exception & e ) {
	    std::cerr << "   alt #" << idx << " failed after "
		      <<  tm.stop() << "\n";
	    ++m_fail[idx];
	    if constexpr ( idx >= num_fns-1 )
		throw timeout_exception( usec, idx );
	    else
		return attempt_fn<idx+1>( usec, std::forward<Args>( args )... );
	}
    }

    template<typename... Args>
    auto attempt_all_fn( uint64_t usec, Args && ... args ) {
	std::array<double,num_fns> tms = { std::numeric_limits<double>::max() };
	using return_type = decltype( std::get<0>( m_fn )( 0ull, args... ) );
	return_type ret;
	bool repeat = true;

	while( repeat ) {
	    try {
		ret = attempt_all_fn_aux<0>(
		    usec, tms, std::forward<Args>( args )... );
		repeat = false;
	    } catch( timeout_exception & e ) {
		usec *= 2;
		std::cerr << "timeout on all variants; doubling time to "
			  << usec << "\n";
	    }
	}

	for( size_t idx=0; idx < num_fns; ++idx ) {
	    double dly = tms[idx];
	    if( dly != std::numeric_limits<double>::max() ) {
		m_success_time_total[idx] += dly;
		m_success_time_max[idx] = std::max( m_success_time_max[idx], dly );
		++m_success[idx];
	    } else
		++m_fail[idx];
	}

	size_t best = std::distance(
	    tms.begin(), std::min_element( tms.begin(), tms.end() ) );
	++m_best[best];
	m_best_time_total[best] += tms[best];

	return ret;
    }

    template<typename Arg0, typename... Args>
    void check_clique( size_t size, VID * clique,
		       Arg0 && arg0, Args && ... args ) {
	::check_clique( std::forward<Arg0>( arg0 ), size, clique );
    }
    
    template<size_t idx, typename... Args>
	auto attempt_all_fn_aux( uint64_t usec, std::array<double,num_fns> & tms, Args && ... args ) {
	using return_type = decltype( std::get<0>( m_fn )( 0ull, args... ) );
	return_type ret;

	timer tm;
	tm.start();

	try {
	    // TODO: pass in ret as argument and use any contents filled in
	    //       even in case of timeout.
	    if( verbose )
		std::cerr << "as: alternative " << idx
			  << " timeout " << usec << "\n";
	    ret = execute_time_limited(
		usec, std::get<idx>( m_fn ), std::forward<Args>( args )... );
	    tms[idx] = tm.stop();
	    // check_clique( ret.size(), &ret[0], std::forward<Args>( args )... );

	    if constexpr ( idx+1 < num_fns ) {
		try {
		    auto r = attempt_all_fn_aux<idx+1>(
			usec, tms, std::forward<Args>( args )... );
		    assert( is_equal( ret, r ) );
		} catch( timeout_exception & e ) {
		    return ret;
		}
	    }

	    return ret;
	} catch( timeout_exception & e ) {
	    tms[idx] = std::numeric_limits<double>::max();
	    if constexpr ( idx+1 < num_fns )
		return attempt_all_fn_aux<idx+1>(
		    usec, tms, std::forward<Args>( args )... );
	    else
		throw timeout_exception( usec, idx );
	}
    }

    template<typename T>
    static bool is_equal( const T & a, const T & b ) {
	return true;
    }
    static bool is_equal( bool a, bool b ) {
	return a == b;
    }
    static bool
    is_equal( const std::vector<VID> & a, const std::vector<VID> & b ) {
	return a.size() == b.size();
    }

private:
    std::tuple<Fn...> m_fn;
    size_t m_success[num_fns];
    size_t m_fail[num_fns];
    size_t m_best[num_fns];
    double m_success_time_total[num_fns];
    double m_success_time_max[num_fns];
    double m_best_time_total[num_fns];
};

template<typename... Fn>
auto make_alternative_selector( Fn && ... fn ) {
    return AlternativeSelector<Fn...>( std::forward<Fn>( fn )... );
}


template<unsigned Bits, typename sVID, typename sEID>
class DenseMatrix {
    using VID = sVID;
    using EID = sEID;
    using DID = std::conditional_t<Bits<=256,uint8_t,uint16_t>;
    using type = std::conditional_t<Bits<=32,uint32_t,uint64_t>;
    static constexpr unsigned bits_per_lane = sizeof(type)*8;
    static constexpr unsigned short VL = Bits / bits_per_lane;
    using tr = vector_type_traits_vl<type,VL>;
    using row_type = typename tr::type;

    static_assert( VL * bits_per_lane == Bits );

public:
    static constexpr size_t MAX_VERTICES = bits_per_lane * VL;

public:
    DenseMatrix( const GraphCSx & G, VID v,
		 VID num_neighbours, const VID * neighbours,
		 const VID * const core_order )
	: m_start_pos( 0 ) {
	VID n = G.numVertices();
	EID m = G.numEdges();
	const EID * const gindex = G.getIndex();
	const VID * const gedges = G.getEdges();

	// Set of eligible neighbours
	VID ns = num_neighbours;
	assert( ns <= MAX_VERTICES );
	std::copy( &neighbours[0], &neighbours[ns], m_s2g );

	sVID * n2s = new sVID[ns];
	std::iota( &m_s2g[0], &m_s2g[ns], 0 );

	// Sort by increasing core_order
	std::sort( &m_s2g[0], &m_s2g[ns],
		   [&]( VID u, VID v ) {
		       return core_order[neighbours[u]]
			   < core_order[neighbours[v]];
		   } );
	// Invert permutation into n2s and create mapping for m_s2g
	for( VID su=0; su < ns; ++su ) {
	    VID x = m_s2g[su];
	    m_s2g[su] = neighbours[x]; // create mapping
	    n2s[x] = su; // invert permutation
	}

	// Determine start position, i.e., vertices less than start_pos
	// are in X by default
	VID * sp2_pos = std::upper_bound(
	    &m_s2g[0], &m_s2g[ns], v,
	    [&]( VID a, VID b ) {
		return core_order[a] < core_order[b];
	    } );
	m_start_pos = sp2_pos - &m_s2g[0];

#if 0
	// Short-cut if we have P=empty and all vertices in X
	// Saves time constructing the matrix.
	// Doesn't require specific checks in mce_bron_kerbosch()
	// Doesn't help performance...
	if( m_start_pos >= ns ) {
	    m_matrix = m_matrix_alc = nullptr;
	    m_m = 0;
	    m_n = ns;
	    delete[] n2s;
	    return;
	}
#endif

	assert( ( ns + bits_per_lane - 1 ) / bits_per_lane <= m_words );
	m_matrix = m_matrix_alc = new type[m_words * ns + 32];
	intptr_t p = reinterpret_cast<intptr_t>( m_matrix );
	if( p & 31 ) // 31 = 256 bits / 8 bits per byte - 1
	    m_matrix = &m_matrix[(p&31)/sizeof(type)];
	static_assert( Bits <= 256, "AVX512 requires 64-byte alignment" );
	// std::fill( &m_matrix[0], &m_matrix[m_words * ns], 0 );

	// Place edges
	VID ni = 0;
	m_m = 0;
	for( VID su=0; su < ns; ++su ) {
	    VID u = m_s2g[su];
	    VID deg = 0;

	    row_type row_u = tr::setzero();

	    contract::detail::intersect_tmpl(
		&neighbours[0], &neighbours[num_neighbours],
		&gedges[gindex[u]], &gedges[gindex[u+1]],
		[&]( VID w ) {
		    const VID * const pos = std::lower_bound(
			&neighbours[0], &neighbours[ns], w );
		    if( pos != &neighbours[ns] && *pos == w && u != w ) {
			VID sw = n2s[pos - neighbours];
			row_u = tr::bitwise_or( row_u, create_row( sw ) );
			++deg;
		    }
		    return true;
		} );

	    tr::store( &m_matrix[VL * su], row_u );
	    m_degree[su] = deg;
	    m_m += deg;
	}

	m_n = ns;

	delete[] n2s;
    }
    // In this variation, hVIDs are already sorted by core_order
    // and we know the separation between X and P sets.
    template<typename hVID, typename hEID, typename Hash>
    DenseMatrix( const graptor::graph::GraphHAdjTable<hVID,hEID,Hash> & G,
		 const hVID * XP,
		 hVID ne, hVID ce )
	: m_start_pos( ne ) {
	static_assert( sizeof(hVID) >= sizeof(sVID) );
	static_assert( sizeof(hEID) >= sizeof(sEID) );

	// Vertices in X and P are independently already sorted by core order.
	// We do not reorder, primarily because only the order in P matters
	// for efficiency of enumeration.
	// Do we need to copy, or can we just keep a pointer to XP?
	VID ns = ce;
	std::copy( XP, XP+ce, m_s2g );

	assert( ( ns + bits_per_lane - 1 ) / bits_per_lane <= m_words );
	m_matrix = m_matrix_alc = new type[m_words * ns + 32];
	intptr_t p = reinterpret_cast<intptr_t>( m_matrix );
	if( p & 31 ) // 31 = 256 bits / 8 bits per byte - 1
	    m_matrix = &m_matrix[(p&31)/sizeof(type)];
	static_assert( Bits <= 256, "AVX512 requires 64-byte alignment" );
	std::fill( &m_matrix[0], &m_matrix[m_words * ns], 0 );

	// Place edges
	VID ni = 0;
	m_m = 0;
	for( VID su=0; su < ns; ++su ) {
	    VID u = m_s2g[su]; // or XP[su]
	    VID deg = 0;

	    row_type row_u = tr::setzero();
	    auto & adj = G.get_adjacency( u );

	    // Intersect XP with adjacency list
	    for( VID l=(su >= m_start_pos ? 0 : ne); l < ce; ++l ) {
		VID xp = XP[l];
		if( adj.contains( xp ) ) {
		    row_u = tr::bitwise_or( row_u, create_row( l ) );
		    ++deg;
		}
	    }

	    tr::store( &m_matrix[VL * su], row_u );
	    m_degree[su] = deg;
	    m_m += deg;
	}

	m_n = ns;
    }

    ~DenseMatrix() {
	if( m_matrix_alc != nullptr )
	    delete[] m_matrix_alc;
    }

    
    // Variations to consider:
    // - bron_kerbosh_nox (excluding X set)
    // - bron_kerbosh_pivot (no X set, pivoting)
    // - bron_kerbosh_pivot_degeneracy (no X set, degeneracy ordering, pivoting)
    // Note that the X set is used only to identify maximal cliques. For
    // maximum clique search, it does not matter as we can only avoid
    // a scalar int comparison, while checking X is zero is slightly more
    // expensive. If a non-maximal clique is considered, we haven't lost time
    // and we are not inaccurate.
    bitset<Bits>
    bron_kerbosch( VID cutoff ) {
	// First try with cutoff, if failing, try without.
	// Useful if we believe we will find a clique of size cutoff
	VID th = cutoff;
	if( cutoff == m_n ) // initial guess is bad for us
	    cutoff = 1;
	else if( cutoff > 1 ) {
	    // set target slightly lower to increase chance of success
	    // for example in those case where we drop from a 5-clique
	    // to a 4-clique
	    --cutoff;
	}
    
	auto ret = bron_kerbosch_with_cutoff( cutoff );
	if( ret.size() >= cutoff )
	    return ret;
	else {
	    // We know a clique of size ret.size() exists, so use this
	    // as a cutoff for the second attempt
	    cutoff = ret.size() >= 1 ? ret.size() : 1;
	    return bron_kerbosch_with_cutoff( cutoff );
	}
    }

    template<typename Enumerate>
    void
    mce_bron_kerbosch( Enumerate && E ) {
	for( VID v=m_start_pos; v < m_n; ++v ) { // implicit X vertices
	    row_type vrow = create_row( v );
	    row_type R = vrow;

	    // if no neighbours in cut-out, then trivial 2-clique
	    if( tr::is_zero( get_row( v ) ) ) {
		E( bitset<Bits>( R ) );
		continue;
	    }

	    // Consider as candidates only those neighbours of u that are
	    // ordered after v to avoid revisiting the vertices
	    // unnecessarily.
	    row_type h = get_himask( v );
	    row_type r = get_row( v );
	    row_type P = tr::bitwise_and( h, r );
	    row_type X = tr::bitwise_andnot( h, r );
	    // std::cerr << "depth " << 0 << " v=" << v << "\n";
	    mce_bk_iterate( E, R, P, X, 1 );
	}
    }
    
private:
    bitset<Bits>
    bron_kerbosch_with_cutoff( VID cutoff ) {
	m_mc = tr::setzero();
	m_mc_size = 0;

	for( VID v=0; v < m_n; ++v ) {
	    if( tr::is_zero( get_row( v ) ) )
		continue;

	    row_type vrow = create_row( v );
	    row_type R = vrow;

	    // Consider as candidates only those neighbours of u that are larger
	    // than u to avoid revisiting the vertices unnecessarily.
	    row_type P = tr::bitwise_and( get_row( v ), get_himask( v ) );

	    bk_iterate( R, P, 1, cutoff );
	}

	return bitset<Bits>( m_mc );
    }

public:
    void erase_incident_edges( bitset<Bits> vset ) {
	// Erase columns
	row_type vs = vset;
	for( VID v=0; v < m_n; ++v )
	    tr::store( &m_matrix[m_words * v],
		       tr::bitwise_andnot(
			   vs, tr::load( &m_matrix[m_words * v] ) ) );
	
	// Erase rows
	for( auto && v : vset ) {
	    assert( v < m_n );
	    tr::store( &m_matrix[m_words * v], tr::setzero() );
	}
    }

    bitset<Bits>
    vertex_cover() {
	m_mc = tr::setone();
	m_mc_size = m_n;

	row_type z = tr::setzero();
	vc_iterate( 0, z, z, 0 );

	// cover vs clique on complement. Invert bitset, mask with valid bits
	m_mc = tr::bitwise_andnot( m_mc, get_himask( m_n ) );
	m_mc_size = m_n - m_mc_size; // for completeness; unused hereafter
	return bitset<Bits>( m_mc );
    }

    VID numVertices() const { return m_n; }
    EID numEdges() const { return m_m; }

    const VID * get_s2g() const { return &m_s2g[0]; }

private:
    void bk_iterate( row_type R, row_type P, int depth, VID cutoff ) {
	// depth == get_size( R )
	if( tr::is_zero( P ) ) {
	    if( depth > m_mc_size ) {
		m_mc = R;
		m_mc_size = depth;
	    }
	    return;
	}
	VID p_size = get_size( P );
	if( depth + p_size < m_mc_size )
	    return;
	if( depth + p_size < cutoff )
	    return;

	row_type x = P;
	while( !tr::is_zero( x ) ) {
	    VID u;
	    row_type x_new;
	    std::tie( u, x_new ) = remove_element( x );
	    row_type u_row = tr::bitwise_andnot( x_new, x );
	    x = x_new;
	    // assert( tr::is_zero( tr::bitwise_and( get_row( u ), create_row( u ) ) ) );
	    row_type Pv = tr::bitwise_and( x, get_row( u ) ); // x vs P?
	    row_type Rv = tr::bitwise_or( R, u_row );
	    bk_iterate( Rv, Pv, depth+1, cutoff );
	}
    }

    template<typename Enumerate>
    void mce_bk_iterate(
	Enumerate && EE,
	row_type R, row_type P, row_type X, int depth ) {
	// depth == get_size( R )
	if( tr::is_zero( P ) ) {
	    if( tr::is_zero( X ) )
		EE( bitset<Bits>( R ) );
	    return;
	}

	VID pivot = mce_get_pivot( P, X );
	row_type pivot_ngh = get_row( pivot );
	row_type x = tr::bitwise_andnot( pivot_ngh, P );
	bitset<Bits> bx( x );
	for( auto I = bx.begin(), E = bx.end(); I != E; ++I ) {
	    VID u = *I;
	    // row_type x_new;
	    // std::tie( u, x_new ) = remove_element( x );
	    // row_type u_only = tr::bitwise_andnot( x_new, x );
	    row_type u_only = tr::setglobaloneval( u );
	    row_type x_new = tr::bitwise_andnot( u_only, x );
	    x = x_new;
	    row_type u_ngh = get_row( u );
	    row_type Pv = tr::bitwise_and( P, u_ngh );
	    row_type Xv = tr::bitwise_and( X, u_ngh );
	    row_type Rv = tr::bitwise_or( R, u_only );
	    P = tr::bitwise_andnot( u_only, P ); // P == x w/o pivoting
	    X = tr::bitwise_or( u_only, X );
	    // std::cerr << "depth " << depth << " u=" << u << "\n";
	    mce_bk_iterate( EE, Rv, Pv, Xv, depth+1 );
	}
    }

    // Could potentially vectorise small matrices by placing
    // one 32/64-bit row in a vector lane and performing a vectorised
    // popcount per lane. Could evaluate doing popcounts on all lanes,
    // or gathering only active lanes. The latter probably most promising
    // in AVX512
    VID mce_get_pivot( row_type P, row_type X ) {
	row_type r = tr::bitwise_or( P, X );
	bitset<Bits> b( r );

	VID p_best = *b.begin();
	VID p_ins = 0; // will be overridden

	// Avoid complexities if there is not much choice
	if( get_size( P ) <= 3 )
	    return p_best;
	
	for( auto I=b.begin(), E=b.end(); I != E; ++I ) {
	    VID v = *I;
	    if( (VID)m_degree[v] < p_ins ) // skip if cannot be best
		continue;
	    row_type v_ngh = get_row( v );
	    row_type pv_ins = tr::bitwise_and( P, v_ngh );
	    VID ins = get_size( pv_ins );
	    if( ins > p_ins ) {
		p_best = v;
		p_ins = ins;
	    }
	}
	assert( ~p_best != 0 );
	return p_best;
    }

    // cin is a bitmask indicating which vertices are in the cover.
    // It is filled up only up to vertex v. Remaining bits are zero.
    // cout indicates the vertices excluded.
    void vc_iterate( VID v, row_type cin, row_type cout, VID cin_sz ) {
	// Leaf node
	if( v == m_n ) {
	    if( cin_sz < m_mc_size ) {
		m_mc = cin;
		m_mc_size = cin_sz;
	    }
	    return;
	}

	// isolated vertex
	row_type v_set = create_row( v );
	row_type v_row = tr::bitwise_andnot(
	    get_himask( m_n ), tr::bitwise_xnor( v_set, get_row( v ) ) );
	VID deg = get_size( v_row );
	if( deg == 0 ) {
	    vc_iterate( v+1, cin, tr::bitwise_or( cout, v_set ), cin_sz );
	    return;
	}

	// Count number of covered neighbours
	VID num_covered = get_size( tr::bitwise_and( v_row, cin ) );

	// In case we don't have choice: including all neighbours would result
	// in a vertex cover larger than the one of interest. In that case,
	// include the vertex and not the (remaining) neighbours
	if( cin_sz + deg - num_covered >= m_mc_size ) {
	    vc_iterate( v+1, tr::bitwise_or( cin, v_set ), cout, cin_sz+1 );
	    return;
	}

	// All neighbours included, so this vertex is not needed
	// Any neighbour not included, then this vertex must be included
	if( num_covered == deg ) {
	    vc_iterate( v+1, cin, tr::bitwise_or( cout, v_set ), cin_sz );
	    return;
	}

	// Count number of uncovered neighbours; only chance we have any
	// if cout_sz is non-zero
	VID cout_sz = v - cin_sz;
	if( cout_sz > 0 ) {
	    VID num_uncovered = get_size( tr::bitwise_and( v_row, cout ) );
	    if( num_uncovered > 0 ) {
		vc_iterate( v+1, tr::bitwise_or( cin, v_set ), cout, cin_sz+1 );
		return;
	    }
	}

	// Otherwise, try both ways.
	vc_iterate( v+1, cin, tr::bitwise_or( cout, v_set ), cin_sz );

	vc_iterate( v+1, tr::bitwise_or( cin, v_set ), cout, cin_sz+1 );
    }
    
    static std::pair<VID,row_type> remove_element( row_type s ) {
	// find_first includes tzcnt; can be avoided because lane() includes
	// a switch, so can check on mask & 1, mask & 2, etc instead of
	// tzcnt == 0, tzcnt == 1, etc
	auto mask = tr::cmpne( s, tr::setzero(), target::mt_mask() );

	type xtr;
	unsigned lane;
	
	if constexpr ( VL == 4 ) {
	    __m128i half;
	    if( ( mask & 0x3 ) == 0 ) {
		half = tr::upper_half( s );
		lane = 2;
		mask >>= 2;
	    } else {
		half = tr::lower_half( s );
		lane = 0;
	    }
	    if( ( mask & 0x1 ) == 0 ) {
		lane += 1;
		xtr = _mm_extract_epi64( half, 1 );
	    } else {
		xtr = _mm_extract_epi64( half, 0 );
	    }
	} else if constexpr ( VL == 2 ) {
	    if( ( mask & 0x1 ) == 0 ) {
		xtr = tr::upper_half( s );
		lane = 1;
	    } else {
		xtr = tr::lower_half( s );
		lane = 0;
	    }
	} else if constexpr ( VL == 1 ) {
	    lane = 0;
	    xtr = s;
	} else
	    assert( 0 && "Oops" );

	assert( xtr != 0 );
	unsigned off = _tzcnt_u64( xtr );
	assert( off != bits_per_lane );
	row_type s_upd = tr::bitwise_and( s, tr::sub( s, tr::setoneval() ) );
	row_type new_s = tr::blend( 1 << lane, s, s_upd );
	return std::make_pair( lane * bits_per_lane + off, new_s );
    }

    row_type get_row( VID v ) {
	return tr::load( &m_matrix[m_words * v] );
    }

    row_type create_row( VID v ) { // TODO: lookup table with 0x1 in precisely one lane
#if 0
	row_type z = tr::setzero();
	row_type o = tr::setoneval();
	row_type p = tr::sll( o, v % bits_per_lane );
	VID lane = v / bits_per_lane;
	row_type r = tr::blend( 1 << lane, z, p );
	return r;
#else
	return tr::setglobaloneval( v );
#endif
	
	// return tr::setlane(
	// tr::setzero(), type(1) << ( v % bits_per_lane ), v / bits_per_lane );
    }

    row_type get_himask( VID v ) {
#if 1
	row_type r = tr::himask( v+1 );
	return r;
#else
#if 1
	row_type z = tr::setzero();
	row_type s = tr::setone();
	row_type o = tr::setoneval();
	// row_type p = tr::bitwise_invert(
	// tr::sub( tr::sll( o, v % bits_per_lane ), o ) );
	row_type p = tr::sll( s, v % bits_per_lane );
	VID lane = v / bits_per_lane;
	VID mask = ( VID(1) << VL ) - ( VID(1) << lane );
	row_type a = tr::blend( mask, z, s );
	row_type r = tr::blend( 1 << lane, a, p );
	return r;
#else
	VID lane = v / bits_per_lane;
	row_type a = tr::load( &himask_starter[lane * VL] );
	row_type b = tr::slli( a, v % bits_per_lane );
	row_type c = tr::sub( b, a );
row_type d = tr::srli( a, 1 );
	row_type e = tr::bitwise_or( c, d );
	row_type f = tr::bitwise_invert( e );
	return f;
#endif
#endif
    }

    void set( VID u, VID v ) {
	assert( u != v );
	VID word, off;
	std::tie( word, off ) = slocate( u, v );
	type w = type(1) << off;
	m_matrix[word] |= w;
	// assert( tr::is_zero( tr::bitwise_and( get_row( u ), create_row( u ) ) ) );
	// assert( tr::is_zero( tr::bitwise_and( get_row( v ), create_row( v ) ) ) );
    }

    static VID get_size( row_type r ) {
	return target::allpopcnt<VID,type,VL>::compute( r );
    }

    std::pair<VID,VID> slocate( VID u, VID v ) const {
	VID col = v / bits_per_lane;
	VID word = u * VL + col;
	return std::make_pair( word, v % bits_per_lane );
    }

    VID get_start_pos() const { return m_start_pos; }
		    
private:
    VID m_n;
    static constexpr unsigned m_words = VL;
    EID m_m;
    type * m_matrix;
    type * m_matrix_alc;

    VID m_mc_size;
    row_type m_mc;
    VID m_start_pos;

    VID m_s2g[Bits];
    DID m_degree[Bits];

    // assumes VL == 4
    alignas(64) static constexpr uint64_t himask_starter[16] = {
	0x1, 0x0, 0x0, 0x0,
	0xffffffffffffffff, 0x1, 0x0, 0x0,
	0xffffffffffffffff, 0xffffffffffffffff, 0x1, 0x0,
	0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0x1
	// 0x1, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff,
	// 0x0, 0x1, 0xffffffffffffffff, 0xffffffffffffffff,
	// 0x0, 0x0, 0x1, 0xffffffffffffffff,
	// 0x0, 0x0, 0x0, 0x1,
    };
};

template<unsigned Bits, typename Enumerator>
void mce_top_level(
    const GraphCSx & G,
    Enumerator & E,
    VID v,
    const NeighbourCutOutAll<VID,EID> & cut,
    const VID * const core_order ) {
    DenseMatrix<Bits,VID,EID>
	IG( G, v, cut.get_num_vertices(), cut.get_vertices(),
	    core_order );

    // Needs to include X? Pivoting?
    IG.mce_bron_kerbosch( [&]( const bitset<Bits> & c ) {
	E.record( 1 + c.size() );
/*
	std::cerr << "MC: " << v;
	for( auto v : c )
	    std::cerr << ' ' << IG.get_s2g()[v];
	std::cerr << " | cutout: ";
	for( auto v : c )
	    std::cerr << ' ' << v;
	std::cerr << "\n";

	std::vector<VID> cc( c.begin(), c.end() );
	std::transform( cc.begin(), cc.end(), cc.begin(),
			[&]( auto v ) { return IG.get_s2g()[v]; } );
	cc.push_back( v );
	std::sort( cc.begin(), cc.end() );
	check_clique( G, cc.size(), &cc[0] );
*/
    } );
}

template<typename Enumerator, typename VID, typename EID, typename Hash>
void mce_top_level(
    const graptor::graph::GraphHAdjTable<VID,EID,Hash> & HG,
    Enumerator & E,
    VID v,
    VID start_pos,
    // const NeighbourCutOutAll<VID,EID> & cut,
    // const VID * const core_order,
    VID degeneracy
    ) {
    auto Ee = [&]( const contract::vertex_set<VID> & c,
		   size_t surplus = 0 ) {
	E.record( 1+c.size()+surplus );
/*
	    std::cerr << "MC: " << v;
	    for( auto v : c )
		std::cerr << ' ' << ibuilder.get_s2g()[v];
	    std::cerr << " | cutout: ";
	    for( auto v : c )
		std::cerr << ' ' << v;
	    std::cerr << "\n";

	    std::vector<VID> cc( c.begin(), c.end() );
	    std::transform( cc.begin(), cc.end(), cc.begin(),
			    [&]( auto v ) { return ibuilder.get_s2g()[v]; } );
	    cc.push_back( v );
	    std::sort( cc.begin(), cc.end() );
	    check_clique( G, cc.size(), &cc[0] );
*/
	};

    
    VID n = HG.numVertices();
    if( n > 1024 )
	mce_bron_kerbosch_par_xp( HG, start_pos, degeneracy, Ee );
    else
	mce_bron_kerbosch_seq_xp( HG, start_pos, degeneracy, Ee );
}

struct variant_statistics {
    variant_statistics() : m_tm( 0 ), m_calls( 0 ) { }
    variant_statistics( double tm, size_t calls )
	: m_tm( tm ), m_calls( calls ) { }

    variant_statistics operator + ( const variant_statistics & s ) const {
	return variant_statistics( m_tm + s.m_tm, m_calls + s.m_calls );
    }

    void record( double atm ) {
	m_tm += atm;
	++m_calls;
    }
    
    double m_tm;
    size_t m_calls;
};

struct all_variant_statistics {
    all_variant_statistics() { }
    all_variant_statistics(
	variant_statistics && v32,
	variant_statistics && v64,
	variant_statistics && v128,
	variant_statistics && v256,
	variant_statistics && vgen,
	variant_statistics && vgenbuild ) :
	m_32( std::forward<variant_statistics>( v32 ) ),
	m_64( std::forward<variant_statistics>( v64 ) ),
	m_128( std::forward<variant_statistics>( v128 ) ),
	m_256( std::forward<variant_statistics>( v256 ) ),
	m_gen( std::forward<variant_statistics>( vgen ) ),
	m_genbuild( std::forward<variant_statistics>( vgenbuild ) ) { }

    all_variant_statistics
    operator + ( const all_variant_statistics & s ) const {
	return all_variant_statistics(
	    m_32 + s.m_32,
	    m_64 + s.m_64,
	    m_128 + s.m_128,
	    m_256 + s.m_256,
	    m_gen + s.m_gen,
	    m_genbuild + s.m_genbuild );
    }

    void record_32( double atm ) { m_32.record( atm ); }
    void record_64( double atm ) { m_64.record( atm ); }
    void record_128( double atm ) { m_128.record( atm ); }
    void record_256( double atm ) { m_256.record( atm ); }
    void record_gen( double atm ) { m_gen.record( atm ); }
    void record_genbuild( double atm ) { m_genbuild.record( atm ); }
    
    variant_statistics m_32, m_64, m_128, m_256, m_gen, m_genbuild;
};

// thread_local static all_variant_statistics * mce_pt_stats = nullptr;

struct per_thread_statistics {
    all_variant_statistics & get_statistics() {
	const pthread_t tid = pthread_self();
	std::lock_guard<std::mutex> guard( m_mutex );
	auto it = m_stats.find( tid );
	if( it == m_stats.end() ) {
	    auto it2 = m_stats.emplace(
		std::make_pair( tid, all_variant_statistics() ) );
	    return it2.first->second;
	}
	return it->second;
    }
    
    all_variant_statistics sum() const {
	return std::accumulate(
	    m_stats.begin(), m_stats.end(), all_variant_statistics(),
	    []( const all_variant_statistics & s,
		const std::pair<pthread_t,all_variant_statistics> & p ) {
		return s + p.second;
	    } );
    }
    
    std::mutex m_mutex;
    std::map<pthread_t,all_variant_statistics> m_stats;
};

per_thread_statistics mce_stats;


template<typename Enumerator>
void mce_top_level(
    const GraphCSx & G,
    Enumerator & E,
    VID v,
    const VID * const core_order,
    VID degeneracy ) {
    NeighbourCutOutAll<VID,EID> cut( G, v );

    all_variant_statistics & stats = mce_stats.get_statistics();

    VID num = cut.get_num_vertices();
    if( num <= 32 ) {
	timer tm;
	tm.start();
	mce_top_level<32>( G, E, v, cut, core_order );
	stats.record_32( tm.stop() );
    } else if( num <= 64 ) {
	timer tm;
	tm.start();
	mce_top_level<64>( G, E, v, cut, core_order );
	stats.record_64( tm.stop() );
    } else if( num <= 128 ) {
	timer tm;
	tm.start();
	mce_top_level<128>( G, E, v, cut, core_order );
	stats.record_128( tm.stop() );
    } else if( num <= 256 ) {
	timer tm;
	tm.start();
	mce_top_level<256>( G, E, v, cut, core_order );
	stats.record_256( tm.stop() );
    } else {
	timer tm;
	tm.start();
	GraphBuilderInduced<
	    graptor::graph::GraphHAdjTable<VID,EID,java_hash<VID>>>
	    ibuilder( G, v, cut, core_order );
	const auto & HG = ibuilder.get_graph();

	stats.record_genbuild( tm.stop() );

	tm.start();
	mce_top_level( HG, E, v, ibuilder.get_start_pos(), degeneracy );
	stats.record_gen( tm.stop() );
    }
}

class MCE_Enumerator {
public:
    MCE_Enumerator( size_t degen = 0 )
	: m_degeneracy( degen ),
	  m_histogram( degen+1 ) { }

    // Recod clique of size s
    void record( size_t s ) {
	assert( s <= m_degeneracy+1 );
	__sync_fetch_and_add( &m_histogram[s-1], 1 );
    }

    std::ostream & report( std::ostream & os ) const {
	assert( m_histogram.size() >= m_degeneracy+1 );

	size_t num_maximal_cliques = 0;
	for( size_t i=0; i < m_histogram.size(); ++i )
	    num_maximal_cliques += m_histogram[i];
	
	os << "Number of maximal cliques: " << num_maximal_cliques
	   << "\n";
	os << "Clique histogram: clique_size, num_of_cliques\n";
	for( size_t i=0; i <= m_degeneracy; ++i ) {
	    if( m_histogram[i] != 0 ) {
		os << (i+1) << ", " << m_histogram[i] << "\n";
	    }
	}
	return os;
    }

private:
    size_t m_degeneracy;
    std::vector<size_t> m_histogram;
};

int main( int argc, char *argv[] ) {
    commandLine P( argc, argv, " help" );
    bool symmetric = P.getOptionValue("-s");
    const char * ifile = P.getOptionValue( "-i" );

    timer tm;
    tm.start();

    GraphCSx G( ifile, -1, symmetric );

    std::cerr << "Reading graph: " << tm.next() << "\n";

    VID n = G.numVertices();
    EID m = G.numEdges();

    assert( G.isSymmetric() );
    std::cerr << "Undirected graph: n=" << n << " m=" << m << std::endl;

    std::cerr << "Calculating coreness...\n";
    GraphCSRAdaptor GA( G, 256 );
    KCv<GraphCSRAdaptor> kcore( GA, P );
    kcore.run();
    auto & coreness = kcore.getCoreness();
    std::cerr << "coreness=" << kcore.getLargestCore() << "\n";

    std::cerr << "Calculating coreness: " << tm.next() << "\n";

    std::cerr << "Calculating sort order...\n";
    mm::buffer<VID> order( n, numa_allocation_interleaved() );
    mm::buffer<VID> rev_order( n, numa_allocation_interleaved() );

    sort_order( order.get(), rev_order.get(),
		coreness.get_ptr(), n, kcore.getLargestCore() );

    std::cerr << "Determining sort order: " << tm.next() << "\n";

    MCE_Enumerator E( kcore.getLargestCore() );

    system( "date" );

    parallel_loop( VID(0), n, 1, [&]( VID i ) {
	VID v = order[i];

/*
	std::cerr << "Iteration " << i << " with v=" << v
		  << " deg=" << G.getDegree( v )
		  << " c=" << coreness[v]
		  << "\n";
*/
	
	mce_top_level( G, E, v, rev_order.get(), kcore.getLargestCore() );

	// std::cerr << "  vertex " << v << " complete\n";
    } );

    std::cerr << "Enumeration: " << tm.next() << "\n";

    all_variant_statistics stats = mce_stats.sum();

    double duration = tm.total();
    std::cerr << "Completed MCE in " << duration << " seconds\n";
    std::cerr << " 32-bit version: " << stats.m_32.m_tm << " seconds in "
	      << stats.m_32.m_calls << " calls @ "
	      << ( stats.m_32.m_tm / double(stats.m_32.m_calls) )
	      << " s/call\n";
    std::cerr << " 64-bit version: " << stats.m_64.m_tm << " seconds in "
	      << stats.m_64.m_calls << " calls @ "
	      << ( stats.m_64.m_tm / double(stats.m_64.m_calls) )
	      << " s/call\n";
    std::cerr << " 128-bit version: " << stats.m_128.m_tm << " seconds in "
	      << stats.m_128.m_calls << " calls @ "
	      << ( stats.m_128.m_tm / double(stats.m_128.m_calls) )
	      << " s/call\n";
    std::cerr << " 256-bit version: " << stats.m_256.m_tm << " seconds in "
	      << stats.m_256.m_calls << " calls @ "
	      << ( stats.m_256.m_tm / double(stats.m_256.m_calls) )
	      << " s/call\n";
    std::cerr << " generic version: " << stats.m_gen.m_tm << " seconds in "
	      << stats.m_gen.m_calls << " calls @ "
	      << ( stats.m_gen.m_tm / double(stats.m_gen.m_calls) )
	      << " s/call\n";
    std::cerr << " generic version building: "
	      << stats.m_genbuild.m_tm << " seconds in "
	      << stats.m_genbuild.m_calls << " calls @ "
	      << ( stats.m_genbuild.m_tm / double(stats.m_genbuild.m_calls) )
	      << " s/call\n";

    std::cerr << " pruning: " << pruning << "\n";

    E.report( std::cerr );

    rev_order.del();
    order.del();
    G.del();

    return 0;
}
