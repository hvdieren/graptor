// -*- C++ -*-
#ifndef GRAPHGRIND_GRAPH_SIMPLE_DENSE_H
#define GRAPHGRIND_GRAPH_SIMPLE_DENSE_H

#include <type_traits>
#include <algorithm>

#include "graptor/graph/GraphCSx.h"
#include "graptor/graph/simple/cutout.h"
#include "graptor/graph/simple/hadjt.h"
#include "graptor/graph/simple/bitconstruct.h"
#include "graptor/graph/simple/xp_set.h"
#include "graptor/target/vector.h"
#include "graptor/container/intersect.h"

#ifndef DENSE_THRESHOLD_SEQUENTIAL
#define DENSE_THRESHOLD_SEQUENTIAL 32.0
#endif

#ifndef DENSE_THRESHOLD_SEQUENTIAL_BITS
#define DENSE_THRESHOLD_SEQUENTIAL_BITS 64
#endif

#ifndef DENSE_THRESHOLD_DENSITY
#define DENSE_THRESHOLD_DENSITY 0.5
#endif


namespace graptor {

namespace graph {

template<typename BitMaskTy, typename lVID>
struct bitmask_output_iterator {
    using bitmask_type = BitMaskTy;
    static constexpr size_t word_size = 8 * sizeof( bitmask_type );
    
    bitmask_output_iterator(
	bitmask_type * bitmask,
	const lVID * n2s,
	const lVID * start,
	lVID deg_start_pos,
	lVID start_pos )
	: m_bitmask( bitmask ), m_n2s( n2s ), m_start( start ),
	  m_deg( 0 ), m_deg_start_pos( deg_start_pos ),
	  m_start_pos( start_pos ) { }

    const bitmask_output_iterator &
    operator = ( const bitmask_output_iterator & it ) {
	// hmmm....
	m_deg = it.m_deg;
	return *this;
    }

    void push_back( const lVID * p, const lVID * = nullptr ) {
	lVID v = m_n2s[p - m_start];
	// no X-X edges
	if( v >= m_start_pos ) {
	    m_bitmask[v/word_size]
		|= bitmask_type(1) << ( v & ( word_size-1 ) );
	    if( v >= m_deg_start_pos )
		m_deg++;
	}
    }

#if 0
    template<unsigned VL>
    void push_back( typename vector_type_traits_vl<lVID,VL>::mask_type m,
		    typename vector_type_traits_vl<lVID,VL>::type v,
		    const lVID * base ) {
	using tr = vector_type_traits_vl<lVID,VL>;
	using mtr = typename tr::mask_traits;
	using type = typename tr::type;
	using mask_type = typename tr::mask_type;

	type vsp = tr::set1( m_start_pos );
	type vns = tr::loadu( &m_n2s[base - m_start] );
	// type vns = tr::set1inc( base - m_start );
	mask_type mm = tr::cmpge( m, vns, vsp, target::mt_mask() );

	// Now need to set bits corresponding to vns if set in mm
	// Makes assumption that elements of vns are a+0,a+1,a+2,etc.
	// which happens when iterating LHS over s2g. However, this is
	// not correct in case of merging, because s2g is sorted
	// by degeneracy and not by ID. Also not correct with partitioned
	// approaches.
	mask_type * b = reinterpret_cast<mask_type *>( m_bitmask );
	// b[(base-m_start)/VL] = mm;
    }
#endif

    VID get_degree() const { return m_deg; }

private:
    bitmask_type * m_bitmask;
    const lVID * m_n2s;
    const lVID * m_start;
    lVID m_deg;
    const lVID m_deg_start_pos;
    const lVID m_start_pos;
};


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
#if 0
    DenseMatrix( const ::GraphCSx & G, VID v,
		 const NeighbourCutOutXP<VID,EID> & cut,
		 const VID * const core_order )
	: DenseMatrix( G, v, cut.get_num_vertices(), cut.get_vertices(),
		       cut.get_s2g(), cut.get_n2s(), cut.get_start_pos(),
		       core_order ) { }
    DenseMatrix( const ::GraphCSx & G, VID v,
		 VID num_neighbours, const VID * neighbours,
		 const VID * const s2g,
		 const VID * const n2s,
		 VID start_pos,
		 const VID * const core_order )
	: m_start_pos( start_pos ) {
	VID n = G.numVertices();
	EID m = G.numEdges();
	const EID * const gindex = G.getIndex();
	const VID * const gedges = G.getEdges();

	// Set of eligible neighbours
	VID ns = num_neighbours;

	assert( ( ns + bits_per_lane - 1 ) / bits_per_lane <= m_words );
	m_matrix = m_matrix_alc = new type[m_words * ns + 64];
	intptr_t p = reinterpret_cast<intptr_t>( m_matrix );
	if( p & 63 ) // 63 = 512 bits / 8 bits per byte - 1
	    m_matrix = &m_matrix[64 - (p&63)/sizeof(type)];
	static_assert( Bits <= 512, "AVX512 requires 64-byte alignment" );

	// Place edges
	VID ni = 0;
	m_m = 0;
	for( VID su=0; su < ns; ++su ) {
	    VID u = s2g[su];
	    VID deg = 0;

	    row_type row_u = tr::setzero();

	    const VID * p = &neighbours[0];
	    const VID * const pe = &neighbours[num_neighbours];
	    const VID * q = &gedges[gindex[u]];
	    const VID * const qe = &gedges[gindex[u+1]];

	    while( p != pe && q != qe ) {
		if( *p == *q ) {
		    // Common neighbour
		    if( *p != u ) {
			VID sw = n2s[p - neighbours];
			// no X-X edges
			if( su >= m_start_pos )
			    ++deg;
			if( sw >= m_start_pos || su >= m_start_pos )
			    row_u = tr::bitwise_or( row_u, create_row( sw ) );
		    }
		    ++p;
		    ++q;
		} else if( *p < *q )
		    ++p;
		else
		    ++q;
	    }

	    tr::store( &m_matrix[VL * su], row_u );
#if !ABLATION_PDEG
	    m_degree[su] = deg;
#endif
	    m_m += deg;
	}

	m_n = ns;
    }
    template<typename utr, typename HGraph>
    DenseMatrix( const ::GraphCSx & G,
		 const HGraph & H,
		 VID v,
		 const NeighbourCutOutXP<VID,EID> & cut,
		 const VID * const core_order,
		 size_t levels,
		 const VID * prestudy,
		 utr )
	: DenseMatrix( G, v, cut.get_num_vertices(), cut.get_vertices(),
		       cut.get_s2g(), cut.get_n2s(), cut.get_start_pos(),
		       core_order, levels, prestudy, utr() ) { }
    template<typename utr, typename HGraph>
    DenseMatrix( const ::GraphCSx & G,
		 const HGraph & H,
		 VID v,
		 VID num_neighbours, const VID * neighbours,
		 const VID * const s2g,
		 const VID * const n2s,
		 VID start_pos,
		 const VID * const core_order,
		 size_t levels,
		 const VID * prestudy,
		 utr )
	: m_start_pos( start_pos ) {
	VID n = G.numVertices();
	EID m = G.numEdges();
	const EID * const gindex = G.getIndex();
	const VID * const gedges = G.getEdges();

	// Set of eligible neighbours
	VID ns = num_neighbours;

	assert( ( ns + bits_per_lane - 1 ) / bits_per_lane <= m_words );
	m_matrix = m_matrix_alc = new type[m_words * ns + 64];
	intptr_t p = reinterpret_cast<intptr_t>( m_matrix );
	if( p & 63 ) // 63 = 512 bits / 8 bits per byte - 1
	    m_matrix = &m_matrix[64 - (p&63)/sizeof(type)];
	static_assert( Bits <= 512, "AVX512 requires 64-byte alignment" );
	std::fill( m_matrix, m_matrix+ns*m_words, type(0) );

	// Place edges
	const VID * vidx = &prestudy[v*((size_t(1)<<levels)+1)];
	VID ni = 0;
	m_m = 0;
	for( VID su=0; su < ns; ++su ) {
	    VID u = s2g[su];

	    const VID * q = &gedges[gindex[u]];
	    const VID * const qe = &gedges[gindex[u+1]];
	    const VID * uidx = &prestudy[u*((size_t(1)<<levels)+1)];

	    bitmask_output_iterator<type, VID>
		row_u( &m_matrix[m_words * su], n2s, neighbours, // s2g,
		       m_start_pos,
		       su >= m_start_pos ? 0 : m_start_pos );

	    row_u = graptor::merge_partitioned<utr>::template intersect<true>(
		    neighbours, neighbours+num_neighbours,
		    // s2g, s2g+ns, // simplifies vectorized creation bitmask, but ONLY scalar hash, not partitioned
		    q, qe,
		    H.get_adjacency( u ),
		    levels, 0, 1<<levels, vidx, uidx, row_u );

	    VID deg = row_u.get_degree();
#if !ABLATION_PDEG
	    m_degree[su] = deg;
#endif
	    m_m += deg;
	}

	m_n = ns;
    }
    template<typename utr, typename HGraph>
    DenseMatrix( const ::GraphCSx & G,
		 const HGraph & H,
		 VID v,
		 const NeighbourCutOutDegeneracyOrder<VID,EID> & cut,
		 utr )
	: m_start_pos( cut.get_start_pos() ) {
	// This version assumes that the graph has been relabeled with
	// vertices placed in order of non-increasing coreness.
	// Consequence is that coreness does not need checked and that
	// coreness[u] <= coreness[v] iff u <= v.
	// In this case, neighbours and s2g should coincide.
	// TODO:
	// * The LHS argument to intersect is typically the longer one (due to
	//   higher coreness), which is sub-optimal.
	// Note:
	// This version simplifies the vectorized creation of the bitmask
	// (row contents) as successive elements in the neighbour list
	// (s2g / adjacency of v) are tested, and these elements correspond
	// to successive indices in the dense matrix.
	const VID * const s2g = cut.get_vertices();
	VID n = G.numVertices();
	EID m = G.numEdges();
	const EID * const gindex = G.getIndex();
	const VID * const gedges = G.getEdges();

#if __AVX512F__
	constexpr VID VL = 64 / sizeof(VID);
#elif __AVX2__
	constexpr VID VL = 32 / sizeof(VID);
#else
	constexpr VID VL = 1;
#endif

	// Set of eligible neighbours
	VID ns = cut.get_num_vertices();

	assert( ( ns + bits_per_lane - 1 ) / bits_per_lane <= m_words );
	m_matrix = m_matrix_alc = new type[m_words * ns + 64];
	intptr_t p = reinterpret_cast<intptr_t>( m_matrix );
	if( p & 63 ) // 63 = 512 bits / 8 bits per byte - 1
	    m_matrix = &m_matrix[64 - (p&63)/sizeof(type)];
	static_assert( Bits <= 512, "AVX512 requires 64-byte alignment" );
	std::fill( m_matrix, m_matrix+ns*m_words, type(0) );

	// Place edges
	VID ni = 0;
	m_m = 0;
	for( VID su=0; su < ns; ++su ) {
	    VID u = s2g[su];

	    bitmask_lhs_sorted_output_iterator<type, VID, true, false>
		row_u( &m_matrix[m_words * su], s2g,
		       m_start_pos,
		       su >= m_start_pos ? 0 : m_start_pos );

	    // Trim off vertices that will be filtered out, but keep alignment.
	    const VID * const s2g_start
		= su >= m_start_pos ? s2g
		: s2g + ( m_start_pos & ~( VL - 1 ) );

	    if constexpr ( utr::uses_hash ) {
		row_u = utr::template intersect<true>(
		    s2g_start, s2g+ns, H.get_adjacency( u ), row_u );
	    } else {
		const VID * q = &gedges[gindex[u]];
		const VID * const qe = &gedges[gindex[u+1]];

		row_u = utr::template intersect<true>(
		    s2g_start, s2g+ns, q, qe, row_u );
	    }
	    
	    VID deg = row_u.get_degree();
#if !ABLATION_PDEG
	    m_degree[su] = deg;
#endif
	    m_m += deg;
	}

	m_n = ns;
    }
    template<typename utr, typename HGraph>
    DenseMatrix( const ::GraphCSx & G,
		 const HGraph & H,
		 VID v,
		 const NeighbourCutOutDegeneracyOrder<VID,EID> & cut,
		 size_t levels,
		 const VID * prestudy,
		 utr )
	: m_start_pos( cut.get_start_pos() ) {
	// This version assumes that the graph has been relabeled with
	// vertices placed in order of non-increasing coreness.
	// Consequence is that coreness does not need checked and that
	// coreness[u] <= coreness[v] iff u <= v.
	// In this case, neighbours and s2g should coincide.
	// TODO:
	// * The LHS argument to intersect is typically the longer one (due to
	//   higher coreness), which is sub-optimal.
	// Note:
	// This version simplifies the vectorized creation of the bitmask
	// (row contents) as successive elements in the neighbour list
	// (s2g / adjacency of v) are tested, and these elements correspond
	// to successive indices in the dense matrix.
	const VID * const s2g = cut.get_vertices();
	VID n = G.numVertices();
	EID m = G.numEdges();
	const EID * const gindex = G.getIndex();
	const VID * const gedges = G.getEdges();

#if __AVX512F__
	constexpr VID VL = 64 / sizeof(VID);
#elif __AVX2__
	constexpr VID VL = 32 / sizeof(VID);
#else
	constexpr VID VL = 1;
#endif

	// Set of eligible neighbours
	VID ns = cut.get_num_vertices();

	assert( ( ns + bits_per_lane - 1 ) / bits_per_lane <= m_words );
	m_matrix = m_matrix_alc = new type[m_words * ns + 64];
	intptr_t p = reinterpret_cast<intptr_t>( m_matrix );
	if( p & 63 ) // 63 = 512 bits / 8 bits per byte - 1
	    m_matrix = &m_matrix[64 - (p&63)/sizeof(type)];
	static_assert( Bits <= 512, "AVX512 requires 64-byte alignment" );
	std::fill( m_matrix, m_matrix+ns*m_words, type(0) );

	// Place edges
	const VID * vidx = &prestudy[v*((size_t(1)<<levels)+1)];
	VID ni = 0;
	m_m = 0;
	for( VID su=0; su < ns; ++su ) {
	    VID u = s2g[su];

	    const VID * q = &gedges[gindex[u]];
	    const VID * const qe = &gedges[gindex[u+1]];
	    const VID * uidx = &prestudy[u*((size_t(1)<<levels)+1)];

	    // Trim off vertices that will be filtered out, but keep alignment.
	    // Note: must keep alignment with prestudy data...
	    // const VID * const s2g_start
	    // = su >= m_start_pos ? s2g
	    // : s2g + ( m_start_pos & ~( VL - 1 ) );

	    bitmask_lhs_sorted_output_iterator<type, VID, false, false>
		row_u( &m_matrix[m_words * su], s2g,
		       m_start_pos,
		       su >= m_start_pos ? 0 : m_start_pos );

	    row_u = graptor::merge_partitioned<utr>::template intersect<true>(
		// s2g_start, s2g+ns,
		s2g, s2g+ns,
		q, qe,
		H.get_adjacency( u ),
		levels, 0, 1<<levels, vidx, uidx, row_u );

	    VID deg = row_u.get_degree();
#if !ABLATION_PDEG
	    m_degree[su] = deg;
#endif
	    m_m += deg;
	}

	m_n = ns;
    }
    DenseMatrix( VID n, size_t edges )
	: m_n( n ), m_m( 0 ), m_start_pos( 0 ) {
	VID ns = n;

	assert( ( ns + bits_per_lane - 1 ) / bits_per_lane <= m_words );
	m_matrix = m_matrix_alc = new type[m_words * ns + 64];
	intptr_t p = reinterpret_cast<intptr_t>( m_matrix );
	if( p & 63 ) // 63 = 512 bits / 8 bits per byte - 1
	    m_matrix = &m_matrix[64 - (p&63)/sizeof(type)];
	static_assert( Bits <= 512, "AVX512 requires 64-byte alignment" );
	std::fill( m_matrix, m_matrix+ns*m_words, type(0) );
#if !ABLATION_PDEG
	std::fill( m_degree, m_degree+ns, DID(0) );
#endif

	size_t e = 1;
	for( VID i=0; i < n-1; ++i ) {
	    VID deg = 0;
	    for( VID j=0; j < i; ++j ) {
		if( ( edges & e ) != 0 ) {
		    set( i, j );
		    set( j, i );
		    m_m += 2;
#if !ABLATION_PDEG
		    m_degree[i]++;
		    m_degree[j]++;
#endif
		}
		e <<= 1;
	    }
	}
    }
#endif
    // In this variation, hVIDs are already sorted by core_order
    // and we know the separation between X and P sets.
    template<typename GGraph, typename HGraph>
    DenseMatrix( const GGraph & G,
		 const HGraph & H,
		 const sVID * XP,
		 sVID ne, sVID ce )
	: m_start_pos( ne ) {
	// Vertices in X and P are independently already sorted by core order.
	// We do not reorder, primarily because only the order in P matters
	// for efficiency of enumeration.
	// Do we need to copy, or can we just keep a pointer to XP?
	VID ns = ce;

	assert( ( ns + bits_per_lane - 1 ) / bits_per_lane <= m_words );
	m_matrix = m_matrix_alc = new type[m_words * ns + 64];
	intptr_t p = reinterpret_cast<intptr_t>( m_matrix );
	if( p & 63 ) // 63 = 512 bits / 8 bits per byte - 1
	    m_matrix = &m_matrix[64 - (p&63)/sizeof(type)];
	static_assert( Bits <= 512, "AVX512 requires 64-byte alignment" );

	std::fill( m_matrix, m_matrix+ns*m_words, type(0) );

#if !ABLATION_DENSE_DISABLE_XP_HASH
	// Build hash table. Note: comparable hash set already exists in
	// the graph H, however, not set up to do remapping.
	hash_table<sVID, sVID, typename HGraph::Hash> XP_hash( ce );
	for( sVID i=0; i < ce; ++i )
	    XP_hash.insert( XP[i], i );
#endif

	// Place edges
	VID ni = 0;
	m_m = 0;
	m_fully_connected = tr::setzero();

	row_type mn = get_himask( ns );
	row_type mx = get_himask( m_start_pos );
	row_type allP = tr::bitwise_xor( mn, mx );
	row_type all = tr::bitwise_invert( mn );

	for( VID su=0; su < ns; ++su ) {
	    VID u = XP[su];

	    // Attempting a hash lookup in XP when XP is much longer than the
	    // neighbour list of the vertex appears not beneficial to
	    // performance in this location

	    VID adeg;
	    row_type row_u;
#if !ABLATION_DENSE_DISABLE_XP_HASH
	    if( ce > 2*G.getDegree( u ) ) {
		std::tie( row_u, adeg )
		    = graptor::graph::construct_row_hash_xp_vec<tr>(
			G, H, XP_hash, ne, ce, su, u,
			( su >= ne ? 0 : ne ),
			ce, (sVID)0 ); 
		tr::store( &m_matrix[VL * su], row_u );
	    } else
#endif
	    {
		// Best option
		adeg = construct_row_hash_adj<tr>(
		    G, H, &m_matrix[VL * su], XP, ne, ce, su,
		    ( su >= ne ? 0 : ne ), ce );
		row_u = tr::load( &m_matrix[VL * su] );
	    }

	    row_type su_only = create_row( su );
	    row_type eql = allP; // su >= m_start_pos ? allP : all;
	    if( tr::cmpeq( tr::bitwise_andnot( su_only, eql ),
			   tr::bitwise_and( row_u, eql ),
			   target::mt_bool() ) )
		m_fully_connected = tr::bitwise_or(
		    m_fully_connected, su_only );

#if !ABLATION_PDEG
	    m_degree[su] = adeg;
#endif
	    m_m += adeg;
	}

	m_n = ns;
    }
    // In this variation, hVIDs are already sorted by core_order
    // and we know the separation between X and P sets.
    template<typename GGraph, typename HGraph>
    DenseMatrix( const GGraph & G,
		 const HGraph & H,
		 const XPSet<sVID> & xp_set,
		 sVID ne, sVID ce )
	: m_start_pos( ne ) {
	// Vertices in X and P are independently already sorted by core order.
	// We do not reorder, primarily because only the order in P matters
	// for efficiency of enumeration.
	// Do we need to copy, or can we just keep a pointer to XP?
	VID ns = ce;

	const sVID * const XP = xp_set.get_set();

	assert( ( ns + bits_per_lane - 1 ) / bits_per_lane <= m_words );
	m_matrix = m_matrix_alc = new type[m_words * ns + 64];
	intptr_t p = reinterpret_cast<intptr_t>( m_matrix );
	if( p & 63 ) // 63 = 512 bits / 8 bits per byte - 1
	    m_matrix = &m_matrix[64 - (p&63)/sizeof(type)];
	static_assert( Bits <= 512, "AVX512 requires 64-byte alignment" );

	std::fill( m_matrix, m_matrix+ns*m_words, type(0) );

	// Place edges
	VID ni = 0;
	m_m = 0;
	m_fully_connected = tr::setzero();

	row_type mn = get_himask( ns );
	row_type mx = get_himask( m_start_pos );
	row_type allP = tr::bitwise_xor( mn, mx );
	row_type all = tr::bitwise_invert( mn );

	for( VID su=0; su < ns; ++su ) {
	    const VID u = XP[su];

	    // Attempt a hash lookup in XP when XP is much longer than the
	    // neighbour list of the vertex.

	    VID adeg;
	    row_type row_u;
#if !ABLATION_DENSE_DISABLE_XP_HASH
	    if( ce > 2*G.getDegree( u ) ) {
		std::tie( row_u, adeg )
		    = graptor::graph::construct_row_hash_xp_vec<tr>(
			G, H, xp_set.hash_table(), ne, ce, su, u,
			( su >= ne ? 0 : ne ),
			ce, (sVID)0 ); 
		tr::store( &m_matrix[VL * su], row_u );
	    } else
#endif
	    {
		// Best option for hashing adjacency
		adeg = construct_row_hash_adj<tr>(
		    G, H, &m_matrix[VL * su], XP, ne, ce, su,
		    ( su >= ne ? 0 : ne ), ce );
		row_u = tr::load( &m_matrix[VL*su] );
	    }

	    row_type su_only = create_row( su );
	    row_type eql = allP; // su >= m_start_pos ? allP : all;
	    if( tr::cmpeq( tr::bitwise_andnot( su_only, eql ),
			   tr::bitwise_and( row_u, eql ),
			   target::mt_bool() ) )
		m_fully_connected = tr::bitwise_or(
		    m_fully_connected, su_only );

#if !ABLATION_PDEG
	    m_degree[su] = adeg;
#endif
	    m_m += adeg;
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
	row_type mn = get_himask( m_n );
	row_type mx = get_himask( m_start_pos );
#if !ABLATION_DENSE_FILTER_FULLY_CONNECTED
	row_type allX = tr::bitwise_invert( mx );
	row_type allP = tr::bitwise_xor( mn, mx );

	// Special case of vertices connected to all other vertices.
	// If we select such a vertex as pivot, only the vertex itself is
	// processed at this level. The remaining vertices remain in P.
	// This loops simulates multiple recurisve calls selecting a fully
	// connected vertex as pivot
	unsigned depth = 0;
	row_type R = tr::setzero();
	if( !tr::is_zero( m_fully_connected ) ) {
	    // If an X vertex is connected to all P vertices, we're done
	    if( !tr::is_zero( tr::bitwise_andnot( mx, m_fully_connected ) ) )
		return;

	    row_type fcP = tr::bitwise_and( mx, m_fully_connected );
	    bitset<Bits> bx( fcP );

	    // fully connected P vertices
	    for( auto I = bx.begin(), E = bx.end(); I != E; ++I ) {
		sVID v = *I;
		row_type v_ngh = get_row( v );
		row_type v_row = create_row( v );
		// Add v to R, it must be included
		R = tr::bitwise_or( v_row, R );
		++depth;
		// Filter X and P with neighbours of v.
		// As v_ngh has only one zero (for v), the effect
		// is to remove v
		allX = tr::bitwise_and( allX, v_ngh );
		allP = tr::bitwise_and( allP, v_ngh );
	    }
	}
	mce_bk_iterate( E, R, allP, allX, depth );
#else
#if !ABLATION_DENSE_NO_PIVOT_TOP
	row_type allX = tr::bitwise_invert( mx );
	row_type allP = tr::bitwise_xor( mn, mx );
	mce_bk_iterate( E, tr::setzero(), allP, allX, 0 );
#else	
	// No pivoting, process all
	parallel_loop( m_start_pos, m_n, 1, [&]( sVID u ) {
	    row_type R = create_row( u );

	    row_type pu_ngh = get_row( u );
	    row_type h = get_himask( u );
	    row_type Ppv = tr::bitwise_and( h, pu_ngh );
	    row_type Xpv = tr::bitwise_andnot( h, pu_ngh );
	    row_type Rv = R;
	    mce_bk_iterate( E, Rv, Ppv, Xpv, 1 );
	} );
#endif
#endif
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
    // EID numEdges() const { return m_m; }

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
	if( tr::is_zero( P ) ) {
	    if( tr::is_zero( X ) )
		EE( bitset<Bits>( R ), depth );
	    return;
	}

	VID pivot = mce_get_pivot( P, X );
	row_type pivot_ngh = get_row( pivot );
	row_type x = tr::bitwise_andnot( pivot_ngh, P );

	auto task = [=,&EE]( VID u, row_type u_only ) {
	    // row_type u_only = tr::setglobaloneval( u );
	    row_type u_ngh = get_row( u );
	    row_type h = get_himask( u );
	    row_type u_done = tr::bitwise_andnot( h, x );

	    row_type P_shrink = tr::bitwise_andnot( u_done, P );
	    row_type Pv = tr::bitwise_and( P_shrink, u_ngh );
	    row_type X_grow = tr::bitwise_or( X, u_done );
	    row_type Xv = tr::bitwise_and( X_grow, u_ngh );
	    row_type Rv = tr::bitwise_or( R, u_only );
	    mce_bk_iterate( EE, Rv, Pv, Xv, depth+1 );
	};
	
	bitset<Bits> bx( x );
	VID nset = get_size( x );
	if( Bits >= DENSE_THRESHOLD_SEQUENTIAL_BITS
	    && float(get_size(P)) >= DENSE_THRESHOLD_SEQUENTIAL
	    && nset > 1 ) {
	    if( float(nset)/float(m_n) >= DENSE_THRESHOLD_DENSITY ) {
		// High number of vertices to process + density
		parallel_loop( (VID)0, (VID)m_n, 1, [&,x]( VID u ) {
		    row_type u_only = tr::setglobaloneval( u );
		    if( !tr::is_zero( tr::bitwise_and( u_only, x ) ) )
			// Vertex needs to be processed
			task( u, u_only );
		} );
	    } else {
		VID elms[Bits];
		VID * elms_end = target::expand_bitset<
		    typename tr::member_type,sVID,tr::vlen>::compute(
			x, m_n, elms, 0 );
		size_t nset = elms_end - elms;
		parallel_loop( (size_t)0, nset, 1, [&]( size_t i ) {
		    VID u = elms[i];
		    row_type u_only = tr::setglobaloneval( u );
		    task( u, u_only );
		} );
	    }
	} else {
	    for( auto I = bx.begin(), E = bx.end(); I != E; ++I ) {
		VID u = *I;
		row_type u_only = tr::setglobaloneval( u );
		task( u, u_only );
	    }
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

#if !ABLATION_DENSE_EXCEED
	// Avoid complexities if there is not much choice
	if( get_size( P ) <= 3 )
	    return p_best;
#endif
	
	for( auto I=b.begin(), E=b.end(); I != E; ++I ) {
	    VID v = *I;
#if !ABLATION_DENSE_EXCEED && !ABLATION_PDEG
	    if( (VID)m_degree[v] < p_ins ) // skip if cannot be best
		continue;
#endif
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

    row_type create_row( VID v ) {
	return tr::setglobaloneval( v );
    }

    row_type get_himask( VID v ) {
	return tr::himask( v+1 );
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

    row_type m_fully_connected;

#if !ABLATION_PDEG
    DID m_degree[Bits]; //!< only left-most columns (P part)
#endif
};

} // namespace graph

} // namespace graptor

#endif // GRAPHGRIND_GRAPH_SIMPLE_DENSE_H
