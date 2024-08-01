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
#include "graptor/graph/simple/csxd.h"
#include "graptor/target/vector.h"
#include "graptor/container/intersect.h"

#ifndef DENSE_THRESHOLD_SEQUENTIAL
#define DENSE_THRESHOLD_SEQUENTIAL 64
#endif

#ifndef DENSE_THRESHOLD_SEQUENTIAL_BITS
#define DENSE_THRESHOLD_SEQUENTIAL_BITS 32
#endif

// As a fraction of 1/1000s
#ifndef DENSE_THRESHOLD_DENSITY
#define DENSE_THRESHOLD_DENSITY 500
#endif


namespace graptor {

namespace graph {

#if 0
std::ostream & operator << ( std::ostream & os, __m512i m ) {
    using tr = vector_type_traits_vl<uint64_t,8>;
    return os << "{ " << std::hex
	      << tr::lane( m, 0 ) << ", "
	      << tr::lane( m, 1 ) << ", "
	      << tr::lane( m, 2 ) << ", "
	      << tr::lane( m, 3 ) << ", "
	      << tr::lane( m, 4 ) << ", "
	      << tr::lane( m, 5 ) << ", "
	      << tr::lane( m, 6 ) << ", "
	      << tr::lane( m, 7 ) << std::dec << " }";
}
#endif

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
	: m_allv_mask( tr::bitwise_invert( tr::himask( ce+1 ) ) ),
	  m_start_pos( ne ) {
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
	m_d_max = 0;
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
	    if( adeg > m_d_max )
		m_d_max = adeg;
	}

	m_n = ns;
    }
    // In this variation, hVIDs are already sorted by core_order
    // and we know the separation between X and P sets.
    DenseMatrix( const GraphCSxDepth<sVID,sEID> & G,
		 const sVID * XP, sVID ce )
	: m_allv_mask( tr::bitwise_invert( tr::himask( ce+1 ) ) ),
	  m_start_pos( 0 ) {
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

	// Build hash table. Note: comparable hash set already exists in
	// the graph H, however, not set up to do remapping.
	hash_table<sVID, sVID, rand_hash<sVID>> XP_hash( ce );
	for( sVID i=0; i < ce; ++i )
	    XP_hash.insert( XP[i], i );

	// Place edges
	VID ni = 0;
	m_m = 0;
	m_d_max = 0;
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

	    std::tie( row_u, adeg )
		= graptor::graph::construct_row_hash_xp_vec<tr>(
		    G, G, XP_hash, (sVID)0, ce, su, u, (sVID)0, ce, (sVID)0 ); 
	    tr::store( &m_matrix[VL * su], row_u );

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
	    if( adeg > m_d_max )
		m_d_max = adeg;
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
	: m_allv_mask( tr::bitwise_invert( tr::himask( ce+1 ) ) ),
	  m_start_pos( ne ) {
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

    DenseMatrix( const DenseMatrix & base )
	: m_allv_mask( base.m_allv_mask ),
	  m_n( base.m_n ), m_start_pos( base.m_start_pos ) {
	VID ns = m_n;
	assert( ( ns + bits_per_lane - 1 ) / bits_per_lane <= m_words );
	m_matrix = m_matrix_alc = new type[m_words * ns + 64];
	intptr_t p = reinterpret_cast<intptr_t>( m_matrix );
	if( p & 63 ) // 63 = 512 bits / 8 bits per byte - 1
	    m_matrix = &m_matrix[64 - (p&63)/sizeof(type)];
	static_assert( Bits <= 512, "AVX512 requires 64-byte alignment" );

	// std::fill( m_matrix, m_matrix+ns*m_words, type(0) );
    }
    DenseMatrix( VID n, const row_type * row )
	: m_allv_mask( tr::bitwise_invert( tr::himask( n+1 ) ) ),
	  m_n( n ), m_start_pos( 0 ) {
	VID ns = m_n;
	assert( ( ns + bits_per_lane - 1 ) / bits_per_lane <= m_words );
	m_matrix = m_matrix_alc = new type[m_words * ns + 64];
	intptr_t p = reinterpret_cast<intptr_t>( m_matrix );
	if( p & 63 ) // 63 = 512 bits / 8 bits per byte - 1
	    m_matrix = &m_matrix[64 - (p&63)/sizeof(type)];
	static_assert( Bits <= 512, "AVX512 requires 64-byte alignment" );
	std::copy( row, row+n, &m_matrix[0] );
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
	//
	// TODO: Also look at all vertices without P neighbours. These
	//       give immediately rise to 1 maximal clique and can be further
	//       discarded from the P set.
	unsigned depth = 0;
	row_type R = tr::setzero();
	if( !tr::is_zero( m_fully_connected ) ) {
	    // If an X vertex is connected to all P vertices, we're done
	    if( !tr::is_zero( tr::bitwise_andnot( mx, m_fully_connected ) ) ) {
		E.count_fully_connected_X();
		return;
	    }

	    row_type fcP = tr::bitwise_and( mx, m_fully_connected );
	    bitset<Bits> bx( fcP );

	    // fully connected P vertices
	    for( auto I = bx.begin(), E = bx.end(); I != E; ++I ) {
		sVID v = *I;
		row_type v_ngh = get_row( v );
		row_type v_row = I.get_mask(); // create_row( v );
		// Add v to R, it must be included
		R = tr::bitwise_or( v_row, R );
		++depth;
		// Filter X and P with neighbours of v.
		// As v_ngh has only one zero (for v), the effect
		// is to remove v
		allX = tr::bitwise_and( allX, v_ngh );
		allP = tr::bitwise_and( allP, v_ngh );
	    }
	    E.count_fully_connected_P( depth );
	}

#if 0
	mce_filter_P( E, R, allP, allX, depth );
#endif

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

    template<typename Enumerate>
    void
    mc_search( Enumerate && E, size_t depth ) {
	m_mc = tr::setzero();
	m_mc_size = 0;

	row_type mn = get_himask( m_n );
	row_type R = m_fully_connected;
	// row_type R = tr::setzero();
	row_type allP = tr::bitwise_andnot( R, tr::bitwise_invert( mn ) );
	// row_type allP = tr::bitwise_invert( mn );

	// depth is 1 for top-level vertex
	// every fully-connected vertex is automatically included in clique
	if( !tr::is_zero( R ) )
	    depth += get_size( R );
#if PIVOT_COLOUR_DENSE
	mc_iterate_colour( E, R, allP, depth );
#else
	mc_iterate( E, R, allP, depth );
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

    /*! \brief compute minimal vertex cover
     *
     * \tparam co true if operating on complement of stored graph
     * \param k_max find a cover of size k_max or less
     */
    template<bool co>
    std::pair<bitset<Bits>,VID>
    vertex_cover_kernelised( VID k_max ) {
	timer tm;
	tm.start();
	
	// Complement graph:
	// Remove fully-connected vertices from the candidate list. They are
	// automatically not part of the vertex cover as they have zero degree
	// in the complement graph.
	row_type all_vertices;
	if constexpr ( co )
	    all_vertices = tr::bitwise_andnot( m_fully_connected, m_allv_mask );
	else
	    all_vertices = m_allv_mask;

	// Attempt to find a vertex cover that is smaller than
	// the size required to improve on the best known clique.
	// This is expected to fail. If it does succeed, we continue
	// searching for smaller covers.
	if( k_max == 0 )
	    return std::make_pair( bitset<Bits>( tr::setzero() ), ~(VID)0 );
	VID best_size = k_max + 1;
	row_type best_cover = tr::setzero();
	VID k_up = k_max - 1;
	VID k_lo = 1;
	VID k_best_size = k_max;
	VID k = k_up;
	bool first_attempt = true;
	while( true ) {
	    VID a_best_size = 0;
	    row_type a_best_cover = tr::setzero();
	    bool any = vck_iterate<true,co>(
		k, 1, all_vertices, a_best_size, a_best_cover );
	    /*
	    std::cout << " vck: k_max=" << k_max << " k=[" << k_lo << ','
		      << k << ',' << k_up << "] bs=" << a_best_size
		      << " ok=" << any
		      << ' ' << tm.next()
		      << "\n";
	    */
	    if( any ) {
		best_size = a_best_size;
		best_cover = a_best_cover;
		// in case we find a better cover than requested
		if( k > best_size )
		    k = best_size;
	    }

	    // Reduce range for k
	    if( any ) // k too high
		k_up = k;
	    else
		k_lo = k;
	    if( k_up <= k_lo+1 )
		break;

	    // Determine next k
	    if( first_attempt ) {
		first_attempt = false;
		k = k_up - 1;
	    } else
		k = ( k_up + k_lo ) / 2;
	}

	// If we can't meet the constraint, return something quick and
	// recognisable as unuseful.
	if( best_size > k_max )
	    return std::make_pair( bitset<Bits>( tr::setzero() ), ~(VID)0 );

	return std::make_pair( bitset<Bits>( best_cover ), best_size );
    }

    bitset<Bits>
    clique_via_vertex_cover( VID k_max ) {
	auto r = vertex_cover_kernelised<true>( k_max );
	if( r.second == ~(VID)0 )
	    return r.first;
	
	row_type best_cover = r.first;
	VID best_size = r.second;

	// cover vs clique on complement. Invert bitset, mask with valid bits
	// best_cover = tr::bitwise_invert(
	// tr::bitwise_or( best_cover, get_himask( m_n ) ) );
	best_cover = tr::bitwise_andnot( best_cover, m_allv_mask );
	best_size = m_n - best_size; // for completeness; unused hereafter

	// Check for clique (debugging)
	for( VID r=0; r < m_n; ++r ) {
	    row_type rr = create_row( r );
	    row_type rm = get_row( r );

	    // Add self vertex and intersect with cover
	    row_type rc = tr::bitwise_or( rr, rm );
	    row_type ins = tr::bitwise_and( rc, best_cover );
	    VID sz = get_size( ins );

	    if( tr::is_bitwise_and_zero( rr, best_cover ) )
		assert( sz < best_size );
	    else
		assert( sz == best_size );
	}
	
	return bitset<Bits>( best_cover );
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
	row_type R, row_type P, row_type X, unsigned int depth ) {
	if( tr::is_zero( P ) ) {
	    if( tr::is_zero( X ) )
		EE( bitset<Bits>( R ), depth );
	    return;
	}

#if 0
	// Apply filtering rules on vertices in P, removing vertices from P
	// or moving them from P to R.
	mce_filter_P( EE, R, P, X, depth );

	if( tr::is_zero( P ) ) {
	    if( tr::is_zero( X ) )
		EE( bitset<Bits>( R ), depth );
	    return;
	}
#endif

	VID pivot, nset;
	row_type pivot_ngh, x;

	while( true ) {
#if !ABLATION_DENSE_PIVOT_FILTER
	    pivot = mce_get_pivot( P, X );
#else
	    pivot = mce_get_pivot_and_filter( EE, R, P, X, depth );
#endif
	    pivot_ngh = get_row( pivot );
	    x = tr::bitwise_andnot( pivot_ngh, P );
	    nset = get_size( x );

#if !ABLATION_DENSE_ITERATE
	    if( nset == 0 )
		// P\N(pivot) is empty. If we were to proceed without
		// pivoting, the pivot would remain in X as it
		// is a neighbour of all elements of P.
		return;
	    else if( nset == 1 ) {
		// Only one vertex is iterated over. Simply adopt
		// and search for next, to reduce depth of recursion
		sVID u = target::alltzcnt<sVID,type,VL>::compute( x );
		row_type u_ngh = get_row( u );
		row_type u_only = x;

		P = tr::bitwise_and( P, u_ngh );
		X = tr::bitwise_and( X, u_ngh );
		R = tr::bitwise_or( R, u_only );
		++depth;

		if( tr::is_zero( P ) ) {
		    if( tr::is_zero( X ) )
			EE( bitset<Bits>( R ), depth );
		    return;
		}
	    } else
#endif // !ABLATION_DENSE_ITERATE
		break;
	};

	auto task = [=,&EE,this]( VID u, row_type u_only ) {
	    row_type u_ngh = get_row( u );
	    row_type h = get_himask( u );
	    row_type u_done = tr::bitwise_andnot( h, x );

	    // row_type P_shrink = tr::bitwise_andnot( u_done, P );
	    // row_type Pv = tr::bitwise_and( P_shrink, u_ngh );
	    row_type Pv = tr::bitwise_andnot( u_done, P, u_ngh );
	    // row_type X_grow = tr::bitwise_or( X, u_done );
	    // row_type Xv = tr::bitwise_and( X_grow, u_ngh );
	    row_type Xv = tr::bitwise_or_and( X, u_done, u_ngh );
	    row_type Rv = tr::bitwise_or( R, u_only );
	    mce_bk_iterate( EE, Rv, Pv, Xv, depth+1 );
	};
	
	if( Bits >= DENSE_THRESHOLD_SEQUENTIAL_BITS
	    && get_size(P) >= DENSE_THRESHOLD_SEQUENTIAL
	    && nset > 1 ) {
	    if( nset*1000 >= DENSE_THRESHOLD_DENSITY*m_n ) {
		// High number of vertices to process + density
		parallel_loop( (VID)0, (VID)m_n, 1, [&,x]( VID u ) {
		    row_type u_only = tr::setglobaloneval( u );
		    row_type ins = tr::bitwise_and( u_only, x );
		    if( !tr::is_zero( ins ) )
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
	    bitset<Bits> bx( x );
	    for( auto I = bx.begin(), E = bx.end(); I != E; ++I ) {
		VID u = *I;
		row_type u_only = I.get_mask();
		task( u, u_only );
	    }
	}
    }

    // Colour graph induced by P
    auto induced_graph_colour( row_type P, VID n_collect ) {
	bitset<Bits> b( P );
	using sgVID = std::make_signed_t<VID>;
	std::vector<sgVID> col( m_n );
	std::vector<uint8_t> histo( Bits );
	sgVID last_col = 0;
	row_type fc = tr::setzero();
	for( auto I=b.begin(), E=b.end(); I != E; ++I ) {
	    VID v = *I;

	    // Colours used by neighbours
	    // row_type histo = tr::setzero();

	    // Neighbours in induced subgraph that were visited before
	    row_type ngh = tr::bitwise_and( P, get_row( v ) );
	    row_type v_mask = get_himask( v );
	    ngh = tr::bitwise_andnot( v_mask, ngh );
	    bitset<Bits> bn( ngh );
	    sgVID max_h = -1;
	    // sgVID max_h = 0;
	    for( auto u : bn ) {
		histo[col[u]] = uint8_t(1); // highest bit set
		// histo[col[u]] = uint8_t(0xff); // highest bit set
		// histo = tr::bitwise_or( histo, create_row( col[u] ) );
		if( max_h < col[u] )
		    max_h = col[u];
	    }

	    // TODO: consider making histo bitset and using alltzcnt for this
	    //       loop
	    if( max_h == -1 ) {
		col[v] = 0;
	    } else {
		col[v] = max_h + 1; // if no better colour found
		VID c = 0;
		while( c <= max_h ) {
		    if( histo[c] == 0 ) {
			col[v] = c;
			break;
		    }
		    histo[c] = uint8_t(0);
		    ++c;
		}
		while( c <= max_h )
		    histo[c++] = uint8_t(0);
	    }
/*
	    static constexpr size_t VLB = 8; // 16 on AVX512
	    using trb = vector_type_traits_vl<uint8_t,VLB>;
	    size_t i;
	    for( i=0; i <= max_h; i += VLB ) {
		auto chunk = trb::loadu( &histo[i] );
		auto mask = ~trb::asmask( chunk );
		trb::storeu( &histo[i], trb::setzero() );
		if( mask != 0 ) {
		    col[v] = _tzcnt_u32( mask ) + ( i & ~(VLB-1) );
		    break;
		}
	    }
	    for( ; i <= max_h; i += VLB )
		trb::storeu( &histo[i], trb::setzero() );
*/
		
/*
	    VID c = target::alltzcnt<sVID,type,VL>::compute( histo );
	    col[v] = c;
*/

	    // Track number of colours
	    if( last_col < col[v] )
		last_col = col[v];
	    // Build bit mask of all vertices in n_collect first colours
	    if( col[v] < n_collect )
		fc = tr::bitwise_or( fc, I.get_mask() );
	}

	return std::make_tuple( last_col+1, fc );
    }

    template<typename Enumerate>
    void mc_iterate_colour(
	Enumerate && EE,
	row_type R, row_type P, unsigned int depth ) {
	if( depth + get_size( P ) < EE.get_max_clique_size() )
	    return;
	
	if( tr::is_zero( P ) ) {
	    bitset<Bits> b( R );
	    EE.record( depth, b.begin(), b.end() );
	    return;
	}

	row_type x;

	while( true ) {
	    VID tol = EE.get_max_clique_size() - depth;
	    auto [ num_col, skipv ] = induced_graph_colour( P, tol );

	    if( depth + num_col < EE.get_max_clique_size() )
		return;

	    x = tr::bitwise_andnot( skipv, P );
	    VID nset = get_size( x );

	    if( nset == 0 )
		// P\N(pivot) is empty. If we were to proceed without
		// pivoting, the pivot would remain in X as it
		// is a neighbour of all elements of P.
		return;
	    else if( nset == 1 ) {
		// Only one vertex is iterated over. Simply adopt
		// and search for next, to reduce depth of recursion
		sVID u = target::alltzcnt<sVID,type,VL>::compute( x );
		row_type u_only = x;
		row_type u_ngh = get_row( u );
		P = tr::bitwise_and( P, u_ngh );
		R = tr::bitwise_or( R, u_only );
		++depth;

		if( tr::is_zero( P ) ) {
		    bitset<Bits> b( R );
		    EE.record( depth, b.begin(), b.end() );
		    return;
		}
	    } else
		break;
	}

	auto task = [=,&EE,this]( VID u, row_type u_only ) {
	    row_type u_ngh = get_row( u );
	    row_type h = get_himask( u );
	    row_type u_done = tr::bitwise_andnot( h, x );

	    row_type Pv = tr::bitwise_andnot( u_done, P, u_ngh );
	    row_type Rv = tr::bitwise_or( R, u_only );
	    mc_iterate_colour( EE, Rv, Pv, depth+1 );
	};
	
	bitset<Bits> bx( x );
	for( auto I = bx.begin(), E = bx.end(); I != E; ++I ) {
	    VID u = *I;
	    row_type u_only = I.get_mask();
	    task( u, u_only );
	}
    }

#if 0
    bool sat_test_viable( row_type P, VID u,
			  const std::vector<VID> & col, VID num_col ) {
	using LID = std::make_signed_t<VID>;
	using CID = std::make_signed_t<EID>;
	row_type u_only = create_row( u );
	row_type vertices = tr::bitwise_or( P, u_only );
	size_t num_vertices = get_size( vertices );
	size_t induced_edges = calculate_num_edges( vertices );
	// All induced non-edges
	size_t hard_clauses =
	    ( vertices * ( vertices - 1 ) - induced_edges ) / 2;
	size_t hard_literals = 2 * hard_clauses;
	// All independent sets
	size_t soft_clauses = num_col + 1; // +1 for u
	size_t soft_literals = vertices;
	size_t num_clauses = hard_clauses + soft_clauses;
	size_t num_literals = hard_literals + soft_literals;
	SATClauses<LID,CID> clauses( num_clauses, num_literals );

	LID * cc = &clauses[0];
	size_t * ci = clauses.get_index();

	// Hard clauses
	size_t c=0;
	bitset<Bits> b( vertices );
	for( auto I=b.begin(), E=b.end(); I != E; ++I ) {
	    VID v = *I;
	    row_type non_edges = get_inv_row( v, I.get_mask(), vertices );
	    bitset<Bits> bne( non_edges );
	    for( auto u : bne ) {
		if( u < v ) {
		    ci[c] = 2*c;
		    cc[2*c] = -(LID)u;
		    cc[2*c+1] = -(LID)v;
		    c++;
		}
	    }
	}
	assert( c == hard_clauses );

	// Soft clauses
	size_t * nc = &ci[c];
	for( auto v : b )
	    if( v != u )
		nc[col[v]]++;
	nc[num_col] = 1; // u in its own class, at this particular colour
	nc[0] += 2*c;
	VID nv = std::exclusive_scan( nc, nc+num_col+1, nc );
	assert( nv == num_literals );
	for( auto v : b ) {
	    VID vcol = v == u ? num_col : col[v];
	    cc[nc[vcol]++] = (LID)v;
	}
	for( size_t i=c; i < c+num_col+1; ++i )
	    ci[i] = ci[i+1];
	ci[c+num_col+1] = num_literals;

	SATUnitPropagation<LID,CID> up( clauses, m_n );
	return up.unit_propagation();
    }

    template<typename Enumerate>
    void mc_iterate_coloursat(
	Enumerate && EE,
	row_type R, row_type P, unsigned int depth ) {
	if( depth + get_size( P ) < EE.get_max_clique_size() )
	    return;
	
	if( tr::is_zero( P ) ) {
	    bitset<Bits> b( R );
	    EE.record( depth, b.begin(), b.end() );
	    return;
	}

	VID tol = EE.get_max_clique_size() - depth;
	auto [ num_col, col, skipv ] = induced_graph_colour( P, tol );

	if( depth + num_col < EE.get_max_clique_size() )
	    return;

	row_type x = tr::bitwise_andnot( skipv, P );

	bitset<Bits> bx( x );
	for( auto I = bx.begin(), E = bx.end(); I != E; ++I ) {
	    VID u = *I;
	    if( !sat_test_viable( x, u, col, tol ) ) {
		std::cout << "SAT remove u=" << u << "\n";
		skipv = tr::bitwise_or( skipv, I.get_mask() );
		col[u] = tol++; // need to recolour? not for first addition...
		// maybe need to test if colour can be same as previously
		// added vertices, so maybe yes, recolour...
	    }
	}

	// Update x, if skipv has grown
	x = tr::bitwise_andnot( skipv, P );


	auto task = [=,&EE,this]( VID u, row_type u_only ) {
	    row_type u_ngh = get_row( u );
	    row_type h = get_himask( u );
	    row_type u_done = tr::bitwise_andnot( h, x );

	    row_type Pv = tr::bitwise_andnot( u_done, P, u_ngh );
	    row_type Rv = tr::bitwise_or( R, u_only );
	    mc_iterate_coloursat( EE, Rv, Pv, depth+1 );
	};
	
	bitset<Bits> bx( x );
	for( auto I = bx.begin(), E = bx.end(); I != E; ++I ) {
	    VID u = *I;
	    
	    row_type u_only = I.get_mask();
	    task( u, u_only );
	}
    }
#endif

    template<typename Enumerate>
    void mc_iterate(
	Enumerate && EE,
	row_type R, row_type P, unsigned int depth ) {
	if( depth + get_size( P ) < EE.get_max_clique_size() )
	    return;
	
	if( tr::is_zero( P ) ) {
	    bitset<Bits> b( R );
	    EE.record( depth, b.begin(), b.end() );
	    return;
	}

	VID pivot, nset;
	row_type pivot_ngh, x;

	while( true ) {
	    // pivot = target::alltzcnt<sVID,type,VL>::compute( P );
	    pivot = mc_get_pivot( P );
	    pivot_ngh = get_row( pivot );
	    x = tr::bitwise_andnot( pivot_ngh, P );
	    nset = get_size( x );

	    if( nset == 0 )
		// P\N(pivot) is empty. If we were to proceed without
		// pivoting, the pivot would remain in X as it
		// is a neighbour of all elements of P.
		return;
	    else if( nset == 1 ) {
		// Only one vertex is iterated over. Simply adopt
		// and search for next, to reduce depth of recursion
		sVID u = target::alltzcnt<sVID,type,VL>::compute( x );
		row_type u_ngh = get_row( u );
		row_type u_only = x;

		P = tr::bitwise_and( P, u_ngh );
		R = tr::bitwise_or( R, u_only );
		++depth;

		if( tr::is_zero( P ) ) {
		    bitset<Bits> b( R );
		    EE.record( depth, b.begin(), b.end() );
		    return;
		}
	    } else
		break;
	};

	auto task = [=,&EE,this]( VID u, row_type u_only ) {
	    row_type u_ngh = get_row( u );
	    row_type h = get_himask( u );
	    row_type u_done = tr::bitwise_andnot( h, x );

	    row_type Pv = tr::bitwise_andnot( u_done, P, u_ngh );
	    row_type Rv = tr::bitwise_or( R, u_only );
	    mc_iterate( EE, Rv, Pv, depth+1 );
	};
	
	bitset<Bits> bx( x );
	for( auto I = bx.begin(), E = bx.end(); I != E; ++I ) {
	    VID u = *I;
	    row_type u_only = I.get_mask();
	    task( u, u_only );
	}
    }

    template<typename Enumerate>
    void mce_filter_P( Enumerate && EE, row_type & R, row_type & P,
		       row_type & X, unsigned int & depth ) {
	// Any vertex v in P where P intersect N(v) is empty can be enumerated
	// and removed from P. It is as if we select this vertex first, then
	// consider the remainder of P as right siblings.
	bitset<Bits> b( P );
	for( auto I=b.begin(), E=b.end(); I != E; ++I ) {
	    VID v = *I;

	    row_type v_ngh = get_row( v );
	    if( tr::is_bitwise_and_zero( P, v_ngh ) ) {
		row_type v_only = I.get_mask();
		if( tr::is_bitwise_and_zero( X, v_ngh ) ) {
		    row_type Rv = tr::bitwise_or( R, v_only );
		    EE( bitset<Bits>( Rv ), depth+1 );
		}

		// Move v from P to X. All other elements of P will be
		// processed as if they are right siblings of v.
		P = tr::bitwise_andnot( v_only, P );
		X = tr::bitwise_or( v_only, X );
	    }
	}

	// Any vertex v in P where P is subset of N(v) is connected to all
	// remaining candidates and must be a member of any maximal clique.
	// It is as if we select these vertices as pivot, which makes them
	// the only vertex at this level, and the remainder of P is processed
	// at the next recursion level.
	b = bitset<Bits>( P ); // P may have changed
	for( auto I=b.begin(), E=b.end(); I != E; ++I ) {
	    VID v = *I;

	    row_type v_ngh = get_row( v );
	    row_type diff = tr::bitwise_xor( P, v_ngh );
	    if( tr::is_bitwise_and_zero( P, diff ) ) {
		row_type v_only = I.get_mask();
		// P = tr::bitwise_and( P, v_ngh ) redundant due to condition
		X = tr::bitwise_and( X, v_ngh );
		R = tr::bitwise_or( R, v_only );
		++depth;

		// Remove v
		P = tr::bitwise_andnot( v_only, P );
	    }
	}
    }

    template<typename Enumerate>
    VID mce_get_pivot_and_filter(
	Enumerate && EE, row_type & R, row_type & P,
	row_type & X, unsigned int & depth ) {
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
	    row_type v_only = I.get_mask();
	    row_type v_ngh = get_row( v );
	    row_type pv_ins = tr::bitwise_and( P, v_ngh );

	    bool in_P = !tr::is_bitwise_and_zero( P, v_only );

	    // Any vertex v in P where P intersect N(v) is empty can be
	    // enumerated and removed from P. It is as if we select this
	    // vertex first, then consider the remainder of P as right siblings.
	    if( in_P && tr::is_zero( pv_ins ) ) {
		if( tr::is_bitwise_and_zero( X, v_ngh ) ) {
		    row_type Rv = tr::bitwise_or( R, v_only );
		    EE( bitset<Bits>( Rv ), depth+1 );
		}

		// Move v from P to X. All other elements of P will be
		// processed as if they are right siblings of v.
		P = tr::bitwise_andnot( v_only, P );
		X = tr::bitwise_or( v_only, X );

		continue; // vertex has been processed
	    }

#if !ABLATION_DENSE_EXCEED && !ABLATION_PDEG
	    if( (VID)m_degree[v] < p_ins ) // skip if cannot be best
		continue;
#endif

	    // Any vertex v in P where P is subset of N(v) is connected to all
	    // remaining candidates and must be a member of any maximal clique.
	    // It is as if we select these vertices as pivot, which makes them
	    // the only vertex at this level, and the remainder of P is
	    // processed at the next recursion level.
	    row_type diff = tr::bitwise_xor( P, v_ngh );
	    if( in_P && tr::is_bitwise_and_zero( P, diff ) ) {
		row_type v_only = I.get_mask();
		// P = tr::bitwise_and( P, v_ngh ) redundant due to condition
		X = tr::bitwise_and( X, v_ngh );
		R = tr::bitwise_or( R, v_only );
		++depth;

		// Remove v
		P = tr::bitwise_andnot( v_only, P );

		continue; // vertex has been processed
	    }

	    // Now search for pivot
#if FURTHER_OPTIMIZATION
	    // Only compute allpopcnt if pv_ins is not subset of p_ins
	    if( !tr::is_zero( tr::bitwise_andnot( p_ins, pv_ins ) ) ) {
#endif
	    VID ins = get_size( pv_ins );
	    if( ins > p_ins ) {
		p_best = v;
		p_ins = ins;
	    }
#if FURTHER_OPTIMIZATION
	    }
#endif
	}
	return p_best;
    }

    // Could potentially vectorise small matrices by placing
    // one 32/64-bit row in a vector lane and performing a vectorised
    // popcount per lane. Could evaluate doing popcounts on all lanes,
    // or gathering only active lanes. The latter probably most promising
    // in AVX512
    VID mce_get_pivot( row_type P, row_type X ) {
	row_type r = tr::bitwise_or( P, X );
	bitset<Bits> b( r );

	auto I = b.begin();
	VID p_best = *I;

#if !ABLATION_DENSE_EXCEED
	// Avoid complexities if there is not much choice
	if( get_size( P ) <= 3 )
	    return p_best;
#endif

	++I;
	row_type p_row = tr::bitwise_and( P, get_row( p_best ) );
	VID p_ins = get_size( p_row );
	
	for( auto E=b.end(); I != E; ++I ) {
	    VID v = *I;
#if !ABLATION_DENSE_EXCEED && !ABLATION_PDEG
	    if( (VID)m_degree[v] < p_ins ) // skip if cannot be best
		continue;
#endif
	    row_type v_ngh = get_row( v );
	    row_type pv_ins = tr::bitwise_and( P, v_ngh );
#if FURTHER_OPTIMIZATION
	    // Only compute allpopcnt if pv_ins is not subset of p_ins
	    if( !tr::is_zero( tr::bitwise_andnot( p_row, pv_ins ) ) ) {
#endif
	    VID ins = get_size( pv_ins );
	    if( ins > p_ins ) {
		p_best = v;
		p_ins = ins;
		p_row = pv_ins;
		// if( p_ins >= P_size ) // cannot improve further
		// break;
	    }
#if FURTHER_OPTIMIZATION
	    }
#endif
	}
	// assert( p_best < m_n );
	return p_best;
    }

    VID mc_get_pivot( row_type r ) {
	bitset<Bits> b( r );

	auto I = b.begin();
	VID p_best = *I;

	// Avoid complexities if there is not much choice
	if( get_size( r ) <= 3 )
	    return p_best;

	++I;
	row_type p_row = tr::bitwise_and( r, get_row( p_best ) );
	VID p_ins = get_size( p_row );
	
	for( auto E=b.end(); I != E; ++I ) {
	    VID v = *I;
	    row_type v_ngh = get_row( v );
	    row_type pv_ins = tr::bitwise_and( r, v_ngh );
	    // Only compute allpopcnt if pv_ins is not subset of p_row
	    if( !tr::is_zero( tr::bitwise_andnot( p_row, pv_ins ) ) ) {
		VID ins = get_size( pv_ins );
		if( ins > p_ins ) {
		    p_best = v;
		    p_ins = ins;
		    p_row = pv_ins;
		}
	    }
	}
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

    /**
     * Greedy graph matching
     *
     * @return bitmask showing vertices that have been matched
     */
    row_type graph_matching_outsiders( row_type eligible ) const {
	// State: 0 if matched, 1 if unmatched
	// Reason: feeds into get_inv_row_unc()
	row_type state = eligible;

	for( VID u=0; u < m_n; ++u ) {
	    row_type u_row = create_row( u );
	    if( tr::is_bitwise_and_zero( u_row, state ) ) // already matched
		continue;

	    // Iterate over neighbours that have not yet been matched
	    row_type u_ngh = get_inv_row_unc( u, u_row, state );
	    bitset<Bits> bx( u_ngh );
	    auto I = bx.begin();
	    if( I != bx.end() ) { // match one edge
		// It is implied that v's state is 1, i.e., unmatched
		VID v = *I;
		row_type v_row = I.get_mask();
		assert( !tr::is_bitwise_and_zero( v_row, state ) );

		// Set u and v as matched
		state = tr::bitwise_andnot( v_row, state );
		state = tr::bitwise_andnot( u_row, state );
	    }
	}

	return tr::bitwise_andnot( state, m_allv_mask );
    }

    /**
     * Greedy auxiliary graph matching
     *
     * @a 
     * @return vector of edges in matching, with vertices encoded in row_type
     */
    std::vector<std::pair<row_type,row_type>>
    auxiliary_graph_matching( row_type eligible,
			      row_type M1,
			      row_type & state_out )
	const {
	row_type state = tr::setzero(); // set of matched vertices
	std::vector<std::pair<row_type,row_type>> match; // edge list

	// Iterate over all vertices that have not been matched yet,
	// i.e. all vertices in O, and find one matching edge for them
	// from among their neighbours.
	bitset<Bits> bu( tr::bitwise_andnot( M1, eligible ) );
	bool set_u = false;
	for( auto Iu = bu.begin(), Eu = bu.end(); Iu != Eu; ++Iu ) {
	    VID u = *Iu;
	    row_type u_row = Iu.get_mask();

	    // u already matched
	    if( !tr::is_bitwise_and_zero( u_row, state ) )
		continue;

	    row_type not_matched = tr::bitwise_andnot( state, eligible );
	    if( tr::is_zero( not_matched ) ) // no more vertices to select
		break;
	    bitset<Bits> bv( get_inv_row_unc( u, u_row, not_matched ) );
	    auto Iv = bv.begin();
	    if( Iv != bv.end() ) { // Match one edge
		VID v = *Iv;
		row_type v_row = Iv.get_mask();
		row_type uv_row = tr::bitwise_or( u_row, v_row );
		match.push_back( { u_row, v_row } );
		state = tr::bitwise_or( uv_row, state );
	    }
	}

	state_out = state;
	return match;
    }

    /**
     * Compute crown kernel for vertex cover
     *
     * @return pair of sets (I,H) such that I is an independent set and
     *         H is the base of the crown.
     */
    std::pair<row_type,row_type>
    vck_crown_kernel( row_type eligible ) const {
	// Primary and auxiliary matchings
	// Let O = { M1 == 0 }
	row_type M1 = graph_matching_outsiders( eligible );
	row_type M2;
	std::vector<std::pair<row_type,row_type>> Ma =
	    auxiliary_graph_matching( eligible, M1, M2 );

	row_type crown_I = tr::setzero();
	row_type crown_H = tr::setzero();

	// Check if every vertex in N(O) is matched by M2
	// Alternatively, could check size of N(O) equals size of I equals
	// size of matching M2
	bool all_ngh_match_M2 = true;
	crown_I = tr::bitwise_andnot( M1, eligible );
	bitset<Bits> bu( crown_I );
	for( auto Iu=bu.begin(), Eu=bu.end();
	     Iu != Eu && all_ngh_match_M2;
	     ++Iu ) {
	    VID u = *Iu;
	    row_type u_row = Iu.get_mask();

	    // Iterate over neighbours of u
	    row_type u_ngh = get_inv_row_unc( u, u_row, eligible );
	    bitset<Bits> bx( get_inv_row_unc( u, u_row, eligible ) );
	    for( auto I = bx.begin(), E = bx.end(); I != E; ++I ) {
		VID v = *I;
		row_type v_row = I.get_mask();
		if( tr::is_bitwise_and_zero( v_row, M2 ) ) { // v not matched
		    all_ngh_match_M2 = false;
		    break;
		}
	    }
	    crown_H = tr::bitwise_or( u_ngh, crown_H );
	}

	if( all_ngh_match_M2 ) {
	    // Intersection of I and H should be empty
	    assert( tr::is_bitwise_and_zero( crown_I, crown_H ) );

	    return { crown_I, crown_H };
	}

	// I0 = { M2 == 0 }: vertices in O (~M1) that are unmatched by M2
	crown_I = tr::bitwise_andnot( M2, crown_I );
	bool change = true;
	while( change ) {
	    // Reset H
	    crown_H = tr::setzero();

	    // Hn = N(In)
	    bitset<Bits> bx( crown_I );
	    for( auto I = bx.begin(), E = bx.end(); I != E; ++I ) {
		VID v = *I;
		row_type v_row = I.get_mask();

		// crown_I intersect neighbours of v should be empty as
		// crown_I is an independent set
		row_type v_ngh = get_inv_row_unc( v, v_row, eligible );
		assert( tr::is_bitwise_and_zero( v_ngh, crown_I ) );
		crown_H = tr::bitwise_or( v_ngh, crown_H );
	    }

	    // In+1 = In union N_M2(Hn)
	    change = false;
	    for( auto I=Ma.begin(), E=Ma.end(); I != E; ++I ) {
		auto [ u_row, v_row ] = *I; // row_types !

		if( !tr::is_bitwise_and_zero( u_row, crown_H )
		    && tr::is_bitwise_and_zero( v_row, crown_I ) ) {
		    // u in Hn, v not in In
		    change = true;
		    crown_I = tr::bitwise_or( v_row, crown_I );
		} else if( !tr::is_bitwise_and_zero( v_row, crown_H )
			   && tr::is_bitwise_and_zero( u_row, crown_I ) ) {
		    // v in Hn, u not in In
		    change = true;
		    crown_I = tr::bitwise_or( u_row, crown_I );
		}
	    }
	    // TODO: always include in I and avoid 1/2 is_zero, check for
	    // change by comparing pre/post crown_I
	}

	// Intersection of I and H should be empty
	assert( tr::is_bitwise_and_zero( crown_I, crown_H ) );

	return { crown_I, crown_H };
    }

    // A greedy algorithm guaranteeing a 2-approximation
    void greedy_cover( VID & size, row_type & cover_out ) const {
	row_type cover = tr::setzero();
	for( VID v=0; v < m_n; ++v ) {
	    row_type v_row = create_row( v );

	    // v already covered
	    if( !tr::is_bitwise_and_zero( v_row, cover ) )
		continue;

	    row_type v_ngh = get_inv_row_unc( v, v_row, m_allv_mask );
	    bitset<Bits> bu( v_ngh );
	    for( auto Iu=bu.begin(), Eu=bu.end(); Iu != Eu; ++Iu ) {
		VID u = *Iu;
		row_type u_row = Iu.get_mask();

		// Is u covered?
		if( !tr::is_bitwise_and_zero( u_row, cover ) )
		    continue;

		// Add (u,v) to cover
		cover = tr::bitwise_or( u_row, cover );
		cover = tr::bitwise_or( v_row, cover );
		break;
	    }
	}

	size = get_size( cover );
	cover_out = cover;

	assert( ( size % 2 ) == 0 );
    }

    template<bool exists, bool co>
    int vck_crown( VID k, VID c, row_type eligible,
		   VID & best_size, row_type & best_cover ) {
	static_assert( co, "not yet adapted for co == false case" );

	// Compute crown kernel: (I=1,H=2)
	auto [ crown_I, crown_H ] = vck_crown_kernel( eligible );

	// Failure to identify crown
	if( tr::is_zero( crown_I ) || tr::is_zero( crown_H ) )
	    return -1;

	VID h_size = get_size( crown_H );
	if( h_size > k )
	    return 0; // no VC of size k or less

	// All vertices in H are included in cover
	row_type tmp_best_cover = tr::setzero();

	// All vertices in I and H are no longer selectable
	row_type crown_IH = tr::bitwise_or( crown_H, crown_I );
	row_type tmp_eligible = tr::bitwise_andnot( crown_IH, eligible );

	// Find a cover for the remaining vertices
	VID tmp_best_size = 0;
	bool rec = vck_iterate<exists,co>( k - h_size, c, tmp_eligible,
					   tmp_best_size, tmp_best_cover );

	if( rec ) {
	    best_size += tmp_best_size + h_size;
	    best_cover =
		tr::bitwise_or( tmp_best_cover,
				tr::bitwise_or( crown_H, best_cover ) );
	}

	return rec ? 1 : 0;
    }

    template<bool exists, bool co>
    bool vck_buss( VID k, VID c, row_type eligible,
		   VID & best_size, row_type & best_cover ) {
	// Set U of vertices of degree higher than k
	VID u_size = 0;
	row_type u_set = tr::setzero();
	for( VID r=0; r < m_n; ++r ) {
	    row_type rr = create_row( r );
	    if( get_size( co_get_row<co>( r, rr, eligible ) ) > k ) {
		++u_size;
		u_set = tr::bitwise_or( u_set, rr );
	    }
	}

	// std::cout << "buss k=" << k << " u_size=" << u_size << "\n";

	// If |U| > k, then there exists no cover of size k
	if( u_size > k )
	    return false;

	assert( u_size > 0 );

	// Calculate eligible vertices
	row_type gp_eligible = tr::bitwise_andnot( u_set, eligible );
	EID m = std::get<2>( co_analyse<co>( gp_eligible ) );

	// If G has more than k(k-|U|) edges, reject
	if( m > k * ( k - u_size ) )
	    return false;

	// Find a cover for the remaining vertices
	// All vertices with degree > k must be included in the cover
	VID gp_best_size = best_size + u_size;
	row_type gp_best_cover = tr::bitwise_or( best_cover, u_set );
	// row_type gp_eligible = tr::bitwise_andnot( u_set, eligible );
	bool rec = vck_iterate<exists,co>( k - u_size, c, gp_eligible,
					   gp_best_size, gp_best_cover );

	if( rec ) {
	    assert( gp_best_size - best_size <= k );

	    best_size = gp_best_size;
	    best_cover = gp_best_cover;
	}
	// std::cout << "buss k=" << k << " u_size=" << u_size
		  // << " return rec=" << rec << " best=" << best_size
		  // << "\n";

	return rec;
    }

    // For path or cycle
    template<bool co>
    void
    vck_trace_path( std::vector<char> & visited,
		    row_type eligible,
		    VID & best_size,
		    row_type & best_cover,
		    VID cur,
		    VID nxt,
		    bool incl ) {
	if( visited[nxt] )
	    return;

	visited[nxt] = true;

	row_type nxt_row = create_row( nxt );
	if( incl ) {
	    best_size++;
	    best_cover = tr::bitwise_or( best_cover, nxt_row );
	    // std::cout << "tracepath set " << nxt << "\n";
	}

	// Done if nxt is degree-1 vertex
	row_type nadj = co_get_row<co>( nxt, nxt_row, eligible );
	VID ndeg = get_size( nadj );
	if( ndeg == 2 ) {
	    // Remove cur from neighbours of nxt
	    row_type cur_row = create_row( cur );
	    row_type x = tr::bitwise_andnot( cur_row, nadj );
	    VID ngh = target::alltzcnt<sVID,type,VL>::compute( x );

	    vck_trace_path<co>( visited, eligible,
				best_size, best_cover, nxt, ngh, !incl );
	}
    }

    template<bool co>
    bool vck_poly( VID k, row_type eligible,
		   VID & best_size, row_type & best_cover ) {
	VID n = numVertices();

	std::vector<char> visited( n, false ); // TODO: row_type

	VID poly_best_size = 0;
	row_type poly_best_cover = tr::setzero();

	// Find paths
	bitset<Bits> bx( eligible );
	for( auto I = bx.begin(), E = bx.end(); I != E; ++I ) {
	    VID v = *I;
	    row_type v_row = I.get_mask();
	    row_type adj = co_get_row_unc<co>( v, v_row, eligible );
	    if( tr::is_zero( adj ) )
		continue;
	    VID deg = get_size( adj );
	    assert( deg <= 2 );
	    if( deg == 1 && !visited[v] ) {
		visited[v] = true;
		VID ngh = target::alltzcnt<sVID,type,VL>::compute( adj );
		vck_trace_path<co>( visited, eligible,
				    poly_best_size, poly_best_cover,
				    v, ngh, true );
	    }
	}
    
	// Find cycles (uses same auxiliary as paths)
	for( auto I = bx.begin(), E = bx.end(); I != E; ++I ) {
	    VID v = *I;
	    row_type v_row = I.get_mask();
	    row_type adj = co_get_row_unc<co>( v, v_row, eligible );
	    if( tr::is_zero( adj ) )
		continue;
	    VID deg = get_size( adj );
	    assert( deg <= 2 );
	    if( deg == 2 && !visited[v] ) {
		visited[v] = true;
	
		// Mark
		poly_best_size++;
		poly_best_cover = tr::bitwise_or( poly_best_cover, v_row );
		// std::cout << "tracepath cycle set " << v << "\n";

		VID ngh = target::alltzcnt<sVID,type,VL>::compute( adj );
		vck_trace_path<co>( visited, eligible,
				    poly_best_size, poly_best_cover,
				    v, ngh, false );
	    }
	}

	if( poly_best_size <= k ) {
	    best_size += poly_best_size;
	    best_cover = tr::bitwise_or( best_cover, poly_best_cover );
	    return true;
	} else
	    return false;
    }
    
    template<bool exists, bool co>
    bool vck_iterate( VID k, VID c, row_type eligible,
		      VID & best_size, row_type & best_cover ) {
	VID n = m_n;
	auto [ max_v, max_deg, m ] = co_analyse<co>( eligible );
	if( k == 0 )
	    return m == 0;

	if( max_deg <= 2 ) {
	    VID sz = 0;
	    row_type c = tr::setzero();
	    bool ret = vck_poly<co>( k, eligible, sz, c );
	    assert( !ret || sz <= k );
	    if( ret ) {
		best_size += sz;
		best_cover = tr::bitwise_or( best_cover, c );
	    }
	    return ret;
	}	

	// Try crown kernel reduction
#if 0 // crown cover not correct
	if constexpr ( co == true ) { // co == false NYI
	    int ret = vck_crown<exists,co>( k, c, eligible, best_size, best_cover );
	    if( ret >= 0 )
		return (bool)ret;
	}
#endif

	if( m/2 > c * k * k && max_deg > k ) {
	    // replace by Buss kernel
	    return vck_buss<exists,co>( k, c, eligible, best_size, best_cover );
	}

	// Must have a vertex with degree >= 3
	assert( max_deg >= 3 );

	// Create two subproblems ; branch on max_v
	VID i_best_size = 0;
	row_type i_best_cover = tr::setzero();
	VID x_best_size = 0;
	row_type x_best_cover = tr::setzero();

	// In case v is excluded, erase v and all its neighbours
	// Make sure our neighbours are sorted. Iterators remain valid after
	// erasing incident edges.
	row_type max_v_row = create_row( max_v );
	row_type rm = co_get_row<co>( max_v, max_v_row, eligible );
	row_type rmv = tr::bitwise_or( rm, max_v_row );
	row_type x_eligible = tr::bitwise_andnot( rmv, eligible );

	VID x_k = std::min( n-1-max_deg, k-max_deg );
	bool x_ok = false;
	if( k >= max_deg )
	    x_ok = vck_iterate<exists,co>( x_k, c, x_eligible, x_best_size, x_best_cover );

	// With exists queries, it suffices to answer positively
	if constexpr ( exists ) {
	    if( x_ok ) {
		assert( x_best_size <= x_k );
		// add max_v, neighbours, and anything in x_best_cover;
		best_size += max_deg + x_best_size;
		best_cover = tr::bitwise_or(
		    best_cover, tr::bitwise_or( x_best_cover, rm ) );
		assert( best_size <= k );
		return true;
	    }
	}

	// In case v is included, erase only v
	row_type i_eligible = tr::bitwise_andnot( max_v_row, eligible );
	VID i_k = x_ok ? std::min( max_deg+x_best_size, k-1 ) : k-1;
	bool i_ok = vck_iterate<exists,co>( i_k, c, i_eligible, i_best_size, i_best_cover );

	VID pre = best_size;
	if( i_ok && ( !x_ok || i_best_size+1 < x_best_size+max_deg ) ) {
	    assert( i_best_size <= i_k );
	    assert( 1 + i_best_size <= k );
	    // add max_v, and anything in i_best_cover;
	    best_size += 1 + i_best_size;
	    best_cover = tr::bitwise_or(
		best_cover, tr::bitwise_or( i_best_cover, max_v_row ) );
	} else if( x_ok ) {
	    assert( max_deg + x_best_size <= k );
	    // add max_v, neighbours, and anything in x_best_cover;
	    best_size += max_deg + x_best_size;
	    best_cover = tr::bitwise_or(
		best_cover, tr::bitwise_or( x_best_cover, rm ) );
	}

	assert( !( i_ok || x_ok ) || best_size - pre <= k );
	return i_ok || x_ok;
    }

    DenseMatrix<Bits,VID,EID> filter_vertices( row_type rm ) {
	DenseMatrix<Bits,VID,EID> G( *this ); // create of same size

	for( VID r=0; r < m_n; ++r ) {
	    row_type rr = create_row( r );
	    if( tr::is_bitwise_and_zero( rr, rm ) ) {
		G.set_row( r, tr::bitwise_andnot( rm, get_row( r ) ) );
	    } else {
		G.set_row( r, tr::setzero() );
	    }
	}

	return G;
    }

    DenseMatrix<Bits,VID,EID> filter_vertex( VID rmv ) {
	DenseMatrix<Bits,VID,EID> G( *this ); // create of same size

	row_type rm = create_row( rmv );

	for( VID r=0; r < m_n; ++r )
	    G.set_row( r, tr::bitwise_andnot( rm, get_row( r ) ) );

	G.set_row( rmv, tr::setzero() );

	return G;
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

    row_type get_row( VID v ) const {
	return tr::load( &m_matrix[m_words * v] );
    }
    row_type get_row( VID v, row_type eligible ) const {
	const row_type r = tr::load( &m_matrix[m_words * v] );
	return tr::bitwise_and( eligible, r );
    }
    row_type get_inv_row_all( VID v, row_type v_single ) const {
	const row_type r = tr::load( &m_matrix[m_words * v] );
	row_type rc = tr::bitwise_or( v_single, r );
	return tr::bitwise_andnot( rc, m_allv_mask );
    }
    template<bool co>
    row_type co_get_row( VID v, row_type v_single, row_type eligible ) const {
	if constexpr ( co )
	    return get_inv_row( v, v_single, eligible );
	else
	    return get_row( v, eligible );
    }
    row_type get_inv_row( VID v, row_type v_single, row_type eligible ) const {
	if( tr::is_bitwise_and_zero( v_single, eligible ) )
	    return tr::setzero();
	const row_type r = tr::load( &m_matrix[m_words * v] );
	row_type rc = tr::bitwise_or( v_single, r );
	return tr::bitwise_andnot( rc, eligible );
    }
    template<bool co>
    row_type co_get_row_unc( VID v, row_type v_single, row_type eligible )
	const {
	if constexpr ( co )
	    return get_inv_row_unc( v, v_single, eligible );
	else
	    return get_row( v, eligible );
    }
    row_type get_inv_row_unc( VID v, row_type v_single, row_type eligible )
	const {
	const row_type r = tr::load( &m_matrix[m_words * v] );
	row_type rc = tr::bitwise_or( v_single, r );
	return tr::bitwise_andnot( rc, eligible );
    }
    void set_row( VID v, row_type r ) {
	return tr::store( &m_matrix[m_words * v], r );
    }

    std::pair<VID,VID> get_max_degree( row_type eligible ) const {
	bitset<Bits> bx( eligible );
	VID max_v = 0;
	VID max_deg = 0;
	for( auto I = bx.begin(), E = bx.end(); I != E; ++I ) {
	    VID v = *I;
	    row_type rv = I.get_mask();
	    row_type adj = get_inv_row( v, rv, eligible );
	    if( !tr::is_zero( adj ) ) {
		VID deg = get_size( adj );
		if( deg > max_deg ) {
		    max_deg = deg;
		    max_v = v;
		}
	    }
	}
	return { max_v, max_deg };
    }

public:
    EID get_num_edges() const { return m_m; }
    VID get_max_degree() const { return m_d_max; }
    EID calculate_num_edges() const {
	return calculate_num_edges( m_allv_mask );
    }
    VID get_num_vertices() const { return m_n; }
    VID get_degree( VID v ) const { return get_size( get_row( v ) ); }

private:
    // Returns max-degree-vertex, its degree, number of edges in graph
    template<bool co>
    std::tuple<VID,VID,EID> co_analyse( row_type eligible ) const {
	EID m = 0;
	VID max_v = 0;
	VID max_deg = 0;
	bitset<Bits> bx( eligible );
	for( auto I = bx.begin(), E = bx.end(); I != E; ++I ) {
	    VID v = *I;
	    row_type vr = I.get_mask();
	    row_type adj = co_get_row<co>( v, vr, eligible );
	    if( !tr::is_zero( adj ) ) { // cheap check to filter out zero-degree
		VID deg = get_size( adj );
		m += deg;
		if( deg > max_deg ) {
		    max_deg = deg;
		    max_v = v;
		}
	    }
	}
	return { max_v, max_deg, m };
    }
    
    EID calculate_num_edges( row_type eligible ) const {
	EID m = 0;
	bitset<Bits> bx( eligible );
	for( auto I = bx.begin(), E = bx.end(); I != E; ++I ) {
	    VID v = *I;
	    row_type vr = I.get_mask();
	    row_type adj = get_inv_row( v, vr, eligible );
	    if( !tr::is_zero( adj ) )
		m += get_size( adj );
	}
	return m;
    }

    static row_type create_row( VID v ) {
	return tr::setglobaloneval( v );
    }

    static row_type get_himask( VID v ) {
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
    const row_type m_allv_mask;
    VID m_n;
    static constexpr unsigned m_words = VL;
    VID m_d_max;
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
