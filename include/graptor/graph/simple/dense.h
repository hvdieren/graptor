// -*- C++ -*-
#ifndef GRAPHGRIND_GRAPH_SIMPLE_DENSE_H
#define GRAPHGRIND_GRAPH_SIMPLE_DENSE_H

#include <type_traits>
#include <algorithm>

#include "graptor/graph/GraphCSx.h"
#include "graptor/graph/simple/cutout.h"
#include "graptor/graph/simple/hadjt.h"
#include "graptor/target/vector.h"
#include "graptor/container/intersect.h"

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

    void push_back( const lVID * p ) {
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

template<typename BitMaskTy, typename lVID, bool aligned, bool two_sided>
struct bitmask_lhs_sorted_output_iterator {
    using bitmask_type = BitMaskTy;
    static constexpr size_t word_size = 8 * sizeof( bitmask_type );
    
    bitmask_lhs_sorted_output_iterator(
	bitmask_type * bitmask,
	const lVID * start,
	lVID deg_start_pos,
	lVID start_pos,
	lVID end_pos = std::numeric_limits<lVID>::max() )
	: m_bitmask( bitmask ), m_start( start ),
	  m_deg( 0 ), m_deg_start_pos( deg_start_pos ),
	  m_start_pos( start_pos ), m_end_pos( end_pos ) { }

    const bitmask_lhs_sorted_output_iterator &
    operator = ( const bitmask_lhs_sorted_output_iterator & it ) {
	// hmmm....
	m_deg = it.m_deg;
	return *this;
    }

    void push_back( const lVID * p ) {
	lVID v = p - m_start;
	lVID vchk = v;
	bool accept = v >= m_start_pos;
	if constexpr ( two_sided ) {
	    accept = accept && v < m_end_pos;
	    v -= m_start_pos;
	}

	// no X-X edges / slice of matrix
	if( accept ) {
	    m_bitmask[v/word_size]
		|= bitmask_type(1) << ( v & ( word_size-1 ) );
	    if( vchk >= m_deg_start_pos )
		m_deg++;
	}
    }

    template<unsigned VL>
    void push_back( typename vector_type_traits_vl<lVID,VL>::mask_type m,
		    typename vector_type_traits_vl<lVID,VL>::type gv,
		    const lVID * base ) {
	using tr = vector_type_traits_vl<lVID,VL>;
	using mtr = typename tr::mask_traits;
	using type = typename tr::type;
	using mask_type = typename tr::mask_type;

	lVID v = base - m_start;
	type vsp = tr::set1( m_start_pos );
	type vns = tr::set1inc( v );
	mask_type mm = tr::cmpge( m, vns, vsp, target::mt_mask() );
	if constexpr ( two_sided ) {
	    type vep = tr::set1( m_end_pos );
	    mm = tr::cmplt( mm, vns, vep, target::mt_mask() );
	    // vns = tr::sub( vns, vsp ); -- vns unused further down
	}
	mask_type * b = reinterpret_cast<mask_type *>( m_bitmask );

	// What if only part of vector is acceptable (blocked + prestudy)???
	bool perform = true;
	if constexpr ( two_sided ) {
	    perform = ( v >= m_start_pos && v < m_end_pos );
	    v -= m_start_pos;
	}
	if( perform ) {
	    if constexpr ( aligned ) {
		b[v/VL] = mm;
	    } else {
		size_t word = v / VL;
		size_t off = v % VL;
		b[word/VL] |= mtr::slli( mm, off );
		b[1+word/VL] |= mtr::srli( mm, VL - off );
	    }
	    type vdsp = tr::set1( m_deg_start_pos );
	    mm = tr::cmpge( mm, vns, vdsp, target::mt_mask() );
	    m_deg += mtr::popcnt( mm );
	}
    }

    VID get_degree() const { return m_deg; }

private:
    bitmask_type * m_bitmask;
    const lVID * m_start;
    lVID m_deg;
    const lVID m_deg_start_pos;
    const lVID m_start_pos;
    const lVID m_end_pos;
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
	m_matrix = m_matrix_alc = new type[m_words * ns + 64];
	intptr_t p = reinterpret_cast<intptr_t>( m_matrix );
	if( p & 63 ) // 63 = 512 bits / 8 bits per byte - 1
	    m_matrix = &m_matrix[64 - (p&63)/sizeof(type)];
	static_assert( Bits <= 512, "AVX512 requires 64-byte alignment" );

	// Place edges
	VID ni = 0;
	m_m = 0;
	for( VID su=0; su < ns; ++su ) {
	    VID u = s2g[su]; // Note: m_s2g not initialised in this variant
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
	    m_degree[su] = deg;
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
	    VID u = s2g[su]; // Note: m_s2g not initialised in this variant

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
	    m_degree[su] = deg;
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
	    VID u = s2g[su]; // Note: m_s2g not initialised in this variant

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
	    m_degree[su] = deg;
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
	    VID u = s2g[su]; // Note: m_s2g not initialised in this variant

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
	    m_degree[su] = deg;
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
	std::fill( m_degree, m_degree+ns, DID(0) );

	size_t e = 1;
	for( VID i=0; i < n-1; ++i ) {
	    VID deg = 0;
	    for( VID j=0; j < i; ++j ) {
		if( ( edges & e ) != 0 ) {
		    set( i, j );
		    set( j, i );
		    m_m += 2;
		    m_degree[i]++;
		    m_degree[j]++;
		}
		e <<= 1;
	    }
	}
    }
#if 0
    DenseMatrix( const ::GraphCSx & G, VID v,
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
	m_matrix = m_matrix_alc = new type[m_words * ns + 64];
	intptr_t p = reinterpret_cast<intptr_t>( m_matrix );
	if( p & 63 ) // 63 = 512 bits / 8 bits per byte - 1
	    m_matrix = &m_matrix[64 - (p&63)/sizeof(type)];
	static_assert( Bits <= 512, "AVX512 requires 64-byte alignment" );

	// Place edges
	VID ni = 0;
	m_m = 0;
	for( VID su=0; su < ns; ++su ) {
	    VID u = m_s2g[su];
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
			if( sw >= m_start_pos || su >= m_start_pos ) {
			    row_u = tr::bitwise_or( row_u, create_row( sw ) );
			    ++deg;
			}
		    }
		    ++p;
		    ++q;
		} else if( *p < *q )
		    ++p;
		else
		    ++q;
	    }

	    tr::store( &m_matrix[VL * su], row_u );
	    m_degree[su] = deg;
	    m_m += deg;
	}

	m_n = ns;

	delete[] n2s;
    }
#endif
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
	// std::copy( XP, XP+ce, m_s2g );

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
	    VID u = XP[su]; // m_s2g[su]; // or XP[su]
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
		E( bitset<Bits>( R ), 1 );
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

#if 0
    const VID * get_s2g() const { return &m_s2g[0]; }
#endif

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
		EE( bitset<Bits>( R ), depth );
	    return;
	}

	VID pivot = mce_get_pivot( P, X );
	row_type pivot_ngh = get_row( pivot );
	row_type x = tr::bitwise_andnot( pivot_ngh, P );
	bitset<Bits> bx( x );
	for( auto I = bx.begin(), E = bx.end(); I != E; ++I ) {
	    VID u = *I;
	    row_type u_only = tr::setglobaloneval( u );
	    row_type u_ngh = get_row( u );
	    row_type Pv = tr::bitwise_and( P, u_ngh );
	    row_type Xv = tr::bitwise_and( X, u_ngh );
	    row_type Rv = tr::bitwise_or( R, u_only );
	    P = tr::bitwise_andnot( u_only, P ); // P == x w/o pivoting
	    X = tr::bitwise_or( u_only, X );
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

#if 0
    VID m_s2g[Bits];
#endif
    DID m_degree[Bits]; //!< only left-most columns (P part)
};

} // namespace graph

} // namespace graptor

#endif // GRAPHGRIND_GRAPH_SIMPLE_DENSE_H
