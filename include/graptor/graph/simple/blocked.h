// -*- C++ -*-
#ifndef GRAPHGRIND_GRAPH_SIMPLE_BLOCKED_H
#define GRAPHGRIND_GRAPH_SIMPLE_BLOCKED_H

#include <type_traits>
#include <algorithm>

#include "graptor/graph/GraphCSx.h"
#include "graptor/graph/simple/cutout.h"
#include "graptor/graph/simple/hadjt.h"
#include "graptor/graph/simple/dense.h" // support iterators
#include "graptor/target/vector.h"

namespace graptor {

namespace graph {

// TODO: matrix construction often takes longer than solving the sub-problem
// Rectangular binary matrix
template<unsigned Bits, typename sVID, typename sEID>
class BinaryMatrix {
    using type = std::conditional_t<Bits<=32,uint32_t,uint64_t>;
    static constexpr unsigned bits_per_lane = sizeof(type)*8;
    static constexpr unsigned short VL = Bits / bits_per_lane;
public:
    using tr = vector_type_traits_vl<type,VL>;
    using row_type = typename tr::type;

    static_assert( VL * bits_per_lane == Bits );

public:
    static constexpr size_t MAX_COL_VERTICES = bits_per_lane * VL;

public:
    BinaryMatrix()
	: m_rows( 0 ), m_cols( 0 ), m_row_start( 0 ), m_col_start( 0 ),
	  m_matrix_alc( nullptr ) { }
    // The neighbours are split up in ineligible neighbours (initial X set)
    // and eligible neighbours (initial P set). They are already sorted
    // by decreasing coreness in the gID array. Their position in this array
    // reflects their relative index in this matrix. Columns are vertices in
    // gID from cs to ce; rows are gID elements rs to re.
    template<typename DID, typename AddDegree,
	     typename gVID = VID, typename gEID = EID>
    BinaryMatrix( const ::GraphCSx & G,
		  gVID rs, gVID re,
		  gVID cs, gVID ce,
		  const sVID * const neighbours, // sorted order of gVID
		  const gVID * const gID, // neighbours in degeneracy order
		  const sVID * const n2s, // map idx in neighbours to idx in gID
		  DID * m_degree,
		  AddDegree && )
	: m_row_start( rs ), m_rows( re-rs ),
	  m_col_start( cs ), m_cols( ce-cs ) {
	assert( m_cols <= MAX_COL_VERTICES
		&& "Cap on number of vertices that fit in bitmask" );
	assert( ( m_cols + bits_per_lane - 1 ) / bits_per_lane <= VL );
	gVID n = G.numVertices();
	gEID m = G.numEdges();
	const gEID * const gindex = G.getIndex();
	const gVID * const gedges = G.getEdges();

	m_matrix = m_matrix_alc = new type[VL * m_rows + 64];
	intptr_t p = reinterpret_cast<intptr_t>( m_matrix );
	if( p & 63 ) // 63 = 512 bits / 8 bits per byte - 1
	    m_matrix = &m_matrix[64 - (p&63)/sizeof(type)];
	static_assert( Bits <= 512, "AVX512 requires 64-byte alignment" );

	// Place edges
	VID ni = 0;
	m_m = 0;
	for( VID r=rs; r < re; ++r ) {
	    VID u = gID[r];
	    VID deg = 0;

	    row_type row_u = tr::setzero();

	    const gVID * p = &neighbours[0];
	    const gVID * pe = &neighbours[re];
	    const gVID * q = &gedges[gindex[u]];
	    const gVID * qe = &gedges[gindex[u+1]];

	    while( p != pe && q != qe ) {
		if( *p == *q ) {
		    // Common neighbour
		    VID c = n2s[p-&neighbours[0]];
		    if( cs <= c && c < ce ) {
			row_u = tr::bitwise_or( row_u, create_singleton( c ) );
			++deg;
		    }
		    ++p;
		    ++q;
		} else if( *p < *q )
		    ++p;
		else
		    ++q;
	    }

	    // assert( deg <= m_cols );
	    // assert( get_size( row_u ) == deg );
	    // assert( VL * (r-rs) <= VL * m_rows );
	    tr::store( &m_matrix[VL * (r-rs)], row_u );
	    if constexpr ( !AddDegree::value )
		// m_degree[r] += deg;
	    // else
		m_degree[r] = deg;
	    m_m += deg;
	}
    }
    // Assumes no relabeling required
    template<typename utr, typename HGraph,
	     typename DID, typename AddDegree,
	     typename gVID = VID, typename gEID = EID>
    BinaryMatrix( const ::GraphCSx & G,
		  const HGraph & H,
		  gVID rs, gVID re,
		  gVID cs, gVID ce,
		  const gVID * const s2g, // neighbours in natural sort and degeneracy sort order at same time
		  DID * m_degree,
		  AddDegree &&,
		  utr )
	: m_row_start( rs ), m_rows( re-rs ),
	  m_col_start( cs ), m_cols( ce-cs ) {
	assert( m_cols <= MAX_COL_VERTICES
		&& "Cap on number of vertices that fit in bitmask" );
	assert( ( m_cols + bits_per_lane - 1 ) / bits_per_lane <= VL );
	gVID n = G.numVertices();
	gEID m = G.numEdges();

	allocate();
	std::fill( m_matrix, m_matrix+VL*m_rows, type(0) );

	// Place edges
	VID ni = 0;
	m_m = 0;
	for( VID r=rs; r < re; ++r ) {
	    VID u = s2g[r];

	    bitmask_lhs_sorted_output_iterator<type, VID, true, true>
		row_u( &m_matrix[VL * (r-rs)], s2g, cs, cs, ce );

	    if constexpr ( utr::uses_hash ) {
		row_u = utr::template intersect<true>(
		    s2g+cs, s2g+re, H.get_adjacency( u ), row_u );
	    } else {
		const VID * const n = G.get_neighbours( u );
		VID deg = G.getDegree( u );

		row_u = utr::template intersect<true>(
		    s2g+cs, s2g+re, n, n+deg, row_u );
	    }
	    
	    VID deg = row_u.get_degree();
	    if constexpr ( !AddDegree::value )
		// m_degree[r] += deg;
	    // else
		m_degree[r] = deg;
	    m_m += deg;
	}
    }

    // In this variation, hVIDs are already sorted by core_order
    // and we know the separation between X and P sets.
    template<typename GGraph, typename HGraph,
	     typename DID, typename AddDegree>
    BinaryMatrix( const GGraph & G,
		  const HGraph & H,
		  const sVID * XP,
		  sVID ne, sVID ce,
		  DID * m_degree,
		  AddDegree && ) // correlates with xp vs px matrix
	: m_row_start( AddDegree::value == false ? 0 : ne ),
	  m_rows( AddDegree::value == false ? ce : ce-ne ), // X
	  m_col_start( AddDegree::value == false ? ne : 0 ),
	  m_cols( AddDegree::value == false ? ce-ne : ne ) { // P
	assert( AddDegree::value != false || ce-ne <= Bits ); // xp - side col
	assert( AddDegree::value != true || ne <= Bits ); // px - bottom rows

	// Vertices in X and P are independently already sorted by core order.
	// We do not reorder, primarily because only the order in P matters
	// for efficiency of enumeration.
	// Do we need to copy, or can we just keep a pointer to XP?

	assert( AddDegree::value != false
		|| ( ce-ne + bits_per_lane - 1 ) / bits_per_lane <= VL );
	assert( AddDegree::value != true
		|| ( ne + bits_per_lane - 1 ) / bits_per_lane <= VL );
	allocate();
	std::fill( &m_matrix[0], &m_matrix[VL * m_rows], 0 );

	// Place XP in hash table for fast intersection
	typename HGraph::hash_set_type XP_hash( XP, XP+ce );

	// Place edges
	sVID ni = 0;
	m_m = 0;
	for( sVID r=m_row_start; r < m_row_start+m_rows; ++r ) {
	    sVID u = XP[r];
	    sVID deg;
	    row_type row_u;
	    sVID udeg = G.getDegree( u );
	    const sVID * n = G.get_neighbours( u );

	    if( HGraph::has_dual_rep && ce > 2*udeg ) {
		std::tie( row_u, deg )
		    = graptor::graph::construct_row_hash_xp<tr>(
			G, H, XP_hash, XP, ne, ce, r, u, m_col_start,
			m_col_start + m_cols, m_col_start );
		tr::store( &m_matrix[VL * (r - m_row_start)], row_u );
	    } else {
		std::tie( row_u, deg )
		    = graptor::graph::construct_row_hash_adj_vec<tr>(
			G, H, XP, ne, ce, r, m_col_start, m_col_start+m_cols,
			m_col_start );
		tr::store( &m_matrix[VL * (r - m_row_start)], row_u );
	    }

	    if constexpr ( !AddDegree::value )
		m_degree[r] = deg;
	    m_m += deg;
	}
    }

    ~BinaryMatrix() {
	if( m_matrix_alc != nullptr ) {
	    delete[] m_matrix_alc;
	    m_matrix_alc = nullptr;
	}
    }

    sVID numRows() const { return m_rows; }
    sVID numCols() const { return m_cols; }
    // sEID numEdges() const { return m_m; }

    row_type get_row( sVID v ) const {
	// assert( m_row_start <= v && v < m_row_start+m_rows );
	return tr::load( &m_matrix[VL * (v - m_row_start)] );
    }

    row_type create_singleton( sVID v ) const {
	// assert( m_col_start <= v && v < m_col_start + m_cols );
	return tr::setglobaloneval( v - m_col_start );
    }

    row_type get_himask( sVID v ) const {
	// assert( m_col_start <= v && v <= m_col_start + m_cols );
	return tr::himask( v+1 - m_col_start );
    }

    row_type create_singleton_rel( sVID v ) const {
	// assert( 0 <= v && v < m_cols );
	return tr::setglobaloneval( v );
    }

    row_type get_himask_rel( sVID v ) const {
	// assert( 0 <= v && v <= m_cols );
	return tr::himask( v+1 );
    }

    sVID get_size( row_type r ) const {
	return target::allpopcnt<sVID,type,VL>::compute( r );
    }

    sVID get_col_start() const { return m_col_start; }

#if 0
    void complement() {
	row_type mask = tr::bitwise_invert( tr::himask( m_cols ) );
	for( VID i=0; i < m_rows; ++i ) {
	    VID r = i + m_row_start;
	    row_type b = tr::load( &m_matrix[i * VL] );
	    row_type c = tr::bitwise_xor( mask, b );
	    if( r >= m_col_start && r < m_col_start + m_cols ) { // TODO: can be faster
		row_type s = create_singleton_rel( i );
		c = tr::bitwise_andnot( s, c );
	    }
	    // Leave disconnected vertices as disconnected
	    if( !tr::is_zero( b ) )
		tr::store( &m_matrix[i * VL], c );
	}
	m_m = m_rows * m_cols - m_m; // flip count
    }
#endif

private:
    void allocate() {
	m_matrix = m_matrix_alc = new type[VL * m_rows + 64];
	intptr_t p = reinterpret_cast<intptr_t>( m_matrix );
	if( p & 63 ) // 63 = 512 bits / 8 bits per byte - 1
	    m_matrix = &m_matrix[64 - (p&63)/sizeof(type)];
	static_assert( Bits <= 512, "AVX512 requires 64-byte alignment" );
    }

private:
    const sVID m_rows;
    const sVID m_cols;
    const sVID m_row_start;
    const sVID m_col_start;
    sEID m_m;
    type * m_matrix;
    type * m_matrix_alc;
};


template<unsigned XBits, unsigned PBits, typename sVID, typename sEID>
class BlockedBinaryMatrix {
    static constexpr unsigned MAX_COL_VERTICES = PBits;
    using DID = std::conditional_t<PBits<=256,uint8_t,uint16_t>;

public:
    // The neighbours are split up in ineligible neighbours (initial X set)
    // and eligible neighbours (initial P set). They are already sorted
    // by decreasing coreness in the gID array. Their position in this array
    // reflects their relative index in this matrix. Columns are vertices in
    // gID from cs to ce; rows are gID elements rs to re.
    template<typename gVID = VID, typename gEID = EID>
    BlockedBinaryMatrix(
	const ::GraphCSx & G, gVID v,
	gVID num_neighbours, const gVID * const neighbours,
	const gVID * const s2g,
	const gVID * const n2s,
	gVID start_pos,
	const gVID * const core_order ) {
	gVID n = G.numVertices();
	gEID m = G.numEdges();
	const gEID * const gindex = G.getIndex();
	const gVID * const gedges = G.getEdges();

	// Set of eligible neighbours
	VID ns = num_neighbours;
	assert( ns - start_pos <= MAX_COL_VERTICES );

	// Construct two matrices
	// Rows: X union P; columns: P
	new ( &m_xp ) BinaryMatrix<PBits,sVID,sEID>(
	    G, VID(0), ns, start_pos, ns, neighbours, neighbours, n2s,
	    m_degree, std::false_type() );
	// Rows: P; columns: X
	new ( &m_px ) BinaryMatrix<XBits,sVID,sEID>(
	    G, start_pos, ns, VID(0), start_pos, neighbours, neighbours, n2s,
	    m_degree, std::true_type() );
    }
    template<typename utr, typename HGraph,
	     typename gVID = VID, typename gEID = EID>
    BlockedBinaryMatrix(
	const ::GraphCSx & G,
	const HGraph & H,
	gVID v,
	const NeighbourCutOutDegeneracyOrder<gVID,gEID> & cut,
	utr ) {
	gVID n = G.numVertices();
	gEID m = G.numEdges();
	const gEID * const gindex = G.getIndex();
	const gVID * const gedges = G.getEdges();

	// Set of eligible neighbours
	gVID ns = cut.get_num_vertices();
	const gVID * vert = cut.get_vertices();
	gVID start_pos = cut.get_start_pos();
	assert( ns - start_pos <= MAX_COL_VERTICES );

	// Construct two matrices
	// Rows: X union P; columns: P
	new ( &m_xp ) BinaryMatrix<PBits,sVID,sEID>(
	    G, H, VID(0), ns, start_pos, ns, vert,
	    m_degree, std::false_type(), utr() );
	// Rows: P; columns: X
	new ( &m_px ) BinaryMatrix<XBits,sVID,sEID>(
	    G, H, start_pos, ns, VID(0), start_pos, vert,
	    m_degree, std::true_type(), utr() );
    }
    // In this variation, hVIDs are already sorted by core_order
    // and we know the separation between X and P sets.
    template<typename GGraph, typename HGraph>
    BlockedBinaryMatrix(
	const GGraph & G,
	const HGraph & H,
	const sVID * XP,
	sVID ne, sVID ce ) {

	// Vertices in X and P are independently already sorted by core order.
	// We do not reorder, primarily because only the order in P matters
	// for efficiency of enumeration.

	// Construct two matrices
	// Rows: X union P; columns: P
	new ( &m_xp ) BinaryMatrix<PBits,sVID,sEID>(
	    G, H, XP, ne, ce, m_degree, std::false_type() );
	// Rows: P; columns: X
	new ( &m_px ) BinaryMatrix<XBits,sVID,sEID>(
	    G, H, XP, ne, ce, m_degree, std::true_type() );
    }

#if 0
    void complement() {
	m_xp.complement();
	m_px.complement();

	VID n = numVertices();
	for( VID i=0; i < n; ++i )
	    m_degree[i] = n - 1 - m_degree[i]; // -1 for self-edge
    }
#endif

    const BinaryMatrix<PBits,sVID,sEID> & get_xp() const { return m_xp; }
    const BinaryMatrix<XBits,sVID,sEID> & get_px() const { return m_px; }
    const DID * get_degree() const { return &m_degree[0]; }

    sVID numVertices() const { return m_xp.numRows(); }
    sVID numXVertices() const { return m_px.numCols(); }
    // sEID numEdges() const { return m_xp.numEdges() + m_px.numEdges(); }

private:
    BinaryMatrix<PBits,sVID,sEID> m_xp; //!< rightmost columns, in full
    BinaryMatrix<XBits,sVID,sEID> m_px; //!< bottommost rows, left parts
    DID m_degree[XBits+PBits]; //!< degree of vertices in cutout, m_xp only
};

template<unsigned XBits, unsigned PBits, typename sVID, typename sEID>
sVID get_pivot(
    const BlockedBinaryMatrix<XBits,PBits,sVID,sEID> & mtx,
    typename BinaryMatrix<PBits,sVID,sEID>::row_type Pp,
    typename BinaryMatrix<PBits,sVID,sEID>::row_type Xp,
    typename BinaryMatrix<XBits,sVID,sEID>::row_type Xx ) {

    using ptr = typename BinaryMatrix<PBits,sVID,sEID>::tr;
    using xtr = typename BinaryMatrix<XBits,sVID,sEID>::tr;

    const BinaryMatrix<PBits,sVID,sEID> & xp = mtx.get_xp();
    const BinaryMatrix<XBits,sVID,sEID> & px = mtx.get_px();
    const auto * degree = mtx.get_degree();

    auto r = ptr::bitwise_or( Pp, Xp );
    
    bitset<PBits> b( r );

    VID cs = xp.get_col_start();
    VID p_best = *b.begin() + cs;
    VID p_ins = 0; // will be overridden

    // Avoid complexities if there is not much choice
    if( xp.get_size( Pp ) <= 3 ) // Tunable: 3
	return p_best;

    for( auto I=b.begin(), E=b.end(); I != E; ++I ) {
	VID v = *I + cs;
	if( (VID)degree[v] < p_ins ) // skip if cannot be best
	    continue;
	auto v_ngh = xp.get_row( v );
	auto pv_ins = ptr::bitwise_and( Pp, v_ngh );
	VID ins = xp.get_size( pv_ins );
	if( ins > p_ins ) {
	    p_best = v;
	    p_ins = ins;
	}
    }

    bitset<XBits> c( Xx );
    assert( px.get_col_start() == 0 && "should always be zero" );
    for( auto I=c.begin(), E=c.end(); I != E; ++I ) {
	VID v = *I;
	if( (VID)degree[v] < p_ins ) // skip if cannot be best
	    continue;
	auto v_ngh = xp.get_row( v );
	auto pv_ins = ptr::bitwise_and( Pp, v_ngh );
	VID ins = xp.get_size( pv_ins );
	if( ins > p_ins ) {
	    p_best = v;
	    p_ins = ins;
	}
    }

    assert( ~p_best != 0 );
    return p_best;
}


template<unsigned XBits, unsigned PBits, typename sVID, typename sEID,
	 typename Enumerate>
void mce_bk_iterate(
    const BlockedBinaryMatrix<XBits,PBits,sVID,sEID> & mtx,
    Enumerate && EE,
    typename BinaryMatrix<PBits,sVID,sEID>::row_type R,
    typename BinaryMatrix<PBits,sVID,sEID>::row_type Pp,
    typename BinaryMatrix<PBits,sVID,sEID>::row_type Xp,
    typename BinaryMatrix<XBits,sVID,sEID>::row_type Xx,
    int depth ) {
	
    using ptr = typename BinaryMatrix<PBits,sVID,sEID>::tr;
    using prow_type = typename BinaryMatrix<PBits,sVID,sEID>::row_type;
    using xtr = typename BinaryMatrix<XBits,sVID,sEID>::tr;
    using xrow_type = typename BinaryMatrix<XBits,sVID,sEID>::row_type;

    const BinaryMatrix<PBits,sVID,sEID> & xp = mtx.get_xp();
    const BinaryMatrix<XBits,sVID,sEID> & px = mtx.get_px();

    sVID n = mtx.numVertices();
    sVID x = px.numCols();

    // depth == get_size( R )
    if( ptr::is_zero( Pp ) ) {
	if( ptr::is_zero( Xp ) && xtr::is_zero( Xx ) )
	    EE( bitset<PBits>( R ), depth ); // note: offset elements by n-x
	return;
    }

    VID pivot = get_pivot<XBits,PBits,sVID,sEID>( mtx, Pp, Xp, Xx );
    prow_type pivot_ngh = xp.get_row( pivot );
    prow_type ins = ptr::bitwise_andnot( pivot_ngh, Pp );
    bitset<PBits> bx( ins );
    VID cs = xp.get_col_start();
    for( auto I = bx.begin(), E = bx.end(); I != E; ++I ) {
	sVID u = *I + cs;
	prow_type u_only = xp.create_singleton_rel( *I );
	// prow_type xp_new = ptr::bitwise_andnot( u_only, ins );
	// xrow_type xx_new = xx; // u in P; u not in X
	// ins = xp_new;
	prow_type pu_ngh = xp.get_row( u );
	prow_type Ppv = ptr::bitwise_and( Pp, pu_ngh );
	prow_type Xpv = ptr::bitwise_and( Xp, pu_ngh );
	xrow_type xu_ngh = px.get_row( u );
	xrow_type Xxv = xtr::bitwise_and( Xx, xu_ngh );
	prow_type Rv = ptr::bitwise_or( R, u_only );
	Pp = ptr::bitwise_andnot( u_only, Pp ); // Pp == ins w/o pivoting
	Xp = ptr::bitwise_or( u_only, Xp );
	// Xx unmodified as u in P
	mce_bk_iterate( mtx, EE, Rv, Ppv, Xpv, Xxv, depth+1 );
    }
}


template<unsigned XBits, unsigned PBits, typename sVID, typename sEID,
	 typename Enumerate>
void
mce_bron_kerbosch(
    const BlockedBinaryMatrix<XBits,PBits,sVID,sEID> & mtx,
    Enumerate && EE ) {
    using ptr = typename BinaryMatrix<PBits,sVID,sEID>::tr;
    using prow_type = typename BinaryMatrix<PBits,sVID,sEID>::row_type;
    using xtr = typename BinaryMatrix<XBits,sVID,sEID>::tr;
    using xrow_type = typename BinaryMatrix<XBits,sVID,sEID>::row_type;

    const BinaryMatrix<PBits,sVID,sEID> & xp = mtx.get_xp();
    const BinaryMatrix<XBits,sVID,sEID> & px = mtx.get_px();
    assert( px.numRows() + px.numCols() == xp.numRows() );
    assert( px.numRows() == xp.numCols() );

    sVID n = mtx.numVertices();
    sVID cs = xp.get_col_start();

    // implicitly skips X vertices; iterate over P vertices
    for( sVID v=cs; v < n; ++v ) {
	prow_type R = xp.create_singleton( v );
	prow_type r = xp.get_row( v );
	xrow_type Xx = px.get_row( v );

	// if no neighbours in cut-out, then trivial 2-clique
	if( ptr::is_zero( r ) && xtr::is_zero( Xx ) ) {
	    EE( bitset<PBits>( R ), 1 );
	    continue;
	}

	// Consider as candidates only those neighbours of u that are
	// ordered after v to avoid revisiting the vertices
	// unnecessarily.
	prow_type h = xp.get_himask( v );
	prow_type Pp = ptr::bitwise_and( h, r );
	prow_type Xp = ptr::bitwise_andnot( h, r );
	// std::cerr << "depth " << 0 << " v=" << v << "\n";
	mce_bk_iterate( mtx, EE, R, Pp, Xp, Xx, 1 );
    }
}


} // namespace graph

} // namespace graptor

#endif // GRAPHGRIND_GRAPH_SIMPLE_BLOCKED_H
