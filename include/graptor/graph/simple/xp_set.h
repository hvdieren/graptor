// -*- C++ -*-
#ifndef GRAPHGRIND_GRAPH_SIMPLE_XPSET_H
#define GRAPHGRIND_GRAPH_SIMPLE_XPSET_H

#include "graptor/target/vector.h"

namespace graptor {

namespace graph {

//! Inspired by Fast Arrays
// Constant-time initialisation
template<typename _lVID = VID>
class XPSet {
public:
    using lVID = _lVID;

    template<bool present_p>
    class hash_set_interface {
    public:
	hash_set_interface( const XPSet & xp, lVID ne )
	    : m_xp( xp ), m_ne( ne ) { }

	bool contains( lVID v ) const {
	    lVID pos = m_xp.m_pos[v];
	    if( pos >= m_xp.m_fill ) // not a certificate
		return false;
	    if( m_xp.m_set[pos] != v ) // not a valid certificate
		return false;
	    if constexpr ( present_p )
		return pos >= m_ne;
	    else
		return pos < m_ne;
	}

	template<typename U, unsigned short VL, typename MT>
	std::conditional_t<std::is_same_v<MT,target::mt_mask>,
			   typename vector_type_traits_vl<U,VL>::mask_type,
			   typename vector_type_traits_vl<U,VL>::vmask_type>
	multi_contains( typename vector_type_traits_vl<U,VL>::type
			index, MT ) const {
	    static_assert( sizeof( U ) >= sizeof( lVID ) );
	    using tr = vector_type_traits_vl<U,VL>;
	    using str = vector_type_traits_vl<std::make_signed_t<U>,VL>;
	    using vtype = typename tr::type;
#if __AVX512F__
	    using mtr = typename tr::mask_traits;
	    using mkind = target::mt_mask;
#else
	    using mtr = typename tr::vmask_traits;
	    using mkind = target::mt_vmask;
#endif
	    using mtype = typename mtr::type;

	    // we silently assume that all values are >= 0
	    static_assert( std::is_unsigned_v<lVID>,
			   "arithmetic requires unsigned type" );

	    // load the positions that might be used
	    vtype pos = tr::gather( m_xp.m_pos, index );
	    // check if positions are valid
	    mtype mc = tr::cmplt( pos, tr::set1( m_xp.m_fill ), mkind() );
	    // load the certificates, if plausible
	    vtype vc = tr::gather( m_xp.m_set, pos, mc );
	    // check which certificates are valid
	    mtype mv = tr::cmpeq( mc, vc, index, mkind() );

	    // determine presence of vertices in either X or P
	    // in AVX2, unsigned cmplt/gt is cheaper than ge/le
	    // use signed comparison as only interested in valid positions
	    // which can be assumed to fit in signed quantities
	    mtype is_x = str::cmplt( pos, str::set1( m_ne ), mkind() );
	    mtype ret;
	    if constexpr ( present_p )
		ret = mtr::logical_andnot( is_x, mv );
	    else
		ret = mtr::logical_and( is_x, mv );
	    if constexpr ( std::is_same_v<mkind,MT> )
		return ret;
	    else if constexpr ( std::is_same_v<MT,target::mt_mask> )
		return tr::asmask( ret );
	    else
		return tr::asvector( ret );
	}

    private:
	const XPSet & m_xp;
	lVID m_ne;
    };

    // The methods are a misnomer. Should be lookup. Using contains to
    // fit in with the intersect template code.
    class hash_table_interface {
    public:
	hash_table_interface( const XPSet & xp ) : m_xp( xp ) { }

	lVID contains( lVID v ) const {
	    lVID pos = m_xp.m_pos[v];
	    if( pos >= m_xp.m_fill ) // not a certificate
		return ~lVID(0);
	    if( m_xp.m_set[pos] != v ) // not a valid certificate
		return ~lVID(0);
	    return pos;
	}

	template<typename U, unsigned short VL, typename MT>
	typename vector_type_traits_vl<U,VL>::type
	multi_contains(
	    typename vector_type_traits_vl<U,VL>::type index,
	    MT ) const {
	    return multi_contains<U,VL>( index );
	}

	template<typename U, unsigned short VL>
	typename vector_type_traits_vl<U,VL>::type
	multi_contains(
	    typename vector_type_traits_vl<U,VL>::type index ) const {
	    static_assert( sizeof( U ) >= sizeof( lVID ) );
	    using tr = vector_type_traits_vl<U,VL>;
	    using str = vector_type_traits_vl<std::make_signed_t<U>,VL>;
	    using vtype = typename tr::type;
#if __AVX512F__
	    using mtr = typename tr::mask_traits;
	    using mkind = target::mt_mask;
#else
	    using mtr = typename tr::vmask_traits;
	    using mkind = target::mt_vmask;
#endif
	    using mtype = typename mtr::type;

	    // we silently assume that all values are >= 0
	    static_assert( std::is_unsigned_v<lVID>,
			   "arithmetic requires unsigned type" );

	    // load the positions that might be used
	    vtype pos = tr::gather( m_xp.m_pos, index );
	    // check if positions are valid
	    mtype mc = tr::cmplt( pos, tr::set1( m_xp.m_fill ), mkind() );
	    // load the certificates, if plausible
	    vtype vc = tr::gather( m_xp.m_set, pos, mc );
	    // check which certificates are valid
	    mtype mv = tr::cmpeq( mc, vc, index, mkind() );

	    // determine presence of vertices in either X or P
	    // in AVX2, unsigned cmplt/gt is cheaper than ge/le
	    // use signed comparison as only interested in valid positions
	    // which can be assumed to fit in signed quantities
	    vtype inv = tr::setone();
	    return tr::blend( mv, inv, pos );
	}

    private:
	const XPSet & m_xp;
    };

public:
    XPSet( VID n, VID ce_max )
	: m_pos( new lVID[n+ce_max] ), m_set( m_pos+n ), m_fill( 0 ) {
	// Silent errors occur when allocating zero elements...
	assert( n > 0 && "require non-empty graph" );
    }
private:
    XPSet( XPSet && xp )
	: m_set( std::forward<lVID*>( xp.m_set ) ),
	  m_pos( std::forward<lVID*>( xp.m_pos ) ),
	  m_fill( std::forward<lVID>( xp.m_fill ) ) {
	xp.m_set = nullptr;
	xp.m_pos = nullptr;
	xp.m_fill = 0;
    }
    XPSet( const XPSet & ) = delete;
    XPSet & operator = ( const XPSet & ) = delete;

public:
    ~XPSet() {
	if( m_pos )
	    delete[] m_pos;
    }

    // A top level iteration for a cut-out graph. v is currently being visited
    // and we are setting up an XPSet to describe its position
    // in the search tree.
    template<typename HGraphTy>
    static XPSet
    create_top_level( const HGraphTy & G, lVID v /*, lVID & ne, lVID & ce */ ) {
	lVID n = G.numVertices();
	const lVID deg = G.getDegree( v );

	// At top level, deg <= n
	XPSet xp( n, deg );

	if constexpr ( HGraphTy::has_dual_rep ) {
	    const auto * ngh = G.get_neighbours( v );
	    for( VID i=0; i < deg; ++i ) {
		// assert( ngh[i] < n );
		xp.m_set[i] = ngh[i];
		xp.m_pos[ngh[i]] = i;
	    }
	    xp.m_fill = deg;
	} else {
	    const auto & adj = G.get_adjacency( v ); 
	    auto end = std::copy_if(
		adj.begin(), adj.end(), xp.m_set,
		[&]( VID v ) { return v != adj.invalid_element; } );
	    assert( end - xp.m_set == deg );

	    // If sorting is required
	    std::sort( xp.m_set, xp.m_set+deg ); // semisort X/P?

	    // Only now that elements in m_set are in place can we build m_pos
	    for( lVID i=0; i < deg; ++i ) {
		// assert( xp.m_set[i] < n );
		xp.m_pos[xp.m_set[i]] = i;
	    }
	    xp.m_fill = deg;
	}

	// v is not a neighbour of itself, hence not a member of this set.
	// Search for the first larger vertex. Requires that xp.m_set is
	// sorted. We need at least semisort between X/P anyway, so sorting
	// is reasonable, especially if we assume that neighbour lists in G
	// have been pre-sorted.
	// Note: now computed outside this function
	// const lVID * pos = std::upper_bound( xp.m_set, xp.m_set+deg, v );
	// ne = pos - xp.m_set;
	// ce = deg;

	return xp;
    }

    template<typename HGraphTy>
    static XPSet
    create_full_set( const HGraphTy & G ) {
	lVID n = G.numVertices();

	XPSet xp( n, n );

	for( lVID i=0; i < n; ++i ) {
	    xp.m_set[i] = i;
	    xp.m_pos[i] = i;
	}
	xp.m_fill = n;

	return xp;
    }

    template<typename Adj>
    void semisort_pivot( lVID ne, lVID pe, lVID ce, const Adj & padj ) {
	assert( ce == m_fill );

	// Semisort P into P\N(pivot) and P\cap N(pivot)
	// Assumption: pe + |intersect(padj,P)| == ce
	lVID P_ins = pe;
	for( lVID i=ne; i < pe; ++i ) {
	    if( padj.contains( m_set[i] ) ) {
		// Find insertion point
		lVID v = m_set[i];
		while( padj.contains( m_set[P_ins] ) )
		    P_ins++;

		assert( P_ins < ce );
		// swap
		lVID u = m_set[P_ins];
		m_set[i] = u;
		m_set[P_ins] = v;
		m_pos[u] = i;
		m_pos[v] = P_ins;
		P_ins++;

		// assert( m_pos[u] < m_fill && m_set[m_pos[u]] == u );
		// assert( m_pos[v] < m_fill && m_set[m_pos[v]] == v );

		// All done - no more vertices available to swap
		if( P_ins == ce )
		    break;
	    }
	}
    }

    template<typename Adj>
    void semisort_pivot_deposit(
	lVID ne, lVID pe, lVID ce, const Adj & padj, lVID n ) {

	lVID p_ins = pe, c_ins = ne;
	for( lVID i=0; i < ne; ++i ) {
	    m_set[i] = i;
	    m_pos[i] = i;
	}
	for( lVID i=ne; i < ce; ++i ) {
	    if( padj.contains( i ) ) {
		m_set[p_ins] = i;
		m_pos[i] = p_ins++;
	    } else {
		m_set[c_ins] = i;
		m_pos[i] = c_ins++;
	    }
	}
	assert( c_ins == pe );
	assert( p_ins == ce );

	m_fill = ce;
    }

    // A hash interface supporting contains, multi_contains
    // that checks for either pos < ne or pos >= ne or pos == ~0
    // present hash_set-like interface for X set
    auto X_hash_set( lVID ne ) const {
	return hash_set_interface<false>( *this, ne );
    }
    // present hash_set-like interface for P set
    auto P_hash_set( lVID ne ) const {
	return hash_set_interface<true>( *this, ne );
    }
    // present hash_table-like interface for X and P (vertex ID translation)
    auto hash_table() const {
	return hash_table_interface( *this );
    }

    template<typename Adj>
    XPSet intersect( lVID n, lVID i, lVID ce,
		     const Adj & adj, const lVID * ngh,
		     lVID & ne_new, lVID & ce_new ) const {
	// assert( i <= ce );
	// assert( ce == m_fill );
	// assert( ce <= n );
	// Pass in ne here as an argument instead of storing it
	// Intersect both X and P with adj, resulting in X' and P'
	lVID deg = adj.size();
	lVID mx = std::min( deg, ce );
	XPSet ins( n, mx+8 ); // hash_vector requires extra space

	if( ce > 2*deg ) {
	    // TODO: find split point for X/P to reduce ranges?
	    //       or make single traversal and determine X/P on the fly?
	    //       need to be careful as m_set may not be sorted, and there
	    //       may exist elements of P that are smaller than some elements
	    //       of X (due to pivoting)
	    ne_new = graptor::hash_vector::intersect(
		ngh, ngh+deg, X_hash_set( i ), ins.m_set ) - ins.m_set;
	    ce_new = graptor::hash_vector::intersect(
		ngh, ngh+deg, P_hash_set( i ), ins.m_set+ne_new ) - ins.m_set;
	} else {
	    // Note: skip m_set[i] as we know there are no self-loops.
	    ne_new = graptor::hash_vector::intersect(
		m_set, m_set+i, adj, ins.m_set ) - ins.m_set;
	    ce_new = graptor::hash_vector::intersect(
		m_set+i+1, m_set+ce, adj, ins.m_set+ne_new ) - ins.m_set;
	}

	// Construct ins.m_pos 
	for( lVID i=0; i < ce_new; ++i ) {
	    // assert( ins.m_set[i] < n );
	    ins.m_pos[ins.m_set[i]] = i;
	}

	ins.m_fill = ce_new;

	return ins;
    }

    template<typename Adj>
    static XPSet intersect_top_level(
	lVID n, lVID ne_strict, lVID ne,
	const Adj & p_adj, const lVID * p_ngh,
	const Adj & adj, const lVID * ngh,
	lVID & ne_new, lVID & ce_new ) {

	// The current XPSet contains all vertices 0..n-1 split in:
	// * 0..ne-1 subtract p_adj: non-edges (X set)
	// * 0..ce-1 intersect p_adj: postponed P neighbours of the pivot
	// * ne..ce-1 subtract p_adj: non-P neighbours currently processed
	// The intersection between this XPSet and the adjacency list adj
	// thus comprises:
	// * X-set  : ( 0..ne-1 subtract p_adj ) intersect adj
	// * P-set 1: 0..ce-1 intersect p_adj intersect adj
	// * P-set 2: ( ne..ce-1 subtract p_adj ) intersect adj
	// OR:
	// * X-set  : N(v) subtract N(p), all below ne
	// * P-set 1: N(v) intersect N(p)
	// * P-set 2: N(v) subtract N(p), above ne
	// Fix: 0..ne_strict-1 is always part of X, regardless of p_adj
	// Thus:
	// * X-set 1: N(v), all below ne_strict
	// * X-set 2: N(v) subtract N(p), in range ne_strict..ne-1
	// * P-set 1: N(v) intersect N(p)
	// * P-set 2: N(v) subtract N(p), above ne

	lVID p_deg = p_adj.size();
	lVID mx = std::min( deg + p_deg, n );
	XPSet ins( n, mx+8 ); // hash_vector requires extra space

	// X set 1
	const lVID * const ne_strict_p
	    = std::lower_bound( ngh, ngh+deg, ne_strict );
	lVID * pos = std::copy( ngh, ne_strict_p, ins.m_set );
	
	// X set 2
	const lVID * const ne_p = std::lower_bound( ne_strict_p, ngh+deg, ne );
	pos = graptor::hash_vector::intersect_invert<false>(
	    ne_strict_p, ne_p, p_adj, pos );

	ne_new = pos - ins.m_set;

	// P set 1
	pos = graptor::hash_vector::intersect( ne_strict_p, ne_p, p_adj, pos );
	// P set 2
	pos = std::copy( ne_p, ngh+deg, pos );

	ins.m_fill = ce_new = pos - ins.m_set;

	// Complete hash info
	for( lVID i=0; i < ce_new; ++i )
	    ins.m_pos[ins.m_set[i]] = i;

	return ins;
    }

    /*const*/ lVID * get_set() const { return m_set; }
    lVID at( lVID pos ) const { return m_set[pos]; }

    void sort( lVID ne ) {
	std::sort( m_set, m_set+ne );
	std::sort( m_set+ne, m_set+m_fill );
    }

private:
    lVID * m_pos;
    lVID * m_set;
    lVID m_fill;
};


} // namespace graph

} // namespace graptor

#endif // GRAPHGRIND_GRAPH_SIMPLE_XPSET_H
