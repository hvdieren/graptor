// -*- C++ -*-
#ifndef GRAPHGRIND_GRAPH_SIMPLE_XPSET_H
#define GRAPHGRIND_GRAPH_SIMPLE_XPSET_H

#include "graptor/target/vector.h"

namespace graptor {

namespace graph {

template<typename HashSet1, typename HashSet2>
class hash_set_or {
    using type = typename HashSet1::type;
    
public:
    hash_set_or( const HashSet1 & h1, const HashSet2 & h2 )
	: m_hash1( h1 ), m_hash2( h2 ) { }

    bool contains( type v ) const {
	return m_hash1.contains( v ) || m_hash2.contains( v );
    }

    template<typename U, unsigned short VL, typename MT>
    std::conditional_t<std::is_same_v<MT,target::mt_mask>,
		       typename vector_type_traits_vl<U,VL>::mask_type,
		       typename vector_type_traits_vl<U,VL>::vmask_type>
    multi_contains( typename vector_type_traits_vl<U,VL>::type index,
		    MT ) const {
	using tr = vector_type_traits_vl<U,VL>;

	auto r1 = m_hash1.template multi_contains<U,VL>( index, MT() );
	auto r2 = m_hash2.template multi_contains<U,VL>( index, MT() );

	if constexpr ( std::is_same_v<MT,target::mt_mask> ) {
	    using mtr = typename tr::mask_traits;
	    return mtr::logical_or( r1, r2 );
	} else
	    return tr::logical_or( r1, r2 );
    }

private:
    const HashSet1 & m_hash1;
    const HashSet2 & m_hash2;
};

template<typename HashSet1, typename HashSet2>
class hash_set_and {
    using type = typename HashSet1::type;
    
public:
    hash_set_and( const HashSet1 & h1, const HashSet2 & h2 )
	: m_hash1( h1 ), m_hash2( h2 ) { }

    bool contains( type v ) const {
	return m_hash1.contains( v ) && m_hash2.contains( v );
    }

    template<typename U, unsigned short VL, typename MT>
    std::conditional_t<std::is_same_v<MT,target::mt_mask>,
		       typename vector_type_traits_vl<U,VL>::mask_type,
		       typename vector_type_traits_vl<U,VL>::vmask_type>
    multi_contains( typename vector_type_traits_vl<U,VL>::type index,
		    MT ) const {
	using tr = vector_type_traits_vl<U,VL>;

	// TODO: invalidate lanes in lookup in hash2 based on r1 result
	auto r1 = m_hash1.template multi_contains<U,VL>( index, MT() );
	auto r2 = m_hash2.template multi_contains<U,VL>( index, MT() );

	if constexpr ( std::is_same_v<MT,target::mt_mask> ) {
	    using mtr = typename tr::mask_traits;
	    return mtr::logical_and( r1, r2 );
	} else
	    return tr::logical_and( r1, r2 );
    }

private:
    const HashSet1 & m_hash1;
    const HashSet2 & m_hash2;
};

template<typename HashSet1, typename HashSet2>
auto make_hash_set_and( const HashSet1 & hset1,
			const HashSet2 & hset2 ) {
    return hash_set_and<HashSet1,HashSet2>( hset1, hset2 );
}

//! Inspired by Fast Arrays
// Constant-time initialisation
template<typename _lVID = VID>
class XPSetBase {
public:
    using lVID = _lVID;
    using type = _lVID;

    template<bool present_p>
    class hash_set_interface {
    public:
	using type = lVID;
	
	hash_set_interface( const XPSetBase & xp, lVID ne )
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

	const lVID * begin() const {
	    if constexpr ( present_p )
		return m_xp.begin() + m_ne;
	    else
		return m_xp.begin();
	}
	const lVID * end() const {
	    if constexpr ( present_p )
		return m_xp.end();
	    else
		return m_xp.begin() + m_ne;
	}
	const lVID size() const {
	    if constexpr ( present_p )
		return m_xp.size() - m_ne;
	    else
		return m_ne;
	}

	const hash_set_interface & trim_range( lVID lo, lVID hi ) const {
	    // TODO: trim P or X range; may adjust ne to accomplish this
	    return *this;
	}

    private:
	const XPSetBase & m_xp;
	lVID m_ne;
    };

    // The methods are a misnomer. Should be lookup. Using contains to
    // fit in with the intersect template code.
    class hash_table_interface {
    public:
	using type = lVID;
	
	hash_table_interface( const XPSetBase & xp ) : m_xp( xp ) { }

	bool contains( lVID v ) const {
	    lVID pos = m_xp.m_pos[v];
	    if( pos >= m_xp.m_fill ) // not a certificate
		return false;
	    if( m_xp.m_set[pos] != v ) // not a valid certificate
		return false;
	    return true;
	}

	lVID lookup( lVID v ) const {
	    lVID pos = m_xp.m_pos[v];
	    if( pos >= m_xp.m_fill ) // not a certificate
		return ~lVID(0);
	    if( m_xp.m_set[pos] != v ) // not a valid certificate
		return ~lVID(0);
	    return pos;
	}

	template<typename U, unsigned short VL, typename MT>
	std::conditional_t<std::is_same_v<MT,target::mt_mask>,
			   typename vector_type_traits_vl<U,VL>::mask_type,
			   typename vector_type_traits_vl<U,VL>::vmask_type>
	multi_contains(
	    typename vector_type_traits_vl<U,VL>::type index,
	    MT ) const {
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

	    mtype present;
	    vtype value;
	    std::tie( present, value ) = multi_helper<U,VL>( index );

	    if constexpr ( std::is_same_v<mkind,MT> )
		return present;
	    else if constexpr ( std::is_same_v<MT,target::mt_mask> )
		return tr::asmask( present );
	    else
		return tr::asvector( present );
	}

	template<typename U, unsigned short VL>
	std::pair<typename vector_type_traits_vl<U,VL>::mask_type,
		  typename vector_type_traits_vl<U,VL>::type>
	multi_lookup(
	    typename vector_type_traits_vl<U,VL>::type index ) const {
	    return multi_helper<U,VL>;
	}
	    
	const lVID * begin() const { return m_xp.begin(); }
	const lVID * end() const { return m_xp.end(); }
	const lVID size() const { return m_xp.size(); }

    private:
	template<typename U, unsigned short VL>
	std::pair<typename vector_type_traits_vl<U,VL>::mask_type,
		  typename vector_type_traits_vl<U,VL>::type>
	multi_helper(
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

	    return std::make_pair( mv, pos );
	}

    private:
	const XPSetBase & m_xp;
    };

public:
    XPSetBase( VID n, VID ce_max )
	: m_pos( new lVID[n+ce_max] ), m_set( m_pos+n ), m_fill( 0 ) {
	// Silent errors occur when allocating zero elements...
	assert( n > 0 && "require non-empty graph" );
    }
protected:
    XPSetBase( XPSetBase && xp )
	: m_set( std::forward<lVID*>( xp.m_set ) ),
	  m_pos( std::forward<lVID*>( xp.m_pos ) ),
	  m_fill( std::forward<lVID>( xp.m_fill ) ) {
	xp.m_set = nullptr;
	xp.m_pos = nullptr;
	xp.m_fill = 0;
    }
    XPSetBase( const XPSetBase & ) = delete;
    XPSetBase & operator = ( const XPSetBase & ) = delete;

public:
    ~XPSetBase() {
	if( m_pos )
	    delete[] m_pos;
    }

    // A top level iteration for a cut-out graph. v is currently being visited
    // and we are setting up an XPSet to describe its position
    // in the search tree.
    template<typename HGraphTy>
    static XPSetBase
    create_top_level( const HGraphTy & G, lVID v ) {
	lVID n = G.numVertices();
	const lVID deg = G.getDegree( v );

	// At top level, deg <= n
	XPSetBase xp( n, deg );

	if constexpr ( HGraphTy::has_dual_rep ) {
	    const auto * ngh = G.get_neighbours( v );
	    for( VID i=0; i < deg; ++i ) {
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
    static XPSetBase
    create_full_set( const HGraphTy & G ) {
	lVID n = G.numVertices();

	XPSetBase xp( n, n );

	for( lVID i=0; i < n; ++i ) {
	    xp.m_set[i] = i;
	    xp.m_pos[i] = i;
	}
	xp.m_fill = n;

	return xp;
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

    /*const*/ lVID * get_set() const { return m_set; }
    lVID at( lVID pos ) const { return m_set[pos]; }
    lVID get_fill() const { return m_fill; }
    lVID size() const { return m_fill; }

    const lVID * begin() const { return m_set; }
    const lVID * end() const { return m_set + m_fill; }

protected:
    lVID * m_pos;
    lVID * m_set;
    lVID m_fill;
};
    
template<typename _lVID = VID>
class XPSet : public XPSetBase<_lVID> {
public:
    using lVID = _lVID;

public:
    XPSet( VID n, VID ce_max )
	: XPSetBase<lVID>( n, ce_max ) { }
protected:
    XPSet( XPSet && xp )
	: XPSetBase<lVID>( std::forward<XPSetBase<lVID>>( xp ) ) { }
    XPSet( const XPSet & ) = delete;
    XPSet & operator = ( const XPSet & ) = delete;

public:
    template<typename Adj>
    void semisort_pivot( lVID ne, lVID pe, lVID ce, const Adj & padj ) {
	assert( ce == this->m_fill );

	// Semisort P into P\N(pivot) and P\cap N(pivot)
	// Assumption: pe + |intersect(padj,P)| == ce
	lVID P_ins = pe;
	for( lVID i=ne; i < pe; ++i ) {
	    if( padj.contains( this->m_set[i] ) ) {
		// Find insertion point
		lVID v = this->m_set[i];
		while( padj.contains( this->m_set[P_ins] ) )
		    P_ins++;

		assert( P_ins < ce );
		// swap
		lVID u = this->m_set[P_ins];
		this->m_set[i] = u;
		this->m_set[P_ins] = v;
		this->m_pos[u] = i;
		this->m_pos[v] = P_ins;
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
	    this->m_set[i] = i;
	    this->m_pos[i] = i;
	}
	for( lVID i=ne; i < ce; ++i ) {
	    if( padj.contains( i ) ) {
		this->m_set[p_ins] = i;
		this->m_pos[i] = p_ins++;
	    } else {
		this->m_set[c_ins] = i;
		this->m_pos[i] = c_ins++;
	    }
	}
	assert( c_ins == pe );
	assert( p_ins == ce );

	this->m_fill = ce;
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
	XPSet ins( n, mx+16 ); // hash_vector requires extra space

	if( ce > 2*deg ) {
	    // TODO: find split point for X/P to reduce ranges?
	    //       or make single traversal and determine X/P on the fly?
	    //       need to be careful as m_set may not be sorted, and there
	    //       may exist elements of P that are smaller than some elements
	    //       of X (due to pivoting)
	    ne_new = graptor::hash_vector::intersect(
		ngh, ngh+deg, this->X_hash_set( i ), ins.m_set ) - ins.m_set;
	    ce_new = graptor::hash_vector::intersect(
		ngh, ngh+deg, this->P_hash_set( i ), ins.m_set+ne_new ) - ins.m_set;
	} else {
	    // Note: skip m_set[i] as we know there are no self-loops.
	    ne_new = graptor::hash_vector::intersect(
		this->m_set, this->m_set+i, adj, ins.m_set ) - ins.m_set;
	    ce_new = graptor::hash_vector::intersect(
		this->m_set+i+1, this->m_set+ce, adj, ins.m_set+ne_new ) - ins.m_set;
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
	lVID deg = adj.size();
	lVID mx = std::min( deg + p_deg, n );
	XPSet ins( n, mx+16 ); // hash_vector requires extra space

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
	for( lVID i=0; i < ins.m_fill; ++i )
	    ins.m_pos[ins.m_set[i]] = i;

	return ins;
    }

    void sort( lVID ne ) {
	std::sort( this->m_set, this->m_set+ne );
	std::sort( this->m_set+ne, this->m_set+this->m_fill );
    }
};

template<typename _lVID = VID>
class PSet : public XPSetBase<_lVID> {
public:
    using lVID = _lVID;

    class p_hash_set_interface {
    public:
	using type = _lVID;

	p_hash_set_interface( const PSet & xp )
	    : m_xp( xp ) { }

	bool contains( lVID v ) const {
	    lVID pos = m_xp.m_pos[v];
	    if( pos >= m_xp.m_fill ) // not a certificate
		return false;
	    if( m_xp.m_set[pos] != v ) // not a valid certificate
		return false;
	    return true;
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

	    if constexpr ( std::is_same_v<mkind,MT> )
		return mv;
	    else if constexpr ( std::is_same_v<MT,target::mt_mask> )
		return tr::asmask( mv );
	    else
		return tr::asvector( mv );
	}

	size_t size() const { return m_xp.get_fill(); }

	const lVID * begin() const { return m_xp.begin(); }
	const lVID * end() const { return m_xp.end(); }

	const p_hash_set_interface & trim_range( lVID lo, lVID hi ) const {
	    // TODO: trim range
	    return *this;
	}

    private:
	const PSet & m_xp;
    };

public:
    PSet( VID n, VID ce_max )
	: XPSetBase<lVID>( n, ce_max ) { }
protected:
    PSet( PSet && xp )
	: XPSetBase<lVID>( std::forward<XPSetBase<lVID>>( xp ) ) { }
    PSet( XPSetBase<lVID> && xp )
	: XPSetBase<lVID>( std::forward<XPSetBase<lVID>>( xp ) ) { }
    PSet( const PSet & ) = delete;
    PSet & operator = ( const PSet & ) = delete;

public:
    // A hash interface supporting contains, multi_contains
    auto hash_set() const {
	return p_hash_set_interface( *this );
    }

    static PSet copy( lVID n, const PSet & p ) {
	PSet<lVID> cp( n, p.size() );
	std::copy( p.m_pos, p.m_pos+n, cp.m_pos );
	std::copy( p.m_set, p.m_set+p.size(), cp.m_set );
	cp.m_fill = p.m_fill;
	return cp;
    }
    
    // Intersect this with adjacency list, interested only in vertices
    // that are higher-numbered than i (right-neighbourhood)
    template<typename Adj>
    PSet intersect( lVID n, lVID i, lVID ce,
		    const Adj & adj, const lVID * ngh,
		    lVID & ce_new ) const {
	lVID deg = adj.size();
	lVID mx = std::min( deg, ce );
	PSet ins( n, mx+16 ); // hash_vector requires extra space

	if( ce > 2*deg ) {
	    // TODO: find split point for X/P to reduce ranges?
	    //       or make single traversal and determine X/P on the fly?
	    //       need to be careful as m_set may not be sorted, and there
	    //       may exist elements of P that are smaller than some elements
	    //       of X (due to pivoting)
	    ce_new = graptor::hash_vector::intersect(
		ngh, ngh+deg, this->P_hash_set( i ), ins.m_set ) - ins.m_set;
	} else {
	    // Note: skip m_set[i] as we know there are no self-loops.
	    ce_new = graptor::hash_vector::intersect(
		this->m_set+i+1, this->m_set+ce, adj, ins.m_set ) - ins.m_set;
	}

	// Construct ins.m_pos 
	for( lVID i=0; i < ce_new; ++i )
	    ins.m_pos[ins.m_set[i]] = i;

	ins.m_fill = ce_new;

	return ins;
    }

    // Intersect this with adjacency list.
    // Validate all entries in this->m_set.
    // TODO: streamline with dual_set
    template<typename DualSet>
    PSet intersect_validate( lVID n, const DualSet & adj ) const {
	lVID deg = adj.size();
	lVID ce = this->m_fill;
	lVID mx = std::min( deg, ce );
	PSet ins( n, mx+16 ); // hash_vector requires extra space

	// Validation: hash set lookup always performs validation
	//             that m_set and m_pos entries correspond.
	// When looking up values in the adjacency hash set, we need to
	// additionally validate the values are present in the Pset
	auto & adj_h = adj.get_hash();
	auto xp_h = this->hash_set();
	auto validate_h = make_hash_set_and( xp_h, adj_h );
	auto validate_ds = make_dual_set( adj.get_seq(), validate_h );

	lVID ce_new =
	    graptor::set_operations<graptor::adaptive_intersect>::intersect_ds(
		validate_ds, xp_h, ins.m_set ) - ins.m_set;

	// Construct ins.m_pos 
	for( lVID i=0; i < ce_new; ++i )
	    ins.m_pos[ins.m_set[i]] = i;

	ins.m_fill = ce_new;

	return ins;
    }

    // Invalidate an element by making the hash entry point away.
    // Any hash set lookup on the element m_set[i] will fail, however,
    // sequential iteration through m_set will identify the element as
    // valid.
    void invalidate_at( lVID i ) {
	this->m_pos[this->m_set[i]] = this->m_fill;
    }

    void invalidate( lVID v ) {
	this->m_pos[v] = this->m_fill;
    }

    template<typename DualSet>
    static PSet create_complement( lVID n, const DualSet & adj ) {
	// Note: could also do merge-like traversal with jumping to
	//       skip many consecutive elements in adjacency when adjacency
	//       set size is large proportion of n.
	PSet co( n, n - adj.size() );
	lVID k = 0;
	for( lVID i=0; i < n; ++i )
	    if( !adj.contains( i ) ) {
		co.m_pos[i] = k;
		co.m_set[k++] = i;
	    }
	co.m_fill = k;

	return co;
    }

    template<typename DualSet>
    static PSet
    left_union_right( lVID n, lVID v,
		      const DualSet & l_adj, const DualSet & r_adj ) {
	// Determine left and right neighbourhoods
	const lVID * const l_ngh
	    = std::lower_bound( l_adj.begin(), l_adj.end(), v );
	const lVID * const r_ngh
	    = std::upper_bound( r_adj.begin(), r_adj.end(), v );

	// Length of left neighbourhood is an upper bound as we need to
	// intersect with r_adj
	const lVID l_len = std::distance( l_adj.begin(), l_ngh );
	const lVID r_len = std::distance( r_ngh, r_adj.end() );

	PSet lur( n, l_len + r_len + 16 );
	lVID * s = lur.m_set;
	s = graptor::set_operations<graptor::adaptive_intersect>::intersect_ds(
	    l_adj.trim_r( l_ngh ), r_adj, s );
	s = std::copy( r_ngh, r_adj.end(), s );

	// Construct lur.m_pos
	const lVID len = s - lur.m_set;
	assert( len <= l_len + r_len );
	for( lVID i=0; i < len; ++i )
	    lur.m_pos[lur.m_set[i]] = i;
	lur.m_fill = len;

	return lur;
    }

    static PSet create_all( lVID n ) {
	PSet co( n, n );
	for( lVID i=0; i < n; ++i ) {
	    co.m_pos[i] = i;
	    co.m_set[i] = i;
	}
	co.m_fill = n;

	return co;
    }

    template<typename DualSet>
    PSet remove( lVID n, const DualSet & adj ) const {
	PSet rm( n, this->size() );
	lVID k = 0;
	for( lVID i=0; i < this->m_fill; ++i ) {
	    lVID v = this->at( i );
	    if( !adj.contains( v ) ) {
		rm.m_pos[v] = k;
		rm.m_set[k++] = v;
	    }
	}
	rm.m_fill = k;

	return rm;
    }

    // Intersect this with adjacency list, interested only in vertices
    // that are higher-numbered than i (right-neighbourhood), and the
    // lower-numbered vertices that are also neighbours of the pivot.
    // Keep the PSet sorted, if it was sorted.
    template<typename Adj>
    [[deprecated]]
    PSet intersect_pivot( lVID n, lVID i, lVID ce,
		    const Adj & adj, const lVID * ngh,
		    const Adj & p_adj,
		    lVID & ce_new ) const {
	const lVID deg = adj.size();
	const lVID p_deg = p_adj.size();
	const lVID mx = std::min( deg+p_deg, ce ); // at most i left-ngh of pivot
	PSet ins( n, mx+16 ); // hash_vector requires extra space

	// Lower-numbered vertices that are neighbours of the pivot
	lVID * p = ins.m_set;
	if( i > 0 )
	    p = graptor::hash_vector::intersect(
		this->m_set, this->m_set+i-1, p_adj, p );
	
	// Note: skip m_set[i] as we know there are no self-loops.
	hash_set_or<Adj,Adj> both_adj( adj, p_adj );
	p = graptor::hash_vector::intersect(
	    this->m_set+i+1, this->m_set+ce, both_adj, p );

	ce_new = p - ins.m_set;
	assert( ce_new <= mx );

	// Construct ins.m_pos 
	for( lVID i=0; i < ce_new; ++i )
	    ins.m_pos[ins.m_set[i]] = i;

	ins.m_fill = ce_new;

	return ins;
    }


    // Intersect-size PSet with adjacency list.
    // Consider all vertices.
    template<typename DualSet>
    lVID intersect_size( const DualSet & adj ) const {
	return graptor::set_operations<graptor::adaptive_intersect>
	    ::intersect_size_ds( adj, this->P_hash_set( 0 ) );
    }

    // Intersect-size-exceed PSet with adjacency list.
    // Consider all vertices.
    template<typename DualSet>
    lVID intersect_size_exceed( const DualSet & adj, lVID x ) const {
	return graptor::set_operations<graptor::adaptive_intersect>
	    ::intersect_size_exceed_ds( adj, this->P_hash_set( 0 ), x );
    }

    // Intersect-size PSet with adjacency list.
    // Consider all vertices up to position i.
    template<typename DualSet>
    lVID intersect_size_from( const DualSet & set, lVID i ) const {
	return graptor::set_operations<graptor::adaptive_intersect>::intersect_size_ds(
	    set, this->P_hash_set( i+1 ) );
    }

    template<typename HGraphTy>
    static PSet
    create_full_set( const HGraphTy & G ) {
	return PSet( std::move( XPSetBase<lVID>::create_full_set( G ) ) );
    }
};


} // namespace graph

} // namespace graptor

#endif // GRAPHGRIND_GRAPH_SIMPLE_XPSET_H
