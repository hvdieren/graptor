// -*- C++ -*-
#ifndef GRAPHGRIND_GRAPH_SIMPLE_BITCONSTRUCT_H
#define GRAPHGRIND_GRAPH_SIMPLE_BITCONSTRUCT_H

#include "graptor/target/vector.h"

namespace graptor {

namespace graph {

template<typename tr, typename GGraph, typename HGraph, typename sVID>
std::pair<typename tr::type,sVID> construct_row_hash_xp(
    const GGraph & G,
    const HGraph & H,
    const typename HGraph::hash_set_type & XP_hash,
    const sVID * XP,
    sVID ne,
    sVID ce,
    sVID su,
    sVID u,
    sVID col_start,
    sVID col_end,
    sVID off ) {
    using row_type = typename tr::type;

    const sVID * n = G.get_neighbours( u );
    sVID udeg = G.getDegree( u );
    sVID deg = 0;
    row_type row = tr::setzero();
    const sVID * n_start
	= su < ne
	? std::lower_bound( n, n+udeg, XP[ne] ) : n;
    for( const sVID * i=n_start; i != n+udeg; ++i ) {
	sVID v = *i;
	if( XP_hash.contains( v ) ) {
	    const VID * pos = std::lower_bound( XP, XP+ne, v );
	    if( *pos != v )
		pos = std::lower_bound( XP+ne, XP+ce, v );
	    sVID sv = pos - XP;
	    if( sv >= col_start && sv < col_end ) {
		row = tr::bitwise_or( row, tr::setglobaloneval( sv - off ) );
		deg++;
	    }
	}
    }
    return std::make_pair( row, deg );
}

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

    void push_back( const lVID * p, const lVID * = nullptr ) {
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
		    const lVID * base,
		    const lVID * = nullptr ) {
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
	    vns = tr::sub( vns, vsp ); // -- vns unused further down
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


template<typename tr, typename GGraph, typename HGraph, typename sVID>
sVID construct_row_hash_adj(
    const GGraph & G,
    const HGraph & H,
    typename tr::member_type * bitmask_ptr,
    const sVID * XP,
    sVID ne,
    sVID ce,
    sVID su,
    sVID col_start,
    sVID col_end ) {
    using type = typename tr::member_type;
    using row_type = typename tr::type;
    constexpr size_t Bits = 8 * sizeof(row_type);

#if __AVX512F__
	constexpr VID VL = 64 / sizeof(VID);
#elif __AVX2__
	constexpr VID VL = 32 / sizeof(VID);
#else
	constexpr VID VL = 1;
#endif

    VID u = XP[su];

    bitmask_lhs_sorted_output_iterator<type, VID, true, false>
	row_u( bitmask_ptr, XP, ne, su >= ne ? 0 : ne );

    // Trim off vertices that will be filtered out, but keep alignment.
    const VID * const XP_start
	= su >= ne ? XP
	: XP + ( ne & ~( VL - 1 ) );

    row_u = graptor::hash_vector::template intersect<true>(
	XP_start, XP+ce, H.get_adjacency( u ), row_u );

    return row_u.get_degree();
}

template<typename tr, typename GGraph, typename HGraph, typename sVID>
sVID construct_row_hash_adj(
    const GGraph & G,
    const HGraph & H,
    typename tr::member_type * bitmask_ptr,
    const sVID * XP,
    sVID ne,
    sVID ce,
    sVID su,
    sVID col_start,
    sVID col_end,
    sVID off ) {
    using type = typename tr::member_type;
    using row_type = typename tr::type;
    constexpr size_t Bits = 8 * sizeof(row_type);

    sVID u = XP[su];

    bitmask_lhs_sorted_output_iterator<type, sVID, true, true>
	row_u( bitmask_ptr, XP, col_start, col_start, col_end );

    row_u = graptor::hash_vector::template intersect<true>(
	XP+col_start, XP+ce, H.get_adjacency( u ), row_u );
	    
    return row_u.get_degree();
}

template<typename tr, typename GGraph, typename HGraph, typename sVID>
std::pair<typename tr::type,sVID> construct_row_hash_adj_vec(
    const GGraph & G,
    const HGraph & H,
    const sVID * XP,
    sVID ne,
    sVID ce,
    sVID su,
    sVID col_start,
    sVID col_end,
    sVID off ) {
    using type = typename tr::member_type;
    using row_type = typename tr::type;
    constexpr size_t Bits = 8 * sizeof(row_type);

    sVID u = XP[su];
    
    row_type row_u = tr::setzero();
    auto & adj = H.get_adjacency( u );

    sVID deg = 0;

    // Intersect XP with adjacency list
#if __AVX512F__
    static constexpr unsigned RVL = 512/Bits;
#elif __AVX2__
    static constexpr unsigned RVL = 256/Bits;
#elif __SSE42__
    static constexpr unsigned RVL = 128/Bits;
#else
    static constexpr unsigned RVL = 1;
#endif
    if constexpr ( sizeof(sVID)*8 == Bits && RVL >= 4 && false ) {
	sVID l = col_start;
	if( col_end - col_start >= RVL ) {
	    // A vertex identifier is not wider than a row of the matrix.
	    // We can fit multiple row_type into a vector
	    using itr = vector_type_traits_vl<sVID,RVL>;
	    using rtr = vector_type_traits_vl<type,RVL>;
	    using itype = typename itr::type;
	    using rtype = typename rtr::type;
	    rtype one = rtr::setoneval();
	    itype ione = itr::setoneval();
	    rtype step = rtr::slli( one, ilog2( RVL ) );
	    rtype off = rtr::set1inc0();
	    rtype mrow = rtr::setzero();
	    itype mdeg = itr::setzero();
	    while( l+RVL <= col_end ) {
		itype v = itr::loadu( &XP[l] );
		rtype c = rtr::sllv( one, off );
#if __AVX512F__
		// using bitmask
		auto b = adj.template multi_contains<sVID,RVL>( v, target::mt_mask() );
		rtype d = rtr::blend( b, rtr::setzero(), c );
		mdeg = itr::blend( b, mdeg, itr::add( mdeg, ione ) );
#elif __AVX2__
		itype b = adj.template multi_contains<sVID,RVL>( v, target::mt_vmask() );
		rtype br = conversion_traits<logical<sizeof(sVID)>,logical<sizeof(type)>,RVL>::convert( b );
		rtype d = rtr::bitwise_and( br, c );
		mdeg = itr::add( mdeg, itr::bitwise_and( b, ione ) ); // 4-argument add( mdeg, b, ione, mdeg );
#else
		assert( 0 && "NYI" );
#endif
		mrow = rtr::bitwise_or( mrow, d );
		off = rtr::add( off, step );
		l += RVL;
	    }
	    row_u = rtr::reduce_bitwiseor( mrow );
	    deg = itr::reduce_add( mdeg );
	}
	while( l < col_end ) {
	    sVID xp = XP[l];
	    if( adj.contains( xp ) ) {
		row_u = tr::bitwise_or(
		    row_u, tr::setglobaloneval( l - off ) );
		++deg;
	    }
	    ++l;
	}
    } else {
	for( sVID l=col_start; l < col_end; ++l ) {
	    sVID xp = XP[l];
	    if( adj.contains( xp ) ) {
		row_u = tr::bitwise_or(
		    row_u, tr::setglobaloneval( l - off ) );
		++deg;
	    }
	}
    }

    return std::make_pair( row_u, deg );
}

} // namespace graph

} // namespace graptor

#endif // GRAPHGRIND_GRAPH_SIMPLE_BITCONSTRUCT_H
