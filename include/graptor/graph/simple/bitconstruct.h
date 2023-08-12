// -*- C++ -*-
#ifndef GRAPHGRIND_GRAPH_SIMPLE_BITCONSTRUCT_H
#define GRAPHGRIND_GRAPH_SIMPLE_BITCONSTRUCT_H

#include "graptor/target/vector.h"
#include "graptor/graph/simple/xp_set.h"

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
	row_u( bitmask_ptr, XP, su >= ne ? 0 : ne, 0, ce );

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
    sVID l = col_start;
    if constexpr ( sizeof(sVID)*8 <= Bits && Bits <= 64
		   && sizeof(sVID) >= 4 && RVL >= 4 ) {
	// When vertex IDs are not wider than the bit mask that needs to be
	// constructed, and the bit masks are at most 64 bits, and the vertex
	// IDs are 32 or 64 bits, then it is possible to perform either
	// 32-bit or 64-bit gather operations on the hash set/table.
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
	    rtype off = rtr::set1inc( l ); // l - off; name confusion
	    rtype mrow = rtr::setzero();
	    rtype ine = rtr::set1( su >= ne ? 0 : ne );
	    while( l+RVL <= col_end ) {
		itype v = itr::loadu( &XP[l] );
		rtype c = rtr::sllv( one, off );

		// TODO: optimise for AVX512F using mask instead of vmask
		// bb is 0xffffffff if element present, 0x0 if not
		itype bb = adj.template multi_contains<sVID,RVL>(
		    v, target::mt_vmask() );
		// Width conversion (if needed)
		rtype br = conversion_traits<sVID,type,RVL>::convert( bb );
		// Remove elements when X-X edges
		rtype b = rtr::bitwise_andnot(
		    rtr::cmplt( off, ine, target::mt_vmask() ), br );
		// Select active lanes (or nullify inactive bitmasks)
		rtype d = rtr::bitwise_and( b, c );

		mrow = rtr::bitwise_or( mrow, d );
		off = rtr::add( off, step );
		l += RVL;
	    }
	    row_u = rtr::reduce_bitwiseor( mrow );
	    deg = _popcnt32( row_u );
	}
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

    return std::make_pair( row_u, deg );
}

template<typename tr, typename GGraph, typename HGraph, typename sVID>
std::pair<typename tr::type,sVID> construct_row_hash_xp_vec(
    const GGraph & G,
    const HGraph & H,
    const XPSet<sVID> & xp_set,
    sVID ne,
    sVID ce,
    sVID su,
    sVID col_start,
    sVID col_end,
    sVID off ) {
    using type = typename tr::member_type;
    using row_type = typename tr::type;
    constexpr size_t Bits = 8 * sizeof(row_type);

    sVID u = xp_set.at( su );
    
    row_type row_u = tr::setzero();
    auto xp_hash = xp_set.hash_table();

    const sVID * ngh = G.get_neighbours( u );
    const sVID udeg = G.getDegree( u );

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
    sVID l = 0;
    if constexpr ( sizeof(sVID)*8 <= Bits && Bits <= 64
		   && sizeof(sVID) >= 4 && RVL >= 4 ) {
	// When vertex IDs are not wider than the bit mask that needs to be
	// constructed, and the bit masks are at most 64 bits, and the vertex
	// IDs are 32 or 64 bits, then it is possible to perform either
	// 32-bit or 64-bit gather operations on the hash set/table.
	if( udeg >= RVL ) {
	    // A vertex identifier is not wider than a row of the matrix.
	    // We can fit multiple row_type into a vector
	    using itr = vector_type_traits_vl<sVID,RVL>;
	    using rtr = vector_type_traits_vl<type,RVL>;
	    using itype = typename itr::type;
	    using rtype = typename rtr::type;
	    rtype one = rtr::setoneval();
	    itype ione = itr::setoneval();
	    rtype step = rtr::slli( one, ilog2( RVL ) );
	    rtype mrow = rtr::setzero();
	    itype ine = itr::set1( su >= ne ? 0 : ne );
	    while( l+RVL <= udeg ) {
		itype v = itr::loadu( &ngh[l] );
		itype bb = xp_hash.template multi_contains<sVID,RVL>( v ); // translate
		// like blend: if v < ine, then invalidate lane, else use bb
		itype b = itr::bitwise_or(
		    itr::cmplt( bb, ine, target::mt_vmask() ), bb );
		// if lane invalid, then shift results in zero mask
		rtype d = rtr::sllv( one, b );
		mrow = rtr::bitwise_or( mrow, d );
		l += RVL;
	    }
	    row_u = rtr::reduce_bitwiseor( mrow );
	    deg = target::allpopcnt<sVID,type,tr::vlen>::compute( row_u );
	}
    } else if constexpr ( sizeof(sVID) >= 4 && Bits == 128 ) {
	// Do vectorized lookup operation, then handle each index one at a time
	// Bitmask restrict to SSE subset
	// Lookup vector length matches total vector width of bitmask vector.
	static constexpr unsigned IVL = sizeof(row_type)*RVL/sizeof(sVID);
	if( udeg >= IVL ) {
	    // A vertex identifier is not wider than a row of the matrix.
	    // We can fit multiple row_type into a vector
	    using itr = vector_type_traits_vl<sVID,IVL>;
	    using rtr = vector_type_traits_vl<type,tr::vlen*RVL>;
	    using itype = typename itr::type;
	    using rtype = typename rtr::type;
	    const itype ione = itr::setoneval();
	    itype step = itr::slli( ione, ilog2( RVL ) );
	    rtype mrow = rtr::setzero();
	    const itype ine = itr::set1( su >= ne ? 0 : ne );
	    //! 4 ... 4 0 ... 0 (if sVID == uint32_t)
	    const itype lno2 = itr::template shuffle<0>( itr::set1inc0() );
	    constexpr unsigned lshift = ilog2( itr::B );
	    // const itype omask = itr::sub( itr::slli( ione, lshift ), ione );
	    const itype omask = itr::srli( itr::setone(), itr::B - lshift );
	    const itype selector = itr::sub( itr::set1inc0(), lno2 );
	    assert( off == 0 );
	    while( l+IVL <= udeg ) {
		itype v = itr::loadu( &ngh[l] );
		itype bb = xp_hash.template multi_contains<sVID,IVL>( v ); // translate
		// like blend: if v < ine, then invalidate lane, else use bb
		itype b = itr::bitwise_or(
		    itr::cmplt( bb, ine, target::mt_vmask() ), bb );

		// To generate bit, split index in lane and offset in lane
		itype bl = itr::srli( b, lshift );
		itype bo = itr::bitwise_and( b, omask );
		itype mo = itr::sllv( ione, bo );

		// handle indices few at a time (one per SSE subvector)
		// for( unsigned li=0; li < IVL; li += RVL ) {// IVL/RVL iterations
		static_assert( IVL/RVL == 4, "unrolling assumption" );
		{
		    // if lane is invalid, its value is 0xffffffff and the
		    // lane number in il will be a large value, which will make
		    // cmpeq fail and ila will be zero across the SSE
		    // subvector that corresponds to invalid lanes

		    // Replicate lowest lane in SSE subvector to all lanes
		    itype il = itr::template shuffle<0>( bl );
		    // Generate relevant 32-bit part of bitmask in each lane
		    itype spec = itr::template shuffle<0>( mo );
		    
		    // only required lane equals _1s
		    itype ila = itr::cmpeq( selector, il, target::mt_vmask() );
		    // Select bitmask from required lane
		    // Following two operations could be one using ternarylogic
		    itype m = itr::bitwise_and( ila, spec );
		    // Merge with accumulator
		    mrow = rtr::bitwise_or( mrow, m );

		    // Shift index to position for next iteration
		    // When unrolling, could remove this and adjust shuffle
		    // at the top of the loop to select appropriate lane
		    // bl = itr::template bsrli<itr::W>( bl );
		    // mo = itr::template bsrli<itr::W>( mo );
		}
		{
		    itype il = itr::template shuffle<0x55>( bl );
		    itype spec = itr::template shuffle<0x55>( mo );
		    itype ila = itr::cmpeq( selector, il, target::mt_vmask() );
		    itype m = itr::bitwise_and( ila, spec );
		    mrow = rtr::bitwise_or( mrow, m );
		}
		{
		    itype il = itr::template shuffle<0xaa>( bl );
		    itype spec = itr::template shuffle<0xaa>( mo );
		    itype ila = itr::cmpeq( selector, il, target::mt_vmask() );
		    itype m = itr::bitwise_and( ila, spec );
		    mrow = rtr::bitwise_or( mrow, m );
		}
		{
		    itype il = itr::template shuffle<0xff>( bl );
		    itype spec = itr::template shuffle<0xff>( mo );
		    itype ila = itr::cmpeq( selector, il, target::mt_vmask() );
		    itype m = itr::bitwise_and( ila, spec );
		    mrow = rtr::bitwise_or( mrow, m );
		}
		
		l += IVL;
	    }
	    for( unsigned ri=0; ri < RVL; ++ri )
		row_u = tr::bitwise_or( row_u, rtr::sse_subvector( mrow, ri ) );
	    deg = target::allpopcnt<sVID,type,tr::vlen>::compute( row_u );
	}
    }
    while( l < udeg ) {
	sVID v = ngh[l];
	sVID sv = xp_hash.contains( v ); // translates ID
	if( sv >= col_start && sv < col_end ) {
	    row_u = tr::bitwise_or(
		row_u, tr::setglobaloneval( sv - off ) );
	    ++deg;
	}
	++l;
    }

    return std::make_pair( row_u, deg );
}


} // namespace graph

} // namespace graptor

#endif // GRAPHGRIND_GRAPH_SIMPLE_BITCONSTRUCT_H
