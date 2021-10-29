// -*- c++ -*-
#ifndef GRAPTOR_DSL_EXTRACTLANE_H
#define GRAPTOR_DSL_EXTRACTLANE_H

#include "graptor/dsl/ast.h"

namespace expr {

/**********************************************************************
 * Utilities for rewriting an AST such that only one vector lane
 * is processe.
 **********************************************************************/

namespace detail {

// Decl
template<unsigned short Lane, bool nt, typename R, typename T>
static constexpr auto extract_lane( storeop<nt,R,T> s );

template<unsigned short Lane, typename E1, typename E2, typename BinOp>
static constexpr auto extract_lane( binop<E1,E2,BinOp> b );

template<unsigned short Lane, unsigned cid, typename Tr>
static constexpr auto extract_lane( cacheop<cid,Tr> c );

template<unsigned short Lane, typename A, typename T, unsigned short VL>
static constexpr auto extract_lane( refop<A,T,VL> r );

template<unsigned short Lane, typename A, typename T, typename M, unsigned short VL>
static constexpr auto extract_lane( maskrefop<A,T,M,VL> r );

template<unsigned short Lane, typename E1, typename E2, typename RedOp>
static constexpr auto extract_lane( redop<E1,E2,RedOp> r );

// Impl
    
template<unsigned short Lane>
static constexpr
auto extract_lane( noop op ) { // No-op has VL=0
    return op;
}

template<unsigned short Lane, typename T, typename U, short AID,
	 typename Enc, bool NT>
static constexpr
auto extract_lane( array_ro<T, U, AID, Enc, NT> a ) { // Array_ro has no VL
    return a;
}

template<unsigned short Lane, typename Tr, value_kind VKind>
static constexpr
auto extract_lane( value<Tr, VKind> v ) {
    using Tr1 = typename Tr::template rebindVL<1>::type;
    return value<Tr1, VKind>();
}

template<unsigned short Lane, typename Tr>
static constexpr
auto extract_lane( value<Tr, vk_any> v ) {
    using Tr1 = typename Tr::template rebindVL<1>::type;
    return value<Tr1, vk_any>( v.data() );
}

template<unsigned short Lane, typename Expr, typename UnOp>
static constexpr
auto extract_lane( unop<Expr,UnOp> u ) {
    return make_unop( extract_lane<Lane>( u.data() ), UnOp() );
}

template<unsigned short Lane, typename Expr, unsigned short VL>
static constexpr
auto extract_lane( unop<Expr,unop_broadcast<VL>> u ) {
    return extract_lane<Lane>( u.data() );
}

template<unsigned short Lane, typename E1, typename E2, typename BinOp>
static constexpr
auto extract_lane( binop<E1,E2,BinOp> b ) {
    return make_binop(
	extract_lane<Lane>( b.data1() ),
	extract_lane<Lane>( b.data2() ),
	BinOp() );
}

template<unsigned short Lane, typename A, typename T, unsigned short VL>
static constexpr
auto extract_lane( refop<A,T,VL> r ) {
    return make_refop( r.array(), extract_lane<Lane>( r.index() ) );
}

template<unsigned short Lane, typename A, typename T, typename M, unsigned short VL>
static constexpr
auto extract_lane( maskrefop<A,T,M,VL> r ) {
    return make_maskrefop( r.array(),
			   extract_lane<Lane>( r.index() ),
			   extract_lane<Lane>( r.mask() ) );
}

template<unsigned short Lane, bool nt, typename R, typename T>
static constexpr
auto extract_lane( storeop<nt,R,T> s ) {
    return make_storeop( extract_lane<Lane>( s.ref() ),
			 extract_lane<Lane>( s.value() ) );
}

template<unsigned short Lane, unsigned cid, typename Tr>
static constexpr
auto extract_lane( cacheop<cid,Tr> c ) {
    // return cacheop<cid,T,1>();
    return cacheop<cid,typename Tr::template rebindVL<1>::type>();
}

template<unsigned short Lane, typename E1, typename E2, typename RedOp>
static constexpr
auto extract_lane( redop<E1,E2,RedOp> r ) {
    return make_redop( extract_lane<Lane>( r.ref() ),
		       extract_lane<Lane>( r.val() ) );
}

} // namespace detail

template<unsigned short Lane, typename AST>
static constexpr
auto extract_lane( AST ast ) {
    return detail::extract_lane<Lane>( ast );
}

} // namespace expr

#endif // GRAPTOR_DSL_EXTRACTLANE_H
