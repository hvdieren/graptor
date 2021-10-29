// -*- c++ -*-
#ifndef GRAPTOR_DSL_COMP_REWRITE_INTERNAL_H
#define GRAPTOR_DSL_COMP_REWRITE_INTERNAL_H

#include "graptor/dsl/ast.h"

namespace expr {

/**********************************************************************
 * Utilities for rewriting an AST for a more efficient implementation.
 * - Every occurence of array_ro is replaced by array_intl. The latter
 *   is stateless. The goal is to avoid run-time copying of the
 *   expressions, which would be unnecessary if they are stateless.
 **********************************************************************/

namespace detail {

// Decl
template<bool NT, typename R, typename T>
static constexpr auto rewrite_internal( storeop<NT,R,T> s );

template<typename E1, typename E2, typename BinOp>
static constexpr auto rewrite_internal( binop<E1,E2,BinOp> b );

template<unsigned cid, typename Tr>
static constexpr auto rewrite_internal( cacheop<cid,Tr> c );

template<typename A, typename T, unsigned short VL>
static constexpr auto rewrite_internal( refop<A,T,VL> r );

template<typename A, typename T, typename M, unsigned short VL>
static constexpr auto rewrite_internal( maskrefop<A,T,M,VL> r );

template<typename E1, typename E2, typename RedOp>
static constexpr auto rewrite_internal( redop<E1,E2,RedOp> r );

// Impl
    
static constexpr
auto rewrite_internal( noop op ) {
    return op;
}

template<typename T, typename U, short AID, typename Enc, bool NT>
static constexpr
auto rewrite_internal( array_ro<T, U, AID, Enc, NT> a ) {
    return array_intl<T,U,AID,Enc,NT>(); // stateless version
}

template<typename T, typename U, short AID, typename Enc, bool NT>
static constexpr
auto rewrite_internal( array_intl<T, U, AID, Enc, NT> a ) {
    return a;
}

template<typename T, typename U, short AID>
static constexpr
auto rewrite_internal( bitarray_ro<T, U, AID> a ) {
    return bitarray_intl<T,U,AID>(); // stateless version
}

template<typename T, typename U, short AID>
static constexpr
auto rewrite_internal( bitarray_intl<T, U, AID> a ) {
    return a;
}

template<typename Tr, value_kind VKind>
static constexpr
auto rewrite_internal( value<Tr, VKind> v ) {
    // TODO: deal with vk_any, the only type of value with state.
    return v;
}

template<typename Expr, typename UnOp>
static constexpr
auto rewrite_internal( unop<Expr,UnOp> u ) {
    return make_unop( rewrite_internal( u.data() ), UnOp() );
}

template<typename E1, typename E2, typename BinOp>
static constexpr
auto rewrite_internal( binop<E1,E2,BinOp> b ) {
    return make_binop(
	rewrite_internal( b.data1() ),
	rewrite_internal( b.data2() ),
	BinOp() );
}

template<typename A, typename T, unsigned short VL>
static constexpr
auto rewrite_internal( refop<A,T,VL> r ) {
    return make_refop( rewrite_internal( r.array() ),
		       rewrite_internal( r.index() ) );
}

template<typename A, typename T, typename M, unsigned short VL>
static constexpr
auto rewrite_internal( maskrefop<A,T,M,VL> r ) {
    return make_maskrefop( r.array(),
			   rewrite_internal( r.index() ),
			   rewrite_internal( r.mask() ) );
}

template<bool nt, typename R, typename T>
static constexpr
auto rewrite_internal( storeop<nt,R,T> s ) {
    return make_storeop( rewrite_internal( s.ref() ),
			 rewrite_internal( s.value() ) );
}

template<unsigned cid, typename Tr>
static constexpr
auto rewrite_internal( cacheop<cid,Tr> c ) {
    return c;
}

template<typename E1, typename E2, typename RedOp>
static constexpr
auto rewrite_internal( redop<E1,E2,RedOp> r ) {
    return make_redop( rewrite_internal( r.ref() ),
		       rewrite_internal( r.val() ), RedOp() );
}

template<typename C, typename E1, typename E2, typename TernOp>
static constexpr
auto rewrite_internal( ternop<C,E1,E2,TernOp> e ) {
    return make_ternop( rewrite_internal( e.data1() ),
			rewrite_internal( e.data2() ),
			rewrite_internal( e.data3() ),
			TernOp() );
}

template<typename S, typename U, typename C, typename DFSAOp>
static constexpr
auto rewrite_internal( dfsaop<S,U,C,DFSAOp> op ) {
    auto s = rewrite_internal( op.state() );
    auto u = rewrite_internal( op.update() );
    auto c = rewrite_internal( op.condition() );
    return make_dfsaop( s, u, c, DFSAOp() );
}


} // namespace detail

template<typename AST>
static constexpr
auto rewrite_internal( AST ast ) {
    return detail::rewrite_internal( ast );
}

} // namespace expr

#endif // GRAPTOR_DSL_COMP_REWRITE_INTERNAL_H
