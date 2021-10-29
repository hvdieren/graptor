// -*- c++ -*-
#ifndef GRAPTOR_DSL_REWRITE_REDOP_TO_STORE_H
#define GRAPTOR_DSL_REWRITE_REDOP_TO_STORE_H

#include "graptor/dsl/ast.h"

namespace expr {

/**********************************************************************
 * Utilities for rewriting an AST such that redop is replaced by store,
 * ignoring the currently held value in the redop's memref. A binop_seq
 * is introduced to indicate the change mask, which is always true.
 **********************************************************************/

namespace detail {

// Decl
template<bool nt, typename R, typename T>
static constexpr auto rewrite_redop_to_store( storeop<nt,R,T> s );

template<typename E1, typename E2, typename BinOp>
static constexpr auto rewrite_redop_to_store( binop<E1,E2,BinOp> b );

template<unsigned cid, typename Tr>
static constexpr auto rewrite_redop_to_store( cacheop<cid,Tr> c );

template<typename A, typename T, unsigned short VL>
static constexpr auto rewrite_redop_to_store( refop<A,T,VL> r );

template<typename A, typename T, typename M, unsigned short VL>
static constexpr auto rewrite_redop_to_store( maskrefop<A,T,M,VL> r );

template<typename E1, typename E2, typename RedOp>
static constexpr auto rewrite_redop_to_store( redop<E1,E2,RedOp> r );

// Impl
    
template<unsigned short Lane>
static constexpr
auto rewrite_redop_to_store( noop op ) { // No-op has VL=0
    return op;
}

template<typename T, typename U, short AID, typename Enc, bool NT>
static constexpr
auto rewrite_redop_to_store( array_ro<T, U, AID, Enc, NT> a ) {
    return a;
}

template<typename T, typename U, short AID>
static constexpr
auto rewrite_redop_to_store( bitarray_ro<T, U, AID> a ) {
    return a;
}

template<typename Tr, value_kind VKind>
static constexpr
auto rewrite_redop_to_store( value<Tr, VKind> v ) {
    return v;
}

template<typename Expr, typename UnOp>
static constexpr
auto rewrite_redop_to_store( unop<Expr,UnOp> u ) {
    return make_unop( rewrite_redop_to_store( u.data() ), UnOp() );
}

template<typename E1, typename E2, typename BinOp>
static constexpr
auto rewrite_redop_to_store( binop<E1,E2,BinOp> b ) {
    return make_binop(
	rewrite_redop_to_store( b.data1() ),
	rewrite_redop_to_store( b.data2() ),
	BinOp() );
}

template<typename A, typename T, unsigned short VL>
static constexpr
auto rewrite_redop_to_store( refop<A,T,VL> r ) {
    return make_refop( rewrite_redop_to_store( r.array() ),
		       rewrite_redop_to_store( r.index() ) );
}

template<typename A, typename T, typename M, unsigned short VL>
static constexpr
auto rewrite_redop_to_store( maskrefop<A,T,M,VL> r ) {
    return make_maskrefop( r.array(),
			   rewrite_redop_to_store( r.index() ),
			   rewrite_redop_to_store( r.mask() ) );
}

template<bool nt, typename R, typename T>
static constexpr
auto rewrite_redop_to_store( storeop<nt,R,T> s ) {
    return make_storeop( rewrite_redop_to_store( s.ref() ),
			 rewrite_redop_to_store( s.value() ) );
}

template<unsigned cid, typename Tr>
static constexpr
auto rewrite_redop_to_store( cacheop<cid,Tr> c ) {
    return c;
}

template<typename E1, typename E2, typename RedOp>
static constexpr
auto rewrite_redop_to_store( redop<E1,E2,RedOp> r ) {
    return make_seq(
	make_storeop( rewrite_redop_to_store( r.ref() ),
		      rewrite_redop_to_store( r.val() ) ),
	get_mask( r.val() ) );
}

template<typename S, typename U, typename C, typename DFSAOp>
static constexpr
auto rewrite_redop_to_store( dfsaop<S,U,C,DFSAOp> op ) {
    auto s = rewrite_redop_to_store( op.state() );
    auto u = rewrite_redop_to_store( op.update() );
    auto c = rewrite_redop_to_store( op.condition() );
    return make_dfsaop( s, u, c, DFSAOp() );
}


} // namespace detail

template<typename AST>
static constexpr
auto rewrite_redop_to_store( AST ast ) {
    return detail::rewrite_redop_to_store( ast );
}

} // namespace expr

#endif // GRAPTOR_DSL_REWRITE_REDOP_TO_STORE_H
