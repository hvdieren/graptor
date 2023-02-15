// -*- c++ -*-
#ifndef GRAPTOR_DSL_TRANSFORM_NT_H
#define GRAPTOR_DSL_TRANSFORM_NT_H

#include "graptor/dsl/ast.h"

namespace expr {

/**********************************************************************
 * Utilities for rewriting an AST such that only one vector lane
 * is processe.
 **********************************************************************/

namespace detail {

// Decl
template<bool NT, bool nt, typename R, typename T>
static constexpr auto transform_nt( storeop<nt,R,T> s );

template<bool NT, typename E1, typename E2, typename BinOp>
static constexpr auto transform_nt( binop<E1,E2,BinOp> b );

template<bool NT, unsigned cid, typename Tr, short aid, cacheop_flags flags>
static constexpr auto transform_nt( cacheop<cid,Tr,aid,flags> c );

template<bool NT, typename A, typename T, unsigned short VL>
static constexpr auto transform_nt( refop<A,T,VL> r );

template<bool NT, typename A, typename T, typename M, unsigned short VL>
static constexpr auto transform_nt( maskrefop<A,T,M,VL> r );

template<bool NT, typename E1, typename E2, typename RedOp>
static constexpr auto transform_nt( redop<E1,E2,RedOp> r );

// Impl
    
template<unsigned short Lane>
static constexpr
auto transform_nt( noop op ) { // No-op has VL=0
    return op;
}

template<bool NT, typename T, typename U, short AID, typename Enc, bool NT_>
static constexpr
auto transform_nt( array_ro<T, U, AID, Enc, NT_> a ) { // Array_ro has no VL
    return a.template rebindNT<NT>();
}

template<bool NT, typename T, typename U, short AID>
static constexpr
auto transform_nt( bitarray_ro<T, U, AID> a ) { // Bitarray_ro has no VL
    // return a.template rebindNT<NT>();
    return a;
}

template<bool NT, typename Tr, value_kind VKind>
static constexpr
auto transform_nt( value<Tr, VKind> v ) {
    return v;
}

template<bool NT, typename Expr, typename UnOp>
static constexpr
auto transform_nt( unop<Expr,UnOp> u ) {
    return make_unop( transform_nt<NT>( u.data() ), UnOp() );
}

template<bool NT, typename E1, typename E2, typename BinOp>
static constexpr
auto transform_nt( binop<E1,E2,BinOp> b ) {
    return make_binop(
	transform_nt<NT>( b.data1() ),
	transform_nt<NT>( b.data2() ),
	BinOp() );
}

template<bool NT, typename A, typename T, unsigned short VL>
static constexpr
auto transform_nt( refop<A,T,VL> r ) {
    return make_refop( transform_nt<NT>( r.array() ),
		       transform_nt<NT>( r.index() ) );
}

template<bool NT, typename A, typename T, typename M, unsigned short VL>
static constexpr
auto transform_nt( maskrefop<A,T,M,VL> r ) {
    return make_maskrefop( r.array(),
			   transform_nt<NT>( r.index() ),
			   transform_nt<NT>( r.mask() ) );
}

template<bool NT, bool nt, typename R, typename T>
static constexpr
auto transform_nt( storeop<nt,R,T> s ) {
    return make_storeop( transform_nt<NT>( s.ref() ),
			 transform_nt<NT>( s.value() ) );
}

template<bool NT, unsigned cid, typename Tr, short aid, cacheop_flags flags>
static constexpr auto transform_nt( cacheop<cid,Tr,aid,flags> c ) {
    return c;
}

template<bool NT, typename E1, typename E2, typename RedOp>
static constexpr
auto transform_nt( redop<E1,E2,RedOp> r ) {
    return make_redop( transform_nt<NT>( r.ref() ),
		       transform_nt<NT>( r.val() ), RedOp() );
}

template<bool NT, typename S, typename U, typename C, typename DFSAOp>
static constexpr
auto transform_nt( dfsaop<S,U,C,DFSAOp> op ) {
    auto s = transform_nt<NT>( op.state() );
    auto u = transform_nt<NT>( op.update() );
    auto c = transform_nt<NT>( op.condition() );
    return make_dfsaop( s, u, c, DFSAOp() );
}


} // namespace detail

template<bool NT, typename AST>
static constexpr
auto transform_nt( AST ast ) {
    return detail::transform_nt<NT>( ast );
}

} // namespace expr

#endif // GRAPTOR_DSL_TRANSFORM_NT_H
