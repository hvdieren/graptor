// -*- c++ -*-
#ifndef GRAPTOR_DSL_INSERT_MASK_H
#define GRAPTOR_DSL_INSERT_MASK_H

#include "graptor/dsl/ast.h"

namespace expr {

/**********************************************************************
 * Utilities for adding a mask to a value in an AST.
 **********************************************************************/

namespace detail {

// Decl
template<value_kind VKind, typename Mask, bool nt, typename R, typename T>
static constexpr auto insert_mask( storeop<nt,R,T> s, Mask mask );

template<value_kind VKind, typename Mask, typename Expr, unsigned short VL>
static constexpr auto insert_mask( unop<Expr,unop_incseq<VL>> u, Mask mask );

template<value_kind VKind, typename Mask, typename Expr, typename UnOp>
static constexpr auto insert_mask( unop<Expr,UnOp> u, Mask mask );

template<value_kind VKind, typename Mask, typename E1, typename E2, typename BinOp>
static constexpr auto insert_mask( binop<E1,E2,BinOp> b, Mask mask );

template<value_kind VKind, typename Mask,
	 unsigned cid, typename Tr, short aid, cacheop_flags flags>
static constexpr auto insert_mask( cacheop<cid,Tr,aid,flags> c, Mask mask );

template<value_kind VKind, typename Mask, typename A, typename T, unsigned short VL>
static constexpr auto insert_mask( refop<A,T,VL> r, Mask mask );

template<value_kind VKind, typename Mask, typename A, typename T, typename M, unsigned short VL>
static constexpr auto insert_mask( maskrefop<A,T,M,VL> r, Mask mask );

template<value_kind VKind, typename Mask, typename E1, typename E2, typename RedOp>
static constexpr auto insert_mask( redop<E1,E2,RedOp> r, Mask mask );

// Impl
    
template<value_kind VKind, typename Mask>
static constexpr
auto insert_mask( noop op, Mask mask ) {
    return op;
}

template<value_kind VKind, typename Mask, typename T, typename U, short AID,
	 typename Enc, bool NT_>
static constexpr
auto insert_mask( array_ro<T, U, AID, Enc, NT_> a, Mask mask ) {
    return a;
}

template<value_kind VKind, typename Mask, typename Tr, value_kind VKind_>
static constexpr
auto insert_mask( value<Tr, VKind_> v, Mask mask ) {
    if constexpr ( VKind == VKind_ ) {
	return add_mask( v, mask );
    } else {
	return v;
    }
}

template<value_kind VKind, typename Mask, typename Expr, typename UnOp>
static constexpr
auto insert_mask( unop<Expr,UnOp> u, Mask mask ) {
    return make_unop( insert_mask<VKind>( u.data(), mask ), UnOp() );
}

template<value_kind VKind, typename Mask, typename Expr, unsigned short VL>
static constexpr
auto insert_mask( unop<Expr,unop_incseq<VL>> u, Mask mask ) {
    return add_mask( u, mask );
}

template<value_kind VKind, typename Mask, typename E1, typename E2, typename BinOp>
static constexpr
auto insert_mask( binop<E1,E2,BinOp> b, Mask mask ) {
    return make_binop(
	insert_mask<VKind>( b.data1(), mask ),
	insert_mask<VKind>( b.data2(), mask ),
	BinOp() );
}

template<value_kind VKind, typename Mask, typename A, typename T, unsigned short VL>
static constexpr
auto insert_mask( refop<A,T,VL> r, Mask mask ) {
    return make_refop( insert_mask<VKind>( r.array(), mask ),
		       insert_mask<VKind>( r.index(), mask ) );
}

template<value_kind VKind, typename Mask, typename A, typename T, typename M, unsigned short VL>
static constexpr
auto insert_mask( maskrefop<A,T,M,VL> r, Mask mask ) {
    return make_maskrefop( r.array(),
			   insert_mask<VKind>( r.index(), mask ),
			   insert_mask<VKind>( r.mask(), mask ) );
}

template<value_kind VKind, typename Mask, bool nt, typename R, typename T>
static constexpr
auto insert_mask( storeop<nt,R,T> s, Mask mask ) {
    return make_storeop( insert_mask<VKind>( s.ref(), mask ),
			 insert_mask<VKind>( s.value(), mask ) );
}

template<value_kind VKind, typename Mask,
	 unsigned cid, typename Tr, short aid, cacheop_flags flags>
static constexpr
auto insert_mask( cacheop<cid,Tr,aid,flags> c, Mask mask ) {
    return c;
}

template<value_kind VKind, typename Mask, typename E1, typename E2, typename RedOp>
static constexpr
auto insert_mask( redop<E1,E2,RedOp> r, Mask mask ) {
    return make_redop( insert_mask<VKind>( r.ref(), mask ),
		       insert_mask<VKind>( r.val(), mask ), RedOp() );
}

} // namespace detail

template<value_kind VKind, typename AST, typename Mask>
static constexpr
auto insert_mask( AST ast, Mask mask ) {
    return detail::insert_mask<VKind>( ast, mask );
}

} // namespace expr

#endif // GRAPTOR_DSL_INSERT_MASK_H
