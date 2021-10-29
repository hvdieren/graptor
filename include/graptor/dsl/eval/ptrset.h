// -*- c++ -*-
#ifndef GRAPTOR_DSL_PTRSET_H
#define GRAPTOR_DSL_PTRSET_H

#include "graptor/dsl/ast.h"
#include "graptor/dsl/comp/cache.h"

namespace expr {

/**********************************************************************
 * Utilities for removing, what are assumed to be, recurring pointers
 * in multiple arrays in an AST. As the AST is a tree, if an array is
 * read multiple times, then the base address of the array is stored
 * multiple times. The C++ compiler, however, does not know these are
 * the same pointers. This pass collects the unique pointers, which
 * reduces the number of pointers used and register pressure, and creates
 * opportunities for reusing data.
 * The method aid_aliased and struct aid_is_aliased are used to determine
 * if AIDs are aliased. This is based on the AIDs used for the arrays.
 **********************************************************************/

namespace ast_ptrset {

// Decl
template<typename PtrSet, bool nt, typename R, typename T>
static constexpr auto extract_ptrset( storeop<nt,R,T> s, PtrSet && set );

template<typename PtrSet, typename E1, typename E2, typename BinOp>
static constexpr auto extract_ptrset( binop<E1,E2,BinOp> b, PtrSet && set );

template<typename PtrSet, typename E1, typename E2, typename E3,
	 typename TernOp>
static constexpr auto extract_ptrset( ternop<E1,E2,E3,TernOp> b, PtrSet && set );

template<typename PtrSet, unsigned cid, typename Tr>
static constexpr auto extract_ptrset( cacheop<cid,Tr> c, PtrSet && set );

template<typename PtrSet, typename A, typename T, unsigned short VL>
static constexpr auto extract_ptrset( refop<A,T,VL> r, PtrSet && set );

template<typename PtrSet, typename A, typename T, typename M, unsigned short VL>
static constexpr auto extract_ptrset( maskrefop<A,T,M,VL> r, PtrSet && set );

template<typename PtrSet, typename E1, typename E2, typename RedOp>
static constexpr auto extract_ptrset( redop<E1,E2,RedOp> r, PtrSet && set );

template<typename PtrSet, typename S, typename U, typename C, typename DFSAOp>
static constexpr auto extract_ptrset( dfsaop<S,U,C,DFSAOp> s, PtrSet && set );

// Auxiliary: make pointers volatile?
namespace {
#if ENABLE_VOLATILE_PTRS
template<typename T>
std::add_volatile_t<T *> cvt( T * ptr ) {
    return const_cast<std::add_volatile_t<T *>>( ptr );
}
#else
template<typename T>
T * cvt( T * ptr ) {
    return ptr;
}
#endif
} // namespace anonymous

// Impl
    
template<typename PtrSet>
static constexpr
auto extract_ptrset( noop, PtrSet && set ) {
    return set;
}

template<typename PtrSet, typename T, typename U, short AID,
	 typename Enc, bool NT>
static constexpr
auto extract_ptrset( array_ro<T, U, AID, Enc, NT> a, PtrSet && set ) {
    return map_set_if_absent<(unsigned)aid_key(array_aid(AID))>(
	std::forward<PtrSet>( set ), cvt( a.ptr() ) );
}

template<typename PtrSet, typename T, typename U, short AID,
	 typename Enc, bool NT>
static constexpr
auto extract_ptrset( array_intl<T, U, AID, Enc, NT> a, PtrSet && set ) {
    return set;
}

template<typename PtrSet, typename T, typename U, short AID>
static constexpr
auto extract_ptrset( bitarray_intl<T, U, AID> a, PtrSet && set ) {
    return set;
}

template<typename PtrSet, typename T, typename U, short AID>
static constexpr
auto extract_ptrset( bitarray_ro<T, U, AID> a, PtrSet && set ) {
    return map_set_if_absent<(unsigned)aid_key(array_aid(AID))>(
	std::forward<PtrSet>( set ), cvt( a.ptr() ) );
}

template<typename PtrSet, typename Tr, value_kind VKind>
static constexpr
auto extract_ptrset( value<Tr, VKind>, PtrSet && set ) {
    return set;
}

template<typename PtrSet, typename Expr, typename UnOp>
static constexpr
auto extract_ptrset( unop<Expr,UnOp> u, PtrSet && set ) {
    return extract_ptrset( u.data(), std::forward<PtrSet>( set ) );
}

template<typename PtrSet, typename E1, typename E2, typename BinOp>
static constexpr
auto extract_ptrset( binop<E1,E2,BinOp> b, PtrSet && set ) {
    return extract_ptrset( b.data1(), extract_ptrset( b.data2(), std::forward<PtrSet>( set ) ) );
}

template<typename PtrSet, typename E1, typename E2, typename E3,
	 typename TernOp>
static constexpr auto extract_ptrset( ternop<E1,E2,E3,TernOp> e, PtrSet && set ) {
    return extract_ptrset( e.data1(),
			   extract_ptrset( e.data2(),
					   extract_ptrset( e.data3(), std::forward<PtrSet>( set ) ) ) );
}

template<typename PtrSet, typename A, typename T, unsigned short VL>
static constexpr
auto extract_ptrset( refop<A,T,VL> r, PtrSet && set ) {
    return extract_ptrset( r.array(), extract_ptrset( r.index(), std::forward<PtrSet>( set ) ) );
}

template<typename PtrSet, typename A, typename T, typename M, unsigned short VL>
static constexpr
auto extract_ptrset( maskrefop<A,T,M,VL> r, PtrSet && set ) {
    return extract_ptrset( r.array(),
			   extract_ptrset( r.index(),
					   extract_ptrset( r.mask(), std::forward<PtrSet>( set ) ) ) );
}

template<typename PtrSet, bool nt, typename R, typename T>
static constexpr
auto extract_ptrset( storeop<nt,R,T> s, PtrSet && set ) {
    return extract_ptrset( s.ref(), extract_ptrset( s.value(), std::forward<PtrSet>( set ) ) );
}

template<typename PtrSet, unsigned cid, typename Tr>
static constexpr
auto extract_ptrset( cacheop<cid,Tr> c, PtrSet && set ) {
    return set;
}

template<typename PtrSet, typename E1, typename E2, typename RedOp>
static constexpr
auto extract_ptrset( redop<E1,E2,RedOp> r, PtrSet && set ) {
    return extract_ptrset( r.ref(), extract_ptrset( r.val(), std::forward<PtrSet>( set ) ) );
}

template<typename PtrSet, typename S, typename U, typename C, typename DFSAOp>
static constexpr
auto extract_ptrset( dfsaop<S,U,C,DFSAOp> s, PtrSet && set ) {
    return extract_ptrset( s.state(),
			   extract_ptrset( s.update(),
					   extract_ptrset( s.condition(),
							   std::forward<PtrSet>( set ) ) ) );
}

template<typename PtrSet>
static constexpr
auto extract_ptrset( expr::cache<> cache, PtrSet && set ) {
    return set;
}

template<typename PtrSet, typename C, typename... Cs>
static constexpr
auto extract_ptrset( expr::cache<C,Cs...> cache, PtrSet && set ) {
    return extract_ptrset( expr::car( cache ).get_ref(),
			   extract_ptrset( expr::cdr( cache ), std::forward<PtrSet>( set ) ) );
}

} // namespace ast_ptrset

// Also accepting expr::cache<> as argument and will iterate of its references
template<typename AST>
static constexpr
auto extract_pointer_set( AST ast ) {
    return ast_ptrset::extract_ptrset( ast, create_map() );
}

template<typename AST0, typename... AST>
static constexpr
auto extract_pointer_set( AST0 ast0, AST... astn ) {
    return ast_ptrset::extract_ptrset( ast0, extract_pointer_set( astn... ) );
}

template<typename PtrSet, typename AST>
static constexpr
auto extract_pointer_set_with( const PtrSet & ptrset, AST ast ) {
    return ast_ptrset::extract_ptrset( ast, ptrset );
}

template<typename PtrSet, typename AST0, typename... AST>
static constexpr
auto extract_pointer_set_with( const PtrSet & ptrset, AST0 ast0, AST... astn ) {
    return ast_ptrset::extract_ptrset( ast0, extract_pointer_set_with( ptrset, astn... ) );
}

} // namespace expr

#endif // GRAPTOR_DSL_PTRSET_H
