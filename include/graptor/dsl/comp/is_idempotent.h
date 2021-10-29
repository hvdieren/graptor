// -*- c++ -*-
#ifndef GRAPTOR_DSL_COMP_IS_IDEMPOTENT_H
#define GRAPTOR_DSL_COMP_IS_IDEMPOTENT_H

#include "graptor/dsl/ast.h"

namespace expr {

/**********************************************************************
 * Utility for checking if an expression is idempotent.
 * An expression is idempotent if executing it multiple times on exactly
 * the same values (src, dst, mask, ...) yields the same values in memory
 * and in cacheops.
 *
 * We use a simple rule that accepts at most one modified memory location
 * or cacheop per expression, and the modification is done by an
 * idempotent redop.
 * -- currently any number of idempotent redop's
 *
 * The difficulty with considering multiple updates in an idempotent 
 * expression is that the result of one expression may feed into another
 * and we need more careful checks that the update of the first does not
 * affect the other. On other words, it is easier to check if all values
 * read from memory are guaranteed to be the same in each execution.
 **********************************************************************/

namespace detail {

// Decl
template<typename Expr>
struct is_idempotent;

// Impl
    
template<>
struct is_idempotent<noop> {
    static constexpr unsigned char seen_mod = 0;
    static constexpr bool value = true;
};

template<typename T, typename U, short AID, typename Enc, bool NT>
struct is_idempotent<array_intl<T, U, AID, Enc, NT>> {
    static constexpr unsigned char seen_mod = 0;
    static constexpr bool value = true;
};

template<typename T, typename U, short AID, typename Enc, bool NT>
struct is_idempotent<array_ro<T, U, AID, Enc, NT>> {
    static constexpr unsigned char seen_mod = 0;
    static constexpr bool value = true;
};

template<typename T, typename U, short AID>
struct is_idempotent<bitarray_ro<T, U, AID>> {
    static constexpr unsigned char seen_mod = 0;
    static constexpr bool value = true;
};

template<typename Tr, value_kind VKind>
struct is_idempotent<value<Tr, VKind>> {
    static constexpr unsigned char seen_mod = 0;
    static constexpr bool value = true;
};

template<typename Expr, typename UnOp>
struct is_idempotent<unop<Expr,UnOp>> : public is_idempotent<Expr> { };


template<typename E1, typename E2, typename BinOp>
struct is_idempotent<binop<E1,E2,BinOp>> {
    static constexpr unsigned char seen_mod =
	is_idempotent<E1>::seen_mod + is_idempotent<E2>::seen_mod;
    static constexpr bool value =
	is_idempotent<E1>::value && is_idempotent<E2>::value;
};

template<typename A, typename T, unsigned short VL>
struct is_idempotent<refop<A,T,VL>> {
    static constexpr unsigned char seen_mod =
	is_idempotent<A>::seen_mod + is_idempotent<T>::seen_mod;
    static constexpr bool value =
	is_idempotent<A>::value && is_idempotent<T>::value;
};

template<typename A, typename T, typename M, unsigned short VL>
struct is_idempotent<maskrefop<A,T,M,VL>> {
    static constexpr unsigned char seen_mod =
	is_idempotent<A>::seen_mod + is_idempotent<T>::seen_mod
	+ is_idempotent<M>::seen_mod;
    static constexpr bool value =
	is_idempotent<A>::value && is_idempotent<T>::value
	&& is_idempotent<M>::value;
};

template<bool nt, typename R, typename T>
struct is_idempotent<storeop<nt,R,T>> {
    static constexpr unsigned char seen_mod =
	is_idempotent<R>::seen_mod + is_idempotent<T>::seen_mod + 1;
    static constexpr bool value = false;
};

template<unsigned cid, typename Tr>
struct is_idempotent<cacheop<cid,Tr>> {
    static constexpr unsigned char seen_mod = 0;
    static constexpr bool value = true;
};

template<typename E1, typename E2, typename RedOp>
struct is_idempotent<redop<E1,E2,RedOp>> {
    static constexpr unsigned char seen_mod =
	is_idempotent<E1>::seen_mod + is_idempotent<E2>::seen_mod + 1;
    static constexpr bool value =
	is_idempotent<E1>::value && is_idempotent<E2>::value
	&& RedOp::is_idempotent;
};

template<typename C, typename E1, typename E2, typename TernOp>
struct is_idempotent<ternop<C,E1,E2,TernOp>> {
    static constexpr unsigned char seen_mod =
	is_idempotent<C>::seen_mod + is_idempotent<E1>::seen_mod
	+ is_idempotent<E2>::seen_mod;
    static constexpr bool value =
	is_idempotent<C>::value && is_idempotent<E1>::value
	&& is_idempotent<E2>::value;
};

template<typename S, typename U, typename C, typename DFSAOp>
struct is_idempotent<dfsaop<S,U,C,DFSAOp>> {
    static constexpr unsigned char seen_mod =
	is_idempotent<S>::seen_mod + is_idempotent<U>::seen_mod
	+ is_idempotent<C>::seen_mod;
    static constexpr bool value =
	is_idempotent<S>::value && is_idempotent<U>::value
	&& is_idempotent<C>::value;
};

} // namespace detail

template<typename Expr>
struct is_idempotent {
    using D = detail::is_idempotent<std::decay_t<Expr>>;
    static constexpr bool value = D::value && D::seen_mod <= 1;
};

} // namespace expr

#endif // GRAPTOR_DSL_COMP_IS_IDEMPOTENT_H
