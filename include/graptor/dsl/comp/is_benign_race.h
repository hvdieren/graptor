// -*- c++ -*-
#ifndef GRAPTOR_DSL_COMP_IS_BENIGN_RACE_H
#define GRAPTOR_DSL_COMP_IS_BENIGN_RACE_H

#include "graptor/dsl/ast.h"

namespace expr {

/**********************************************************************
 * Utility for checking if an expression would result in benign race
 * conditions if the modified memory location(s) were subject to
 * concurrent updates.
 *
 * Any number of updates are allowed. All updates must be benign for
 * the expression to have benign races (only).
 *
 * Old behaviour:
 * Store operations are not benign
 * races as we do not know anything about the value being stored.
 * Only for specific redop operations will data races be benign.
 * New behaviour:
 * Stores are not benign, except when the stored value is known constant.
 * In that case, it does not matter which store executes.
 **********************************************************************/

namespace detail {

// Decl
template<typename Expr>
struct is_benign_race;

// Impl
    
template<>
struct is_benign_race<noop> {
    static constexpr bool value = true;
};

template<typename T, typename U, short AID, typename Enc, bool NT>
struct is_benign_race<array_intl<T, U, AID, Enc, NT>> {
    static constexpr bool value = true;
};

template<typename T, typename U, short AID, typename Enc, bool NT>
struct is_benign_race<array_ro<T, U, AID, Enc, NT>> {
    static constexpr bool value = true;
};

template<typename T, typename U, short AID>
struct is_benign_race<bitarray_ro<T, U, AID>> {
    static constexpr bool value = true;
};

template<typename Tr, value_kind VKind>
struct is_benign_race<value<Tr, VKind>> {
    static constexpr bool value = true;
};

template<typename Expr, typename UnOp>
struct is_benign_race<unop<Expr,UnOp>> : public is_benign_race<Expr> { };


template<typename E1, typename E2, typename BinOp>
struct is_benign_race<binop<E1,E2,BinOp>> {
    static constexpr bool value =
	is_benign_race<E1>::value && is_benign_race<E2>::value;
};

template<typename A, typename T, unsigned short VL>
struct is_benign_race<refop<A,T,VL>> {
    static constexpr bool value =
	is_benign_race<A>::value && is_benign_race<T>::value;
};

template<typename A, typename T, typename M, unsigned short VL>
struct is_benign_race<maskrefop<A,T,M,VL>> {
    static constexpr bool value =
	is_benign_race<A>::value && is_benign_race<T>::value
	&& is_benign_race<M>::value;
};

template<bool nt, typename R, typename T>
struct is_benign_race<storeop<nt,R,T>> {
    // benign only if stored value is constant
    static constexpr bool value = is_constant_expr<T>::value;
};

template<unsigned cid, typename Tr>
struct is_benign_race<cacheop<cid,Tr>> {
    static constexpr bool value = true;
};

template<typename E1, typename E2, typename RedOp>
struct is_benign_race<redop<E1,E2,RedOp>> {
    static constexpr bool value =
	is_benign_race<E1>::value && is_benign_race<E2>::value
	&& RedOp::is_benign_race;
};

template<typename C, typename E1, typename E2, typename TernOp>
struct is_benign_race<ternop<C,E1,E2,TernOp>> {
    static constexpr bool value =
	is_benign_race<C>::value && is_benign_race<E1>::value
	&& is_benign_race<E2>::value;
};

template<typename S, typename U, typename C, typename DFSAOp>
struct is_benign_race<dfsaop<S,U,C,DFSAOp>> {
    static constexpr bool value = false; // NOT benign
};

} // namespace detail

template<typename Expr>
struct is_benign_race {
    using D = detail::is_benign_race<std::decay_t<Expr>>;
    static constexpr bool value = D::value;
};

template<typename Operator>
struct is_benign_race_op {
    static constexpr bool value = is_benign_race<
	decltype(
	    ((Operator*)nullptr)
	    ->relax( expr::value<simd::ty<VID,1>,expr::vk_src>(),
		     expr::value<simd::ty<VID,1>,expr::vk_dst>(),
		     expr::value<simd::ty<EID,1>,expr::vk_edge>() ))>
	::value;
};

} // namespace expr

#endif // GRAPTOR_DSL_COMP_IS_BENIGN_RACE_H
