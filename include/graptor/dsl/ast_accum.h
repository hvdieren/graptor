// -*- c++ -*-
#ifndef GRAPTOR_DSL_AST_ACCUM_H
#define GRAPTOR_DSL_AST_ACCUM_H

#include "graptor/dsl/ast.h"

namespace expr {

namespace detail {
template<typename Expr, typename Accum>
struct reads_accum;

template<typename Tr, value_kind VKind, typename Accum>
struct reads_accum<value<Tr,VKind>,Accum> : public std::false_type { };

template<typename Expr, typename UnOp, typename Accum>
struct reads_accum<unop<Expr,UnOp>,Accum> : public reads_accum<Expr,Accum> { };

template<typename E1, typename E2, typename BinOp, typename Accum>
struct reads_accum<binop<E1,E2,BinOp>,Accum> {
    static constexpr bool value = reads_accum<E1,Accum>::value
	|| reads_accum<E2,Accum>::value; 
};
    
template<typename A, typename T, unsigned short VL, typename Accum>
struct reads_accum<refop<A,T,VL>,Accum> {
    static constexpr bool value = reads_accum<A,Accum>::value
	|| reads_accum<T,Accum>::value;
};

template<typename T, typename U, short AID, typename Enc, bool NT,
	 typename Accum>
struct reads_accum<array_ro<T,U,AID,Enc,NT>,Accum> {
    static constexpr bool value = accum_updates<AID,Accum>::value;
};

template<typename E1, typename E2, typename RedOp, typename Accum>
struct reads_accum<redop<E1,E2,RedOp>,Accum> {
    static constexpr bool value = reads_accum<E1,Accum>::value
	|| reads_accum<E2,Accum>::value;
};

template<bool nt, typename R, typename T, typename Accum>
struct reads_accum<storeop<nt,R,T>,Accum> {
    static constexpr bool value = reads_accum<T,Accum>::value
	|| reads_accum<R,Accum>::value;
};

} // namespace detail

template<typename Expr, typename Accum>
using reads_accum = detail::reads_accum<Expr,Accum>;

} // namespace expr


#endif // GRAPTOR_DSL_AST_ACCUM_H
