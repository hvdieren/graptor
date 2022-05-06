// -*- c++ -*-
#ifndef GRAPTOR_DSL_COMP_REWRITEACCUM_H
#define GRAPTOR_DSL_COMP_REWRITEACCUM_H

namespace expr {

namespace detail {
// Declarations
template<typename E1, typename E2, typename RedOp, typename Cache, typename PIDExpr>
static constexpr
auto rewrite_privatize_accumulators( redop<E1,E2,RedOp> r, Cache & c, PIDExpr pid );

template<typename Expr, typename UnOp, typename Cache, typename PIDExpr>
static constexpr
auto rewrite_privatize_accumulators( unop<Expr,UnOp> u, Cache & c, PIDExpr pid );

template<typename E1, typename E2, typename BinOp, typename Cache, typename PIDExpr>
static constexpr
auto rewrite_privatize_accumulators( binop<E1,E2,BinOp> b, Cache & c, PIDExpr pid );

template<typename E1, typename E2, typename E3, typename TernOp, typename Cache, typename PIDExpr>
static constexpr
auto rewrite_privatize_accumulators( ternop<E1,E2,E3,TernOp> b, Cache & c, PIDExpr pid );

template<typename A, typename T, typename Cache, unsigned short VL, typename Expr>
static constexpr
auto rewrite_privatize_accumulators( refop<A,T,VL> r, Cache & c, Expr pid );

// Implementations
template<typename Cache, typename Expr>
static constexpr
auto rewrite_privatize_accumulators( noop n, Cache &, Expr ) {
    return n;
}

template<typename Tr, value_kind VKind, typename Cache, typename Expr>
static constexpr
auto rewrite_privatize_accumulators( value<Tr, VKind> v, Cache &, Expr ) {
    return v;
}

// Case I: indexing not using constant zero vertex ID (syntax to be revisited)
template<typename A, typename T, typename Cache, unsigned short VL, typename Expr>
static constexpr
auto rewrite_privatize_accumulators_ref(
    refop<A,T,VL> r, Cache & c, Expr pid,
    typename std::enable_if<!cache_contains<A::AID,Cache>::value
    || !is_accumulator<refop<A,T,VL>>::value>::type * = nullptr ) {
    // A let variable in vertexmap looks like an accumulator at this point, but
    // isn't. This would trigger fail_expose incorrectly.
    // fail_expose<is_accumulator>( r );
    return refop<A,T,VL>( r.array(), rewrite_privatize_accumulators( r.index(), c, pid ) );
}

// Case II: the refop has index [0] or [0 w/ mask]
// Rewrite REFOP such that same AID is used but pointer is substituted
// to point to accumulator register and index is no longer vk_zero or
// vk_zero with mask, but is now vk_pid with mask as appropriate.
// TODO: should check whether the mask points to old value of accumulator?
//       If mask points to accum[0], then will be translated, but if the
//       index is accum[constant(0)] or accum[something that may alias to 0]
//       then the translation is incorrect.
template<typename A, typename T, typename Cache, unsigned short VL,
	 typename Expr>
static constexpr
auto rewrite_privatize_accumulators_ref(
    refop<A,T,VL> r, Cache & c, Expr pid,
    typename std::enable_if<cache_contains<A::AID,Cache>::value
    && is_accumulator<refop<A,T,VL>>::value>::type * = nullptr ) {
    auto item = cache_select<A::AID>( c );
    return r.template replace_array<
	cid_to_aid(decltype(item)::cid)>( item.get_accum(), pid );
}

template<bool nt, typename R, typename T, typename Cache, typename PIDExpr>
static constexpr
auto rewrite_privatize_accumulators( storeop<nt,R,T> s, Cache & c, PIDExpr pid ) {
    return make_storeop_like<nt>(
	rewrite_privatize_accumulators_ref( s.ref(), c, pid ),
	rewrite_privatize_accumulators( s.value(), c, pid ) );
}

template<typename A, typename T, typename Cache, unsigned short VL, typename PIDExpr>
static constexpr
auto rewrite_privatize_accumulators( refop<A,T,VL> r, Cache & c, PIDExpr pid ) {
    return rewrite_privatize_accumulators_ref( r, c, pid );
}

template<typename E1, typename E2, typename E3, typename TernOp, typename Cache, typename PIDExpr>
static constexpr
auto rewrite_privatize_accumulators( ternop<E1,E2,E3,TernOp> b, Cache & c, PIDExpr pid ) {
    return make_ternop( rewrite_privatize_accumulators( b.data1(), c, pid ),
			rewrite_privatize_accumulators( b.data2(), c, pid ),
			rewrite_privatize_accumulators( b.data3(), c, pid ),
			TernOp() );
}


template<typename E1, typename E2, typename BinOp, typename Cache, typename PIDExpr>
static constexpr
auto rewrite_privatize_accumulators( binop<E1,E2,BinOp> b, Cache & c, PIDExpr pid ) {
    return make_binop( rewrite_privatize_accumulators( b.data1(), c, pid ),
		       rewrite_privatize_accumulators( b.data2(), c, pid ),
		       BinOp() );
}

template<typename Expr, typename UnOp, typename Cache, typename PIDExpr>
static constexpr
auto rewrite_privatize_accumulators( unop<Expr,UnOp> u, Cache & c, PIDExpr pid ) {
    return make_unop( rewrite_privatize_accumulators( u.data(), c, pid ), UnOp() );
}

template<typename E1, typename E2, typename RedOp, typename Cache, typename PIDExpr>
constexpr
auto rewrite_privatize_accumulators( redop<E1,E2,RedOp> r, Cache & c, PIDExpr pid ) {
    return make_redop( rewrite_privatize_accumulators( r.ref(), c, pid ),
		       rewrite_privatize_accumulators( r.val(), c, pid ),
		       RedOp() );
}

} // namespace detail

template<typename Expr, typename PIDExpr>
static constexpr
Expr rewrite_privatize_accumulators( Expr e, const partitioner & part,
				     cache<> & c, PIDExpr p ) {
    return e;
}

template<typename Expr, typename Cache, typename PIDExpr>
static constexpr
auto rewrite_privatize_accumulators(
    Expr e, const partitioner & part, Cache & c, PIDExpr p,
    typename std::enable_if<!std::is_same<Cache,cache<>>::value>::type * = nullptr ) {
    accum_create( part, c );
    return detail::rewrite_privatize_accumulators( e, c, p );
}

} // namespace expr

#endif // GRAPTOR_DSL_COMP_REWRITEACCUM_H
