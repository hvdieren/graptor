// -*- c++ -*-
#ifndef GRAPTOR_DSL_REWRITE_SCALARIZE
#define GRAPTOR_DSL_REWRITE_SCALARIZE

/***********************************************************************
 * This is a highly specific code transformation for the csc_vreduce_ngh
 * edgemap variant. This operation is called in a context where code is
 * generated for some vector length, and a cache is created at that
 * vector length, but actually this specific part of the computation
 * should be scalar.
 *
 * vcaches is a base cache which describes a subset of the cached values
 * that have already been calculated at vector length VL. Other cached
 * values related to scan operations across all vertices. These values
 * should be reduced to a scalar when accessed. The remainder
 * of the computation should be performed in a scalar manner although
 * values should be deposited in the vectorized cache, in any lane.
 *
 * The translation works as follows:
 * - rvalue cacheop<cid,Tr> is replaced by unop_reduce( cacheop<cid,Tr> )
 *   using the reduction operator in the context where the cacheop
 *   originally appeared. This holds if cid is in vcaches
 * - lvalue cacheop<cid,Tr> has unop_setl0 applied to the rvalue (either
 *   in a storeop or redop).
 *   rule: if cid in vcaches -> only used as rvalue;
 *         else -> only used as lvalue
 * - refop with ref not a cacheop: fully transform to scalar (scalar store)
 * - all refop and arithmetic is transformed from vector length VL
 *   to scalar (VL=1)
 ***********************************************************************/

namespace expr {

namespace detail {
// Declarations
template<typename Cache>
static constexpr
auto rewrite_scalarize( noop n, const Cache & c );

template<typename Tr, value_kind VKind, typename Cache>
static constexpr
auto rewrite_scalarize( value<Tr, VKind> v, const Cache & c );

template<unsigned cid, typename Tr, typename Cache>
auto rewrite_scalarize( cacheop<cid, Tr> o, const Cache & c,
			typename std::enable_if<cache_contains_cid<cid,Cache>::value>::type * = nullptr );

template<typename Expr, typename UnOp, typename Cache>
static constexpr
auto rewrite_scalarize(
    unop<Expr,UnOp> u, const Cache & c,
    std::enable_if_t<!unop_is_broadcast<UnOp>::value
    && !unop_is_reduce<UnOp>::value
    && !unop_is_incseq<UnOp>::value> * = nullptr );

template<typename E1, typename E2, typename BinOp, typename Cache>
static constexpr
auto rewrite_scalarize( binop<E1,E2,BinOp> b, const Cache & c );

template<bool nt, unsigned cid, typename Tr, typename U,
	 typename Cache>
static constexpr
auto rewrite_scalarize( storeop<nt,cacheop<cid,Tr>,U> r, const Cache & c,
			typename std::enable_if<cache_contains_cid<cid,Cache>::value>::type * = nullptr );

template<bool nt, typename R, typename T, typename Cache>
static constexpr
auto rewrite_scalarize( storeop<nt,R,T> s, const Cache & c,
			typename std::enable_if<!is_cacheop<R>::value>::type * = nullptr );

template<typename A, typename T, unsigned short VL, typename Cache>
static constexpr
auto rewrite_scalarize( refop<A,T,VL> r, const Cache & c,
			typename std::enable_if<!is_cacheop<A>::value>::type * = nullptr );

// Case II: not-Cache-cached cacheop used as lvalue
template<unsigned cid, typename Tr,
	 typename E2, typename RedOp, typename Cache>
constexpr
auto rewrite_scalarize( redop<cacheop<cid,Tr>,E2,RedOp> r, const Cache & c,
			typename std::enable_if<!cache_contains_cid<cid,Cache>::value>::type * = nullptr );

template<typename E1, typename E2, typename RedOp, typename Cache>
constexpr
auto rewrite_scalarize( redop<E1,E2,RedOp> r, const Cache & c,
			typename std::enable_if<!is_cacheop<E1>::value>::type * = nullptr );


// Implementations
template<typename Cache>
static constexpr
auto rewrite_scalarize( noop n, const Cache & c ) {
    return n;
}

template<typename Tr, value_kind VKind, typename Cache>
static constexpr
auto rewrite_scalarize( value<Tr, VKind> v, const Cache & c ) {
    // Values only hold a scalar value
    using Tr1 = typename Tr::template rebindVL<1>::type;
    if constexpr( value_kind_has_value<VKind>::value ) {
	return value<Tr1,VKind>( v.data() );
    } else {
	return value<Tr1,VKind>();
    }
}

template<unsigned cid, typename Tr, typename Cache>
auto rewrite_scalarize( cacheop<cid, Tr> o, const Cache & c,
			typename std::enable_if<cache_contains_cid<cid,Cache>::value>::type * ) {
    // R-value usage of cacheop
    // return c.template get<cid>().get_reduce_expr( c );
    return cache_select_cid<cid>( c ).template get_reduce_expr<Tr::VL>( c ) ;
}

// Case I: Cache-cached cacheop used as lvalue - error
/*
template<unsigned cid, typename T, typename U, unsigned short VL,
	 typename Cache>
static constexpr
void rewrite_scalarize( storeop<cacheop<cid,T,VL>,U> r, const Cache & c,
			typename std::enable_if<!cache_contains_cid<cid,Cache>::value>::type * = nullptr ) {
    static_assert( 0, "Error: not expecting that this cacheop "
		   "is used as an lvalue" );
}
*/

// Case II: not-Cache-cached cacheop used as lvalue
template<unsigned cid, bool nt, typename Tr, typename U,
	 typename Cache>
static constexpr
auto rewrite_scalarize( storeop<nt,cacheop<cid,Tr>,U> r, const Cache & c,
			typename std::enable_if<cache_contains_cid<cid,Cache>::value>::type * ) {
    return make_storeop_like<nt>(
	r.ref(), make_unop_setl0<Tr>( rewrite_scalarize( r.value(), c ) ) );
}

template<bool nt, typename R, typename T, typename Cache>
static constexpr
auto rewrite_scalarize( storeop<nt,R,T> r, const Cache & c,
			typename std::enable_if<!is_cacheop<R>::value>::type * ) {
    return make_storeop_like<nt>(
	rewrite_scalarize( r.ref(), c ),
	rewrite_scalarize( r.value(), c ) );
}

template<typename A, typename T, unsigned short VL, typename Cache>
static constexpr
auto rewrite_scalarize( refop<A,T,VL> r, const Cache & c,
			typename std::enable_if<!is_cacheop<A>::value>::type * ) {
    return make_refop( r.array(),
		       rewrite_scalarize( r.index(), c ) );
}

template<typename A, typename T, typename M, unsigned short VL, typename Cache>
static constexpr
auto rewrite_scalarize( maskrefop<A,T,M,VL> r, const Cache & c ) {
    return make_maskrefop( r.array(),
			   rewrite_scalarize( r.index(), c ),
			   rewrite_scalarize( r.mask(), c ) );
}

/*
template<unsigned cid, typename T, unsigned short VL, typename Cache>
static constexpr
auto rewrite_scalarize( cacheop<cid,T,VL> c, const Cache & c ) {
    static_assert( 0, "should not be called" );
}
*/


template<typename E1, typename E2, typename BinOp, typename Cache>
static constexpr
auto rewrite_scalarize( binop<E1,E2,BinOp> b, const Cache & c ) {
    return make_binop( rewrite_scalarize( b.data1(), c ),
		       rewrite_scalarize( b.data2(), c ),
		       BinOp() );
}

template<typename Expr, typename UnOp, typename Cache>
static constexpr
auto rewrite_scalarize(
    unop<Expr,UnOp> u, const Cache & c,
    std::enable_if_t<!unop_is_broadcast<UnOp>::value
    && !unop_is_reduce<UnOp>::value
    && !unop_is_incseq<UnOp>::value> * ) {
    using UnOp1 = typename UnOp::template rebindVL<1>;
    return make_unop( rewrite_scalarize( u.data(), c ), UnOp1() );
}

template<typename Expr, unsigned short VL, typename Cache>
static constexpr
auto rewrite_scalarize( unop<Expr,unop_broadcast<VL>> u, const Cache & c ) {
    return rewrite_scalarize( u.data(), c );
}

template<typename Expr, typename RedOp, typename Cache>
static constexpr
auto rewrite_scalarize( unop<Expr,unop_reduce<RedOp>> u, const Cache & c ) {
    return rewrite_scalarize( u.data(), c );
}

template<typename Expr, unsigned short VL, typename Cache>
static constexpr
auto rewrite_scalarize( unop<Expr,unop_incseq<VL>> u, const Cache & c ) {
    return rewrite_scalarize( u.data(), c );
}

// Case I: Cache-cached cacheop used as lvalue - error
/*
template<unsigned cid, typename T, unsigned short VL,
	 typename E2, typename RedOp, typename Cache>
constexpr
void rewrite_scalarize( redop<cacheop<cid,T,VL>,E2,RedOp> r, const Cache & c,
			typename std::enable_if<cache_contains_cid<cid,Cache>::value>::type * ) {
    static_assert( 0, "Error: not expecting that this cacheop "
		   "is used as an lvalue" );
}
*/

// Case II: not-Cache-cached cacheop used as lvalue
template<unsigned cid, typename Tr,
	 typename E2, typename RedOp, typename Cache>
constexpr
auto rewrite_scalarize( redop<cacheop<cid,Tr>,E2,RedOp> r, const Cache & c,
			typename std::enable_if<!cache_contains_cid<cid,Cache>::value>::type * ) {
    return make_redop( r.ref(),
		       make_unop_setl0<Tr>( rewrite_scalarize( r.val(), c ) ),
		       RedOp() );
}

template<typename E1, typename E2, typename RedOp, typename Cache>
constexpr
auto rewrite_scalarize( redop<E1,E2,RedOp> r, const Cache & c,
			typename std::enable_if<!is_cacheop<E1>::value>::type * ) {
    return make_redop( rewrite_scalarize( r.ref(), c ),
		       rewrite_scalarize( r.val(), c ),
		       RedOp() );
}

} // namespace detail


template<typename Expr, typename Cache>
static constexpr
auto rewrite_scalarize( Expr e, const Cache & c ) {
    return detail::rewrite_scalarize( e, c );
}

} // namespace expr

#endif // GRAPTOR_DSL_REWRITE_SCALARIZE
