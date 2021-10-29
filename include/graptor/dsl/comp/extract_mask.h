// -*- c++ -*-
#ifndef GRAPTOR_DSL_COMP_EXTRACTMASK_H
#define GRAPTOR_DSL_COMP_EXTRACTMASK_H

namespace expr {

namespace detail {

/***********************************************************************
 * Identify what result mask would be applied. Returns an expression
 * for the mask.
 ***********************************************************************/
static __attribute__((always_inline)) inline constexpr
auto extract_mask( noop n ) {
    return n; // wouldn't know vector length/type for a true_val
}

template<typename Tr, value_kind VKind>
static __attribute__((always_inline)) inline constexpr
auto extract_mask( value<Tr, VKind> v ) {
    return value<Tr, vk_true>();
}

template<unsigned cid, typename Tr>
static __attribute__((always_inline)) inline constexpr
auto extract_mask( cacheop<cid,Tr> c ) {
    return value<Tr, vk_true>();
}

template<typename C, typename E1, typename E2, typename TernOp>
static __attribute__((always_inline)) inline constexpr
auto extract_mask( ternop<C,E1,E2,TernOp> e ) {
    return extract_mask( e.data1() )
	&& extract_mask( e.data2() )
	&& extract_mask( e.data3() );
}

template<bool nt, typename R, typename T>
static __attribute__((always_inline)) inline constexpr
auto extract_mask( storeop<nt,R,T> s ) {
    return extract_mask( s.ref() )
	&& extract_mask( s.value() );
}

template<typename Expr, unsigned short VL>
static inline __attribute__((always_inline))
auto extract_mask( unop<Expr, unop_incseq<VL>> b ) {
    auto m = extract_mask( b.data() );
    if constexpr ( std::is_same_v<decltype(m),noop>
		   || is_value_vk<decltype(m),vk_true>::value )
	return value<typename Expr::data_type::template rebindVL<VL>::type, vk_true>();
    else
	return make_unop_incseq<VL>( m );
}

template<typename Expr, typename UnOp>
static inline __attribute__((always_inline))
auto extract_mask( unop<Expr, UnOp> b ) {
    return extract_mask( b.data() );
}

template<typename E1, typename E2>
static inline __attribute__((always_inline))
auto extract_mask( binop<E1,E2,binop_mask> b ) {
    return b.data2();
}

template<typename E1, typename E2, typename BinOp>
static inline __attribute__((always_inline))
auto extract_mask( binop<E1,E2,BinOp> b ) {
    return extract_mask( b.data1() )
	&& extract_mask( b.data2() );
}

template<typename A, typename T, unsigned short VL>
static inline __attribute__((always_inline))
auto extract_mask( refop<A,T,VL> r ) {
    return extract_mask( r.index() );
}

template<typename A, typename T, typename M, unsigned short VL>
static inline __attribute__((always_inline))
auto extract_mask( maskrefop<A,T,M,VL> r ) {
    return extract_mask( r.index() ) && r.mask();
}

template<typename E1, typename E2, typename RedOp>
static inline __attribute__((always_inline))
auto extract_mask( redop<E1,E2,RedOp> r ) {
    return extract_mask( r.ref() )
	&& extract_mask( r.val() );
}

template<typename S, typename U, typename C, typename DFSAOp>
static constexpr
auto extract_mask( dfsaop<S,U,C,DFSAOp> op ) {
    return extract_mask( op.state() )
	&& extract_mask( op.update() )
	&& extract_mask( op.condition() );
}

} // namespace detail;

template<typename Expr>
static inline __attribute__((always_inline))
auto extract_mask( Expr e ) {
    return detail::extract_mask( e );
}

} // namespace expr

#endif // GRAPTOR_DSL_COMP_EXTRACTMASK_H
