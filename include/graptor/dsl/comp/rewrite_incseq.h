// -*- c++ -*-
#ifndef GRAPTOR_DSL_REWRITE_INCSEQ
#define GRAPTOR_DSL_REWRITE_INCSEQ

/***********************************************************************
 * Try to convert vector load with index unop_incseq( scalar ) to
 * vector load with scalar index
 ***********************************************************************/

namespace expr {

namespace detail {
// Declarations
template<typename E1, typename E2, typename RedOp>
static constexpr
auto rewrite_incseq( redop<E1,E2,RedOp> r );

template<typename Expr, typename UnOp>
static constexpr
auto rewrite_incseq( unop<Expr,UnOp> u );

template<typename E1, typename Tr, typename BinOp>
static constexpr
auto rewrite_incseq( binop<unop<E1,unop_incseq<Tr::VL>>,value<Tr,vk_any>,BinOp> b );

template<typename E1, typename E2, typename BinOp>
static constexpr
auto rewrite_incseq( binop<E1,E2,BinOp> b,
		     typename std::enable_if<!is_unop_incseq<E1>::value
		     || !is_value_vk<E2,vk_any>::value>::type * = nullptr );

template<typename A, typename T, unsigned short VL>
static constexpr
auto rewrite_incseq( refop<A,T,VL> r );

template<typename A, typename T, typename M, unsigned short VL>
static constexpr
auto rewrite_incseq( maskrefop<A,T,M,VL> r );

// Implementations
template<typename Cache>
static constexpr
auto rewrite_incseq( noop n ) {
    return n;
}

template<typename Tr, value_kind VKind>
static constexpr
auto rewrite_incseq( value<Tr, VKind> v ) {
    return v;
}

// Case I: indexing unsing unop_incseq - cut out the unop
template<typename A, typename T, unsigned short VL>
static constexpr
auto rewrite_incseq_ref( refop<A,unop<T,unop_incseq<VL>>,VL> r ) {
    return refop<A,T,VL>( r.array(), r.index().data() );
}

// Case II: other indexing
template<typename A, typename T, unsigned short VL>
static constexpr
auto rewrite_incseq_ref(
    refop<A,T,VL> r,
    typename std::enable_if<!is_unop_incseq<T>::value>::type * = nullptr ) {
    return make_refop( r.array(), rewrite_incseq( r.index() ) );
}

template<typename A, typename T, unsigned short VL>
static constexpr
auto rewrite_incseq( refop<A,T,VL> r ) {
    return rewrite_incseq_ref(
	make_refop( r.array(), rewrite_incseq( r.index() ) ) );
}

// Case I: indexing unsing unop_incseq - cut out the unop
template<typename A, typename T, typename M, unsigned short VL>
static constexpr
auto rewrite_incseq_ref( maskrefop<A,unop<T,unop_incseq<VL>>,M,VL> r ) {
    return maskrefop<A,T,M,VL>( r.array(), r.index().data(), r.mask() );
}

// Case II: other indexing
template<typename A, typename T, typename M, unsigned short VL>
static constexpr
auto rewrite_incseq_ref(
    maskrefop<A,T,M,VL> r,
    typename std::enable_if<!is_unop_incseq<T>::value>::type * = nullptr ) {
    return make_maskrefop( r.array(), rewrite_incseq( r.index() ), r.mask() );
}


template<typename A, typename T, typename M, unsigned short VL>
static constexpr
auto rewrite_incseq( maskrefop<A,T,M,VL> r ) {
    return rewrite_incseq_ref(
	make_maskrefop( r.array(), rewrite_incseq( r.index() ), r.mask() ) );
}

template<unsigned cid, typename Tr>
static constexpr
auto rewrite_incseq( cacheop<cid,Tr> c ) {
    return c;
}


template<bool nt, typename R, typename T>
static constexpr
auto rewrite_incseq( storeop<nt,R,T> s ) {
    return make_storeop_like<nt>( rewrite_incseq( s.ref() ),
				  rewrite_incseq( s.value() ) );
}

template<typename E1, typename Tr, typename BinOp>
static constexpr
auto rewrite_incseq( binop<unop<E1,unop_incseq<Tr::VL>>,value<Tr,vk_any>,BinOp> b ) {
    // Perform scalar arithmetic, then convert to increasing sequence
    using Tr1 = typename Tr::template rebindVL<1>::type;
    return make_unop_incseq<Tr::VL>(
	make_binop( b.data1().data(),
		    value<Tr1,vk_any>( b.data2().data() ),
		    BinOp() ) );
}

template<typename E1, typename E2, typename BinOp>
static constexpr
auto rewrite_incseq( binop<E1,E2,BinOp> b,
		     typename std::enable_if<!is_unop_incseq<E1>::value
		     || !is_value_vk<E2,vk_any>::value>::type * ) {
    return make_binop( rewrite_incseq( b.data1() ),
		       rewrite_incseq( b.data2() ),
		       BinOp() );
}

template<typename Expr, typename UnOp>
static constexpr
auto rewrite_incseq( unop<Expr,UnOp> u ) {
    return make_unop( rewrite_incseq( u.data() ), UnOp() );
}

template<typename E1, typename E2, typename RedOp>
constexpr
auto rewrite_incseq( redop<E1,E2,RedOp> r ) {
    return make_redop( rewrite_incseq( r.ref() ),
		       rewrite_incseq( r.val() ),
		       RedOp() );
}
} // namespace detail

template<typename Expr>
static constexpr
auto rewrite_incseq( Expr e ) {
    return detail::rewrite_incseq( e );
}

} // namespace expr

#endif // GRAPTOR_DSL_REWRITE_INCSEQ
