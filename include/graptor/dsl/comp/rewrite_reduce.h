// -*- c++ -*-
#ifndef GRAPTOR_DSL_REWRITE_REDUCE
#define GRAPTOR_DSL_REWRITE_REDUCE

/***********************************************************************
 * Convert redop with unop_broadcast argument in the ref to a
 * scalar redop with reduction of value. Similar for dfsaop.
 ***********************************************************************/

namespace expr {

namespace rw_reduce {
// Declarations

static constexpr
auto rewrite_reduce( noop n );

template<typename E1, typename E2, typename RedOp>
static constexpr
auto rewrite_reduce( redop<E1,E2,RedOp> r );

template<typename Expr, typename UnOp>
static constexpr
auto rewrite_reduce( unop<Expr,UnOp> u );

template<typename E1, typename E2, typename BinOp>
static constexpr
auto rewrite_reduce( binop<E1,E2,BinOp> b );

template<typename E1, typename E2, typename E3, typename TernOp>
static constexpr
auto rewrite_reduce( ternop<E1,E2,E3,TernOp> t );

template<unsigned cid, typename Tr>
static constexpr
auto rewrite_reduce( cacheop<cid,Tr> c );

template<typename A, typename T, unsigned short VL>
static constexpr
auto rewrite_reduce( refop<A,T,VL> r );

template<typename A, typename T, typename M, unsigned short VL>
static constexpr
auto rewrite_reduce( maskrefop<A,T,M,VL> r );

template<bool nt, typename R, typename T>
static constexpr
auto rewrite_reduce( storeop<nt,R,T> s );

template<typename S, typename U, typename C, typename DFSAOp>
static constexpr
auto rewrite_reduce( dfsaop<S,U,C,DFSAOp> s );

// Implementations

static constexpr
auto rewrite_reduce( noop n ) {
    return n;
}

template<typename Tr, value_kind VKind>
static constexpr
auto rewrite_reduce( value<Tr, VKind> v ) {
    return v;
}

template<typename A, typename T, unsigned short VL>
static constexpr
auto rewrite_reduce( refop<A,T,VL> r ) {
    return make_refop( r.array(), rewrite_reduce( r.index() ) );
}

template<typename A, typename T, typename M, unsigned short VL>
static constexpr
auto rewrite_reduce( maskrefop<A,T,M,VL> r ) {
    return make_maskrefop( r.array(), rewrite_reduce( r.index() ), r.mask() );
}

template<unsigned cid, typename Tr>
static constexpr
auto rewrite_reduce( cacheop<cid,Tr> c ) {
    return c;
}

template<bool nt, typename R, typename T>
static constexpr
auto rewrite_reduce_storeop( storeop<nt,R,T> s ) {
    return make_storeop_like<nt>( rewrite_reduce( s.ref() ),
				  rewrite_reduce( s.value() ) );
}

template<bool nt, typename A, typename T, typename V, unsigned short VL>
static constexpr
auto rewrite_reduce_storeop( storeop<nt,refop<A,unop<T,unop_broadcast<VL>>,VL>,
			     unop<V,unop_broadcast<VL>>> s ) {
    return make_storeop_like<nt>( make_refop( s.ref().array(),
					      rewrite_reduce( s.ref().data() ) ),
				  rewrite_reduce( s.value().data() ) );
#if 0
    return make_storeop( make_refop( s.ref().array(),
				     s.ref().index().data() ), // scalar ref 
			 make_unop_reduce( rewrite_reduce( s.value() ),
					   RedOp: likely redop_logicalor() ) ); // TODO: <--- how to reduce? ---> from cache or accumulator info
#endif
}

template<bool nt, typename R, typename T>
static constexpr
auto rewrite_reduce( storeop<nt,R,T> s ) {
    return rewrite_reduce_storeop(
	make_storeop_like<nt>( rewrite_reduce( s.ref() ),
			       rewrite_reduce( s.value() ) ) );
}

template<typename S, typename U, typename C, typename DFSAOp>
static constexpr
auto rewrite_reduce_dfsaop( S s, U u, C c, DFSAOp op ) {
    return expr::noop(); // make_dfsaop( s, u, c, op );
}

template<typename A, typename T, typename M, typename U,
	 typename C, unsigned short VL, typename DFSAOp>
static constexpr
auto rewrite_reduce_dfsaop( maskrefop<A,unop<T,unop_broadcast<VL>>,M,VL> s,
			    U u, C c, DFSAOp op ) {
    // A scalar reduce operation will call evaluate1() on the DFSAOp to
    // reduce all lanes onto one, after which a store operation follows.
    // What if all lanes are inactive?
    return make_unop_select_mask(
	make_storeop(
	    make_refop( s.array(), s.index().data() ), // scalar ref
	    make_unop_reduce( make_ternop( s, u, c, op ), DFSAOp() )
	    ) );
}

template<typename A, typename T, typename U,
	 typename C, unsigned short VL, typename DFSAOp>
static constexpr
auto rewrite_reduce_dfsaop( refop<A,unop<T,unop_broadcast<VL>>,VL> s,
			    U u, C c, DFSAOp op ) {
    // A scalar reduce operation will call evaluate1() on the DFSAOp to
    // reduce all lanes onto one, after which a store operation follows.
    return make_unop_select_mask(
	make_storeop(
	    make_refop( s.array(), s.index().data() ), // scalar ref
	    make_unop_reduce( make_ternop( s, u, c, op ), DFSAOp() )
	    ) );
}

template<typename S, typename U, typename C, typename DFSAOp>
static constexpr
auto rewrite_reduce( dfsaop<S,U,C,DFSAOp> s ) {
    return rewrite_reduce_dfsaop(
	rewrite_reduce( s.state() ),
	rewrite_reduce( s.update() ),
	rewrite_reduce( s.condition() ),
	DFSAOp() );
}


template<typename E1, typename Tr, typename BinOp>
static constexpr
auto rewrite_reduce_binop(
    unop<E1,unop_incseq<Tr::VL>> u, value<Tr,vk_any> v, BinOp b,
    typename std::enable_if<
    std::is_same<typename BinOp::op_type, binop_mask>::value
    || std::is_same<typename BinOp::op_type, binop_add>::value
    >::type * = nullptr ) {
    // Perform scalar arithmetic, then convert to increasing sequence
    // TODO: restrict to binop_add, binop_mask ??
    using Tr1 = typename Tr::template rebindVL<1>::type;
    return make_unop_incseq<Tr::VL>(
	make_binop( b.data1().data(),
		    value<Tr1,vk_any>( b.data2().data() ),
		    BinOp() ) );
}

template<typename Tr, typename Tr1, typename BinOp>
static constexpr
auto rewrite_reduce_binop(
    value<Tr,vk_any> v,
    unop<value<Tr1,vk_dst>,unop_broadcast<Tr::VL>> u,
    BinOp b,
    typename std::enable_if<std::is_same<
    typename Tr::element_type,
    typename Tr1::element_type>::value
    && Tr1::VL == 1
    && is_binop_arithmetic<BinOp>::value>::type * = nullptr ) {
    return make_unop( make_binop( value<Tr1,vk_any>( v.data() ),
				  value<Tr1,vk_dst>(),
				  BinOp() ),
		      unop_broadcast<Tr::VL>() );
}

template<typename E1, typename E2, typename BinOp>
static constexpr
auto rewrite_reduce_binop( E1 e1, E2 e2, BinOp b ) {
    return make_binop( e1, e2, b );
}

template<typename E1, typename E2, typename BinOp>
static constexpr
auto rewrite_reduce( binop<E1,E2,BinOp> b ) {
    return rewrite_reduce_binop( rewrite_reduce( b.data1() ),
				 rewrite_reduce( b.data2() ),
				 BinOp() );
}

template<typename E1, typename E2, typename E3, typename TernOp>
static constexpr
auto rewrite_reduce( ternop<E1,E2,E3,TernOp> t ) {
    return make_ternop( rewrite_reduce( t.data1() ),
			rewrite_reduce( t.data2() ),
			rewrite_reduce( t.data3() ),
			TernOp() );
}

template<typename Expr, typename UnOp>
static constexpr
auto rewrite_reduce( unop<Expr,UnOp> u ) {
    return make_unop( rewrite_reduce( u.data() ), UnOp() );
}

template<typename E1, typename E2, typename RedOp>
static constexpr
auto rewrite_reduce( redop<E1,E2,RedOp> r ) {
    return make_redop( rewrite_reduce( r.ref() ),
		       rewrite_reduce( r.val() ),
		       RedOp() );
}

template<typename A, typename E1, typename M, typename E2, typename RedOp,
	 unsigned short VL>
static constexpr
auto rewrite_reduce( redop<maskrefop<A,unop<E1,unop_broadcast<VL>>,M,VL>,E2,RedOp> r ) {
    static_assert( VL == E2::VL, "need matching vector lengths" );
    return ( assert( false && "NYI" ), r );
}

template<typename A, typename E1, typename E2, typename RedOp,
	 unsigned short VL>
static constexpr
auto rewrite_reduce( redop<refop<A,unop<E1,unop_broadcast<VL>>,VL>,E2,RedOp> r ) {
    static_assert( VL == E2::VL, "need matching vector lengths" );
    return make_redop( make_refop( r.ref().array(), r.ref().index().data() ), // scalar ref
		       make_unop_reduce( rewrite_reduce( r.val() ), RedOp() ),
		       RedOp() );
}

} // namespace rw_reduce

template<typename Expr>
static constexpr
auto rewrite_reduce( Expr e ) {
    return rw_reduce::rewrite_reduce( e );
}

} // namespace expr

#endif // GRAPTOR_DSL_REWRITE_REDUCE
