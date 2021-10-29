// -*- c++ -*-
#ifndef GRAPTOR_DSL_COMP_REWRITEMASK_H
#define GRAPTOR_DSL_COMP_REWRITEMASK_H

namespace expr {

template<typename M1, typename M2>
auto merge_mask( M1 m1, M2 m2 ) {
    // Expressions containing vk_any are not identical
    if constexpr ( is_identical<M1,M2>::value ) {
	return m1;
    } else {
	return make_land( m1, m2 );
    }
}

template<typename M1, typename M2, typename M3>
auto merge_mask( M1 m1, M2 m2, M3 m3 ) {
    // Expressions containing vk_any are not identical
    if constexpr ( is_identical<M1,M2>::value ) {
	return merge_mask( m1, m3 );
    } else if constexpr ( is_identical<M1,M3>::value ) {
	return merge_mask( m1, m2 );
    } else if constexpr ( is_identical<M2,M3>::value ) {
	return merge_mask( m1, m2 );
    } else {
	return make_land( m1, make_land( m2, m3 ) );
    }
}

template<unsigned short VL, typename A, typename T>
static inline __attribute__((always_inline))
typename std::enable_if<!is_binop_mask<T>::value, refop<A,T,VL>>::type
apply_mask( A array, T index );

template<unsigned short VL, typename A, typename T, typename M>
static inline __attribute__((always_inline))
auto apply_mask( A array, binop<T,M,binop_mask> index );

/***********************************************************************
 * Rewriting to move source/destination masks. These are associated
 * specifically to load/store operations.
 ***********************************************************************/
template<bool LV>
static __attribute__((always_inline)) inline constexpr
auto rewrite_mask( noop n ) {
    return n;
}

template<bool LV, typename Tr, value_kind VKind>
static __attribute__((always_inline)) inline constexpr
auto rewrite_mask( value<Tr, VKind> v ) {
    return v;
}

template<bool LV, unsigned cid, typename Tr>
static __attribute__((always_inline)) inline constexpr
auto rewrite_mask( cacheop<cid,Tr> c ) {
    return c;
}

template<typename Expr, typename UnOp>
static __attribute__((always_inline)) inline constexpr
auto rotate_mask_unop( Expr e, UnOp op ) {
    // Apply unop to expression, as it was originally
    return make_unop( e, op );
}

template<bool LV, typename Expr, typename Mask, typename UnOp>
static __attribute__((always_inline)) inline constexpr
auto rotate_mask_unop( binop<Expr,Mask,binop_mask> e, UnOp op,
		       typename std::enable_if<UnOp::VL == Expr::VL>::value * = nullptr ) {
    // Transform unop(mask(e,m)) to mask(unop(e),m)
    // but only if the UnOp retains the vector length. Otherwise we loose
    // connection between the values and the mask
    // TODO: this is wrong for reductions!
    return add_mask( make_unop( e.data1(), op ), e.data2() );
}

template<bool LV, typename C, typename E1, typename E2, typename TernOp>
static __attribute__((always_inline)) inline constexpr
auto rewrite_mask( ternop<C,E1,E2,TernOp> e ) {
    return make_ternop( rewrite_mask<false>( e.data1() ),
			rewrite_mask<false>( e.data2() ),
			rewrite_mask<false>( e.data3() ),
			TernOp() );
}

template<bool LV, typename C, typename E1, typename Mask1,
	 typename E2, typename Mask2, typename TernOp>
static __attribute__((always_inline)) inline constexpr
auto rewrite_mask( ternop<C,binop<E1,Mask1,binop_mask>,
		   binop<E2,Mask2,binop_mask>,TernOp> e ) {
    // Move mask up over selection.
    return add_mask( make_ternop( e.data1(), e.data2().data1(),
				  e.data3().data1(), TernOp() ),
		     merge_mask( e.data2().data2(), e.data3().data2() ) );
}

template<bool LV, bool nt, typename R, typename T>
static __attribute__((always_inline)) inline constexpr
auto rewrite_mask( storeop<nt,R,T> s ) {
    auto r = rewrite_mask<true>( s.ref() );
    auto t = rewrite_mask<false>( s.value() );
    return make_storeop_like<nt>( r, t );
}

template<bool LV, typename Expr, typename UnOp>
static inline __attribute__((always_inline))
auto rewrite_mask( unop<Expr, UnOp> b ) {
    auto e = rewrite_mask<false>( b.data() );
    // If e is an add_mask operation, then move up over current unop
    // TODO: rewrite with get_mask() and remove_mask()
    return rotate_mask_unop( e, UnOp() );
}

template<typename E1, typename E2, typename BinOp>
static inline __attribute__((always_inline))
auto rotate_mask_binop( E1 e1, E2 e2, BinOp op,
			typename std::enable_if<!is_binop_mask<E1>::value
			&& !is_binop_mask<E1>::value>::type
			* = nullptr ) {
    // Leave op(e1,e2) as is
    return make_binop( e1, e2, BinOp() );
}

template<typename A, typename T, typename M, typename V, unsigned short VL>
static inline __attribute__((always_inline))
auto rotate_mask_binop( maskrefop<A,T,M,VL> r,
			V m,
			binop_mask op ) {
    // return make_maskrefop( r.array(), r.index(), merge_mask( r.mask(), m ) );
    return apply_mask( r.array(),
		       add_mask( r.index(), merge_mask( r.mask(), m ) ) );
}

template<typename E1, typename E2, typename Mask, typename BinOp>
static inline __attribute__((always_inline))
auto
rotate_mask_binop( binop<E1,Mask,binop_mask> e1, E2 e2, BinOp op,
		   typename std::enable_if<!is_binop_mask<E2>::value>::type * = nullptr ) {
    return add_mask( make_binop( e1.data1(), e2, BinOp() ), e1.data2() );
}

template<typename E1, typename E2, typename Mask, typename BinOp>
static inline __attribute__((always_inline))
auto
rotate_mask_binop( E1 e1, binop<E2,Mask,binop_mask> e2, BinOp op,
		   typename std::enable_if<!is_binop_mask<E1>::value>::type * = nullptr ) {
    // Transform op( e1, mask(e2,m) ) to mask( op(e1,e2), m )
    // ??? what if op is binop_mask
    return add_mask( make_binop( e1, e2.data1(), BinOp() ), e2.data2() );
}

template<typename E1, typename E2, typename M1, typename M2, typename BinOp>
static inline __attribute__((always_inline))
auto
rotate_mask_binop( binop<E1,M1,binop_mask> e1,
		   binop<E2,M2,binop_mask> e2, BinOp op,
		   typename std::enable_if<!std::is_same<BinOp,binop_mask>::value>::type * = nullptr ) {
    // Transform op( mask(e1,m1), mask(e2,m2) ) to mask( op(e1,e2), m1&m2 )
    // Require that m1 == m2? That is a runtime property (e.g., m is frontier)
    // ??? what if op is binop_mask
    return add_mask( make_binop( e1.data1(), e2.data1(), BinOp() ),
		     merge_mask( e1.data2(), e2.data2() ) );
}

template<typename E1, typename M1, typename M2>
static inline __attribute__((always_inline))
auto
rotate_mask_binop( binop<E1,M1,binop_mask> e1, M2 e2, binop_mask op,
		   typename std::enable_if<!is_binop_mask<M2>::value>::type * = nullptr ) {
    // Tranform mask( mask(e1,m1), m2 ) to mask( e1, m1 & m2 )
    return add_mask( e1.data1(), merge_mask( e1.data2(), e2 ) );
}

template<typename E1, typename M1, typename E2, typename M2>
static inline __attribute__((always_inline))
auto
rotate_mask_binop( binop<E1,M1,binop_mask> e1,
		   binop<E2,M2,binop_mask> e2, binop_mask op ) {
    // Tranform mask( mask(e1,m1), mask(e2,m2) ) to mask( e1, e2 & m1 & m2 )
    return add_mask( e1.data1(),
		     merge_mask( e2.data1(), e2.data2(), e1.data2() ) );
}

template<bool LV, typename E1, typename E2, typename BinOp>
static inline __attribute__((always_inline))
auto rewrite_mask( binop<E1,E2,BinOp> b ) {
    auto e1 = rewrite_mask<false>( b.data1() );
    auto e2 = rewrite_mask<false>( b.data2() );
    // If e1 and/or e2 is an add_mask operation, then move up over current binop
    return rotate_mask_binop( e1, e2, BinOp() );
}

template<unsigned short VL, typename A, typename T>
static inline __attribute__((always_inline))
typename std::enable_if<!is_binop_mask<T>::value, refop<A,T,VL>>::type
    apply_mask( A array, T index ) {
    return refop<A,T,VL>( array, index );
}

template<unsigned short VL, typename A, typename T, typename M>
static inline __attribute__((always_inline))
auto apply_mask( A array, binop<T,M,binop_mask> index ) {
    auto idx = rewrite_mask<false>( index );
    static_assert( std::is_same_v<typename decltype(idx)::op_type,binop_mask>,
		   "transformation should retain mask at top level" );
    return make_maskrefop( array, idx.data1(), idx.data2() );
}


template<bool LV, typename A, typename T, unsigned short VL>
static inline __attribute__((always_inline))
auto rewrite_mask( refop<A,T,VL> r ) {
    auto i = rewrite_mask<false>( r.index() );
    return apply_mask<VL>( r.array(), i );
}

template<bool LV, typename A, typename T, typename M, unsigned short VL>
static inline constexpr __attribute__((always_inline))
auto rewrite_mask( maskrefop<A,T,M,VL> r ) {
    return r;
}

template<bool LV, typename E1, typename E2, typename RedOp>
static inline __attribute__((always_inline))
auto rewrite_mask( redop<E1,E2,RedOp> r ) {
    auto e1 = rewrite_mask<true>( r.ref() );
    auto e2 = rewrite_mask<false>( r.val() );
    // add_mask returns arg if mask is nomask
    auto m = merge_mask( get_mask( e1 ), get_mask( e2 ) );
    return make_redop( remove_mask( e1 ),
		       add_mask( remove_mask( e2 ), m ),
		       RedOp() );
}

template<bool LV, typename S, typename U, typename C, typename DFSAOp>
static constexpr
auto rewrite_mask( dfsaop<S,U,C,DFSAOp> op ) {
    auto s = rewrite_mask<false>( op.state() );
    auto u = rewrite_mask<false>( op.update() );
    auto c = rewrite_mask<false>( op.condition() );
    return make_dfsaop( s, u, c, DFSAOp() );
}


template<typename Expr>
static inline __attribute__((always_inline))
auto rewrite_mask_main( Expr e ) {
    // auto r = rewrite_mask<false>( e );
    auto r = e; // rewrite_mask<false>( e );
    // Throwing away the top-level mask entirely seems erroneous as it
    // is necessary, e.g., for BFS. Perhaps a sufficient portion of the mask
    // is kept down for other algorihms, but for BFS, we don't have a refop
    // to do that.
    // return remove_mask( r ); // strip the mask in the end
#if 0 // causing issues graph colouring
    static_assert( !is_binop_mask<decltype(r)>::value,
		   "Should not have top-level mask" );
#endif
    return r;
}

} // namespace expr

#endif // GRAPTOR_DSL_COMP_REWRITEMASK_H
