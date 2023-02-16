// -*- c++ -*-
#ifndef GRAPTOR_DSL_AST_H
#define GRAPTOR_DSL_AST_H

#include <type_traits>
#include <algorithm>
#include <tuple>
#include <utility>
#include <memory>

#include "graptor/utils.h"
#include "graptor/frontier.h"
#include "graptor/dsl/simd_vector.h"

#include "graptor/dsl/aval.h"

#include "graptor/dsl/ast/decl.h"
#include "graptor/dsl/ast/value.h"
#include "graptor/dsl/ast/constant.h"
#include "graptor/dsl/ast/unop.h"
#include "graptor/dsl/ast/binop.h"
#include "graptor/dsl/ast/ternop.h"
#include "graptor/dsl/ast/memref.h"
#include "graptor/dsl/ast/redop.h"
#include "graptor/dsl/ast/dfsaop.h"
#include "graptor/dsl/ast/ctrl.h"
// #include "graptor/dsl/ast/random.h"

#include "graptor/dsl/value_map.h"

#include "graptor/dsl/comp/frontier_select.h"
    
namespace expr {

/**********************************************************************
 * Pre-declarations
 **********************************************************************/
template<typename value_map_type, typename Cache>
struct printer;

template<typename... T>
struct cache;

template<typename value_map_type, typename Cache,
	 typename array_map_type, typename Accum = cache<>,
	 bool AtomicUpdate = false>
struct evaluator;

} // namespace expr

#include "graptor/dsl/comp/extract_lane.h"
#include "graptor/dsl/comp/cache.h"

#include "graptor/dsl/comp/accum.h"
#include "graptor/dsl/comp/licm.h"

#include "graptor/dsl/comp/rewrite_accum.h"
#include "graptor/dsl/comp/rewrite_mask.h"
#include "graptor/dsl/comp/rewrite_incseq.h"
#include "graptor/dsl/comp/rewrite_reduce.h"
#include "graptor/dsl/comp/rewrite_scalarize.h"

#include "graptor/dsl/eval/ptrset.h"
#include "graptor/dsl/eval/evaluator.h"

namespace expr {

#if 0 // - appears unused 
/***********************************************************************
 * Calculate vector length of the value of an expression
 ***********************************************************************/
template<typename Expr>
struct vector_length;

template<typename Tr, value_kind VKind>
struct vector_length<value<Tr, VKind>> {
    static constexpr unsigned short value = Tr::VL;
};

template<unsigned cid, typename Tr, array_aid aid, cacheop_flags flags>
struct vector_length<cacheop<cid,Tr,aid,flags>> {
    static constexpr unsigned short value = Tr::VL;
};

template<typename T, typename R>
struct vector_length<storeop<T,R>> {
    static_assert( vector_length<T>::value == vector_length<R>::value,
		   "vector length of index and value must match" );
    static constexpr unsigned short value = vector_length<T>::value;
};
    
template<typename E, typename UnOp>
struct vector_length<unop<E,UnOp>> {
    static constexpr unsigned short value = UnOp::VL;
};

template<typename E1, typename E2, typename BinOp>
struct vector_length<binop<E1,E2,BinOp>> {
    // Assume if e1 != e2, then one of e1 or e2 is 1
    static constexpr unsigned short value
	= std::max( vector_length<typename binop<E1,E2,BinOp>::left_type>::value,
		    vector_length<typename binop<E1,E2,BinOp>::right_type>::value );
};

template<typename A, typename T, unsigned short VL>
struct vector_length<refop<A,T,VL>> {
    static constexpr unsigned short value
	= vector_length<typename refop<A,T,VL>::idx_type>::value;
};

template<typename A, typename T, typename M, unsigned short VL>
struct vector_length<maskrefop<A,T,M,VL>> {
    static_assert( vector_length<T>::value == vector_length<M>::value,
		   "vector length of index and mask must match" );
    static constexpr unsigned short value = vector_length<T>::value;
};

template<typename E1, typename E2, typename RedOp>
struct vector_length<redop<E1,E2,RedOp>> {
    static constexpr unsigned short value
	= std::max( vector_length<E1>::value, vector_length<E2>::value );
};

#endif // 0 - appears unused 


/***********************************************************************
 * Rewriting to match vector lengths: insert replications or reductions.
 ***********************************************************************/
#if 0
static constexpr
GG_INLINE auto rewrite_vectors( noop n ) {
    return n;
}

template<typename Tr, value_kind VKind>
static constexpr
GG_INLINE auto rewrite_vectors( value<Tr, VKind> v ) {
    return v;
}

template<unsigned cid, typename Tr>
static constexpr
GG_INLINE auto rewrite_vectors( cacheop<cid,Tr> c ) {
    return c;
}

template<typename T, typename R>
static constexpr
GG_INLINE auto rewrite_vectors( storeop<T,R> s ) {
    return make_storeop( rewrite_vectors( s.ref() ),
			 rewrite_vectors( s.value() ) );
}

template<typename Expr, typename UnOp>
static constexpr
GG_INLINE auto rewrite_vectors( unop<Expr,UnOp> u ) {
    // NYI: just keep as is -- should adjust vector length in UnOp type?
    return make_unop( rewrite_vectors( u.data() ), UnOp() );
}

template<typename E1, typename E2, typename BinOp>
static inline GG_INLINE
auto rewrite_vectors( binop<E1,E2,BinOp> b,
		      typename std::enable_if<
		      vector_length<decltype(rewrite_vectors(b.data1()))>::value
		      == vector_length<decltype(rewrite_vectors(b.data2()))>::value>::type * = nullptr ) {
    auto e1 = rewrite_vectors( b.data1() );
    auto e2 = rewrite_vectors( b.data2() );
    constexpr unsigned short len1 = vector_length<decltype(e1)>::value;
    constexpr unsigned short len2 = vector_length<decltype(e2)>::value;
    static_assert( len1 == len2 || len1 == 1 || len2 == 1,
		   "Expect either equal-length vectors or vector and scalar" );
    return make_binop( e1, e2, BinOp() );
}

template<typename E1, typename E2, typename BinOp>
static GG_INLINE
auto rewrite_vectors( binop<E1,E2,BinOp> b,
		      typename std::enable_if<
		      vector_length<decltype(rewrite_vectors(b.data1()))>::value == 1 &&
		      vector_length<decltype(rewrite_vectors(b.data2()))>::value != 1>::type * = nullptr ) {
    assert( false && "should not allow this" );
    auto e1 = rewrite_vectors( b.data1() );
    auto e2 = rewrite_vectors( b.data2() );
    constexpr unsigned short len1 = vector_length<decltype(e1)>::value;
    constexpr unsigned short len2 = vector_length<decltype(e2)>::value;
    static_assert( len1 == len2 || len1 == 1 || len2 == 1,
		   "Expect either equal-length vectors or vector and scalar" );
    // Insert an operation to expand from scalar to higher vector length
    auto e1v = make_unop_broadcast<len2>( e1 );
    return make_binop( e1v, e2, BinOp() );
}

template<typename E1, typename E2, typename BinOp>
static GG_INLINE
auto rewrite_vectors( binop<E1,E2,BinOp> b,
		      typename std::enable_if<
		      vector_length<decltype(rewrite_vectors(b.data1()))>::value != 1 &&
		      vector_length<decltype(rewrite_vectors(b.data2()))>::value == 1>::type * = nullptr ) {
    auto e1 = rewrite_vectors( b.data1() );
    auto e2 = rewrite_vectors( b.data2() );
    assert( false && "should not allow this" );
    constexpr unsigned short len1 = vector_length<decltype(e1)>::value;
    constexpr unsigned short len2 = vector_length<decltype(e2)>::value;
    static_assert( len1 == len2 || len1 == 1 || len2 == 1,
		   "Expect either equal-length vectors or vector and scalar" );
    // Insert an operation to expand from scalar to higher vector length
    auto e2v = make_unop_broadcast<len1>( e2 );
    return make_binop( e1, e2v, BinOp() );
}

template<typename A, typename T, unsigned short VL>
static constexpr GG_INLINE
auto rewrite_vectors( refop<A,T,VL> r ) {
    return refop<A,T,VL>( r.array(), rewrite_vectors( r.index() ) );
};

template<typename A, typename T, typename M, unsigned short VL>
static constexpr GG_INLINE
auto rewrite_vectors( maskrefop<A,T,M,VL> r ) {
    return maskrefop<A,T,M,VL>( r.array(), 
				rewrite_vectors( r.index() ),
				rewrite_vectors( r.mask() ) );
};

template<typename E1, typename E2, typename RedOp>
static constexpr GG_INLINE
auto rewrite_vectors_redop( E1 e1, E2 e2, RedOp op,
			    typename std::enable_if<
			    vector_length<E1>::value
			    == vector_length<E2>::value>::type * = nullptr ) {
    return make_redop( e1, e2, RedOp() );
}

template<typename E1, typename E2, typename RedOp>
static GG_INLINE
auto rewrite_vectors_redop( E1 e1, E2 e2, RedOp op,
			    typename std::enable_if<
			    vector_length<E1>::value == 1 &&
			    vector_length<E2>::value != 1>::type * = nullptr ) {
    assert( false && "should not allow this" );
    constexpr unsigned short len1 = vector_length<E1>::value;
    constexpr unsigned short len2 = vector_length<E2>::value;
    static_assert( len1 == len2 || len1 == 1 || len2 == 1,
		   "Expect either equal-length vectors or vector and scalar" );
    // The destination is a single value while there is a vector of updates.
    // The vector of updates need to be reduced first, then merged in with a
    // scalar operation. However, if using a cache, the cached value should
    // be a vector, in which case the destination is a vector. For that reason,
    // introduce caches before rewriting vectors.
    auto e2v = make_unop_reduce( e2, RedOp() );
    return make_redop( e1, e2v, RedOp() );
}

template<typename E1, typename E2, typename RedOp>
static GG_INLINE
auto rewrite_vectors_redop( E1 e1, E2 e2, RedOp op,
			    typename std::enable_if<
			    vector_length<E1>::value != 1 &&
			    vector_length<E2>::value == 1>::type * = nullptr ) {
    assert( false && "should not allow this" );
    constexpr unsigned short len1 = vector_length<E1>::value;
    constexpr unsigned short len2 = vector_length<E2>::value;
    static_assert( len1 == len2 || len1 == 1 || len2 == 1,
		   "Expect either equal-length vectors or vector and scalar" );
    // A single update is applied to multiple targets. Expand the vector.
    auto e2v = make_unop_broadcast<len1>( e2 );
    return make_redop(
	e1, e2v,
	op );
    // typename RedOp::template rebind<decltype(e1v),decltype(e2)>::type() );
}

template<typename E1, typename E2, typename RedOp>
static GG_INLINE
auto rewrite_vectors_redop( E1 e1, E2 e2, RedOp op,
    typename std::enable_if<vector_length<E1>::value != 1
			    && vector_length<E2>::value != 1
			    && vector_length<E1>::value != vector_length<E2>::value
			    >::type * = nullptr ) {
    // Bad thing. error
    assert( 0 && "Bad" );
}

// Auxiliary to skip over a top-most refop or maskrefop in the AST.
template<unsigned cid, typename Tr>
static constexpr GG_INLINE
auto rewrite_vectors_skip( cacheop<cid,Tr> c ) {
    return c;
}

template<typename E1, typename E2>
static constexpr GG_INLINE
auto rewrite_vectors_skip( binop<E1,E2,binop_mask> bop ) {
    return add_mask( rewrite_vectors_skip( bop.data1() ),
		     rewrite_vectors_skip( bop.data2() ) );
}

template<typename A, typename T, unsigned short VL>
static constexpr GG_INLINE
auto rewrite_vectors_skip( refop<A,T,VL> r ) {
    return refop<A,T,VL>( r.array(),
			  rewrite_vectors( r.index() ) );
}

template<typename A, typename T, typename M, unsigned short VL>
static constexpr GG_INLINE
auto rewrite_vectors_skip( maskrefop<A,T,M,VL> r ) {
    return maskrefop<A,T,M,VL>( r.array(),
				rewrite_vectors( r.index() ),
				rewrite_vectors( r.mask() ) );
}

template<typename E1, typename E2, typename RedOp>
static constexpr GG_INLINE
auto rewrite_vectors_skip( redop<E1,E2,RedOp> r ) {
    return make_redop( rewrite_vectors( r.ref() ),
		       rewrite_vectors( r.val() ),
		       RedOp() );
}


template<typename E1, typename E2, typename RedOp>
static constexpr GG_INLINE
auto rewrite_vectors( redop<E1,E2,RedOp> r ) {
    return rewrite_vectors_redop(
	rewrite_vectors_skip( r.ref() ), // skip over refop/maskrefop
	rewrite_vectors( r.val() ),
	RedOp() );
}

template<typename E>
static constexpr GG_INLINE
auto rewrite_vectors_main( E e ) {
    return rewrite_vectors( e );
}

#else
template<typename E>
static constexpr GG_INLINE
auto rewrite_vectors_main( E e ) {
    return e;
}
#endif 

} // namespace expr 

#endif // GRAPTOR_DSL_AST_H

