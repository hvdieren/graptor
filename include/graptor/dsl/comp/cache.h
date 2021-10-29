// -*- c++ -*-
#ifndef GRAPTOR_DSL_CACHE_H
#define GRAPTOR_DSL_CACHE_H

#include <type_traits>
#include <algorithm>
#include <tuple>
#include <utility>
#include <memory>

#include "graptor/utils.h"
#include "graptor/dsl/simd_vector.h"
#include "graptor/dsl/comp/extract_lane.h"
#include "graptor/dsl/comp/transform_nt.h"
#include "graptor/encoding.h"

#ifndef PFV_DISTANCE
#define PFV_DISTANCE 32
#endif

#ifndef PFE1_DISTANCE
#define PFE1_DISTANCE 32
#endif

#ifndef PFE2_DISTANCE
#define PFE2_DISTANCE 32
#endif

#ifndef PFT_DISTANCE
#define PFT_DISTANCE 48
#endif

#ifndef ENABLE_NT_VMAP
#define ENABLE_NT_VMAP 0
#endif

#ifndef ENABLE_NT_EMAP
#define ENABLE_NT_EMAP 0
#endif

#ifndef ENABLE_NT_TOPO
#define ENABLE_NT_TOPO 0
#endif

namespace expr {

enum memory_access_method {
    mam_cached = 0,
    mam_nontemporal = 1,
    mam_N = 2
};

template<typename... T>
struct cache;

template<typename... T>
struct cache {
    using type = std::tuple<T...>;

    cache( type t_ ) : t( t_ ) { }

    type t;
};

template<>
struct cache<> {
    using type = std::tuple<>;

    constexpr cache() { }
    constexpr cache( type t_ ) : t( t_ ) { }

    type t;
};

/* -- unused
template<typename T>
static constexpr
auto make_cache( T t ) {
    return cache<T>( t );
}
*/
    
template<typename T>
struct is_cache : std::false_type { };

template<typename... Ts>
struct is_cache<cache<Ts...>> : std::true_type { };


/**********************************************************************
 * Infrastructure for caching values in registers (or on stack)
 * Cached values are either reduction values (redop), which are then
 * reduced while held in registers/on stack (no random memory accesses);
 * or they can be read-only values (refop) that are explicitly loaded
 *  into faster memory.
 **********************************************************************/

// Make cache-create and cache-commit expressions
template<unsigned CID, typename R> // redop or refop
struct reduction_info;

template<typename VTr,layout_t Layout, typename MTr>
GG_INLINE inline
void assign( simd::container<VTr> & l,
	     rvalue<VTr,Layout,MTr> && r ) { // ignore mask
    l = std::move( r.value() );
}

template<typename VTr,layout_t Layout>
GG_INLINE inline
void assign( simd::container<VTr> & l,
	     rvalue<void,Layout,VTr> && r ) {
    l = std::move( r.mask() );
}

template<typename VTr,layout_t Layout, typename... MTr>
GG_INLINE inline
void assign( simd::container<VTr> & l,
	     sb::rvalue<VTr,Layout,MTr...> && r ) { // ignore mask
    l = std::move( r.value() );
}

template<unsigned CID, typename E1, typename E2, typename RedOp>
struct reduction_info<CID,redop<E1,E2,RedOp>> {
    static constexpr unsigned cid = CID;
    static constexpr unsigned next_cid = CID+1;
    static constexpr short id = redop<E1,E2,RedOp>::ref_type::array_type::AID;

    using orig_redop_type = redop<E1,E2,RedOp>;
    // using value_type = typename orig_redop_type::val_type::type;
    using data_type = typename orig_redop_type::ref_type::data_type;
    using orig_ref_type = typename orig_redop_type::ref_type;
    using cached_type = simd::container<data_type>;
    template<unsigned short VL>
    using cacheop_type
	= cacheop<cid,typename data_type::template rebindVL<VL>::type>;

    // Every cached value should have a natural/default vector length
    static constexpr unsigned short VL = orig_redop_type::VL;

    reduction_info( const orig_redop_type & red ) : m_ref( red.ref() ) { }

    orig_ref_type get_ref() const { return m_ref; }

    template<typename value_map_type>
    auto create( const value_map_type & m ) const {
	// static constexpr unsigned short VL
	// = std::max(value_map_type::VLS,value_map_type::VLD);
	return create_map_entry<cid, cached_type>( cached_type() );
    }
    template<typename Cache, typename Environment, typename value_map_type,
	     typename mask_pack_type>
    bool init( const Environment & env, Cache & c,
	       const value_map_type & m,
	       const mask_pack_type & mpack ) const {
	static constexpr unsigned short VLc
	    = std::remove_reference<decltype(c.template get<cid>())>::type::VL;
	static_assert( VLc == VL, "Vector lengths must match" );
	auto cast_ref = expr::make_unop_cvt_data_type<data_type>( m_ref );
	if constexpr ( orig_ref_type::VL != VL && orig_ref_type::VL == 1 ) {
	    auto replicate_ref = expr::make_unop_broadcast<VL>( cast_ref );
	    c.template get<cid>()
		= env.evaluate( c, m, mpack, replicate_ref ).value();
	} else {
	    auto r = env.evaluate( c, m, mpack, cast_ref );
	    assign( c.template get<cid>(), std::move( r ) );
	}
	return true;
    }
    template<typename Environment, typename Cache, typename value_map_type,
	     typename mask_pack_type>
    __attribute__((always_inline)) inline
    bool commit( const Environment & env, Cache &c,
		 const value_map_type & m, const mask_pack_type & mpack ) const {
	commit( env, c, m, mpack, m_ref );
	return true;
    }
    template<typename value_map_type, typename CacheI, typename CacheF,
	     typename Environment>
    __attribute__((always_inline)) inline
    bool commit_if_changed( const Environment & env,
			    CacheI &c_init, CacheF &c_final,
			    const value_map_type & m ) const {
	commit_if_changed( env, c_init, c_final, m, m_ref );
	return true;
    }
    template<bool Atomic, typename Environment, typename value_map_type, typename cache_map_type>
    __attribute__((always_inline)) inline
    bool commit_with_reduce( const Environment & env,
			     cache_map_type &c, const value_map_type & m ) const {
	using cached_type = typename std::remove_reference<decltype(c.template get<cid>())>::type;
	constexpr unsigned short VL = cached_type::VL;
	auto rexpr = make_redop( m_ref, cacheop_type<VL>(), RedOp() );
#if ENABLE_NT_EMAP
	auto ntexpr = transform_nt<true>( rexpr );
#else
	auto ntexpr = rexpr;
#endif
	auto env2 = env.replace_ptrs( ntexpr );
	env2.template evaluate<Atomic>( c, m, ntexpr );
	return true;
    }
    template<typename Environment, typename Cache, typename value_map_type>
    bool clear( const Environment & env,
		Cache & c, const value_map_type & m ) const {
	using cached_type = typename std::remove_reference<decltype(c.template get<cid>())>::type;
	constexpr unsigned short VL = cached_type::VL;
	
	// Fill the cache with a vector consisting of the unit value of the
	// reduction operation
	auto vunit = constant_val3<data_type>( orig_redop_type::unit() );
	auto vclear = make_storeop( cacheop_type<VL>(), vunit );
	// auto array_map = extract_pointer_set( vclear );
	// evaluator<value_map_type, Cache, decltype(array_map)>( c, m, array_map ).evaluate( vclear );
	env.evaluate( c, m, vclear );
	return true;
    }
    template<typename Cache, typename value_map_type>
    bool reduce( Cache & c, const value_map_type & m ) const {
	using cached_type = typename std::remove_reference<decltype(c.template get<cid>())>::type;
	constexpr unsigned short VL = cached_type::VL;
	
	// Reduce the vector we have in cache to a single element.
	// TODO: check what needs to be done if this is already a scalar value.
	auto vreduce = make_unop_reduce( cacheop_type<VL>(), typename orig_redop_type::redop_type() );
	// Cast type of scalar value to appropriate type for the target location
	auto scast = make_unop_cvt_type<typename orig_ref_type::type>( vreduce );
	// Scalar target memory location
	auto sref = extract_lane<0>( m_ref );
	
	// Don't need load-modify-store, just store suffices
	auto scommit // = make_storeop( sref, scast );
	    = make_redop( sref, scast, typename orig_redop_type::redop_type() );

	// Execute expression
	auto array_map = extract_pointer_set( scommit );
	evaluator<value_map_type, Cache, decltype(array_map)>( c, m, array_map ).evaluate( scommit );
	return true;
    }

    template<typename Cache, typename value_map_type>
    bool reduce_and_commit( Cache & c, const value_map_type & m ) const {
	using cached_type = typename std::remove_reference<decltype(c.template get<cid>())>::type;
	constexpr unsigned short VL = cached_type::VL;
	
	// Reduce the vector we have in cache to a single element.
	auto vreduce = make_unop_reduce( cacheop_type<VL>(),
					 typename orig_redop_type::redop_type() );
	// Cast type of scalar value to appropriate type for the target location
	auto scast = make_unop_cvt_type<typename orig_ref_type::type>( vreduce );
	// Rebind to scalar target memory location
	cache<> empty;
	auto sref = rewrite_scalarize( m_ref, empty );
	
	// Store the reduced value, don't reduce with the value in memory
	auto scommit = make_storeop( sref, scast );

	// Execute expression
	auto array_map = extract_pointer_set( scommit );
	evaluator<value_map_type, Cache, decltype(array_map)>( c, m, array_map ).evaluate( scommit );

	// done
	return true;
    }

    template<unsigned short VL, typename Cache>
    auto get_reduce_expr( Cache & c ) const {
	// using cached_type = typename std::remove_reference<decltype(c.template get<cid>())>::type;
	// constexpr unsigned short VL = cached_type::VL;
	
	// Reduce the vector we have in cache to a single element.
	// TODO: check what needs to be done if this is already a scalar value.
	auto vreduce = make_unop_reduce( cacheop_type<VL>(), typename orig_redop_type::redop_type() );
	// Cast type of scalar value to appropriate type for the target location
	auto scast = make_unop_cvt_type<typename orig_ref_type::type>( vreduce );

	return scast;
    }

private:
    template<typename Environment, typename value_map_type, typename Cache,
	     typename mask_pack_type,
	     typename Expr>
    __attribute__((always_inline)) inline
    void commit( const Environment & env, Cache &c, const value_map_type & m,
		 const mask_pack_type & mpack,
		 Expr e,
		 typename std::enable_if<value_map_type::VLD==1>::type * = nullptr ) const {
	using cached_type = typename std::remove_reference<decltype(c.template get<cid>())>::type;
	constexpr unsigned short VL = cached_type::VL;
	
	// Reduce the vector we have in cache to a single element.
	// TODO: check what needs to be done if this is already a scalar value.
	auto vreduce = make_unop_reduce( cacheop_type<VL>(), typename orig_redop_type::redop_type() );
	// Cast type of scalar value to appropriate type for the target location
	auto scast = make_unop_cvt_type<typename orig_ref_type::type>( vreduce );
	// Don't need load-modify-store, just store suffices
#if ENABLE_NT_EMAP
	auto scommit = make_ntstoreop( e, scast );
#else
	auto scommit = make_storeop( e, scast );
#endif
	// Execute expression
	auto env2 = env.replace_ptrs( scommit );
	env2.evaluate( c, m, mpack, scommit );
    }
    template<typename Environment, typename value_map_type, typename Cache,
	     typename mask_pack_type,
	     typename Expr>
    __attribute__((always_inline)) inline
    void commit( const Environment & env, Cache &c, const value_map_type & m,
		 const mask_pack_type & mpack,
		 Expr e,
		 typename std::enable_if<value_map_type::VLD!=1>::type * = nullptr ) const {
	using cached_type = typename std::remove_reference_t<decltype(c.template get<cid>())>;
	constexpr unsigned short VL = cached_type::VL;
#if ENABLE_NT_EMAP
	auto vcommit = make_ntstoreop( e, cacheop_type<VL>() );
#else
	auto vcommit = make_storeop( e, cacheop_type<VL>() );
#endif
	// auto array_map = extract_pointer_set( vcommit );
	env.evaluate( c, m, mpack, vcommit );
    }
    template<typename value_map_type, typename CacheI, typename CacheF, typename Expr,
	     typename Environment>
    __attribute__((always_inline)) inline
    void commit_if_changed( const Environment & env,
			    CacheI &c_init, CacheF &c_final,
			    const value_map_type & m, Expr e ) const {
	// using cached_type = typename std::remove_reference<decltype(c_final.template get<cid>())>::type;
	constexpr unsigned short VL = cached_type::VL;

	auto c_merge = map_merge( c_final, c_init );

	// Retrieve initial value
	auto iv_expr = expr::cacheop<cid+std::tuple_size<decltype(c_final.get_entries())>::value,
				     data_type>();
	auto fv_expr = expr::cacheop<cid,data_type>();

#if ENABLE_NT_EMAP
	auto vcommit = make_ntstoreop( e, cacheop_type<VL>() );
#else
	auto vcommit = make_storeop( e, cacheop_type<VL>() );
#endif
	auto vcommit_cond = add_mask( vcommit, iv_expr != fv_expr );

	// auto array_map = extract_pointer_set( vcommit_cond );
	// evaluator<value_map_type, decltype(c_merge), decltype(array_map)>(
	// c_merge, m, array_map ).evaluate( vcommit_cond );
	env.replace_ptrs( vcommit_cond ).evaluate( c_merge, m, vcommit_cond );
    }

public:
    template<unsigned long Distance,
	     typename value_map_type, typename Cache>
    __attribute__((always_inline)) inline
    bool prefetch( Cache &c, const value_map_type & m ) const {
	if constexpr ( true || Distance > 0 ) {
	    static constexpr unsigned short VLc
		= std::remove_reference<decltype(c.template get<cid>())>::type::VL;
	    static_assert( VLc == VL, "Vector lengths must match" );
	    // Evaluate index
	    auto iexpr = m_ref.index();
	    auto array_map = extract_pointer_set( iexpr );
	    auto rval = evaluator<value_map_type, Cache, decltype(array_map)>(
		c, m, array_map ).evaluate( iexpr );

	    using index_type = typename decltype(rval.value())::member_type;
	    using encoding = typename orig_ref_type::array_type::encoding;
	    // simd::detail::vector_ref_impl<data_type,index_type,encoding,false>
	    // ref( m_ref.array().ptr() + Distance, rval.value() );
	    auto ref = simd::template create_vector_ref_vec<
		data_type,index_type,encoding,false,decltype(rval.value())::layout>(
		    m_ref.array().ptr() + Distance, rval.value() );
		    

	    ref.template prefetch<_MM_HINT_NTA>( rval.mask() );
	}
	return true;
    }

private:
    const orig_ref_type m_ref;
};

template<unsigned CID, typename S, typename U, typename C, typename DFSAOp>
struct reduction_info<CID,dfsaop<S,U,C,DFSAOp>> {
    static constexpr unsigned cid = CID;
    static constexpr unsigned next_cid = CID+1;
    static constexpr short id = dfsaop<S,U,C,DFSAOp>::state_type::array_type::AID;

    using orig_redop_type = dfsaop<S,U,C,DFSAOp>;
    using data_type = typename orig_redop_type::state_type::data_type;
    using orig_ref_type = typename orig_redop_type::state_type;
    using cached_type = simd::container<data_type>;
    template<unsigned short VL>
    using cacheop_type
	= cacheop<cid,typename data_type::template rebindVL<VL>::type>;

    // Every cached value should have a natural/default vector length
    static constexpr unsigned short VL = orig_redop_type::VL;

    reduction_info( orig_redop_type red ) : m_ref( red.state() ) { }

    orig_ref_type get_ref() const { return m_ref; }

    template<typename value_map_type>
    auto create( const value_map_type & m ) const {
	return create_map_entry<cid, cached_type>( cached_type() );
    }
    template<typename Environment, typename Cache,
	     typename value_map_type, typename mask_pack_type>
    bool init( const Environment & env, Cache & c,
	       const value_map_type & m,
	       const mask_pack_type & mpack ) const {
	static constexpr unsigned short VLc
	    // = std::max(value_map_type::VLS,value_map_type::VLD);
	    = std::remove_reference<decltype(c.template get<cid>())>::type::VL;
	static_assert( VLc == VL, "Vector lengths must match" );
	// Remove mask: cache anchored on densely iterated side (src/dst)
	// so no gather required and indices should be valid even if masked out.
	auto uc_ref =
	    make_refop( m_ref.array(), remove_mask( m_ref.index() ) );
	auto cast_ref = expr::make_unop_cvt_data_type<data_type>( uc_ref );
	init_rest<VL>( env, c, m, mpack, cast_ref );
	return true;
    }
    template<typename Environment, typename value_map_type,
	     typename mask_pack_type, typename Cache>
    __attribute__((always_inline)) inline
    bool commit( const Environment & env,
		 Cache &c,
		 const value_map_type & m,
		 const mask_pack_type & mpack ) const {
	using cached_type = typename std::remove_reference<
	    decltype(c.template get<cid>())>::type;
	constexpr unsigned short VL = cached_type::VL;

	static_assert( value_map_type::VLD == VL, "seems logical" );

	if constexpr ( value_map_type::VLD == 1 ) {
	    // Reduce the vector we have in cache to a single element.
	    // TODO: check what needs to be done if this is already
	    //       a scalar value.
	    auto vreduce = make_unop_reduce(
		cacheop_type<VL>(), typename orig_redop_type::op_type() );
	    // Cast type of scalar value to appropriate type for the
	    // target location
	    auto scast = make_unop_cvt_type<typename orig_ref_type::type>(
		vreduce );
	    // Don't need load-modify-store, just store suffices
	    auto scommit = make_storeop( m_ref, scast );
	    // Execute expression
	    env.evaluate( c, m, mpack, scommit );
	} else {
	    auto vcommit = make_storeop( m_ref, cacheop_type<VL>() );
	    env.evaluate( c, m, mpack, vcommit );
	}
	return true;
    }
    template<typename value_map_type, typename CacheI, typename CacheF,
	     typename Environment>
    __attribute__((always_inline)) inline
    bool commit_if_changed( const Environment & env,
			    CacheI &c_init, CacheF &c_final,
			    const value_map_type & m ) const {
	commit_if_changed( env, c_init, c_final, m, m_ref );
	return true;
    }
    template<typename Environment, typename Cache, typename value_map_type>
    bool clear( const Environment & env,
		Cache & c, const value_map_type & m ) const {
	// using cached_type = typename std::remove_reference<decltype(c.template get<cid>())>::type;
	constexpr unsigned short VL = cached_type::VL;
	
	// Fill the cache with a vector consisting of the unit value of the
	// reduction operation
	auto vunit = constant_val3<data_type>( orig_redop_type::unit() );
	auto vclear = make_storeop( cacheop_type<VL>(), vunit );
	//auto array_map = extract_pointer_set( vclear );
	//evaluator<value_map_type, Cache, decltype(array_map)>( c, m, array_map ).evaluate( vclear );
	env.evaluate( c, m, vclear );
	return true;
    }
    template<typename Cache, typename value_map_type>
    bool reduce( Cache & c, const value_map_type & m ) const {
	using cached_type = typename std::remove_reference<decltype(c.template get<cid>())>::type;
	constexpr unsigned short VL = cached_type::VL;
	
	// Reduce the vector we have in cache to a single element.
	// TODO: check what needs to be done if this is already a scalar value.
	auto vreduce = make_unop_reduce( cacheop_type<VL>(), typename orig_redop_type::op_type() );
	// Cast type of scalar value to appropriate type for the target location
	auto scast = make_unop_cvt_type<typename orig_ref_type::type>( vreduce );
	// Scalar target memory location
	auto sref = extract_lane<0>( m_ref );
	
	// Don't need load-modify-store, just store suffices
	auto scommit // = make_storeop( sref, scast );
	    = make_redop( sref, scast, typename orig_redop_type::op_type() );

	// Execute expression
	auto array_map = extract_pointer_set( scommit );
	evaluator<value_map_type, Cache, decltype(array_map)>( c, m, array_map ).evaluate( scommit );
	return true;
    }

    template<typename Cache, typename value_map_type>
    bool reduce_and_commit( Cache & c, const value_map_type & m ) const {
	using cached_type = typename std::remove_reference<decltype(c.template get<cid>())>::type;
	constexpr unsigned short VL = cached_type::VL;
	
	// Reduce the vector we have in cache to a single element.
	auto vreduce = make_unop_reduce( cacheop_type<VL>(),
					 typename orig_redop_type::op_type() );
	// Cast type of scalar value to appropriate type for the target location
	auto scast = make_unop_cvt_type<typename orig_ref_type::type>( vreduce );
	// Rebind to scalar target memory location
	cache<> empty;
	auto sref = rewrite_scalarize( m_ref, empty );
	
	// Store the reduced value, don't reduce with the value in memory
	auto scommit = make_storeop( sref, scast );

	// Execute expression
	auto array_map = extract_pointer_set( scommit );
	evaluator<value_map_type, Cache, decltype(array_map)>( c, m, array_map ).evaluate( scommit );

	// done
	return true;
    }

    template<unsigned short VL, typename Cache>
    auto get_reduce_expr( Cache & c ) const {
	// using cached_type = typename std::remove_reference<decltype(c.template get<cid>())>::type;
	// constexpr unsigned short VL = cached_type::VL;
	
	// Reduce the vector we have in cache to a single element.
	// TODO: check what needs to be done if this is already a scalar value.
	auto vreduce = make_unop_reduce( cacheop_type<VL>(), typename orig_redop_type::op_type() );
	// Cast type of scalar value to appropriate type for the target location
	auto scast = make_unop_cvt_type<typename orig_ref_type::type>( vreduce );

	return scast;
    }

private:
    template<unsigned short VL, typename Environment, typename Cache,
	     typename value_map_type, typename mask_pack_type, typename Expr>
    void init_rest( const Environment & env, Cache & c,
		    const value_map_type & m, const mask_pack_type & mpack,
		    Expr cast_ref ) const {
	if constexpr ( Expr::VL != VL && Expr::VL == 1 ) {
	    auto replicate_ref = expr::make_unop_broadcast<VL>( cast_ref );
	    // TODO: need to implement a selective update of the cache as the
	    //       evaluator may return a mask with the values it has calculated.
	    c.template get<cid>() = env.evaluate( c, m, mpack, replicate_ref )
		.value();
	} else {
	    auto r = env.evaluate( c, m, mpack, cast_ref );
	    assign( c.template get<cid>(), std::move( r ) );
	}
    }
    template<typename value_map_type, typename CacheI, typename CacheF, typename Expr,
	     typename Environment>
    __attribute__((always_inline)) inline
    void commit_if_changed( const Environment & env,
			    CacheI &c_init, CacheF &c_final,
			    const value_map_type & m, Expr e ) const {
	// using cached_type = typename std::remove_reference<decltype(c_final.template get<cid>())>::type;
	constexpr unsigned short VL = cached_type::VL;

	auto c_merge = map_merge( c_final, c_init );

	// Retrieve initial value
	auto iv_expr = expr::cacheop<cid+std::tuple_size<decltype(c_final.get_entries())>::value,
				     data_type>();
	auto fv_expr = expr::cacheop<cid,data_type>();

#if ENABLE_NT_EMAP
	auto vcommit = make_ntstoreop( e, cacheop_type<VL>() );
#else
	auto vcommit = make_storeop( e, cacheop_type<VL>() );
#endif
	auto vcommit_cond = add_mask( vcommit, iv_expr != fv_expr );

	// auto array_map = extract_pointer_set( vcommit_cond );
	// evaluator<value_map_type, decltype(c_merge), decltype(array_map)>(
	// c_merge, m, array_map ).evaluate( vcommit_cond );
	env.replace_ptrs( vcommit_cond ).evaluate( c_merge, m, vcommit_cond );
    }
    
private:
    orig_ref_type m_ref;
};


// Note: this one isn't actually a reduction. It is meant to build a read-only
//       cache of arrays indexed by vk_src; or a cache in vertex_map.
template<unsigned CID, typename A, typename T, unsigned short VL_>
struct reduction_info<CID,refop<A,T,VL_>> {
    // Every cached value should have a natural/default vector length
    static constexpr unsigned short VL = VL_;

    static constexpr unsigned cid = CID;
    static constexpr unsigned next_cid = CID+1;
    static constexpr short id = refop<A,T,VL>::array_type::AID; // A::AID

    using orig_ref_type = refop<A,T,VL>;
    using value_type = typename orig_ref_type::type;
    using data_type = typename orig_ref_type::data_type;
    using cached_type = simd::container<data_type>;
    template<unsigned short VL__>
    using cacheop_type
	= cacheop<cid,typename data_type::template rebindVL<VL__>::type>;

    reduction_info( orig_ref_type ref ) : m_ref( ref ) { }

    orig_ref_type get_ref() const { return m_ref; }

    template<typename value_map_type>
    auto create( const value_map_type & m ) const {
	return create_map_entry<cid, cached_type>( cached_type() );
    }
    template<typename Cache, typename Environment, typename value_map_type,
	     typename mask_pack_type >
    __attribute__((always_inline))
    bool init( const Environment & env, Cache & c,
	       const value_map_type & m, const mask_pack_type & mpack ) const {
	using cached_type =
	    std::remove_reference_t<decltype(c.template get<cid>())>;
	constexpr unsigned short VLc = cached_type::VL;
	auto cast_ref = expr::make_unop_cvt_data_type<data_type>( m_ref );
	auto r = env.evaluate( c, m, mpack, cast_ref );
	assign( c.template get<cid>(), std::move( r ) );
	return true;
    }

protected:
    orig_ref_type m_ref;
};

enum use_rw_t {
    use_rw_read_only,		//!< read-only array
    use_rw_read_before_write,	//!< read and write, read old value, or may
     	 	 	 	//   not overwrite old value due to masks
    use_rw_write_kill		//!< old value no longer necessary but be
     	 	 	 	//   careful with partially written cache blocks
};

// Prior execution analysed as use, followed by a read/write, possible masked
template<use_rw_t Use, bool Write, bool Masked>
struct use_rw_merge {
    static constexpr use_rw_t value =
	Write
	? ( Masked
	    ? use_rw_read_before_write
	    : ( Use == use_rw_write_kill
		? use_rw_write_kill
		: use_rw_read_before_write ) )
	: Use;
};

template<use_rw_t Use, bool Write, bool Masked>
constexpr use_rw_t use_rw_merge_v = use_rw_merge<Use,Write,Masked>::value;

// TODO: also differentiate if init is necessary (read-before-write)
template<unsigned CID, use_rw_t RW, bool IsCached, bool IsLocalVar, typename R>
struct use_info : public reduction_info<CID,R> {

    static constexpr use_rw_t rw_status = RW;
    static constexpr bool is_cached = IsCached;
    static constexpr bool is_local_var = IsLocalVar;
    using parent_type = reduction_info<CID,R>;
    using parent_type::cid;
    using parent_type::m_ref;
    template<unsigned short VL__>
    using cacheop_type = typename parent_type::template cacheop_type<VL__>;

    static_assert( !is_local_var || is_cached,
		   "All local variables must be cached" );

    use_info( typename parent_type::orig_ref_type ref ) : parent_type( ref ) { }

    template<typename Cache, typename Environment, typename value_map_type,
	     typename mask_pack_type>
    __attribute__((always_inline))
    bool init( const Environment & env, Cache & c,
	       const value_map_type & m, const mask_pack_type & mpack ) const {
	if constexpr ( !is_local_var
		       && is_cached && rw_status != use_rw_write_kill )
	    return parent_type::init( env, c, m, mpack );
	return true;
    }

    template<typename Environment, typename Cache, typename value_map_type,
	     typename mask_pack_type>
    __attribute__((always_inline)) inline
    bool commit( const Environment & env, Cache &c,
		 const value_map_type & m,
		 const mask_pack_type & mpack ) const {
	if constexpr ( !is_local_var
		       && is_cached && rw_status != use_rw_read_only ) {
	    using cached_type
		= typename std::remove_reference_t<
		    decltype(c.template get<cid>())>;
	    constexpr unsigned short VL = cached_type::VL;
	    auto vcommit = make_storeop( m_ref, cacheop_type<VL>() );
	    env.evaluate( c, m, mpack, vcommit );
	}
	return true;
    }

    template<unsigned long Distance, typename value_map_type, typename Cache>
    __attribute__((always_inline)) inline
    bool prefetch( Cache &c, const value_map_type & m ) const {
	if constexpr ( Distance > 0 && is_cached
		       && !is_local_var
		       && rw_status != use_rw_write_kill ) {
		const typename parent_type::orig_ref_type::type * addr
		    = m_ref.array().ptr();
		VID vid = m.template get<vk_vid>().data();
		_mm_prefetch( addr + vid + Distance, _MM_HINT_NTA );
	}
	return true;
    }
};

template<unsigned CID, typename R>
reduction_info<CID,R> make_reduction_info( R r ) {
    return reduction_info<CID,R>( r );
}

template<unsigned CID, typename A, typename E1, typename E2, unsigned short VL>
auto make_reduction_info( refop<A,binop<E1,E2,binop_mask>,VL> r ) {
    return make_reduction_info<CID>(
	make_refop( r.array(), remove_mask( r.index() ) ) );
}

template<unsigned CID, typename A, typename E1, typename E2, unsigned short VL>
auto make_reduction_info( refop<A,binop<E1,E2,binop_predicate>,VL> r ) {
    return make_reduction_info<CID>(
	make_refop( r.array(), r.index().data2() ) );
}

template<unsigned CID, use_rw_t RW, bool IsCached, bool IsLocalVar, typename R>
auto make_use_info( R r ) {
    // All uses relate to vertexmap, i.e., STREAM-like behaviour, and should
    // be accessed using non-temporal instructions
#if ENABLE_NT_VMAP
    auto nt_r = transform_nt<true>( r );
#else
    auto nt_r = r;
#endif
    return use_info<CID,RW,IsCached,IsLocalVar,decltype(nt_r)>( nt_r );
}

// Do not remove the mask on a use that is indexed by vk_src, as it
// may be a masked gather. This is a little bit of a hack; should have a
// cleaner way of deciding when to remove mask and when not.
template<unsigned CID, use_rw_t RW, bool IsCached, bool IsLocalVar,
	 typename A, typename E1, typename E2, unsigned short VL>
auto make_use_info( refop<A,binop<E1,E2,binop_mask>,VL> r,
		    std::enable_if_t<!is_indexed_by_vk<vk_src,decltype(r)>::value>
		    * = nullptr ) {
    return make_use_info<CID,RW,IsCached,IsLocalVar>(
	make_refop( r.array(), remove_mask( r.index() ) ) );
}


template<typename... T>
auto car( const cache<T...> & c ) {
    return std::get<0>( c.t );
}

template<std::size_t... Ns, typename... Ts>
auto cdr_impl( std::index_sequence<Ns...>, const std::tuple<Ts...> & t ) {
   return std::make_tuple( std::get<Ns+1u>(t)... );
}

template<typename T0, typename... Ts>
auto cdr( const cache<T0,Ts...> & c ) {
    return cache<Ts...>(
	cdr_impl( std::make_index_sequence<sizeof...(Ts)>(), c.t ) );
}

/***********************************************************************
 * cache_contains
 ***********************************************************************/
template<short id, typename C, typename Enable = void>
struct cache_contains : std::false_type { };

template<short id, typename C0, typename... Cs>
struct cache_contains<id,cache<C0,Cs...>,
		      typename std::enable_if<id==C0::id>::type> 
    : std::true_type { };

template<short id, typename C0, typename... Cs>
struct cache_contains<id,cache<C0,Cs...>,
		      typename std::enable_if<id!=C0::id>::type> 
    : cache_contains<id,cache<Cs...>> { };

template<unsigned cid, typename C, typename Enable = void>
struct cache_contains_cid : std::false_type { };

template<unsigned cid, typename C0, typename... Cs>
struct cache_contains_cid<cid,cache<C0,Cs...>,
		      typename std::enable_if<cid==C0::cid>::type> 
    : std::true_type { };

template<unsigned cid, typename C0, typename... Cs>
struct cache_contains_cid<cid,cache<C0,Cs...>,
		      typename std::enable_if<cid!=C0::cid>::type> 
    : cache_contains_cid<cid,cache<Cs...>> { };

/***********************************************************************
 * cache_requires_rewrite
 ***********************************************************************/
// Default case for reduction_info - always rewrite
template<short id, typename RU>
struct rewrite_required : public std::true_type { };

template<short id,
	 unsigned CID, use_rw_t RW, bool IsCached, bool IsLocalVar, typename R>
struct rewrite_required<id,use_info<CID,RW,IsCached,IsLocalVar,R>> {
    static constexpr bool value = IsCached;
};

template<short id, typename C, typename Enable = void>
struct cache_requires_rewrite : std::false_type { };

template<short id, typename C0, typename... Cs>
struct cache_requires_rewrite<
    id, cache<C0,Cs...>,
    std::enable_if_t<id == C0::id && rewrite_required<id,C0>::value>>
    : std::true_type { };

template<short id, typename C0, typename... Cs>
struct cache_requires_rewrite<
    id, cache<C0,Cs...>,
    std::enable_if_t<id != C0::id || !rewrite_required<id,C0>::value>>
    : cache_requires_rewrite<id,cache<Cs...>> { };

/***********************************************************************
 * cache_select
 ***********************************************************************/
/*
template<short id, typename C>
auto cache_select( const cache<C> & c,
		   typename std::enable_if<id==C::id>::type * = nullptr ) {
    return car(c);
}
*/

template<short id, typename C0, typename... C>
auto cache_select( const cache<C0,C...> & c,
		   typename std::enable_if<id==C0::id>::type * = nullptr ) {
    return car(c);
}

template<short id, typename C0, typename... C>
auto cache_select( const cache<C0,C...> & c,
		   typename std::enable_if<id!=C0::id>::type * = nullptr ) {
    return cache_select<id>( cdr(c) );
}

template<unsigned cid, typename C0, typename... C>
auto cache_select_cid( const cache<C0,C...> & c,
		       typename std::enable_if<cid==C0::cid>::type * = nullptr ) {
    return car(c);
}

template<unsigned cid, typename C0, typename... C>
auto cache_select_cid( const cache<C0,C...> & c,
		       typename std::enable_if<cid!=C0::cid>::type * = nullptr ) {
    return cache_select_cid<cid>( cdr(c) );
}

template<typename... C0n, typename... C1n>
cache<C0n...,C1n...>
cache_cat( const cache<C0n...> & c0, const cache<C1n...> & c1 ) {
    return cache<C0n...,C1n...>( std::tuple_cat( c0.t, c1.t ) );
}

template<typename... C0n, typename... C1n, typename... C2n>
cache<C0n...,C1n...,C2n...>
cache_cat( const cache<C0n...> & c0, const cache<C1n...> & c1,
	   const cache<C2n...> & c2 ) {
    return cache<C0n...,C1n...,C2n...>(
	std::tuple_cat( c0.t, std::tuple_cat( c1.t, c2.t ) ) );
}

template<typename C0, typename... C>
static constexpr
cache<C0,C...> cache_cons( const C0 & c0, const cache<C...> & c ) {
    static_assert( !is_cache<C0>::value, "Cache elements should not be caches" );
    return cache<C0,C...>( std::tuple_cat( std::make_tuple( c0 ), c.t ) );
}

template<short id, typename Cache>
struct cache_find;

template<short id>
struct cache_find<id,cache<>> {
    static constexpr bool value = false;
};

template<short id, typename C>
struct cache_find<id,cache<C>> {
    static constexpr bool value = id == C::id;
};

template<short id, typename C0, typename... C>
struct cache_find<id,cache<C0,C...>> {
    static constexpr bool value
	= id == C0::id || cache_find<id,cache<C...>>::value;
    
};

template<short id>
static constexpr
cache<> cache_remove( const cache<> & c ) {
    return c;
}

template<short id, typename C>
static constexpr
cache<C> cache_remove( const cache<C> & c,
		       typename std::enable_if<id!=C::id>::type * = nullptr ) {
    return c;
}

template<short id, typename C>
static constexpr
cache<> cache_remove( const cache<C> &,
		      typename std::enable_if<id==C::id>::type * = nullptr ) {
    return cache<>();
}

template<short id, typename C0, typename... C>
static constexpr
auto cache_remove( const cache<C0,C...> & c,
		   typename std::enable_if<id==C0::id>::type * = nullptr ) {
    return cache_remove<id>( cdr( c ) );
}

template<short id, typename C0, typename... C>
static constexpr
auto cache_remove( const cache<C0,C...> & c,
		   typename std::enable_if<id!=C0::id>::type * = nullptr ) {
    return cache_cons( car( c ), cache_remove<id>( cdr( c ) ) );
}

/***********************************************************************
 * Detecting conflicting uses.
 * We conservatively do not cache all uses that access the same array
 * using a different index expression. We need to be careful about
 * examples such as:
 *   index[v+1] - index[v]
 * where we need to differentiate the index expression and cache each
 * seperately.
 * Another difficulty is:
 *   A[v] = 1;
 *   A[v+constant 0] = 2;
 * the baseline code would consider those as two different cached uses.
 * Committing those updates to memory if cached separately may result in
 * errors. Analysing that a constant may be zero, is not possible
 * in the type system (we could find vk_zero, but not vk_any with 0).
 * We avoid difficulties by only caching uses where the array is used
 * with the exact same index expression. Here, all constants are
 * considered different (same type, different value).
 ***********************************************************************/
template<typename Use0, typename Use1>
struct is_conflicting_use {
    static constexpr bool value = 
	Use0::id == Use1::id
	&& ( !std::is_same_v<typename Use0::type, typename Use1::type>
	     || is_indexed_by_vk<vk_any,typename Use0::type>::value );
};

template<typename Use0, typename Use1>
constexpr bool is_conflicting_use_v = is_conflicting_use<Use0,Use1>::value;

template<typename Ref0, typename Ref1>
struct is_conflicting_ref {
    static constexpr bool value = 
	Ref0::array_type::AID == Ref1::array_type::AID
	&& ( !std::is_same_v<Ref0, Ref1>
	     || is_indexed_by_vk<vk_any,Ref0>::value );
};

template<typename Ref0, typename Ref1>
constexpr bool is_conflicting_ref_v = is_conflicting_ref<Ref0,Ref1>::value;

template<typename Use, typename Cache>
struct cache_has_conflicting_use : public std::false_type { };

template<typename Use, typename U0>
struct cache_has_conflicting_use<Use,cache<U0>> {
    static constexpr bool value = is_conflicting_use_v<Use,U0>;
};

template<typename Use, typename C0, typename... C>
struct cache_has_conflicting_use<Use,cache<C0,C...>> {
    static constexpr bool value =
	is_conflicting_use_v<Use,C0>
	|| cache_has_conflicting_use<Use,cache<C...>>::value;
    
};

template<typename Use0, typename Use1>
constexpr bool cache_has_conflicting_use_v =
    cache_has_conflicting_use<Use0,Use1>::value;


/***********************************************************************
 * A local variable is encoded as an array indexed by zero, no backing
 * storage (array_encoding<void> or array_encoding_zero) and having
 * array_encoding<void>:
 *    use_rw_write_kill, i.e., it is written before being used.
 *    use_rw_read_before_write, in order to admit masked initialisation
 * array_encoding_zero<Ty>:
 *    any use (although a use of rw_read_only could be replaced by value zero).
 * Notes:
 * - a global variable is encoded as an array indexed by constant zero,
 *   having backing storage.
 * - an array indexed by constant zero, not having backing storage and
 *   a usage that sees it used prior to being defined is an error.
 ***********************************************************************/
template<use_rw_t rw_status, typename T, typename Enable = void>
struct is_local_var : public std::false_type { };

template<use_rw_t rw_status, typename A, typename T, unsigned short VL>
struct is_local_var<rw_status,
		    refop<A,T,VL>,
		    std::enable_if_t<
    (
	( std::is_same_v<typename A::encoding,array_encoding<void>>
	  // && rw_status == use_rw_write_kill )
	  && rw_status != use_rw_read_only )
	|| is_array_encoding_zero_v<typename A::encoding>
	) && is_value_vk<T,vk_zero>::value>>
    : public std::true_type { };

template<use_rw_t rw_status, typename T, typename Enable = void>
constexpr bool is_local_var_v = is_local_var<rw_status,T>::value;


/***********************************************************************
 * Does any cache entry write to a particular use?
 * This is applied to uses (vertex map) only; reduction info does not
 * have the modified entry (they are all modified).
 ***********************************************************************/
template<short id, typename Cache, typename Enable = void>
struct cache_writes : public std::false_type { };

template<short id, typename C0, typename... C>
struct cache_writes<id,cache<C0,C...>,
		    std::enable_if_t<id==C0::id && C0::modified>>
    : public std::true_type { };

template<short id, typename C0, typename... C>
struct cache_writes<id,cache<C0,C...>,
		    std::enable_if_t<id!=C0::id || !C0::modified>>
    : public cache_writes<id,cache<C...>> { };

static constexpr
cache<> cache_dedup( const cache<> & c ) {
    return c;
}

template<typename C0, typename... C>
static constexpr
auto cache_dedup( const cache<C0,C...> & c,
		  typename std::enable_if<cache_find<C0::id,cache<C...>>::value>::type * = nullptr ) {
    // Not clear what the aim was:
    // + if repeated, then remove all instances
    // + if repeated, retain at most once
    // Seems first: remove all instances
    return cache_dedup( cache_remove<C0::id>( cdr( c ) ) );
}

template<typename C0, typename... C>
static constexpr
auto cache_dedup( const cache<C0,C...> & c,
		  typename std::enable_if<!cache_find<C0::id,cache<C...>>::value>::type * = nullptr ) {
    return cache_cons( car( c ), cache_dedup( cdr( c ) ) );
}

static constexpr
cache<> cache_dedup_uses( const cache<> & c ) {
    return c;
}

template<typename C0, typename... C>
static constexpr
auto cache_dedup_uses(
    const cache<C0,C...> & c,
    std::enable_if_t<!cache_has_conflicting_use_v<C0,cache<C...>>> * = nullptr
    ) {
    // Only keep each array reference once. Weed out any use of an array
    // that has multiple (potentially) different index expressions.
    return cache_cons( car( c ).template rebindWrite<cache_writes<C0::id,cache<C0,C...>>::value>(),
		       cache_dedup_uses( cache_remove<C0::id>( cdr( c ) ) ) );
}

template<typename C0, typename... C>
static constexpr
auto cache_dedup_uses(
    const cache<C0,C...> & c,
    std::enable_if_t<cache_has_conflicting_use_v<C0,cache<C...>>> * = nullptr
    ) {
    return cache_dedup_uses( cache_remove<C0::id>( cdr( c ) ) );
}


template<typename... C>
constexpr size_t cache_count( const cache<C...> & ) {
    return sizeof...(C);
}

template<typename T> // redop or refop
struct cache_ref;

template<typename E1, typename E2, typename RedOp>
struct cache_ref<redop<E1,E2,RedOp>> {
    using type = redop<E1,E2,RedOp>;
    using ref_type = typename redop<E1,E2,RedOp>::ref_type;
    static constexpr short id = ref_type::array_type::AID;

    cache_ref( redop<E1,E2,RedOp> r_ ) : r( r_ ) { }

    type r;
};

template<typename A, typename T, unsigned short VL>
struct cache_ref<refop<A,T,VL>> {
    using type = refop<A,T,VL>;
    using ref_type = refop<A,T,VL>;
    static constexpr short id = ref_type::array_type::AID;

    cache_ref( refop<A,T,VL> r_ ) : r( r_ ) { }

    type r;
};

template<typename S, typename U, typename C, typename DFSAOp>
struct cache_ref<dfsaop<S,U,C,DFSAOp>> {
    using type = dfsaop<S,U,C,DFSAOp>;
    using ref_type = S;
    static constexpr short id = ref_type::array_type::AID;

    cache_ref( type r_ ) : r( r_ ) { }

    type r;
};

template<use_rw_t RW, bool IsCached, typename T> // redop or refop
struct use_ref;


template<use_rw_t RW, bool IsCached, typename A, typename T, unsigned short VL>
struct use_ref<RW,IsCached,refop<A,T,VL>> {
    static constexpr use_rw_t rw_status = RW;
    static constexpr bool is_cached = IsCached;
    using type = refop<A,T,VL>;
    using ref_type = refop<A,T,VL>;
    static constexpr short id = ref_type::array_type::AID;

    use_ref( refop<A,T,VL> r_ ) : r( r_ ) { }

    template<bool rIsCached>
    use_ref<rw_status,rIsCached,ref_type> rebindIsCached() const {
	return use_ref<rw_status,rIsCached,ref_type>( r );
    }

    type r;
};

template<typename E1, typename E2, typename RedOp>
auto list_ref( redop<E1,E2,RedOp> r ) {
    return cache<cache_ref<redop<E1,E2,RedOp>>>( cache_ref<redop<E1,E2,RedOp>>(r) );
}

template<typename A, typename T, unsigned short VL>
auto list_ref( refop<A,T,VL> r ) {
    return cache<cache_ref<refop<A,T,VL>>>( cache_ref<refop<A,T,VL>>(r) );
}

template<typename S, typename U, typename C, typename DFSAOp>
auto list_ref( dfsaop<S,U,C,DFSAOp> r ) {
    return cache<cache_ref<dfsaop<S,U,C,DFSAOp>>>(
	cache_ref<dfsaop<S,U,C,DFSAOp>>( r ) );
}

/***********************************************************************
 * Operations on use lists
 ***********************************************************************/
// Find an entry in a use list. Identify key properties of the use.
template<short AID, typename UseList, typename Enable = void>
struct use_list_find;

template<short AID>
struct use_list_find<AID,cache<>> {
    static constexpr bool found = false;
};

template<short AID, typename Use0, typename... Uses>
struct use_list_find<AID,cache<Use0,Uses...>,
		     std::enable_if_t<AID == Use0::id>> {
    static constexpr bool found = true;
    static constexpr use_rw_t rw_status = Use0::rw_status;
    static constexpr bool is_cached = Use0::is_cached;
    using ref_type = typename Use0::ref_type;
};

template<short AID, typename Use0, typename... Uses>
struct use_list_find<AID,cache<Use0,Uses...>,
		     std::enable_if_t<AID != Use0::id>>
    : public use_list_find<AID,cache<Uses...>> { };

// Create one-element use list
template<use_rw_t RW, bool IsCached, typename A, typename T, unsigned short VL>
auto list_use( refop<A,T,VL> r ) {
    return cache<use_ref<RW,IsCached,refop<A,T,VL>>>(
	use_ref<RW,IsCached,refop<A,T,VL>>(r) );
}

// Update a use list entry (type change only).
// Assumes each AID is listed at most once.
template<short AID, use_rw_t RW, bool IsCached, typename UseList>
auto use_list_change( UseList use_list );

template<short AID, use_rw_t RW, bool IsCached>
auto use_list_change( cache<> use_list ) {
    return use_list;
}

template<short AID, use_rw_t RW, bool IsCached,
	 typename Use0, typename... Uses>
auto use_list_change( cache<Use0,Uses...> use_list,
		      std::enable_if_t<AID != Use0::id> * = nullptr ) {
    return cache_cons(
	car( use_list ),
	use_list_change<AID,RW,IsCached>( cdr( use_list ) ) );
}

template<short AID, use_rw_t RW, bool IsCached,
	 typename Use0, typename... Uses>
auto use_list_change( cache<Use0,Uses...> use_list,
		      std::enable_if_t<AID == Use0::id> * = nullptr ) {
    return cache_cat(
	list_use<RW,IsCached>( car( use_list ).r ), cdr( use_list ) );	
}

/***********************************************************************
 * Dispatch operations over all items of a cache/use list
 ***********************************************************************/
namespace detail {
template<std::size_t... Ns, typename... Cs,
	 typename Environment, typename value_map_type,
	 typename mask_pack_type>
__attribute__((always_inline))
inline auto cache_create( std::index_sequence<Ns...>,
			  const cache<Cs...> & desc,
			  const Environment & env, value_map_type m,
			  const mask_pack_type & mpack ) {
    auto c = create_map( std::get<Ns>( desc.t ).create( m )... );
    std::make_tuple( std::get<Ns>( desc.t ).init( env, c, m, mpack )... );
    return c;
}

template<std::size_t... Ns, typename... Cs,
	 typename value_map_type>
__attribute__((always_inline)) inline
auto cache_create_no_init( std::index_sequence<Ns...>,
			   const cache<Cs...> & desc, const value_map_type & m ) {
    return create_map( std::get<Ns>( desc.t ).create( m )... );
}

template<std::size_t... Ns, typename Cache, typename... Cs,
	 typename Environment, typename value_map_type,
	 typename mask_pack_type>
__attribute__((always_inline)) inline
void cache_init( std::index_sequence<Ns...>,
		 Cache & c, const cache<Cs...> & desc,
		 const Environment & env, value_map_type m,
		 const mask_pack_type & mpack ) {
    std::make_tuple( std::get<Ns>( desc.t ).init( env, c, m, mpack )... );
}

template<std::size_t... Ns, typename Environment,
	 typename Cache, typename... Cs,
	 typename value_map_type>
__attribute__((always_inline)) inline
void cache_clear( std::index_sequence<Ns...>,
		  const Environment & env,
		  Cache & c, const cache<Cs...> & desc, value_map_type m ) {
    std::make_tuple( std::get<Ns>( desc.t ).clear( env, c, m )... );
}

template<std::size_t... Ns, typename... Cs,
	 typename Cache, typename Environment, typename value_map_type,
	 typename mask_pack_type>
__attribute__((always_inline)) inline
void cache_commit( std::index_sequence<Ns...>,
		   const cache<Cs...> & c, Cache & cc,
		   const Environment & env,
		   const value_map_type & m,
		   const mask_pack_type & mpack ) {
    // The make_tuple call here is a hack to evaluate commit on all
    // elements of the cache.
    std::make_tuple( std::get<Ns>( c.t ).commit( env, cc, m, mpack )... );
}

template<std::size_t... Ns, typename Environment, typename... Cs,
	 typename CacheI, typename CacheF,
	 typename value_map_type>
__attribute__((always_inline)) inline
void cache_commit_if_changed( std::index_sequence<Ns...>,
			      const Environment & env,
			      const cache<Cs...> & c,
			      CacheI & cc_init,
			      CacheF & cc_final,
			      const value_map_type & m ) {
    // The make_tuple call here is a hack to evaluate commit on all
    // elements of the cache.
    std::make_tuple( std::get<Ns>( c.t ).commit_if_changed(
			 env, cc_init, cc_final, m )... );
}

template<bool Atomic, std::size_t... Ns, typename... Cs,
	 typename Environment,
	 typename cache_map_type, typename value_map_type>
__attribute__((always_inline)) inline
void cache_commit_with_reduce( std::index_sequence<Ns...>,
			       const Environment & env,
			       const cache<Cs...> & c, cache_map_type & cc,
			       const value_map_type & m ) {
    std::make_tuple( std::get<Ns>( c.t ).
		     template commit_with_reduce<Atomic>( env, cc, m )... );
}


template<std::size_t... Ns, typename... Cs,
	 typename Cache, typename value_map_type>
__attribute__((always_inline)) inline
void cache_reduce_and_commit( std::index_sequence<Ns...>,
			      const cache<Cs...> & c, Cache & cc,
			      const value_map_type & m ) {
    // The make_tuple call here is a hack to evaluate commit on all
    // elements of the cache.
    std::make_tuple( std::get<Ns>( c.t ).reduce_and_commit( cc, m )... );
}

template<std::size_t... Ns, typename... Cs,
	 typename Cache, typename value_map_type>
__attribute__((always_inline)) inline
void cache_reduce( std::index_sequence<Ns...>,
		   const cache<Cs...> & c, Cache & cc,
		   const value_map_type & m ) {
    // The make_tuple call here is a hack to evaluate commit on all
    // elements of the cache.
    std::make_tuple( std::get<Ns>( c.t ).reduce( cc, m )... );
}

// For uses only
template<unsigned long Distance, std::size_t... Ns, typename... Cs,
	 typename Cache, typename value_map_type>
__attribute__((always_inline)) inline
void cache_prefetch( std::index_sequence<Ns...>,
		     Cache & cc,
		     const cache<Cs...> & c,
		     const value_map_type & m ) {
    // The make_tuple call here is a hack to evaluate prefetch on all
    // elements of the cache.
    std::make_tuple( std::get<Ns>( c.t ).template prefetch<Distance>( cc, m )... );
}

} // namespace detail

template<typename... Cs, typename Environment, typename value_map_type,
	 typename mask_pack_type>
__attribute__((always_inline)) inline
auto cache_create( const Environment & env,
		   const cache<Cs...> & c,
		   const value_map_type & m,
		   const mask_pack_type & mpack ) {
    return detail::cache_create( std::index_sequence_for<Cs...>(),
				 c, env, m, mpack );
}

template<typename... Cs, typename Environment, typename value_map_type>
__attribute__((always_inline)) inline
auto cache_create( const Environment & env,
		   const cache<Cs...> & c,
		   const value_map_type & m ) {
    auto mpack_empty = expr::sb::mask_pack<>();
    return cache_create( env, c, m, mpack_empty );
}

template<typename... Cs, typename value_map_type>
__attribute__((always_inline)) inline
auto cache_create_no_init( const cache<Cs...> & c, const value_map_type & m ) {
    return detail::cache_create_no_init(
	std::index_sequence_for<Cs...>(), c, m );
}

template<typename Cache, typename Environment,
	 typename value_map_type, typename mask_pack_type>
__attribute__((always_inline)) inline
auto cache_init( const Environment & env, Cache & c, const cache<> & desc,
		 const value_map_type & m, const mask_pack_type & mpack ) {
}

template<typename Cache, typename... Cs, typename Environment,
	 typename value_map_type, typename mask_pack_type>
__attribute__((always_inline)) inline
auto cache_init( const Environment & env, Cache & c, const cache<Cs...> & desc,
		 const value_map_type & m, const mask_pack_type & mpack ) {
    return detail::cache_init<false>( std::index_sequence_for<Cs...>(),
				      c, desc, env, m, mpack );
}

template<typename Cache, typename... Cs, typename Environment,
	 typename value_map_type>
__attribute__((always_inline)) inline
auto cache_init( const Environment & env, Cache & c, const cache<Cs...> & desc,
		 const value_map_type & m ) {
    auto mpack_empty = expr::sb::mask_pack<>();
    return cache_init( env, c, desc, m, mpack_empty );
}

template<typename Cache, typename Environment, typename value_map_type,
	 typename mask_pack_type>
[[deprecated("duplicate def")]]
__attribute__((always_inline)) inline
auto cache_init( Cache & c, const cache<> & desc,
		 const Environment & env, const value_map_type & m,
		 const mask_pack_type & mpack ) {
}

template<typename Environment, typename Cache, typename... Cs,
	 typename value_map_type>
__attribute__((always_inline)) inline
auto cache_clear( const Environment & env, Cache & c, const cache<Cs...> & desc,
		 const value_map_type & m ) {
    return detail::cache_clear( std::index_sequence_for<Cs...>(),
				env, c, desc, m );
}

template<typename... Cs, typename Cache, typename Environment,
	 typename value_map_type, typename mask_pack_type>
__attribute__((always_inline)) inline
void cache_commit( const Environment & env,
		   const cache<Cs...> & c, Cache & cc,
		   const value_map_type & m, const mask_pack_type & mpack ) {
    detail::cache_commit( std::index_sequence_for<Cs...>(),
			  c, cc, env, m, mpack );
}

template<typename Cache, typename Environment,
	 typename value_map_type, typename mask_pack_type>
__attribute__((always_inline)) inline
void cache_commit( const Environment & env, const cache<> & c, Cache & cc,
		   const value_map_type & m, const mask_pack_type & mpack ) {
}

template<typename... Cs, typename Cache, typename Environment, typename value_map_type>
__attribute__((always_inline)) inline
void cache_commit( const Environment & env,
		   const cache<Cs...> & c, Cache & cc,
		   const value_map_type & m ) {
    auto mpack_empty = expr::sb::mask_pack<>();
    cache_commit( env, c, cc, m, mpack_empty );
}

template<typename Cache, typename Environment, typename value_map_type>
__attribute__((always_inline)) inline
void cache_commit( const Environment & env, const cache<> & c, Cache & cc,
		   const value_map_type & m ) {
}

template<typename Environment, typename... Cs, typename CacheI, typename CacheF, typename value_map_type>
__attribute__((always_inline)) inline
void cache_commit_if_changed( const Environment & env,
			      const cache<Cs...> & c,
			      CacheI & cc_init,
			      CacheF & cc_final,
			      const value_map_type & m ) {
    detail::cache_commit_if_changed( std::index_sequence_for<Cs...>(),
				     env, c, cc_init, cc_final, m );
}

template<typename... Cs,
	 typename Environment,
	 typename cache_map_type, typename value_map_type>
__attribute__((always_inline)) inline
void cache_commit_with_reduce( const Environment & env,
			       const cache<Cs...> & c, cache_map_type & cc,
			       const value_map_type & m ) {
    detail::cache_commit_with_reduce<false>( std::index_sequence_for<Cs...>(),
					     env, c, cc, m );
}

template<typename... Cs,
	 typename Environment,
	 typename cache_map_type, typename value_map_type>
__attribute__((always_inline)) inline
void cache_commit_with_reduce_atomic( const Environment & env,
				      const cache<Cs...> & c,
				      cache_map_type & cc,
				      const value_map_type & m ) {
    detail::cache_commit_with_reduce<true>( std::index_sequence_for<Cs...>(),
					    env, c, cc, m );
}

template<typename... Cs, typename Cache, typename value_map_type>
__attribute__((always_inline)) inline
void cache_reduce_and_commit( const cache<Cs...> & c, Cache & cc,
			      const value_map_type & m ) {
    detail::cache_reduce_and_commit( std::index_sequence_for<Cs...>(),
				     c, cc, m );
}

template<typename... Cs, typename Cache, typename value_map_type>
__attribute__((always_inline)) inline
void cache_reduce( const cache<Cs...> & c, Cache & cc,
		   const value_map_type & m ) {
    detail::cache_reduce( std::index_sequence_for<Cs...>(), c, cc, m );
}

template<unsigned long Distance, typename Cache, typename... Cs,
	 typename value_map_type>
__attribute__((always_inline)) inline
auto cache_prefetch( Cache & c, const cache<Cs...> & desc,
		     const value_map_type & m ) {
    detail::cache_prefetch<Distance>( std::index_sequence_for<Cs...>(),
				      c, desc, m );
}

#if DISABLE_CACHES_HARD
template<value_kind VKind, typename Expr>
static constexpr
auto extract_cacheable_refs( Expr ) {
    return cache<>(); // caches disabled means nothing is cached
}

template<value_kind VKind, typename Expr, typename Cache>
static constexpr
auto extract_cacheable_refs( Expr, Cache ) {
    return cache<>(); // caches disabled means nothing is cached
}

template<typename Expr>
static constexpr
auto extract_cacheable_refs_helper( Expr ) {
    return cache<>();
}

template<value_kind VKind, typename Expr>
static constexpr
auto extract_readonly_refs( Expr ) {
    return cache<>(); // caches disabled means nothing is cached
}

template<value_kind VKind, typename Expr, typename Cache>
static constexpr
auto extract_readonly_refs( Expr, Cache ) {
    return cache<>(); // caches disabled means nothing is cached
}

template<value_kind VKind, typename Expr, typename Cache,
	 memory_access_method = mam_cached>
static constexpr
auto rewrite_caches( Expr e, Cache ) {
    return e;
}
#endif // DISABLE_CACHES_HARD

/**********************************************************************
 * Utility: converting an initial cache structure to reduction_info
 *          objects for eligible variables.
 **********************************************************************/
template<value_kind VKind, typename CRef, typename Enable = void>
struct is_cacheable_ref : std::false_type { };

template<value_kind VKind, typename CRef>
struct is_cacheable_ref<VKind, CRef, typename std::enable_if<is_indexed_by_vk<VKind,typename CRef::ref_type>::value && ( is_redop<typename CRef::type>::value || is_dfsaop<typename CRef::type>::value || VKind == vk_src )>::type> : std::true_type { };

template<value_kind VKind, memory_access_method mam, unsigned cid>
static constexpr
auto cache_convert_rinfo( const cache<> & c ) {
    return c;
}

template<value_kind VKind, memory_access_method mam, unsigned cid, typename C0, typename... Cs>
__attribute__((always_inline))
inline auto cache_convert_rinfo(
    const cache<C0,Cs...> & c,
    typename std::enable_if<!is_cacheable_ref<VKind,C0>::value>::type * = nullptr ) {
    fail_expose<is_indexed_by_src>( car(c) );
    return cache_convert_rinfo<VKind,mam,cid>( cdr( c ) );
}

template<value_kind VKind, memory_access_method mam,
	 unsigned cid, typename C0, typename... Cs>
static constexpr
auto cache_convert_rinfo(
    const cache<C0,Cs...> & c,
    typename std::enable_if<is_cacheable_ref<VKind,C0>::value>::type * = nullptr ) {
#if ENABLE_NT_EMAP
    auto nt_r = transform_nt<mam==mam_nontemporal>( car( c ).r );
#else
    auto nt_r = car( c ).r;
#endif
    return cache_cons( make_reduction_info<cid>( nt_r ), 
		       cache_convert_rinfo<VKind,mam,cid+1>( cdr( c ) ) );
}

/**********************************************************************
 * Utility: converting an initial cache structure to reduction_info
 *          objects for eligible variables; retaining all contents.
 **********************************************************************/
template<unsigned cid>
static constexpr
auto cache_convert_use_info( const cache<> & c ) {
    return c;
}

template<unsigned cid, typename C0, typename... Cs>
__attribute__((always_inline))
inline auto cache_convert_use_info( const cache<C0,Cs...> & c ) {
    return cache_cons(
	make_use_info<cid,C0::rw_status,C0::is_cached,false>( car( c ).r ),
	cache_convert_use_info<cid+1>( cdr( c ) ) );
}

/**********************************************************************
 * Utility: converting an initial cache structure to reduction_info
 *          objects for local variables; retaining all contents.
 **********************************************************************/
template<unsigned cid>
static constexpr
auto cache_convert_local_vars( const cache<> & c ) {
    return c;
}

template<unsigned cid, typename C0, typename... Cs>
__attribute__((always_inline))
inline auto cache_convert_local_vars( const cache<C0,Cs...> & c ) {
    // The is_cached flag is not relevant to local variables.
    if constexpr ( is_local_var_v<C0::rw_status, typename C0::ref_type> )
	// For local variables, override the is_cached flag; they should all
	// have backing storage created
	return cache_cons(
	    make_use_info<cid,C0::rw_status,true,true>( car( c ).r ),
	    cache_convert_local_vars<cid+1>( cdr( c ) ) );
    else
	return cache_convert_local_vars<cid>( cdr( c ) );
}

/**********************************************************************
 * Utility: extracting all cacheable references from an expression.
 * This returns all r-values which occur exactly once, i.e., the same
 * array is not accessed through another r-value or an l-value.
 **********************************************************************/
#if ! DISABLE_CACHES_HARD
// Implementations
static constexpr
auto extract_cacheable_refs_helper( noop ) {
    return cache<>();
}

template<typename Tr, value_kind VKind>
static constexpr
auto extract_cacheable_refs_helper( value<Tr, VKind> ) {
    return cache<>();
}

template<typename Expr, typename UnOp>
static constexpr
auto extract_cacheable_refs_helper( unop<Expr,UnOp> u ) {
    return extract_cacheable_refs_helper( u.data() );
}

template<typename E1, typename E2, typename BinOp>
static constexpr
auto extract_cacheable_refs_helper( binop<E1,E2,BinOp> b ) {
    return cache_cat(
	extract_cacheable_refs_helper( b.data1() ),
	extract_cacheable_refs_helper( b.data2() ) );
}

template<typename E1, typename E2, typename E3, typename TernOp>
static constexpr
auto extract_cacheable_refs_helper( ternop<E1,E2,E3,TernOp> t ) {
    return cache_cat(
	extract_cacheable_refs_helper( t.data1() ),
	cache_cat(
	    extract_cacheable_refs_helper( t.data2() ),
	    extract_cacheable_refs_helper( t.data3() ) ) );
}

template<typename A, typename T, unsigned short VL>
static constexpr
auto extract_cacheable_refs_helper( refop<A,T,VL> r ) {
    return extract_cacheable_refs_helper( r.index() );
}

template<bool nt, typename R, typename T>
static constexpr
auto extract_cacheable_refs_helper( storeop<nt,R,T> s ) {
    return cache_cat( extract_cacheable_refs_helper( s.ref() ),
		      extract_cacheable_refs_helper( s.value() ) );
}

template<unsigned cid, typename Tr>
static constexpr
auto extract_cacheable_refs_helper( cacheop<cid,Tr> c ) {
    return cache<>();
}

template<typename E1, typename E2, typename RedOp>
static constexpr
auto extract_cacheable_refs_helper(
    redop<E1,E2,RedOp> r,
    typename std::enable_if<!is_refop<E1>::value>::type * = nullptr ) {
    auto cache1 = extract_cacheable_refs_helper( r.ref() );
    auto cache2 = extract_cacheable_refs_helper( r.val() );
    return cache_cat( cache1, cache2 );
}

template<typename E1, typename E2, typename RedOp>
static constexpr
auto extract_cacheable_refs_helper(
    redop<E1,E2,RedOp> r,
    typename std::enable_if<is_refop<E1>::value>::type * = nullptr ) {
    if constexpr ( std::is_same_v<typename E1::array_type::encoding,
		   array_encoding<void>> ) {
	// Skip ref, it should be a local var
	return extract_cacheable_refs_helper( r.val() );
    } else {
	auto cache1 = list_ref( r );
	auto cache2 = extract_cacheable_refs_helper( r.val() );
	return cache_cat( cache1, cache2 );
    }
}

template<typename S, typename U, typename C, typename DFSAOp>
static constexpr
auto extract_cacheable_refs_helper(
    dfsaop<S,U,C,DFSAOp> r,
    std::enable_if_t<!is_refop<S>::value && !is_maskrefop<S>::value> *
    = nullptr ) {
    auto cache1 = extract_cacheable_refs_helper( r.state() );
    auto cache2 = extract_cacheable_refs_helper( r.update() );
    auto cache3 = extract_cacheable_refs_helper( r.condition() );
    return cache_cat( cache1, cache_cat( cache2, cache3 ) );
}

template<typename S, typename U, typename C, typename DFSAOp>
static constexpr
auto extract_cacheable_refs_helper(
    dfsaop<S,U,C,DFSAOp> r,
    std::enable_if_t<is_refop<S>::value || is_maskrefop<S>::value> *
    = nullptr ) {
    auto cache1 = list_ref( r );
    auto cache2 = extract_cacheable_refs_helper( r.update() );
    auto cache3 = extract_cacheable_refs_helper( r.condition() );
    return cache_cat( cache1, cache_cat( cache2, cache3 ) );
}

template<value_kind VKind, memory_access_method mam, typename Expr>
static constexpr
auto extract_cacheable_refs( Expr e ) {
#if ENABLE_NT_EMAP
    return cache_convert_rinfo<VKind,mam,0>(
	cache_dedup( extract_cacheable_refs_helper( e ) ) );
#else
    return cache_convert_rinfo<VKind,mam_cached,0>(
	cache_dedup( extract_cacheable_refs_helper( e ) ) );
#endif
}

template<value_kind VKind, memory_access_method mam, typename Expr, typename... C>
static constexpr
auto extract_cacheable_refs( Expr e, cache<C...> c ) {
    auto er = extract_cacheable_refs_helper( e );
    auto cr = cache_dedup( er );
#if ENABLE_NT_EMAP
    auto ci = cache_convert_rinfo<VKind,mam,sizeof...(C)>( cr );
#else
    auto ci = cache_convert_rinfo<VKind,mam_cached,sizeof...(C)>( cr );
#endif
    return ci;
}

template<value_kind VKind, typename Expr>
static constexpr
auto extract_cacheable_refs( Expr e ) {
    return extract_cacheable_refs<VKind,mam_cached>( e );
}

template<value_kind VKind, typename Expr, typename... C>
static constexpr
auto extract_cacheable_refs( Expr e, cache<C...> c ) {
    return extract_cacheable_refs<VKind,mam_cached>( e, c );
}
#endif // DISABLE_CACHES_HARD

/**********************************************************************
 * Utility: extracting all reads and writes from an expression.
 * This is for use in vertex_map, where a straight-line piece of code
 * is executed.
 * This code assumes a strict depth-first, left-to-right evaluation order
 * for non-mutating operations, and read-before-write for storeop, redop
 * and dfsaop, in order to assess whether a store occurs before a load.
 **********************************************************************/
// Implementations
template<value_kind VKX, bool Write, bool Masked, typename LeftLdSt>
static constexpr
auto extract_uses_helper( noop, LeftLdSt left_ls ) {
    return left_ls;
}

template<value_kind VKX, bool Write, bool Masked, typename LeftLdSt,
	 typename Tr, value_kind VKind>
static constexpr
auto extract_uses_helper( value<Tr, VKind>, LeftLdSt left_ls ) {
    return left_ls;
}

template<value_kind VKX, bool Write, bool Masked, typename LeftLdSt,
	 typename Expr, typename UnOp>
static constexpr
auto extract_uses_helper( unop<Expr,UnOp> u, LeftLdSt left_ls ) {
    return extract_uses_helper<VKX,Write,Masked>( u.data(), left_ls );
}

template<value_kind VKX, bool Write, bool Masked, typename LeftLdSt,
	 typename E1, typename E2, typename BinOp>
static constexpr
auto extract_uses_helper( binop<E1,E2,BinOp> b, LeftLdSt left_ls ) {
    auto intm_ls = extract_uses_helper<VKX,Write,Masked>( b.data1(), left_ls );
    return extract_uses_helper<VKX,Write,Masked>( b.data2(), intm_ls );
}

template<value_kind VKX, bool Write, bool Masked, typename LeftLdSt, typename E1, typename E2, typename E3, typename TernOp>
static constexpr
auto extract_uses_helper( ternop<E1,E2,E3,TernOp> t, LeftLdSt left_ls ) {
    auto intm_ls1 = extract_uses_helper<VKX,Write,Masked>( t.data1(), left_ls );
    auto intm_ls2 = extract_uses_helper<VKX,Write,Masked>( t.data2(), intm_ls1 );
    return extract_uses_helper<VKX,Write,Masked>( t.data3(), intm_ls2 );
}

template<value_kind VKX, bool Write, bool Masked, typename LeftLdSt,
	 typename A, typename T, unsigned short VL>
static constexpr
auto extract_uses_helper(
    refop<A,T,VL> r, LeftLdSt left_ls,
    std::enable_if_t<!is_indexed_by_vk<VKX,refop<A,T,VL>>::value> * = nullptr
    ) {
    return extract_uses_helper<VKX,false,false>( r.index(), left_ls );
}

template<value_kind VKX, bool Write, bool Masked, typename LeftLdSt,
	 typename A, typename T, unsigned short VL>
static constexpr
auto extract_uses_helper(
    refop<A,T,VL> r, LeftLdSt left_ls,
    std::enable_if_t<is_indexed_by_vk<VKX,refop<A,T,VL>>::value> * = nullptr
    ) {
    auto intm_ls = extract_uses_helper<VKX,false,false>( r.index(), left_ls );

    using Found = use_list_find<A::AID,decltype(intm_ls)>;
    constexpr bool found = Found::found;

    // Pre-existing load/store to same array ID?
    if constexpr ( found ) {
	constexpr use_rw_t rw_status
	    = use_rw_merge_v<Found::rw_status,Write,Masked || expr_contains_mask_v<T>>;
	constexpr bool is_cached
	    = Found::is_cached
	    && !is_conflicting_ref_v<typename Found::ref_type,refop<A,T,VL>>;

	if constexpr ( Found::rw_status != rw_status
		       || Found::is_cached != is_cached ) {
	    return use_list_change<A::AID,rw_status,is_cached>( intm_ls );
	} else {
	    return intm_ls;
	}
    } else {
	// First use of array ID A::AID
	// If the index expression contains a mask, be conservative
	// and assume this involves an exposed load.
	// On the first reference, we always assume it will be cached.
	if constexpr ( !Write ) {
	    return cache_cat( list_use<use_rw_read_only,true>( r ), intm_ls );
	} else if constexpr ( Masked || expr_contains_mask_v<T> ) {
	    return cache_cat(
		list_use<use_rw_read_before_write,true>( r ), intm_ls );
	} else {
	    return cache_cat( list_use<use_rw_write_kill,true>( r ), intm_ls );
	}
    }
}

template<value_kind VKX, bool Write, bool Masked, typename LeftLdSt,
	 bool nt, typename R, typename T>
static constexpr
auto extract_uses_helper( storeop<nt,R,T> s, LeftLdSt left_ls ) {
    // Compute RHS before updating LHS
    auto intm_ls = extract_uses_helper<VKX,false,false>( s.value(), left_ls );
// Problem:
    // let<> when vmap w/ frontier creates local variable where initializer
    // has mask. That is not recognised as a local variable by is_local_var
    // ...
    return extract_uses_helper<VKX,true,Masked || expr_contains_mask_v<T>>( s.ref(), intm_ls );
}

template<value_kind VKX, bool Write, bool Masked, typename LeftLdSt,
	 unsigned cid, typename Tr>
static constexpr
auto extract_uses_helper( cacheop<cid,Tr> c, LeftLdSt left_ls ) {
    return left_ls;
}

template<value_kind VKX, bool Write, bool Masked, typename LeftLdSt,
	 typename E1, typename E2, typename RedOp>
static constexpr
auto extract_uses_helper( redop<E1,E2,RedOp> r, LeftLdSt left_ls ) {
    // Compute LHS and RHS before updating LHS
    constexpr bool M = expr_contains_mask_v<E2>;
    auto intm_ls1 = extract_uses_helper<VKX,false,false>( r.ref(), left_ls );
    auto intm_ls2 = extract_uses_helper<VKX,false,false>( r.val(), intm_ls1 );
    return extract_uses_helper<VKX,true,M>( r.ref(), intm_ls2 );
}

template<value_kind VKX, bool Write, bool Masked, typename LeftLdSt,
	 typename S, typename U, typename C, typename DFSAOp>
static constexpr
auto extract_uses_helper( dfsaop<S,U,C,DFSAOp> r, LeftLdSt left_ls ) {
    // Compute RHS and condition before updating LHS
    constexpr bool M = expr_contains_mask_v<U> || expr_contains_mask_v<C>;
    auto intm_ls1 = extract_uses_helper<VKX,false,false>( r.state(), left_ls );
    auto intm_ls2 = extract_uses_helper<VKX,false,false>( r.update(), intm_ls1 );
    auto intm_ls3 = extract_uses_helper<VKX,false,false>( r.condition(), intm_ls2 );
    return extract_uses_helper<VKX,true,Masked>( r.state(), intm_ls3 );
}

#if DISABLE_CACHES_HARD
template<value_kind VKind, typename Expr>
static constexpr
auto extract_uses( Expr e ) {
    return cache<>();
}

template<value_kind VKind, typename Expr, typename... C>
static constexpr
auto extract_uses( Expr e, cache<C...> & c ) {
    return cache<>();
}

template<typename Expr>
static constexpr
auto extract_local_vars( Expr e ) {
    return cache<>();
}

template<typename Expr, typename... C>
static constexpr
auto extract_local_vars( Expr e, cache<C...> & c ) {
    return cache<>();
}
#else

// TODO: can we remove entries where IsCached == false after all duplicates
//       and conflicting refs have been identified?
template<value_kind VKind, typename Expr>
static constexpr
auto extract_uses( Expr e ) {
    return cache_convert_use_info<0>(
	extract_uses_helper<VKind,false,false>( e, cache<>() ) );
}

template<value_kind VKind, typename Expr, typename... C>
static constexpr
auto extract_uses( Expr e, cache<C...> c ) {
    auto er = extract_uses_helper<VKind,false,false>( e, cache<>() );
    auto ci = cache_convert_use_info<sizeof...(C)>( er );
    // fail_expose<std::is_class>( ci );
    return ci;
}

template<typename Expr>
static constexpr
auto extract_local_vars( Expr e ) {
    return cache_convert_local_vars<0>(
	extract_uses_helper<vk_zero,false,false>( e, cache<>() ) );
}

template<typename Expr, typename... C>
static constexpr
auto extract_local_vars( Expr e, cache<C...> c ) {
    auto er = extract_uses_helper<vk_zero,false,false>( e, cache<>() );
    auto ci = cache_convert_local_vars<sizeof...(C)>( er );
    // fail_expose<std::is_class>( ci );
    return ci;
}
#endif // DISABLE_CACHES_HARD

/**********************************************************************
 * Utility: extracting all readonly references from an expression.
 * Note that this function returns all the r-values, but does not check
 * if the same memory locations may also be an l-value.
 **********************************************************************/
#if ! DISABLE_CACHES_HARD

template<value_kind VKind, typename Expr>
static constexpr
auto extract_readonly_refs( Expr e ) {
    return cache_convert_rinfo<VKind,mam_cached,0>(
	cache_dedup( extract_readonly_refs_helper( e ) ) );
}


// Implementations
static constexpr
auto extract_readonly_refs_helper( noop ) {
    return cache<>();
}

template<typename Tr, value_kind VKind>
static constexpr
auto extract_readonly_refs_helper( value<Tr, VKind> ) {
    return cache<>();
}

template<typename Expr, typename UnOp>
static constexpr
auto extract_readonly_refs_helper( unop<Expr,UnOp> u ) {
    return extract_readonly_refs_helper( u.data() );
}

template<typename E1, typename E2, typename BinOp>
static constexpr
auto extract_readonly_refs_helper( binop<E1,E2,BinOp> b ) {
    return cache_cat(
	extract_readonly_refs_helper( b.data1() ),
	extract_readonly_refs_helper( b.data2() ) );
}

template<typename E1, typename E2, typename E3, typename TernOp>
static constexpr
auto extract_readonly_refs_helper( ternop<E1,E2,E3,TernOp> b ) {
    return cache_cat(
	extract_readonly_refs_helper( b.data1() ),
	cache_cat(
	    extract_readonly_refs_helper( b.data2() ),
	    extract_readonly_refs_helper( b.data3() ) ) );
}

template<typename A, typename T, unsigned short VL>
static constexpr
auto extract_readonly_refs_helper( refop<A,T,VL> r ) {
    return cache_cat(
	list_ref( r ),
	extract_readonly_refs_helper( r.index() ) );
}

template<bool nt, typename R, typename T>
static constexpr
auto extract_readonly_refs_helper( storeop<nt,R,T> s ) {
    // l-value is modified, don't return it; but scan r-value
    return extract_readonly_refs_helper( s.value() );
}

template<unsigned cid, typename Tr>
static constexpr
auto extract_readonly_refs_helper( cacheop<cid,Tr> c ) {
    return cache<>();
}

template<typename E1, typename E2, typename RedOp>
static constexpr
auto extract_readonly_refs_helper(
    redop<E1,E2,RedOp> r ) {
    // l-value is modified, don't return it; but scan r-value
    return extract_readonly_refs_helper( r.val() );
}

template<typename S, typename U, typename C, typename DFSAOp>
static constexpr
auto extract_readonly_refs_helper( dfsaop<S,U,C,DFSAOp> s ) {
    // l-value is modified, don't return it; but scan r-values
    return cache_cat(
	extract_readonly_refs_helper( s.update() ),
	extract_readonly_refs_helper( s.condition() ) );
}

template<value_kind VKind, typename Expr, typename... C>
static constexpr
auto extract_readonly_refs( Expr e, cache<C...> & c ) {
    auto er = extract_readonly_refs_helper( e );
    auto cr = cache_dedup( er );
    return cache_convert_rinfo<VKind,mam_cached,sizeof...(C)>( cr );
}
#endif // DISABLE_CACHES_HARD


/**********************************************************************
 * Rewrite rules: rewriting expressions to reference cache elements
 *    instead of referencing arrays.
 **********************************************************************/
#if ! DISABLE_CACHES_HARD

namespace detail {

template<value_kind VKind, memory_access_method mam, typename Cache>
static constexpr
auto rewrite_caches( noop n, const Cache & ) {
    return n;
}

template<value_kind VKind, memory_access_method mam,
	 bool nt, typename R, typename T, typename Cache>
static constexpr
auto rewrite_caches( storeop<nt,R,T> s, const Cache & c ) {
    return make_storeop_like<nt>( rewrite_caches<VKind,mam>( s.ref(), c ),
				  rewrite_caches<VKind,mam>( s.value(), c ) );
}

template<value_kind VKind, memory_access_method mam,
	 typename Tr, value_kind VKind2, typename Cache>
static constexpr
auto rewrite_caches( value<Tr, VKind2> v, const Cache & ) {
    return v;
}

template<value_kind VKind, memory_access_method mam,
	 typename Expr, typename UnOp, typename Cache>
static constexpr
auto rewrite_caches( unop<Expr,UnOp> u, const Cache & c ) {
    return make_unop( rewrite_caches<VKind,mam>( u.data(), c ), UnOp() );
}

template<value_kind VKind, memory_access_method mam,
	 typename E1, typename E2, typename BinOp, typename Cache>
static constexpr
auto rewrite_caches( binop<E1,E2,BinOp> b, const Cache & c ) {
    return make_binop( rewrite_caches<VKind,mam>( b.data1(), c ),
		       rewrite_caches<VKind,mam>( b.data2(), c ),
		       BinOp() );
}

template<value_kind VKind, memory_access_method mam,
	 typename E1, typename E2, typename E3, typename TernOp, typename Cache>
static constexpr
auto rewrite_caches( ternop<E1,E2,E3,TernOp> t, const Cache & c ) {
    return make_ternop( rewrite_caches<VKind,mam>( t.data1(), c ),
			rewrite_caches<VKind,mam>( t.data2(), c ),
			rewrite_caches<VKind,mam>( t.data3(), c ),
			TernOp() );
}

template<value_kind VKind, memory_access_method mam,
	 unsigned cid, typename Tr, typename Cache>
static constexpr
auto rewrite_caches( cacheop<cid,Tr> o, const Cache & c ) {
    return o;
}

// Case I: indexing not using destination vertex ID
template<value_kind VKind, memory_access_method mam,
	 unsigned short tgtVL, typename A, typename T,
	 typename Cache, unsigned short VL>
static constexpr
auto rewrite_caches_ref( refop<A,T,VL> r, const Cache & c,
			 typename std::enable_if<!cache_requires_rewrite<A::AID,Cache>::value || !( is_indexed_by_vk<VKind,refop<A,T,VL>>::value || VKind == vk_zero)>::type * = nullptr ) {
    // auto ci = rewrite_caches( r.index(), c );
    // return refop<A,T,VL>( r.array(), ci );
    return make_refop(
#if ENABLE_NT_EMAP
	transform_nt<mam == mam_nontemporal>( r.array() ),
#else
	r.array(),
#endif
	rewrite_caches<VKind,mam>( r.index(), c ) );
}

// Case II: the refop has index [d] or [d w/ mask]
template<value_kind VKind, memory_access_method mam,
	 unsigned short tgtVL, typename A, typename T,
	 typename Cache, unsigned short VL>
static constexpr
auto rewrite_caches_ref( refop<A,T,VL> r, const Cache & c,
			 typename std::enable_if<cache_requires_rewrite<A::AID,Cache>::value && (is_indexed_by_vk<VKind,refop<A,T,VL>>::value || VKind == vk_zero)>::type * = nullptr ) {
    // return typename decltype(cache_select<A::AID>( c ))
    // ::template cacheop_type<tgtVL>();
    // Re-instate mask on index of refop.
    // Also allow for index to contain cached data. Normally, index would
    // be value<...,VKind>, with an optional mask attached.
    auto idx = rewrite_caches<VKind,mam>( r.index(), c );
    auto ri = cache_select<A::AID>( c );
    auto rc = typename decltype(ri)::template cacheop_type<tgtVL>(); // decltype(ri)::VL>();
    return add_mask( rc, get_mask( idx ) );
}

// Case III: the target of the refop is a cached value
// This is a separate case because A::AID is not defined if A is a cacheop
// and results in SFINAE skipping Case I, II
// TODO: should we nest cacheop in refop at all?
template<value_kind VKind, memory_access_method mam,
	 unsigned cid, typename Tr, typename U, typename Cache>
static constexpr
auto rewrite_caches_ref( refop<cacheop<cid,Tr>,U,Tr::VL> r, const Cache & c ) {
    assert( 0 && "Is this case relevant at all?" );
    return refop<cacheop<cid,Tr>,U,Tr::VL>(
	transform_nt<mam == mam_nontemporal>( r.array() ),
	rewrite_caches<VKind,mam>( r.index(), c ) );
}

template<value_kind VKind, memory_access_method mam,
	 typename A, typename T, typename Cache, unsigned short VL>
static constexpr
auto rewrite_caches( refop<A,T,VL> r, const Cache & c ) {
    return rewrite_caches_ref<VKind,mam,VL>( r, c );
}

template<value_kind VKind, memory_access_method mam,
	 typename E1, typename E2, typename RedOp, typename Cache>
static constexpr
auto rewrite_caches( redop<E1,E2,RedOp> r, const Cache & c ) {
    return make_redop( rewrite_caches<VKind,mam>( r.ref(), c ),
		       rewrite_caches<VKind,mam>( r.val(), c ),
		       RedOp() );
}

template<value_kind VKind, memory_access_method mam,
	 typename S, typename U, typename C, typename DFSAOp, typename Cache>
static constexpr
auto rewrite_caches( dfsaop<S,U,C,DFSAOp> s, const Cache & c ) {
    return make_dfsaop( rewrite_caches<VKind,mam>( s.state(), c ),
			rewrite_caches<VKind,mam>( s.update(), c ),
			rewrite_caches<VKind,mam>( s.condition(), c ),
			DFSAOp() );
}

} // namespace detail

template<value_kind VKind, typename Expr, typename Cache>
static constexpr
auto rewrite_caches( Expr e, const Cache & c ) {
    return detail::rewrite_caches<VKind,mam_cached>( e, c );
}

template<value_kind VKind, memory_access_method mam,
	 typename Expr, typename Cache>
static constexpr
auto rewrite_caches( Expr e, const Cache & c ) {
    return detail::rewrite_caches<VKind,mam>( e, c );
}

#endif // DISABLE_CACHES_HARD

} // namespace expr


#endif // GRAPTOR_DSL_CACHE_H

