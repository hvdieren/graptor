// -*- c++ -*-
#ifndef GRAPTOR_DSL_EMAP_GRAPHCSX_CSR_H
#define GRAPTOR_DSL_EMAP_GRAPHCSX_CSR_H

#include "graptor/dsl/eval/environment.h"
#include "graptor/dsl/emap/utils.h"

namespace GraphCSx_csr {

template<bool Assoc, typename Expr>
auto ownrd_push_extract_cacheable_refs( Expr expr0 ) {
    if constexpr ( Assoc )
	return expr::extract_cacheable_refs<expr::vk_dst>( expr0 );
    else
	return expr::cache<>();
}

#if 0
template<typename EMapConfig, bool is_ownwr, typename GraphType,
	 typename EIDRetriever, typename Operator>
__attribute__((always_inline))
static inline void DBG_NOINLINE csr_loop(
    const GraphType & G, const EIDRetriever & eid_retriever,
    Operator & op, const partitioner & part,
    typename std::enable_if<!Operator::is_scan>::type * = nullptr ) {
    // If the operator does convergence checking, and is idempotent,
    // load the old value such that we can accelerate convergence checking
    static constexpr bool Assoc = has_redop_operator_t<Operator>::value;

    const VID n = G.numVertices();

    auto pid = expr::value<simd::ty<VID,1>,expr::vk_pid>();
    auto v_src = expr::value<simd::ty<VID,1>,expr::vk_src>();
    auto v_dst = expr::value<simd::ty<VID,1>,expr::vk_dst>();
    auto v_edge = expr::value<simd::ty<EID,1>,expr::vk_edge>();

    auto expr0 = op.relax( v_src, v_dst, v_edge );
    auto vop0 = op.vertexop( v_dst );

    auto licm = expr::licm_split_main( expr0 );
    auto expr1 = licm.le();
    auto rexpr0 = licm.pe();
    auto vop1 = append_pe( rexpr0, vop0 );

    auto d_cache = ownrd_push_extract_cacheable_refs<Assoc>( expr0 );
    auto expr2 = expr::rewrite_caches<expr::vk_dst>( expr1, d_cache );

    auto s_cache = expr::extract_uses<expr::vk_src>( expr2, d_cache );
    auto expr3 = expr::rewrite_caches<expr::vk_src>( expr2, s_cache );

    auto l_cache = expr::extract_local_vars(
	expr3, expr::cache_cat( d_cache, s_cache ) );
    auto expr4 = expr::rewrite_caches<expr::vk_zero>( expr3, l_cache );

    auto expr = expr::rewrite_mask_main( expr4 );

#if ELIDE_LOAD_ZERO
    // If source has a property that implies convergence, i.e., it supersedes
    // any other value, no need to read those values, can simply store.
    auto aexpr0 = op.active( v_src );
    auto aexpr1 = expr::rewrite_caches<expr::vk_src>( aexpr0, s_cache );
    auto aexpr = expr::rewrite_mask_main( aexpr1 );

    auto zexpr = expr::rewrite_redop_to_store( expr );
#endif

#if CCSR_ELIDE_ACTV
    auto adexpr0 = op.active( v_dst );
    auto adexpr1 = expr::rewrite_caches<expr::vk_dst>( adexpr0, d_cache );
    auto adexpr = expr::rewrite_mask_main( adexpr1 );
#endif

    auto fexpr0 = op.enabled( v_src );
#if CCSR_ELIDE_SLD
    auto sd_cache = cache_cat( s_cache, d_cache );
    auto f_cache = expr::extract_uses<expr::vk_src>( fexpr0, sd_cache );
#else
    auto f_cache = expr::cache<>();
#endif
    auto fexpr1 = expr::rewrite_caches<expr::vk_src>( fexpr0, f_cache );
    auto fexpr = expr::rewrite_mask_main( fexpr1 );

    // Post-processing
    auto vop = expr::rewrite_mask_main( vop1 );

    // fail_expose<std::is_class>( expr );

    static constexpr bool ID = expr::is_idempotent<decltype(expr)>::value;
    static constexpr bool InitZero
	= Assoc && ( !has_active_t<Operator>::value || !ID );
    static constexpr bool Atomic = /*!Assoc &&*/ !is_ownwr;

    auto env = expr::eval::create_execution_environment_with(
	op.get_ptrset(),
	s_cache, d_cache, l_cache, f_cache, expr,
#if ELIDE_LOAD_ZERO
	aexpr,
#endif
	vop, fexpr );

    using Cfg = std::decay_t<decltype(op.get_config())>;

    map_partition<Cfg::is_parallel()>( part, [&]( unsigned int p ) {
	auto pid = simd::create_scalar( (VID)p );

	auto ps = G.part_vertex_begin( part, p );
	auto pe = G.part_vertex_end( part, p );
	for( auto ii=ps; ii != pe; ++ii ) { // owner-reads partitioning
	    VertexInfo vi = *ii;
	    VID i = vi.v;
	    VID deg = vi.degree;
	    const VID * out = vi.neighbours;
	    auto src = simd::create_scalar( i );

	    auto m_src = expr::create_value_map_new2<1,expr::vk_src>( src );
	    auto c = cache_create_no_init(
		cache_cat( cache_cat( s_cache, d_cache ),
			   cache_cat( l_cache, f_cache ) ), m_src );
#if CCSR_ELIDE_SLD
	    cache_init( env, c, cache_cat( s_cache, f_cache ), m_src );
#endif

	    // fail_expose<std::is_class>( expr );

	    // Frontier check
	    if( !env.evaluate_bool( c, m_src, fexpr ) )
		continue;

#if ELIDE_LOAD_ZERO
	    if( env.evaluate_bool( c, m_src, aexpr ) ) {
#endif

	    for( VID j=0; j < deg; ++j ) {
		auto dst = simd::create_scalar( out[j] );

		auto m = expr::create_value_map_new2<1,expr::vk_src,expr::vk_dst>(
		    src, dst );
#if !CCSR_ELIDE_SLD
		cache_init( env, c, s_cache, m );
#endif
		if constexpr ( InitZero )
		    cache_clear( env, c, d_cache, m );
		else
		    cache_init( env, c, d_cache, m );

#if CCSR_ELIDE_ACTV
		if( !env.evaluate_bool( c, m, adexpr ) )
		    continue;
#endif

		auto mod = env.template evaluate<Atomic>( c, m, expr );

#if CCSR_ELIDE_WB
		constexpr size_t n_elms
		    = std::tuple_size<typename decltype(d_cache)::type>::value;
		// Elision of storing back result is correct only when
		// a single element is cached, as only one changed element
		// is reflected in the return value of expr::evaluate
		static_assert( n_elms <= 1, "oops" );
		if( n_elms > 1 || mod.mask().data() ) {
		    if constexpr ( Assoc ) {
			if constexpr ( is_ownwr ) // owner-writes partitioning
			    cache_commit_with_reduce( env, d_cache, c, m );
			else
			    cache_commit_with_reduce_atomic( env, d_cache, c, m );
		    } else
			cache_commit( env, d_cache, c, m );
		}
#else
		// Atomic not necessary if owner-writes (CCSR)
		if constexpr ( Assoc ) {
		    if constexpr ( is_ownwr ) // owner-writes partitioning
			cache_commit_with_reduce( env, d_cache, c, m );
		    else
			cache_commit_with_reduce_atomic( env, d_cache, c, m );
		} else
		    cache_commit( env, d_cache, c, m );
#endif
	    }

#if ELIDE_LOAD_ZERO
	    } else { // Converged, simply store value
		const VID * out = &edge[idx[i]];
		for( VID j=0; j < deg; ++j ) {
		    auto dst = simd::create_scalar( out[j] );

		    auto m = expr::create_value_map_new2<1,expr::vk_src,expr::vk_dst>(
			src, dst );
#if !CCSR_ELIDE_SLD
		    cache_init( env, c, s_cache, m );
#endif
		    // Do not read vertex property of destination
		    cache_clear( env, c, d_cache, m );

		    auto mod = expr::evaluate<true>( c, m, zexpr );

		    cache_commit( env, d_cache, c, m );
		}
	    }
#endif
	}
    } );

    map_partition<Cfg::is_parallel()>( part, [&]( unsigned int p ) {
	expr::cache<> c;
        GraptorCSRVertexOp<1>( env, p, part, vop, c );
     } );
}
#endif

template<typename EMapConfig, bool is_ownwr, typename GraphType,
	 typename EIDRetriever, typename Operator>
__attribute__((always_inline))
static inline void DBG_NOINLINE csr_loop(
    const GraphType & G, const EIDRetriever & eid_retriever,
    Operator & op, const partitioner & part ) { // ,
    // typename std::enable_if<Operator::is_scan>::type * = nullptr ) {
    // If the operator does convergence checking, and is idempotent,
    // load the old value such that we can accelerate convergence checking
    static constexpr bool Assoc = false; // has_redop_operator_t<Operator>::value;

    const VID n = G.numVertices();

    auto pid = expr::value<simd::ty<VID,1>,expr::vk_pid>();
    auto v_src = expr::value<simd::ty<VID,1>,expr::vk_src>();
    auto v_dst = expr::value<simd::ty<VID,1>,expr::vk_dst>();
    auto v_edge = expr::value<simd::ty<EID,1>,expr::vk_edge>();

    auto expr0 = op.relax( v_src, v_dst, v_edge );
    auto vop0 = op.vertexop( v_dst );

    auto licm = expr::template licm_split<is_ownwr>( expr0 );
    auto expr1 = licm.le();
    auto rexpr0 = licm.pe();
    auto vop1 = append_pe( rexpr0, vop0 );

    auto d_cache = ownrd_push_extract_cacheable_refs<Assoc>( expr0 );
    auto expr2 = expr::rewrite_caches<expr::vk_dst>( expr1, d_cache );

    auto s_cache = expr::extract_uses<expr::vk_src>( expr2, d_cache );
    auto expr3 = expr::rewrite_caches<expr::vk_src>( expr2, s_cache );

    auto l_cache = expr::extract_local_vars(
	expr3, expr::cache_cat( d_cache, s_cache ) );
    auto expr4 = expr::rewrite_caches<expr::vk_zero>( expr3, l_cache );

    auto expr5 = expr::rewrite_mask_main( expr4 );

#if ELIDE_LOAD_ZERO
    // If source has a property that implies convergence, i.e., it supersedes
    // any other value, no need to read those values, can simply store.
    auto aexpr0 = op.active( v_src );
    auto aexpr1 = expr::rewrite_caches<expr::vk_src>( aexpr0, s_cache );
    auto aexpr = expr::rewrite_mask_main( aexpr1 );

    auto zexpr = expr::rewrite_redop_to_store( expr );
#endif

#if CCSR_ELIDE_ACTV
    auto adexpr0 = op.active( v_dst );
    auto adexpr1 = expr::rewrite_caches<expr::vk_dst>( adexpr0, d_cache );
    auto adexpr = expr::rewrite_mask_main( adexpr1 );
#endif

    auto fexpr0 = op.enabled( v_src );
#if CCSR_ELIDE_SLD
    auto sd_cache = cache_cat( s_cache, d_cache );
    auto f_cache = expr::extract_uses<expr::vk_src>( fexpr0, sd_cache );
#else
    auto f_cache = expr::cache<>();
#endif
    auto fexpr1 = expr::rewrite_caches<expr::vk_src>( fexpr0, f_cache );
    auto fexpr = expr::rewrite_mask_main( fexpr1 );

    // Post-processing
    auto accum = expr::extract_accumulators( vop1 );
// TODO: spread out accumulators for different partitions on different cache blocks ???
    auto vop2 = expr::rewrite_privatize_accumulators( vop1, part, accum, pid );
    auto pvop = expr::accumulate_privatized_accumulators( pid, accum );
    auto pvopf = expr::final_accumulate_privatized_accumulators( pid, accum );
    auto vop = expr::rewrite_mask_main( vop2 );

    auto env = expr::eval::create_execution_environment_with(
	op.get_ptrset(),
	s_cache, d_cache, l_cache, f_cache, accum, expr5,
#if ELIDE_LOAD_ZERO
	aexpr,
#endif
	vop, fexpr, pvop, pvopf );

    auto expr = rewrite_internal( expr5 );

    static constexpr bool ID = expr::is_idempotent<decltype(expr)>::value;
    static constexpr bool InitZero
	= Assoc && ( !has_active_t<Operator>::value || !ID );
    static constexpr bool Atomic = /*!Assoc &&*/ !is_ownwr;

    using Cfg = std::decay_t<decltype(op.get_config())>;

    map_partition<Cfg::is_parallel()>( part, [&]( unsigned int p ) {
	simd_vector<VID,1> pid( p );

	EID estart = part.edge_start_of( p );

	auto ps = G.part_vertex_begin( part, p );
	auto pe = G.part_vertex_end( part, p );
	for( auto ii=ps; ii != pe; ++ii ) { // owner-reads partitioning
	    VertexInfo vi = *ii;
	    VID i = vi.v;
	    VID deg = vi.degree;
	    const VID * out = vi.neighbours;
	    auto src = simd::create_scalar( i );

	    auto m_src = expr::create_value_map_new2<1,expr::vk_src>( src );
	    auto c = cache_create_no_init(
		cache_cat( cache_cat( s_cache, d_cache ),
			   cache_cat( l_cache, f_cache ) ), m_src );
#if CCSR_ELIDE_SLD
	    cache_init( c, cache_cat( s_cache, f_cache ), m_src );
#endif

	    EID estartv = estart;
	    estart += deg;

	    // fail_expose<std::is_class>( expr );

	    // Frontier check
	    if( !env.evaluate_bool( c, m_src, fexpr ) )
		continue;

#if ELIDE_LOAD_ZERO
	    if( env.evaluate_bool( c, m_src, aexpr ) ) {
#endif

	    for( VID j=0; j < deg; ++j ) {
		auto dst = simd::create_scalar( out[j] );
		auto edge = simd::create_scalar( estartv+(EID)j );

		auto m = expr::create_value_map_new2<
		    1,expr::vk_src,expr::vk_dst,expr::vk_edge>(
			src, dst, edge );
#if !CCSR_ELIDE_SLD
		cache_init( env, c, s_cache, m );
#endif
		if constexpr ( InitZero )
		    cache_clear( env, c, d_cache, m );
		else
		    cache_init( env, c, d_cache, m );

#if CCSR_ELIDE_ACTV
		if( !env.evaluate_bool( c, m, adexpr ) )
		    continue;
#endif

		auto mod = env.template evaluate<Atomic>( c, m, expr );

#if CCSR_ELIDE_WB
		constexpr size_t n_elms
		    = std::tuple_size<typename decltype(d_cache)::type>::value;
		// Elision of storing back result is correct only when
		// a single element is cached, as only one changed element
		// is reflected in the return value of expr::evaluate
		static_assert( n_elms <= 1, "oops" );
		if( n_elms > 1 || mod.mask().data() ) {
		    if constexpr ( Assoc ) {
			if constexpr ( is_ownwr ) // owner-writes partitioning
			    cache_commit_with_reduce( env, d_cache, c, m );
			else
			    cache_commit_with_reduce_atomic( env, d_cache, c, m );
		    } else
			cache_commit( d_cache, c, m );
		}
#else
		if constexpr ( Assoc ) {
		    if constexpr ( is_ownwr ) // owner-writes partitioning
			cache_commit_with_reduce( env, d_cache, c, m );
		    else
			cache_commit_with_reduce_atomic( env, d_cache, c, m );
		} else
		    cache_commit( env, d_cache, c, m );
#endif
	    }

#if ELIDE_LOAD_ZERO
	    } else { // Converged, simply store value
		const VID * out = &edge[idx[i]];
		for( VID j=0; j < deg; ++j ) {
		    auto dst = simd::create_scalar( out[j] );

		    auto m = expr::create_value_map_new2<1,expr::vk_src,expr::vk_dst>(
			src, dst );
#if !CCSR_ELIDE_SLD
		    cache_init( env, c, s_cache, m );
#endif
		    // Do not read vertex property of destination
		    cache_clear( env, c, d_cache, m );

		    auto mod = expr::evaluate<true>( c, m, zexpr );

		    cache_commit( d_cache, c, m );
		}
	    }
#endif
	}
    } );

    map_partition<Cfg::is_parallel()>( part, [&]( unsigned int p ) {
	expr::cache<> c;
        GraptorCSRVertexOp<1>( env, p, part, vop, c );
     } );

    // Scan acros partitions
    {
	int np = part.get_num_partitions();
	for( int p=1; p < np; ++p ) {
#if ACCUM_PAD
	    auto pp = simd::template create_set1inc<VID,1,true>( p*16 );
#else
	    auto pp = simd::template create_set1inc<VID,1,true>( p*1 );
#endif
	    auto m = expr::create_value_map2<1,1,expr::vk_pid>( pp );
	    expr::cache<> c;
	    env.evaluate( c, m, pvop );
	}
	{
	    auto pp = simd::template create_set1inc0<VID,1>();
	    auto m = expr::create_value_map2<1,1,expr::vk_pid>( pp );
	    expr::cache<> c;
	    env.evaluate( c, m, pvopf );
	}
    }

    // Free up memory
    accum_destroy( accum );
}

} // namespace GraphCSx_csr

template<typename EMapConfig, typename Operator>
GG_INLINE
static inline void emap_push(
    const GraphVEBOPartCCSR & G, Operator & op, const partitioner & part ) {
    constexpr unsigned short VL = EMapConfig::VL;
    static_assert( VL == 1, "GraphCSx designed for scalar execution only" );
    GraphCSx_csr::csr_loop<EMapConfig,true>(
	G, G.get_eid_retriever(), op, part );
}

// Assumes symmetric graph
template<typename EMapConfig, typename Operator>
GG_INLINE
static inline void emap_push(
    const GraphGGVEBO & G, Operator & op, const partitioner & part ) {
    constexpr unsigned short VL = EMapConfig::VL;
    static_assert( VL == 1, "GraphCSx designed for scalar execution only" );
    assert( G.isSymmetric() && "This interpretes graph in inverse direction" );

    GraphCSx_csr::csr_loop<EMapConfig,false>(
	G, G.get_eid_retriever(), op, part );
}

#endif // GRAPTOR_DSL_EMAP_GRAPHCSX_CSR_H
