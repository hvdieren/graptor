// -*- c++ -*-
#ifndef GRAPTOR_DSL_EMAP_GRAPHCSX_CSC_H
#define GRAPTOR_DSL_EMAP_GRAPHCSX_CSC_H

#include "graptor/dsl/eval/environment.h"
#include "graptor/dsl/comp/is_idempotent.h"

namespace GraphCSx_csc {

template<bool OwnWr, typename Expr>
auto ownrd_pull_extract_cacheable_refs( Expr expr0 ) {
    if constexpr( OwnWr )
	return expr::extract_cacheable_refs<expr::vk_dst>( expr0 );
    else
	return expr::cache<>();
}

template<bool Idempotent, bool OwnWr>
struct cache_ops {
    template<typename Environment, typename Cache, typename VCache,
	     typename ValueMap, typename ValueMap1>
    // TODO: for low-degree vertices, it is probably better to not cache the
    //       values
    static void
    construct( const Environment & env,
	       const VCache & vcaches, Cache & c,
	       const ValueMap & m, const ValueMap1 & m1 ) {
	if constexpr ( Idempotent )
	    cache_clear( env, c, vcaches, m ); // set to zero of enclosing redop
	else
	    cache_init( env, c, vcaches, m ); // initialise with stored value
    }
    template<typename Environment, typename Cache, typename VCache,
	     typename ValueMap, typename ValueMap1>
    static void
    destruct( const Environment & env,
	      const VCache & vcaches, Cache & c,
	      const ValueMap & m, const ValueMap1 & m1 ) {
	if constexpr ( Idempotent ) {
	    if constexpr ( OwnWr )
		cache_commit_with_reduce( env, vcaches, c, m ); // reduce with value stored in memory
	    else
		cache_commit_with_reduce_atomic( env, vcaches, c, m ); // reduce with value stored in memory
	} else {
	    cache_commit( env, vcaches, c, m ); // write back to memory
	}
    }
};

template<bool Idempotent, bool Assoc, bool is_ownwr,
	 typename EIDRetriever, typename Environment,
	 typename Expr, typename AExpr, typename RExpr, typename UExpr,
	 typename Cache,
	 typename CacheDescriptor, typename UseDescriptor>
GG_INLINE inline
void
process_in_v_e_simd1_VL1( const VID *in, EID deg,
			  VID scalar_dst, int scalar_pid, EID scalar_eid,
			  const EIDRetriever & eid_retriever,
			  const Environment & env,
			  const Expr & e, const AExpr & ea,
			  const RExpr & er, const UExpr & eu,
			  Cache & c,
			  const CacheDescriptor & cache_desc,
			  const UseDescriptor & use_desc ) {
#if CSC_ELIDE_WB_ZERO
    constexpr bool InitZero = true;
#else
    constexpr bool InitZero = false;
#endif
    constexpr bool Atomic = /*!Assoc &&*/ !is_ownwr;

    EID j;

    auto dst = simd::create_scalar( (VID)scalar_dst );
    auto pid = simd::create_scalar( (VID)scalar_pid );

    auto m = expr::create_value_map_new<1>(
	expr::create_entry<expr::vk_dst>( dst ),
	expr::create_entry<expr::vk_pid>( pid ) );

    // TODO: can we initialise cache to "zero" of redop, then reduce in
    //       commit only if value changed?
    // auto c = cache_create_no_init( cache_desc, m );
    cache_ops<InitZero,is_ownwr>::construct( env, cache_desc, c, m, m );

#if CSC_ELIDE_ACTV
    // Won't combine well with CSC_ELIDE_WB_ZERO
    if( !env.evaluate_bool( c, m, ea ) )
	return;
#endif // CSC_ELIDE_ACTV
    
#if CSC_ELIDE_WB_CHANGED
    auto c_initial = expr::map_shift<c.num_entries>( c );
#endif

    static_assert( !std::is_same<void, typename Expr::type>::value,
		   "type must be non-void" );
    // typename simd_vector<typename Expr::type, 1>::simd_mask_type output
    // = simd_vector<typename Expr::type, 1>::false_mask();
    using output_type = simd::container<
	simd::ty<typename Expr::type, Expr::VL>>;
    auto output = output_type::false_mask();

    for( j=0; j < deg; j++ ) {
	// Check all lanes are active.
	// Type-check on the expression to see if it is constant true.
	// If so, omit the check.
	//if( !expr::evaluate_bool( c, m, ea ) )
	if( !env.evaluate_bool( c, m, ea ) )
	    goto vertex_op;

	auto src = simd::create_scalar( in[j] );
	// TODO: edge-eid recovery is incorrect
	// EID seid = eid_retriever.get_edge_eid( in[j], j );
	EID seid = scalar_eid + j;
	auto eid = simd::template create_scalar<simd::ty<EID,1>>( seid );
	auto m = expr::create_value_map_new<1>(
	    expr::create_entry<expr::vk_src>( src ),
	    expr::create_entry<expr::vk_dst>( dst ),
	    expr::create_entry<expr::vk_edge>( eid ),
	    expr::create_entry<expr::vk_pid>( pid ) );
	cache_init( env, c, use_desc, m ); // partial init of uses (src)
	auto rval_output = env.template evaluate<Atomic>( c, m, e ); // rvalue space
	auto mpack = rval_output.mpack();
	output.lor_assign( mpack.template get_mask<simd::detail::mask_bool_traits>() );
    }

    // Self-edge - only in case of WB_ZERO and certain benchmarks (idempotent!)
    if constexpr ( InitZero && Idempotent ) {
	if( env.evaluate_bool( c, m, ea ) ) {
	    auto m = expr::create_value_map_new<1>(
		expr::create_entry<expr::vk_src>( dst ),
		expr::create_entry<expr::vk_dst>( dst ),
		expr::create_entry<expr::vk_pid>( pid ) );
	    cache_init( env, c, use_desc, m ); // partial init of uses (src)
	    auto rval_output = env.template evaluate<Atomic>( c, m, e );
	    auto mpack = rval_output.mpack();
	    output.lor_assign( mpack.template get_mask<simd::detail::mask_bool_traits>() );
	}
    }

vertex_op:
    // Code that was moved out of loop.
    // Perform this per vertex only in owner-writes mode
    if( is_ownwr ) {
	auto mr = expr::create_value_map_new<1>(
	    expr::create_entry<expr::vk_dst>( dst ),
	    expr::create_entry<expr::vk_smk>( output ),
	    expr::create_entry<expr::vk_pid>( pid ) );
	env.evaluate( c, mr, er );
    }

#if CSC_ELIDE_WB
/*
    constexpr size_t n_elms
	= std::tuple_size<typename std::remove_reference<decltype(cache_desc)>::type::type>::value;
    static_assert( n_elms <= 1, "oops" );
*/

#if CSC_ELIDE_WB_ZERO
    cache_ops<InitZero,is_ownwr>::destruct( env, cache_desc, c, m, m );
#elif CSC_ELIDE_WB_ALWAYS
    if( output.data() )
	cache_commit( env, cache_desc, c, m );
#elif CSC_ELIDE_WB_CHANGED
    cache_commit_if_changed( env, cache_desc, c_initial, c, m );
#else
    if constexpr ( expr::is_constant_true<UExpr>::value ) {
	if( output.data() )
	    cache_commit( env, cache_desc, c, m );
    } else {
	auto mu = expr::create_value_map_new<1>(
	    expr::create_entry<expr::vk_dst>( dst ),
	    expr::create_entry<expr::vk_pid>( pid ) );
	bool upd = env.evaluate_bool( c, mu, eu );
	if( upd )
	    cache_ops<InitZero,is_ownwr>::destruct( env, cache_desc, c, m, m );
    }
#endif

#else
    // No WB elision
    cache_ops<InitZero,is_ownwr>::destruct( env, cache_desc, c, m, m );
#endif
}


#if 0
// CSC scalar edge map with fused vertex map operation
template<typename EMapConfig, bool is_ownwr, typename GraphType,
	 typename EIDRetriever, typename Operator>
void DBG_NOINLINE csc_loop(
    const GraphType & G,
    const EIDRetriever & eid_retriever,
    Operator & op, const partitioner & part,
    typename std::enable_if<!Operator::is_scan>::type * = nullptr ) {
    // If the operator does convergence checking, and is idempotent,
    // load the old value such that we can accelerate convergence checking
    static constexpr bool Assoc = has_redop_operator_t<Operator>::value;

    auto vexpr0 = op.relax( expr::value<simd::ty<VID,1>,expr::vk_src>(),
			    expr::value<simd::ty<VID,1>,expr::vk_dst>(),
			    expr::value<simd::ty<EID,1>,expr::vk_edge>() );
    auto vop0 = op.vertexop( expr::value<simd::ty<VID,1>,expr::vk_dst>() );
#if CSC_ELIDE_WB_DIFF
    auto uexpr0 = op.different( expr::value<simd::ty<VID,1>,expr::vk_dst>() );
#else
    auto uexpr0 = op.update( expr::value<simd::ty<VID,1>,expr::vk_dst>() );
#endif

    // LICM: only inner-most redop needs to be performed inside loop.
    //       the other relates to the new frontier and can be carried over
    //       through registers as opposed to variables.
    auto licm = expr::licm_split_main( vexpr0 );
    auto vexpr1 = licm.le();
    auto rexpr0 = licm.pe();
    auto rexpr1 = append_pe( rexpr0, vop0 );

    // Loop part
#if DISABLE_CACHING
    auto vcaches_dst = expr::cache<>();
    auto vcaches_use = expr::cache<>();
#else
    auto vcaches_dst = ownrd_pull_extract_cacheable_refs<is_ownwr>( vexpr1 );
    auto vcaches_use0 = expr::extract_uses<expr::vk_src>( vexpr1, vcaches_dst );
#endif
    // This cannot be disabled
    auto vcaches_let = expr::extract_local_vars( vexpr1,
						 cache_cat( vcaches_dst, vcaches_use0 ) );
    // auto vcaches = expr::cache_cat( vcaches_dst, vcaches_let );
    auto vcaches = vcaches_dst;
    auto vcaches_use = expr::cache_cat( vcaches_use0, vcaches_let );

    auto vexpr2 = expr::rewrite_caches<expr::vk_dst>( vexpr1, vcaches_dst );
    auto vexpr3 = expr::rewrite_caches<expr::vk_src>( vexpr2, vcaches_use );
    auto vexpr4 = expr::rewrite_caches<expr::vk_zero>( vexpr3, vcaches_let );
    auto vexpr5 = expr::rewrite_vectors_main( vexpr4 );
    auto vexpr = expr::rewrite_mask_main( vexpr5 );

    // Post-loop part
    auto rexpr2 = expr::rewrite_caches<expr::vk_dst>( rexpr1, vcaches_dst );
    auto rexpr3 = expr::rewrite_caches<expr::vk_src>( rexpr2, vcaches_use );
    auto rexpr4 = expr::rewrite_caches<expr::vk_zero>( rexpr3, vcaches_let );
    auto rexpr5 = expr::rewrite_vectors_main( rexpr4 );
    auto rexpr = expr::rewrite_mask_main( rexpr5 );

    // Update condition
    auto uexpr1 = expr::rewrite_caches<expr::vk_dst>( uexpr0, vcaches_dst );
    auto uexpr2 = expr::rewrite_vectors_main( uexpr1 );
    auto uexpr = expr::rewrite_mask_main( uexpr2 );

    // Loop termination condition (active check)
    auto aexpr0 = op.active( expr::value<simd::ty<VID,1>,expr::vk_dst>() );
    // auto aexpr1 = expr::make_unop_reduce( aexpr0, expr::redop_logicalor() );
    auto aexpr2 = expr::rewrite_caches<expr::vk_dst>( aexpr0, vcaches_dst );
    auto aexpr3 = expr::rewrite_vectors_main( aexpr2 );
    auto aexpr = expr::rewrite_mask_main( aexpr3 );

    // Override pointer for aid_eweight with the relevant permuation of the
    // weights for the G graph.
    auto ew_pset = expr::create_map2<expr::aid_eweight>(
	G.getWeights() ? G.getWeights()->get() : nullptr );
					 
    auto env = expr::eval::create_execution_environment_with(
	op.get_ptrset( ew_pset ),
	vcaches, vexpr, aexpr, rexpr, uexpr );

    using Cfg = std::decay_t<decltype(op.get_config())>;

    map_partition<Cfg::is_parallel()>( part, [&]( int p ) {
	    auto m_empty = expr::create_value_map_new2<1>();

	    // A single partition is processed here. Use a sequential loop.
	    auto ps = G.part_vertex_begin( part, p );
	    auto pe = G.part_vertex_end( part, p );
	    EID e = part.edge_start_of( p );
	    for( auto ii=ps; ii != pe; ++ii ) {
		VertexInfo vi = *ii;
		VID i = vi.v;
		VID deg = vi.degree;
		const VID * ngh = vi.neighbours;

		auto c = cache_create_no_init(
		    cache_cat( vcaches, vcaches_use ), m_empty );

		constexpr bool ID = expr::is_idempotent<decltype(vexpr)>::value;
		process_in_v_e_simd1_VL1<ID,Assoc,is_ownwr>(
		    ngh, deg, i, p, e, eid_retriever, env,
		    rewrite_internal( vexpr ),
		    rewrite_internal( aexpr ),
		    rewrite_internal( rexpr ),
		    rewrite_internal( uexpr ),
		    c, vcaches, vcaches_use );

		e += deg;
	    }
	} );
}
#endif

// CSC scalar edge map with fused vertex scan operation
template<typename EMapConfig, bool is_ownwr, typename GraphType,
	 typename EIDRetriever, typename Operator>
void DBG_NOINLINE csc_loop(
    const GraphType & G,
    const EIDRetriever & eid_retriever,
    Operator & op, const partitioner & part ) { // ,
    // typename std::enable_if<Operator::is_scan>::type * = nullptr ) {
    // If the operator does convergence checking, and is idempotent,
    // load the old value such that we can accelerate convergence checking
    static constexpr bool Assoc = has_redop_operator_t<Operator>::value;

    auto pid = expr::value<simd::ty<VID,1>,expr::vk_pid>();

    auto vexpr0 = op.relax( expr::value<simd::ty<VID,1>,expr::vk_src>(),
			    expr::value<simd::ty<VID,1>,expr::vk_dst>(),
			    expr::value<simd::ty<EID,1>,expr::vk_edge>() );
    auto vop0 = op.vertexop( expr::value<simd::ty<VID,1>,expr::vk_dst>() );
    auto accum = expr::extract_accumulators( vop0 );
// TODO: spread out accumulators for different partitions on different cache blocks ???
    expr::accum_create( part, accum );
    auto vop1 = expr::rewrite_privatize_accumulators( vop0, part, accum, pid );
    auto pvop = expr::accumulate_privatized_accumulators( pid, accum );
    auto pvopf = expr::final_accumulate_privatized_accumulators( pid, accum );

    // TODO: accumulator values, indexed by pid, should be cached in rexpr

#if CSC_ELIDE_WB_DIFF
    auto uexpr0 = op.different( expr::value<simd::ty<VID,1>,expr::vk_dst>() );
#else
    auto uexpr0 = op.update( expr::value<simd::ty<VID,1>,expr::vk_dst>() );
#endif

    //       also add caches at outer loop level for accumulators indexed
    //       by vk_pid

    // LICM: only inner-most redop needs to be performed inside loop.
    //       the other relates to the new frontier and can be carried over
    //       through registers as opposed to variables.
    auto licm = expr::licm_split_main( vexpr0 );
    auto vexpr1 = licm.le();
    auto rexpr0 = licm.pe();
    auto rexpr1 = append_pe( rexpr0, vop1 );

#if DISABLE_CACHING
    auto vcaches_dst = expr::cache<>();
    auto vcaches_use = expr::cache<>();
#else
    auto vcaches_dst = ownrd_pull_extract_cacheable_refs<is_ownwr>( vexpr1 );
    auto vcaches_use = expr::extract_uses<expr::vk_src>( vexpr1, vcaches_dst );
#endif

    auto vop_caches = expr::extract_cacheable_refs<expr::vk_pid>(
	rexpr1, cache_cat( vcaches_dst, vcaches_use ) );
    auto rexpr1b = expr::rewrite_caches<expr::vk_pid>( rexpr1, vop_caches );

    // This cannot be disabled
    auto vcaches_let
	= expr::extract_local_vars(
	    vexpr1, expr::cache_cat( vop_caches, vcaches_dst, vcaches_use ) );
    auto vcaches = expr::cache_cat( vcaches_dst, vcaches_let );

    // Loop part
    auto vexpr2 = expr::rewrite_caches<expr::vk_dst>( vexpr1, vcaches_dst );
    auto vexpr3 = expr::rewrite_caches<expr::vk_src>( vexpr2, vcaches_use );
    auto vexpr4 = expr::rewrite_caches<expr::vk_zero>( vexpr3, vcaches_let );
    auto vexpr5 = expr::rewrite_vectors_main( vexpr4 );
    auto vexpr = expr::rewrite_mask_main( vexpr5 );

    // Post-loop part
    auto rexpr2 = expr::rewrite_caches<expr::vk_dst>( rexpr1b, vcaches_dst );
    auto rexpr3 = expr::rewrite_caches<expr::vk_src>( rexpr2, vcaches_use );
    auto rexpr4 = expr::rewrite_caches<expr::vk_zero>( rexpr3, vcaches_let );
    auto rexpr5 = expr::rewrite_vectors_main( rexpr4 );
    auto rexpr = expr::rewrite_mask_main( rexpr5 );

    // Update condition
    auto uexpr1 = expr::rewrite_caches<expr::vk_dst>( uexpr0, vcaches_dst );
    auto uexpr2 = expr::rewrite_vectors_main( uexpr1 );
    auto uexpr = expr::rewrite_mask_main( uexpr2 );

    // Loop termination condition (active check)
    auto aexpr0 = op.active( expr::value<simd::ty<VID,1>,expr::vk_dst>() );
    // auto aexpr1 = expr::make_unop_reduce( aexpr0, expr::redop_logicalor() );
    auto aexpr2 = expr::rewrite_caches<expr::vk_dst>( aexpr0, vcaches_dst );
    auto aexpr3 = expr::rewrite_vectors_main( aexpr2 );
    auto aexpr = expr::rewrite_mask_main( aexpr3 );

    // Override pointer for aid_eweight with the relevant permuation of the
    // weights for the G graph.
    // auto ew_pset = expr::create_map2<expr::aid_eweight>(
    // G.getWeights() ? G.getWeights()->get() : nullptr );
					 
    // auto env = expr::eval::create_execution_environment_with(
    // op.get_ptrset( ew_pset ),
    // vcaches, vop_caches, accum, vexpr, aexpr, rexpr, uexpr, pvop, pvopf );

    auto all_caches = expr::cache_cat(
	expr::cache_cat( vcaches, vop_caches ), accum );
    auto env = expr::eval::create_execution_environment_op(
	op, all_caches,
	G.getWeights() ? G.getWeights()->get() : nullptr );

    using Cfg = std::decay_t<decltype(op.get_config())>;

    // Note: If not privatized (owner-reads), frontier must be initialized
    //       to all zero. This is done in the frontier constructor.

    map_partition<Cfg::is_parallel()>( part, [&]( int p ) {
	    auto pvec = simd::create_scalar( (VID)p );
	    auto m_pid = expr::create_value_map_new2<1,expr::vk_pid>( pvec );

	    auto c = cache_create_no_init(
		expr::cache_cat( vop_caches, vcaches, vcaches_use ), m_pid );
	    cache_init( env, c, vop_caches, m_pid );

	    // A single partition is processed here. Use a sequential loop.
	    auto ps = G.part_vertex_begin( part, p );
	    auto pe = G.part_vertex_end( part, p );
	    EID e = part.edge_start_of( p );
	    for( auto ii=ps; ii != pe; ++ii ) {
		VertexInfo vi = *ii;
		VID i = vi.v;
		VID deg = vi.degree;
		const VID * ngh = vi.neighbours;

		constexpr bool ID = expr::is_idempotent<decltype(vexpr)>::value;
		process_in_v_e_simd1_VL1<ID,Assoc,is_ownwr>(
		    ngh, deg, i,
#if ACCUM_PAD
		    p*16,
#else
		    p,
#endif
		    e,
		    eid_retriever,
		    env,
		    rewrite_internal( vexpr ),
		    rewrite_internal( aexpr ),
		    rewrite_internal( rexpr ),
		    rewrite_internal( uexpr ),
		    c, vcaches, vcaches_use );

		e += deg;
	    }

	    cache_commit( env, vop_caches, c, m_pid );
	} );

    if constexpr ( !is_ownwr ) {
	map_partition<Cfg::is_parallel()>( part, [&]( unsigned int p ) {
	    auto pvec = simd::create_scalar( (VID)p );
	    auto m_pid = expr::create_value_map_new2<1,expr::vk_pid>( pvec );

	    auto c = cache_create_no_init(
		expr::cache_cat( vop_caches, vcaches, vcaches_use ), m_pid );
	    cache_init( env, c, vop_caches, m_pid );

	    GraptorCSRVertexOp<1>( env, p, part, rexpr, c );

	    cache_commit( env, vop_caches, c, m_pid );
	} );
    }

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

} // namespace GraphCSx_csc

template<typename EMapConfig, typename EIDRetriever, typename Operator>
GG_INLINE
static inline void emap_pull(
    const GraphCSx & G, const EIDRetriever & eid_retriever,
    Operator & op, const partitioner & part ) {
    constexpr unsigned short VL = EMapConfig::VL;
    static_assert( VL == 1, "GraphCSx designed for scalar execution only" );

    // const EID * idx = G.getIndex();
    // const VID * edge = G.getEdges();

    GraphCSx_csc::csc_loop<EMapConfig,true>( G, eid_retriever, /*idx, edge,*/ op, part );
}

template<typename EMapConfig, typename Operator>
GG_INLINE
static inline void emap_pull(
    const GraphGGVEBO & G, Operator & op, const partitioner & part ) {
    constexpr unsigned short VL = EMapConfig::VL;
    static_assert( VL == 1, "GraphCSx designed for scalar execution only" );

    GraphCSx_csc::csc_loop<EMapConfig,true>( G.getCSC(), G.get_eid_retriever(), op, part );
}

// Assumes symmetric graph
template<typename EMapConfig, typename Operator>
GG_INLINE
static inline void emap_pull(
    const GraphVEBOPartCCSR & G, Operator & op, const partitioner & part ) {
    constexpr unsigned short VL = EMapConfig::VL;
    static_assert( VL == 1, "GraphCSx designed for scalar execution only" );
    assert( G.isSymmetric() && "This interpretes graph in inverse direction" );

    GraphCSx_csc::csc_loop<EMapConfig,false>( G, G.get_eid_retriever(), op, part );
}

// Assumes symmetric graph
template<typename EMapConfig, typename Operator>
GG_INLINE
static inline void emap_pull(
    const GraphCSRAdaptor & G, Operator & op, const partitioner & part ) {
    constexpr unsigned short VL = EMapConfig::VL;
    static_assert( VL == 1, "GraphCSx designed for scalar execution only" );
    assert( G.isSymmetric() && "This interpretes graph in inverse direction" );

    GraphCSx_csc::csc_loop<EMapConfig,false>(
	G.getCSR(), G.get_eid_retriever(), op, part );
}

#endif // GRAPTOR_DSL_EMAP_GRAPHCSX_CSC_H
