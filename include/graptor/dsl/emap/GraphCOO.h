#ifndef GRAPTOR_DSL_EMAP_GRAPHCOO_H
#define GRAPTOR_DSL_EMAP_GRAPHCOO_H

#include "graptor/dsl/eval/environment.h"

template<unsigned short VL, typename EIDRetriever, typename Operator>
static __attribute__((flatten)) void coo_vector_loop(
    const GraphCOO & el,
    EIDRetriever && eid_retriever,
    Operator op,
    std::enable_if_t<(VL > 1)> * = nullptr ) {
    EID m = el.numEdges();

    // COO does not use the active check. It should be optional to apply
    // the active check.
    auto vexpr0 = op.relax( expr::value<simd::ty<VID,VL>,expr::vk_src>(),
			    expr::value<simd::ty<VID,VL>,expr::vk_dst>(),
			    expr::value<simd::ty<EID,VL>,expr::vk_edge>() );

    auto vexpr1 = expr::rewrite_vectors_main( vexpr0 );

    // Rewrite local variables
    auto vcaches = expr::extract_local_vars( vexpr1 );
    auto vexpr2 = expr::rewrite_caches<expr::vk_zero>( vexpr1, vcaches );

    // Match vector lengths and move masks
    auto vexpr = expr::rewrite_mask_main( vexpr2 );

    const VID *src_arr = el.getSrc();
    const VID *dst_arr = el.getDst();

    constexpr EID BLOCK_SIZE = 64 / sizeof(VID);
    constexpr EID PREFETCH_DISTANCE = 128; // (2<<20) / sizeof(VID);
    
    // Override pointer for vk_eweight with the relevant permuation of the
    // weights for the el graph.
    auto ew_pset = expr::create_map2<expr::vk_eweight>( el.getWeights() );
					 
    auto venv = expr::eval::create_execution_environment_with(
	op.get_ptrset( ew_pset ), vcaches, vexpr );

    EID i = 0;
    for( ; i+VL-1 < m; i += VL ) {
	auto src = simd_vector<VID,VL>::load_from( &src_arr[i] );	
	auto dst = simd_vector<VID,VL>::load_from( &dst_arr[i] );	

	if( i % BLOCK_SIZE == 0 ) {
	    _mm_prefetch( &src_arr[i+PREFETCH_DISTANCE], _MM_HINT_NTA );
	    _mm_prefetch( &dst_arr[i+PREFETCH_DISTANCE], _MM_HINT_NTA );
	}

	EID seid = i; // eid_retriever.get_edge_eid( i );
	auto eid = simd::template create_scalar<simd::ty<EID,1>>( seid );
	
	auto mp = expr::create_value_map_new<VL>(
	    expr::create_entry<expr::vk_src>( src ),
	    expr::create_entry<expr::vk_dst>( dst ),
	    expr::create_entry<expr::vk_edge>( eid ) );
	auto c = expr::cache_create( venv, vcaches, mp );
	venv.evaluate( c, mp, vexpr );
    }

    // epilogue (remainder iterations < VL)
    if constexpr ( VL == 1 )
	return;
    if( i == m )
	return;

    // Epilogue - scalar
    auto sexpr0 = op.relax( expr::value<simd::ty<VID,1>,expr::vk_src>(),
			    expr::value<simd::ty<VID,1>,expr::vk_dst>(),
			    expr::value<simd::ty<EID,1>,expr::vk_edge>() );
    auto sexpr1 = expr::rewrite_vectors_main( sexpr0 );
    auto scaches = expr::extract_local_vars( sexpr1 );
    auto sexpr2 = expr::rewrite_caches<expr::vk_zero>( sexpr1, scaches );
    auto sexpr = expr::rewrite_mask_main( sexpr2 );

    auto senv = expr::eval::create_execution_environment_with(
	op.get_ptrset( ew_pset ), scaches, sexpr );

    for( ; i < m; ++i ) {
	auto src = simd_vector<VID,1>::load_from( &src_arr[i] );	
	auto dst = simd_vector<VID,1>::load_from( &dst_arr[i] );	

	EID seid = eid_retriever.get_edge_eid( i );
	auto eid = simd::template create_scalar<simd::ty<EID,1>>( seid );
	
	auto mp = expr::create_value_map_new<1>(
	    expr::create_entry<expr::vk_src>( src ),
	    expr::create_entry<expr::vk_dst>( dst ),
	    expr::create_entry<expr::vk_edge>( eid ) );
	auto c = expr::cache_create( senv, vcaches, mp );
	senv.evaluate( c, mp, sexpr );
    }
}

template<unsigned short VL, typename Environment,
	 typename EIDRetriever, typename Expr, typename VCache>
static __attribute__((flatten)) void coo_vector_loop(
    const GraphCOO & el,
    const Environment & env, 
    EIDRetriever && eid_retriever,
    const Expr & expr,
    const VCache & vcaches,
    std::enable_if_t<VL == 1> * = nullptr ) {
    EID m = el.numEdges();

    const VID *src_arr = el.getSrc();
    const VID *dst_arr = el.getDst();

    for( EID i=0; i < m; ++i ) {
	auto src = simd_vector<VID,1>::load_from( &src_arr[i] );	
	auto dst = simd_vector<VID,1>::load_from( &dst_arr[i] );	

	EID seid = i; // eid_retriever.get_edge_eid( i );
	auto eid = simd::template create_scalar<simd::ty<EID,1>>( seid );
	
	auto mp = expr::create_value_map_new<1>(
	    expr::create_entry<expr::vk_src>( src ),
	    expr::create_entry<expr::vk_dst>( dst ),
	    expr::create_entry<expr::vk_edge>( eid ) );
	auto c = expr::cache_create( env, vcaches, mp );
	env.evaluate( c, mp, expr );
    }
}


template<typename EMapConfig, typename GraphType, typename Operator>
static inline void emap_ireg_graphgrind(
    const GraphType & G, Operator & op, const partitioner & part ) {
    constexpr unsigned short VL = EMapConfig::VL;

    EID m = G.numEdges();

    auto vid = expr::value<simd::ty<VID,VL>,expr::vk_vid>();
    auto pid = expr::value<simd::ty<VID,VL>,expr::vk_pid>();

    auto vexpr0 = op.relax( expr::value<simd::ty<VID,VL>,expr::vk_src>(),
			    expr::value<simd::ty<VID,VL>,expr::vk_dst>(),
			    expr::value<simd::ty<EID,VL>,expr::vk_edge>() );

    auto vexpr1 = expr::rewrite_vectors_main( vexpr0 );

    // Rewrite local variables
    auto vcaches = expr::extract_local_vars( vexpr1 );
    auto vexpr2 = expr::rewrite_caches<expr::vk_zero>( vexpr1, vcaches );

    // Match vector lengths and move masks
    auto vexpr3 = expr::rewrite_mask_main( vexpr2 );
    auto vexpr = rewrite_internal( vexpr3 );

    auto expr0 = op.vertexop( vid );
    auto accum = expr::extract_accumulators( expr0 );
    expr::accum_create( part, accum );
    auto expr1 = expr::rewrite_privatize_accumulators( expr0, part, accum, pid );
    auto pvop0 = expr::accumulate_privatized_accumulators( pid, accum );
    auto pvopf0 = expr::final_accumulate_privatized_accumulators( pid, accum );
    // 1. op0->st->ld->op1 becomes op0->st ; op0->op1
    //    this may become too complex for ASTs; C++ compiler cannot do this
    //    because it only knows the types, not the pointers. We can only
    //    work on the AIDs, however, this may be impractical as AIDs would
    //    need to match across vertex map operators.
    // 2. step_vscan: contains an accumulator: privatize the accumulator
    //    in each partition (cacheop) and follow on by a scan across accu's
    // 3. step_vscan: in case output is used, do not fuse across a vscan?
    //    or can we find a way to identify when output is used?
    // 4. step_vscan: do we need to differentiate from vmap in programming
    //    model, or can we figure this out ourselves through analysis?
    auto expr2 = expr::rewrite_mask_main( expr1 ); // expr::remove_reload( expr0 );
    auto expr = rewrite_internal( expr2 );

    map_partitionL( part, [&]( int p ) {
	// Override pointer for vk_eweight with the relevant permuation of the
	// weights for the el graph.
	auto ew_pset = expr::create_map2<expr::vk_eweight>(
	    const_cast<float *>( G.get_edge_list_partition( p ).getWeights() )
	    );
					 
	auto env = expr::eval::create_execution_environment_with(
	    op.get_ptrset( ew_pset ), vcaches, vexpr3, expr2, pvop0, pvopf0  );

	// edgemap part
	coo_vector_loop<VL>( G.get_edge_list_partition( p ),
			     env,
			     G.get_edge_list_eid_retriever( p ),
			     vexpr, vcaches );
	
	// vertexop (map/scan)
	// simd_vector<VID,VL> pp( (VID)p );
	auto pp = simd::template create_constant<VID,VL>( p );
	VID s = part.start_of( p );
	VID e = part.end_of( p );

	for( VID v=s; v < e; ++v ) {
	    // simd_vector<VID,VL> vv( v );
	    static_assert( VL == 1, "Looks like VL==1, otherwise ???" );
	    auto vv = simd::template create_constant<VID,VL>( v );
	    auto m = expr::create_value_map_new<VL>(
		expr::create_entry<expr::vk_vid>( vv ),
		expr::create_entry<expr::vk_pid>( pp ) );
	    // auto m = expr::create_value_map2<
	    // VL,VL,expr::vk_vid,expr::vk_pid>( vv, pp );
	    expr::cache<> c;
	    env.evaluate( c, m, expr );
	}
    } );

    // Scan acros partitions
    {
	auto env = expr::eval::create_execution_environment_with(
	    op.get_ptrset(), vcaches, vexpr3, expr2, pvop0, pvopf0  );

	int np = part.get_num_partitions();
	for( int p=1; p < np; ++p ) {
	    auto pp = simd::template create_set1inc<VID,VL,true>( p*VL );
	    auto m = expr::create_value_map2<VL,VL,expr::vk_pid>( pp );
	    expr::cache<> c;
	    env.evaluate( c, m, pvop0 );
	}
	{
	    auto pp = simd::template create_set1inc0<VID,VL>();
	    auto m = expr::create_value_map2<VL,VL,expr::vk_pid>( pp );
	    expr::cache<> c;
	    env.evaluate( c, m, pvopf0 );
	}
    }

    accum_destroy( accum );
}

template<typename EMapConfig, typename Operator>
static inline void emap_ireg(
    const GraphGGVEBO & G, Operator & op, const partitioner & part ) {
    emap_ireg_graphgrind<EMapConfig>( G, op, part );
}

template<typename EMapConfig, typename COOType, typename Operator>
static inline void emap_ireg(
    const GraphGG_tmpl<COOType> & G, Operator & op, const partitioner & part ) {
    emap_ireg_graphgrind<EMapConfig>( G, op, part );
}

#endif // GRAPTOR_DSL_EMAP_GRAPHCOO_H
