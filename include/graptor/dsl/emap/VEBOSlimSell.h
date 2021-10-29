// -*- c++ -*-
#ifndef GRAPTOR_DSL_EMAP_VEBOSLIMSELL_H
#define GRAPTOR_DSL_EMAP_VEBOSLIMSELL_H

#include "graptor/dsl/eval/environment.h"
#include "graptor/dsl/comp/insert_mask.h"

namespace vebosell {

template<bool Idempotent>
struct cache_ops;

template<>
struct cache_ops<false> {
    template<typename Environment, typename Cache, typename VCache,
	     typename ValueMap, typename ValueMap1>
    // TODO: for low-degree vertices, it is probably better to not cache the
    //       values
    static void
    construct( const Environment & env,
	       const VCache & vcaches, Cache & c,
	       const ValueMap & m, const ValueMap1 & m1 ) {
	cache_init( c, vcaches, m ); // initialise with stored value
    }
    template<typename Environment, typename Cache, typename VCache,
	     typename ValueMap, typename ValueMap1>
    static void
    destruct( const Environment & env,
	      const VCache & vcaches, Cache & c,
	      const ValueMap & m, const ValueMap1 & m1 ) {
	cache_commit( vcaches, c, m ); // reduce and write back to memory
    }
};

template<>
struct cache_ops<true> {
    template<typename Environment, typename Cache, typename VCache,
	     typename ValueMap, typename ValueMap1>
    static void
     construct( const Environment & env,
		const VCache & vcaches, Cache & c,
		const ValueMap & m, const ValueMap1 & m1 ) {
	cache_clear( c, vcaches, m ); // set to zero of enclosing redop
    }
    template<typename Environment, typename Cache, typename VCache,
	     typename ValueMap, typename ValueMap1>
    static void
     destruct( const Environment & env,
	       const VCache & vcaches, Cache & c,
	       const ValueMap & m, const ValueMap1 & m1 ) {
	cache_commit_with_reduce( env, vcaches, c, m ); // reduce with value stored in memory
	// For benchmarks where init value equals zero of semiring, simply
	// write back, don't merge. Does this equal not idempotent?
	// cache_commit( vcaches, c, m ); // reduce and write back to memory
    }
};


template<unsigned short VL, bool Idempotent,
	 typename Environment,
	 typename Expr, typename AExpr, typename RExpr, typename MExpr,
	 typename UExpr,
	 typename CacheDesc, typename CacheInst>
__attribute__((always_inline))
inline void process_in_v_e_simd(
    const VID *in, EID deg, EID mdeg, VID scalar_dst, VID scalar_pid,
    unsigned short storedVL,
    const Environment & env,
    const Expr & e, const AExpr & ea, const RExpr & er, const MExpr & m_e,
    const UExpr & eu,
    const CacheDesc & cache_desc, CacheInst &c ) {
    // Note: deg and mdeg are measured in SIMD groups of size storedVL

#if CSC_ELIDE_WB_ZERO
    constexpr bool InitZero = true;
#else
    constexpr bool InitZero = false;
#endif

    EID j;

    // fail_expose<std::is_class>( er );

    simd_vector<VID,1> dst( scalar_dst );

    simd_vector<VID,VL> pid;
    pid.set1inc( VL*scalar_pid ); // we have sufficient privatized storage

    // Note: create caches at vector length VL of source
    auto m = expr::create_value_map_new2<VL,expr::vk_dst,expr::vk_pid>(
	dst, pid );
    cache_ops<InitZero>::construct( env, cache_desc, c, m, m );

#if CSC_ELIDE_ACTV
    // Problem: this could preclude operations in vertexop that must
    // be executed regardless of convergence.
    if( !env.evaluate_bool( c, m, ea ) )
	return;
#endif // CSC_ELIDE_ACTV
    
#if CSC_ELIDE_WB_CHANGED
    auto c_initial = expr::map_shift<c.num_entries>( c );
#endif

    typename simd_vector<typename Expr::type, VL>::simd_mask_type output
	= simd_vector<typename Expr::type, VL>::false_mask();

    // No mask applied
    for( j=0; j < deg; j += storedVL ) {
	// Check all lanes are active.
	if( !env.evaluate_bool( c, m, ea ) )
	    goto DONE;

	// First of all, issue critical load to memory
	simd_vector<VID, VL> src;
#if ENABLE_NT_TOPO
	src.ntload( &in[j] ); // linear read, non-temporal
#else
	src.load( &in[j] ); // linear read
#endif

	if constexpr ( PFT_DISTANCE > 0 ) {
	    // Topology data, far ahead
	    _mm_prefetch( &in[j+PFT_DISTANCE], _MM_HINT_NTA );

	    if constexpr ( PFE1_DISTANCE > 0 ) {
	        // Topology data and vertex properties, not as far ahead
		src.load( &in[j+PFE1_DISTANCE] );
		auto m = expr::create_value_map_new2<
		    VL,expr::vk_src,expr::vk_dst>( src, dst );
		expr::cache_prefetch<PFE2_DISTANCE>( c, cache_desc, m );
	    }
	}

#if ENABLE_FLUSH_TOPO
	// Assuming vector width sufficient to step a full cache block per
	// iteration of this loop.
	// Factor of 2 is out of conservatism - in case of lack of alignment
	static_assert( sizeof(in[j])*VL*2 >= 64, "flush distance requirement" );
	// _mm_clflushopt( const_cast<VID *>( &in[j-2*storedVL] ) );
	_mm_clflush( const_cast<VID *>( &in[j-2*storedVL] ) );
#endif

	auto m = expr::create_value_map_new2<VL,expr::vk_src,expr::vk_dst>(
	    src, dst );
	auto rval_output = env.evaluate( c, m, e ); // rvalue space
	output.lor_assign( rval_output.mask() ); // simd_vector/simd_mask space
    }
    // Remaining iterations with mask to check valid input vectors
    for( ; j < mdeg; j += storedVL ) {
	// Check all lanes are active.
	if( !env.evaluate_bool( c, m, ea ) )
	    goto DONE;

	simd_vector<VID, VL> src;
	if constexpr ( PFT_DISTANCE > 0 )
	    _mm_prefetch( &in[j+PFT_DISTANCE], _MM_HINT_NTA );
#if ENABLE_NT_TOPO
	src.ntload( &in[j] ); // linear read, non-temporal
#else
	src.load( &in[j] ); // linear read
#endif
	auto m = expr::create_value_map_new2<VL,expr::vk_src,expr::vk_dst>(
	    src, dst );
	auto rval_output = env.evaluate( c, m, m_e ); // rvalue space
	output.lor_assign( rval_output.mask() ); // simd_vector/simd_mask space
    }

    // Self-edge - only in case of WB_ZERO and certain benchmarks (idempotent!)
    if constexpr ( InitZero && Idempotent ) {
	// To be really effective, need to merge this with the write-back:
	// - load own value
	// - redop
	// - compare with current update
	// - store if changed
	// At the moment, we load 'old' value instead of 'new' - ok if ASYNC
	if( env.evaluate_bool( c, m, ea ) ) {
	    simd_vector<VID, VL> src( scalar_dst );
	    auto m = expr::create_value_map_new2<VL,expr::vk_src,expr::vk_dst>(
		src, dst );
	    auto rval_output = env.evaluate( c, m, e );
	    output.lor_assign( rval_output.mask() );
	}
    }

DONE:
    {
	using logicalVID = typename add_logical<VID>::type;
	auto input = output.template asvector<logicalVID>();

	auto m = expr::create_value_map_new2<
	    VL,expr::vk_smk,expr::vk_pid,expr::vk_dst>( input, pid, dst );
	env.evaluate( c, m, er );
    }
#if CSC_ELIDE_WB
    constexpr size_t n_elms
	= std::tuple_size<typename std::remove_reference<decltype(cache_desc)>::type::type>::value;
    static_assert( n_elms <= 1, "oops" );

#if CSC_ELIDE_WB_ZERO
    cache_ops<InitZero>::destruct( env, cache_desc, c, m, m );
#elif CSC_ELIDE_WB_ALWAYS
    bool outputs = simd_vector<typename Expr::type, VL>::traits::cmpne(
	output.data(),
	simd_vector<typename Expr::type, VL>::false_mask().data(),
	target::mt_bool() );
    if( outputs )
	cache_commit( cache_desc, c, m );
#elif CSC_ELIDE_WB_CHANGED
    cache_commit_if_changed( cache_desc, c_initial, c, m );
#else
    if constexpr ( expr::is_constant_true<UExpr>::value ) {
	bool outputs = simd_vector<typename Expr::type, VL>::traits::cmpne(
	    output.data(),
	    simd_vector<typename Expr::type, VL>::false_mask().data(),
	    target::mt_bool() );
	if( outputs )
	    cache_commit( cache_desc, c, m );
    } else {
	auto mu = expr::create_value_map_new<VL>(
	    expr::create_entry<expr::vk_dst>( dst ),
	    expr::create_entry<expr::vk_pid>( pid ) );
	bool upd = env.evaluate_bool( c, mu, eu );
	if( upd )
	    cache_ops<InitZero>::destruct( env, cache_desc, c, m, m );
    }
#endif

#else
    // No WB elision
    cache_ops<InitZero>::destruct( env, cache_desc, c, m, m );
#endif
}

template<unsigned short VL, bool Idempotent,
	 typename Environment,
	 typename MExpr, typename AExpr, typename RExpr, typename UExpr,
	 typename CacheDesc, typename CacheInst>
__attribute__((always_inline))
inline void process_in_v_e_simd_mask(
    const VID *in, EID mdeg, VID scalar_dst, VID scalar_pid,
    unsigned short storedVL,
    VID nexist,
    const Environment & env,
    const MExpr & m_e, const AExpr & ea, const RExpr & er,
    const UExpr & eu,
    const CacheDesc & cache_desc, CacheInst &c ) {
    // Note: mdeg is measured in SIMD groups of size storedVL
    
#if CSC_ELIDE_WB_ZERO
    constexpr bool InitZero = true;
#else
    constexpr bool InitZero = false;
#endif

    EID j;

    // fail_expose<std::is_class>( er );

    simd_vector<VID,1> dst( scalar_dst );

    simd_vector<VID,VL> pid;
    pid.set1inc( VL*scalar_pid ); // we have sufficient privatized storage

    // Note: create caches at vector length VL of source
    auto m = expr::create_value_map_new2<VL,expr::vk_dst,expr::vk_pid>(
	dst, pid );
    cache_ops<InitZero>::construct( env, cache_desc, c, m, m );

#if CSC_ELIDE_ACTV
    if( !env.evaluate_bool( c, m, ea ) )
	return;
#endif // CSC_ELIDE_ACTV
    
#if CSC_ELIDE_WB_CHANGED
    auto c_initial = expr::map_shift<c.num_entries>( c );
#endif

    typename simd_vector<typename MExpr::type, VL>::simd_mask_type output
	= simd_vector<typename MExpr::type, VL>::false_mask();

    // Iterations with mask to check valid input vectors.
    // In this loop, non-existent vertices have no edges, hence no side effects
    // can occur for them.
    for( j=0; j < mdeg; j += storedVL ) {
	// Check all lanes are active.
	if( !env.evaluate_bool( c, m, ea ) )
	    break;

	simd_vector<VID, VL> src;
#if ENABLE_NT_TOPO
	src.ntload( &in[j] ); // linear read, non-temporal
#else
	src.load( &in[j] ); // linear read
#endif
	auto m = expr::create_value_map_new2<VL,expr::vk_src,expr::vk_dst>(
	    src, dst );
	auto rval_output = env.evaluate( c, m, m_e ); // rvalue space
	output.lor_assign( rval_output.mask() ); // simd_vector/simd_mask space
    }

    // Self-edge - only in case of WB_ZERO and certain benchmarks (idempotent!)
    if constexpr ( InitZero && Idempotent ) {
	if( env.evaluate_bool( c, m, ea ) ) {
	    simd_vector<VID,VL> src( scalar_dst );
	    auto m = expr::create_value_map_new<VL>(
		expr::create_entry<expr::vk_src>( src ),
		expr::create_entry<expr::vk_dst>( dst ),
		expr::create_entry<expr::vk_pid>( pid ) );
	    auto rval_output = env.evaluate( c, m, m_e );
	    output.lor_assign( rval_output.mask() );
	}
    }

DONE:
    {
	using logicalVID = typename add_logical<VID>::type;
	auto input = output.template asvector<logicalVID>();
	simd_mask<sizeof(VID), VL> msk;
	// Warning: VL==64 requires a 64-bit integer here
	msk.from_int( (1<<nexist)-1 );

	auto m = expr::create_value_map_new2<
	    VL,expr::vk_smk,expr::vk_pid,expr::vk_dst,expr::vk_mask>(
		input, pid, dst, msk );
	env.evaluate( c, m, er );
    }
    
#if CSC_ELIDE_WB
    constexpr size_t n_elms
	= std::tuple_size<typename std::remove_reference<decltype(cache_desc)>::type::type>::value;
    static_assert( n_elms <= 1, "oops" );

#if CSC_ELIDE_WB_ZERO
    cache_ops<InitZero>::destruct( env, cache_desc, c, m, m );
#elif CSC_ELIDE_WB_ALWAYS
    bool outputs = simd_vector<typename MExpr::type, VL>::traits::cmpne(
	output.data(),
	simd_vector<typename MExpr::type, VL>::false_mask().data(),
	target::mt_bool() );
    if( outputs )
	cache_commit( cache_desc, c, m );
#elif CSC_ELIDE_WB_CHANGED
    cache_commit_if_changed( cache_desc, c_initial, c, m );
#else
    if constexpr ( expr::is_constant_true<UExpr>::value ) {
	bool outputs = simd_vector<typename MExpr::type, VL>::traits::cmpne(
	    output.data(),
	    simd_vector<typename MExpr::type, VL>::false_mask().data(),
	    target::mt_bool() );
	if( outputs )
	    cache_commit( cache_desc, c, m );
    } else {
	auto mu = expr::create_value_map_new<VL>(
	    expr::create_entry<expr::vk_dst>( dst ),
	    expr::create_entry<expr::vk_pid>( pid ) );
	bool upd = env.evaluate_bool( c, mu, eu );
	if( upd )
	    cache_ops<InitZero>::destruct( env, cache_desc, c, m, m );
    }
#endif

#else
    // No WB elision
    cache_ops<InitZero>::destruct( env, cache_desc, c, m, m );
#endif
}

} // namespace vebosell


template<unsigned short VL, bool doVEBO, typename Operator>
__attribute__((always_inline))
static inline void csc_vector_with_cache_loop(
    const GraphVEBOSlimSell_template<doVEBO> & GA,
    Operator & op, const partitioner & part ) {
    // typename std::enable_if<Operator::is_scan>::type * = nullptr ) {
    // Fused CSC + vertexop version where vertexop may involve a scan
    unsigned short storedVL = GA.getMaxVL();

    assert( storedVL >= VL && "Storage must be designed to meet VL" );
    assert( part.get_num_elements() % VL == 0
	    && "number of elements must be multiple of VL" );

    constexpr bool parpar = false;

    // Variables (actually parameters because they are constant within the AST)
    auto v_src = expr::value<simd::ty<VID,VL>,expr::vk_src>();
    auto v_dst1 = expr::value<simd::ty<VID,1>,expr::vk_dst>();
    auto v_dst = expr::make_unop_incseq<VL>( v_dst1 );
    auto v_one = expr::value<simd::ty<VID,VL>,expr::vk_truemask>();
    auto v_pid0 = expr::value<simd::ty<VID,VL>,expr::vk_pid>();
    auto v_mask = expr::value<simd::ty<VID,VL>,expr::vk_mask>();

    // First version without mask
    auto vexpr0 = op.relax( v_src, v_dst );

    // Second version with mask for inactive lanes.
    auto m_vexpr0
	= op.relax( add_mask( v_src, make_cmpne( v_src, v_one ) ), v_dst );

    // Vertexop - no mask
    auto vop0 = op.vertexop( v_dst );
    auto accum = expr::extract_accumulators( vop0 );
    auto vop1 = expr::rewrite_privatize_accumulators( vop0, part, accum, v_pid0 );
    auto pvop0 = expr::accumulate_privatized_accumulators( v_pid0, accum );
    auto pvopf0 = expr::final_accumulate_privatized_accumulators( v_pid0, accum );

    auto vop_caches = expr::extract_cacheable_refs<expr::vk_pid>( vop1 );
    auto vop2 = expr::rewrite_caches<expr::vk_pid>( vop1, vop_caches );

    // LICM: only inner-most redop needs to be performed inside loop.
    //       the other relates to the new frontier and can be carried over
    //       through registers as opposed to variables.
    auto licm = expr::licm_split_main( vexpr0 );
    auto vexpr1 = licm.le();
    auto rexpr0 = licm.pe();
    auto rexpr1 = append_pe( rexpr0, vop2 );

    // It is assumed cache is the same / masked version may have additional
    // update bitmask
    auto vcaches
	= expr::extract_cacheable_refs<expr::vk_dst,expr::mam_nontemporal>( m_vexpr0, vop_caches );

    // Loop part
    auto vexpr2 = expr::rewrite_caches<expr::vk_dst>( vexpr1, vcaches );
    auto vexpr = expr::rewrite_mask_main( vexpr2 );

    // Post-loop part
    auto rexpr2 = expr::rewrite_caches<expr::vk_dst>( rexpr1, vcaches );
    // Any remaining references should be temporal
#if ENABLE_NT_VMAP
    auto rexpr3 = expr::transform_nt<true>( rexpr2 );
#else
    auto rexpr3 = rexpr2;
#endif
    auto rexpr = expr::rewrite_mask_main( rexpr3 );

    // Update
#if CSC_ELIDE_WB_DIFF
    auto uexpr0 = op.different( v_dst );
#else
    auto uexpr0 = op.update( v_dst );
#endif
    auto uexpr1 = expr::make_unop_reduce( uexpr0, expr::redop_logicalor() );
    auto uexpr2 = expr::rewrite_caches<expr::vk_dst>( uexpr1, vcaches );
    auto uexpr = expr::rewrite_mask_main( uexpr2 );

    // Loop termination condition (active check)
    auto aexpr0 = op.active( v_dst );
    auto aexpr1 = expr::make_unop_reduce( aexpr0, expr::redop_logicalor() );
    auto aexpr2 = expr::rewrite_caches<expr::vk_dst>( aexpr1, vcaches );
    auto aexpr = expr::rewrite_mask_main( aexpr2 );

    // LICM: only inner-most redop needs to be performed inside loop.
    //       the other relates to the new frontier and can be carried over
    //       through registers as opposed to variables.
    auto m_licm = expr::licm_split_main( m_vexpr0 );
    auto m_vexpr1 = m_licm.le();
    auto m_rexpr0 = m_licm.pe();
    auto m_rexpr1 = append_pe( m_rexpr0, vop2 );

    auto m_rexpr2 = expr::rewrite_caches<expr::vk_dst>( m_rexpr1, vcaches );
    auto m_rexpr3 = expr::rewrite_mask_main( m_rexpr2 );
    auto m_rexpr = expr::insert_mask<expr::vk_dst>( m_rexpr3, v_mask );

    // TODO: vertexop

    // Loop part
    auto m_vexpr2 = expr::rewrite_caches<expr::vk_dst>( m_vexpr1, vcaches );
    auto m_vexpr = expr::rewrite_mask_main( m_vexpr2 );

    // TODO:
    // * strength reduction on division by storedVL
    // * pass expressions by reference (appears some copying is done)

    auto env = expr::eval::create_execution_environment(
	vcaches, vop_caches, accum,
	vexpr, m_vexpr, aexpr, rexpr, m_rexpr, uexpr, pvop0, pvopf0 );

    map_partitionL( part, [&]( int p ) {
	    auto GP = GA.getCSC( p );
	    const EID * idx = GP.getIndex();
	    const EID * midx = GP.getMaskIndex();
	    const VID * edge = GP.getEdges();

	    VID from = part.start_of( p );
	    VID to = part.end_of( p );
	    // It is very important to prune the iteration range to the number
	    // of vectorized vertices stored, especially for hybrid cases
	    to = std::min( to, from+GP.numSIMDVertices() );

	    assert( from % storedVL == 0 );
	    assert( part.start_of( p+1 ) % storedVL == 0 );

	    simd_vector<VID, VL> pvec;
	    pvec.set1inc( VL*VID(p) );
	    auto m_pid = expr::create_value_map2<VL,VL,expr::vk_pid>( pvec );
	    simd_vector<VID, VL> pzero;

	    if( parpar ) {
		unsigned short lsVL = rt_ilog2( storedVL );
		constexpr unsigned short lVL = ilog2( VL );
		VID e = (to - from + storedVL - 1) >> lsVL;
		parallel_for( VID s=0; s < e; ++s ) {
		    auto c = cache_create_no_init( cache_cat( vcaches, vop_caches ), m_pid );
		    cache_init( c, vop_caches, m_pid );
		    for( VID ss=0; ss < storedVL; ss += VL ) {
			VID i = from + (s << lsVL) + ss;
			// Reconstruct the index based on the availability only of one
			// index value out of storedVL. Each of those index values,
			// however, is one higher than the previous.
			vebosell::process_in_v_e_simd<VL,Operator::is_idempotent>(
			    &edge[idx[s]+ss],
			    midx[s]-idx[s],
			    idx[s+1]-idx[s],
			    i, VID(p),
			    storedVL, env, vexpr, aexpr, rexpr, m_vexpr, uexpr,
			    vcaches, c );
		    }
		    cache_commit( vop_caches, c, m_pid );
		}
	    } else {
		// TODO: split caches with pid cache hoisted out of loop
		// only valid when storedVL == VL.
		unsigned short lsVL = rt_ilog2( storedVL );
		constexpr unsigned short lVL = ilog2( VL );
		VID ehi = (to - from + storedVL - 1) >> lsVL;
		VID e = (to - from) >> lsVL;
		auto c = cache_create_no_init( cache_cat( vop_caches, vcaches ), m_pid );

		// TODO: apply WB_ZERO
		cache_init( c, vop_caches, m_pid );

		VID s = 0;

		for( ; s < e; ++s ) {
		    for( VID ss=0; ss < storedVL; ss += VL ) {
			VID i = from + (s << lsVL) + ss;
			// Reconstruct the index based on the availability only of one
			// index value out of storedVL. Each of those index values,
			// however, is one higher than the previous.
			vebosell::process_in_v_e_simd<VL,Operator::is_idempotent>(
			    &edge[idx[s]+ss],
			    midx[s]-idx[s],
			    idx[s+1]-idx[s],
			    i, VID(p),
			    storedVL, env, vexpr, aexpr, rexpr, m_vexpr, uexpr,
			    vcaches, c );
		    }
		}

		for( ; s < ehi; ++s ) {
		    VID nexist = (to - from) % storedVL;
		    for( VID ss=0; ss < storedVL; ss += VL ) {
			VID i = from + (s << lsVL) + ss;
			vebosell::process_in_v_e_simd_mask<VL,Operator::is_idempotent>(
			    &edge[idx[s]+ss],
			    idx[s+1]-idx[s],
			    i, VID(p), storedVL,
			    (nexist / VL) == 0 ? nexist : VL,
			    env, m_vexpr, aexpr, m_rexpr, uexpr,
			    vcaches, c );
		    }
		}

		cache_commit( vop_caches, c, m_pid );
	    }

	    // In case we use streaming load/stores
	    _mm_mfence();
	} );

    // In case we use streaming load/stores
    _mm_mfence();

    // Scan across partitions
    if( Operator::is_scan ) {
	int np = part.get_num_partitions();

#if 0
	std::cerr << "nactv      :";
	constexpr unsigned CID = decltype( expr::cache_select<-64>( vop_caches ) )::cid;
	VID * vpos = std::get<CID>( accum.t ).get_accum();
	for( int p=0; p < np; ++p ) {
	    VID nav = 0;
	    for( unsigned short l=0; l < VL; ++l )
		nav += vpos[p*VL+l];
	    std::cerr << ' ' << nav;
	}
	std::cerr << "\n";
#endif

	for( int p=1; p < np; ++p ) {
	    simd_vector<VID,VL> pp;
	    pp.set1inc( p*VL );
	    auto m = expr::create_value_map2<VL,VL,expr::vk_pid>( pp );
	    expr::cache<> c;
	    env.evaluate( c, m, pvop0 );
	}
	{
	    simd_vector<VID,VL> pp;
	    pp.set1inc0();
	    auto m = expr::create_value_map2<VL,VL,expr::vk_pid>( pp );
	    expr::cache<> c;
	    env.evaluate( c, m, pvopf0 );
	}
    }

    accum_destroy( accum );
}

#endif // GRAPTOR_DSL_EMAP_VEBOSLIMSELL_H
