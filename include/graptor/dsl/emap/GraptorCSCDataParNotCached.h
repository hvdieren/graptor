// -*- c++ -*-
#ifndef GRAPTOR_DSL_EMAP_GRAPTORCSCDATAPARNOTCACHED_H
#define GRAPTOR_DSL_EMAP_GRAPTORCSCDATAPARNOTCACHED_H

// This function implements a SlimSell-like vectorization although extended
// to short-circuit inactive vertices.
template<unsigned short VL, typename GraphType, typename GraphPartitionType,
	 typename Extractor,
	 typename VDstType,
	 typename AExpr, typename MVExpr, typename MRExpr,
	 typename Cache, typename Environment>
__attribute__((always_inline, flatten))
static inline void GraptorCSCDataParNotCached(
    const GraphType & GA,
    const GraphPartitionType & GP,
    int p,
    const partitioner & part,
    const Extractor extractor,
    const VID * edge,
    const EID nvec,
    VDstType pvec,
    VDstType & vdst,
    const AExpr & aexpr,
    const MVExpr & m_vexpr,
    const MRExpr & m_rexpr,
    Cache & rc,
    const Environment & env ) {

    constexpr unsigned short lgVL = ilog2( VL );

    using vid_type = simd::ty<VID,VL>;
    using svid_type = simd::ty<VID,1>;

    auto vdst_start = vdst;

    auto pvec1 = simd::template create_constant<svid_type>( VL*(VID)p );
    const simd::vec<vid_type,lo_unknown> vmask( extractor.get_mask() );

    expr::cache<> c;
    const EID dmax = 1 << (GP.getMaxVL() * GP.getDegreeBits());
    EID s = 0;
    auto sdst = simd::template create_constant<svid_type>( vdst_start.at(0) );

    while( s < nvec /* && sdst.at(0) < part.end_of(p) */ ) {
	// std::cerr << "vdst[0]=" << vdst.at(0) << "\n";

	// Extract degree information. Increment by one to adjust encoding
	auto edata = simd::template load_from<vid_type>( &edge[s] );
	_mm_prefetch( &edge[s + 256], _MM_HINT_NTA );

	VID code = extractor.extract_degree( edata.data() );

	// std::cerr << "SIMD group v0=" << vdst.at(0) << " s=" << s << " code=" << code << " deg=" << deg << " vdelta=" << vdelta << " vdeltas=" << vdeltas << " nvec=" << nvec << "\n";

	// Extract the source VIDs
	auto vdecode = simd::template create_unknown<vid_type>(
	    extractor.extract_source( edata.data() ) );

	// Check all lanes are active.
	auto md = expr::create_value_map_new<VL>(
	    expr::create_entry<expr::vk_dst>( sdst ), // vdst
	    expr::create_entry<expr::vk_pid>( pvec ) );
	if( !env.evaluate_bool( c, md, aexpr ) ) {
// TODO: encode skip distance (non-VEBO version) -> only useful non-VEBO
	    // The code represents either a degree or a delta value for the
	    // destination.
	    // TODO: is this equivalent to vdeltas = (code &1) << lgVL; as in non-convergence case?
	    VID vdeltas = code == dmax-2 ? 0 : VL;
	    VID deg = code >> 1;
	    deg = GRAPTOR_DEGREE_MULTIPLIER * deg;
	    deg &= ~( (code & 1) << (sizeof(deg)*8-1) );
	    ++deg;

	    s += VL * EID(deg);

	    sdst += simd::template create_constant<svid_type>( vdeltas );
	} else {
	    auto vsrc = vdecode;

/*
	    for( VID l=0; l < VL; ++l ) {
		assert( vsrc.at(l) != ~VID(0) );
		assert( vsrc.at(l) == (~VID(0))>>1 || GA.getCSR().hasEdge( vsrc.at(l), vdst.at(l) ) );
	    }
*/

	    // The code represents either a degree or a delta value for the
	    // destination.
	    VID vdelta = (code & 1) << lgVL; // VL if code & 1 is set

	    // apply op vsrc, vdst;
	    auto m = expr::create_value_map_new<VL>(
		expr::create_entry<expr::vk_pid>( pvec1 ),
		expr::create_entry<expr::vk_src>( vsrc ),
		expr::create_entry<expr::vk_mask>( vmask ),
		expr::create_entry<expr::vk_dst>( sdst ) ); // vdst ) );
	    auto rval_output = env.evaluate( c, m, m_vexpr );

	    // Proceed to next vector
	    s += VL;

	    vdecode = simd::template load_from<vid_type>( &edge[s] );

	    sdst += simd::template create_constant<svid_type>( vdelta );
	}
    }

    // Ensure that all vertices are processed, not only those with non-zero
    // degree.
    VID vs = vdst_start.at(0); // part.start_of( p );
    VID ve = part.end_of( p ); // sdst.at(0); // part.end_of( p );
    auto vv = simd::template create_constant<svid_type>( vs );
    
    for( VID v=vs; v < ve; v += VL ) {
	// Evaluate hoisted part of expression.
	auto m = expr::create_value_map_new<VL>(
	    expr::create_entry<expr::vk_dst>( vv ),
	    expr::create_entry<expr::vk_pid>( pvec ) );
	env.evaluate( rc, m, m_rexpr );

	vv += simd::template create_constant<svid_type>( (VID)VL );
    }

    assert( s == nvec );
    vdst.template s_set1inc<true>( sdst.at(0) );
}

template<unsigned short VL, graptor_mode_t M, typename AExpr, typename MVExpr,
	 typename MRExpr, typename MRExprV, typename MRExprVOP,
	 typename VOPCache, typename Environment>
__attribute__((always_inline))
static inline void GraptorCSCDataParNotCachedDriver(
    const GraphVEBOGraptor<M> & GA,
    int p,
    const GraphCSxSIMDDegreeMixed<M> & GP,
    const partitioner & part,
    const AExpr & aexpr,
    const MVExpr & m_vexpr,
    const MRExpr & m_rexpr,
    const MRExprV & m_rexpr_v,
    const MRExprVOP & m_rexpr_vop,
    const VOPCache & vop_caches,
    const Environment & env ) {
    // Get list of edges
    const VID * edge = GP.getEdges();
    const EID nvec2 = GP.numSIMDEdgesDeg2();
    const EID nvec1 = GP.numSIMDEdgesDeg1();
    const EID nvec_d1 = GP.numSIMDEdgesDelta1();
    const EID nvec_dpar = GP.numSIMDEdgesDeltaPar();
    const EID nvec = GP.numSIMDEdges();

    using vid_type = simd::ty<VID,VL>;

    // std::cerr << "PARTITION " << p << " nvec=" << nvec
	      // << " from=" << part.start_of(p)
	      // << " to=" << part.end_of(p)
	      // << "\n";
    // assert( part.end_of(p) >= part.start_of(p) );

    simd_vector<VID, 1> pvec1( VL*(VID)p );
    auto pvec = simd::create_set1inc<vid_type,true>( VL*VID(p) );

    auto m_pid = expr::create_value_map_new<VL>(
	expr::create_entry<expr::vk_pid>( pvec1 ) );
    simd_vector<VID, VL> pzero;

    auto extractor = simd_vector<VID,VL>::traits::create_extractor(
	GP.getDegreeBits(), GP.getDegreeShift() );

    auto c = cache_create( env, vop_caches, m_pid );

    assert( GP.getSlimVertex() % VL == 0  && "Alignment" );
    auto vdst = simd::create_set1inc<vid_type,true>( GP.getSlimVertex() );

    GraptorCSCDataParNotCached<VL>( GA, GP, p, part, extractor, &edge[nvec_d1],
				    nvec_dpar, pvec, vdst, aexpr,
				    m_vexpr, m_rexpr, c, env );

    assert( (vdst.at(0) - GP.getSlimVertex()) + nvec2/2 + nvec1
	    <= GP.numSIMDVertices() );
    
    cache_commit( env, vop_caches, c, m_pid );
}

template<typename EMapConfig, typename Operator>
__attribute__((flatten))
static inline void emap_pull(
    const GraphVEBOGraptor<gm_csc_datapar_not_cached> & GA,
    Operator & op, const partitioner & part ) {
    constexpr unsigned short VL = EMapConfig::VL;
    unsigned short storedVL = GA.getMaxVL();

    // std::cerr << "VEBOGraptor with VL=" << VL << " VEL=" << sizeof(VID)
    // << " vlen=" << (VL * sizeof(VID)) << " bytes\n";

    assert( VL == storedVL && "restriction" ); // TODO: for now...

    // Variables (actually parameters because they are constant within the AST)
    auto v_src = expr::value<simd::ty<VID,VL>,expr::vk_src>();
    auto v_pid0 = expr::make_unop_incseq<VL>(
	expr::value<simd::ty<VID,1>,expr::vk_pid>() );
    // auto v_dst = expr::value<simd::ty<VID,VL>,expr::vk_dst>();
    auto v_dst = expr::make_unop_incseq<VL>(
	expr::value<simd::ty<VID,1>,expr::vk_dst>() );
    auto v_edge = expr::template make_unop_incseq<VL>(
	expr::value<simd::ty<EID,1>,expr::vk_edge>() );
    auto v_adst = v_dst;

    // TODO: create two aexpr's: an initial scalar one (not cached), another cached and vectorized
    // auto v_pid0 = expr::value<simd::ty<VID,1>,expr::vk_pid>();
    auto v_one = expr::value<simd::ty<VID,VL>,expr::vk_mask>();

    // First version without mask
    // auto vexpr0 = op.relax( v_src, v_dst );

    // Second version with mask.
    auto m_vexpr0
	= op.relax( add_mask( v_src, make_cmpne( v_src, v_one ) ), v_dst,
		    v_edge );

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
    // auto vexpr1 = vexpr0;
    auto rexpr1 = vop2;

    // Post-loop part
    auto rexpr = expr::rewrite_mask_main( rexpr1 );

    // Loop termination condition (active check)
    auto aexpr0 = expr::make_unop_switch_to_vector( op.active( v_adst ) );
    auto aexpr1 = expr::make_unop_reduce( aexpr0, expr::redop_logicalor() );
    auto aexpr = expr::rewrite_mask_main( aexpr1 );

    // LICM: only inner-most redop needs to be performed inside loop.
    //       the other relates to the new frontier and can be carried over
    //       through registers as opposed to variables.
    auto m_vexpr1 = m_vexpr0;
    auto m_rexpr0 = expr::noop();
    auto m_rexpr1 = append_pe( m_rexpr0, vop2 );
    // csc_vreduce_ requires these two as separate expressions, one vectorized
    // and one for scalar execution.
    auto m_rexpr1_v = m_rexpr0;
    auto m_rexpr1_vop = vop2;

    // Loop part
    auto m_vexpr = expr::rewrite_mask_main( m_vexpr1 );

    // Post-loop part
    auto m_rexpr = expr::rewrite_mask_main( m_rexpr1 );
    auto m_rexpr_v = expr::rewrite_mask_main( m_rexpr1_v );
    auto m_rexpr_vop = expr::rewrite_mask_main( m_rexpr1_vop );

    auto env = expr::eval::create_execution_environment_with(
	op.get_ptrset(), vop_caches,
	aexpr, m_vexpr, m_rexpr, m_rexpr_v, m_rexpr_vop );

    // TODO:
    // * strength reduction on division by storedVL
    // * pass expressions by reference (appears some copying is done)

    using Cfg = std::decay_t<decltype(op.get_config())>;

    static_assert( Cfg::max_vector_length() >= VL,
		   "Cannot respect config option of maximum vector length" );

    map_partition<Cfg::is_parallel()>( part, [&]( int p ) {
	    GraptorCSCDataParNotCachedDriver<VL>(
		GA, p, GA.getCSC( p ), part,
		aexpr, m_vexpr, m_rexpr, m_rexpr_v, m_rexpr_vop,
		vop_caches, env );
	} );

    // Scan across partitions
    if( Operator::is_scan )
	emap_scan<VL,VID>( env, part, pvop0, pvopf0 );

    accum_destroy( accum );
}

#endif // GRAPTOR_DSL_EMAP_GRAPTORCSCDATAPARNOTCACHED_H
