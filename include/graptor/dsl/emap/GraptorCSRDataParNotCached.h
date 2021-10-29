// -*- c++ -*-
#ifndef GRAPTOR_DSL_EMAP_GRAPTORCSRDATAPARNOTCACHED_H
#define GRAPTOR_DSL_EMAP_GRAPTORCSRDATAPARNOTCACHED_H

template<unsigned short VL, typename GraphType,
	 typename Extractor,
	 typename AExpr, typename VExpr, typename MVExpr, typename MRExpr,
	 typename Environment, typename Config>
__attribute__((always_inline, flatten))
static inline VID GraptorCSRDataParNotCached(
    const GraphType & GA,
    const GraphCSRSIMDDegreeMixed & GP,
    int p,
    const partitioner & part,
    const Extractor extractor,
    const VID * edge,
    const EID nvec,
    simd::detail::vec<simd::ty<VID,1>,simd::lo_constant> pvec1,
    const AExpr & aexpr,
    const VExpr & vexpr,
    const MVExpr & m_vexpr,
    const MRExpr & m_rexpr,
    const Environment & env,
    const Config & config ) {

    using vid_type = simd::ty<VID,VL>;

    // constexpr unsigned short lgVL = ilog2( VL );
    simd_vector<VID, VL> vstep( (VID)VL );
    VID sidx = 0;
    VID n = GA.numVertices();

    const simd_vector<VID, VL> vmask( extractor.get_mask() );

#if GRAPTOR_CSR_INDIR == 0
    simd_vector<VID, VL> vidx;
    vidx.set1inc0();
    VID vmax = n;
#else
    const VID * redir_p = GP.getRedirP();
    VID redir_nnz = GP.getRedirNNZ();
    VID vmax = redir_nnz;
    if( vmax % VL ) // roundup
	vmax += VL - ( vmax % VL );
#endif

    for( EID s=0; s < nvec; s += VL ) {
	// std::cerr << "vidx[0]=" << vidx.at(0) << "\n";
	// assert( vdst.at(0) < part.end_of(p) );
	// assert( vdst.at(0) < GA.numVertices() );

#if GRAPTOR_CSR_INDIR == 0
	simd_vector<VID, VL> vsrc = vidx;
#else
	// load vsrc from redir array on the basis of scalar index
	// vsrc should be lo_unknown and trigger gather instruction later on
	// simd_vector_ref<VID, VID, VL> vredir( const_cast<VID *>(redir_p), sidx );
	auto vredir = simd::template create_vector_ref_cacheop<
	    vid_type,VID,array_encoding<VID>,false>(
		const_cast<VID *>( redir_p ), sidx );
	auto vsrc = vredir.load();
#endif

	auto edata = simd::template load_from<vid_type>( &edge[s] );
	_mm_prefetch( &edge[s + 256], _MM_HINT_NTA );

	VID code = extractor.extract_degree( edata.data() );
	auto vdst = simd::template create_unknown<vid_type>(
	    extractor.extract_source( edata.data() ) );

/*
	for( unsigned short l=0; l < VL; ++l ) {
	    assert( vdst.at(l) != ~VID(0) );
	    assert( vdst.at(l) == (~VID(0))>>1 || GA.getCSR().hasEdge( vsrc.at(l), vdst.at(l) ) );
	}
*/
	// std::cerr << "SIMD sidx=" << sidx << " vsrc[0]=" << vsrc.at(0) << " vdst[0]=" << vdst.at(0) << " code=" << code << "\n";

	// apply op vsrc, vdst;
	auto m = expr::create_value_map_new<VL>(
	    expr::create_entry<expr::vk_pid>( pvec1 ),
	    expr::create_entry<expr::vk_src>( vsrc ),
	    expr::create_entry<expr::vk_mask>( vmask ),
	    expr::create_entry<expr::vk_dst>( vdst ) );
	expr::cache<> c;
	auto rval_output = env.evaluate( c, m, m_vexpr );

	// VID vstep = ( code & 1 ) << lgVL;
	VID vstep = code == 1 ? VL : 0;
#if GRAPTOR_CSR_INDIR == 0
	vidx += simd_vector<VID, VL>( (VID)vstep );
#endif
	sidx += vstep;
    }

    assert( nvec == 0 || sidx == vmax );
    
    return sidx;
}

template<unsigned short VL, graptor_mode_t M, typename AExpr, typename VExpr, typename MVExpr,
	 typename RExpr, typename MRExpr, typename MRExprVOP, typename VCache,
	 typename Environment, typename Config>
static inline void GraptorCSRDataParNotCachedDriver(
    const GraphVEBOGraptor<M> & GA,
    int p,
    const GraphCSRSIMDDegreeMixed & GP,
    const partitioner & part,
    const AExpr & aexpr,
    const VExpr & vexpr,
    const MVExpr & m_vexpr,
    const RExpr & rexpr,
    const MRExpr & m_rexpr,
    const MRExprVOP & m_rexpr_vop,
    const VCache & vcaches,
    const Environment & env,
    const Config & config ) {
    // Get list of edges
    const VID * edge = GP.getEdges();
    const EID nvec2 = GP.numSIMDEdgesDeg2();
    const EID nvec1 = GP.numSIMDEdgesDeg1();
    const EID nvec_d1 = GP.numSIMDEdgesDelta1();
    const EID nvec_dpar = GP.numSIMDEdgesDeltaPar();
    const EID nvec = GP.numSIMDEdges();

    // std::cerr << "PARTITION " << p << " nvec=" << nvec
	      // << " from=" << part.start_of(p)
	      // << " to=" << part.end_of(p)
	      // << "\n";
    // assert( part.end_of(p) >= part.start_of(p) );

    using vid_type = simd::ty<VID,VL>;

    auto pvec1 = simd::template create_constant<simd::ty<VID,1>>( VL*(VID)p );
    auto m_pid = expr::create_value_map_new<VL>(
	expr::create_entry<expr::vk_pid>( pvec1 ) );
    simd_vector<VID, VL> pzero;

    auto extractor = simd_vector<VID,VL>::traits::create_extractor(
	GP.getDegreeSkipBits(), GP.getDegreeSkipShift() );

    VID sidx;

    sidx = GraptorCSRDataParNotCached<VL>(
	GA, GP, p, part, extractor, &edge[nvec_d1], nvec_dpar,
	pvec1, aexpr, vexpr, m_vexpr, m_rexpr, env, config );

/* should be working
    sidx = csr_dpar_deg2<VL>( GA, GP, p, part,
			      &edge[nvec_d1+nvec_dpar], nvec2, sidx, pvec,
			      vexpr, m_vexpr, m_rexpr, m_rexpr_vop, vcaches );

    sidx = csr_dpar_deg1<VL>( GA, GP, p, part,
			      &edge[nvec_d1+nvec_dpar+nvec2], nvec1, sidx, pvec,
			      vexpr, m_vexpr, m_rexpr, m_rexpr_vop, vcaches );
*/

    expr::cache<> c;
    GraptorCSRVertexOp<VL>( env, p, part, rexpr/*m_rexpr_vop*/, c );
}


template<typename EMapConfig, typename Operator>
__attribute__((flatten))
static inline void emap_push(
    const GraphVEBOGraptor<gm_csr_datapar_not_cached> & GA,
    Operator & op, const partitioner & part ) {
    constexpr unsigned short VL = EMapConfig::VL;
    unsigned short storedVL = GA.getMaxVL();

    // std::cerr << "VEBOGraptor with VL=" << VL << " VEL=" << sizeof(VID)
    // << " vlen=" << (VL * sizeof(VID)) << " bytes\n";

    assert( VL == storedVL && "restriction" ); // TODO: for now...

    // Variables (actually parameters because they are constant within the AST)
    auto v_src = expr::value<simd::ty<VID,VL>,expr::vk_src>();
    auto v_dst = expr::value<simd::ty<VID,VL>,expr::vk_dst>();
    auto v_one = expr::value<simd::ty<VID,VL>,expr::vk_mask>();
    auto v_pid0 = expr::make_unop_incseq<VL>(
	expr::value<simd::ty<VID,1>,expr::vk_pid>() );
    auto v_edge = expr::template make_unop_incseq<VL>(
	expr::value<simd::ty<EID,1>,expr::vk_edge>() );

    // First version without mask
    auto vexpr0 = op.relax( v_src, v_dst, v_edge );

    // Second version with mask.
    auto m_vexpr0
	= op.relax( v_src, add_mask( v_dst, make_cmpne( v_dst, v_one ) ),
		    v_edge );

    // Vertexop - no mask
    auto vop0 = op.vertexop( v_dst );
    auto accum = expr::extract_accumulators( vop0 );
    auto vop1 = expr::rewrite_privatize_accumulators( vop0, part, accum, v_pid0 );
    auto pvop0 = expr::accumulate_privatized_accumulators( v_pid0, accum );
    auto pvopf0 = expr::final_accumulate_privatized_accumulators( v_pid0, accum );

    auto vop2 = vop1;

    // LICM: only inner-most redop needs to be performed inside loop.
    //       the other relates to the new frontier and can be carried over
    //       through registers as opposed to variables.
    auto vexpr1 = vexpr0;
    auto rexpr1 = vop2;

    // It is assumed cache is the same / masked version may have additional
    // update bitmask
    expr::cache<> vcaches;

    // Loop part
    auto vexpr2 = expr::rewrite_caches<expr::vk_src>( vexpr1, vcaches );
    // auto vexpr2 = vexpr1;
    auto vexpr = expr::rewrite_mask_main( vexpr2 );
    // using Expr = decltype( vexpr );

    // Post-loop part
    auto rexpr2 = expr::rewrite_caches<expr::vk_src>( rexpr1, vcaches );
    // auto rexpr2 = rexpr1;
    auto rexpr = expr::rewrite_mask_main( rexpr2 );

    // Loop termination condition (convergence check)
    auto aexpr0 = op.active( v_dst );
    auto aexpr1 = expr::make_unop_reduce( aexpr0, expr::redop_logicalor() );
    auto aexpr2 = expr::rewrite_caches<expr::vk_src>( aexpr1, vcaches );
    // auto aexpr2 = aexpr1;
    auto aexpr = expr::rewrite_mask_main( aexpr2 );

    // LICM: only inner-most redop needs to be performed inside loop.
    //       the other relates to the new frontier and can be carried over
    //       through registers as opposed to variables.
    auto m_vexpr1 = m_vexpr0;
    // auto m_rexpr0 = nop;
    auto m_rexpr1 = vop2;
    // csc_vreduce_ requires these two as separate expressions, one vectorized
    // and one for scalar execution.
    // auto m_rexpr1_v = m_rexpr0;
    auto m_rexpr1_vop = vop2;

    // Loop part
    auto m_vexpr2 = expr::rewrite_caches<expr::vk_src>( m_vexpr1, vcaches );
    // auto m_vexpr2 = m_vexpr1;
    auto m_vexpr = expr::rewrite_mask_main( m_vexpr2 );

    // Post-loop part
    auto m_rexpr2 = expr::rewrite_caches<expr::vk_src>( m_rexpr1, vcaches );
    // auto m_rexpr2 = m_rexpr1;
    auto m_rexpr = expr::rewrite_mask_main( m_rexpr2 );
    // auto m_rexpr2_v = expr::rewrite_caches<expr::vk_src>( m_rexpr1_v, vcaches );
    // auto m_rexpr_v = expr::rewrite_mask_main( m_rexpr2_v );
    auto m_rexpr2_vop
	= expr::rewrite_caches<expr::vk_src>( m_rexpr1_vop, vcaches );
    // auto m_rexpr2_vop = m_rexpr1_vop;
    auto m_rexpr_vop = expr::rewrite_mask_main( m_rexpr2_vop );

    // TODO:
    // * strength reduction on division by storedVL
    // * pass expressions by reference (appears some copying is done)

    using Cfg = std::decay_t<decltype(op.get_config())>;
    
    static_assert( Cfg::max_vector_length() >= VL,
		   "Cannot respect config option of maximum vector length" );

    auto env = expr::eval::create_execution_environment_with(
	op.get_ptrset(), vcaches,
	vexpr, m_vexpr, rexpr, m_rexpr, m_rexpr_vop,
	m_rexpr_vop, pvop0, pvopf0 );

    map_partition<Cfg::is_parallel()>( part, [&]( int p ) {
	    GraptorCSRDataParNotCachedDriver<VL>(
		GA, p, GA.getCSC( p ), part,
		aexpr, vexpr, m_vexpr, rexpr, m_rexpr,
		m_rexpr_vop, vcaches, env, op.get_config() );
	} );

    // Scan across partitions
    if constexpr ( Operator::is_scan )
	emap_scan<VL,VID>( env, part, pvop0, pvopf0 );
/*
    if( Operator::is_scan ) {
	int np = part.get_num_partitions();
	for( int p=1; p < np; ++p ) {
	    simd_vector<VID,VL> pp;
	    pp.set1inc( p*VL );
	    auto m = expr::create_value_map_new<VL>(
		expr::create_entry<expr::vk_pid>( pp ) );
	    expr::cache<> c;
	    expr::evaluate( c, m, pvop0 );
	}
	{
	    simd_vector<VID,VL> pp;
	    pp.set1inc0();
	    auto m = expr::create_value_map_new<VL>(
		expr::create_entry<expr::vk_pid>( pp ) );
	    expr::cache<> c;
	    expr::evaluate( c, m, pvopf0 );
	}
    }
*/

    accum_destroy( accum );
}


#endif // GRAPTOR_DSL_EMAP_GRAPTORCSRDATAPARNOTCACHED_H
