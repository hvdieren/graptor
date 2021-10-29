// -*- c++ -*-
#ifndef GRAPTOR_DSL_EMAP_GRAPTORCSRVPUSHNOTCACHED_H
#define GRAPTOR_DSL_EMAP_GRAPTORCSRVPUSHNOTCACHED_H

// TODO:
// - use emap_scan

template<unsigned short VL, graptor_mode_t M, typename MVExpr,
	 typename RExpr, typename VOPCaches,
	 typename Environment, typename Config>
__attribute__((flatten))
static inline void GraptorCSRVPushNotCachedDriver(
    const GraphVEBOGraptor<M> & GA,
    int p,
    const GraphCSRSIMDDegreeMixed & GP,
    const partitioner & part,
    const MVExpr & m_vexpr,
    const RExpr & rexpr,
    const VOPCaches & vop_caches,
    const Environment & env,
    const Config & config ) {
    // Get list of edges
    const VID * edge = GP.getEdges();
    const EID nvec = GP.numSIMDEdgesDelta1();

    // std::cerr << "PARTITION " << p << " nvec=" << nvec
	      // << " from=" << part.start_of(p)
	      // << " to=" << part.end_of(p)
	      // << "\n";
    // assert( part.end_of(p) >= part.start_of(p) );

    using vid_type = simd::ty<VID,VL>;

    auto pvec1 = simd::template create_constant<simd::ty<VID,1>>( VL*(VID)p );
    auto m_pid = expr::create_value_map_new<VL>(
	expr::create_entry<expr::vk_pid>( pvec1 ) );

    auto extractor = vid_type::traits::create_extractor(
	GP.getDegreeSkipBits(), GP.getDegreeSkipShift() );

    const simd::vec<vid_type,lo_unknown> vmask( extractor.get_mask() );

    // Split caches with pid cache hoisted out of loop
    // This is only valid when storedVL == VL.
    auto c = cache_create_no_init( vop_caches, m_pid );
    cache_init( env, c, vop_caches, m_pid ); // partial init

    // GraptorCSRVPushNotCached<VL>( GA, GP, p, part, extractor, edge, nvec,
    // pvec, m_vexpr, c );

#if GRAPTOR_CSR_INDIR == 1
    const VID * redir = GP.getRedirP();
    VID sidx = 0;
    // simd_vector<VID, VL> vsrc( (VID)redir[sidx], lo_constant );
    auto vsrc = simd::template create_constant<vid_type>( (VID)redir[sidx] );
#else
    // simd_vector<VID, VL> vsrc( (VID)0, lo_constant );
    auto vsrc = simd::template create_constant<vid_type>( (VID)0 );
#endif

    for( EID s=0; s < nvec; s += VL ) {
	//std::cerr << "= vertex s=" << s << " vsrc[0]=" << vsrc.at(0) << "\n";
	
	// Extract degree. Increment by one to adjust encoding
	auto edata = simd::template load_from<vid_type>( &edge[s] );

	VID code = extractor.extract_degree( edata.data() );

	_mm_prefetch( &edge[s + 256], _MM_HINT_NTA );

	auto vdst = simd::template create_unknown<vid_type>(
	    extractor.extract_source( edata.data() ) );

	// TODO: if not active (frontier check expression needs to be separated
	//       out like in convergence), then skip over distance as given
	//       by code.

	// std::cerr << "SIMD s=" << s << " vsrc[0]=" << vsrc.at(0) << " vdst[0]=" << vdst.at(0) << " code=" << code << "\n";

	// validate( GA.getCSR(), vsrc, vdst );
	
	// apply op vsrc, vdst;
	auto m = expr::create_value_map_new<VL>(
	    expr::create_entry<expr::vk_pid>( simd::vector<VID,1>( p ) ),
	    expr::create_entry<expr::vk_src>( vsrc ),
	    expr::create_entry<expr::vk_mask>( vmask ),
	    expr::create_entry<expr::vk_dst>( vdst ) );
	auto rval_output = expr::evaluate( c, m, m_vexpr );

	// Using a blend: vsrc += iif( (bool)(code & 1), vzero, vone );
	// instead of a broadcast completes in 3 vs 2 clock cycles on KNL
	// (see https://www.agner.org/optimize/instruction_tables.pdf)
	// but requires an additional live register throughout the loop
	// VID vstep = code & 1;
	VID vstep = ( code == 0 ) ? 1 : 0;
#if GRAPTOR_CSR_INDIR == 1
	sidx += vstep;
	// vsrc = simd_vector<VID, VL>( (VID)redir[sidx], lo_constant );
	vsrc = simd::template create_constant<vid_type>( (VID)redir[sidx] );
#else
	vsrc += simd_vector<VID, VL>( (VID)vstep );
#endif
    }

#if GRAPTOR_CSR_INDIR == 1
    assert( sidx <= GP.getRedirNNZ() && sidx+VL > GP.getRedirNNZ() );
#endif

    GraptorCSRVertexOp<VL>( p, part, rexpr, c );

    cache_commit( env, vop_caches, c, m_pid );
}

template<typename EMapConfig, typename Operator>
__attribute__((flatten))
static inline void emap_push(
    const GraphVEBOGraptor<gm_csr_vpush_not_cached> & GA,
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

    // Relax with mask applied to destinations.
    auto m_vexpr0
	= op.relax( v_src, add_mask( v_dst, make_cmpne( v_dst, v_one ) ),
		    v_edge );

    // Vertexop - no mask
    auto vop0 = op.vertexop( v_dst );
    auto accum = expr::extract_accumulators( vop0 );
    auto vop1 = expr::rewrite_privatize_accumulators( vop0, part, accum, v_pid0 );
    auto pvop0 = expr::accumulate_privatized_accumulators( v_pid0, accum );
    auto pvopf0 = expr::final_accumulate_privatized_accumulators( v_pid0, accum );

    // Post-loop part
    auto vop_caches = expr::extract_cacheable_refs<expr::vk_pid>( vop1 );
    auto rexpr1 = expr::rewrite_caches<expr::vk_pid>( vop1, vop_caches );
    auto rexpr = expr::rewrite_mask_main( rexpr1 );

    // Loop part
    auto m_vexpr = expr::rewrite_mask_main( m_vexpr0 );

    auto env = expr::eval::create_execution_environment_with(
	op.get_ptrset(), vop_caches, m_vexpr, rexpr, pvop0, pvopf0 );

    using Cfg = std::decay_t<decltype(op.get_config())>;
    
    static_assert( Cfg::max_vector_length() >= VL,
		   "Cannot respect config option of maximum vector length" );

    map_partition<Cfg::is_parallel()>( part, [&]( int p ) {
	    GraptorCSRVPushNotCachedDriver<VL>(
		GA, p, GA.getCSC( p ), part,
		m_vexpr, rexpr, vop_caches, env, op.get_config() );
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


#endif // GRAPTOR_DSL_EMAP_GRAPTORCSRVPUSHNOTCACHED_H

