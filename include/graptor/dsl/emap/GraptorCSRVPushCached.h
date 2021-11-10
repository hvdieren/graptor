// -*- c++ -*-
#ifndef GRAPTOR_DSL_EMAP_GRAPTORCSRVPUSHCACHED_H
#define GRAPTOR_DSL_EMAP_GRAPTORCSRVPUSHCACHED_H

// TODO:
// - LICM is disabled???
// - use unop_incseq( scalar vk_pid ) vs vectorized vk_pid
// - use emap_scan

// This function implements a reverse Grazelle-like vectorization,
// using push-style traversal as opposed to pull-style
template<unsigned short VL, typename Extractor,
	 typename MVExpr, typename VCache,
	 typename Cache>
__attribute__((always_inline, flatten))
static inline void GraptorCSRVPushCached(
    const GraphVEBOGraptor<gm_csr_vpush_cached> & GA,
    const GraphCSRSIMDDegreeMixed<gm_csr_vpush_cached> & GP,
    int p,
    const partitioner & part,
    const Extractor extractor,
    const VID * edge,
    const EID nvec,
    const VID vslim,
    simd_vector<VID,VL> pvec,
    const MVExpr & m_vexpr,
    const VCache & vcaches,
    Cache & c ) {

    // std::cerr << "\n*** csr_value p=" << p << " lo=" << part.start_of(p) << " hi=" << part.end_of(p) << " ***\n";

    // TODO: instead of jumping by at least 1, we know that in this
    //       encoding jumps are of size at least maxVL. This would allow
    //       slightly larger values to be encoded.
    const simd_vector<VID, VL> vone( (VID)VL );
    const simd_vector<VID, VL> vmask( extractor.get_mask() );

    const EID dmax = 1 << (GP.getMaxVL() * GP.getDegreeBits());

#if GRAPTOR_CSR_INDIR == 1
    const VID * redir = GP.getRedirP();
    VID sidx = 0;
    simd_vector<VID, VL> vsrc( (VID)redir[sidx], lo_constant );
#else
    simd_vector<VID, VL> vsrc( (VID)0, lo_constant );
#endif

    EID s=0;
    while( s < nvec ) {
	// Load cache for vdst.
	// The current implementation has not been tested for the treatment
	// of values loaded for pvec (intermediates for scans in the vertexop).
	// As such, pvec is temporarily removed from the value map in order
	// to generate an error in case scans on pvec are performed.
	auto m = expr::create_value_map_new<VL>(
	    expr::create_entry<expr::vk_src>( vsrc ) );

	// The cache is read-only, based on vk_src; no need to write back.
	// auto c = cache_create( vcaches, m );
	cache_init( c, vcaches, m );

	EID code;
	do {
	    // Extract degree. Increment by one to adjust encoding
	    simd_vector<VID, VL> edata;
	    edata.load( &edge[s] );
	    code = extractor.extract_degree( edata.data() );

	    VID deg = code;
	    if( deg < dmax-1 )
		++deg;

	    // Do extract_source only first time, rest uses
	    // raw source data (no degree encoded)
	    simd_vector<VID, VL> vdst(
		extractor.extract_source( edata.data() ) );

	    EID smax = std::min( s + deg * VL, nvec );

	    // std::cerr << "= run s=" << s << " smax=" << smax << " sidx=" << sidx << " src=" << redir[sidx] << " deg=" << (VL * deg) << " edge=" << edge << "\n";

	    while( s < smax ) {
		_mm_prefetch( &edge[s + 256], _MM_HINT_NTA );

		// std::cerr << "      s=" << (s-VL) << " smax=" << smax << " vdst[0]=" << vdst.at(0) << " vdst[15]=" << vdst.at(15) << " edge[s]=" << edge[s-VL] << "\n";

		// validate( GA.getCSR(), vsrc, vdst );

 		// Note: consider to use AExpr to tighten mask on scatter
		// Note: we cannot use the unmasked expression here (assuming
		//       that all destinations are initially valid) because
		//       of the way that inter-lane conflicts are handled.
		// apply op vsrc, vdst;
		auto m = expr::create_value_map_new<VL>(
		    expr::create_entry<expr::vk_pid>( simd::vector<VID,1>( p ) ),
		    expr::create_entry<expr::vk_mask>( vmask ),
		    expr::create_entry<expr::vk_src>( vsrc ),
		    expr::create_entry<expr::vk_dst>( vdst ) );
		expr::evaluate( c, m, m_vexpr );

		s += VL;
		vdst.load( &edge[s] );
	    }
	} while( code == dmax-1 );

#if GRAPTOR_CSR_INDIR == 1
	++sidx;
	vsrc = simd_vector<VID, VL>( (VID)redir[sidx], lo_constant );
#else
	vsrc += simd_vector<VID, VL>( (VID)1 );
#endif
    }

    assert( s == nvec ); // || s + VL/2 == nvec );
}

template<unsigned short VL, graptor_mode_t M, typename MVExpr,
	 typename RExpr, typename VCache, typename VOPCache>
static inline void GraptorCSRVPushCachedDriver(
    const GraphVEBOGraptor<gm_csr_vpush_cached> & GA,
    int p,
    const GraphCSRSIMDDegreeMixed<gm_csr_vpush_cached> & GP,
    const partitioner & part,
    const MVExpr & m_vexpr,
    const RExpr & rexpr,
    const VCache & vcaches,
    const VOPCache & vop_caches ) {
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

    simd_vector<VID, VL> pvec;
    pvec.set1inc( VL*VID(p) );
    auto m_pid = expr::create_value_map_new<VL>(
	expr::create_entry<expr::vk_pid>( pvec ) );
    simd_vector<VID, VL> pzero;

    auto extractor = simd_vector<VID,VL>::traits::create_extractor(
	GP.getDegreeSkipBits(), GP.getDegreeSkipShift() );

    // Split caches with pid cache hoisted out of loop
    // This is only valid when storedVL == VL.
    auto c = cache_create_no_init(
	cache_cat( vop_caches, vcaches ), m_pid );
    cache_init( c, vop_caches, m_pid ); // partial init

    GraptorCSRVPushCached<VL>( GA, GP, p, part, extractor, edge, nvec_d1,
			       GP.getSlimVertex(), pvec,
			       m_vexpr, vcaches, c );

    GraptorCSRVertexOp<VL>( p, part, rexpr, c );

    cache_commit( vop_caches, c, m_pid );
}

template<typename EMapConfig, typename Operator>
__attribute__((flatten))
static inline void emap_push(
    const GraphVEBOGraptor<gm_csr_vpush_cached> & GA,
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
    auto v_pid0 = expr::value<simd::ty<VID,VL>,expr::vk_pid>();

    // Vertexop - no mask
    auto vop0 = op.vertexop( v_dst );
    auto accum = expr::extract_accumulators( vop0 );
    auto vop1 = expr::rewrite_privatize_accumulators( vop0, part, accum, v_pid0 );
    auto pvop0 = expr::accumulate_privatized_accumulators( v_pid0, accum );
    auto pvopf0 = expr::final_accumulate_privatized_accumulators( v_pid0, accum );

    auto vop_caches = expr::extract_cacheable_refs<expr::vk_pid>( vop1 );
    auto vop2 = expr::rewrite_caches<expr::vk_pid>( vop1, vop_caches );
    auto rexpr = expr::rewrite_mask_main( vop2 );

    // Relax operation with mask to indicate validity of destination vertex IDs.
    auto m_vexpr0
	= op.relax( v_src, add_mask( v_dst, make_cmpne( v_dst, v_one ) ) );
    auto vcaches
	= expr::extract_readonly_refs<expr::vk_src>( m_vexpr0, vop_caches );
    auto m_vexpr1 = expr::rewrite_caches<expr::vk_src>( m_vexpr0, vcaches );
    auto m_vexpr = expr::rewrite_mask_main( m_vexpr1 );

    map_partitionL( part, [&]( int p ) {
	    GraptorCSRVPushCachedDriver<VL>(
		GA, p, GA.getCSC( p ), part,
		m_vexpr, rexpr,
		vcaches, vop_caches );
	} );

    // Scan across partitions
    if( Operator::is_scan )
	// emap_scan<VL,VID>( part, pvop0, pvopf0 ); -- uses unop_incseq(scalar pid)
    {
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

    accum_destroy( accum );
}

#endif // GRAPTOR_DSL_EMAP_GRAPTORCSRVPUSHCACHED_H

