// -*- c++ -*-
#ifndef GRAPTOR_DSL_EMAP_GRAPTORCSCVREDUCENOTCACHED_H
#define GRAPTOR_DSL_EMAP_GRAPTORCSCVREDUCENOTCACHED_H

// This function implements a Grazelle-like vectorization although extended
// to short-circuit inactive vertices.
template<unsigned short VL, typename GraphType, typename GraphPartitionType,
	 typename Extractor, typename AExpr, typename VExpr, typename MVExpr,
	 typename MRExpr, typename Cache>
__attribute__((always_inline, flatten))
static inline void GraptorCSCVReduceNotCached(
    const GraphType & GA,
    const GraphPartitionType & GP,
    int p,
    const partitioner & part,
    const Extractor & extractor,
    const VID * edge,
    const EID nvec,
    const VID vslim,
    simd_vector<VID,1> pvec, // VL> pvec,
    const AExpr & aexpr,
    const VExpr & vexpr,
    const MVExpr & m_vexpr,
    const MRExpr & m_rexpr,
    Cache & c ) {
    const EID dmax = 1 << (GP.getMaxVL() * GP.getDegreeBits());

    simd_vector<VID, 1> sdst( (VID)part.start_of(p) );
    const simd_vector<VID, VL> vmask( extractor.get_mask() );

    for( EID s=0; s < nvec && sdst.at(0) < vslim; s += VL ) {
	simd_vector<VID, VL> edata;
	edata.load( &edge[s] );
	_mm_prefetch( &edge[s + 256], _MM_HINT_NTA );

	VID code = extractor.extract_degree( edata.data() );

	simd_vector<VID, VL> vdecode(
	    extractor.extract_source( edata.data() ) );

	// std::cerr << "= vertex s=" << s << " nvec=" << nvec << " sdst=" << sdst.at(0) << " code=" << code << "\n";
	
	// Check all lanes are active.
	// This is a vector expression, followed by logical reduction.
	// However, as there is only one destination, if any lane
	// indicates convergence, then all lanes should terminate.
	// The reduction should be performed using logical AND, while
	// in the case of SELL where destinations are different,
	// reduction should use logical OR.
	auto md = expr::create_value_map_new<VL>(
	    expr::create_entry<expr::vk_dst>( sdst ), // vdst
	    expr::create_entry<expr::vk_pid>( pvec ) );
	if( unlikely( !expr::evaluate_bool_any( c, md, aexpr ) ) ) {
	    VID vdeltas = code == dmax-2 ? 0 : 1;
	    VID deg = code >> 1;
	    deg = GRAPTOR_DEGREE_MULTIPLIER * deg;
	    using SVID = typename std::make_signed<VID>::type;
	    deg &= ~-SVID(code == 1);
	    s += VL * EID(deg);
	    sdst += simd::vector<VID,1>( vdeltas );
	} else {
	    simd_vector<VID, VL> vsrc = vdecode;

	    // The code represents either a degree or a delta value for the
	    // destination. Code == 1 in the final vector (degree is zero
	    // and the termination bit is set)
	    if( __builtin_expect( (code != 1), 1 ) ) {
		// No mask required on validity of vsrc
		// apply op vsrc, vdst;
		auto m = expr::create_value_map_new<VL>(
		    expr::create_entry<expr::vk_pid>( simd::vector<VID,1>( p ) ),
		    expr::create_entry<expr::vk_src>( vsrc ),
		    expr::create_entry<expr::vk_dst>( sdst ) );
		auto rval_output = expr::evaluate( c, m, vexpr );
	    } else { // Final iteration - mask
		// apply op vsrc, vdst;
		auto m = expr::create_value_map_new<VL>(
		    expr::create_entry<expr::vk_pid>( simd::vector<VID,1>( p ) ),
		    expr::create_entry<expr::vk_src>( vsrc ),
		    expr::create_entry<expr::vk_mask>( vmask ),
		    expr::create_entry<expr::vk_dst>( sdst ) );
		auto rval_output = expr::evaluate( c, m, m_vexpr );
		sdst += simd::vector<VID,1>( 1 );
	    }
	}
    }

    // Note: this is scalar. Should be vectorized, but differently than above.
    VID vs = part.start_of( p );
    VID ve = sdst.at(0); // part.end_of( p );
    simd::vector<VID,1> vv( vs );
    
    for( VID v=vs; v < ve; v++ ) { // v += VL ) {
	// Evaluate hoisted part of expression.
	auto m = expr::create_value_map_new<VL>(
	    expr::create_entry<expr::vk_dst>( vv ),
	    expr::create_entry<expr::vk_pid>( pvec ) );
	expr::evaluate( c, m, m_rexpr );

	vv += simd::vector<VID,1>( (VID)1 );
    }

    assert( sdst.at(0) == vslim );
}

template<unsigned short VL, graptor_mode_t M, typename AExpr,
	 typename VExpr, typename MVExpr, typename MRExpr,
	 typename VOPCache>
__attribute__((always_inline))
static inline void GraptorCSCVReduceNotCachedDriver(
    const GraphVEBOGraptor<M> & GA,
    int p,
    const GraphCSxSIMDDegreeMixed & GP,
    const partitioner & part,
    const AExpr & aexpr,
    const VExpr & vexpr,
    const MVExpr & m_vexpr,
    const MRExpr & m_rexpr,
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

    simd_vector<VID, 1> pvec1( (VID)p );

    auto m_pid = expr::create_value_map_new<VL>(
	expr::create_entry<expr::vk_pid>( pvec1 ) );
    simd_vector<VID, VL> pzero;

    auto extractor = simd_vector<VID,VL>::traits::create_extractor(
	GP.getDegreeBits(), GP.getDegreeShift() );

    // Split caches with pid cache hoisted out of loop
    // This is only valid when storedVL == VL.
    auto c = cache_create_no_init( vop_caches, m_pid );
    cache_init( c, vop_caches, m_pid ); // partial init

    // timer tm;
    // tm.start();

    GraptorCSCVReduceNotCached<VL>( GA, GP, p, part, extractor, edge, nvec_d1,
				    GP.getSlimVertex(),
				    pvec1, aexpr, vexpr, m_vexpr, m_rexpr, c );

    cache_commit( vop_caches, c, m_pid );
}

template<unsigned short VL, typename Operator>
__attribute__((always_inline))
static inline void csc_vector_with_cache_loop(
    const GraphVEBOGraptor<gm_csc_vreduce_not_cached> & GA,
    Operator & op, const partitioner & part ) {
    // Fused CSC + vertexop version where vertexop may involve a scan
    unsigned short storedVL = GA.getMaxVL();

    // std::cerr << "VEBOGraptor with VL=" << VL << " VEL=" << sizeof(VID)
    // << " vlen=" << (VL * sizeof(VID)) << " bytes\n";

    assert( VL == storedVL && "restriction" ); // TODO: for now...

    // Variables (actually parameters because they are constant within the AST)
    auto v_src = expr::value<simd::ty<VID,VL>,expr::vk_src>();
    auto v_pid0 = // expr::make_unop_incseq<VL>(
	expr::value<simd::ty<VID,1>,expr::vk_pid>(); // );
    auto v_sdst = expr::value<simd::ty<VID,1>,expr::vk_dst>();
    auto v_dst = expr::make_unop_broadcast<VL>( v_sdst );
    auto v_adst = expr::value<simd::ty<VID,1>,expr::vk_dst>();
    auto v_one = expr::value<simd::ty<VID,VL>,expr::vk_mask>();

    // First version without mask
    auto vexpr0 = op.relax( v_src, v_dst );

    // Second version with mask.
    auto m_vexpr0
	= op.relax( add_mask( v_src, make_cmpne( v_src, v_one ) ), v_dst );

    // Vertexop - no mask
    auto vop0 = op.vertexop( v_sdst );
    auto accum = expr::extract_accumulators( vop0 );
    auto vop1 = expr::rewrite_privatize_accumulators( vop0, part, accum, v_pid0 );
    auto pvop0 = expr::accumulate_privatized_accumulators( v_pid0, accum );
    auto pvopf0 = expr::final_accumulate_privatized_accumulators( v_pid0, accum );

    auto vop_caches = expr::extract_cacheable_refs<expr::vk_pid>( vop1 );
    auto vop2 = expr::rewrite_caches<expr::vk_pid>( vop1, vop_caches );

    // Loop part
    auto vexpr3 = expr::rewrite_mask_main( vexpr0 );
    auto vexpr = expr::rewrite_reduce( vexpr3 );

    // Post-loop part
    auto rexpr = expr::rewrite_mask_main( vop2 );

    // Loop termination condition (active check)
    auto aexpr0 = expr::make_unop_switch_to_vector( op.active( v_adst ) );
    auto aexpr1 = expr::make_unop_reduce( aexpr0, expr::redop_logicalor() );
    auto aexpr = expr::rewrite_mask_main( aexpr1 );

    // Loop part
    auto m_vexpr3 = expr::rewrite_mask_main( m_vexpr0 );
    auto m_vexpr = expr::rewrite_reduce( m_vexpr3 );

    // Post-loop part
    auto m_rexpr3 = expr::rewrite_mask_main( vop2 );
    auto m_rexpr = expr::rewrite_reduce( m_rexpr3 );

    // TODO:
    // * strength reduction on division by storedVL
    // * pass expressions by reference (appears some copying is done)

    map_partitionL( part, [&]( int p ) {
	    GraptorCSCVReduceNotCachedDriver<VL>(
		GA, p, GA.getCSC( p ), part,
		aexpr, vexpr, m_vexpr, m_rexpr,
		vop_caches );
	} );

    // Scan across partitions; scalar - tied to data layout
    if( Operator::is_scan )
	emap_scan<1,VID>( part, pvop0, pvopf0 );

    accum_destroy( accum );
}

#endif // GRAPTOR_DSL_EMAP_GRAPTORCSCVREDUCENOTCACHED_H
