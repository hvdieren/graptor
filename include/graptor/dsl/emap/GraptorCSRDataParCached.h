// -*- c++ -*-
#ifndef GRAPTOR_DSL_EMAP_GRAPTORCSRDATAPARCACHED_H
#define GRAPTOR_DSL_EMAP_GRAPTORCSRDATAPARCACHED_H

template<unsigned short VL, typename GraphType,
	 typename Extractor,
	 typename AExpr, typename MVExpr, typename VCache,
	 typename Cache,
	 typename Environment, typename Config, graptor_mode_t Mode>
__attribute__((flatten))
static inline VID GraptorCSRDataParCached(
    const GraphType & GA,
    const GraphCSRSIMDDegreeMixed<Mode> & GP,
    int p,
    const partitioner & part,
    const Extractor extractor,
    const VID * edge,
    const EID nvec,
    simd::detail::vec<simd::ty<VID,1>,simd::lo_constant> pvec1,
    const AExpr & aexpr,
    const MVExpr & m_vexpr,
    const VCache & vcaches,
    Cache & c,
    const Environment & env,
    const Config & config ) {

    using vid_type = simd::ty<VID,VL>;
    using seid_type = simd::ty<EID,1>;

    VID sidx = 0;
    EID s = 0;
    VID n = GA.numVertices();

    const simd_vector<VID, VL> vmask( extractor.get_mask() );

    EID sedge = part.edge_start_of( p );

#if GRAPTOR_CSR_INDIR == 0
    VID vmax = n;
#else
    const VID * redir_p = GP.getRedirP();
    VID redir_nnz = GP.getRedirNNZ();
    const VID vmax = ( redir_nnz % VL ) ?
	redir_nnz + VL - ( redir_nnz % VL ) : redir_nnz;
#endif

    const EID dmax = 1 << (GP.getMaxVL() * GP.getDegreeBits());

    while( s < nvec && sidx < vmax ) {
	// std::cerr << "vidx[0]=" << vidx.at(0) << "\n";
	// assert( vdst.at(0) < part.end_of(p) );
	// assert( vdst.at(0) < GA.numVertices() );

#if GRAPTOR_CSR_INDIR == 0
	auto vsrc = simd::create_set1inc<vid_type,true>( sidx );
#else
	// load vsrc from redir array on the basis of scalar index
	// vsrc should be lo_unknown and trigger gather instruction later on
	auto vredir = simd::create_vector_ref_cacheop<
	    vid_type,VID,array_encoding<VID>,false>(
		const_cast<VID *>(redir_p), sidx );
	auto vsrc = vredir.load();
#endif

	// Load cache for vsrc -- readonly cache, no commit/reduce
	auto m = expr::create_value_map_new<VL>(
	    expr::create_entry<expr::vk_src>( vsrc ),
	    expr::create_entry<expr::vk_pid>( pvec1 ) );
	cache_init( env, c, vcaches, m );

	VID code;
	VID skip;
	do {
	    // Extract degree. Increment by one to adjust encoding
	    auto edata = simd::template load_from<vid_type>( &edge[s] );
	    code = extractor.extract_degree( edata.data() );
	    VID deg = code >> 1;
	    deg = GRAPTOR_DEGREE_MULTIPLIER * deg + 1;

	    // Should do extract_source only first time, rest should
	    // be raw source data (no degree encoded)
	    auto vdst = simd::template create_unknown<vid_type>(
		extractor.extract_source( edata.data() ) );

	    // check vdst active
	    EID smax = std::min( s + deg * VL, nvec );
	    // std::cerr << "  s=" << s << " smax=" << smax << " deg=" << deg << "\n";

	    auto ma = expr::create_value_map_new<VL>(
		expr::create_entry<expr::vk_pid>( pvec1 ),
		expr::create_entry<expr::vk_src>( vsrc ) );
	    auto r = env.evaluate( c, ma, aexpr );
	    if( expr::is_false( r ) ) {
		s = smax;
	    } else {
		while( s < smax ) {
		    // std::cerr << "   s=" << s << " smax=" << smax << " vsrc[0]=" << vsrc.at(0) << " vdst[0]=" << vdst.at(0) << "\n";

		    // apply op vsrc, vdst;
		    auto sv = simd::create_scalar<seid_type>( sedge+s );
		    auto m = expr::create_value_map_new<VL>(
			expr::create_entry<expr::vk_pid>( pvec1 ),
			expr::create_entry<expr::vk_mask>( vmask ),
			expr::create_entry<expr::vk_src>( vsrc ),
			expr::create_entry<expr::vk_dst>( vdst ),
			expr::create_entry<expr::vk_edge>( sv ) );
		    auto mpack = expr::sb::create_mask_pack(
			vdst != vmask && r.value().asmask() );
		    auto rval_output = env.evaluate( c, m, mpack, m_vexpr );
		    // output.lor_assign( rval_output.mask() );

		    // Proceed to next vector of sources
		    s += VL;

		    _mm_prefetch( &edge[s + 256], _MM_HINT_NTA );

		    vdst = simd::template load_from<vid_type>( &edge[s] );
		}
	    }
	} while( (code & 1) == 0 );

	sidx += VL;
    }

    assert( nvec == 0 || sidx == vmax );
    assert( s == nvec );

    return sidx;
}

template<unsigned short VL, graptor_mode_t M,
	 typename AExpr, typename MVExpr,
	 typename RExpr, typename VCache,
	 typename VOPCache,
	 typename Environment, typename Config>
static inline void GraptorCSRDataParCachedDriver(
    const GraphVEBOGraptor<M> & GA,
    int p,
    const GraphCSRSIMDDegreeMixed<M> & GP,
    const partitioner & part,
    const AExpr & aexpr,
    const MVExpr & m_vexpr,
    const RExpr & rexpr,
    const VCache & vcaches,
    const VOPCache & vop_caches,
    const Environment & env,
    const Config & config ) {
    // Get list of edges
    const VID * edge = GP.getEdges();
    const EID nvec_d1 = GP.numSIMDEdgesDelta1();
    const EID nvec_dpar = GP.numSIMDEdgesDeltaPar();
    const EID nvec = GP.numSIMDEdges();

    // std::cerr << "PARTITION " << p << " nvec=" << nvec
	      // << " from=" << part.start_of(p)
	      // << " to=" << part.end_of(p)
	      // << "\n";
    // assert( part.end_of(p) >= part.start_of(p) );

    auto pvec1 = simd::template create_constant<simd::ty<VID,1>>( VL*(VID)p );
    auto m_pid = expr::create_value_map_new<VL>(
	expr::create_entry<expr::vk_pid>( pvec1 ) );
    simd_vector<VID, VL> pzero;

    auto extractor = vid_type::traits::template create_extractor<
	GraphCSRSIMDDegreeMixed<M>::getDegreeBits(),
	GraphCSRSIMDDegreeMixed<M>::getDegreeShift()>();

    // Split caches with pid cache hoisted out of loop
    // This is only valid when storedVL == VL.
    auto c = cache_create_no_init(
	cache_cat( vop_caches, vcaches ), m_pid );
    cache_init( env, c, vop_caches, m_pid ); // partial init

    VID sidx;

    sidx = GraptorCSRDataParCached<VL>(
	GA, GP, p, part, extractor, &edge[nvec_d1], nvec_dpar,
	pvec1, aexpr, m_vexpr, vcaches, c, env, config );

    GraptorCSRVertexOp<VL>( env, p, part, rexpr, c );

    cache_commit( env, vop_caches, c, m_pid );
}

template<typename EMapConfig, typename Operator>
__attribute__((flatten))
static inline void emap_push(
    const GraphVEBOGraptor<gm_csr_datapar_cached> & GA,
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

    // Call relax with mask.
    auto m_vexpr0
	= op.relax( v_src, add_mask( v_dst, make_cmpne( v_dst, v_one ) ),
		    v_edge );

    // Vertexop - no mask
    auto vop0 = op.vertexop( v_dst );
    auto accum = expr::extract_accumulators( vop0 );
    expr::accum_create( part, accum );
    auto vop1 = expr::rewrite_privatize_accumulators( vop0, part, accum, v_pid0 );
    auto pvop0 = expr::accumulate_privatized_accumulators( v_pid0, accum );
    auto pvopf0 = expr::final_accumulate_privatized_accumulators( v_pid0, accum );

    auto vop_caches = expr::extract_cacheable_refs<expr::vk_pid>( vop1 );
    auto vop2 = expr::rewrite_caches<expr::vk_pid>( vop1, vop_caches );

    // It is assumed cache is the same / masked version may have additional
    // update bitmask
    auto vcaches
	= expr::extract_readonly_refs<expr::vk_src>( m_vexpr0, vop_caches );

    // Post-loop part
    auto rexpr1 = vop2;
    auto rexpr2 = expr::rewrite_caches<expr::vk_src>( rexpr1, vcaches );
    auto rexpr3 = rewrite_internal( rexpr2 );
    auto rexpr = expr::rewrite_mask_main( rexpr3 );

    // Loop part
    auto m_vexpr1 = expr::rewrite_caches<expr::vk_src>( m_vexpr0, vcaches );
    auto m_vexpr2 = rewrite_internal( m_vexpr1 );
    auto m_vexpr = expr::rewrite_mask_main( m_vexpr2 );

    // Frontier check
    auto aexpr0 = op.enabled( v_src );
    auto aexpr1 = rewrite_internal( aexpr0 );
    auto aexpr2 = expr::rewrite_caches<expr::vk_src>( aexpr1, vcaches );
    auto aexpr = expr::rewrite_mask_main( aexpr2 );

    // TODO:
    // * strength reduction on division by storedVL
    // * pass expressions by reference (appears some copying is done)

    using Cfg = std::decay_t<decltype(op.get_config())>;
    
    static_assert( Cfg::max_vector_length() >= VL,
		   "Cannot respect config option of maximum vector length" );

    // Override pointer for aid_eweight with the relevant permutation of the
    // weights for the GA graph.
    auto env = expr::eval::create_execution_environment_op(
	op, cache_cat( vcaches, vop_caches ),
	GA.getWeights() ? GA.getWeights()->get() : nullptr );

    map_partition<Cfg::is_parallel()>( part, [&]( int p ) {
	    GraptorCSRDataParCachedDriver<VL>(
		GA, p, GA.getCSC( p ), part,
		aexpr, m_vexpr, rexpr,
		vcaches, vop_caches, env, op.get_config() );
	} );

    // Scan across partitions
    if constexpr ( Operator::is_scan )
	emap_scan<VL,VID>( env, part, pvop0, pvopf0 );

    accum_destroy( accum );
}

#endif // GRAPTOR_DSL_EMAP_GRAPTORCSRDATAPARCACHED_H
