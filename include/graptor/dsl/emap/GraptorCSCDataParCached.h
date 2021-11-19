// -*- c++ -*-
#ifndef GRAPTOR_DSL_EMAP_GRAPTORCSCDATAPARCACHED_H
#define GRAPTOR_DSL_EMAP_GRAPTORCSCDATAPARCACHED_H

// TODO:
// + non-VEBO case, encode skip distance in final SIMD word.

// This function implements a SlimSell-like vectorization although extended
// to short-circuit inactive vertices.
template<bool Idempotent, unsigned short VL, graptor_mode_t M,
	 typename AExpr, typename MVExpr,
	 typename MRExpr, typename VOPCache, typename VCache,
	 typename VCacheUse, typename AllCaches, typename Environment,
	 typename Config, graptor_mode_t Mode>
__attribute__((always_inline, flatten))
static inline void GraptorCSCDataParCached(
    const GraphVEBOGraptor<M> & GA,
    int p,
    const GraphCSxSIMDDegreeMixed<Mode> & GP,
    const partitioner & part,
    const AExpr & aexpr,
    const MVExpr & m_vexpr,
    const MRExpr & m_rexpr,
    const VOPCache & vop_caches,
    const VCache & vcaches,
    const VCacheUse & vcaches_use,
    const AllCaches & all_caches,
    const Environment & env,
    const Config & config ) {
    // Get list of edges
    const VID * edge = GP.getEdges();
    const EID nvec_d1 = GP.numSIMDEdgesDelta1();
    const EID nvec = GP.numSIMDEdgesDeltaPar();
    edge = &edge[nvec_d1];

    // std::cerr << "PARTITION " << p << " nvec=" << nvec
	      // << " from=" << part.start_of(p)
	      // << " to=" << part.end_of(p)
	      // << "\n";
    // assert( part.end_of(p) >= part.start_of(p) );

    using vid_type = simd::ty<VID,VL>;
    using svid_type = simd::ty<VID,1>;
    using seid_type = simd::ty<EID,1>;

    using output_type = simd::container<typename MVExpr::data_type>;

    auto pvec1 = simd::template create_constant<svid_type>( VL*(VID)p );
    auto m_pid = expr::create_value_map_new<VL>(
	expr::create_entry<expr::vk_pid>( pvec1 ) );

    auto extractor = vid_type::traits::create_extractor(
	GP.getDegreeBits(), GP.getDegreeShift() );

    const simd::vec<vid_type,lo_unknown> vmask( extractor.get_mask() );

    // Split caches with pid cache hoisted out of loop
    // This is only valid when storedVL == VL.
    auto c = cache_create_no_init( all_caches, m_pid );
    cache_init( env, c, vop_caches, m_pid ); // partial init

    // Carried over from one phase to the next
    VID sstart = GP.getSlimVertex();
    auto sdst = simd::template create_constant<svid_type>( sstart );
    auto sstep = simd::template create_constant<svid_type>( (VID)VL );

    VID send = part.end_of(p);
    auto vpend = simd::template create_constant<vid_type>( send );

    EID sedge = part.edge_start_of( p );
    EID s = 0;
    const VID dmax = 1 << (GP.getMaxVL() * GP.getDegreeBits());
    while( s < nvec && sdst.at(0) < send ) {
	// Load cache for vdst
	auto m = expr::create_value_map_new<VL>(
	    expr::create_entry<expr::vk_dst>( sdst ),
	    expr::create_entry<expr::vk_mask>( vpend ),
	    expr::create_entry<expr::vk_pid>( pvec1 ) );
	cache_init( env, c, vcaches, m ); // partial init

	using OTr = typename output_type::data_type;
	static_assert( simd::detail::is_mask_traits_v<OTr>,
		       "expect edgemap operation to return update mask" );
	auto output = output_type::false_mask();

	EID code;
	do {
	    // Extract degree. Increment by one to adjust encoding
	    auto edata = simd::template load_from<vid_type>( &edge[s] );
	    code = extractor.extract_degree( edata.data() );
#if GRAPTOR_STOP_BIT_HIGH
	    VID deg = code & (dmax/2-1);
	    if( deg == dmax/2-1 )
		deg = GRAPTOR_DEGREE_MULTIPLIER * deg;
	    else
		deg = GRAPTOR_DEGREE_MULTIPLIER * deg + 1;
#else
	    VID deg = code >> 1;
	    if( deg == dmax/2-1 )
		deg = GRAPTOR_DEGREE_MULTIPLIER * deg;
	    else
		deg = GRAPTOR_DEGREE_MULTIPLIER * deg + 1;
#endif

	    // End of run of SIMD groups
	    // EID smax = std::min( s + EID(deg) * VL, nvec );
	    EID smax = s + EID(deg) * VL;

	    // std::cerr << "SIMD group v0=" << vdst.at(0) << " s=" << s << " deg=" << deg << " nvec=" << nvec << " - new\n";

	    // Should do extract_source only first time, rest should
	    // be raw source data (no degree encoded)
	    auto vdecode = simd::template create_unknown<vid_type>(
		extractor.extract_source( edata.data() ) );

	    while( s < smax ) {
		// Check all lanes are active; using cached values.
		if( !env.evaluate_bool( c, m, aexpr ) ) {
		    s = smax;
		    break;
		}

		_mm_prefetch( &edge[s + 256], _MM_HINT_NTA );

		// apply op vsrc, vdst;
		auto sv = simd::create_scalar<seid_type>( sedge+s );
		auto m = expr::create_value_map_new<VL>(
		    expr::create_entry<expr::vk_pid>( pvec1 ),
		    expr::create_entry<expr::vk_mask>( vmask ),
		    expr::create_entry<expr::vk_src>( vdecode ),
		    expr::create_entry<expr::vk_dst>( sdst ),
		    expr::create_entry<expr::vk_edge>( sv ) );
		// It is necessary to calculate the masks separately
		// since the change to mask_pack and binop_setmask
		// instead of binop_mask. The mask is no longer captured
		// inside the refop, hence the cache has no knowledge of
		// the mask. As such, masks are lost to cache_init/commit,
		// and the main expression evaluation is oblivious to it
		// as the memory references are likely captured in the cache.
		auto mpack = expr::sb::create_mask_pack( vdecode != vmask );
		cache_init( env, c, vcaches_use, m, mpack ); // partial init of uses (src)
		auto rval_output = env.evaluate( c, m, mpack, m_vexpr );
		output.lor_assign( rval_output.value().template convert_data_type<OTr>() );

		// Proceed to next vector of sources
		s += VL;

		vdecode = simd::template load_from<vid_type>( &edge[s] );
	    }
	} while(
#if GRAPTOR_STOP_BIT_HIGH
	    code == (dmax/2-1)
#else
	    code == dmax-2
#endif
	    && s < nvec );

	// std::cerr << "SIMD group END s=" << s << " code="
	// << code << " vdst.at(0)=" << vdst.at(0) << "\n";

	// for( unsigned short l=0; l < VL; ++l ) {
	    // if( output.at(l) )
		// std::cerr << VID(sdst.at(0)+l) << "\n";
	// }

	// auto vdst = simd::create_set1inc<vid_type,true>( sdst.data() );
	// auto mpack = expr::sb::create_mask_pack( vdst < vpend );

	// Evaluate hoisted part of expression.
	{
	    using logicalVID = typename add_logical<VID>::type;
	    // auto input = output.template asvector<logicalVID>();
	    auto input = output;

	    auto m = expr::create_value_map_new<VL>(
		expr::create_entry<expr::vk_smk>( input ),
		expr::create_entry<expr::vk_mask>( vpend ),
		expr::create_entry<expr::vk_dst>( sdst ),
		expr::create_entry<expr::vk_pid>( pvec1 ) );
	    env.evaluate( c, m, /*mpack,*/ m_rexpr );
	}

	cache_commit( env, vcaches, c, m /*, mpack*/ );
	// vdst.set_unsafe( vdst + vstep );
	sdst += sstep;
    }

    // From now on, only zero-degree vertices. This only executes the
    // vertexop() part and should be a no-op for frontier calculation.
    // We might benefit from creating a variation of m_rexpr that does not
    // include frontier calculation, i.e., constant-propagate a false
    // vk_smk.
    while( sdst.at(0) < send ) {
	// Load cache for vdst
	auto m = expr::create_value_map_new<VL>(
	    expr::create_entry<expr::vk_dst>( sdst ),
	    expr::create_entry<expr::vk_mask>( vpend ),
	    expr::create_entry<expr::vk_pid>( pvec1 ) );
	cache_init( env, c, vcaches, m ); // partial init

	// Determine valid lanes
	// auto vdst = simd::create_set1inc<vid_type,true>( sdst.data() );
	// auto mpack = expr::sb::create_mask_pack( vdst < vpend );

	// Evaluate hoisted part of expression.
	{
	    using logicalVID = typename add_logical<VID>::type;
	    auto input = output_type::false_mask();

	    auto m = expr::create_value_map_new<VL>(
		expr::create_entry<expr::vk_smk>( input ),
		expr::create_entry<expr::vk_mask>( vpend ),
		expr::create_entry<expr::vk_dst>( sdst ),
		expr::create_entry<expr::vk_pid>( pvec1 ) );
	    env.evaluate( c, m, /*mpack,*/ m_rexpr );
	}

	cache_commit( env, vcaches, c, m /*, mpack*/ );
	// vdst.set_unsafe( vdst + vstep );
	sdst += sstep;
    }

    // std::cerr << "end vdest_ngh: vdst[0]=" << vdst.at(0)
    // << " s=" << s << " nvec=" << nvec << "\n";

    /* Need to call m_rexpr also for zero-degree vertices?
     * although would be no-op as there are no active in-edges.
     * However, there may be algorithms where some action needs to be
     * performed (the api::record part admits any side effects)
    if( vdst.at(0) != part.end_of(p) )
	std::cerr << p << ": final vertex processed: " << vdst.at(0)
		  << " final vertex in partitioned: " << part.end_of(p) << "\n";
    */

    assert( s == nvec );

    cache_commit( env, vop_caches, c, m_pid );
}

template<typename EMapConfig, typename Operator>
__attribute__((flatten))
static inline void emap_pull(
    const GraphVEBOGraptor<gm_csc_datapar_cached> & GA,
    Operator & op, const partitioner & part ) {
    constexpr unsigned short VL = EMapConfig::VL;
    unsigned short storedVL = GA.getMaxVL();

    // Restrictions:
    // Meta-information is recoved only from the vector that is processed
    assert( VL == storedVL && "restriction" );

    // Variables (actually parameters because they are constant within the AST)
    auto v_src = expr::value<simd::ty<VID,VL>,expr::vk_src>();
    auto s_dst = expr::value<simd::ty<VID,1>,expr::vk_dst>();
    auto v_dst = expr::make_unop_incseq<VL>( s_dst );
    auto v_edge = expr::template make_unop_incseq<VL>(
	expr::value<simd::ty<EID,1>,expr::vk_edge>() );
    // disables padding vertices
    auto v_adst_cond = v_dst < expr::value<simd::ty<VID,VL>,expr::vk_mask>();
    auto v_pid0 = expr::make_unop_incseq<VL>(
	expr::value<simd::ty<VID,1>,expr::vk_pid>() );
    auto v_one = expr::value<simd::ty<VID,VL>,expr::vk_mask>();

    // All accesses made using a mask.
    auto m_vexpr0a
	= op.relax( add_mask( v_src, make_cmpne( v_src, v_one ) ), v_dst,
		    v_edge );
    // Vertexop - no mask
    auto vop0 = op.vertexop( v_dst );

    // Extract accumulators, also on edges. Generate expressions to reduce
    // accumulators
    auto accum = expr::extract_accumulators( make_seq( vop0, m_vexpr0a ) );
    auto pvop0 = expr::accumulate_privatized_accumulators( v_pid0, accum );
    auto pvopf0 = expr::final_accumulate_privatized_accumulators( v_pid0, accum );

    // Rewrite vertexop expression
    auto vop1 = expr::rewrite_privatize_accumulators(
	vop0, part, accum, v_pid0 );
    auto vop_caches = expr::extract_cacheable_refs<expr::vk_pid>( vop1 );
    auto vop2 = expr::rewrite_caches<expr::vk_pid>( vop1, vop_caches );

    // Rewrite edge expression
    auto m_vexpr0 = expr::rewrite_privatize_accumulators(
	m_vexpr0a, part, accum, v_pid0 );

    // Determine cached values
    auto vcaches_dst
	= expr::extract_cacheable_refs<expr::vk_dst>( m_vexpr0, vop_caches );
    auto vcaches_use = expr::extract_uses<expr::vk_src>(
	m_vexpr0, cache_cat( vop_caches, vcaches_dst ) );
    auto vcaches_let
	= expr::extract_local_vars(
	    expr::make_seq( m_vexpr0, vop2 ),
	    cache_cat( vop_caches, cache_cat( vcaches_dst, vcaches_use ) ) );
    auto vcaches = cache_cat( vcaches_dst, vcaches_let );

    auto all_caches
	= cache_cat( vop_caches, cache_cat( vcaches, vcaches_use ) );

    // TODO maybe need to include aexpr0
    auto env = expr::eval::create_execution_environment_with(
	op.get_ptrset(), all_caches,
	/*aexpr,*/ m_vexpr0, /*m_rexpr,*/ pvop0, pvopf0 );

    // Loop termination condition (active check)
    auto aexpr0 = op.active( v_dst ); // mask
    auto aexpr1 = rewrite_internal( aexpr0 );
    auto aexpr2 = expr::rewrite_caches<expr::vk_dst>( aexpr1, vcaches_dst );
    auto aexpr3 = expr::set_mask( v_adst_cond, aexpr2 );
    auto aexpr = expr::rewrite_mask_main( aexpr3 );

    // LICM: only inner-most redop needs to be performed inside loop.
    //       the other relates to the new frontier and can be carried over
    //       through registers as opposed to variables.
    auto m_vexpr0i = rewrite_internal( m_vexpr0 );
    auto m_licm = expr::licm_split_main( m_vexpr0i );
    auto m_vexpr1 = m_licm.le();
    auto m_rexpr0 = m_licm.pe();
    auto m_rexpr1 = append_pe( m_rexpr0, vop2 );

    // Loop part
    auto m_vexpr2 = expr::rewrite_caches<expr::vk_dst>( m_vexpr1, vcaches_dst );
    auto m_vexpr3 = expr::rewrite_caches<expr::vk_src>( m_vexpr2, vcaches_use );
    auto m_vexpr4 = expr::rewrite_caches<expr::vk_zero>( m_vexpr3, vcaches_let );
    auto m_vexpr = expr::rewrite_mask_main( m_vexpr4 );

    // fail_expose<std::is_class>( vcaches_dst );


    // Post-loop part
    auto m_rexpr2 = expr::rewrite_caches<expr::vk_dst>( m_rexpr1, vcaches_dst );
    auto m_rexpr3 = expr::rewrite_caches<expr::vk_src>( m_rexpr2, vcaches_use );
    auto m_rexpr4 = expr::rewrite_caches<expr::vk_zero>( m_rexpr3, vcaches_let );
    auto m_rexpr5 = expr::set_mask( v_adst_cond, m_rexpr4 );
    auto m_rexpr = expr::rewrite_mask_main( m_rexpr5 );

    using Cfg = std::decay_t<decltype(op.get_config())>;

    static_assert( Cfg::max_vector_length() >= VL,
		   "Cannot respect config option of maximum vector length" );

    map_partition<Cfg::is_parallel()>( part, [&]( int p ) {
	    constexpr bool ID = expr::is_idempotent<decltype(m_vexpr)>::value;
	    GraptorCSCDataParCached<ID,VL>(
		GA, p, GA.getCSC( p ), part,
		aexpr, // rewrite_internal( aexpr ),
		m_vexpr, // rewrite_internal( m_vexpr ),
		m_rexpr, // rewrite_internal( m_rexpr ),
		vop_caches, vcaches, vcaches_use, all_caches, env,
		op.get_config() );
	} );

    // Scan across partitions
    if constexpr ( !expr::is_noop<decltype(pvop0)>::value
		   || !expr::is_noop<decltype(pvopf0)>::value )
	emap_scan<VL,VID>( env, part, pvop0, pvopf0 );

    accum_destroy( accum );
}

#endif // GRAPTOR_DSL_EMAP_GRAPTORCSCDATAPARCACHED_H
