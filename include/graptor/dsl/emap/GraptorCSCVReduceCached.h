// -*- c++ -*-
#ifndef GRAPTOR_DSL_EMAP_GRAPTORCSCVREDUCECACHED_H
#define GRAPTOR_DSL_EMAP_GRAPTORCSCVREDUCECACHED_H

#include "graptor/dsl/emap/emap_scan.h"

template<bool Idempotent>
struct cache_ops;

template<>
struct cache_ops<true> {
    template<typename Cache, typename VCache,
	     typename ValueMap, typename ValueMap1>
    // TODO: for low-degree vertices, it is probably better to not cache the
    //       values
    static void
    construct( const VCache & vcaches, Cache & c,
	       const ValueMap & m, const ValueMap1 & m1 ) {
	cache_init( c, vcaches, m ); // initialise with stored value
    }
    template<typename Cache, typename VCache,
	     typename ValueMap, typename ValueMap1>
    static void
    destruct( const VCache & vcaches, Cache & c,
	      const ValueMap & m, const ValueMap1 & m1 ) {
	cache_reduce_and_commit( vcaches, c, m1 ); // reduce and write back to memory
    }
};

template<>
struct cache_ops<false> {
    template<typename Cache, typename VCache,
	     typename ValueMap, typename ValueMap1>
    static void
     construct( const VCache & vcaches, Cache & c,
		const ValueMap & m, const ValueMap1 & m1 ) {
	cache_clear( c, vcaches, m1 ); // set to zero of enclosing redop
    }
    template<typename Cache, typename VCache,
	     typename ValueMap, typename ValueMap1>
    static void
     destruct( const VCache & vcaches, Cache & c,
	       const ValueMap & m, const ValueMap1 & m1 ) {
	cache_reduce( vcaches, c, m1 ); // reduce with value stored in memory
    }
};

// This function implements a Grazelle-like vectorization although extended
// to short-circuit inactive vertices. It also uses a double-nested loop
// to enable caching of intermediates in memory and limiting reduction of
// updates to once per destination vertex.
template<bool Idempotent,
	 unsigned short VL, typename GraphType, typename GraphPartitionType,
	 typename Extractor,
	 typename AExpr, typename VExpr, typename MVExpr,
	 typename MRExprVOP, typename MRExprV,
	 typename VOPCache, typename VCache, typename Cache>
__attribute__((always_inline, flatten))
static inline void GraptorCSCVReduceCached(
    const GraphType & GA,
    const GraphPartitionType & GP,
    int p,
    const partitioner & part,
    const Extractor extractor,
    const VID * edge,
    const EID nvec,
    const VID vslim,
    simd_vector<VID,VL> pvec,
    const AExpr & aexpr,
    const VExpr & vexpr,
    const MVExpr & m_vexpr,
    const MRExprV & m_rexpr_v,
    const MRExprVOP & m_rexpr_vop,
    const VOPCache & vop_caches,
    const VCache & vcaches,
    Cache & c ) {

    // std::cerr << "\n*** GraptorCSCVReduceCached p=" << p << " ***\n";

    // TODO: instead of jumping by at least 1, we know that in this
    //       encoding jumps are of size at least maxVL. This would allow
    //       slightly larger values to be encoded.
    // const simd_vector<VID, VL> vone = simd_vector<VID, VL>::one_val();
    const simd_vector<VID, VL> vone( (VID)VL );
    const simd_vector<VID, VL> vmask( extractor.get_mask() );

    const EID dmax = 1 << (GP.getMaxVL() * GP.getDegreeBits());

    simd_vector<VID, 1> sdst( (VID)part.start_of(p) );
    simd_vector<VID, VL> vdst( (VID)part.start_of(p) );
    simd_vector<VID, 1> sstep( (VID)1 );
    simd_vector<VID, VL> vstep( (VID)1 );
    simd_vector<VID, 1> spid( (VID)p );

    EID s=0;
    while( sdst.at(0) < vslim && s < nvec ) {
	// std::cerr << "= vertex s=" << s << " sdst=" << sdst.at(0) << "\n";
	
	// Load cache for vdst.
	// The current implementation has not been tested for the treatment
	// of values loaded for pvec (intermediates for scans in the vertexop).
	// As such, pvec is temporarily removed from the value map in order
	// to generate an error in case scans on pvec are performed.
	auto m = expr::create_value_map_new<VL>(
	    expr::create_entry<expr::vk_dst>( sdst ) );
	auto m1 = expr::create_value_map_new<1>(
	    expr::create_entry<expr::vk_dst>( sdst ),
	    expr::create_entry<expr::vk_pid>( spid ) );

	// It depends on idempotency and the need for knowing the frontier
	// how we handle the cache in this case
	// + Idempotent, frontier = difference in update (min, max, or, ...)
        //   then copy the value to the cache, and derive the frontier from
	//   the cached value.
	// + Non-itempotent: frontier = destination was visited (has in-edges)
	//   then copy do cache_clear.
	// Both cases require commit.
	cache_ops<Idempotent>::construct( vcaches, c, m, m1 );

	// typename simd_vector<typename MVExpr::type, MVExpr::VL>::simd_mask_type output
	// = simd_vector<typename MVExpr::type, MVExpr::VL>::false_mask();
	using output_type = simd::container<
	    simd::ty<typename MVExpr::type, MVExpr::VL>>;
	auto output = output_type::false_mask();

	EID code;
	VID diff;
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
	    simd_vector<VID, VL> vdecode(
		extractor.extract_source( edata.data() ) );

	    EID smax = std::min( s + deg * VL, nvec );

	    // std::cerr << "= run s=" << s << " smax=" << smax << " deg=" << deg << "\n";

	    while( true ) {
		simd_vector<VID, VL> vsrc = vdecode;

		// std::cerr << "      s=" << s << " vsrc[0]=" << vsrc.at(0) <<" vdst[0]=" << vdst.at(0) << "\n";

		// assert( vdst.at(0) < GA.numVertices() );
		// Check all lanes are active.
		// This is a vector expression, followed by logical reduction.
		// However, as there is only one destination, if any lane
		// indicates convergence, then all lanes should terminate.
		// The reduction should be performed using logical AND, while
		// in the case of SELL where destinations are different,
		// reduction should use logical OR.
		if( unlikely( !expr::evaluate_bool_any( c, m, aexpr ) ) ) {
		    s = smax;
		    break;
		}

		// Proceed to next vector of sources
		s += VL;

		_mm_prefetch( &edge[s + 256], _MM_HINT_NTA );

		// validate( GA.getCSR(), vsrc, vdst );

		if( __builtin_expect( s < smax, 1 ) ) { // No mask required on validity of vsrc
		    // assuming it suffices that only highest bit in vdecode
		    // needs to be set to disable lane
		    // apply op vsrc, vdst;
		    auto m = expr::create_value_map_new<VL>(
			expr::create_entry<expr::vk_pid>( simd::vector<VID,1>( p ) ),
			expr::create_entry<expr::vk_dst>( sdst ),
			expr::create_entry<expr::vk_src>( vsrc ) );
		    auto rval_output = expr::evaluate( c, m, vexpr );
		    output.lor_assign( rval_output.mask() );

		    vdecode.load( &edge[s] );
		} else { // Final iteration - mask
		    // apply op vsrc, vdst;
		    auto m = expr::create_value_map_new<VL>(
			expr::create_entry<expr::vk_pid>( simd::vector<VID,1>( p ) ),
			expr::create_entry<expr::vk_mask>( vmask ),
			expr::create_entry<expr::vk_dst>( sdst ),
			expr::create_entry<expr::vk_src>( vsrc ) );
		    auto rval_output = expr::evaluate( c, m, m_vexpr );
		    output.lor_assign( rval_output.mask() );

		    break;
		}
	    }
	    // std::cerr << "check: code=" << code << " dmax-1=" << (dmax-1) << "\n";
	} while( code == dmax-1 );

	// std::cerr << "SIMD group END s=" << s << " code="
	// << code << " " << " vdst.at(0)=" << vdst.at(0) << "\n";

	// Evaluate hoisted part of expression.
	{
	    using logicalVID = typename add_logical<VID>::type;
	    auto input = output.template asvector<logicalVID>();

	    // This will include updating the cached version of
	    // the new frontier
	    auto m = expr::create_value_map_new<VL>(
		expr::create_entry<expr::vk_smk>( input ),
		expr::create_entry<expr::vk_dst>( sdst ) );
	    expr::evaluate( c, m, m_rexpr_v );

	    // Only requires sdst. This will include the count of active
	    // vertices and edges in the new frontier.
	    auto m_rexpr_vop_s = expr::rewrite_scalarize( m_rexpr_vop, vcaches );
	    expr::evaluate( c, m1, m_rexpr_vop_s );
	}

	// Cache_reduce should store only lane 0 which is the reduced
	// version of the elements dependent on vk_dst, which in this
	// case is all equal. We do need to do a reduction to scalar 
	// of the vector stored in cache.
	cache_ops<Idempotent>::destruct( vcaches, c, m, m1 );
	sdst += sstep;
	vdst += vstep;
    }

    // std::cerr << "end of vreduce_ngh: vdst[0]=" << vdst.at(0)
    // << " s=" << s
    // << " nvec=" << nvec << "\n";

    assert( sdst.at(0) == vslim );
    // std::cerr << "vslim=" << vslim << " sdst=" << sdst.at(0) << " s=" << s << " nvec=" << nvec << "\n";
    assert( s == nvec );
}

template<bool Idempotent, unsigned short VL, graptor_mode_t M,
	 typename AExpr, typename VExpr, typename MVExpr,
	 typename MRExprV, typename MRExprVOP,
	 typename VOPCache, typename VCache>
__attribute__((always_inline))
static inline void GraptorCSCVReduceCachedDriver(
    const GraphVEBOGraptor<M> & GA,
    int p,
    const GraphCSxSIMDDegreeMixed & GP,
    const partitioner & part,
    const AExpr & aexpr,
    const VExpr & vexpr,
    const MVExpr & m_vexpr,
    const MRExprV & m_rexpr_v,
    const MRExprVOP & m_rexpr_vop,
    const VOPCache & vop_caches,
    const VCache & vcaches ) {
    // Get list of edges
    const VID * edge = GP.getEdges();
    const EID nvec2 = GP.numSIMDEdgesDeg2();
    const EID nvec1 = GP.numSIMDEdgesDeg1();
    const EID nvec_d1 = GP.numSIMDEdgesDelta1();
    const EID nvec_dpar = GP.numSIMDEdgesDeltaPar();
    const EID nvec = GP.numSIMDEdges();

    simd_vector<VID, 1> pvec1( VL*(VID)p );
    simd_vector<VID, VL> pvec;
    pvec.set1inc( VL*VID(p) );
    auto m_pid = expr::create_value_map_new<VL>(
	expr::create_entry<expr::vk_pid>( pvec1 ) );

    auto extractor = simd_vector<VID,VL>::traits::create_extractor(
	GP.getDegreeBits(), GP.getDegreeShift() );

    // Split caches with pid cache hoisted out of loop
    // This is only valid when storedVL == VL.
    auto c = cache_create_no_init(
	cache_cat( vop_caches, vcaches ), m_pid );
    cache_init( c, vop_caches, m_pid ); // partial init

    // timer tm;
    // tm.start();

    GraptorCSCVReduceCached<Idempotent,VL>(
	GA, GP, p, part, extractor, edge, nvec_d1, GP.getSlimVertex(),
	pvec, aexpr, vexpr, m_vexpr, m_rexpr_v, m_rexpr_vop,
	vop_caches, vcaches, c );

    cache_commit( vop_caches, c, m_pid );
}

template<unsigned short VL, typename Operator>
__attribute__((always_inline))
static inline void csc_vector_with_cache_loop(
    const GraphVEBOGraptor<gm_csc_vreduce_cached> & GA,
    Operator & op, const partitioner & part ) {
    // Fused CSC + vertexop version where vertexop may involve a scan
    unsigned short storedVL = GA.getMaxVL();

    // std::cerr << "VEBOGraptor with VL=" << VL << " VEL=" << sizeof(VID)
    // << " vlen=" << (VL * sizeof(VID)) << " bytes\n";

    assert( VL == storedVL && "restriction" ); // TODO: for now...

    // Variables (actually parameters because they are constant within the AST)
    auto v_src = expr::value<simd::ty<VID,VL>,expr::vk_src>();
    auto v_sdst = expr::value<simd::ty<VID,1>,expr::vk_dst>();
    auto v_dst = expr::make_unop_broadcast<VL>( v_sdst );
    auto v_pid0 = expr::make_unop_incseq<VL>(
	expr::value<simd::ty<VID,1>,expr::vk_pid>() );
    auto v_one = expr::value<simd::ty<VID,VL>,expr::vk_mask>();

    // First version without mask
    auto vexpr0 = op.relax( v_src, v_dst );

    // Second version with mask.
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
    // auto rexpr1 = append_pe( rexpr0, vop2 );

    // It is assumed cache is the same / masked version may have additional
    // update bitmask
    auto vcaches
	= expr::extract_cacheable_refs<expr::vk_dst>( m_vexpr0, vop_caches );

    // Loop part
    auto vexpr2 = expr::rewrite_caches<expr::vk_dst>( vexpr1, vcaches );
    auto vexpr3 = vexpr2;
    auto vexpr = expr::rewrite_mask_main( vexpr3 );

    // Post-loop part
    // auto rexpr2 = expr::rewrite_caches<expr::vk_dst>( rexpr1, vcaches );
    // auto rexpr = expr::rewrite_mask_main( rexpr2 );

    // Loop termination condition (active check)
    auto aexpr0 = expr::make_unop_switch_to_vector( op.active( v_dst ) );
    auto aexpr1 = expr::make_unop_reduce( aexpr0, expr::redop_logicalor() );
    auto aexpr2 = expr::rewrite_caches<expr::vk_dst>( aexpr1, vcaches );
    auto aexpr = expr::rewrite_mask_main( aexpr2 );

    // LICM: only inner-most redop needs to be performed inside loop.
    //       the other relates to the new frontier and can be carried over
    //       through registers as opposed to variables.
    auto m_licm = expr::licm_split_main( m_vexpr0 );
    auto m_vexpr1 = m_licm.le();
    auto m_rexpr1_v = m_licm.pe(); // per vertex, vectorized
    auto m_rexpr1_vop = vop2; // per vertex, scalar

    // Loop part
    auto m_vexpr2 = expr::rewrite_caches<expr::vk_dst>( m_vexpr1, vcaches );
    auto m_vexpr = expr::rewrite_mask_main( m_vexpr2 );

    // Post-loop part
    // auto m_rexpr2 = expr::rewrite_caches<expr::vk_dst>( m_rexpr1, vcaches );
    // auto m_rexpr = expr::rewrite_mask_main( m_rexpr2 );
    auto m_rexpr2_v = expr::rewrite_caches<expr::vk_dst>( m_rexpr1_v, vcaches );
    auto m_rexpr_v = expr::rewrite_mask_main( m_rexpr2_v );
    auto m_rexpr2_vop
	= expr::rewrite_caches<expr::vk_dst>( m_rexpr1_vop, vcaches );
    auto m_rexpr3_vop = expr::rewrite_mask_main( m_rexpr2_vop );
    auto m_rexpr_vop = expr::rewrite_reduce( m_rexpr3_vop );

    // TODO:
    // * strength reduction on division by storedVL
    // * pass expressions by reference (appears some copying is done)

    map_partitionL( part, [&]( int p ) {
	    GraptorCSCVReduceCachedDriver<Operator::is_idempotent,VL>(
		GA, p, GA.getCSC( p ), part, aexpr, vexpr, m_vexpr,
		m_rexpr_v, m_rexpr_vop, vop_caches, vcaches );
	} );

    // Scan across partitions
    if( Operator::is_scan )
	emap_scan<VL,VID>( part, pvop0, pvopf0 );

    accum_destroy( accum );
}

#endif // GRAPTOR_DSL_EMAP_GRAPTORCSCVREDUCECACHED_H

