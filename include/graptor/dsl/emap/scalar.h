// -*- c++ -*-
#ifndef GRAPTOR_DSL_EMAP_SCALAR_H
#define GRAPTOR_DSL_EMAP_SCALAR_H

#include "graptor/dsl/eval/environment.h"
#include "graptor/dsl/emap/GraptorCSCVReduceCached.h"

template<unsigned short VL, typename Expr, typename CacheTy>
static void process_in_v_e( const VID *in, EID deg, VID dst, Expr e, CacheTy vcaches ) {
    EID j;

    // Note: create caches at vector length VL of source
    // auto m = expr::value_map_dst<VID, VL, 1>( simd_vector<VID, 1>( dst ) );
    auto m = expr::create_value_map<VL,1>(
	expr::create_entry<expr::vk_dst>( simd_vector<VID, 1>( dst ) ) );

    auto c = cache_create( vcaches, m ); // don't need full value_map as only dst is used

    // Potential optimisation:
    //   If some source vertices inactive: remove them from vector and bring
    //   in new values until fully active vector is obtained, or if not possible
    //   Then, continue with fully active vector (requires no mask - maximum
    //   efficiency on arithmetic; however expensive shuffles)
    
    for( j=0; j+VL-1 < deg; j += VL ) {
	// std::cerr << "in_v_e with j=" << j << " deg=" << deg << " VL=" << VL << "\n";
	simd_vector<VID, VL> src;
	src.loadu( &in[j] ); // linear read
	auto smk = simd_vector<VID,VL>::true_mask();
	auto dmk = simd_vector<VID,1>::true_mask();
	expr::value_map_full<VID,VL,VID,1> m( src, smk,
					      simd_vector<VID,1>(dst), dmk );
	auto r = expr::evaluate( c, m, e );
	    // expr::evaluator<decltype(m), decltype(c)>( c, m ).evaluate( e );
    }
    // TODO: consider: vector with mask if #iters < VL/2, else scalar?
    if( j < deg ) { // epilogue
	simd_vector<VID, VL> src;
	simd_mask<sizeof(VID), VL> smk;
	// Warning: VL==64 requires a 64-bit integer here
	smk.from_int( (1<<(deg-j))-1 );
	// auto src_mask = expr::add_mask( src, smk );
	src.loadu( &in[j] ); // linear read
	auto dmk = simd_vector<VID,1>::true_mask();
	expr::value_map_full<VID,VL,VID,1> m( src, smk,
					      simd_vector<VID,1>(dst), dmk );
	auto r = expr::evaluate( c, m, e );
	// expr::evaluator<decltype(m), decltype(c)>( c, m ).evaluate( e );
	// Related issue: how to put conditionals in source code?
	// what is reasonable user syntax for a maskrefop? A[s] -> A[s,m]: illegal syntax -> can overload operator,
    }

    // TODO: write-back reduction should take into account update mask!
    //       or can we prove this may be ommitted because this is a reduction
    //       and non-updated lanes hold the value that is improved on?
    //       Ok for min and logical_or. What if we have a sum reduction?
    //       Either create()/init() or commit() needs to take this into account.
    // vcaches.commit( c, m );
    cache_commit( vcaches, c, m );
}

template<unsigned short VL, typename Expr, typename AExpr, typename RExpr,
	 typename Cache>
static void
process_in_v_e_simd1( const VID *in, EID deg, VID scalar_dst,
		      Expr e, AExpr ea, RExpr er, Cache cache ) {
    EID j;

    simd_vector<VID,1> dst( scalar_dst );

    // Note: create caches at vector length VL of source
    // auto m = expr::value_map_dst<VID, VL, 1>( dst );
    auto m = expr::create_value_map2<VL,1,expr::vk_dst>( dst );
    auto c = cache_create( cache, m ); // don't need full value_map as only dst is used

    // Check all lanes are active.
    if( !expr::evaluate_bool( c, m, ea ) )
	return;

    typename simd_vector<typename Expr::type, VL>::simd_mask_type output
	= simd_vector<typename Expr::type, VL>::false_mask();

    for( j=0; j+VL-1 < deg; j += VL ) {
	simd_vector<VID, VL> src;
	src.loadu( &in[j] ); // linear read
	auto smk = simd_vector<VID,VL>::true_mask();
	auto dmk = simd_vector<VID,1>::true_mask();
	expr::value_map_full<VID,VL,VID,1> m( src, smk, dst, dmk );
	auto rval_output // rvalue space
	    = expr::evaluate( c, m, e );
	// = expr::evaluator<decltype(m), decltype(c)>( c, m ).evaluate( e );
	output |= rval_output.mask(); // simd_vector/simd_mask space

	// Check all lanes are active.
	if( !expr::evaluate_bool( c, m, ea ) )
	    break;
    }

    if( j < deg ) { // epilogue
	simd_vector<VID, VL> src;
	src.loadu( &in[j] ); // linear read
	simd_mask<sizeof(VID), VL> smk;
	// Warning: VL==64 requires a 64-bit integer here
	smk.from_int( (1<<(deg-j))-1 );
	auto dmk = simd_vector<VID,1>::true_mask();
	expr::value_map_full<VID,VL,VID,1> m( src, smk, dst, dmk );
	auto rval_output = expr::evaluate( c, m, e ); // rvalue space
	output |= rval_output.mask(); // simd_vector/simd_mask space

	// There is no need to perform the active check. We are done anyway.
    }

    // Code that was moved out of loop
    {
	auto dmk = simd_vector<VID,1>::true_mask();
	simd_vector<VID,VL> src; // unused!
	// TODO: value in licm is licm.cop() -> need to set/input value
	expr::value_map_full<VID,VL,VID,1> m( src, output, dst, dmk );
	expr::evaluate( c, m, er );
    }

    // TODO: write-back reduction should take into account update mask!
    //       or can we prove this may be ommitted because this is a reduction
    //       and non-updated lanes hold the value that is improved on?
    //       Ok for min and logical_or. What if we have a sum reduction?
    //       Either create()/init() or commit() needs to take this into account.
    // TODO: the store operation made by the cache should know that the target
    //       addresses are sequential (linear vector), and use store instead of
    //       scatter
    cache_commit( cache, c, m );
}

template<unsigned short VL, typename Expr, typename AExpr, typename RExpr,
	 typename Cache>
__attribute__((always_inline))
inline void process_in_v_e_simd(
    const VID *in, EID deg, VID scalar_dst, unsigned short storedVL,
    Expr e, AExpr ea, RExpr er, Cache cache ) {
    EID j;

    simd_vector<VID,VL> dst;
    dst.set1inc( scalar_dst );

    // Note: create caches at vector length VL of source
    auto m = expr::create_value_map2<VL,VL,expr::vk_dst>( dst );
    auto c = cache_create( cache, m ); // don't need full value_map as only dst is used

    typename simd_vector<typename Expr::type, VL>::simd_mask_type output
	= simd_vector<typename Expr::type, VL>::false_mask();

    // TODO: figure out a second degree from which source needs to be validated
    //       (i.e., shortest degree in vector group), such that we can avoid
    //       the checks for src != ~(VID)0 during most of the computation
    //       Note: not present in SlimSell?
    //       Could take the degree of vertex j+storedVL-1 as a boundary
    for( j=0; j < deg; j += storedVL ) {
	// Check all lanes are active.
	if( !expr::evaluate_bool( c, m, ea ) )
	    break;

	simd_vector<VID, VL> src;
	src.load( &in[j] ); // linear read
	// expr::value_map_nomask<VID,VL,VID,VL> m( src, dst );
	auto mp = expr::create_value_map_new<VL>(
	    expr::create_entry<expr::vk_src>( src ),
	    expr::create_entry<expr::vk_dst>( dst ) );
	auto rval_output = expr::evaluate( c, m, e ); // rvalue space
	output |= rval_output.mask(); // simd_vector/simd_mask space
    }

    {
	auto dmk = simd_vector<VID,VL>::true_mask();
	simd_vector<VID,VL> src;
	// TODO: value in licm is licm.cop() -> need to set/input value
	expr::value_map_full<VID,VL,VID,VL> m( src, output, dst, dmk );
	// expr::evaluator<decltype(m), decltype(c)>( c, m ).evaluate( er );
	expr::evaluate( c, m, er );
    }

    // TODO: write-back reduction should take into account update mask!
    //       or can we prove this may be ommitted because this is a reduction
    //       and non-updated lanes hold the value that is improved on?
    //       Ok for min and logical_or. What if we have a sum reduction?
    //       Either create()/init() or commit() needs to take this into account.
    // TODO: the store operation made by the cache should know that the target
    //       addresses are sequential (linear vector), and use store instead of
    //       scatter
    cache_commit( cache, c, m );
}


#if 0
template<typename UTr, typename VTr>
auto as_mask( rvalue<VTr,void> r ) {
    return r.value().asmask();
}

template<typename UTr, typename VTr, typename MTr>
auto as_mask( rvalue<VTr,MTr> r,
	      typename std::enable_if<VTr::VL == MTr::VL>::type * = nullptr ) {
    return r.value().asmask() & r.mask();
}

template<typename UTr, typename MTr>
auto as_mask( rvalue<void,MTr> r ) {
    return r.mask();
}

template<unsigned short VL, typename vertex, typename Operator>
__attribute__((always_inline))
static inline void DBG_NOINLINE csc_vector_with_cache_loop(
    const graph<vertex> & GA,
    Operator op, VID from, VID to ) {

    typedef typename graph<vertex>::vertex_type vertex_type;
    const pair<intT,vertex_type> *G = GA.CSCV;
    VID n = GA.CSCVn;

    auto vexpr0 = op.relax( add_mask( expr::value<simd::ty<VID,VL>,expr::vk_src>(),
				      expr::value<simd::ty<logical<sizeof(VID)>,VL>,expr::vk_smk>() ),
			    expr::value<simd::ty<VID,1>,expr::vk_dst>() );
    auto vcaches = expr::extract_cacheable_refs<expr::vk_dst>( vexpr0 );
    // First introduce caches, then match vector lengths and move masks
    auto vexpr1 = expr::rewrite_caches<expr::vk_dst>( vexpr0, vcaches );
    auto vexpr = expr::rewrite_mask_main( vexpr1 );

    auto mm = expr::value_map_full<VID, VL, VID, 1>(
	simd_vector<VID, VL>( (VID)-1 ),
	simd_vector<typename add_logical<VID>::type, VL>(
	    logical<sizeof(VID)>( ~VID(0) ) ),
	simd_vector<VID, 1>( (VID)-1 ),
	simd_vector<typename add_logical<VID>::type, 1>( ~(VID)0 ) );
    auto c = vcaches.create( mm );

    auto is_active = op.active( expr::value<simd::ty<VID,1>,expr::vk_dst>() );

    for( VID i=from; i < to; ++i ) {
	VID dst = G[i].first;
	tuple<> ac;
	// expr::value_map_dst<VID, VL, 1> am( (simd_vector<VID, 1>( dst )) );
	auto am = expr::create_value_map2<VL,1,expr::vk_dst>(
	    simd_vector<VID, 1>( dst ) );
	simd_vector<typename add_logical<VID>::type,1> mfalse = simd_vector<typename add_logical<VID>::type,1>( (VID)0 );
	simd_vector<typename add_logical<VID>::type,1> mexpr =
	    as_mask<simd::detail::mask_logical_traits<sizeof(VID),VL>(
		expr::evaluate( ac, am, is_active ) );
	// Some compiler error if we try to use operator != (mask,mask)
	// if( operator != ( mfalse, mexpr ) ) {
	if( mfalse.at(0) != mexpr.at(0) ) {
	// if( true /* op.active_destination( dst ) */ ) {
	    vertex_type V = G[i].second;
	    EID deg = V.getInDegree();
	    const VID *in = V.getInNeighborPtr();
	    process_in_v_e<VL>( in, deg, dst, vexpr, vcaches );
	}
    }
}
#endif


template<unsigned short VL, typename Operator>
__attribute__((always_inline))
static inline void DBG_NOINLINE csc_loop(
    const EID * idx, const VID * edge,
    Operator & op, const partitioner & part,
    typename std::enable_if<VL != 1>::type * = nullptr ) {

    static_assert( !Operator::is_scan,
		   "lost track; don't if this should or not" );

    auto vexpr0 = op.relax(
	add_mask( expr::value<simd::ty<VID,VL>,expr::vk_src>(),
		  expr::value<simd::ty<VID,VL>,expr::vk_smk>() ),
	expr::value<simd::ty<VID,1>,expr::vk_dst>() );

#if DISABLE_CACHING
    auto vcaches = expr::cache<>();
#else
    auto vcaches = expr::extract_cacheable_refs<expr::vk_dst>( vexpr0 );
#endif

    // LICM: only inner-most redop needs to be performed inside loop.
    //       the other relates to the new frontier and can be carried over
    //       through registers as opposed to variables.
    auto licm = expr::licm_split_main( vexpr0 );
    auto vexpr1 = licm.le();
    auto rexpr1 = licm.pe();

    // Loop part
    auto vexpr2 = expr::rewrite_caches<expr::vk_dst>( vexpr1, vcaches );
    auto vexpr3 = expr::rewrite_vectors_main( vexpr2 );
    auto vexpr = expr::rewrite_mask_main( vexpr3 );

    // Post-loop part
    auto rexpr2 = expr::rewrite_caches<expr::vk_dst>( rexpr1, vcaches );
    auto rexpr3 = expr::rewrite_vectors_main( rexpr2 );
    auto rexpr = expr::rewrite_mask_main( rexpr3 );

    // Loop termination condition (active check). VL!=1 because acting on cache
    auto aexpr0 = op.active( expr::value<simd::ty<VID,VL>,expr::vk_dst>() );
    auto aexpr1 = expr::make_unop_reduce( aexpr0, expr::redop_logicalor() );
    auto aexpr2 = expr::rewrite_caches<expr::vk_dst>( aexpr1, vcaches );
    auto aexpr3 = expr::rewrite_vectors_main( aexpr2 );
    auto aexpr = expr::rewrite_mask_main( aexpr3 );

    map_partitionL( part, [&]( int p ) {
	    VID s = part.start_of(p);
	    VID e = part.end_of(p);

	    // A single partition is processed here. Use a sequential loop.
	    for( VID i=s; i < e; i++ ) {
		process_in_v_e_simd1<VL>( &edge[idx[i]], idx[i+1]-idx[i], i,
					  vexpr, aexpr, rexpr, vcaches );
	    }
	} );
}

template<unsigned short VL, typename Operator>
__attribute__((always_inline))
static inline void DBG_NOINLINE csc_vector_with_cache_loop(
    const GraphCSxSlice & GA,
    Operator & op, const partitioner & part ) {
    // Important to restore:
    // from = std::max( from, GA.getLowVertex() );
    // to = std::min( to, GA.getHighVertex() );

    const EID * idx = GA.getIndex() - GA.getLowVertex(); // offset
    const VID * edge = GA.getEdges();

    csc_loop<VL>( idx, edge, op, part ); // was from, to instead of part
}

#endif // GRAPTOR_DSL_EMAP_SCALAR_H
