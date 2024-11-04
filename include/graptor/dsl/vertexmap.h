// -*- c++ -*-

#ifndef GRAPHGRIND_DSL_VERTEXMAP_H
#define GRAPHGRIND_DSL_VERTEXMAP_H

/***********************************************************************
 * \page page_vmap Higher-order methods on vertex sets
 *
 * Higher order methods over vertex sets apply operations over the
 * vertex set of a graph. Supported methods are map, which applies a method
 * for each vertex, and scan, which applies a method for each vertex and
 * implies the calculation of a scalar value, e.g., the value of a property
 * summed over all vertices.
 *
 * Vertex map and scan methods are applied on a #lazy_executor object,
 * see also \ref page_lazy.
 ***********************************************************************/

/***********************************************************************
 * \page page_lazy Lazy executors
 * Graptor stages higher-order methods on vertex or edge sets for lazy
 * execution. The purpose of lazy execution is to merge operations where
 * possible. For instance, merging two subsequent vertex map operations
 * into a single vertex map has the advantage that a larger amount of
 * code may be executed per fetched data element, provided the operations
 * share data streams.
 *
 * The lazy executors may also merge a vertex map operation into an edge
 * map. This may result in highly increased efficiency as the vertex map
 * operation can often be performed off the back of the edge map.
 *
 * Merging of operations is based on dependencies, which are affected
 * by:
 * \li data flow dependencies on vertex properties
 * \li reductions, i.e., edge map and vertex scan
 * \li frontiers, which introduce additional control flow when
 *     distinguishing sparse from dense frontiers
 *
 * \see lazy_executor
 ***********************************************************************/

#include <cstdlib>
#include <typeinfo>
#include <mutex>

#include "graptor/frontier/frontier.h"
#include "graptor/dsl/ast.h"
#include "graptor/dsl/ast_accum.h"
#include "graptor/dsl/frontier_wrap.h"
#include "graptor/dsl/eval/environment.h"
#include "graptor/api/fusion.h"
#include "graptor/vmap_timing.h"

// Simplifying assumptions:
// * postpone calculations until first reason to execute
// * reasons to execute are materialize() (only)
//   (because we want to be able to postpone the nActive[VE] calculation
//    on frontiers...)
// * lazy_executor destructor calls materialize()
// * every lazy_array will eventually be materialized (might consider in
//   some cases to avoid materialization but seems hard with templates)
// * it may be handy to have a lazy_array.read() call which automatically
//   materializes (if not, throw error)
// * there may be no need for a lazy_array type, may capture everything
//   with array_ro? Reason: they will be materialized, so need to allocate
//   memory anyway.
//   + consequence: we may not be able to tolerate WAR, WAW
//   + best to retain program order

// Including edgemap
// * old frontier -> COO/CSC/CSR depending on density
//   (read old frontier/materialise)
// 1. Dense and choose COO
//    edgemap is crisscross, thus not efficient to merge; fully materialise
//    before proceeding
// 2. Dense and choose CSC
//    edgemap is gathering vmap -> materialise and register edgemap as if it
//    were a vmap
// 3. Sparse (CSR)
//    sparse frontier, few accesses, clearly separate case; cannot be lazy
//    output as random vertices updated -> don't know when vertex fully
//    calculated.
//    fully materialise before proceeding
// Also: when edgemap starts, materialise everything (at least frontier)
// -> need to know frontier density

// Rewrite rules
// step_vmap<is_scan=true>: contains an accumulator: privatize the accumulator
// in each partition (cacheop) and follow on by a scan across accu's
// - recognize array_ro with index [0] (constant 0)
// - create accum array, and replace
// ... for now, put this in programming model

// TODO:
// - vertexmap should parallelize with balanced vertex counts, not balanced
//   edge counts

template<unsigned short VL, typename VID>
VID roundupVL( VID n ) {
    return n == 0 ? n : n - (n - 1) % VL + VL - 1;
}

enum vmap_kind {
    vmap_map,
    vmap_scan,
    vmap_filter
};

template<vmap_kind vk1, vmap_kind vk2>
struct vmap_kind_merge {
    static constexpr vmap_kind value = 
	vk1 != vmap_map || vk2 != vmap_map
	? vk1 == vmap_filter || vk2 == vmap_filter ? vmap_filter : vmap_scan
	: vmap_map;
};


/***********************************************************************
 * Combining edge and vertex operators.
 ***********************************************************************/
template<vmap_kind VKind, typename VOperator1, typename VOperator0>
struct AppendVOpToVOp
{
    using self_type = AppendVOpToVOp<VKind, VOperator1, VOperator0>;

    VOperator1 vop1;
    VOperator0 vop0;

    // static constexpr bool is_scan = IsScan;
    static constexpr vmap_kind vkind = VKind;
    static constexpr bool is_filter = VKind == vmap_filter;

    constexpr AppendVOpToVOp( VOperator1 _vop1, VOperator0 _vop0 )
	: vop1( _vop1 ), vop0( _vop0 ) { }

    template<typename VIDDst>
    auto operator ()( VIDDst d ) {
	return expr::make_seq( vop0( d ), vop1( d ) );
    }

    template<typename map_type0>
    struct ptrset {
	using map_type1 = typename VOperator0::template ptrset<map_type0>::map_type;
	using map_type = typename VOperator1::template ptrset<map_type1>::map_type;


	template<typename MapTy>
	static void initialize( MapTy & map, const self_type & op ) {
	    VOperator0::template ptrset<map_type0>::initialize( map, op.vop0 );
	    VOperator1::template ptrset<map_type1>::initialize( map, op.vop1 );
	}
    };
};

template<typename VOperator1, typename VOperator0>
constexpr auto append_vop2vop( VOperator1 vop1, VOperator0 vop0 ) {
    return AppendVOpToVOp<vmap_kind_merge<VOperator1::vkind,
					  VOperator0::vkind>::value,
			  VOperator1, VOperator0>( vop1, vop0 );
}

template<bool IsScan, typename VOperator, typename EOperator>
struct AppendVOpToEOp
{
    using self_type = AppendVOpToEOp<IsScan, VOperator, EOperator>;
    
    VOperator vop;
    EOperator eop;

    static constexpr frontier_mode new_frontier = EOperator::new_frontier;
    static constexpr bool is_scan = IsScan;
    static constexpr bool defines_frontier = EOperator::defines_frontier;
    static constexpr bool may_omit_frontier_rd =
	EOperator::may_omit_frontier_rd;
    static constexpr bool may_omit_frontier_wr =
	EOperator::may_omit_frontier_wr;

    constexpr AppendVOpToEOp( VOperator _vop, EOperator _eop )
	: vop( _vop ), eop( _eop ) { }

    template<typename VIDSrc, typename VIDDst, typename EIDEdge>
    auto relax( VIDSrc s, VIDDst d, EIDEdge e ) { return eop.relax( s, d, e ); }

    template<typename VIDDst>
    auto update( VIDDst d ) { return eop.update( d ); }

    template<typename VIDSrc, typename VIDDst, typename EIDEdge>
    auto fusionop( VIDSrc s, VIDDst d, EIDEdge e ) {
	return eop.fusionop( s, d, e );
    }

    constexpr api::fusion_flags get_fusion_flags() const {
	return eop.get_fusion_flags();
    }

/*
    template<typename VIDDst>
    auto different( VIDDst d ) { return eop.different( d ); }
*/

    template<typename VIDDst>
    auto enabled( VIDDst d ) { return eop.enabled( d ); }

    template<typename VIDDst>
    auto active( VIDDst d ) { return eop.active( d ); }

    template<typename VIDDst>
    auto any_activated( VIDDst d ) { return eop.any_activated( d ); }

    template<typename VIDDst>
    auto vertexop( VIDDst d ) {
	return expr::make_seq( eop.vertexop( d ), vop( d ) );
    }

    template<typename PSet>
    auto get_ptrset( const PSet & pset ) const {
	auto v = expr::value<simd::ty<VID,1>,expr::vk_vid>();
	return map_merge(
	    eop.get_ptrset( pset ),
	    expr::extract_pointer_set( vop( v ) ) );
    }

    auto get_ptrset() const { // TODO - redundant
	return get_ptrset( expr::create_map() );
    }

    template<typename map_type0>
    struct ptrset {
	using map_type1 = typename EOperator::template ptrset<map_type0>::map_type;
	static_assert( expr::map_contains_v<(unsigned)expr::aid_key(expr::array_aid(expr::aid_eweight)),map_type1>, "check 0" );
	using map_type = typename VOperator::template ptrset<map_type1>::map_type;
	static_assert( expr::map_contains_v<(unsigned)expr::aid_key(expr::array_aid(expr::aid_eweight)),map_type>, "check 1" );


	template<typename MapTy>
	static void initialize( MapTy & map, const self_type & op ) {
	    EOperator::template ptrset<map_type0>::initialize( map, op.eop );
	    VOperator::template ptrset<map_type1>::initialize( map, op.vop );
	}
    };


    bool is_true_src_frontier() const { return eop.is_true_src_frontier(); }
    bool is_true_dst_frontier() const { return eop.is_true_dst_frontier(); }

    template<graph_traversal_kind gtk, bool FStrue, bool FDtrue, bool IsPriv,
	     typename cfg>
    auto variant( const VID * degree ) {
	// rebind vop to annotated eop
	// xAW analysis is not affected as eop.relax() only depends on relax
	// method, not the frontier read/write
	auto var = eop.template variant<gtk,FStrue,FDtrue,IsPriv,cfg>( degree );
	return AppendVOpToEOp<is_scan,VOperator,decltype(var)>( vop, var );
    }

    auto get_config() const { return eop.get_config(); }

    frontier & get_frontier() { return eop.get_frontier(); }
};


/***********************************************************************
 * Definitions of map steps. Differentiates by type of frontier used.
 ***********************************************************************/
struct step_empty {
    static const bool has_frontier = false;
    void execute( const partitioner & ) { }
    void execute( const frontier &, const partitioner & ) { }
};

// This Step encodes dense vmap and vscan operations
template<bool IsScan,typename OperatorTy>
struct step_vmap_dense;

template<typename OperatorTy>
struct step_vmap_dense<false,OperatorTy> { // vmap
    using Operator = OperatorTy;
    using self_type = step_vmap_dense<false,Operator>;
    using vexpr_type = decltype( ((Operator*)nullptr)->operator()( expr::value<simd::ty<VID,1>,expr::vk_vid>() ) );

    static constexpr vmap_kind vkind = vmap_map;
    static const bool has_frontier = false;
    
    constexpr step_vmap_dense( Operator _op ) : op( _op ) { }

    auto get_operator() const { return op; }

    void execute( const frontier & cfrontier, const partitioner & part ) {
	if( cfrontier.isEmpty() )
	    return;

	// TODO: if cfrontier is sparse, then convert to sparse.
	
	switch( cfrontier.getType() ) {
	case frontier_type::ft_true:
	    execute( part );
	    break;
	case frontier_type::ft_unbacked:
	    assert( 0 && "ERROR! CANNOT READ UNBACKED FRONTIER" );
	    break;
	case frontier_type::ft_sparse:
	    assert( 0 && "ERROR! SHOULD NOT GET HERE" );
	    break;
	case frontier_type::ft_bool:
	    execute<frontier_type::ft_bool>( cfrontier, part, op );
	    break;
	case frontier_type::ft_bit:
	    execute<frontier_type::ft_bit>( cfrontier, part, op );
	    break;
	case frontier_type::ft_bit2:
	    execute<frontier_type::ft_bit2>( cfrontier, part, op );
	    break;
	case frontier_type::ft_logical1:
	    execute<frontier_type::ft_logical1>( cfrontier, part, op );
	    break;
	case frontier_type::ft_logical2:
	    execute<frontier_type::ft_logical2>( cfrontier, part, op );
	    break;
	case frontier_type::ft_logical4:
	    execute<frontier_type::ft_logical4>( cfrontier, part, op );
	    break;
	case frontier_type::ft_logical8:
	    execute<frontier_type::ft_logical8>( cfrontier, part, op );
	    break;
	default:
	    UNREACHABLE_CASE_STATEMENT;
	}
    }
    void execute( bool * d_bool, const partitioner & part ) {
	using frontier_spec = expr::determine_frontier_vmap<VID,Operator>;
	constexpr unsigned short VL = frontier_spec::VL;
	wrap_frontier_read_m<Operator,frontier_type::ft_bool,VL> d_op( d_bool, op );
	execute<VL>( part, d_op );
    }
    void execute( const partitioner & part ) {
	using frontier_spec = expr::determine_frontier_vmap<VID,Operator>;
	constexpr unsigned short VL = frontier_spec::VL;
	execute<VL>( part, op );
    }
private:
    template<frontier_type ftype>
    void execute( const frontier & cfrontier, const partitioner & part,
		  Operator op ) {
	using frontier_spec = expr::determine_frontier_vmap<VID,Operator>;
	constexpr unsigned short VL = frontier_spec::VL;

	wrap_frontier_read_m<Operator,ftype,VL>
	    f_op( cfrontier.getDense<ftype>(), op );

	execute<VL>( part, f_op );
    }

    // Note: there may be an opportunity when caching a frontier to pre-
    //       calculate the conversion of the frontier to a more suitable type,
    //       as opposed to re-converting everytime it is loaded from cache.
    //       Will only make a difference if the frontier is re-loaded/used
    //       multiple times.
    template<unsigned short VL, typename FrOperatorTy>
    void execute( const partitioner & part, FrOperatorTy op ) {
	vmap_report<Operator,VL>( std::cout, "vmap_dense/map:" );
#if VMAP_TIMING
	timer tm;
	tm.start();
#endif

	// Map
	map_partitionL( part, [&]( int p ) {

	auto vid1 = expr::value<simd::ty<VID,1>,expr::vk_vid>();
	auto vid = expr::make_unop_incseq<VL>( vid1 );
	auto spid = expr::value<simd::ty<VID,1>,expr::vk_pid>();
	auto pid = expr::make_unop_incseq<VL>( spid );

	auto expr0 = op( vid );
	auto cache_pid = expr::extract_cacheable_refs<expr::vk_pid>( expr0 );
	auto expr1 = expr::rewrite_caches<expr::vk_pid>( expr0, cache_pid );
	auto cache_use = expr::extract_uses<expr::vk_vid>( expr1, cache_pid );
	auto expr2 = expr::rewrite_caches<expr::vk_vid,expr::mam_nontemporal>( expr1, cache_use );
	auto cache_let = expr::extract_local_vars( expr2, cache_use );
	auto expr3 = expr::rewrite_caches<expr::vk_zero>( expr2, cache_let );
	auto expr4 = expr::rewrite_vectors_main( expr3 );
	auto expr5 = expr::rewrite_mask_main( expr4 );
	// auto expr = expr::rewrite_incseq( expr4 ); // TODO -- errors in VL match
	auto expr = expr5;

	auto sexpr0 = op( vid1 );
	auto cache_pid1 = expr::extract_cacheable_refs<expr::vk_pid>( sexpr0 );
	auto sexpr1 = expr::rewrite_caches<expr::vk_pid>( sexpr0, cache_pid1 );
	auto cache_use1 = expr::extract_uses<expr::vk_vid>( sexpr1 );
	auto sexpr2 = expr::rewrite_caches<expr::vk_vid,expr::mam_nontemporal>( sexpr1, cache_use1 );
	auto cache_let1 = expr::extract_local_vars( sexpr2, cache_use1 );
	auto sexpr3 = expr::rewrite_caches<expr::vk_zero>( sexpr2, cache_let1 );
	auto sexpr4 = expr::rewrite_vectors_main( sexpr3 );
	auto sexpr = expr::rewrite_mask_main( sexpr4 );

	// Build environment
	auto cache_vid = cache_cat( cache_pid, cache_use, cache_let );
	auto cache_vid1 = cache_cat( cache_pid1, cache_use1, cache_let1 );
	auto cache_all = cache_cat( cache_vid, cache_vid1 );
/*
	auto env = expr::eval::create_execution_environment(
	    // cache_cat( cache_pid, cache_pid1, cache_vid, cache_vid1 ),
	    cache_all,
	    sexpr, expr ); 
*/
	auto env = expr::eval::create_execution_environment_op( op, cache_all );

	// fail_expose<std::is_class>( cache_vid1 );

		VID s = part.start_of_vbal( p );
		VID e = part.end_of_vbal( p );

		// There is a potential issue here as the end of one partition
		// will partially update a vector while the start of the next
		// will do the same. To avoid races, we should only allow
		// partitions with a number of vertices that is a multiple of
		// VL. Only the final partition can have a number of vertices
		// that is not a multiple of VL and every partition starts
		// at a multiple of VL.
		// The code here fixes this problem by using scalar operations
		// at the start and end of a partition.
		// assert( (s % VL) == 0 && "partition should be vector-aligned" );

		VID v=s;
		VID vstart = std::min( e, roundupVL<VL>( s ) );
		for( ; v < vstart; ++v ) {
		    auto vv = simd::template create_constant<simd::ty<VID,1>>( v );
		    auto pp = simd::template create_constant<simd::ty<VID,1>>( VL*VID(p) );
		    auto m = expr::create_value_map_new2<
			1,expr::vk_vid,expr::vk_pid>( vv, pp );
		    auto c = cache_create( env, cache_vid1, m );
		    auto r = env.evaluate( c, m, sexpr );
		    cache_commit( env, cache_vid1, c, m, r.mpack() );
		}
		if( v < e ) {
		    assert( (v % VL) == 0 && "vector-alignment failed" );

		    auto pp = simd::template create_constant<simd::ty<VID,1>>( VL*(VID)p );
		    // auto pp = simd::template create_set1inc<simd::ty<VID,VL>,true>(
		    // VL*VID(p) );

		    auto m_pid = expr::create_value_map_new2<VL,expr::vk_pid>( pp );
		    auto c = cache_create_no_init( cache_all, m_pid );
		    cache_init( env, c, cache_pid, m_pid );

		    for( ; v+VL*4 <= e; v += VL*4 ) {
			{
			    auto vv = simd::template
				create_constant<simd::ty<VID,1>>( v );
			    auto m = expr::create_value_map_new2<
				VL,expr::vk_vid,expr::vk_pid>( vv, pp );
			    cache_init( env, c, cache_vid, m );
			    auto r = env.evaluate( c, m, expr );
			    cache_commit( env, cache_vid, c, m, r.mpack() );
			}
			{
			    auto vv = simd::template
				create_constant<simd::ty<VID,1>>( v + VL );
			    auto m = expr::create_value_map_new2<
				VL,expr::vk_vid,expr::vk_pid>( vv, pp );
			    cache_init( env, c, cache_vid, m );
			    auto r = env.evaluate( c, m, expr );
			    cache_commit( env, cache_vid, c, m, r.mpack() );
			}
			{
			    auto vv = simd::template
				create_constant<simd::ty<VID,1>>( v + 2*VL );
			    auto m = expr::create_value_map_new2<
				VL,expr::vk_vid,expr::vk_pid>( vv, pp );
			    cache_init( env, c, cache_vid, m );
			    auto r = env.evaluate( c, m, expr );
			    cache_commit( env, cache_vid, c, m, r.mpack() );
			}
			{
			    auto vv = simd::template
				create_constant<simd::ty<VID,1>>( v + 3*VL );
			    auto m = expr::create_value_map_new2<
				VL,expr::vk_vid,expr::vk_pid>( vv, pp );
			    cache_init( env, c, cache_vid, m );
			    auto r = env.evaluate( c, m, expr );
			    cache_commit( env, cache_vid, c, m, r.mpack() );
			}
		    }

		    cache_commit( env, cache_pid, c, m_pid );

		    for( ; v < e; ++v ) {
			auto vv = simd::template create_constant<simd::ty<VID,1>>( v );
			auto pp = simd::template create_constant<simd::ty<VID,1>>( VL*VID(p) );
			auto m = expr::create_value_map_new2<
			    1,expr::vk_vid,expr::vk_pid>( vv, pp );
			auto c = cache_create( env, cache_vid1, m );
			auto r = env.evaluate( c, m, sexpr );
			cache_commit( env, cache_vid1, c, m, r.mpack() );
		    }
		}

		_mm_mfence(); // for streaming memory operations
	    } );

#if VMAP_TIMING
	vmap_record_time<Operator>( tm.next() );
#endif
    }

public:
    template<typename map_type0>
    struct ptrset {
	using map_type = typename expr::ast_ptrset::ptrset_list<
	    map_type0,vexpr_type>::map_type;

	template<typename MapTy>
	static void initialize( MapTy & map, const self_type & op ) {
	    auto v = expr::value<simd::ty<VID,1>,expr::vk_vid>();

	    expr::ast_ptrset::ptrset_list<map_type0,vexpr_type>
		::initialize( map, op.op( v ) );
	}
    };

private:
    Operator op;
};

template<typename OperatorTy>
struct step_vmap_dense<true,OperatorTy> { // vscan
    using Operator = OperatorTy;
    using self_type = step_vmap_dense<true,Operator>;
    using vexpr_type = decltype( ((Operator*)nullptr)->operator()( expr::value<simd::ty<VID,1>,expr::vk_vid>() ) );

    // The accumulator type is only meaningfull for scan steps
    using accum_type = decltype( expr::extract_accumulators( ((Operator*)nullptr)->operator()( expr::value<simd::ty<VID,1>,expr::vk_vid>() ) ) );

    static constexpr vmap_kind vkind = vmap_scan;
    static const bool has_frontier = false;
    
    constexpr step_vmap_dense( Operator _op ) : op( _op ) { }

    auto get_operator() const { return op; }

    void execute( const frontier & cfrontier, const partitioner & part ) {
	if constexpr ( Operator::is_filter ) {
	    execute( op.m_G, cfrontier, op.m_f, part );
	} else {
	    if( cfrontier.isEmpty() )
		return;

	    switch( cfrontier.getType() ) {
	    case frontier_type::ft_true:
		execute( part );
		break;
	    case frontier_type::ft_unbacked:
		assert( 0 && "ERROR! CANNOT READ UNBACKED FRONTIER" );
		break;
	    case frontier_type::ft_sparse:
		assert( 0 && "ERROR! SHOULD NOT GET HERE" );
		break;
	    case frontier_type::ft_bool:
		execute<frontier_type::ft_bool>( cfrontier, part, op );
		break;
	    case frontier_type::ft_bit:
		// execute<frontier_type::ft_bit>( cfrontier, part, op );
		assert( 0 && "NYI" );
		break;
	    case frontier_type::ft_bit2:
		execute<frontier_type::ft_bit2>( cfrontier, part, op );
		break;
	    case frontier_type::ft_logical1:
	    case frontier_type::ft_logical2:
		assert( 0 && "Not instantiated to save compilation time" );
		break;
	    case frontier_type::ft_logical4:
		execute<frontier_type::ft_logical4>( cfrontier, part, op );
		break;
	    case frontier_type::ft_logical8:
		execute<frontier_type::ft_logical8>( cfrontier, part, op );
		break;
	    default:
		UNREACHABLE_CASE_STATEMENT;
	    }
	}
    }
private:
    template<typename GraphType>
    void execute( const GraphType & G,
		  const frontier & cfrontier,
		  frontier & nfrontier,
		  const partitioner & part ) {
	nfrontier = frontier::empty();

	if( cfrontier.isEmpty() )
	    return;

	switch( cfrontier.getType() ) {
	case frontier_type::ft_true:
	    execute( part );
	    break;
	case frontier_type::ft_unbacked:
	    assert( 0 && "ERROR! CANNOT READ UNBACKED FRONTIER" );
	    break;
	case frontier_type::ft_sparse:
	    assert( 0 && "ERROR! SHOULD NOT GET HERE" );
	    break;
	case frontier_type::ft_bool:
	    execute<frontier_type::ft_bool>( G, cfrontier, nfrontier, part, op );
	    break;
	case frontier_type::ft_bit:
	    // execute<frontier_type::ft_bit>( cfrontier, part, op );
	    assert( 0 && "NYI" );
	    break;
	case frontier_type::ft_bit2:
	    execute<frontier_type::ft_bit2>( G, cfrontier, nfrontier, part, op );
	    break;
	case frontier_type::ft_logical1:
	    execute<frontier_type::ft_logical1>( G, cfrontier, nfrontier, part, op );
	    break;
	case frontier_type::ft_logical2:
	    assert( 0 && "Not instantiated to save compilation time" );
	    break;
	case frontier_type::ft_logical4:
	    execute<frontier_type::ft_logical4>( G, cfrontier, nfrontier, part, op );
	    break;
	case frontier_type::ft_logical8:
	    execute<frontier_type::ft_logical8>( G, cfrontier, nfrontier, part, op );
	    break;
	default:
	    UNREACHABLE_CASE_STATEMENT;
	}
    }
public:
    void execute( const partitioner & part ) {
	if constexpr ( Operator::is_filter ) {
	    execute( op.m_G, op.m_f, part, op );
	} else {
	    using frontier_spec = expr::determine_frontier_vmap<VID,Operator>;
	    execute<frontier_spec>( part, op );
	}
    }

private:
    template<frontier_type ftype>
    void execute( const frontier & cfrontier, const partitioner & part,
		  Operator op ) {
	using frontier_spec = expr::determine_frontier_vmap<VID,Operator>;
	constexpr unsigned short VL = frontier_spec::VL;

	wrap_frontier_read_m<Operator,ftype,VL>
	    f_op( cfrontier.getDense<ftype>(), op );

	execute<frontier_spec>( part, f_op );
    }
    template<frontier_type ftype, typename GraphType>
    void execute( const GraphType & G, const frontier & cfrontier,
		  frontier & nfrontier, const partitioner & part, Operator op ) {
	using frontier_spec = expr::determine_frontier_vmap<VID,Operator>;
	constexpr unsigned short VL = frontier_spec::VL;

	nfrontier = frontier::template create<ftype>( part );

	// Apply wrap_filter before applying wrap_frontier_read_m such
	// that masks aplied by ..._read_m apply also to wrap_filter.
	wrap_filter<GraphType,Operator,ftype,VL> op1( G, nfrontier, op );

	wrap_frontier_read_m<wrap_filter<GraphType,Operator,ftype,VL>,ftype,VL>
	    op2( cfrontier.getDense<ftype>(), op1 );

	execute<frontier_spec>( part, op2 );
    }

    template<typename GraphType>
    void execute( const GraphType & G,
		  frontier & nfrontier, const partitioner & part, Operator op ) {
	using frontier_spec = expr::determine_frontier_vmap<VID,Operator>;
	constexpr unsigned short VL = frontier_spec::VL;
	constexpr frontier_type ftype = VL == 1
		      ? frontier_type::ft_bool
		      : frontier_type::ft_logical4;

	nfrontier = frontier::template create<ftype>( part );

	wrap_filter<GraphType,Operator,ftype,VL> f_opw( G, nfrontier, op );

	execute<frontier_spec>( part, f_opw );
    }

    template<typename frontier_spec, typename FrOperatorTy>
    void execute( const partitioner & part, FrOperatorTy op ) {
	constexpr unsigned short VL = frontier_spec::VL;
	vmap_report<Operator,frontier_spec>( std::cout, "vmap_dense/scan:" );
#if VMAP_TIMING
	timer tm;
	tm.start();
#endif

	auto svid = expr::value<simd::ty<VID,1>,expr::vk_vid>();
	auto vid = expr::make_unop_incseq<VL>( svid );
	auto pid = expr::value<simd::ty<VID,VL>,expr::vk_pid>();
	auto spid = expr::value<simd::ty<VID,1>,expr::vk_pid>();
	auto expr0 = op( vid );
	auto sexpr0 = op( svid );

	auto accum = expr::extract_accumulators( expr0 );
	expr::accum_create( part, accum );
	auto expr1 = expr::rewrite_privatize_accumulators( expr0, part, accum, pid );
	auto pexpr = expr::accumulate_privatized_accumulators( pid, accum );
	auto pfexpr = expr::final_accumulate_privatized_accumulators( pid, accum );

	auto cache_l = expr::extract_local_vars( expr1, expr::cache<>() );
	auto expr2 = expr::rewrite_caches<expr::vk_zero>( expr1, cache_l );

	// 1. op0->st->ld->op1 becomes op0->st ; op0->op1
	//    this may become too complex for ASTs; C++ compiler cannot do this
	//    because it only knows the types, not the pointers. We can only
	//    work on the AIDs, however, this may be impractical as AIDs would
	//    need to match across vertex map operators.
	// 2. step_vscan: contains an accumulator: privatize the accumulator
	//    in each partition (cacheop) and follow on by a scan across accu's
	// 3. step_vscan: in case output is used, do not fuse across a vscan?
	//    or can we find a way to identify when output is used?
	// 4. step_vscan: do we need to differentiate from vmap in programming
	//    model, or can we figure this out ourselves through analysis?
	auto cache_pid = expr::extract_cacheable_refs<expr::vk_pid>( expr2, cache_l );
	auto expr3 = expr::rewrite_caches<expr::vk_pid>( expr2, cache_pid );
	auto cache_vid = expr::extract_uses<expr::vk_vid>( expr3, expr::cache_cat( cache_l, cache_pid ) );
	auto expr4 = expr::rewrite_caches<expr::vk_vid>( expr3, cache_vid );
	auto expr5 = expr::rewrite_vectors_main( expr4 );
	auto expr6 = expr::rewrite_mask_main( expr5 );
	// auto expr = expr::rewrite_incseq( expr6 ); // TODO -- errors in VL match
	auto expr = expr6;

	// TODO: this creates new, scalar accumulators. Are we adding up
	//       vector and scalar parts correctly?
	auto cache_vid1 = expr::extract_uses<expr::vk_vid>(
	    sexpr0, cache_cat( cache_cat( cache_l, cache_pid ), cache_vid ) );
	auto sexpr1 = expr::rewrite_privatize_accumulators( sexpr0, part, accum, spid );
	auto sexpr2 = expr::rewrite_caches<expr::vk_vid,expr::mam_nontemporal>( sexpr1, cache_vid1 );
	auto sexpr3 = expr::rewrite_caches<expr::vk_pid>( sexpr2, cache_pid );
	auto cache_l1 = expr::extract_local_vars(
	    sexpr3,
	    cache_cat( cache_cat( cache_l, cache_pid ),
		       cache_cat( cache_vid, cache_vid1 ) ) );
	auto sexpr4 = expr::rewrite_caches<expr::vk_zero>( sexpr3, cache_l1 );
	auto sexpr5 = expr::rewrite_vectors_main( sexpr4 );
	auto sexpr = expr::rewrite_mask_main( sexpr5 );

	auto cache_all = cache_cat( cache_cat( cache_l, cache_pid ),
				    cache_cat(
					cache_cat( cache_vid, cache_vid1 ),
					cache_l1 ) );

/*
	auto env = expr::eval::create_execution_environment(
	    cache_all, sexpr, expr, pexpr, pfexpr ); 
*/
	auto env = expr::eval::create_execution_environment_op( op, cache_all );

	// fail_expose<std::is_class>( sexpr0 );
	// fail_expose<std::is_class>( cache_l );

/*
	// Same again now with mask applied
	// TODO: There is a potential issue here as the end of one partition
	//       will partially update a vector while the start of the next
	//       will do the same. To avoid this, we should only allow
	//       partitions with a number of vertices that is a multiple of VL.
	//       Only the final partition can have a number of vertices that is
	//       not a multiple of VL and every partition starts at a multiple
	//       of VL.
	auto m_expr0 = op( vidm );
	auto m_expr1 = expr::rewrite_privatize_accumulators( m_expr0, part, accum, pid );
	auto m_expr2 = expr::rewrite_caches<expr::vk_pid>( m_expr1, cache_pid );
	auto m_expr3 = expr::rewrite_caches<expr::vk_vid>( m_expr2, cache_vid );
	auto m_expr4 = expr::rewrite_vectors_main( m_expr3 );
	auto m_expr = expr::rewrite_mask_main( m_expr4 );
*/

	// fail_expose<std::is_class>( expr );
	// fail_expose<std::is_class>( cache_vid );

	// TODO:
	// * avoid reload of values
	// * keep accumulator in register (cache)
	// * avoid multiple integer registers resolving to the same index
	//
	// extract_cacheable_refs should work to provide a list of
	// all refs indexed by vk_dst (not vk_pid -> replace or parameterize?)
	// Then we should be able to use the existing cache_create/cache_commit
	// before and after the map loop
	//
	// Current status; AddConstant + SeqSum: constant not in register but
	//  loaded from memory on each iteration

	// Map
	map_partitionL( part, [&]( int p ) {
	      // auto pp = simd_vector<VID,VL>::template s_set1inc<true>(
				  // VL*VID(p) );
		auto pp = simd::template create_set1inc<simd::ty<VID,VL>,true>(
		    VL*VID(p) );
		VID s = part.start_of_vbal( p );
		VID e = part.end_of_vbal( p );

		// assert( (s % VL) == 0 && "partition should be vertex-aligned" );

		// Load per-partition values
		auto m_pid = expr::create_value_map_new2<VL,expr::vk_pid>( pp );
		auto c = cache_create_no_init( cache_all, m_pid );
		cache_init( env, c, cache_pid, m_pid );

		VID v=s;
		VID vstart = std::min( e, roundupVL<VL>( s ) );
		for( ; v < vstart; ++v ) {
		    auto vv = simd::template create_constant<simd::ty<VID,1>>( v );
		    auto m = expr::create_value_map_new2<1,expr::vk_vid>( vv );
		    cache_init( env, c, cache_vid1, m );
		    auto r = env.evaluate( c, m, sexpr );
		    cache_commit( env, cache_vid1, c, m, r.mpack() );
		}

		// TODO: prefetch -> extract_ptrset_indexed_by<vid> !
		//       and if write-before-read, then instantiate cache
		//       block in cache before writing (use hint on assembly)
		// We need to extract all arrays + index expressions + R/W
		// Embed info in use list -> set flag if to be cached or not!
		// Use attributes: read-before-write; modified; cached
		// For safety, only create uninit block in cache if array
		//   indexed in one way only -> also cached
		//   it will also need to be not read-before-write
		//   then, this write only occurs in the commit!
		// ... cannot create the cache block if write is masked!
		if( v < e ) {
		    for( ; v+VL <= e; v += VL ) {
			auto vv = simd::template create_constant<simd::ty<VID,1>>( v );
			auto m = expr::create_value_map_new2<
			    VL,expr::vk_vid,expr::vk_pid>( vv, pp );
			cache_init( env, c, cache_vid, m );
			// expr::cache_prefetch<PFV_DISTANCE>( c, cache_vid, m );
			auto r = env.evaluate( c, m, expr );
			cache_commit( env, cache_vid, c, m, r.mpack() );
		    }

		    for( ; v < e; ++v ) {
			auto vv = simd::template create_constant<simd::ty<VID,1>>( v );
			auto m = expr::create_value_map_new2<1,expr::vk_vid>( vv );
			cache_init( env, c, cache_vid1, m );
			auto r = env.evaluate( c, m, sexpr );
			cache_commit( env, cache_vid1, c, m, r.mpack() );
		    }
		}

		cache_commit( env, cache_pid, c, m_pid );

		_mm_mfence(); // for streaming memory operations
	    } );

	// Scan across partitions
	int np = part.get_num_partitions();
	for( int p=1; p < np; ++p ) {
	    // TODO: replace by unop_incseq
	    auto pp = simd::template create_set1inc<simd::ty<VID,VL>,true>(
		VL*VID(p) );
	    auto m = expr::create_value_map_new2<VL,expr::vk_pid>( pp );
	    expr::cache<> c;
	    env.evaluate( c, m, pexpr );
	}
	{
	    auto pp = simd::template create_set1inc0<simd::ty<VID,VL>>();
	    auto m = expr::create_value_map_new2<VL,expr::vk_pid>( pp );
	    expr::cache<> c;
	    env.evaluate( c, m, pfexpr );
	}

	accum_destroy( accum );

#if VMAP_TIMING
	vmap_record_time<Operator>( tm.next() );
#endif
    }

public:
    template<typename map_type0>
    struct ptrset {
	using map_type = typename expr::ast_ptrset::ptrset_list<
	    map_type0,vexpr_type>::map_type;

	template<typename MapTy>
	static void initialize( MapTy & map, const self_type & op ) {
	    auto v = expr::value<simd::ty<VID,1>,expr::vk_vid>();

	    expr::ast_ptrset::ptrset_list<map_type0,vexpr_type>
		::initialize( map, op.op( v ) );
	}
    };

private:
    Operator op;
};

template<typename OperatorTy>
struct step_vmap_sparse {
    using Operator = OperatorTy;
    using self_type = step_vmap_sparse<Operator>;
    using vexpr_type = decltype( ((Operator*)nullptr)->operator()( expr::value<simd::ty<VID,1>,expr::vk_vid>() ) );

    static const bool has_frontier = true;
    
    constexpr step_vmap_sparse( Operator _op, const frontier & f )
	: op( _op ), fref( f ) { }

    auto get_operator() const { return op; }

    void execute( const partitioner & part ) {
	if constexpr ( Operator::is_filter ) {
	    execute_filter( part );
	} else if constexpr ( Operator::is_scan )
	    execute_scan( part );
	else
	    execute_map( part );

    }

private:
    void execute_map( const partitioner & part ) {
	vmap_report<Operator,1>( std::cout, "vmap_sparse:" );
#if VMAP_TIMING
	timer tm;
	tm.start();
#endif

	// TODO: sparse vmap could make use of gather/scatter
	static constexpr unsigned short VL = 1; // sparse, thus scalar
	auto spid = expr::value<simd::ty<VID,1>,expr::vk_pid>();
	auto expr0 = op( expr::value<simd::ty<VID,VL>,expr::vk_vid>() );

	auto l_cache = expr::extract_local_vars( expr0, expr::cache<>() );
	auto expr1 = expr::rewrite_caches<expr::vk_zero>( expr0, l_cache );
	auto cache_pid1 = expr::extract_cacheable_refs<expr::vk_pid>( expr1, l_cache );
	auto expr2 = expr::rewrite_caches<expr::vk_pid>( expr1, cache_pid1 );

	auto cache_all = cache_cat( l_cache, cache_pid1 );
	// auto env = expr::eval::create_execution_environment( expr2, cache_all ); 
	auto env = expr::eval::create_execution_environment_op( op, cache_all );

	auto expr = rewrite_internal( expr2 );


	VID nactv = fref.nActiveVertices();
	const VID * f_array = fref.getSparse();
	parallel_loop( (VID)0, nactv, 256, [&]( auto i ) {
	    VID v = f_array[i];
	    auto vv = simd::template create_unknown<simd::ty<VID,VL>>( v );
// TODO: race conditions on p...
	    auto pp = simd::template create_unknown<simd::ty<VID,VL>>( VL*VID(0) );
	    auto m = expr::create_value_map_new2<VL,expr::vk_vid,expr::vk_pid>(
		vv, pp );
	    auto c = cache_create( env, cache_all, m );
	    auto r = env.evaluate( c, m, expr );
	    cache_commit( env, cache_all, c, m, r.mpack() );
	} );

#if VMAP_TIMING
	vmap_record_time<Operator>( tm.next() );
#endif
    }

    void execute_scan( const partitioner & part ) {
	vmap_report<Operator,1>( std::cout, "vmap_sparse/scan:" );
#if VMAP_TIMING
	timer tm;
	tm.start();
#endif

	// TODO: sparse vscan could make use of gather/scatter
	// TODO: this is a very simplistic sequential implementation
	// TODO: add caching of accumulator variable
	static constexpr unsigned short VL = 1; // sparse, thus scalar
	auto expr0 = op( expr::value<simd::ty<VID,VL>,expr::vk_vid>() );

	auto l_cache = expr::extract_local_vars( expr0, expr::cache<>() );
	auto expr1 = expr::rewrite_caches<expr::vk_zero>( expr0, l_cache );

	// auto env = expr::eval::create_execution_environment( expr1, l_cache ); 
	auto env = expr::eval::create_execution_environment_op( op, l_cache );

	auto expr = rewrite_internal( expr1 );


	VID nactv = fref.nActiveVertices();
	const VID * f_array = fref.getSparse();
	/*parallel_*/for( VID i=0; i < nactv; ++i ) {
	    VID v = f_array[i];
	    auto vv = simd::template create_unknown<simd::ty<VID,VL>>( v );
	    auto m = expr::create_value_map_new2<VL,expr::vk_vid>( vv );
	    auto c = cache_create( env, l_cache, m );
	    env.evaluate( c, m, expr );
	}

#if VMAP_TIMING
	vmap_record_time<Operator>( tm.next() );
#endif
    }
    
    
    void execute_filter( const partitioner & part ) {
	vmap_report<Operator,1>( std::cout, "vmap_sparse:" );
#if VMAP_TIMING
	timer tm;
	tm.start();
#endif

	VID nactv = fref.nActiveVertices();
	VID n = op.m_G.numVertices();

	frontier & nfrontier = op.m_f;
	nfrontier = frontier::sparse( n, nactv );

	static constexpr unsigned short VL = 1; // sparse, thus scalar
	auto expr0 = op( expr::value<simd::ty<VID,VL>,expr::vk_vid>() );

	auto l_cache = expr::extract_local_vars( expr0, expr::cache<>() );
	auto expr1 = expr::rewrite_caches<expr::vk_zero>( expr0, l_cache );

	// auto env = expr::eval::create_execution_environment( expr1, l_cache ); 
	auto env = expr::eval::create_execution_environment_op( op, l_cache );

	auto expr = rewrite_internal( expr1 );

	const VID * f_array = fref.getSparse();
	VID * n_array = nfrontier.getSparse();
	parallel_loop( (VID)0, nactv, 256, [&]( auto i ) {
	    VID v = f_array[i];
	    auto vv = simd::template create_unknown<simd::ty<VID,VL>>( v );
	    auto m = expr::create_value_map_new2<VL,expr::vk_vid>( vv );
	    auto c = cache_create( env, l_cache, m );
	    auto ret = env.evaluate( c, m, expr );
	    if constexpr ( ret.has_mask() ) {
		if( ret.value().data()
		    && ret.mpack().template get_any<simd::detail::mask_bool_traits>().data() ) { // VL == 1
		    n_array[i] = v;
		} else {
		    n_array[i] = ~(VID)0;
		}
	    } else {
		if( ret.value().data() )
		    n_array[i] = v;
		else
		    n_array[i] = ~(VID)0;
	    }
	} );

	// Compact array.
	// TODO: parallel scan
	VID l = 0;
	EID e = 0;
	for( VID k=0; k < nactv; ++k ) {
	    if( n_array[k] != ~(VID)0 ) {
		n_array[l++] = n_array[k];
		e += op.m_G.getOutDegree( n_array[k] );
	    }
	}

	nfrontier.setActiveCounts( l, e );

#if VMAP_TIMING
	vmap_record_time<Operator>( tm.next() );
#endif
    }

public:
    template<typename map_type0>
    struct ptrset {
	using map_type = typename expr::ast_ptrset::ptrset_list<
	    map_type0,vexpr_type>::map_type;

	template<typename MapTy>
	static void initialize( MapTy & map, const self_type & op ) {
	    auto v = expr::value<simd::ty<VID,1>,expr::vk_vid>();

	    expr::ast_ptrset::ptrset_list<map_type0,vexpr_type>
		::initialize( map, op.op( v ) );
	}
    };
    
private:
    Operator op;
    const frontier & fref;
};


/***********************************************************************
 * Auxiliary constructors for step_vmap_dense<> and step_vmap_sparse<>.
 ***********************************************************************/
template<bool IsScan,typename Operator>
constexpr auto make_step_vmap_dense( Operator op ) {
    return step_vmap_dense<IsScan,Operator>( op );
}

template<typename Operator>
constexpr auto make_step_vmap_sparse( Operator op, const frontier & f ) {
    return step_vmap_sparse<Operator>( op, f );
}

/***********************************************************************
 * Definition of CSC edge map step. Assumes dense frontier.
 * This step can only be the first step in a sequence because all
 * computations are materialised at the start of an edgemap.
 * The step already encodes the vector length. Whoever creates this, must
 * decide the appropriate vector length.
 * Any vmap operations appended to it must be able to use the same
 * vector length if they are to be merged in.
 * The types of the old and new frontier are also fixed. Whoever creates
 * this emap step must decide on appropriate frontier types, considering
 * also the vector length, and transform or create frontiers as needed.
 * The final boolean argument indicates whether the step operates with
 * a frontier or not.
 ***********************************************************************/
template<unsigned short VL_, frontier_type old_frontier_,
	 frontier_type new_frontier_>
struct emap_config {
    static constexpr unsigned short VL = VL_;
    static constexpr frontier_type old_frontier = old_frontier_;
    static constexpr frontier_type new_frontier = new_frontier_;

    static std::ostream & dump( std::ostream & os ) {
	return os << "emap_config { VL=" << VL << ", old_fr=" << old_frontier
		  << ", new_fr=" << new_frontier << " }";
    }
};

// TODO: move FTrue into emap_config
template<typename GraphType, typename EdgeOperator, typename EMapConfig,
	 bool FTrue>
struct step_emap_dense;

template<typename EMapConfig,typename GraphType, typename EdgeOperator>
step_emap_dense<GraphType,EdgeOperator,EMapConfig,true>
make_step_emap_dense( const GraphType & G, EdgeOperator op,
		      graph_traversal_kind gtk );

template<typename EMapConfig,typename GraphType, typename EdgeOperator>
step_emap_dense<GraphType,EdgeOperator,EMapConfig,false>
make_step_emap_dense( const GraphType & G, const frontier & old_frontier,
		      frontier & new_frontier, EdgeOperator op );

template<typename GraphType, typename EdgeOperator, typename EMapConfig>
struct step_emap_dense<GraphType,EdgeOperator,EMapConfig,false> {
    using config = EMapConfig;
    static const bool has_frontier = true;

    constexpr step_emap_dense( const GraphType & _G,
			       const frontier & _old_frontier,
			       frontier & _new_frontier,
			       EdgeOperator _op )
	: G( _G ), old_frontier( _old_frontier ), new_frontier( _new_frontier ),
	  op( _op ) { }

    // Definition is located in edgemap.h
    __attribute__((noinline))
    void execute( const partitioner & part );

    // This may be called from step_sdchoice, which also passes in the control
    // frontier.
    void execute( const frontier &, const partitioner & part ) {
	execute( part );
    }

    template<typename VOperator,bool IsScan>
    constexpr auto append_post( step_vmap_dense<IsScan,VOperator> step ) const {
	// Create new step_emap_dense with modified Operator that encodes
	// the VOperator in its post-calculation
	return make_step_emap_dense<config>(
	    G, old_frontier, new_frontier,
	    AppendVOpToEOp<IsScan || EdgeOperator::is_scan,VOperator,EdgeOperator>(
		step.get_operator(), op ) );

	// Could also have post-operation for COO, but then applied sequentially
	// within partition, not immediately after vertex. Key point: don't need
	// to wait for other partitions to complete...
    }

    void materialize_frontier( const partitioner & part );

    const GraphType & getGraph() const { return G; }
    const frontier & getOldFrontier() const { return old_frontier; }
    frontier & getNewFrontier() { return new_frontier; }
    EdgeOperator & get_operator() { return op; }

private:
    const GraphType & G;
    const frontier & old_frontier;
    frontier & new_frontier;
    EdgeOperator op;
};

template<typename GraphType, typename EdgeOperator,typename EMapConfig>
struct step_emap_dense<GraphType,EdgeOperator,EMapConfig,true> {
    using config = EMapConfig;
    static const bool has_frontier = false;

    constexpr step_emap_dense( const GraphType & G,
			       // frontier & _new_frontier,
			       EdgeOperator op,
			       graph_traversal_kind gtk )
	: m_G( G ), /* new_frontier( _new_frontier ), */
	  m_op( op ),
	  m_gtk( gtk ) { }

    // Definition is located in edgemap.h
    __attribute__((noinline))
    void execute( const partitioner & part );

    template<graph_traversal_kind> // TODO: move to class scope
    __attribute__((noinline))
    void execute( const partitioner & part );

    [[deprecated("replace by interface with frontier")]]
    void execute( bool *, const partitioner & part ) {
	execute( part );
    }

    void execute( const frontier &, const partitioner & part ) {
	execute( part );
    }
    
    template<typename VOperator,bool IsScan>
    constexpr auto append_post( step_vmap_dense<IsScan,VOperator> step ) const {
	// Create new step_emap_dense with modified Operator that encodes
	// the VOperator in its post-calculation
	return make_step_emap_dense<config>(
	    m_G, // new_frontier,
	    AppendVOpToEOp<IsScan || EdgeOperator::is_scan,VOperator,EdgeOperator>(
		step.get_operator(), m_op ), m_gtk );

	// Could also have post-operation for COO, but then applied sequentially
	// within partition, not immediately after vertex. Key point: don't need
	// to wait for other partitions to complete...
    }

    void materialize_frontier( const partitioner & part );

    const GraphType & getGraph() const { return m_G; }
    // frontier & getNewFrontier() { return new_frontier; }
    EdgeOperator & get_operator() { return m_op; }

private:
    const GraphType & m_G;
    // frontier & new_frontier;
    EdgeOperator m_op;
    graph_traversal_kind m_gtk;
};

template<typename EMapConfig,typename GraphType, typename EdgeOperator>
step_emap_dense<GraphType,EdgeOperator,EMapConfig,true>
make_step_emap_dense( const GraphType & G, EdgeOperator op,
		      graph_traversal_kind gtk) {
    return step_emap_dense<GraphType,EdgeOperator,EMapConfig,true>( G, op, gtk );
}

template<typename EMapConfig,typename GraphType, typename EdgeOperator>
step_emap_dense<GraphType,EdgeOperator,EMapConfig,false>
make_step_emap_dense( const GraphType & G, const frontier & old_frontier,
		      frontier & new_frontier, EdgeOperator op ) {
    return step_emap_dense<GraphType,EdgeOperator,EMapConfig,false>(
	G, old_frontier, new_frontier, op );
}

template<typename GraphType, typename EdgeOperator, typename EMapConfig,
	 bool FTrue>
struct step_emap_dense;

template<typename EMapConfig,typename GraphType, typename EdgeOperator>
step_emap_dense<GraphType,EdgeOperator,EMapConfig,true>
make_step_emap_dense( const GraphType & G,
		      frontier & new_frontier, EdgeOperator op );

template<typename EMapConfig,typename GraphType, typename EdgeOperator>
step_emap_dense<GraphType,EdgeOperator,EMapConfig,false>
make_step_emap_dense( const GraphType & G, const frontier & old_frontier,
		      frontier & new_frontier, EdgeOperator op );

/***********************************************************************
 * Flat edge_map step
 ***********************************************************************/
template<typename EdgeOperator>
struct step_flat_emap {
    constexpr step_flat_emap( EdgeOperator _op ) : m_op( _op ) { }

    void execute( const partitioner & part ) {
	using frontier_spec = expr::determine_frontier_vmap<EID,EdgeOperator>;
	constexpr unsigned short VL = frontier_spec::VL;
	vmap_report<EdgeOperator,frontier_spec>( std::cout, "flat_emap:" );
#if VMAP_TIMING
	timer tm;
	tm.start();
#endif

	auto seid = expr::value<simd::ty<EID,1>,expr::vk_edge>();
	auto eid = expr::make_unop_incseq<VL>( seid );
	auto spid = expr::value<simd::ty<VID,1>,expr::vk_pid>();
	auto pid = expr::make_unop_incseq<VL>( spid );
	auto expr0 = m_op( eid );
	auto sexpr0 = m_op( seid );

	auto accum = expr::extract_accumulators( expr0 );
	expr::accum_create( part, accum );
	auto expr1 = expr::rewrite_privatize_accumulators( expr0, part, accum, pid );
	auto pexpr = expr::accumulate_privatized_accumulators( pid, accum );
	auto pfexpr = expr::final_accumulate_privatized_accumulators( pid, accum );

	auto cache_l = expr::extract_local_vars( expr1, expr::cache<>() );
	auto expr2 = expr::rewrite_caches<expr::vk_zero>( expr1, cache_l );

	auto cache_pid = expr::extract_cacheable_refs<expr::vk_pid>(
	    expr2, cache_l );
	auto expr3 = expr::rewrite_caches<expr::vk_pid>( expr2, cache_pid );
	auto cache_vid = expr::extract_uses<expr::vk_vid>(
	    expr3, expr::cache_cat( cache_l, cache_pid ) );
	auto expr4 = expr::rewrite_caches<expr::vk_vid>( expr3, cache_vid );
	auto expr5 = expr::rewrite_vectors_main( expr4 );
	auto expr6 = expr::rewrite_mask_main( expr5 );
	auto expr = expr6;

	// TODO: this creates new, scalar accumulators. Are we adding up
	//       vector and scalar parts correctly?
	auto cache_vid1 = expr::extract_uses<expr::vk_vid>(
	    sexpr0, cache_cat( cache_cat( cache_l, cache_pid ), cache_vid ) );
	auto sexpr1 = expr::rewrite_privatize_accumulators(
	    sexpr0, part, accum, spid );
	auto sexpr2 = expr::rewrite_caches<expr::vk_vid,expr::mam_nontemporal>(
	    sexpr1, cache_vid1 );
	auto sexpr3 = expr::rewrite_caches<expr::vk_pid>( sexpr2, cache_pid );
	auto cache_l1 = expr::extract_local_vars(
	    sexpr3,
	    cache_cat( cache_cat( cache_l, cache_pid ),
		       cache_cat( cache_vid, cache_vid1 ) ) );
	auto sexpr4 = expr::rewrite_caches<expr::vk_zero>( sexpr3, cache_l1 );
	auto sexpr5 = expr::rewrite_vectors_main( sexpr4 );
	auto sexpr = expr::rewrite_mask_main( sexpr5 );

	auto cache_all = cache_cat( cache_cat( cache_l, cache_pid ),
				    cache_cat(
					cache_cat( cache_vid, cache_vid1 ),
					cache_l1 ) );

/*
	auto env = expr::eval::create_execution_environment(
	    cache_all, sexpr, expr, pexpr, pfexpr ); 
*/
	auto env = expr::eval::create_execution_environment_op(
	    m_op, cache_all );

	// Map
	map_partitionL( part, [&]( int p ) {
	    auto pp = simd::template create_constant<simd::ty<VID,1>>(
		VL*VID(p) );
	    EID es = part.edge_start_of( p );
	    EID ee = part.edge_end_of( p );

	    // Load per-partition values
	    auto m_pid = expr::create_value_map_new2<VL,expr::vk_pid>( pp );
	    auto c = cache_create_no_init( cache_all, m_pid );
	    cache_init( env, c, cache_pid, m_pid );

	    EID e=es;
	    EID estart = std::min( e, roundupVL<VL>( e ) );
	    for( ; e < estart; ++e ) {
		auto ee = simd::template create_constant<simd::ty<EID,1>>( e );
		auto m = expr::create_value_map_new2<1,expr::vk_edge>( ee );
		cache_init( env, c, cache_vid1, m );
		auto r = env.evaluate( c, m, sexpr );
		cache_commit( env, cache_vid1, c, m, r.mpack() );
	    }

	    if( e < ee ) {
		for( ; e+VL <= ee; e += VL ) {
		    auto ee = simd::template create_constant<simd::ty<EID,1>>(
			e );
		    auto m = expr::create_value_map_new2<
			VL,expr::vk_edge,expr::vk_pid>( ee, pp );
		    cache_init( env, c, cache_vid, m );
		    auto r = env.evaluate( c, m, expr );
		    cache_commit( env, cache_vid, c, m, r.mpack() );
		}

		for( ; e < ee; ++e ) {
		    auto ee = simd::template create_constant<simd::ty<EID,1>>( e );
		    auto m = expr::create_value_map_new2<1,expr::vk_edge>( ee );
		    cache_init( env, c, cache_vid1, m );
		    auto r = env.evaluate( c, m, sexpr );
		    cache_commit( env, cache_vid1, c, m, r.mpack() );
		}
	    }

	    cache_commit( env, cache_pid, c, m_pid );

	    _mm_mfence(); // for streaming memory operations
	} );

	// Scan across partitions
	int np = part.get_num_partitions();
	for( int p=1; p < np; ++p ) {
	    // TODO: replace by unop_incseq
	    auto pp = simd::template create_constant<simd::ty<VID,1>>(
		VL*VID(p) );
	    auto m = expr::create_value_map_new2<VL,expr::vk_pid>( pp );
	    expr::cache<> c;
	    env.evaluate( c, m, pexpr );
	}
	{
	    auto pp = simd::template create_constant<simd::ty<VID,1>>( 0 );
	    auto m = expr::create_value_map_new2<VL,expr::vk_pid>( pp );
	    expr::cache<> c;
	    env.evaluate( c, m, pfexpr );
	}

	accum_destroy( accum );

#if VMAP_TIMING
	vmap_record_time<EdgeOperator>( tm.next() );
#endif
    }

    EdgeOperator & get_operator() { return m_op; }

private:
    EdgeOperator m_op;
};


template<typename EdgeOperator>
auto make_step_flat_emap( EdgeOperator op ) {
    return step_flat_emap<EdgeOperator>( op );
}


/***********************************************************************
 * step for dense edge map, with 3 operators provided
 ***********************************************************************/
#if 0
template<typename GraphType,
	 typename PushOperator, typename PullOperator, typename IRegOperator>
struct step_emap_dense3 {
    static const bool has_frontier =
	PushOperator::has_frontier || PullOperator::has_frontier ||
	IRegOperator::has_frontier;

    constexpr step_emap_dense3( const GraphType & G,
				graph_traversal_kind kind,
				PushOperator push,
				PullOperator pull,
				IRegOperator ireg )
	: m_G( G ), m_kind( kind ), m_push( push ), m_pull( pull ),
	  m_ireg( ireg ) { }

    // Definition is located in edgemap.h
    __attribute__((noinline))
    void execute( const partitioner & part );

    // This may be called from step_sdchoice, which also passes in the control
    // frontier.
    void execute( const frontier &, const partitioner & part ) {
	execute( part );
    }

    template<typename VOperator,bool IsScan>
    constexpr auto append_post( step_vmap_dense<IsScan,VOperator> step ) const {
	// Create new step_emap_dense with modified Operator that encodes
	// the VOperator in its post-calculation
	return make_step_emap_dense3(
	    m_G, m_kind,
	    AppendVOpToEOp<IsScan || PushOperator::is_scan,VOperator,PushOperator>(
		step.get_operator(), m_push ),
	    AppendVOpToEOp<IsScan || PullOperator::is_scan,VOperator,PullOperator>(
		step.get_operator(), m_pull ),
	    AppendVOpToEOp<IsScan || IRegOperator::is_scan,VOperator,IRegOperator>(
		step.get_operator(), m_ireg )
	    );
    }

    const GraphType & get_graph() const { return m_G; }
    graph_traversal_kind get_traversal() const { return m_kind; }

    PushOperator get_step1() const { return m_push; }
    PullOperator get_step2() const { return m_pull; }
    IRegOperator get_step3() const { return m_ireg; }

private:
    const GraphType & m_G;
    const graph_traversal_kind m_kind;
    PushOperator m_push;
    PullOperator m_pull;
    IRegOperator m_ireg;
};

template<typename GraphType,
	 typename PushOperator, typename PullOperator, typename IRegOperator>
step_emap_dense3<GraphType,PushOperator,PullOperator,IRegOperator>
make_step_emap_dense3( const GraphType & G,
		       graph_traversal_kind kind,
		       PushOperator push,
		       PullOperator pull,
		       IRegOperator ireg ) {
    return step_emap_dense3<GraphType,PushOperator,PullOperator,IRegOperator>(
	G, kind, push, pull, ireg );
}

template<typename Step>
struct is_step_emap_dense3 : std::false_type { };

template<typename GraphType,
	 typename PushOperator, typename PullOperator, typename IRegOperator>
struct is_step_emap_dense3<step_emap_dense3<GraphType,PushOperator,
					    PullOperator,IRegOperator>>
    : std::true_type { };

#else

template<typename Step>
struct is_step_emap_dense3 : std::false_type { };

#endif

#if 0 // not currently in use
// Auxiliary class to turn the vertexop into a separate dense vmap step
// in order to execute the vertexop in isolation while reusing existing code
// for efficient implementation of step_vmap_dense
template<typename EdgeOperator>
class vmap_vertexop {
    using emap_step = EdgeOperator;

public:
    static constexpr bool is_scan = emap_step::is_scan;

    vmap_vertexop( emap_step estep_ ) : estep( estep_ ) { }

    template<typename VIDTy>
    auto operator() ( VIDTy vid ) {
	return estep.vertexop( vid );
    }

private:
    emap_step estep;
};

template<typename EdgeOperator>
vmap_vertexop<EdgeOperator> make_vmap_vertexop( EdgeOperator step ) {
    return vmap_vertexop<EdgeOperator>( step );
}
#endif // 0 // not currently in use

/***********************************************************************
 * Choice step: this structure is required because of the meta-programming
 * approach used in the DSL: all control flow paths must use the same type,
 * yet the required computations may differ across paths. The choice step
 * unifies the types by tracking all possible paths that may be taken in
 * the type and recording which of those paths is currently active in the
 * data field.
 ***********************************************************************/
template<typename... Step>
struct step_vchoice;

template<typename Step>
struct is_step_vchoice : std::false_type { };

template<typename... Steps>
struct is_step_vchoice<step_vchoice<Steps...>> : std::true_type { };

// Choice should perhaps be between steps and not between Expr...
template<typename... Step>
struct step_vchoice {
    static const bool has_frontier = true; // pessimistic

    constexpr step_vchoice( int c, Step... e )
	: choice( c ), steps( e... ) {
	static_assert( sizeof...(Step) > 0, "Need at least one Step" );
    }

    void execute( const partitioner & part ) {
	// std::cerr << "step_vchoice::execute: " << __PRETTY_FUNCTION__
	// << " choice=" << choice << "\n";
	execute_recurse<sizeof...(Step)-1>( part );
    }

    // peculiarity in step_sdchoice
    void execute( const frontier & f, const partitioner & part ) {
	execute( part );
    }

    int get_choice() const { return choice; }

    template<int i>
    auto get_step() const { return std::get<i>( steps ); }

private:
    template<int i>
    typename std::enable_if<(i>0)>::type
    execute_recurse( const partitioner & part ) {
	if( choice == i )
	    std::get<i>( steps ).execute( part );
	else
	    execute_recurse<i-1>( part );
    }
    template<int i>
    typename std::enable_if<(i==0)>::type
    execute_recurse( const partitioner & part ) {
	if( choice == i )
	    std::get<i>( steps ).execute( part );
    }

private:
    std::tuple<Step...> steps;
    int choice;
};

/***********************************************************************
 * Auxiliary constructor functions for step_vchoice<>
 ***********************************************************************/
template<typename... Step>
constexpr auto make_step_vchoice( int c, Step... s ) {
    return step_vchoice<Step...>( c, s... );
}

template<typename Operator>
auto make_step_vchoice( const frontier & f, Operator op ) {
    // using logicalVID = typename add_logical<VID>::type;
	
    auto t_step = make_step_vmap_dense<false>( op );

/*
TODO: dont decide on data type yet; register the frontier but leave type open. How? we dont know what type of pointer the frontier is. This should be generic? Or do we create a choice for every common width (bool, VID-wide logical), but then we also need to match these in merge so we need to extend sdchoice
sdchoice needs choices for: sparse, dense-true, dense-bool, dense-VID-wide-logical
A consequence is that we dont know the choice taken yet
*/

    bool * d_bool = f.getDense<bool>();
    wrap_frontier_read3<Operator,bool> d_op( d_bool, op );
    auto d_step = make_step_vmap_dense<false>( d_op );

    VID * s_vid = f.getSparse();
    auto s_step = make_step_vmap_sparse( op, f );

    return make_step_vchoice(
	f.allTrue() ? 0 : d_bool ? 1 : 2, t_step, d_step, s_step );
}

/***********************************************************************
* step_sdchoice: a sparse/dense frontier choice
* The key differentiator with step_vchoice is that we can leverage
* knowledge of sparse and dense cases and avoid merging a sparse step
* with a dense step.
***********************************************************************/
template<typename SparseStep, typename DenseStep>
struct step_sdchoice;

template<typename Step>
struct is_step_sdchoice : std::false_type { };

template<typename SparseStep, typename DenseStep>
struct is_step_sdchoice<step_sdchoice<SparseStep,DenseStep>>
    : std::true_type { };

// TODO: extend with IsScan template argument similar to dense steps
//       API currently does not offer a vertex_scan method with frontier,
//       so not needed until that happens.
template<typename SparseStep, typename DenseStep>
struct step_sdchoice {
    static const bool has_frontier = true;

    constexpr step_sdchoice(
	const frontier & f, SparseStep s, DenseStep d )
	: cfrontier( f ), sstep( s ), dstep( d ) { }

    void execute( const partitioner & part ) {
	// std::cerr << "step_sdchoice::execute: " << __PRETTY_FUNCTION__
	// << " choice=" << choice << "\n";
	if( cfrontier.isEmpty() )
	    return;
	if( cfrontier.getType() == frontier_type::ft_sparse )
	    sstep.execute( part );
	else
	    dstep.execute( cfrontier, part );
    }

    // int get_choice() const { return choice; }

    SparseStep get_sparse_step() const { return sstep; }
    DenseStep get_dense_step() const { return dstep; }

    const frontier & get_frontier() const { return cfrontier; }

private:
    const frontier & cfrontier;
    SparseStep       sstep;
    DenseStep        dstep;
};

/***********************************************************************
 * Auxiliary constructor functions for step_sdchoice<>
 ***********************************************************************/
template<typename SparseStep, typename DenseStep>
constexpr auto make_step_sdchoice(
    const frontier & f, SparseStep s, DenseStep d ) {
    return step_sdchoice<SparseStep,DenseStep>( f, s, d );
}

template<typename Operator>
auto make_step_sdchoice( const frontier & f, Operator op ) {
    using frontier_spec
	= expr::determine_frontier_vmap<VID,Operator>;
    constexpr unsigned short VL = frontier_spec::VL;

    auto d_step = make_step_vmap_dense<Operator::is_scan>( op );

// TODO: sparse is_scan
    auto s_step = make_step_vmap_sparse( op, f );
	
    return make_step_sdchoice( f, s_step, d_step );
}

template<typename Operator>
auto make_step_sdchoice_scan( const frontier & f, Operator op ) {
    using frontier_spec
	= expr::determine_frontier_vmap<VID,Operator>;
    constexpr unsigned short VL = frontier_spec::VL;

    static_assert( Operator::is_scan, "enforce variant" );
    auto d_step = make_step_vmap_dense<true>( op );

// TODO: sparse is_scan
    auto s_step = make_step_vmap_sparse( op, f );
	
    return make_step_sdchoice( f, s_step, d_step );
}

/***********************************************************************
 * Analysing read/write-after-write dependencies on steps and choice steps.
 * An xAW is diagnosed if an xAW occurs on any execution path.
 * An xAW is not diagnosed for vmap and emap_dense because it is assumed
 * that any dependent operations will be executed back-to-back by the
 * same thread. As such there is an xAW that will be handled at execution time.
 * An xAW is diagnosed when the output of a step_vscan is used, because
 * materialisation needs to take place before the output is available.
 * Dependencies are diagnosed on the basis of the AID field of an
 * expr::array_ro (or variations).
 *
 * The step1 templates pivot over parallel options in the first argument
 * (the x=R/W); the step2 templates pivot over parallel options in the
 * second argument (W).
 ***********************************************************************/
namespace detail {
template<typename xStep1, typename wStep2>
struct step_has_xAW;


// wStep2 has to be different from step_vchoice<>
template<typename xChoiceStep1, typename wStep2, typename = void>
struct choice_step1_has_xAW;

template<typename xStep1, typename wStep2>
struct choice_step1_has_xAW<
    xStep1,wStep2,
    typename std::enable_if<!is_step_vchoice<xStep1>::value
			    && !is_step_sdchoice<xStep1>::value
			    && !is_step_emap_dense3<xStep1>::value>::type>
    : step_has_xAW<xStep1,wStep2> { };

template<typename wStep2>
struct choice_step1_has_xAW<step_vchoice<>,wStep2,void> : std::false_type { };

template<typename Step, typename wStep2, typename... Steps>
struct choice_step1_has_xAW<step_vchoice<Step,Steps...>,wStep2,void> {
    static constexpr bool value = step_has_xAW<Step,wStep2>::value
	|| choice_step1_has_xAW<step_vchoice<Steps...>,wStep2>::value;
};

template<typename wStep2, typename SStep, typename DStep>
struct choice_step1_has_xAW<step_sdchoice<SStep,DStep>,wStep2,void> {
    static constexpr bool value = step_has_xAW<SStep,wStep2>::value
	|| step_has_xAW<DStep,wStep2>::value;
};

#if 0
template<typename GT, typename S1, typename S2, typename S3, typename wStep2>
struct choice_step1_has_xAW<step_emap_dense3<GT,S1,S2,S3>,wStep2,void> {
    static constexpr bool value = step_has_xAW<S1,wStep2>::value
	|| step_has_xAW<S2,wStep2>::value
	|| step_has_xAW<S3,wStep2>::value;
};
#endif

template<typename xStep1, typename wChoiceStep2>
struct choice_step2_has_xAW;

template<typename xStep1>
struct choice_step2_has_xAW<xStep1,step_vchoice<>> : std::false_type { };

template<typename xStep1, typename Step, typename... Steps>
struct choice_step2_has_xAW<xStep1,step_vchoice<Step,Steps...>> {
    static constexpr bool value = choice_step1_has_xAW<xStep1,Step>::value
	|| choice_step2_has_xAW<xStep1,step_vchoice<Steps...>>::value;
};

#if 0
template<typename GT, typename S1, typename S2, typename S3, typename xStep1>
struct choice_step2_has_xAW<xStep1,step_emap_dense3<GT,S1,S2,S3>> {
    static constexpr bool value = choice_step1_has_xAW<xStep1,S1>::value
	|| choice_step1_has_xAW<xStep1,S2>::value
	|| choice_step1_has_xAW<xStep1,S3>::value;
};
#endif

// Main template
template<typename Step1>
struct step_has_xAW<Step1,step_empty> : std::false_type { };

template<typename Step1, typename Operator>
struct step_has_xAW<Step1,step_vmap_dense<false,Operator>>
    : std::false_type { };

template<typename Step1, typename Operator>
struct step_has_xAW<Step1,step_vmap_sparse<Operator>> : std::false_type { };

template<typename... Step, typename Operator>
struct step_has_xAW<step_vchoice<Step...>,step_vmap_dense<true,Operator>>
    : choice_step1_has_xAW<step_vchoice<Step...>,step_vmap_dense<true,Operator>>
{ };

template<typename Step1, typename Operator>
struct step_has_xAW<Step1,step_vmap_dense<true,Operator>>
    : expr::reads_accum<typename Step1::vexpr_type,
			typename step_vmap_dense<true,Operator>::accum_type> { };

// A step_emap_dense applies only to CSC and COO dense traversals (the argument
// applies to dense CSR as well). The scan may be merged as long as the CSC/COO
// template code keeps in mind that
// (i)  the scan may require data produced during the edgemap operation
// (ii) consistency of caches and access thereto is maintained across the
//      ASTs for the edgemap and scanning vertexmap.
template<typename Step, typename GraphType, typename Operator,
	 typename EMapConfig, bool FTrue>
struct step_has_xAW<Step,step_emap_dense<GraphType,Operator,EMapConfig,FTrue>>
    : std::false_type { };

template<typename Step, typename... Steps>
struct step_has_xAW<Step,step_vchoice<Steps...>>
    : choice_step2_has_xAW<Step,step_vchoice<Steps...>> { };

template<typename Step, typename SStep, typename DStep>
struct step_has_xAW<Step,step_sdchoice<SStep,DStep>>
    : choice_step2_has_xAW<Step,step_sdchoice<SStep,DStep>> { };

#if 0
template<typename Step, typename GT, typename S1, typename S2, typename S3>
struct step_has_xAW<Step,step_emap_dense3<GT,S1,S2,S3>>
    : choice_step2_has_xAW<Step,step_emap_dense3<GT,S1,S2,S3>> { };
#endif

} // namespace detail

/***********************************************************************
 * Merging steps.
 * We don't merge sparse and dense steps.
 * Choice steps are merged by considering all combinations.
 * Steps should only be merged when there are no dependencies between
 * them.
 ***********************************************************************/
template<typename Step>
auto merge_step( Step step, step_empty ) {
    return step;
}

template<bool IsScan1, typename Op1, bool IsScan2, typename Op2>
auto merge_step( step_vmap_dense<IsScan1,Op1> a,
		 step_vmap_dense<IsScan2,Op2> b ) {
    return make_step_vmap_dense<IsScan1 || IsScan2>(
	append_vop2vop( a.get_operator(), b.get_operator() )
	);
}

// TODO: merge_step( sparse, sparse ) case missing. Difficulty: what if
//       frontiers differ?

template<typename SStep, typename DStep,
	 typename GraphType, typename Operator, typename EMapConfig, bool FTrue>
auto merge_step( step_sdchoice<SStep,DStep> xstep,
		 step_emap_dense<GraphType,Operator,EMapConfig,FTrue> wstep ) {
    // Drop sparse control flow path in this combination of paths
    // return merge_step( xstep.get_dense_step(), wstep );
    return make_step_dchoice(
	merge_step( xstep.get_true_step(), wstep ),
	merge_step( xstep.get_dense_step(), wstep ) );
}

template<typename Step, typename GraphType, typename Operator,
	 typename EMapConfig, bool FTrue>
auto merge_step( Step xstep,
		 step_emap_dense<GraphType,Operator,EMapConfig,FTrue> wstep,
		 typename std::enable_if<!is_step_vchoice<Step>::value && !Step::has_frontier>::type *
		 = nullptr ) {
    // Only if xstep has no frontier!
    return wstep.append_post( xstep );
}

template<typename Step1, std::size_t... Ns, typename... Steps>
auto merge_step_vchoice( Step1 step1,
			 std::index_sequence<Ns...>,
			 step_vchoice<Steps...> step2 ) {
    // This might actually work out for two choice steps if we additionally
    // define merge_step( step_vchoice<...>, non-choice-step ) { }
    // The resulting construct will have nested choice steps, but that is
    // workable
    static_assert( !is_step_vchoice<Step1>::value,
		   "assumes step1 is not a choice step" );
    return make_step_vchoice( step2.get_choice(),
			      merge_step( step1,
					  step2.template get_step<Ns>() )... );
}

template<typename Step2, std::size_t... Ns, typename... Steps>
auto merge_step_vchoice( std::index_sequence<Ns...>,
			 step_vchoice<Steps...> step1,
			 Step2 step2 ) {
    return make_step_vchoice( step1.get_choice(),
			      merge_step( step1.template get_step<Ns>(),
					  step2 )... );
}

template<typename... Steps1, typename Step2>
auto merge_step( step_vchoice<Steps1...> step1, Step2 step2 ) {
    return merge_step_vchoice( std::make_index_sequence<sizeof...(Steps1)>(),
			       step1,
			       step2 );
}

template<typename Step1, typename... Steps>
auto merge_step( Step1 step1, step_vchoice<Steps...> step2,
		 typename std::enable_if<!is_step_vchoice<Step1>::value>::type *
		 = nullptr ) {
    return merge_step_vchoice( step1,
			       std::make_index_sequence<sizeof...(Steps)>(),
			       step2 );
}

template<typename Step1, typename SStep, typename DStep>
auto merge_step( Step1 step1, step_sdchoice<SStep,DStep> step2,
		 typename std::enable_if<!is_step_sdchoice<Step1>::value>::type *
		 = nullptr ) {
    // TODO: assumption is made that steps have same frontier!
    return make_step_sdchoice( step2.get_frontier(),
			       merge_step( step1, step2.get_sparse_step() ),
			       merge_step( step1, step2.get_dense_step() ) );
}

template<typename SStep1, typename DStep1, typename SStep2, typename DStep2>
auto merge_step( step_sdchoice<SStep1,DStep1> step1,
		 step_sdchoice<SStep2,DStep2> step2 ) {
    // Merge corresponding versions only
    // TODO: It is difficult to merge the sparse steps when they have
    //       a different frontier. However, it would be possible to handle
    //       the case where the output frontier of the first is used as input
    //       to the second.
    //       To by-pass issues, vertex_map with a frontier materializes
    //       prior computations.
    assert( &step1.get_frontier() == &step2.get_frontier() );
    return make_step_sdchoice( step1.get_frontier(),
			       merge_step( step1.get_sparse_step(),
					   step2.get_sparse_step() ),
			       merge_step( step1.get_dense_step(),
					   step2.get_dense_step() ) );
}

#if 0
template<typename Step1, typename GT, typename S1, typename S2, typename S3>
auto merge_step( Step1 step1,
		 step_emap_dense3<GT,S1,S2,S3> step2 ) {
    return make_step_emap_dense3(
	step2.get_graph(),
	step2.get_traversal(),
	merge_step( step1, step2.get_step1() ),
	merge_step( step1, step2.get_step2() ),
	merge_step( step1, step2.get_step3() ) );
}
#endif

/***********************************************************************
 * A vertex/frontier filter operation
 ***********************************************************************/
template<typename GraphType, typename Operator>
class vmap_wrap_filter {
public:
    using self_type = vmap_wrap_filter<GraphType,Operator>;
    using vexpr_type = decltype( ((Operator*)nullptr)->operator()( expr::value<simd::ty<VID,1>,expr::vk_vid>() ) );

    // static constexpr bool is_scan = true; // always a scan (on frontier count)
    static constexpr vmap_kind vkind = vmap_filter;
    static constexpr bool is_filter = true;
    static constexpr bool is_scan = true;

    vmap_wrap_filter( const GraphType & G, frontier & f, Operator op )
	: m_G( G ), m_f( f ), m_op( op ) {
	// Set target frontier to empty. In case execution is filtered by
	// an empty frontier, we won't have an opportunity to do so.
	m_f = frontier::empty();
    }

    template<typename VIDTy>
    auto operator() ( VIDTy vid ) {
	return m_op( vid );
    }

    template<typename map_type0>
    struct ptrset {
	using map_type = typename expr::ast_ptrset::ptrset_list<
	    map_type0,vexpr_type>::map_type;

	template<typename MapTy>
	static void initialize( MapTy & map, const self_type & op ) {
	    auto v = expr::value<simd::ty<VID,1>,expr::vk_vid>();

	    expr::ast_ptrset::ptrset_list<map_type0,vexpr_type>
		::initialize( map, op.m_op( v ) );
	}
    };

    const GraphType & m_G;
    frontier & m_f;
    Operator m_op;
};

/***********************************************************************
 * Wrap a lambda/Operator in a struct with a bool is_scan member.
 ***********************************************************************/
template<bool IsScan, typename T>
struct vmap_wrap_scan : public T {
    using self_type = vmap_wrap_scan<IsScan,T>;
    using vexpr_type = decltype( ((T*)nullptr)->operator()( expr::value<simd::ty<VID,1>,expr::vk_vid>() ) );
    
    // static constexpr bool is_scan = IsScan;
    static constexpr vmap_kind vkind = IsScan ? vmap_scan : vmap_map;
    static constexpr bool is_filter = false;
    static constexpr bool is_scan = IsScan;

    vmap_wrap_scan( T t ) : T( t ) { }

/*
    template<frontier_type ftype, unsigned short VL, typename VIDTy>
    auto operator() ( VIDTy vid ) {
	return T::operator() ( vid );
    }
*/

    template<typename map_type0>
    struct ptrset {
	using map_type = typename expr::ast_ptrset::ptrset_list<
	    map_type0,vexpr_type>::map_type;

	static_assert( expr::map_contains_v<(unsigned)expr::aid_key(expr::array_aid(expr::aid_eweight)),map_type0>, "check 0" );
	static_assert( expr::map_contains_v<(unsigned)expr::aid_key(expr::array_aid(expr::aid_eweight)),map_type>, "check 1" );

	template<typename MapTy>
	static void initialize( MapTy & map, const self_type & op ) {
	    auto v = expr::value<simd::ty<VID,1>,expr::vk_vid>();

	    expr::ast_ptrset::ptrset_list<map_type0,vexpr_type>
		::initialize( map, op.operator () ( v ) );
	}
    };
};

template<typename Operator>
inline
vmap_wrap_scan<true,Operator>
wrap_scan( Operator op ) {
    return vmap_wrap_scan<true,Operator>( op );
}

template<typename GraphType, typename Operator>
inline
vmap_wrap_filter<GraphType,Operator>
wrap_scan_filter( const GraphType & G, frontier & f, Operator op ) {
    return vmap_wrap_filter<GraphType,Operator>( G, f, op );
}

template<typename Operator>
inline
vmap_wrap_scan<false,Operator>
wrap_map( Operator op ) {
    return vmap_wrap_scan<false,Operator>( op );
}


/***********************************************************************
 * Lazy executor: collects a number of vmap/vscan/emap steps and
 * merges their computations.
 * The lazy_executor is a convenience class to hold a computational step,
 * append further steps to it prior to evaluation, or execute it by calling
 * #materialize().
 ***********************************************************************/
template<class Step>
class lazy_executor;

/**
 * \brief Create a lazy_executor holding the specified step
 *
 * \param part partitioner object
 * \param s computational step
 * \return a lazy_executor for the step #s
 */
template<class Step>
constexpr lazy_executor<Step>
make_lazy_executor( const partitioner & part, Step s ) {
    return lazy_executor<Step>( part, s );
}

static constexpr
lazy_executor<step_empty> make_lazy_executor( const partitioner & part );

/**
 * \brief A version of the lazy_executor class for an empty step, i.e., a no-op
 */
template<>
class lazy_executor<step_empty> {
public:
    /**
     * \brief Construct a lazy_executor with empty step
     * \param part graph partitioner object
     */
    constexpr lazy_executor( const partitioner & part ) : m_part( part ) { }
    
    /**
     * \brief Retrieve the step held in the object
     * \return a computational step of type #step_empty
     */
    step_empty step() const { return step_empty(); }

    /**
     * \brief Materialize the data. This is a noop.
     */
    constexpr void materialize() { }

    /**
     * \brief Append an edge_map step to the executor
     * \param G Graph to map operation over
     * \param op Operator to apply
     * \param gtk Graph traversal type (push/pull/ireg)
     * \return An executor holding the edge_map step.
     */
    template<typename EMapConfig, typename GraphType, typename Operator>
    constexpr auto edge_map( const GraphType & G, Operator op,
			     graph_traversal_kind gtk ) {
	return make_lazy_executor( m_part,
				   make_step_emap_dense<EMapConfig>( G, op, gtk ) );
    }

    /**
     * \brief Append an edge_map step to the executor
     * \param G Graph to map operation over
     * \param of #frontier to filter by
     * \param nf #frontier to record
     * \param op Operator to apply
     * \return An executor holding the edge_map step.
     */
    template<typename EMapConfig, typename GraphType, typename Operator>
    constexpr auto edge_map( const GraphType & G, const frontier & of,
			     frontier &nf, Operator op ) {
	return make_lazy_executor( m_part,
				   make_step_emap_dense<EMapConfig>( G, of, nf, op ) );
    }

    /**
     * \brief Append a vertex_map step to the executor
     * \param op Operator to apply
     * \return An executor holding the vertex_map step.
     */
    template<typename Operator>
    constexpr auto vertex_map( Operator op ) {
	return make_lazy_executor( m_part, make_step_vmap_dense<false>( wrap_map( op ) ) );
    }

    /**
     * \brief Append a vertex_map step to the executor
     * \param f #frontier to filter by
     * \param op Operator to apply
     * \return An executor holding the vertex_map step.
     */
    template<typename Operator>
    constexpr auto vertex_map( const frontier & f, Operator op ) {
	return make_lazy_executor( m_part, make_step_sdchoice( f, wrap_map( op ) ) );
    }

    /**
     * \brief Append a vertex_filter step to the executor
     * \param G graph data structure to retrieve degrees from
     * \param f #frontier to filter by
     * \param nf filtered #frontier that is created
     * \param op Operator to apply
     * \return An executor holding the vertex_filter step.
     */
    template<typename GraphType, typename Operator>
    auto vertex_filter( const GraphType & G, const frontier & f,
			frontier & nf, Operator op ) {
	return make_lazy_executor(
	    m_part,
	    make_step_sdchoice( f, wrap_scan_filter( G, nf, op ) ) );
    }

    /**
     * \brief Append a vertex_scan step to the executor
     * \param op Operator to apply
     * \return An executor holding the vertex_scan step.
     */
    template<typename Operator>
    constexpr auto vertex_scan( Operator op ) {
	return make_lazy_executor( m_part, make_step_vmap_dense<true>( wrap_scan( op ) ) );
    }

    /**
     * \brief Append a vertex_scan step to the executor
     * \param f #frontier to filter by
     * \param op Operator to apply
     * \return An executor holding the vertex_scan step.
     */
    template<typename Operator>
    constexpr auto vertex_scan( const frontier & f, Operator op ) {
	return make_lazy_executor(
	    m_part,
	    make_step_sdchoice_scan( f, wrap_scan( op ) ) );
    }

    /**
     * \brief Append a vertex_map step to the executor without dependence checking
     * This is for internal use only
     * \param op Operator to apply
     * \return An executor holding the vertex_map step.
     */
    template<typename Operator>
    constexpr auto vertex_map_unsafe( Operator op ) {
	return vertex_map( wrap_map( op ) );
    }

    /**
     * \brief Append a vertex_scan step to the executor without dependence checking
     * This is for internal use only
     * \param op Operator to apply
     * \return An executor holding the vertex_scan step.
     */
    template<typename Operator>
    constexpr auto vertex_scan_unsafe( Operator op ) {
	return vertex_scan( wrap_scan( op ) );
    }

    /**
     * \brief Append a flat_emap step to the executor.
     *
     * For the time being, it is not attempted to merge this operation with
     * other steps. Merging is meaningful only with edge_map and flat_edge_map
     * steps.
     *
     * \param op Operator to apply
     * \return An executor holding the flat_edge_map step.
     */
    template<typename Operator>
    constexpr auto flat_edge_map( Operator op ) {
	auto step = make_step_flat_emap( op );
	step.execute( m_part );
	return make_lazy_executor( m_part );
    }

private:
    const partitioner & m_part; //!< graph partitioning object
};

/**
 * \brief A version of the lazy_executor class for a non-empty step
 */
template<class Step>
class lazy_executor {
public:
    /**
     * \brief Construct a lazy_executor with empty step
     * \param part graph partitioner object
     * \param step computational step
     */
    lazy_executor( const partitioner & part, Step step )
	: m_part( part ), m_step( step ) {
	static_assert( !std::is_same<Step,step_empty>::value,
		       "Template designed for non-empty step" );
    }

    /**
     * \brief Retrieve the step held in the object
     * \return a computational step of type #step_empty
     */
    Step step() const { return m_step; }

    /**
     * \brief Materialize the data by executing the step
     */
    __attribute__((noinline))
    void materialize() {
	m_step.execute( m_part );
    }

    /**
     * \brief Append a vertex_map step to the executor
     * This operation checks if materialisation should occur.
     * In particular, when the result of a vscan is used, then the vscan
     * needs to be materialized. The vscan results are summarized in
     * #m_step.accum.
     * \param op Operator to apply
     * \return An executor holding the sequence of #m_step and a new vertex_map step.
     */
    template<typename Operator>
    auto vertex_map( Operator op ) {
	if constexpr ( std::is_class<Operator>::value && Step::has_frontier ) {
	    materialize();
	    return make_lazy_executor(
		m_part,
		make_step_vmap_dense<false>( wrap_map( op ) ) );
	} else if constexpr( std::is_class<Operator>::value && !Step::has_frontier ) {
	    return enqueue_step( make_step_vmap_dense<false>( wrap_map( op ) ) );
	}
    }

    /**
     * \brief Append a vertex_map step to the executor
     * This operation checks if materialisation should occur.
     * In particular, when the result of a vscan is used, then the vscan
     * needs to be materialized. The vscan results are summarized in
     * #m_step.accum.
     * \param f #frontier to filter by
     * \param op Operator to apply
     * \return An executor holding the sequence of #m_step and a new vertex_map step.
     */
    template<typename Operator>
    auto vertex_map( const frontier & f, Operator op ) {
	// return enqueue_step( make_step_sdchoice( f, wrap_map( op ) ) );
	// TODO: It is hard to merge steps when the frontier is sparse.
	materialize(); // Calculate all of m_step
	return make_lazy_executor( m_part,
				   make_step_sdchoice( f, wrap_map( op ) ) );
    }

    /**
     * \brief Append a vertex_filter step to the executor
     * \param G graph data structure to retrieve degrees from
     * \param f #frontier to filter by
     * \param nf filtered #frontier that is created
     * \param op Operator to apply
     * \return An executor holding the vertex_filter step.
     */
    template<typename GraphType, typename Operator>
    auto vertex_filter( const GraphType & G, const frontier & f,
			frontier & nf, Operator op ) {
	// TODO: It is hard to merge steps when the frontier is sparse.
	materialize(); // Calculate all of m_step
	return make_lazy_executor(
	    m_part,
	    make_step_sdchoice( f, wrap_scan_filter( G, nf, op ) ) );
    }

    /**
     * \brief Append a vertex_scan step to the executor
     * This operation checks if materialisation should occur.
     * In particular, when the result of a vscan is used, then the vscan
     * needs to be materialized. The vscan results are summarized in
     * #m_step.accum.
     * \param op Operator to apply
     * \return An executor holding the sequence of #m_step and a new vertex_scan step.
     */
    template<typename Operator>
    auto vertex_scan( Operator op ) {
	if constexpr ( std::is_class<Operator>::value && Step::has_frontier ) {
	    materialize(); // Calculate all of m_step
	    return make_lazy_executor(
		m_part,
		make_step_vmap_dense<true>( wrap_scan( op ) ) );
	} else if constexpr( std::is_class<Operator>::value && !Step::has_frontier ) {
	    return enqueue_step( make_step_vmap_dense<true>( wrap_scan( op ) ) );
	}
    }

    /**
     * \brief Append a vertex_map step to the executor without dependency checking
     * This operation does not check if materialisation should occur.
     * It is for internal use only.
     * \param op Operator to apply
     * \return An executor holding the sequence of #m_step and a new vertex_map step.
     */
    template<typename Operator>
    auto vertex_map_unsafe( Operator op ) {
	auto step = make_step_vmap_dense<false>( wrap_map( op ) );
	return make_lazy_executor( m_part, merge_step( step, m_step ) );
    }

    /**
     * \brief Append a vertex_scan step to the executor without dependency checking
     * This operation does not check if materialisation should occur.
     * It is for internal use only.
     * \param op Operator to apply
     * \return An executor holding the sequence of #m_step and a new vertex_scan step.
     */
    template<typename Operator>
    auto vertex_scan_unsafe( Operator op ) {
	auto step = make_step_vmap_dense<true>( wrap_scan( op ) );
	return make_lazy_executor( m_part, merge_step( step, m_step ) );
    }
    
    
private:
    /**
     * \brief check for dependencies and concatenate steps
     * New step reads/writes accumulator calculated in previous steps.
     * Materialize all data (i.e., execute), then start a new sequence
     * of steps.
     * \param step New step to append
     * \return remaining step to be executed
     */
    template<typename NewStep>
    auto enqueue_step( NewStep step,
		       typename std::enable_if<detail::step_has_xAW<NewStep,Step>::value>::type * = nullptr ) {
	materialize(); // Calculate all of m_step
	return make_lazy_executor( m_part, step ); // m_step is done
    }

    /**
     * \brief check for dependencies and concatenate steps
     * New step does not read/write accumulator calculated in previous steps.
     * Merge the step into the calculation already queued up
     * \param step New step to append
     * \return remaining step to be executed
     */
    template<typename NewStep>
    auto enqueue_step( NewStep step,
		       typename std::enable_if<!detail::step_has_xAW<NewStep,Step>::value>::type * = nullptr ) {
#if ALWAYS_MATERIALIZE
	materialize(); // Calculate all of m_step
	return make_lazy_executor( m_part, step ); // m_step is done
#else
	return make_lazy_executor( m_part, merge_step( step, m_step ) );
#endif // ALWAYS_MATERIALIZE
    }

private:
    Step m_step; //!< the computational step
    const partitioner & m_part; //!< graph partitioning object
};

/**
 * \brief create lazy_executor with empty step
 * \param part graph partitioner object
 * \return empty lazy_executor
 */
static constexpr
lazy_executor<step_empty> make_lazy_executor( const partitioner & part ) {
    return lazy_executor<step_empty>( part );
}

/**
 * \brief create lazy_executor making a choice of one of n steps
 * \param part graph partitioner object
 * \param c conditional choice of which step to execute
 * \param e... steps
 * \return lazy executor holding the conditional step
 */
template<typename... Executor>
constexpr auto
make_lazy_executor( const partitioner & part, int c, Executor... e ) {
    // static_assert( false, "can this be phased out, together with vchoice?" );
    return make_lazy_executor( part, make_step_vchoice( c, e.step()... ) );
}

#if 0
/**
 * \brief create lazy_executor making a choice of sparse or dense frontiers
 * \param part graph partitioner object
 * \param f #frontier to apply
 * \param s step to execute in case of a sparse frontier
 * \param d step to execute in case of a dense frontier
 * \return lazy executor holding the conditional step
 */
template<typename SparseStep, typename DenseStep>
constexpr auto make_edge_map_executor(
    const partitioner & part, frontier & f, SparseStep s, DenseStep d ) {
    return make_lazy_executor(
	part, make_step_sdchoice( f, s.step(), d.step() ) );
}
/**
 * \brief create lazy_executor selecting edgemap variations
 * \param part graph partitioner object
 * \param kind choice of step
 * \param push step to execute if kind is push
 * \param pull step to execute if kind is pull
 * \param ireg step to execute if kind is ireg
 * \return lazy executor holding the conditional step
 */
template<typename GraphType, typename PushStep, typename PullStep,
	 typename IRegStep>
constexpr auto make_edge_map_executor(
    const GraphType & G, graph_traversal_kind kind,
    PushStep push, PullStep pull, IRegStep ireg ) {
    return make_lazy_executor(
	G.get_partitioner(), make_step_emap_dense3(
	    G, kind, push.step(), pull.step(), ireg.step() ) );
}
#endif

/**
 * \brief Maintain the copies of two vertex properties
 *
 * \param part Graph partitioner object
 * \param change Frontier of changed elements
 * \param source Vertex property as source for copy
 * \param dest Vertex property as destination for copy
 */
template<typename DstTy, typename SrcTy>
auto maintain_copies( const partitioner & part, frontier & change,
		      DstTy & dst, SrcTy & src ) {
    bool do_all = change.getType() == frontier_type::ft_unbacked
	|| change.getType() == frontier_type::ft_true
	|| change.nActiveVertices() > change.nVertices() / 4;

    if( do_all ) {
	make_lazy_executor( part )
	    .vertex_map( [&]( auto v ) {
		return dst[v] = src[v];
	    } )
	    .materialize();
    } else {
	make_lazy_executor( part )
	    .vertex_map( change, [&]( auto v ) {
		return dst[v] = src[v];
	    } )
	    .materialize();
    }
}


#endif // GRAPHGRIND_DSL_VERTEXMAP_H
