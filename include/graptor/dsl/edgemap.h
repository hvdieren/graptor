// -*- c++ -*-
#ifndef GRAPTOR_DSL_EDGEMAP_H
#define GRAPTOR_DSL_EDGEMAP_H

#include <cstdio>
#include <cstddef>
#include <cassert>
#include <type_traits>
#include <algorithm>
#include <iostream>

#include <immintrin.h>

#include "graptor/frontier/frontier.h"
#include "graptor/target/vector.h"
#include "graptor/dsl/simd_vector.h"
#include "graptor/dsl/ast.h"
#include "graptor/dsl/vertexmap.h"

#include "graptor/dsl/comp/rewrite_redop_to_store.h"
#include "graptor/dsl/comp/rewrite_internal.h"
#include "graptor/dsl/comp/is_idempotent.h"
#include "graptor/dsl/comp/is_benign_race.h"

#include "graptor/dsl/emap/utils.h"
#include "graptor/dsl/emap/emap_scan.h"

#ifndef EMAP_BLOCK_SIZE
#define EMAP_BLOCK_SIZE 2048
#endif

enum update_method {
    um_none,
    um_list,
    um_list_must_init,
    um_list_must_init_unique,
    um_flags_only,
    um_flags_only_unique
};

#include "graptor/dsl/emap/fusion.h"
#include "graptor/dsl/emap/edgechunk.h"

#include "graptor/dsl/emap/GraphCOO.h"
#include "graptor/dsl/emap/GraphCSx_csc.h"
#include "graptor/dsl/emap/GraphCSx_csr.h"

// #include "graptor/dsl/emap/scalar.h"
// #include "graptor/dsl/emap/VEBOSlimSell.h"
// #include "graptor/dsl/emap/GraptorCSCVReduceCached.h"
// #include "graptor/dsl/emap/GraptorCSCVReduceNotCached.h"
#include "graptor/dsl/emap/GraptorCSCDataParCached.h"
#include "graptor/dsl/emap/GraptorCSCDataParNotCached.h"
#include "graptor/dsl/emap/GraptorCSRVPushCached.h"
#include "graptor/dsl/emap/GraptorCSRVPushNotCached.h"
#include "graptor/dsl/emap/GraptorCSRDataParCached.h"
#include "graptor/dsl/emap/GraptorCSRDataParNotCached.h"

#include "graptor/frontier/impl.h"
#include "graptor/api/fusion.h"

template<typename EMapConfig, typename GraphType, typename EdgeOperator>
static inline void emap_push(
    const GraphType &, EdgeOperator &, const partitioner & ) {
    assert( 0 && "ERROR: default push operator for specific graph type "
	    "has not been overridden\n" );
    abort();
}

template<typename EMapConfig, typename GraphType, typename EdgeOperator>
static inline void emap_pull(
    const GraphType &, EdgeOperator &, const partitioner & ) {
    assert( 0 && "ERROR: default pull operator for specific graph type "
	    "has not been overridden\n" );
    abort();
}

template<typename EMapConfig, typename GraphType, typename EdgeOperator>
static inline void emap_ireg(
    const GraphType &, EdgeOperator &, const partitioner & ) {
    assert( 0 && "ERROR: default irregular operator for specific graph type "
	    "has not been overridden\n" );
    abort();
}

template<typename GraphType, typename Operator, typename EMapConfig,
	 graph_traversal_kind gtk, bool is_priv>
static std::ostream & emap_report_dyn( std::ostream & os, Operator op ) {
    static bool printed = false;
    if( printed ) 
	return os;

    printed = true;

    return EMapConfig::report( std::cout )
	<< " Operator { is_scan: " << Operator::is_scan
	<< ", is_idempotent: " << expr::is_idempotent_op<Operator>::value
	<< ", fusion: " << api::has_fusion_op_v<Operator>
	<< " } config { is_parallel: " << op.get_config().is_parallel()
	<< ", maxVL: " << op.get_config().max_vector_length()
	<< ", s/d threshold: " << op.get_config().get_threshold()
	<< " } traversal: " << gtk << ( is_priv ? ",privatized" : ",atomic" )
	<< "\n";
}

template<typename GraphType, typename EdgeOperator,typename EMapConfig>
void step_emap_dense<GraphType,EdgeOperator,EMapConfig,true>::execute(
    const partitioner & part ) {
    // Not sure what it means here if part != m_G.get_partitioner()
    assert( &part == &m_G.get_partitioner() );

    // TODO: compilation time can be saved if m_gtk is made a template
    //       parameter, as it will avoid many instantiations.
    switch( m_gtk ) {
    case graph_traversal_kind::gt_push:
    {
	using push_cfg = typename config::push;
	static constexpr bool is_priv
	    = GraphType::is_privatized( graph_traversal_kind::gt_push );
	emap_report_dyn<GraphType,EdgeOperator,push_cfg,
			graph_traversal_kind::gt_push,is_priv>(
			    std::cout, m_op );
	// Frontier is handled by push traversal code template
	if( m_op.is_true_src_frontier() ) {
	    if( m_op.is_true_dst_frontier() ) {
		auto push_op = m_op.template
		    variant<graph_traversal_kind::gt_push,true,true,is_priv,config>(
			m_G.getOutDegree() );
		emap_push<push_cfg>( m_G, push_op, part );
	    } else {
		auto push_op = m_op.template
		    variant<graph_traversal_kind::gt_push,true,false,is_priv,config>(
			m_G.getOutDegree() );
		emap_push<push_cfg>( m_G, push_op, part );
	    }
	} else {
	    if( m_op.is_true_dst_frontier() ) {
		auto push_op = m_op.template
		    variant<graph_traversal_kind::gt_push,false,true,is_priv,config>(
			m_G.getOutDegree() );
		emap_push<push_cfg>( m_G, push_op, part );
	    } else {
		auto push_op = m_op.template
		    variant<graph_traversal_kind::gt_push,false,false,is_priv,config>(
			m_G.getOutDegree() );
		emap_push<push_cfg>( m_G, push_op, part );
	    }
	}
	break;
    }
    case graph_traversal_kind::gt_pull:
    {
	using pull_cfg = typename config::pull;
	static constexpr bool is_priv
	    = GraphType::is_privatized( graph_traversal_kind::gt_pull );
	emap_report_dyn<GraphType,EdgeOperator,pull_cfg,
			graph_traversal_kind::gt_pull,is_priv>(
			    std::cout, m_op );
	if( m_op.is_true_src_frontier() ) {
	    if( m_op.is_true_dst_frontier() ) {
		auto pull_op = m_op.template
		    variant<graph_traversal_kind::gt_pull,true,true,is_priv,config>(
			m_G.getOutDegree() );
		emap_pull<pull_cfg>( m_G, pull_op, part );
	    } else {
		auto pull_op = m_op.template
		    variant<graph_traversal_kind::gt_pull,true,false,is_priv,config>(
			m_G.getOutDegree() );
		emap_pull<pull_cfg>( m_G, pull_op, part );
	    }
	} else {
	    if( m_op.is_true_dst_frontier() ) {
		auto pull_op = m_op.template
		    variant<graph_traversal_kind::gt_pull,false,true,is_priv,config>(
			m_G.getOutDegree() );
		emap_pull<pull_cfg>( m_G, pull_op, part );
	    } else {
		auto pull_op = m_op.template
		    variant<graph_traversal_kind::gt_pull,false,false,is_priv,config>(
			m_G.getOutDegree() );
		emap_pull<pull_cfg>( m_G, pull_op, part );
	    }
	}
	break;
    }
    case graph_traversal_kind::gt_ireg:
    {
	// assert( 0 && "revisit" );
	using ireg_cfg = typename config::ireg;
	static constexpr bool is_priv
	    = GraphType::is_privatized( graph_traversal_kind::gt_ireg );
	emap_report_dyn<GraphType,EdgeOperator,ireg_cfg,
			graph_traversal_kind::gt_ireg,is_priv>(
			    std::cout, m_op );
	if( m_op.is_true_src_frontier() ) {
	    if( m_op.is_true_dst_frontier() ) {
		auto ireg_op = m_op.template
		    variant<graph_traversal_kind::gt_ireg,true,true,is_priv,config>(
			m_G.getOutDegree() );
		emap_ireg<ireg_cfg>( m_G, ireg_op, part );
	    } else {
		auto ireg_op = m_op.template
		    variant<graph_traversal_kind::gt_ireg,true,false,is_priv,config>(
			m_G.getOutDegree() );
		emap_ireg<ireg_cfg>( m_G, ireg_op, part );
	    }
	} else {
	    if( m_op.is_true_dst_frontier() ) {
		auto ireg_op = m_op.template
		    variant<graph_traversal_kind::gt_ireg,false,true,is_priv,config>(
			m_G.getOutDegree() );
		emap_ireg<ireg_cfg>( m_G, ireg_op, part );
	    } else {
		auto ireg_op = m_op.template
		    variant<graph_traversal_kind::gt_ireg,false,false,is_priv,config>(
			m_G.getOutDegree() );
		emap_ireg<ireg_cfg>( m_G, ireg_op, part );
	    }
	}
	break;
    }
    case graph_traversal_kind::gt_sparse:
	// Reports as not privatized, however, depends on traversal direction
	emap_report_dyn<GraphType,EdgeOperator,typename EMapConfig::scalar,
			graph_traversal_kind::gt_sparse,false>( std::cout, m_op );
	// nothing to do
	break;
    default:
	UNREACHABLE_CASE_STATEMENT;
    }

    // If unbacked frontier generated that proves sparse, then
    // materialise frontier using record-method, if possible.
    if constexpr ( EdgeOperator::defines_frontier )
	materialize_frontier( part );
}

template<typename GraphType, typename EdgeOperator,typename EMapConfig>
void step_emap_dense<GraphType,EdgeOperator,EMapConfig,false>
::materialize_frontier( const partitioner & part ) {
    frontier & fr = op.get_frontier();
    if( fr.getType() == frontier_type::ft_unbacked
	&& op.get_config().get_threshold().is_sparse( fr, G.numEdges() ) ) {
	calculate_sparse_frontier( fr, part, op );
    }
}

template<typename GraphType, typename EdgeOperator,typename EMapConfig>
void step_emap_dense<GraphType,EdgeOperator,EMapConfig,true>
::materialize_frontier( const partitioner & part ) {
    frontier & fr = m_op.get_frontier();
    if( fr.getType() == frontier_type::ft_unbacked
	&& m_op.get_config().get_threshold().is_sparse( fr, m_G.numEdges() ) ) {
	calculate_sparse_frontier( fr, part, m_op );
    }
}




struct ValidVID
{
    bool operator() ( VID a ) {
        return a != ~(VID)0;
    }
};

// Returns true on first activation, or any activation if first one is not
// identified
template<update_method um>
bool conditional_set( VID * f, bool * zf, bool updated, VID d ) {
    if constexpr ( um == um_list ) {
	if( updated )
	    *f = d;
	return updated;
    } else if constexpr ( um == um_list_must_init_unique ) {
	// at least initialise *f
	if( updated ) { // set true
#if 1
	    // For some reason, the fetch-and-or, which includes cmp-xchg
	    // in a loop, performs slightly faster
	    if( *zf != 0 || __sync_fetch_and_or( (unsigned char *)zf,
						 (unsigned char)1 ) )
#else
	    if( __atomic_exchange_n( (unsigned char *)zf, (unsigned char)1,
				     __ATOMIC_ACQ_REL ) != 0 )
#endif
	    {
		// already set, don't set twice
		*f = ~(VID)0;
		return false;
	    } else {
		*f = d;
		return true;
	    }
	} else { // set false
	    *f = ~(VID)0;
	    return false;
	}
    } else if constexpr ( um == um_list_must_init ) {
	    *f = updated ? d : ~(VID)0;
	    return updated;
    } else if constexpr ( um == um_flags_only ) {
	if( updated )
	    *zf = true;
	return updated;
    } else if constexpr ( um == um_flags_only_unique ) {
	// at least initialise *f
	if( updated ) { // set true
#if 1
	    // For some reason, the fetch-and-or, which includes cmp-xchg
	    // in a loop, performs slightly faster
	    if( *zf != 0 || __sync_fetch_and_or( (unsigned char *)zf,
						 (unsigned char)1 ) )
#else
	    if( __atomic_exchange_n( (unsigned char *)zf, (unsigned char)1,
				     __ATOMIC_ACQ_REL ) != 0 )
#endif
	    {
		// already set, don't set twice
		return false;
	    } else {
		return true;
	    }
	} else { // set false
	    return false;
	}
    } else {
	// Do nothing for um_none
	static_assert( um == um_none, "invalid value for um" );
	return false;
    }
}


template<bool setf, bool zerof, bool once>
void conditional_set( VID * f, bool * zf, bool updated, VID d ) {
    if constexpr ( setf ) {
	if constexpr ( once ) { // at least initialise *f
	    if constexpr ( zerof ) {
		if( updated ) { // set true
		    if( *zf != 0 || __sync_fetch_and_or( (unsigned char *)zf,
							 (unsigned char)1 ) ) {
			// already set, don't set twice
			*f = ~(VID)0;
		    } else
			*f = d;
		} else { // set false
		    *f = ~(VID)0;
		}
	    } else {
		*f = updated ? d : ~(VID)0;
	    }
	} else if( updated )
	    *f = d;
    }
}

template<bool setf, bool once>
[[deprecated("seems out of use")]]
void conditional_set( VID * f, simd::nomask<1>, VID d ) {
    if constexpr ( setf )
	*f = d;
}


template<bool setf, typename Expr, typename Cache, typename CacheDesc,
	 typename Environment>
static DBG_NOINLINE void
process_csc_sparse_aset( const VID *out, VID deg, VID p, VID dstv, EID eid,
			 VID *fr,
			 const Expr & e, const Cache & vcaches,
			 const CacheDesc & vcaches_dst,
			 const Environment & env ) {
    static_assert( Expr::VL == 1, "Sparse traversal requires VL == 1" );

    auto dst = simd::template create_constant<simd::ty<VID,1>>( dstv );

    // Cache containing local variables only
    auto mdst = expr::create_value_map_new<1>( 
	expr::create_entry<expr::vk_dst>( dst ) );
    auto c = expr::cache_create_no_init( vcaches, mdst );
    expr::cache_init( env, c, vcaches_dst, mdst );

    // using output_type = simd::container<simd::ty<typename Expr::type, 1>>;
    using output_type = simd::container<typename Expr::data_type>;
    auto output = output_type::false_mask();

    auto pvec1 = simd::template create_constant<simd::ty<VID,1>>( 8*(VID)p );

    for( VID j=0; j < deg; j++ ) {
	auto src = simd::template load_from<simd::ty<VID,1>>( &out[j] );
	auto edg = simd::template create_constant<simd::ty<EID,1>>( eid+EID(j) );
	auto m = expr::create_value_map_new<1>(
	    // expr::create_entry<expr::vk_pid>( pvec1 ),
	    expr::create_entry<expr::vk_edge>( edg ),
	    expr::create_entry<expr::vk_dst>( dst ),
	    expr::create_entry<expr::vk_src>( src ) );
	// Note: CSC, sequential by degree, no atomics required
	auto ret = env.template evaluate<false>( c, m, e );
	if constexpr ( setf && !std::is_same_v<decltype(ret.value()),nomask<1>> )
	    output.lor_assign( ret.value() );
    }
    conditional_set<setf,false,true>( fr, nullptr, output.data(), dstv );

    expr::cache_commit( env, vcaches_dst, c, mdst );
}

template<bool setf, typename Expr, typename RExpr,
	 typename Cache, typename Environment>
static DBG_NOINLINE void
process_csc_sparse_aset( const VID *out, VID deg, VID p, VID dstv, VID *fr,
			 const Expr & e, const RExpr & re,
			 const Cache & vcaches, const Environment & env ) {
    static_assert( Expr::VL == 1, "Sparse traversal requires VL == 1" );

    auto dst = simd::template create_constant<simd::ty<VID,1>>( dstv );

    // Cache containing local variables only
    auto c = expr::cache_create( env, vcaches, expr::create_value_map_new<1>() );

    auto pvec1 = simd::template create_constant<simd::ty<VID,1>>( 8*(VID)p );

    for( VID j=0; j < deg; j++ ) {
	auto src = simd::template load_from<simd::ty<VID,1>>( &out[j] );
	auto m = expr::create_value_map_new<1>(
	    // expr::create_entry<expr::vk_pid>( pvec1 ),
	    expr::create_entry<expr::vk_dst>( dst ),
	    expr::create_entry<expr::vk_src>( src ) );
	// Note: CSC, sequential by degree, no atomics required
	env.template evaluate<false>( c, m, e );
    }

    // Now evaluate record method
    auto m = expr::create_value_map_new<1>(
	// expr::create_entry<expr::vk_pid>( pvec1 ),
	expr::create_entry<expr::vk_dst>( dst ) );
    auto output = env.template evaluate<false>( c, m, re );
    
    conditional_set<setf,false,true>( fr, nullptr, output.value().data(), dstv );
}


template<bool setf, typename Expr, typename Cache, typename Environment>
static DBG_NOINLINE void
process_csc_sparse_aset_parallel( const VID *out, VID deg, VID dstv,
				  VID *fr, const Expr & e,
				  const Cache & vcaches,
				  const Environment & env ) {
    static_assert( Expr::VL == 1, "Sparse traversal requires VL == 1" );

    // Cache containing local variables only
    auto c = expr::cache_create( env, vcaches, expr::create_value_map_new<1>() );

    auto dst = simd::template create_constant<simd::ty<VID,1>>( dstv );

    parallel_loop( VID(0), deg, [&]( VID j ) {
	// Need to (re-)initialise dst inside loop as otherwise compiler
	// cannot figure out the layout of the vector due to the vector
	// being passed in as an argument to the cilk_for function.
	auto src = simd::template load_from<simd::ty<VID,1>>( &out[j] );
	auto m = expr::create_value_map_new<1>(
	    expr::create_entry<expr::vk_dst>( dst ),
	    expr::create_entry<expr::vk_src>( src ) );
	// Note: set evaluator to use atomics as we are in parallel
	auto ret = env.template evaluate<true>( c, m, e );
	conditional_set<setf,false,false>( fr, nullptr, ret.value(), dstv );
    } );
}

// Sparse CSR traversal (GraphCSx) with a new frontier
template<typename config, typename Operator>
static __attribute__((noinline)) frontier csc_sparse_aset_with_f(
    config && cfg,
    const GraphCSx & GA,
    const partitioner & part,
    Operator op,
    frontier & active_set ) {
    const EID * idx = GA.getIndex();
    VID m = active_set.nActiveVertices();
    VID * s = active_set.getSparse();

    if( m == 0 )
	return frontier::empty();

    frontier status = frontier::sparse( GA.numVertices(), m );
    VID * o = status.getSparse();

    // CSR requires no check that destination is active.
    auto vexpr0 = op.relax( expr::value<simd::ty<VID,1>,expr::vk_src>(),
			    expr::value<simd::ty<VID,1>,expr::vk_dst>(),
			    expr::value<simd::ty<EID,1>,expr::vk_edge>() );

    // Rewrite local variables
    expr::cache<> vcaches_dst;
    auto vcaches = expr::extract_local_vars( vexpr0 );
    auto vexpr1 = expr::rewrite_caches<expr::vk_zero>( vexpr0, vcaches );

    // Match vector lengths and move masks
    auto vexpr = expr::rewrite_mask_main( vexpr1 );

    auto env = expr::eval::create_execution_environment_with(
	op.get_ptrset(), vcaches, vexpr );

    const VID *edge = GA.getEdges();

    if constexpr ( !std::decay_t<config>::is_parallel() ) {
	for( VID k = 0; k < m; k++ ) {
	    VID v = s[k];
	    intT d = idx[v+1]-idx[v];
	    o[k] = ~(VID)0;
	    process_csc_sparse_aset<true>(
		&edge[idx[v]], d, 0 /*part.part_of(v)*/, v, idx[v], &o[k],
		vexpr, vcaches, vcaches_dst, env );
	}
    } else {
	parallel_loop( VID(0), m, 1, [&]( VID k ) {
	    VID v = s[k];
	    intT d = idx[v+1]-idx[v];
	    // We cannot actually do parallel processing of neighbours as we
	    // have not privatized the array elements corresponding to the
	    // destination vertex. Vectorisation however may be feasible
	    // for very high-degree vertices.
	    o[k] = ~(VID)0;
	    process_csc_sparse_aset<true>(
		&edge[idx[v]], d, 0 /*part.part_of(v)*/, v, idx[v], &o[k],
		vexpr, vcaches, vcaches_dst, env );
	} );
    }

    // Filter out the empty slots (marked with -1)
    frontier output = frontier::sparse( GA.numVertices(), m );
    VID * oo = output.getSparse();
    VID m_valid = sequence::filter( o, oo, m, ValidVID() );
    status.del();
    output.calculateActiveCounts( GA, part, m_valid );

    return output;
}

// Sparse CSR traversal (GraphCSx) with a new frontier, record method
template<typename config, typename Operator>
static __attribute__((noinline)) frontier csc_sparse_aset_with_f_record(
    config && cfg,
    const GraphCSx & GA,
    const partitioner & part,
    Operator op,
    frontier & active_set ) {
    const EID * idx = GA.getIndex();
    VID m = active_set.nActiveVertices();
    VID * s = active_set.getSparse();

    if( m == 0 )
	return frontier::empty();

    frontier status = frontier::sparse( GA.numVertices(), m );
    VID * o = status.getSparse();

    // CSC requires no check that destination is active.
    // Use update method to check if vertex becomes active. Don't call vertexop
    // as it contains code to count active vertices, which we don't want
    // (required accumulators). However, this is a problem for user-specified
    // vertexop's and vertex map/scan merging.
    auto vexpr0 = op.relax( expr::value<simd::ty<VID,1>,expr::vk_src>(),
			    expr::value<simd::ty<VID,1>,expr::vk_dst>(),
			    expr::value<simd::ty<EID,1>,expr::vk_edge>() );
    auto rexpr0 = op.update( expr::value<simd::ty<VID,1>,expr::vk_dst>() );

    // Rewrite local variables
    auto vcaches = expr::extract_local_vars( vexpr0 );
    auto vexpr1 = expr::rewrite_caches<expr::vk_zero>( vexpr0, vcaches );
    auto rexpr1 = expr::rewrite_caches<expr::vk_zero>( rexpr0, vcaches );

    // Match vector lengths and move ma,sks
    auto vexpr = expr::rewrite_mask_main( vexpr1 );
    auto rexpr = expr::rewrite_mask_main( rexpr1 );

    auto env = expr::eval::create_execution_environment_with(
	op.get_ptrset(), vcaches, vexpr, rexpr );

    const VID *edge = GA.getEdges();

    if constexpr ( !std::decay_t<config>::is_parallel() ) {
	for( VID k = 0; k < m; k++ ) {
	    VID v = s[k];
	    intT d = idx[v+1]-idx[v];
	    o[k] = ~(VID)0;
	    process_csc_sparse_aset<true>(
		&edge[idx[v]], d, 0 /*part.part_of(v)*/, v, &o[k], vexpr, rexpr,
		vcaches, env );
	}
    } else {
	parallel_loop( VID(0), m, 1, [&]( VID k ) {
	    VID v = s[k];
	    intT d = idx[v+1]-idx[v];
	    // We cannot actually do parallel processing of neighbours as we
	    // have not privatized the array elements corresponding to the
	    // destination vertex. Vectorisation however may be feasible
	    // for very high-degree vertices.
	    o[k] = ~(VID)0;
	    process_csc_sparse_aset<true>(
		&edge[idx[v]], d, 0 /*part.part_of(v)*/, v, &o[k], vexpr, rexpr,
		vcaches, env );
	} );
    }

    // Filter out the empty slots (marked with -1)
    frontier output = frontier::sparse( GA.numVertices(), m );
    VID * oo = output.getSparse();
    VID m_valid = sequence::filter( o, oo, m, ValidVID() );
    status.del();
    output.calculateActiveCounts( GA, part, m_valid );

    return output;
}

/**=====================================================================*
 * \brief Calculate the number of parts in an edge-balanced partitioning
 *
 * If the ret_degree argument is provided, then it must point to an array
 * of length 2*m+1 at least. The first m entries will be filled with the
 * degrees of the m vertices; the next m+1 entries will be filled with
 * the cumulative degree (plus scan).
 *
 * \param s list of vertex IDs
 * \param m number of vertex IDs in list \see s
 * \param idx graph's index array to determine degree
 * \param ret_degree optional return value of degree and cumulative degree
 *
 * \return the number of advised parts for parallel execution
 *======================================================================*/
inline VID calculate_edge_balanced_parts(
    VID * const s,
    VID m,
    const EID * const idx,
    EID * ret_degree = nullptr ) {
    // Calculate info on the degrees of the vertices and total number of edges
    EID mm;

    // Is there enough data to warrant parallel processing?
    if( m > 2048 ) {
	// Reserve space if none was provided
	EID * degree = ret_degree;
	if( !ret_degree )
	    degree = new EID[m];
	
	parallel_loop( (VID)0, (VID)m, [&]( auto i ) {
	    degree[i] = idx[s[i]+1] - idx[s[i]];
	} );
	if( ret_degree ) {
	    EID * const voffsets = &degree[m];
	    voffsets[m] = mm = sequence::plusScan( degree, voffsets, m );
	} else {
	    // TODO: apply plusReduce with operator that calculates degree
	    //       without storing to memory. Avoids allocation of temporary 
	    //       storage.
	    mm = sequence::plusReduce( degree, m );
	    delete[] degree;
	}
    } else {
	if( ret_degree ) {
	    EID * const degree = ret_degree;
	    EID * const voffsets = &degree[m];
	    mm = 0;
	    for( VID i=0; i < (VID)m; ++i ) {
		EID deg = idx[s[i]+1] - idx[s[i]];
		degree[i] = deg;
		voffsets[i] = mm;
		mm += deg;
	    }
	    voffsets[m] = mm;
	} else {
	    // No additional space required if no return of details and
	    // doing sequential processing
	    mm = 0;
	    for( VID i=0; i < (VID)m; ++i ) {
		mm += idx[s[i]+1] - idx[s[i]];
	    }
	}
    }

    static constexpr EID mm_block = EMAP_BLOCK_SIZE;
    static constexpr EID mm_threshold = EMAP_BLOCK_SIZE;
    VID mm_parts = std::min( graptor_num_threads() * 16,
			     VID( ( mm + mm_block - 1 ) / mm_block ) );

    return mm_parts;
}

// Sparse CSC traversal (GraphCSx) with no new frontier calculated
template<typename config, typename Operator>
static __attribute__((noinline)) frontier csc_sparse_aset_no_f(
    config && cfg,
    const GraphCSx & GA,
    const partitioner & part,
    Operator op,
    frontier & active_set ) {
    const EID * idx = GA.getIndex();
    VID m = active_set.nActiveVertices();
    VID * s = active_set.getSparse();

    // CSR requires no check that destination is active.
    auto vexpr0 = op.relax( expr::value<simd::ty<VID,1>,expr::vk_src>(),
			    expr::value<simd::ty<VID,1>,expr::vk_dst>(),
			    expr::value<simd::ty<EID,1>,expr::vk_edge>() );
    auto aexpr0 = op.active( expr::value<simd::ty<VID,1>,expr::vk_dst>() );

    // Rewrite local variables; extract cacheable references
    auto vcaches_dst = expr::extract_cacheable_refs<expr::vk_dst>( vexpr0 );
    auto vcaches_let = expr::extract_local_vars( vexpr0, vcaches_dst );
    auto vcaches = expr::cache_cat( vcaches_dst, vcaches_let );

    auto vexpr1 = expr::rewrite_caches<expr::vk_zero>( vexpr0, vcaches_let );
    auto vexpr2 = expr::rewrite_caches<expr::vk_dst>( vexpr1, vcaches_dst );
    auto aexpr1 = expr::rewrite_caches<expr::vk_zero>( aexpr0, vcaches_let );
    auto aexpr2 = expr::rewrite_caches<expr::vk_dst>( aexpr1, vcaches_dst );

    // Replace array_ro by array_intl
    auto vexpr = expr::rewrite_internal( vexpr2 );
    auto aexpr = expr::rewrite_internal( aexpr2 );

    // auto env = expr::eval::create_execution_environment_with(
    // op.get_ptrset(), vcaches, vexpr );
    auto env = expr::eval::create_execution_environment_op(
	op, vcaches, GA.getWeights() ? GA.getWeights()->get() : nullptr );

    // Number of partitions
    EID * degree = new EID[2*m+1];
    VID mm_parts = calculate_edge_balanced_parts( s, m, idx, degree );

    // Require at least 32 parts before running in parallel. This is based on
    // the sequential version not requiring atomics and being much more
    // efficient than the parallel version, so a high degree of parallelism
    // is needed to offset the efficiency loss.
    static constexpr VID mm_min = 32;

    // Process edges
    if( !std::decay_t<config>::is_parallel() || mm_parts < mm_min ) {
	// Edge array (convenience)
	const VID *edge = GA.getEdges();

	for( VID k = 0; k < m; k++ ) {
	    VID v = s[k];
	    intT d = idx[v+1]-idx[v];
	    process_csc_sparse_aset<false>(
		&edge[idx[v]], d, 0 /*part.part_of(v)*/, v, idx[v], nullptr,
		vexpr, vcaches, vcaches_dst, env );
	}
    } else {
	static constexpr EID mm_block = EMAP_BLOCK_SIZE;
	static constexpr EID mm_threshold = EMAP_BLOCK_SIZE;

	EID * voffsets = &degree[m];
	EID mm = voffsets[m];
	vertex_partition<VID,EID> * parts = new vertex_partition<VID,EID>[mm_parts];
	partition_vertex_list<mm_block,mm_threshold,VID,EID>(
	    s, m, voffsets, idx, mm, mm_parts,
	    [&]( VID p, VID from, VID to ) {
		new ( &parts[p] ) vertex_partition<VID,EID>( from, to );
	    } );
#if 0
	parallel_loop( VID(0), m, 1, [&]( VID k ) {
	    VID v = s[k];
	    intT d = idx[v+1]-idx[v];
	    // We cannot actually process neighbours in parallel as we have not
	    // privatized the array elements corresponding to the destination
	    // vertex. Vectorisation however may be feasible for very
	    // high-degree vertices.
	    process_csc_sparse_aset<false>( &edge[idx[v]], d, 0 /*part.part_of(v)*/, v, idx[v], nullptr, vexpr,
					    vcaches, env );
	} );
#else
	parallel_loop( VID(0), mm_parts, 1, [&]( VID p ) {
	    // Edge array (convenience)
	    const VID *edge = GA.getEdges();

	    parts[p].process_pull( s, idx, edge, vcaches, vcaches_dst,
				   env, vexpr, aexpr );
	} );
#endif
    }

    delete[] degree;

    // return
    return frontier::all_true( GA.numVertices(), GA.numEdges() );
}

// Note: normally, we would need this method only for sparse frontiers,
//       where m ~ n and so fits in a VID rather than an EID. We need to check
//       this however.
inline void removeDuplicates( VID * ids, EID me, const partitioner & part ) {
    static mmap_ptr<VID> flags;

    constexpr VID undef = std::numeric_limits<VID>::max();

    VID m = (VID) me;
    assert( me == (EID)m && "length of array should be small enough" );

    // Allocate working memory. Try to retain it across calls.
    VID n = part.get_num_elements();
    if( flags.get_length() != n ) {
	if( flags.get_length() > 0 )
	    flags.del();
	flags.allocate( numa_allocation_partitioned( part ) );
	parallel_loop( VID(0), n, [&]( VID i ) {
	    flags[i] = undef;
	} );
    }

    // Figure out which vertices are listed in ids
    parallel_loop( VID(0), m, [&]( VID i ) {
	if( ids[i] != undef && flags[ids[i]] == undef )
	    CAS( &flags[ids[i]], undef, i );
    } );

    // Cancel out duplicates
    parallel_loop( VID(0), m, [&]( VID i ) {
	if( ids[i] != (VID)-1 ) {
	    if( flags[ids[i]] == i ) {  // win
		flags[ids[i]] = undef; //reset
	    } else
		ids[i] = undef; // lose
	}
    } );
    // Flags is restored to all -1 now
    // TODO: we could now already know how many distinct values there are
    //       by counting them above, which can speedup the subsequent filter op.
}

inline VID removeDuplicatesAndFilter_seq( VID * ids, EID me,
					  VID * tgt, const partitioner & part ) {
    VID m = (VID) me;
    assert( me == (EID)m && "length of array should be small enough" );

    constexpr VID undef = std::numeric_limits<VID>::max();

    if( me > 150000 ) {
	static mmap_ptr<VID> flags;

	// Allocate working memory. Try to retain it across calls.
	VID n = part.get_num_elements();
	if( flags.get_length() != n ) {
	    if( flags.get_length() > 0 )
		flags.del();
	    flags.allocate( numa_allocation_partitioned( part ) );
	}

	// Always reset. Can do in parallel (large numbers)
	parallel_loop( VID(0), n, [&]( VID i ) {
	    flags[i] = (VID)-1;
	} );

	// Figure out which vertices are listed in ids
	VID i = 0;
	VID t = 0;
	while( i < m ) {
	    if( ids[i] == undef ) {
		ids[i] = ids[--m];
	    } else if( flags[ids[i]] == undef ) {
		flags[ids[i]] = i;
		tgt[t++] = ids[i];
		++i;
	    } else
		ids[i] = ids[--m];
	}
	assert( t == m );
	return m; // Number of unique values
    } else if( m > 10 ) {
	std::sort( &ids[0], &ids[m] );

	assert( !std::is_signed_v<VID> ); // should be ok with signed, double-check
	if( ids[0] == undef ) // Assuming sort is unsigned, ~VID(0) at back
	    return 0;

	tgt[0] = ids[0];

	VID t=1;
	for( VID s=1; s < m; ++s ) {
	    if( ids[s] == ids[s-1] ) {
		// next
	    } else if( ids[s] == undef ) {
		break;
	    } else {
		tgt[t++] = ids[s];
	    }
	}
	return t;
    } else {
	// Simplistic algorithm, should work well on very short arrays
	VID s = 0;
	while( s < m ) {
	    if( ids[s] == undef )
		ids[s] = ids[--m];
	    else
		++s;
	}
	VID t = 0;
	for( s=0; s < m; ++s ) {
	    VID sval = ids[s];
	    tgt[t++] = sval;
	    VID i = s+1;
	    while( i < m ) {
		if( ids[i] == sval ) { // duplicate
		    ids[i] = ids[--m];
		} else
		    ++i;
	    }
	}
	assert( t == m );
	return m; // Number of unique values
    }
}

template <typename T>
class has_add_in_frontier_out
{
    typedef char one;
    struct two { char x[2]; };

    template <typename C> static one test( decltype(&C::add_in_frontier_out) ) ;
    template <typename C> static two test(...);    

public:
    static constexpr bool value = sizeof(test<T>(0)) == sizeof(char);
};

template<typename Operator, typename Enable = void>
struct emap_do_add_in_frontier_out
    : public std::false_type { };

template<typename Operator>
struct emap_do_add_in_frontier_out<
    Operator,std::enable_if_t<has_add_in_frontier_out<Operator>::value>>
    : public std::integral_constant<bool,Operator::add_in_frontier_out> { };

template<bool zerof, typename EIDRetriever, typename Expr>
static VID
process_csr_sparse_seq( const EIDRetriever & eid_retriever,
			const VID *out, VID deg, VID srcv, VID *frontier,
			bool *zf, const Expr & e ) {
    tuple<> c; // empty cache

    static_assert( Expr::VL == 1, "Sparse traversal requires VL == 1" );

    auto src = simd::template create_scalar<simd::ty<VID,1>>( srcv );

    VID o = 0;
    for( VID j=0; j < deg; j++ ) {
	EID seid = eid_retriever.get_edge_eid( srcv, j );
	auto dst = simd::template load_from<simd::ty<VID,1>>( &out[j] );
	auto eid = simd::template create_scalar<simd::ty<EID,1>>( seid );
	auto m = expr::create_value_map_new<1>(
	    expr::create_entry<expr::vk_dst>( dst ),
	    expr::create_entry<expr::vk_edge>( eid ),
	    expr::create_entry<expr::vk_src>( src ) );
	// Note: set evaluator to _not_ use atomics
	auto ret = expr::evaluate<false>( c, m, e );
	if( ret.value().data() ) {
	    if constexpr ( zerof ) {
		if( !zf[out[j]] ) {
		    zf[out[j]] = true;
		    frontier[o++] = out[j];
		}
	    } else {
		frontier[o++] = out[j];
	    }
	}
    }

    return o;
}

template<bool atomic, bool setf, bool zerof,
	 typename EIDRetriever, typename Cache, typename Environment,
	 typename Expr>
static void
process_csr_sparse( const EIDRetriever & eid_retriever,
		    const VID *out, VID deg, VID srcv, VID *frontier,
		    bool *zf, Cache & c,
		    const Environment & env, const Expr & e ) {

    static_assert( Expr::VL == 1, "Sparse traversal requires VL == 1" );

    auto src = simd::template create_scalar<simd::ty<VID,1>>( srcv );

    for( VID j=0; j < deg; j++ ) {
	EID seid = eid_retriever.get_edge_eid( srcv, j );
	auto dst = simd::template load_from<simd::ty<VID,1>>( &out[j] );
	auto eid = simd::template create_scalar<simd::ty<EID,1>>( seid );
	auto m = expr::create_value_map_new<1>(
	    expr::create_entry<expr::vk_dst>( dst ),
	    expr::create_entry<expr::vk_edge>( eid ),
	    expr::create_entry<expr::vk_src>( src ) );
	// Note: set evaluator to use atomics
	auto ret = env.template evaluate<atomic>( c, m, e );
	conditional_set<setf,zerof,true>( &frontier[j], &zf[dst.data()],
					  ret.value().data(), dst.data() );
    }
}

template<bool atomic, bool setf, bool zerof, typename EIDRetriever,
	 typename Cache, typename Environment, typename Expr>
static void
process_csr_sparse_parallel( const EIDRetriever & eid_retriever,
			     const VID *out, VID deg, VID srcv, VID *frontier,
			     bool *zf,
			     Cache & c,
			     const Environment & env,
			     const Expr & e ) {
    static_assert( Expr::VL == 1, "Sparse traversal requires VL == 1" );

    parallel_loop( VID(0), deg, [&]( VID j ) {
	// Need to (re-)initialise src inside loop as otherwise compiler
	// cannot figure out the layout of the vector due to the vector
	// being passed in as an argument to the cilk_for function.
	EID seid = eid_retriever.get_edge_eid( srcv, j );
	auto eid = simd::template create_scalar<simd::ty<EID,1>>( seid );
	auto src = simd::template create_scalar<simd::ty<VID,1>>( srcv );
	auto dst = simd::template load_from<simd::ty<VID,1>>( &out[j] );
	// auto dst = simd_vector<VID,1>::load_from( &out[j] ); // linear read
	auto m = expr::create_value_map_new<1>(
	    expr::create_entry<expr::vk_dst>( dst ),
	    expr::create_entry<expr::vk_edge>( eid ),
	    expr::create_entry<expr::vk_src>( src ) );
	// Note: set evaluator to use atomics
	auto ret = env.template evaluate<atomic>( c, m, e );
	conditional_set<setf,zerof,true>( &frontier[j], &zf[dst.data()],
					  ret.value().data(), dst.data() );
    } );
}

template<bool atomic, update_method um, typename FActiv,
	 typename Cache, typename Environment, typename Expr>
static void
process_csr_sparse( const VID *out, EID be, VID deg, VID srcv, VID *frontier,
		    bool *zf, FActiv & factiv,
		    Cache & c, const Environment & env,
		    const Expr & e ) {
    static_assert( Expr::VL == 1, "Sparse traversal requires VL == 1" );

    auto src = simd::template create_scalar<simd::ty<VID,1>>( srcv );

    for( VID j=0; j < deg; j++ ) {
	// EID seid = eid_retriever.get_edge_eid( srcv, j );
	EID seid = be + (EID)j;
	auto dst = simd::template load_from<simd::ty<VID,1>>( &out[j] );
	auto eid = simd::template create_scalar<simd::ty<EID,1>>( seid );
	auto m = expr::create_value_map_new<1>(
	    expr::create_entry<expr::vk_dst>( dst ),
	    expr::create_entry<expr::vk_edge>( eid ),
	    expr::create_entry<expr::vk_src>( src ) );
	// Note: set evaluator to use atomics
	auto ret = env.template evaluate<atomic>( c, m, e );
	bool act = conditional_set<um>( &frontier[j], &zf[dst.data()],
					ret.value().data(), dst.data() );
	if( act )
	    factiv( dst.data() );
    }
}

template<bool atomic,
	 typename Cache, typename Environment, typename Expr>
static VID
process_csr_append( const VID *out, EID be, VID deg, VID srcv,
		    VID * frontier, bool *zf,
		    Cache & c, const Environment & env, const Expr & e ) {
    static_assert( Expr::VL == 1, "Sparse traversal requires VL == 1" );

    auto src = simd::template create_scalar<simd::ty<VID,1>>( srcv );

    VID fidx = 0;

    for( VID j=0; j < deg; j++ ) {
	// EID seid = eid_retriever.get_edge_eid( srcv, j );
	EID seid = be + (EID)j;
	auto dst = simd::template load_from<simd::ty<VID,1>>( &out[j] );
	auto eid = simd::template create_scalar<simd::ty<EID,1>>( seid );
	auto m = expr::create_value_map_new<1>(
	    expr::create_entry<expr::vk_dst>( dst ),
	    expr::create_entry<expr::vk_edge>( eid ),
	    expr::create_entry<expr::vk_src>( src ) );
	// Note: set evaluator to use atomics
	auto ret = env.template evaluate<atomic>( c, m, e );
	if( ret.value().data() ) {
	    if constexpr ( !expr::is_single_trigger<Expr>::value ) {
		// for particular cases, no need to use zerof (e.g. count_down)
		// Set frontier, once.
		if( zf[dst.data()] == 0
		    && __sync_fetch_and_or( (unsigned char *)&zf[dst.data()],
					    (unsigned char)1 ) == 0 )
		    // first time being set
		    frontier[fidx++] = dst.data();
	    } else {
		// first time and only time being set
		frontier[fidx++] = dst.data();
	    }
	}
    }

    return fidx;
}


template<bool atomic, update_method um,
	 typename Cache, typename Environment, typename Expr>
static void
process_csr_sparse_parallel( const VID *out, EID be,
			     VID deg, VID srcv, VID *frontier,
			     bool *zf,
			     Cache & c, 
			     const Environment & env,
			     const Expr & e ) {
    static_assert( Expr::VL == 1, "Sparse traversal requires VL == 1" );

    parallel_loop( VID(0), deg, [&]( VID j ) {
	// Need to (re-)initialise src inside loop as otherwise compiler
	// cannot figure out the layout of the vector due to the vector
	// being passed in as an argument to the cilk_for function.
	// EID seid = eid_retriever.get_edge_eid( srcv, j );
	EID seid = be + (EID)j;
	auto eid = simd::template create_scalar<simd::ty<EID,1>>( seid );
	auto src = simd::template create_scalar<simd::ty<VID,1>>( srcv );
	auto dst = simd::template load_from<simd::ty<VID,1>>( &out[j] );
	// auto dst = simd_vector<VID,1>::load_from( &out[j] ); // linear read
	auto m = expr::create_value_map_new<1>(
	    expr::create_entry<expr::vk_dst>( dst ),
	    expr::create_entry<expr::vk_edge>( eid ),
	    expr::create_entry<expr::vk_src>( src ) );
	// Note: set evaluator to use atomics
	auto ret = env.template evaluate<atomic>( c, m, e );
	conditional_set<um>( &frontier[j], &zf[dst.data()],
			     ret.value().data(), dst.data() );
    } );
}


template<bool atomic, bool setf, bool zerof,
	 typename EIDRetriever, typename Expr>
static void
process_csr_sparse( const EIDRetriever & eid_retriever,
		    const VID *out, VID deg, VID srcv, VID *frontier,
		    bool *zf, const Expr & e ) {
    tuple<> c; // empty cache

    static_assert( Expr::VL == 1, "Sparse traversal requires VL == 1" );

    auto src = simd::template create_scalar<simd::ty<VID,1>>( srcv );

    for( VID j=0; j < deg; j++ ) {
	EID seid = eid_retriever.get_edge_eid( srcv, j );
	auto dst = simd::template load_from<simd::ty<VID,1>>( &out[j] );
	auto eid = simd::template create_scalar<simd::ty<EID,1>>( seid );
	auto m = expr::create_value_map_new<1>(
	    expr::create_entry<expr::vk_dst>( dst ),
	    expr::create_entry<expr::vk_edge>( eid ),
	    expr::create_entry<expr::vk_src>( src ) );
	// Note: set evaluator to use atomics
	auto ret = expr::evaluate<atomic>( c, m, e );
	conditional_set<setf,zerof,true>( &frontier[j], &zf[dst.data()],
					  ret.value().data(), dst.data() );
    }
}

template<bool atomic, bool setf, bool zerof, typename EIDRetriever,
	 typename Expr>
static void
process_csr_sparse_parallel( const EIDRetriever & eid_retriever,
			     const VID *out, VID deg, VID srcv, VID *frontier,
			     bool *zf,
			     const Expr & e ) {
    tuple<> c; // empty cache

    static_assert( Expr::VL == 1, "Sparse traversal requires VL == 1" );

    parallel_loop( VID(0), deg, [&]( VID j ) {
	// Need to (re-)initialise src inside loop as otherwise compiler
	// cannot figure out the layout of the vector due to the vector
	// being passed in as an argument to the cilk_for function.
	EID seid = eid_retriever.get_edge_eid( srcv, j );
	auto eid = simd::template create_scalar<simd::ty<EID,1>>( seid );
	auto src = simd::template create_scalar<simd::ty<VID,1>>( srcv );
	auto dst = simd::template load_from<simd::ty<VID,1>>( &out[j] );
	// auto dst = simd_vector<VID,1>::load_from( &out[j] ); // linear read
	auto m = expr::create_value_map_new<1>(
	    expr::create_entry<expr::vk_dst>( dst ),
	    expr::create_entry<expr::vk_edge>( eid ),
	    expr::create_entry<expr::vk_src>( src ) );
	// Note: set evaluator to use atomics
	auto ret = expr::evaluate<atomic>( c, m, e );
	conditional_set<setf,zerof,true>( &frontier[j], &zf[dst.data()],
					  ret.value().data(), dst.data() );
    } );
}


template<typename Operator>
void csr_sparse_update_frontier( VID nextM, VID * nextIndices, Operator op ) {
    return;
    
    auto vop = op.update( expr::value<simd::ty<VID,1>,expr::vk_dst>() );
    if constexpr ( !expr::is_noop<decltype(vop)>::value ) {
	// TODO: sparse vmap could make use of gather/scatter
	static constexpr unsigned short VL = 1; // sparse, thus scalar
	auto expr0 = vop;

	auto l_cache = expr::extract_local_vars( expr0, expr::cache<>() );
	auto expr1 = expr::rewrite_caches<expr::vk_zero>( expr0, l_cache );

	auto env = expr::eval::create_execution_environment_with(
	    op.get_ptrset(), l_cache, expr1 );

	auto expr = rewrite_internal( expr1 );

	parallel_loop( VID(0), nextM, [&]( VID i ) {
	    VID v = nextIndices[i];
	    auto vv = simd::template create_unknown<simd::ty<VID,VL>>( v );
	    auto m = expr::create_value_map_new<VL>(
		expr::create_entry<expr::vk_dst>( vv ) );
	    auto c = cache_create( env, l_cache, m );
	    auto ret = env.evaluate( c, m, expr );
	    if( !ret.value().data() )
		nextIndices[i] = ~(VID)0;
	} );
    }
}

template<bool zerof, typename config, typename EIDRetriever, typename Operator>
static __attribute__((noinline)) frontier csr_sparse_with_f_seq(
    config & cfg,
    const GraphCSx & GA,
    const EIDRetriever & eid_retriever,
    const partitioner & part,
    EID outEdgeCount,
    frontier & old_frontier,
    const Operator & op ) {
    const EID * idx = GA.getIndex();
    VID m = old_frontier.nActiveVertices();
    VID * s = old_frontier.getSparse();

    bool * zf = nullptr;
    if constexpr ( zerof )
	zf = GA.get_flags( part );

    // if constexpr ( emap_do_add_in_frontier_out<Operator>::value )
    // outEdgeCount += m; // Add in m vertices in old frontier
    VID *outEdges = new VID[outEdgeCount];

    // CSR requires no check that destination is active.
    auto vexpr0 = op.relax( expr::value<simd::ty<VID,1>,expr::vk_src>(),
			    expr::value<simd::ty<VID,1>,expr::vk_dst>(),
			    expr::value<simd::ty<EID,1>,expr::vk_edge>() );
    auto uexpr0 = op.update( expr::value<simd::ty<VID,1>,expr::vk_dst>() );

    // Append. TODO: ensure short-circuit evaluation in scalar execution
    // auto vexpr1 = expr::set_mask( vexpr0, uexpr0 );
    auto vexpr1 = vexpr0 && uexpr0;
    // auto vexpr1 = vexpr0;

    // Rewrite local variables
    auto l_cache = expr::extract_local_vars( vexpr1, expr::cache<>() );
    auto vexpr2 = expr::rewrite_caches<expr::vk_zero>( vexpr1, l_cache );

    // Match vector lengths and move masks
    auto vexpr = expr::rewrite_mask_main( vexpr2 );
    using Expr = std::decay_t<decltype(vexpr)>;

    auto env = expr::eval::create_execution_environment_op(
	op, l_cache, GA.getWeights() ? GA.getWeights()->get() : nullptr );

    // Cache local variables; no need to destruct/commit cache
    auto mi = expr::create_value_map_new<1>();
    auto c = cache_create( env, l_cache, mi );

    const VID *edge = GA.getEdges();

    VID nactv = 0;
    EID nacte = 0;
    for( VID k = 0; k < m; k++ ) {
	VID v = s[k];
	EID x = idx[v];
	EID y = idx[v+1];

	// Sequential, so dump in sequential locations in outEdges[]
	auto src = simd::template create_scalar<simd::ty<VID,1>>( v );

	for( EID e=x; e < y; ++e ) {
	    // EID seid = eid_retriever.get_edge_eid( v, j );
	    auto dst = simd::template load_from<simd::ty<VID,1>>( &edge[e] );
	    auto eid = simd::template create_scalar<simd::ty<EID,1>>( e );
	    auto m = expr::create_value_map_new<1>(
		expr::create_entry<expr::vk_dst>( dst ),
		expr::create_entry<expr::vk_edge>( eid ),
		expr::create_entry<expr::vk_src>( src ) );
	    // Note: sets evaluator to _not_ use atomics
	    auto ret = env.evaluate( c, m, vexpr );
	    if( ret.value().data() ) {
		bool add = false;
		VID sdst = dst.data();
		if constexpr ( !expr::is_single_trigger<Expr>::value ) {
		    if constexpr ( zerof ) {
			if( !zf[sdst] ) {
			    zf[sdst] = true;
			    add = true;
			}
		    } else
			add = true;
		} else
		    add = true;

		if( add ) {
		    outEdges[nactv++] = sdst;
		    nacte += idx[sdst+1] - idx[sdst];
		}
	    }
	}
    }

    // assert( nactv <= outEdgeCount );

    outEdgeCount = nactv;

    // Add in vertices in old_frontier, if not present yet
/*
    if constexpr ( emap_do_add_in_frontier_out<Operator>::value ) {
	static_assert( zerof, "missing code for !zerof" );
	for( VID r=0; r < m; ++r )
	    if( !zf[s[r]] )
		outEdges[nactv++] = s[r];
    }
*/

    // Restore zero frontier to all zeros; do not access old frontiers copied
    // in as we did not set their flag in the loop above
    if constexpr ( zerof && !expr::is_single_trigger<Expr>::value ) {
	for( VID k=0; k < outEdgeCount; ++k ) {
	    // assert( zf[outEdges[k]] );
	    zf[outEdges[k]] = false;
	}
    }

    // TODO: edge count is wrong in case of !zerof because
    //       new frontier may contain same vertex multiple times.
    frontier new_frontier
	= frontier::sparse( GA.numVertices(), nactv, outEdges );
    new_frontier.setActiveCounts( nactv, nacte );

    return new_frontier;
}

EID calculate_edge_count( const VID * s, VID m, const EID * const idx,
			  EID * degree, EID * voffsets ) {
    EID mm0;
    if( m > 2048 ) {
	parallel_loop( (VID)0, (VID)m, [&]( auto i ) {
	    degree[i] = idx[s[i]+1] - idx[s[i]];
	} );
	mm0 = sequence::plusScan( degree, voffsets, m );
    } else {
	mm0 = 0;
	for( VID i=0; i < (VID)m; ++i ) {
	    EID deg = idx[s[i]+1] - idx[s[i]];
	    // degree[i] = deg; -- unused
	    voffsets[i] = mm0;
	    mm0 += deg;
	}
    }
    voffsets[m] = mm0;
    return mm0;
}

// Sparse CSR traversal (GraphCSx) with a new frontier
// Uses internal array for removing duplicates, contrasted to the explicit
// removeDuplicates step
template<typename config, typename EIDRetriever, typename Operator>
static __attribute__((noinline)) frontier csr_sparse_with_f(
    config & cfg,
    const GraphCSx & GA,
    const EIDRetriever & eid_retriever,
    const partitioner & part,
    frontier & old_frontier,
    const Operator & op ) {
    const EID * const idx = GA.getIndex();
    VID m = old_frontier.nActiveVertices();
    const VID * s = old_frontier.getSparse();
    EID mm = old_frontier.nActiveEdges();

    // timer tm;
    // tm.start();

    // Use a dense array of boolean flags to weed out duplicate activations
    // of the same vertex. Disable if the atomic is a single-trigger operation,
    // i.e., it will activate any vertex at most once by design.
    static constexpr bool zerof = !expr::is_single_trigger_op<Operator>::value;

    // Arrays to support edge-balanced partitioning. We may not need these
    // if we hand over to sequential or fusion methods, except if the number
    // of edges was not known.
    EID * degree = nullptr, * voffsets = nullptr;

    // Unknown number of edges, not calculated yet
    if( mm == ~(EID)0 ) {
	degree = new EID[2*m+1];
	voffsets = &degree[m];
	mm = calculate_edge_count( s, m, idx, degree, voffsets );
	old_frontier.setActiveCounts( m, mm ); // for what it's worth
    }

    // Block sizes for parallelisation.
    static constexpr EID mm_block = EMAP_BLOCK_SIZE;
    static constexpr EID mm_threshold = EMAP_BLOCK_SIZE;
    VID mm_parts =
	std::min( graptor_num_threads() * 16,
		  VID( ( mm + mm_block - 1 ) / mm_block ) );
    // Require at least 32 parts before running in parallel. This is based on
    // the sequential version not requiring atomics and being much more
    // efficient than the parallel version, so a high degree of parallelism
    // is needed to offset the efficiency loss.
    static constexpr VID mm_min = 32;

    // Should we execute sequentially?
    const bool do_seq = mm_parts < mm_min || !cfg.is_parallel();

    // Every call with a fusion operation defined will be executed in the
    // fusion-based traversal. Exceptions are made when the fusion operation
    // is read-only, i.e., it has no side effects. In this case we assume that
    // it is correct to also execute with a non-fusion traversal, which we will
    // do if a sequential execution is preferred.
    if constexpr ( api::has_fusion_op_v<Operator> ) {
	if( cfg.is_parallel()
	    && graptor_num_threads() > 1
	    && cfg.do_fusion( m, mm, GA.numEdges() ) ) {
	    return csr_sparse_with_f_fusion_stealing(
		cfg, GA, eid_retriever, part, old_frontier, op );
	}
    }

    if( do_seq ) {
	return csr_sparse_with_f_seq<zerof>(
	    cfg, GA, eid_retriever, part, mm, old_frontier, op );
    }

    // Calculate info on the degrees of the vertices and total number of edges
    // to support edge-balanced partitioning.
    // While some calls may not require edge-balanced partitioning (e.g., when
    // the number of active edges is similar to the number of active vertices),
    // the approach ensures that long neighbour lists are parallelised without
    // further effort.
    if( degree == nullptr ) {
	degree = new EID[2*m+1];
	voffsets = &degree[m];
	[[maybe_unused]] EID mm0 =
	    calculate_edge_count( s, m, idx, degree, voffsets );
	assert( mm == mm0 );
    }
    
    // auto tm1 = tm.next();

    // Break high-degree vertices over multiple blocks
    edge_partition<VID,EID> * parts = new edge_partition<VID,EID>[mm_parts];
    mm_parts = partition_edge_list<mm_threshold,VID,EID>(
	s, m, voffsets, idx, mm, mm_parts,
	[&]( VID p, VID from, VID to, EID fstart, EID lend, EID offset ) {
	    new ( &parts[p] )
		edge_partition<VID,EID>( from, to, fstart, lend, offset );
	} );

    // auto tm2 = tm.next();

    // Compilation steps
    
    // CSR requires no check that destination is active.
    auto vexpr0 = op.relax( expr::value<simd::ty<VID,1>,expr::vk_src>(),
			    expr::value<simd::ty<VID,1>,expr::vk_dst>(),
			    expr::value<simd::ty<EID,1>,expr::vk_edge>() );
    auto uexpr0 = op.update( expr::value<simd::ty<VID,1>,expr::vk_dst>() );

    // Append. TODO: ensure short-circuit evaluation in scalar execution
    // auto vexpr1 = expr::set_mask( vexpr0, uexpr0 );
    auto vexpr1 = vexpr0 && uexpr0;
    // auto vexpr1 = vexpr0;

    // Rewrite local variables
    auto l_cache = expr::extract_local_vars( vexpr1, expr::cache<>() );
    auto vexpr2 = expr::rewrite_caches<expr::vk_zero>( vexpr1, l_cache );

    // Match vector lengths and move masks
    auto vexpr = expr::rewrite_mask_main( vexpr2 );

    auto env = expr::eval::create_execution_environment_op(
	op, l_cache, GA.getWeights() ? GA.getWeights()->get() : nullptr );
    

    const VID *edge = GA.getEdges();

    constexpr bool need_atomic = !expr::is_benign_race<decltype(vexpr)>::value;

    // Cache local variables; no need to destruct/commit cache
    auto mi = expr::create_value_map_new<1>();
    auto c = cache_create( env, l_cache, mi );

    // auto tm3 = tm.next();
    // decltype(tm3) tm4a, tm4b, tm4c, tm4d, tm4e;
    // decltype(tm3) * tm_parts = new decltype(tm3)[mm_parts]();

    frontier new_frontier;

    // Stores number of activated vertices, scan of #activated vertices
    // and number of activated edges for each part
    EID* n_out_block = new EID[mm_parts*3+1];

#if 0
    if( false && zerof && mm > EID(GA.numVertices()) ) {
	// Case of potentially high number of activated vertices
	parallel_loop( VID(0), mm_parts, 1, [&]( VID p ) {
	    parts[p].template process_push_many<need_atomic,um_flags_only>(
		s, idx, edge, zf, c, env, vexpr );
	} );

	// Construct frontier based on zerof flags and simultaneously
	// restore zero flags to all zeros
	GraphCSRAdaptor GA_csr( GA );
	frontier ftrue = frontier::all_true( GA.numVertices(), GA.numEdges() );
	expr::array_ro<bool,VID,expr::aid_emap_zerof> a_zerof( zf );
	make_lazy_executor( part )
	    .vertex_filter(
		GA_csr,
		ftrue,
		new_frontier,
		[&]( auto v ) {
		    return expr::let<expr::aid_emap_let>(
			a_zerof[v],
			[&]( auto z ) {
			    return expr::make_seq(
				a_zerof[v] = expr::_0,
				z != expr::_0 ); }
			);
		} )
	    .materialize();
	ftrue.del();
    } else
#endif
    if( mm > EID(GA.numVertices()) ) {
	// Case of potentially high number of activated vertices
	// We do not use zero flags as they copy the dense frontier.
	// Update a dense boolean frontier direct as we would do with the
	// zero flags.

	// Initialize bool frontier. Sets bool flags to zero
	new_frontier = frontier::create<frontier_type::ft_bool>( part );

	// Get private zero flags; no need to clear :)
	bool * pzf = new_frontier.template getDense<frontier_type::ft_bool>();

	// Iterate over edges, setting frontier en lieu of zero flags
	parallel_loop( VID(0), mm_parts, 1, [&]( VID p ) {
	    std::tie( n_out_block[p], n_out_block[p+mm_parts] ) =
		parts[p].template process_push_many<need_atomic,um_flags_only_unique>(
		    s, idx, edge, pzf, c, env, vexpr );
	} );

	// Calculate active counts, based on counts obtained during edge
	// processing.
	VID nactv = n_out_block[0];
	EID nacte = n_out_block[mm_parts];
	for( VID p=1; p < mm_parts; ++p ) {
	    nactv += n_out_block[p];
	    nacte += n_out_block[p+mm_parts];
	}
	new_frontier.setActiveCounts( nactv, nacte );
    } else {
	bool * zf = nullptr;
	if constexpr ( zerof )
	    zf = GA.get_flags( part );

	// Case of likely sparse output frontier
	assert( mm <= EID(std::numeric_limits<VID>::max())
		&& "limitation due to cast in frontier construction" );

	VID* outEdges = new VID[mm]; //!< activated edges

	// Count number of activated vertices for each block. Append them to
	// a per-block list.
	parallel_loop( VID(0), mm_parts, 1, [=,&c,&env,&vexpr]( VID p ) {
	    // timer tm;
	    // tm.start();
	    std::tie( n_out_block[p], n_out_block[p+mm_parts] ) =
		parts[p].template process_push<need_atomic>(
		    s, outEdges, zf, idx, edge, c, env, vexpr );
	    // tm_parts[p] = tm.next();
	} );

	// tm4a = tm.next();

	// Scan on number of activated vertices per block to calculate
	// offsets for aggregating the lists in a single list.
	EID n_out = sequence::plusScan(
	    n_out_block, &n_out_block[2*mm_parts], mm_parts );
	n_out_block[3*mm_parts] = n_out;

	// tm4b = tm.next();

	// Compact the per-block lists into a single list
	new_frontier = frontier::sparse( GA.numVertices(), n_out );
	VID* nextIndices = new_frontier.getSparse();
	if( n_out > 2048 ) {
	    parallel_loop( VID(0), mm_parts, 1, [&]( VID p ) {
		std::copy( &outEdges[parts[p].get_offset()],
			   &outEdges[parts[p].get_offset()+n_out_block[p]],
			   &nextIndices[n_out_block[2*mm_parts+p]] );
	    } );
	} else {
	    for( VID p=0; p < mm_parts; ++p ) {
		std::copy( &outEdges[parts[p].get_offset()],
			   &outEdges[parts[p].get_offset()+n_out_block[p]],
			   &nextIndices[n_out_block[2*mm_parts+p]] );
	    }
	}
	VID nextM = n_out;

	// tm4c = tm.next();

	// Calculate the number of active edges
	// Note: there are no duplicates if zerof == true
	// Note: there are no duplicates if the expression triggers at most
	//       once per vertex
	if constexpr ( expr::is_idempotent<decltype(vexpr)>::value || zerof
		       || expr::is_single_trigger<decltype(vexpr)>::value ) {
	    // Operation is idempotent, i.e., we are allowed to process each
	    // edge multiple times. No need to remove duplicates.
	    // OR: we used the zero flags array to avoid duplicates.
	    // No need to filter empty slots (-1) as they are removed
	    // by copying active slices.
	    // ... nothing further to do
	    EID nacte = 0;
	    for( VID p=0; p < mm_parts; ++p )
		nacte += n_out_block[p+mm_parts];
	    VID nactv = n_out_block[3*mm_parts];
	    new_frontier.setActiveCounts( nactv, nacte );
	} else {
	    // Default code assuming not idempotent and not zerof
	    frontier tmp = frontier::sparse( GA.numVertices(), n_out );
	    std::swap( tmp, new_frontier );
	    VID * tmpIndices = tmp.getSparse();
	    nextIndices = new_frontier.getSparse();

	    if( n_out > GA.numEdges() / 1000 ) {
		// We have relatively many edges. Better to process this
		// in parallel.
		removeDuplicates( tmpIndices, n_out, part );
		// Filter out the empty slots (marked with -1)
		nextM = sequence::filter( tmpIndices, nextIndices, n_out,
					  ValidVID() );
	    } else {
		// Remove duplicates sequentially
		nextM = removeDuplicatesAndFilter_seq( tmpIndices, n_out,
						       nextIndices, part );
	    }

	    tmp.del();

	    new_frontier.calculateActiveCounts( GA, part, nextM );
	}


	// Restore zero frontier to all zeros
	if constexpr ( zerof ) {
	    parallel_loop( VID(0), nextM, [&]( VID k ) {
		assert( nextIndices[k] != (VID)-1 );
		zf[nextIndices[k]] = false;
	    } );
	}

	delete [] outEdges;
	// delete [] n_out_block;
    }

/*
    auto tm4 = tm.next();
    std::cerr << tm1 << ' ' << tm2 << ' ' << tm3 << ' '
	      << mm_parts << ' '
	      << tm4a << ' '
	      << tm4b << ' '
	      << tm4c << ' '
	      << tm4d << ' '
	      << tm4e << ' '
	      << tm4
	      << " parts: ";
    decltype(tm1) tmin = std::numeric_limits<decltype(tm1)>::max();
    decltype(tm1) tmax = 0;
    decltype(tm1) tsum = 0;
    for( VID p=0; p < mm_parts; ++p ) {
	auto nv = parts[p].num_vertices();
	auto ne = parts[p].num_edges( s, idx );
	std::cerr << tm_parts[p] << "(" << nv << ',' << ne << ',' << n_out_block[p] << ") ";
	tmax = std::max( tm_parts[p], tmax );
	tmin = std::min( tm_parts[p], tmin );
	tsum = tm_parts[p] + tsum;
    }
    std::cerr << " [" << tsum << ' ' << tmin << ' ' << tmax << ']';
    std::cerr << '\n';
*/

    // Clean up and return
    delete[] degree;
    delete[] parts;
    
    return new_frontier;
}


// Sparse CSR traversal (GraphCSx) with no new frontier calculated
template<typename config, typename EIDRetriever, typename Operator>
static __attribute__((noinline)) frontier csr_sparse_no_f(
    config & cfg,
    const GraphCSx & GA,
    const EIDRetriever & eid_retriever,
    frontier & old_frontier,
    Operator op ) {
    const EID * idx = GA.getIndex();
    VID m = old_frontier.nActiveVertices();
    VID * s = old_frontier.getSparse();

    // CSR requires no check that destination is active.
    auto vexpr0 = op.relax( expr::value<simd::ty<VID,1>,expr::vk_src>(),
			    expr::value<simd::ty<VID,1>,expr::vk_dst>(),
			    expr::value<simd::ty<EID,1>,expr::vk_edge>() );
    auto uexpr0 = op.update( expr::value<simd::ty<VID,1>,expr::vk_dst>() );

    // Append. TODO: ensure short-circuit evaluation in scalar execution
    auto vexpr1 = vexpr0 && uexpr0;

    // Rewrite local variables
    auto l_cache = expr::extract_local_vars( vexpr1, expr::cache<>() );
    auto vexpr2 = expr::rewrite_caches<expr::vk_zero>( vexpr1, l_cache );

    // Match vector lengths and move masks
    auto vexpr = expr::rewrite_mask_main( vexpr0 );

/*
    auto ew_pset = expr::create_map2<(unsigned)aid_key(array_aid(expr::aid_eweight)>(
	GA.getWeights() ? GA.getWeights()->get() : nullptr );

    auto env = expr::eval::create_execution_environment_with(
	op.get_ptrset( ew_pset ), l_cache, vexpr );
*/
    auto env = expr::eval::create_execution_environment_op(
	op, l_cache, GA.getWeights() ? GA.getWeights()->get() : nullptr );

    const VID *edge = GA.getEdges();

    constexpr bool need_atomic = !expr::is_benign_race<decltype(vexpr)>::value;

    // Cache local variables; no need to destruct/commit cache
    auto mi = expr::create_value_map_new<1>();
    auto c = cache_create( env, l_cache, mi );

    if constexpr ( !std::decay_t<config>::is_parallel() ) {
	for( VID k = 0; k < m; k++ ) {
	    VID v = s[k];
	    intT d = idx[v+1]-idx[v];
	    process_csr_sparse<false,um_none>(
		/*eid_retriever,*/ &edge[idx[v]], idx[v], d, v,
		nullptr, nullptr, c, env, vexpr );
	}
    } else {
	parallel_loop( VID(0), m, 1, [&]( VID k ) {
	    VID v = s[k];
	    intT d = idx[v+1]-idx[v];
	    if( __builtin_expect( d < 1000, 1 ) ) {
		process_csr_sparse<need_atomic,um_none>(
		    /*eid_retriever,*/ &edge[idx[v]], idx[v], d, v,
		    nullptr, nullptr, c, env, vexpr );
	    } else {
		// Note: it is highly likely that running this in parallel
		//       will hardly ever occur, as in the sparse part high-degree
		//       vertices may have converged (e.g. PRDelta, maybe not BFS)
		process_csr_sparse_parallel<need_atomic,um_none>(
		    /*eid_retriever,*/ &edge[idx[v]], idx[v],
		    d, v, nullptr, nullptr, c, env, vexpr );
	    }
	} );
    }

    // return
    return frontier::all_true( GA.numVertices(), GA.numEdges() );
}


#endif // GRAPTOR_DSL_EDGEMAP_H
