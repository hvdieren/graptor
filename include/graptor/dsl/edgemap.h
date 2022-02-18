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

#include "graptor/frontier.h"
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

#include "graptor/frontier_impl.h"
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

#if 0
template<typename GraphType, typename EdgeOperator,typename EMapConfig>
template<graph_traversal_kind gtk>
void step_emap_dense<GraphType,EdgeOperator,EMapConfig,true>::execute(
    const partitioner & part ) {
    // Not sure what it means here if part != m_G.get_partitioner()

    emap_report_dyn<GraphType,EdgeOperator,EMapConfig,gtk>( std::cout );

    if constexpr ( gtk == graph_traversal_kind::gt_push )
	emap_push<EMapConfig>( getGraph(), get_operator(), part );
    else if constexpr ( gtk == graph_traversal_kind::gt_pull )
	emap_pull<EMapConfig>( getGraph(), get_operator(), part );
    else if constexpr ( gtk == graph_traversal_kind::gt_ireg )
	emap_ireg<EMapConfig>( getGraph(), get_operator(), part );
    else if constexpr ( gtk == graph_traversal_kind::gt_sparse )
	; // nothing to do
    else
	UNREACHABLE_CASE_STATEMENT;

    std::cout << "done emap dense\n";
}

template<typename GraphType,
	 typename PushOperator, typename PullOperator, typename IRegOperator>
void
step_emap_dense3<GraphType,PushOperator,PullOperator,IRegOperator>::execute(
    const partitioner & part ) {
    // Not sure what it means here if part != m_G.get_partitioner()

    switch( m_kind ) {
    case graph_traversal_kind::gt_push:
	m_push.template execute<graph_traversal_kind::gt_push>( part ); 
	break;
    case graph_traversal_kind::gt_pull:
	m_pull.template execute<graph_traversal_kind::gt_pull>( part ); 
	break;
    case graph_traversal_kind::gt_ireg:
	m_ireg.template execute<graph_traversal_kind::gt_ireg>( part ); 
	break;
    case graph_traversal_kind::gt_sparse:
	// nothing to do
	break;
    default:
	UNREACHABLE_CASE_STATEMENT;
    }
    
    // emap_report_dyn<GraphType,Operator,EMapConfig>( std::cout );

}
#endif



struct ValidVID
{
    bool operator() ( VID a ) {
        return a != ~(VID)0;
    }
};


template<bool setf, bool zerof, bool once>
void conditional_set( VID * f, bool * zf, bool updated, VID d ) {
    if constexpr ( setf ) {
	if constexpr ( once ) { // at least initialise *f
	    if constexpr ( zerof ) {
		if( updated ) { // set true
		    if( !*zf && __sync_fetch_and_or( (unsigned char *)zf,
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


template<bool setf, typename Expr, typename Cache, typename Environment>
static DBG_NOINLINE void
process_csc_sparse_aset( const VID *out, VID deg, VID p, VID dstv, VID *fr,
			 const Expr & e, const Cache & vcaches,
			 const Environment & env ) {
    static_assert( Expr::VL == 1, "Sparse traversal requires VL == 1" );

    auto dst = simd::template create_constant<simd::ty<VID,1>>( dstv );

    // Cache containing local variables only
    auto c = expr::cache_create( env, vcaches, expr::create_value_map_new<1>() );

    // using output_type = simd::container<simd::ty<typename Expr::type, 1>>;
    using output_type = simd::container<typename Expr::data_type>;
    auto output = output_type::false_mask();

    auto pvec1 = simd::template create_constant<simd::ty<VID,1>>( 8*(VID)p );

    for( VID j=0; j < deg; j++ ) {
	auto src = simd::template load_from<simd::ty<VID,1>>( &out[j] );
	auto m = expr::create_value_map_new<1>(
	    // expr::create_entry<expr::vk_pid>( pvec1 ),
	    expr::create_entry<expr::vk_dst>( dst ),
	    expr::create_entry<expr::vk_src>( src ) );
	// Note: CSC, sequential by degree, no atomics required
	auto ret = env.template evaluate<false>( c, m, e );
	if constexpr ( setf && !std::is_same_v<decltype(ret.value()),nomask<1>> )
	    output.lor_assign( ret.value() );
    }
    conditional_set<setf,false,true>( fr, nullptr, output.data(), dstv );
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

    parallel_for( VID j=0; j < deg; j++ ) {
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
    }
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
		&edge[idx[v]], d, 0 /*part.part_of(v)*/, v, &o[k], vexpr, vcaches, env );
	}
    } else {
	parallel_for( VID k = 0; k < m; k++ ) {
	    VID v = s[k];
	    intT d = idx[v+1]-idx[v];
	    // We cannot actually do parallel processing of neighbours as we
	    // have not privatized the array elements corresponding to the
	    // destination vertex. Vectorisation however may be feasible
	    // for very high-degree vertices.
	    o[k] = ~(VID)0;
	    process_csc_sparse_aset<true>(
		&edge[idx[v]], d, 0 /*part.part_of(v)*/, v, &o[k], vexpr, vcaches, env );
	}
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
	parallel_for( VID k = 0; k < m; k++ ) {
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
	}
    }

    // Filter out the empty slots (marked with -1)
    frontier output = frontier::sparse( GA.numVertices(), m );
    VID * oo = output.getSparse();
    VID m_valid = sequence::filter( o, oo, m, ValidVID() );
    status.del();
    output.calculateActiveCounts( GA, part, m_valid );

    return output;
}

// Sparse CSR traversal (GraphCSx) with no new frontier calculated
//template<typename config, typename Operator>
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

    // Rewrite local variables
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
	    process_csc_sparse_aset<false>( &edge[idx[v]], d, 0 /*part.part_of(v)*/, v, nullptr, vexpr,
					    vcaches, env );
	}
    } else {
	parallel_for( VID k = 0; k < m; k++ ) {
	    VID v = s[k];
	    intT d = idx[v+1]-idx[v];
	    // We cannot actually do parallel processing as we have not
	    // privatized the array elements corresponding to the destination
	    // vertex. Vectorisation however may be feasible for very
	    // high-degree vertices.
	    process_csc_sparse_aset<false>( &edge[idx[v]], d, 0 /*part.part_of(v)*/, v, nullptr, vexpr,
					    vcaches, env );
	}
    }

    // return
    return frontier::all_true( GA.numVertices(), GA.numEdges() );
}

// Note: normally, we would need this method only for sparse frontiers,
//       where m ~ n and so fits in a VID rather than an EID. We need to check
//       this however.
inline void removeDuplicates( VID * ids, EID me, const partitioner & part ) {
    static mmap_ptr<VID> flags;

    VID m = (VID) me;
    assert( me == (EID)m && "length of array should be small enough" );

    // Allocate working memory. Try to retain it across calls.
    VID n = part.get_num_elements();
    if( flags.get_length() != n ) {
	if( flags.get_length() > 0 )
	    flags.del();
	flags.allocate( numa_allocation_partitioned( part ) );
	parallel_for( VID i=0; i<n; i++ ) flags[i] = (VID)-1;
    }

    // Figure out which vertices are listed in ids
    parallel_for( VID i=0; i < m; i++ ) {
	if( ids[i] != (VID)-1 && flags[ids[i]] == (VID)-1 )
	    CAS( &flags[ids[i]], (VID)-1, i );
    }

    // Cancel out duplicates
    parallel_for( VID i=0; i < m; i++ ) {
	if( ids[i] != (VID)-1 ) {
	    if( flags[ids[i]] == i ) {  // win
		flags[ids[i]] = (VID)-1; //reset
	    } else
		ids[i] = (VID)-1; // lose
	}
    }
    // Flags is restored to all -1 now
    // TODO: we could now already know how many distinct values there are
    //       by counting them above, which can speedup the subsequent filter op.
}

inline VID removeDuplicatesAndFilter_seq( VID * ids, EID me,
				   VID * tgt, const partitioner & part ) {
    VID m = (VID) me;
    assert( me == (EID)m && "length of array should be small enough" );

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
	parallel_for( VID i=0; i<n; i++ ) flags[i] = (VID)-1;

	// Figure out which vertices are listed in ids
	VID i = 0;
	VID t = 0;
	while( i < m ) {
	    if( ids[i] == ~VID(0) ) {
		ids[i] = ids[--m];
	    } else if( flags[ids[i]] == ~VID(0) ) {
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

	if( ids[0] == ~VID(0) ) // Assuming sort is unsigned, ~VID(0) at back
	    return 0;

	tgt[0] = ids[0];

	VID t=1;
	for( VID s=1; s < m; ++s ) {
	    if( ids[s] == ids[s-1] ) {
		// next
	    } else if( ids[s] == ~VID(0) ) {
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
	    if( ids[s] == ~VID(0) )
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
	 typename EIDRetriever, typename Environment, typename Expr>
static void
process_csr_sparse( const EIDRetriever & eid_retriever,
		    const VID *out, VID deg, VID srcv, VID *frontier,
		    bool *zf, const Environment & env, const Expr & e ) {
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
	auto ret = env.template evaluate<atomic>( c, m, e );
	conditional_set<setf,zerof,true>( &frontier[j], &zf[dst.data()],
					  ret.value().data(), dst.data() );
    }
}

template<bool atomic, bool setf, bool zerof, typename EIDRetriever,
	 typename Environment, typename Expr>
static void
process_csr_sparse_parallel( const EIDRetriever & eid_retriever,
			     const VID *out, VID deg, VID srcv, VID *frontier,
			     bool *zf,
			     const Environment & env,
			     const Expr & e ) {
    tuple<> c; // empty cache

    static_assert( Expr::VL == 1, "Sparse traversal requires VL == 1" );

    parallel_for( VID j=0; j < deg; j++ ) {
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
    }
}

template<bool atomic, bool setf, bool zerof,
	 typename Environment, typename Expr>
static void
process_csr_sparse( const VID *out, EID be, VID deg, VID srcv, VID *frontier,
		    bool *zf, const Environment & env, const Expr & e ) {
    tuple<> c; // empty cache

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
	conditional_set<setf,zerof,true>( &frontier[j], &zf[dst.data()],
					  ret.value().data(), dst.data() );
    }
}

template<bool atomic, bool setf, bool zerof,
	 typename Environment, typename Expr>
static void
process_csr_sparse_parallel( const VID *out, EID be,
			     VID deg, VID srcv, VID *frontier,
			     bool *zf,
			     const Environment & env,
			     const Expr & e ) {
    tuple<> c; // empty cache

    static_assert( Expr::VL == 1, "Sparse traversal requires VL == 1" );

    parallel_for( VID j=0; j < deg; j++ ) {
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
	conditional_set<setf,zerof,true>( &frontier[j], &zf[dst.data()],
					  ret.value().data(), dst.data() );
    }
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

    parallel_for( VID j=0; j < deg; j++ ) {
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
    }
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

	parallel_for( VID i=0; i < nextM; ++i ) {
	    VID v = nextIndices[i];
	    auto vv = simd::template create_unknown<simd::ty<VID,VL>>( v );
	    auto m = expr::create_value_map_new<VL>(
		expr::create_entry<expr::vk_dst>( vv ) );
	    auto c = cache_create( env, l_cache, m );
	    auto ret = env.evaluate( c, m, expr );
	    if( !ret.value().data() )
		nextIndices[i] = ~(VID)0;
	}
    }
}

template<bool zerof, typename config, typename EIDRetriever, typename Operator>
static __attribute__((noinline)) frontier csr_sparse_with_f_seq(
    config & cfg,
    const GraphCSx & GA,
    const EIDRetriever & eid_retriever,
    const partitioner & part,
    frontier & old_frontier,
    bool * zf,
    Operator op ) {
    const EID * idx = GA.getIndex();
    VID m = old_frontier.nActiveVertices();
    VID * s = old_frontier.getSparse();

    // Calculate maximum number of activated out-edges
    EID outEdgeCount = 0;
    for( VID i=0; i < m; ++i )
	outEdgeCount += idx[s[i]+1] - idx[s[i]];

    if constexpr ( emap_do_add_in_frontier_out<Operator>::value )
	outEdgeCount += m; // Add in m vertices in old frontier
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

    auto ew_pset = expr::create_map2<expr::vk_eweight>(
	GA.getWeights() ? GA.getWeights()->get() : nullptr );
					 
    auto env = expr::eval::create_execution_environment_with(
	op.get_ptrset( ew_pset ), l_cache, vexpr );

    tuple<> c; // empty cache

    const VID *edge = GA.getEdges();

    VID nactv = 0;
    EID nacte = 0;
    for( VID k = 0; k < m; k++ ) {
	VID v = s[k];
	VID d = idx[s[k]+1] - idx[s[k]];
	const VID * out = &edge[idx[s[k]]];

	// Sequential, so dump in sequential locations in outEdges[]
	auto src = simd::template create_scalar<simd::ty<VID,1>>( v );

	for( VID j=0; j < d; j++ ) {
	    // EID seid = eid_retriever.get_edge_eid( v, j );
	    EID seid = idx[s[k]]+j;
	    auto dst = simd::template load_from<simd::ty<VID,1>>( &out[j] );
	    auto eid = simd::template create_scalar<simd::ty<EID,1>>( seid );
	    auto m = expr::create_value_map_new<1>(
		expr::create_entry<expr::vk_dst>( dst ),
		expr::create_entry<expr::vk_edge>( eid ),
		expr::create_entry<expr::vk_src>( src ) );
	    // Note: sets evaluator to _not_ use atomics
	    auto ret = env.evaluate( c, m, vexpr );
	    if( ret.value().data() ) {
		bool add = false;
		VID dst = out[j];
		if constexpr ( zerof ) {
		    if( !zf[dst] ) {
			zf[dst] = true;
			add = true;
		    }
		} else
		    add = true;

		if( add ) {
		    outEdges[nactv++] = dst;
		    nacte += idx[dst+1] - idx[dst];
		}
	    }
	}
    }

    // assert( nactv <= outEdgeCount );

    outEdgeCount = nactv;

    // Add in vertices in old_frontier, if not present yet
    if constexpr ( emap_do_add_in_frontier_out<Operator>::value ) {
	static_assert( zerof, "missing code for !zerof" );
	for( VID r=0; r < m; ++r )
	    if( !zf[s[r]] )
		outEdges[nactv++] = s[r];
    }

    // Restore zero frontier to all zeros; do not access old frontiers copied
    // in as we did not set their flag in the loop above
    if constexpr ( zerof ) {
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

template<bool zerof, typename config, typename Operator>
frontier csr_sparse_with_f_seq_fusion(
    config & cfg,
    const GraphCSx & GA,
    const partitioner & part,
    frontier actv,
    std::vector<VID> & unprocessed,
    bool * zf,
    Operator op ) {
    const EID * idx = GA.getIndex();
    VID m = actv.nActiveVertices();
    VID * s = actv.getSparse();

    // Calculate maximum number of activated out-edges
    EID outEdgeCount = 0;
    for( VID i=0; i < m; ++i )
	outEdgeCount += idx[s[i]+1] - idx[s[i]];
    if constexpr ( emap_do_add_in_frontier_out<Operator>::value )
	outEdgeCount += m; // Add in m vertices in old frontier

    // Reserve space to store the activated vertices. As we don't know if
    // we will process them immediately with the fused operation, or if we
    // will postpone processing them, we reserve space for them twice.
    VID *outEdges = new VID[outEdgeCount];
    unprocessed.reserve( unprocessed.size() + outEdgeCount );

    // CSR requires no check that destination is active.
    auto vexpr0 = op.relax( expr::value<simd::ty<VID,1>,expr::vk_src>(),
			    expr::value<simd::ty<VID,1>,expr::vk_dst>(),
			    expr::value<simd::ty<EID,1>,expr::vk_edge>() );
    auto uexpr0 = op.update( expr::value<simd::ty<VID,1>,expr::vk_dst>() );
    auto fexpr0 = op.fusionop( expr::value<simd::ty<VID,1>,expr::vk_dst>() );

    // Append. TODO: ensure short-circuit evaluation in scalar execution
    // auto vexpr1 = expr::set_mask( vexpr0, uexpr0 );
    auto vexpr1 = vexpr0 && uexpr0;
    // auto vexpr1 = vexpr0;

    // Rewrite local variables
    auto l_cache_v = expr::extract_local_vars( vexpr1, expr::cache<>() );
    auto vexpr2 = expr::rewrite_caches<expr::vk_zero>( vexpr1, l_cache_v );

    auto l_cache_f = expr::extract_local_vars( fexpr0, expr::cache<>() );
    auto fexpr2 = expr::rewrite_caches<expr::vk_zero>( fexpr0, l_cache_f );

    auto l_cache = expr::cache_cat( l_cache_v, l_cache_f );

    // Match vector lengths and move masks
    auto vexpr = expr::rewrite_mask_main( vexpr2 );
    auto fexpr = expr::rewrite_mask_main( fexpr2 );

    auto ew_pset = expr::create_map2<expr::vk_eweight>(
	GA.getWeights() ? GA.getWeights()->get() : nullptr );
					 
    auto env = expr::eval::create_execution_environment_with(
	op.get_ptrset( ew_pset ), l_cache, vexpr, fexpr );

    auto mi = expr::create_value_map_new<1>();
    auto c = cache_create( env, l_cache, mi );

    const VID *edge = GA.getEdges();

    VID nactv = 0;
    for( VID k = 0; k < m; k++ ) {
	VID v = s[k];
	VID d = idx[s[k]+1] - idx[s[k]];
	const VID * out = &edge[idx[s[k]]];

	// Sequential, so dump in sequential locations in outEdges[]
	auto src = simd::template create_scalar<simd::ty<VID,1>>( v );

	for( VID j=0; j < d; j++ ) {
	    // EID seid = eid_retriever.get_edge_eid( v, j );
	    EID seid = idx[s[k]]+j;
	    auto dst = simd::template load_from<simd::ty<VID,1>>( &out[j] );
	    auto eid = simd::template create_scalar<simd::ty<EID,1>>( seid );
	    auto m = expr::create_value_map_new<1>(
		expr::create_entry<expr::vk_dst>( dst ),
		expr::create_entry<expr::vk_edge>( eid ),
		expr::create_entry<expr::vk_src>( src ) );

	    // Set evaluator to use atomics
	    auto ret = env.template evaluate<true>( c, m, vexpr );

	    if( ret.value().data() ) {
		// Query whether to process immediately
		auto immediate_val = env.evaluate( c, m, fexpr );
		bool immediate = immediate_val.value().data() ? true : false;

		VID dst = out[j];

		if( immediate ) {
		    // Process now. Differentiate idempotent operators
		    // to avoid the repeated processing book-keeping.
		    if constexpr (
			expr::is_idempotent<decltype(vexpr)>::value ) {
			// A vertex may be processed multiple times, by
			// multiple threads, possibly concurrently.
			// Multiple threads may insert the vertex in their
			// active list, then concurrent processing may occur.
			outEdges[nactv++] = dst;
		    } else {
			// A vertex may first be unprocessed,
			// then on a later edge become immediately ready.
			// Record differently.
			unsigned char flg = 2;
			unsigned char old
			    = __sync_fetch_and_or( (unsigned char *)&zf[dst], flg );
			if( ( old & flg ) == 0 )
			    outEdges[nactv++] = dst;
		    }
		} else {
		    // Include in unprocessed list, if not already done so
		    unsigned char flg = 1;
		    unsigned char old
			= __sync_fetch_and_or( (unsigned char *)&zf[dst], flg );
		    if( ( old & flg ) == 0 )
			unprocessed.push_back( dst );
		}
	    }
	}
    }

    // assert( nactv <= outEdgeCount );

    // Add in vertices in old_frontier, if not present yet
    if constexpr ( emap_do_add_in_frontier_out<Operator>::value ) {
	static_assert( zerof, "missing code for !zerof" );
	// static_assert( false, "questionable if this is safe here" );
	for( VID r=0; r < m; ++r )
	    if( !zf[s[r]] )
		outEdges[nactv++] = s[r];
    }

    // TODO: edge count is wrong in case of !zerof because
    //       new frontier may contain same vertex multiple times.
    frontier new_frontier
	= frontier::sparse( GA.numVertices(), nactv, outEdges );
    // We don't really care how many edges are activated, just duplicate
    // number of activated vertices to have zero/non-zero status
    new_frontier.setActiveCounts( nactv, (EID)nactv );

    return new_frontier;
}


template<bool zerof, typename config, typename Operator>
std::vector<VID> csr_sparse_with_f_seq_fusion_driver(
    config & cfg,
    const GraphCSx & GA,
    const partitioner & part,
    frontier actv,
    bool * zf,
    Operator op ) {
    // Assumptions
    static_assert( zerof, "Assuming presence of joint flag array for now" );

    std::vector<VID> unprocessed;
    bool first = true;
    
    // Repeat until work depleted
    while( !actv.isEmpty() ) {
	// F.first = vertices amenable for further processing
	// F.second = vertices not amenable for immediate processing
	frontier F = csr_sparse_with_f_seq_fusion<zerof>(
	    cfg, GA, part, actv, unprocessed, zf, op );

	// We are done with this
	if( first )
	    first = false;
	else
	    actv.del();

	// Next, process F.first
	actv = F;
    }

    if( !first )
	actv.del();

    return unprocessed;
}


template<bool zerof, typename config, typename EIDRetriever, typename Operator>
static __attribute__((noinline)) frontier csr_sparse_with_f_fusion(
    config & cfg,
    const GraphCSx & GA,
    const EIDRetriever & eid_retriever,
    const partitioner & part,
    frontier & old_frontier,
    bool * zf,
    Operator op ) {

    if( !cfg.is_parallel() )
	return csr_sparse_with_f_seq<zerof>(
	    cfg, GA, eid_retriever, part, old_frontier, zf, op );

    // Split the list of active vertices over the number of threads, and
    // have each thread run through multiple iterations until complete.
    VID m = old_frontier.nActiveVertices();
    VID * s = old_frontier.getSparse();

    VID num_threads = graptor_num_threads();
    std::vector<VID> * F = new std::vector<VID>[num_threads]();
    parallel_for( uint32_t t=0; t < num_threads; ++t ) {
	VID from = ( t * m ) / num_threads;
	VID to = t == num_threads ? m : ( (t + 1) * m ) / num_threads;
	frontier f = frontier::sparse( GA.numVertices(), to-from, &s[from] );
	f.setActiveCounts( to-from, to-from );
	F[t] = csr_sparse_with_f_seq_fusion_driver<zerof>(
	    cfg, GA, part, f, zf, op );
	// do not delete frontier f; doesn't own data
    }

    // Restore zero frontier to all zeros; do not access old frontiers copied
    // in as we did not set their flag in the loop above
    if constexpr ( zerof ) {
	if constexpr ( expr::is_idempotent_op<Operator>::value ) {
	    // Vertices already processed leave no info in zf[]
	    parallel_for( uint32_t t=0; t < num_threads; ++t ) {
		VID outEdgeCount = F[t].size();
		VID * outEdges = &F[t][0];
		for( VID k=0; k < outEdgeCount; ++k )
		    zf[outEdges[k]] = false;
	    }
	} else {
	    // Vertices already processed leave info in zf[] that is not
	    // easily cleared.
	    VID n = GA.numVertices();
	    parallel_for( VID v=0; v < n; ++v )
		zf[v] = false;
	}
    }

    // Tally all activated vertices
    VID nactv = 0;
    VID * inspt = new VID[num_threads];
    for( uint32_t t=0; t < num_threads; ++t ) {
	inspt[t] = nactv;
	nactv += F[t].size();
    }

    // Merge all the resultant frontiers (copy - could do in parallel)
    frontier merged = frontier::sparse( GA.numVertices(), nactv );
    s = merged.getSparse();
    parallel_for( uint32_t t=0; t < num_threads; ++t ) {
	std::copy( F[t].begin(), F[t].end(), &s[inspt[t]] );
    }

    // Cleanup. Deletes contents of vectors also.
    delete[] inspt;
    delete[] F;

    merged.setActiveCounts( nactv, nactv );
    return merged;
}


// Sparse CSR traversal (GraphCSx) with a new frontier
template<typename config, typename EIDRetriever, typename Operator>
static __attribute__((noinline)) frontier csr_sparse_with_f_old(
    config & cfg,
    const GraphCSx & GA,
    const EIDRetriever & eid_retriever,
    const partitioner & part,
    frontier & old_frontier,
    Operator op ) {
    // timer tm;
    // tm.start();

    const EID * idx = GA.getIndex();
    VID m = old_frontier.nActiveVertices();
    EID * degrees = new EID[m];
    VID * s = old_frontier.getSparse();

    // By definition of sparsity, don't expect this loop to be worthwhile to
    // parallelize.
    if( m > 4096 )
	parallel_for( VID i=0; i < m; ++i )
	    degrees[i] = idx[s[i]+1] - idx[s[i]];
    else
	for( VID i=0; i < m; ++i )
	    degrees[i] = idx[s[i]+1] - idx[s[i]];

    // TODO: retain degrees array for use in the main parallel_for
    //       and accumulate offsets in distinct array. This will improve
    //       memory locality in main parallel_for as idx[] won't need to be
    //       accessed again, which occurs in a sparse and unpredictable way.
    EID* offsets = degrees;
    EID outEdgeCount = sequence::plusScan(degrees, offsets, m);
    if constexpr ( emap_do_add_in_frontier_out<Operator>::value )
	outEdgeCount += m; // Add in m vertices in old frontier
    VID* outEdges = new VID [outEdgeCount];

    // CSR requires no check that destination is active.
    auto vexpr0 = op.relax( expr::value<simd::ty<VID,1>,expr::vk_src>(),
			    expr::value<simd::ty<VID,1>,expr::vk_dst>(),
			    expr::value<simd::ty<EID,1>,expr::vk_edge>() );

    // TODO: (needs environment in csr_process...)
    // TODO: give evaluate for set_mask short-cut semantics in scalar mode
    // auto uexpr0 = op.update( expr::value<simd::ty<VID,1>,expr::vk_dst>() );
    // auto vexpr1 = expr::set_mask( vexpr0, uexpr0 );
    auto vexpr1 = vexpr0;

    // Match vector lengths and move masks
    auto vexpr = expr::rewrite_mask_main( vexpr1 );

    const VID *edge = GA.getEdges();

    if constexpr ( !std::decay_t<config>::is_parallel() )  {
	for( VID k = 0; k < m; k++ ) {
	    VID v = s[k];
	    EID o = offsets[k];
	    intT d = idx[v+1]-idx[v];
	    process_csr_sparse<false,true,false>(
		eid_retriever, &edge[idx[v]], d, v,
		&outEdges[o], nullptr, vexpr );
	}
    } else {
	parallel_for( VID k = 0; k < m; k++ ) {
	    VID v = s[k];
	    EID o = offsets[k];
	    intT d = idx[v+1]-idx[v];
	    if( __builtin_expect( d < 1000, 1 ) ) {
		process_csr_sparse<true,true,false>(
		    eid_retriever, &edge[idx[v]], d, v,
		    &outEdges[o], nullptr, vexpr );
	    } else {
		// Note: it is highly likely that running this in parallel
		//       will hardly ever occur, as in the sparse part
		//       high-degree vertices may have converged (e.g.
		//       PRDelta, maybe not BFS)
		process_csr_sparse_parallel<true,false>(
		    eid_retriever, &edge[idx[v]],
		    d, v, &outEdges[o], nullptr, vexpr );
	    }
	}
    }

    if constexpr ( emap_do_add_in_frontier_out<Operator>::value )
	for( VID r=0; r < m; ++r )
	    outEdges[outEdgeCount-m+r] = s[r];

    // Calculate the number of active edges
    frontier new_frontier = frontier::sparse( GA.numVertices(), outEdgeCount );
    VID* nextIndices = new_frontier.getSparse();
    // Currently not required for any of the algorithms. This is due to
    // converting the frontier to a dense one on every iteration, which
    // has the same function.
    // double tmrd= 0;
    // timer tmd;
    // tmd.start();
    VID nextM = 0;
    if constexpr ( expr::is_idempotent<decltype(vexpr)>::value ) {
	// Operation is idempotent, i.e., we are allowed to process each
	// edge multiple times. No need to remove duplicates. We will filter
	// 'empty' slots (marked -1).

	// Re-evaluate if vertices are active. Duplicate vertices will be
	// evaluated multiple times.
	csr_sparse_update_frontier( nextM, nextIndices, op );
	// Filter out the empty slots (marked with -1)
	nextM = sequence::filter( outEdges, nextIndices, outEdgeCount,
				  ValidVID() );
    } else if( outEdgeCount > GA.numEdges() / 1000 ) {
	// We have relatively many edges. Better to process this in parallel.
	removeDuplicates( outEdges, outEdgeCount, part );
	// Re-evaluate if vertices are active
	csr_sparse_update_frontier( nextM, nextIndices, op );
	// Filter out the empty slots (marked with -1)
	nextM = sequence::filter( outEdges, nextIndices, outEdgeCount,
				  ValidVID() );
    } else {
	// Re-evaluate if vertices are active. Duplicate vertices will be
	// evaluated multiple times.
	csr_sparse_update_frontier( nextM, nextIndices, op );

	// Remove duplicates sequentially
	nextM = removeDuplicatesAndFilter_seq( outEdges, outEdgeCount,
					       nextIndices, part );
    }

    new_frontier.calculateActiveCounts( GA, part, nextM );

    // tmrd = tmd.next();
    // new_frontier.calculateActiveCounts( GA, part, nextM );
    // double tmra = tmd.stop();

    // Clean up and return
    delete [] outEdges;
    delete [] degrees;
// double tmr = tm.stop();
// std::cerr << "csr_sparse_with_f nv=" << m << " noute=" << outEdgeCount << ' ' << tmr << " of which dedup: " << tmrd << " of which count: " << tmra << "\n";
    
    return new_frontier;
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
    Operator op ) {
    const EID * idx = GA.getIndex();
    VID m = old_frontier.nActiveVertices();
    EID * degrees = new EID[2*m];
    VID * s = old_frontier.getSparse();

    static constexpr bool zerof = true; // !expr::is_idempotent<decltype(vexpr)>::value;
    bool * zf = nullptr;
    if constexpr ( zerof ) {
	// A dense bool frontier initially all zero, returned to all zero
	extern frontier * zero_frontier;
	if( zero_frontier != nullptr
	    && zero_frontier->nVertices() != part.get_num_elements() ) {
	    zero_frontier->del();
	    delete zero_frontier;
	    zero_frontier = nullptr;
	}
	if( zero_frontier == nullptr ) {
	    zero_frontier = new frontier;
	    *zero_frontier
		= frontier::template create<frontier_type::ft_bool>( part );
	}
	zf = zero_frontier->template getDense<frontier_type::ft_bool>();
    }

    if constexpr ( api::has_fusion_op_v<Operator> )
	return csr_sparse_with_f_fusion<zerof>(
	    cfg, GA, eid_retriever, part, old_frontier, zf, op );

    if( m < 1024 || !cfg.is_parallel() )
	return csr_sparse_with_f_seq<zerof>(
	    cfg, GA, eid_retriever, part, old_frontier, zf, op );

    // By definition of sparsity, don't expect this loop to be worthwhile to
    // parallelize.
    if( m > 4096 )
	parallel_for( VID i=0; i < m; ++i )
	    degrees[i] = idx[s[i]+1] - idx[s[i]];
    else
	for( VID i=0; i < m; ++i )
	    degrees[i] = idx[s[i]+1] - idx[s[i]];

    // TODO: retain degrees array for use in the main parallel_for
    //       and accumulate offsets in distinct array. This will improve
    //       memory locality in main parallel_for as idx[] won't need to be
    //       accessed again, which occurs in a sparse and unpredictable way.
    EID* offsets = &degrees[m];
    EID outEdgeCountF = sequence::plusScan(degrees, offsets, m);
    EID outEdgeCount = outEdgeCountF;
    if constexpr ( emap_do_add_in_frontier_out<Operator>::value )
	outEdgeCount += m; // Add in m vertices in old frontier
    VID* outEdges = new VID[outEdgeCount];

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

    auto ew_pset = expr::create_map2<expr::vk_eweight>(
	GA.getWeights() ? GA.getWeights()->get() : nullptr );

    auto env = expr::eval::create_execution_environment_with(
	op.get_ptrset( ew_pset ), l_cache, vexpr );

    const VID *edge = GA.getEdges();

    constexpr bool need_atomic = !expr::is_benign_race<decltype(vexpr)>::value;

    parallel_for( VID k = 0; k < m; k++ ) {
	VID v = s[k];
	EID o = offsets[k];
	intT d = degrees[k];
	if( __builtin_expect( d < 1000, 1 ) ) {
	    process_csr_sparse<need_atomic,true,zerof>(
		&edge[idx[v]], idx[v], d, v,
		&outEdges[o], zf, env, vexpr );
	} else {
	    // Note: it is highly likely that running this in parallel
	    //       will hardly ever occur, as in the sparse part
	    //       high-degree vertices may have converged (e.g.
	    //       PRDelta, maybe not BFS)
	    process_csr_sparse_parallel<need_atomic,true,zerof>(
		&edge[idx[v]], idx[v],
		d, v, &outEdges[o], zf, env, vexpr );
	}
    }

    if constexpr ( emap_do_add_in_frontier_out<Operator>::value )
	for( VID r=0; r < m; ++r ) {
	    bool add = false;
	    if constexpr ( zerof ) {
		if( !zf[s[r]] )
		    add = true;
	    } else
		add = true;
	    outEdges[outEdgeCountF+r] = add ? s[r] : ~(VID)0;
	}

    // Restore zero frontier to all zeros
    if constexpr ( zerof ) {
	parallel_for( VID k=0; k < outEdgeCountF; ++k )
	    if( outEdges[k] != (VID)-1 ) {
		// assert( zf[outEdges[k]] );
		zf[outEdges[k]] = false;
	    }
    }

    // Calculate the number of active edges
    frontier new_frontier = frontier::sparse( GA.numVertices(), outEdgeCount );
    VID* nextIndices = new_frontier.getSparse();
    // Currently not required for any of the algorithms. This is due to
    // converting the frontier to a dense one on every iteration, which
    // has the same function.
    VID nextM = 0;

    // Note: there are no duplicates if zerof == true
    if constexpr ( expr::is_idempotent<decltype(vexpr)>::value || zerof ) {
	// Operation is idempotent, i.e., we are allowed to process each
	// edge multiple times. No need to remove duplicates. We will filter
	// 'empty' slots (marked -1).
	nextM = sequence::filter( outEdges, nextIndices, outEdgeCount,
				  ValidVID() );
    } else {
	// Default code assuming not idempotent and not zerof
	if( outEdgeCount > GA.numEdges() / 1000 ) {
	    // We have relatively many edges. Better to process this in parallel.
	    removeDuplicates( outEdges, outEdgeCount, part );
	    // Filter out the empty slots (marked with -1)
	    nextM = sequence::filter( outEdges, nextIndices, outEdgeCount,
				      ValidVID() );
	} else {
	    // Remove duplicates sequentially
	    nextM = removeDuplicatesAndFilter_seq( outEdges, outEdgeCount,
						   nextIndices, part );
	}
    }
    
    new_frontier.calculateActiveCounts( GA, part, nextM );

    // Clean up and return
    delete [] outEdges;
    delete [] degrees;
    
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

    auto ew_pset = expr::create_map2<expr::vk_eweight>(
	GA.getWeights() ? GA.getWeights()->get() : nullptr );

    auto env = expr::eval::create_execution_environment_with(
	op.get_ptrset( ew_pset ), l_cache, vexpr );

    const VID *edge = GA.getEdges();

    constexpr bool need_atomic = !expr::is_benign_race<decltype(vexpr)>::value;

    if constexpr ( !std::decay_t<config>::is_parallel() ) {
	for( VID k = 0; k < m; k++ ) {
	    VID v = s[k];
	    intT d = idx[v+1]-idx[v];
	    process_csr_sparse<false,false,false>(
		/*eid_retriever,*/ &edge[idx[v]], idx[v], d, v,
		nullptr, nullptr, env, vexpr );
	}
    } else {
	parallel_for( VID k = 0; k < m; k++ ) {
	    VID v = s[k];
	    intT d = idx[v+1]-idx[v];
	    if( __builtin_expect( d < 1000, 1 ) ) {
		process_csr_sparse<need_atomic,false,false>(
		    /*eid_retriever,*/ &edge[idx[v]], idx[v], d, v,
		    nullptr, nullptr, env, vexpr );
	    } else {
		// Note: it is highly likely that running this in parallel
		//       will hardly ever occur, as in the sparse part high-degree
		//       vertices may have converged (e.g. PRDelta, maybe not BFS)
		process_csr_sparse_parallel<need_atomic,false,false>(
		    /*eid_retriever,*/ &edge[idx[v]], idx[v],
		    d, v, nullptr, nullptr, env, vexpr );
	    }
	}
    }

    // return
    return frontier::all_true( GA.numVertices(), GA.numEdges() );
}


#endif // GRAPTOR_DSL_EDGEMAP_H
