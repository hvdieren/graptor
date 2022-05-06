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

#include "graptor/target/vector.h"
#include "graptor/dsl/simd_vector.h"
#include "graptor/dsl/ast.h"
#include "graptor/dsl/vertexmap.h"

#include "graptor/dsl/comp/rewrite_redop_to_store.h"
#include "graptor/dsl/comp/rewrite_internal.h"
#include "graptor/dsl/comp/is_idempotent.h"

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
// #include "graptor/dsl/emap/GraptorCSCDataParNotCached.h"
// #include "graptor/dsl/emap/GraptorCSRVPushCached.h"
// #include "graptor/dsl/emap/GraptorCSRVPushNotCached.h"
#include "graptor/dsl/emap/GraptorCSRDataParCached.h"
#include "graptor/dsl/emap/GraptorCSRDataParNotCached.h"

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
	// << ", is_idempotent: " << Operator::is_idempotent
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
}


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

#if 0
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
		if( *zf ) // already set, don't set twice
		    *f = ~(VID)0;
		else if( updated ) { // set true, first time
		    *f = d;
		    *zf = true;
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

template<typename Operator, bool V = false>
struct emap_do_add_in_frontier_out_helper {
    static constexpr bool value = V;
};

template<typename Operator, typename Enable = void>
struct emap_do_add_in_frontier_out
    : emap_do_add_in_frontier_out_helper<Operator,false> { };

template<typename Operator>
struct emap_do_add_in_frontier_out<
    Operator,std::enable_if_t<has_add_in_frontier_out<Operator>::value>>
    : emap_do_add_in_frontier_out_helper<
    Operator,Operator::add_in_frontier_out> { };


template<bool setf, bool zerof, typename EIDRetriever, typename Expr>
static DBG_NOINLINE void
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
	auto ret = expr::evaluate<true>( c, m, e );
	conditional_set<setf,zerof,true>( &frontier[j], &zf[dst.data()],
					  ret.value().data(), dst.data() );
    }
}

template<bool setf, bool zerof, typename EIDRetriever, typename Expr>
static DBG_NOINLINE void
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
	auto ret = expr::evaluate<true>( c, m, e );
	conditional_set<setf,zerof,true>( &frontier[j], &zf[dst.data()],
					  ret.value().data(), dst.data() );
    }
}


// Sparse CSR traversal (GraphCSx) with a new frontier
template<typename config, typename EIDRetriever, typename Operator>
static __attribute__((noinline)) frontier csr_sparse_with_f(
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
    EID outEdgeCount = sequence::plusScan(offsets, degrees, m);
    if constexpr ( emap_do_add_in_frontier_out<Operator>::value )
	outEdgeCount += m; // Add in m vertices in old frontier
    VID* outEdges = new VID [outEdgeCount];

    // CSR requires no check that destination is active.
    auto vexpr0 = op.relax( expr::value<simd::ty<VID,1>,expr::vk_src>(),
			    expr::value<simd::ty<VID,1>,expr::vk_dst>(),
			    expr::value<simd::ty<EID,1>,expr::vk_edge>() );

    // Match vector lengths and move masks
    auto vexpr = expr::rewrite_mask_main( vexpr0 );

    const VID *edge = GA.getEdges();

    if constexpr ( !std::decay_t<config>::is_parallel() )  {
	for( VID k = 0; k < m; k++ ) {
	    VID v = s[k];
	    EID o = offsets[k];
	    intT d = idx[v+1]-idx[v];
	    process_csr_sparse<true,false>( eid_retriever, &edge[idx[v]], d, v,
					    &outEdges[o], nullptr, vexpr );
	}
    } else {
	parallel_for( VID k = 0; k < m; k++ ) {
	    VID v = s[k];
	    EID o = offsets[k];
	    intT d = idx[v+1]-idx[v];
	    if( __builtin_expect( d < 1000, 1 ) ) {
		process_csr_sparse<true,false>(
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
	nextM = sequence::filter( outEdges, nextIndices, outEdgeCount,
				  ValidVID() );
    } else if( outEdgeCount > GA.numEdges() / 1000 ) {
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
    // tmrd = tmd.next();
    new_frontier.calculateActiveCounts( GA, part, nextM );
    // double tmra = tmd.stop();

    // Clean up and return
    delete [] outEdges;
    delete [] degrees;
// double tmr = tm.stop();
// std::cerr << "csr_sparse_with_f nv=" << m << " noute=" << outEdgeCount << ' ' << tmr << " of which dedup: " << tmrd << " of which count: " << tmra << "\n";
    
    return new_frontier;
}

// Sparse CSR traversal (GraphCSx) with a new frontier
template<typename config, typename EIDRetriever, typename Operator>
static __attribute__((noinline)) frontier csr_sparse_with_f_new(
    config & cfg,
    const GraphCSx & GA,
    const EIDRetriever & eid_retriever,
    const partitioner & part,
    frontier & old_frontier,
    Operator op ) {
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
    EID outEdgeCount = sequence::plusScan(offsets, degrees, m);
    if constexpr ( emap_do_add_in_frontier_out<Operator>::value )
	outEdgeCount += m; // Add in m vertices in old frontier
    VID* outEdges = new VID [outEdgeCount];

    // CSR requires no check that destination is active.
    auto vexpr0 = op.relax( expr::value<simd::ty<VID,1>,expr::vk_src>(),
			    expr::value<simd::ty<VID,1>,expr::vk_dst>(),
			    expr::value<simd::ty<EID,1>,expr::vk_edge>() );

    // Match vector lengths and move masks
    auto vexpr = expr::rewrite_mask_main( vexpr0 );

    const VID *edge = GA.getEdges();

    static constexpr bool zerof = !expr::is_idempotent<decltype(vexpr)>::value;
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
	bool * zf = zero_frontier->template getDense<frontier_type::ft_bool>();
    }

    if constexpr ( !std::decay_t<config>::is_parallel() )  {
	for( VID k = 0; k < m; k++ ) {
	    VID v = s[k];
	    EID o = offsets[k];
	    intT d = idx[v+1]-idx[v];
	    process_csr_sparse<true,zerof>(
		eid_retriever, &edge[idx[v]], d, v,
		&outEdges[o], zf, vexpr );
	}
    } else {
	parallel_for( VID k = 0; k < m; k++ ) {
	    VID v = s[k];
	    EID o = offsets[k];
	    intT d = idx[v+1]-idx[v];
	    if( __builtin_expect( d < 1000, 1 ) ) {
		process_csr_sparse<true,zerof>(
		    eid_retriever, &edge[idx[v]], d, v,
		    &outEdges[o], zf, vexpr );
	    } else {
		// Note: it is highly likely that running this in parallel
		//       will hardly ever occur, as in the sparse part
		//       high-degree vertices may have converged (e.g.
		//       PRDelta, maybe not BFS)
		process_csr_sparse_parallel<true,zerof>(
		    eid_retriever, &edge[idx[v]],
		    d, v, &outEdges[o], zf, vexpr );
	    }
	}
    }

    // Restore zero frontier to all zeros
    if constexpr ( zerof ) {
	for( VID k=0; k < outEdgeCount-m; ++k )
	    zf[outEdges[k]] = false;
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
    VID nextM = 0;

    // Note: there are no duplicates if zerof == true
    if constexpr ( expr::is_idempotent<decltype(vexpr)>::value ) {
	// Operation is idempotent, i.e., we are allowed to process each
	// edge multiple times. No need to remove duplicates. We will filter
	// 'empty' slots (marked -1).
	nextM = sequence::filter( outEdges, nextIndices, outEdgeCount,
				  ValidVID() );
    } else if constexpr( zerof ) {
	// There are no duplicates by design
	// Filter out the empty slots (marked with -1)
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

    // Match vector lengths and move masks
    auto vexpr = expr::rewrite_mask_main( vexpr0 );

    const VID *edge = GA.getEdges();

    if constexpr ( !std::decay_t<config>::is_parallel() ) {
	for( VID k = 0; k < m; k++ ) {
	    VID v = s[k];
	    intT d = idx[v+1]-idx[v];
	    process_csr_sparse<false,false>( eid_retriever, &edge[idx[v]], d, v,
					     nullptr, nullptr, vexpr );
	}
    } else {
	parallel_for( VID k = 0; k < m; k++ ) {
	    VID v = s[k];
	    intT d = idx[v+1]-idx[v];
	    if( __builtin_expect( d < 1000, 1 ) ) {
		process_csr_sparse<false,false>(
		    eid_retriever, &edge[idx[v]], d, v,
		    nullptr, nullptr, vexpr );
	    } else {
		// Note: it is highly likely that running this in parallel
		//       will hardly ever occur, as in the sparse part high-degree
		//       vertices may have converged (e.g. PRDelta, maybe not BFS)
		process_csr_sparse_parallel<false,false>(
		    eid_retriever, &edge[idx[v]],
		    d, v, nullptr, nullptr, vexpr );
	    }
	}
    }

    // return
    return frontier::all_true( GA.numVertices(), GA.numEdges() );
}


#endif // GRAPTOR_DSL_EDGEMAP_H
