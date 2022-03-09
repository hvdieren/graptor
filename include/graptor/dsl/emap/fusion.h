// -*- c++ -*-
#ifndef GRAPTOR_DSL_EMAP_FUSION_H
#define GRAPTOR_DSL_EMAP_FUSION_H

#include "graptor/partitioner.h"
#include "graptor/graph/GraphCSx.h"

#include "graptor/dsl/simd_vector.h"
#include "graptor/dsl/ast.h"
#include "graptor/dsl/vertexmap.h"

#include "graptor/dsl/comp/rewrite_redop_to_store.h"
#include "graptor/dsl/comp/rewrite_internal.h"
#include "graptor/dsl/comp/is_idempotent.h"
#include "graptor/dsl/comp/is_benign_race.h"

#include "graptor/dsl/emap/work_list.h"

template<bool zerof, bool need_strong_checking,
	 typename config, typename WorkQueues, typename Operator>
std::vector<VID> csr_sparse_with_f_seq_fusion_stealing(
    config & cfg,
    const GraphCSx & GA,
    const partitioner & part,
    // frontier actv,
    WorkQueues & work_queues,
    unsigned self_id,
    bool * zf,
    Operator op ) {
    const EID * idx = GA.getIndex();

    unsigned char * uczf = (unsigned char*)zf;

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

    std::vector<VID> unprocessed;
    
    while( auto * buffer = work_queues.steal( self_id ) ) {
	auto I = buffer->begin();
	auto E = buffer->end();
	for( ; I != E; ++I ) {
	    VID v = *I;
	    VID d = idx[v+1] - idx[v];
	    const VID * out = &edge[idx[v]];

	    // Sequential, so dump in sequential locations in outEdges[]
	    auto src = simd::template create_scalar<simd::ty<VID,1>>( v );

	    for( VID j=0; j < d; j++ ) {
		// EID seid = eid_retriever.get_edge_eid( v, j );
		EID seid = idx[v]+j;
		VID sdst = out[j];
		auto dst = simd::template create_scalar<simd::ty<VID,1>>( sdst );
		auto eid = simd::template create_scalar<simd::ty<EID,1>>( seid );
		auto m = expr::create_value_map_new<1>(
		    expr::create_entry<expr::vk_dst>( dst ),
		    expr::create_entry<expr::vk_edge>( eid ),
		    expr::create_entry<expr::vk_src>( src ) );

		// Set evaluator to use atomics
		// TODO: can drop atomics in case of benign races
		auto ret = env.template evaluate<true>( c, m, vexpr );

		if( ret.value().data() ) {
		    // Query whether to process immediately. Note: the fusion
		    // operation may have side-effects, and is executed with
		    // atomics. However, it is not executed atomically with the
		    // relax operation!
		    auto immediate_val = env.template evaluate<true>( c, m, fexpr );
		    int immediate = immediate_val.value().data();

		    if( immediate > 0 ) {
			// Process now. Differentiate idempotent operators
			// to avoid the repeated processing book-keeping.
			if constexpr ( need_strong_checking ) {
			    // A vertex may first be unprocessed,
			    // then on a later edge become immediately ready.
			    // Record differently.
			    unsigned char flg = 2;
			    unsigned char val = uczf[sdst];
			    if( ( val & flg ) == 0 ) {
				unsigned char old
				    = __sync_fetch_and_or( &uczf[sdst], flg );
				if( ( old & flg ) == 0 ) {
				    // outEdges[nactv++] = sdst;
				    work_queues.push( self_id, sdst );
				}
			    }
			} else {
			    // A vertex may be processed multiple times, by
			    // multiple threads, possibly concurrently.
			    // Multiple threads may insert the vertex in their
			    // active list, then concurrent processing may occur.
			    // outEdges[nactv++] = sdst;
			    work_queues.push( self_id, sdst );
			}
		    } else if( immediate == 0 ) {
			// Include in unprocessed list, if not already done so
			unsigned char flg = 1;
			unsigned char val = uczf[sdst];
			if( ( val & flg ) == 0 ) {
			    unsigned char old
				= __sync_fetch_and_or( &uczf[sdst], flg );
			    if( ( old & (unsigned char)3 ) == 0 )
				unprocessed.push_back( sdst );
			}
		    }
		}
	    }
	}
	buffer->destroy();
    }

    return unprocessed;
}

template<bool zerof, typename config, typename EIDRetriever, typename Operator>
static __attribute__((noinline)) frontier csr_sparse_with_f_fusion_stealing(
    config & cfg,
    const GraphCSx & GA,
    const EIDRetriever & eid_retriever,
    const partitioner & part,
    frontier & old_frontier,
    bool * zf,
    Operator op ) {

    // TODO: Address initial inefficiency when the initial frontier is
    // very small, as it will leave many threads idle initially.
    assert( cfg.is_parallel() && "this will execute in parallel" );

    // Instantiate relax operation only to check if it requires strict exclusion
    // of replica's of the activated vertices in the immediate or postponed
    // vertex lists.
    auto vexpr0 = op.relax( expr::value<simd::ty<VID,1>,expr::vk_src>(),
			   expr::value<simd::ty<VID,1>,expr::vk_dst>(),
			   expr::value<simd::ty<EID,1>,expr::vk_edge>() );
    auto uexpr0 = op.update( expr::value<simd::ty<VID,1>,expr::vk_dst>() );
    auto fexpr = op.fusionop( expr::value<simd::ty<VID,1>,expr::vk_dst>() );
    auto vexpr = vexpr0 && uexpr0;
    constexpr bool need_strong_checking
		  = !( expr::is_idempotent<decltype(fexpr)>::value
		       && expr::is_idempotent<decltype(vexpr)>::value );

    // Split the list of active vertices over the number of threads, and
    // have each thread run through multiple iterations until complete.
    VID m = old_frontier.nActiveVertices();
    VID * s = old_frontier.getSparse();

    VID num_threads = graptor_num_threads();
    work_stealing<VID> work_queues;
    
    std::vector<VID> * F = new std::vector<VID>[num_threads]();

    parallel_for( uint32_t t=0; t < num_threads; ++t ) {
	VID from = ( t * m ) / num_threads;
	VID to = t == num_threads ? m : ( (t + 1) * m ) / num_threads;
	work_queues.create_buffer( t, &s[from], to-from );
	F[t] =
	    csr_sparse_with_f_seq_fusion_stealing<zerof,need_strong_checking>(
		cfg, GA, part, work_queues, t, zf, op );
    }

    // Restore zero frontier to all zeros; do not access old frontiers copied
    // in as we did not set their flag in the loop above
    if constexpr ( zerof ) {
	if constexpr ( need_strong_checking ) {
	    // Vertices already processed leave info in zf[] that is not
	    // easily cleared. So we need to traverse the whole array.
	    parallel_for( uint32_t t=0; t < num_threads; ++t ) {
		// Strange stuff with zf check, compiler erroneously
		// evading it because it looks like bool(?). Hence the cast.
		unsigned char * uczf = (unsigned char*)zf;
		VID outEdgeCount = F[t].size();
		VID * outEdges = &F[t][0];
		// If a vertex has been processed immediately, then any copy
		// introduced previously for postponed processing can be
		// removed.
		VID nv_new = 0;
		for( VID k=0; k < outEdgeCount; ++k ) {
		    if( uczf[outEdges[k]] & (unsigned char)2 ) {
			; // outEdges[k] = ~(VID)0;
		    } else
			outEdges[nv_new++] = outEdges[k];
		}
		F[t].resize( nv_new );
	    }

	    VID n = GA.numVertices();
	    parallel_for( VID v=0; v < n; ++v )
		zf[v] = false;
	} else {
	    // Vertices already processed leave no info in zf[]
	    parallel_for( uint32_t t=0; t < num_threads; ++t ) {
		VID outEdgeCount = F[t].size();
		VID * outEdges = &F[t][0];
		for( VID k=0; k < outEdgeCount; ++k )
		    zf[outEdges[k]] = false;
	    }
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

    // We likely don't need an accurate count of active edges
    merged.setActiveCounts( nactv, nactv );
    return merged;
}

#endif // GRAPTOR_DSL_EMAP_FUSION_H
