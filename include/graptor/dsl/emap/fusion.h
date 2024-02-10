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

template<bool no_duplicate_reporting,
	 bool no_duplicate_processing,
	 bool no_reporting_processed,
	 typename config, typename WorkQueues, typename Operator>
std::vector<VID> csr_sparse_with_f_seq_fusion_stealing(
    config & cfg,
    const GraphCSx & GA,
    const partitioner & part,
    WorkQueues & work_queues,
    unsigned self_id,
    bool * zf,
    Operator op ) {
    const EID * idx = GA.getIndex();

    unsigned char * uczf = (unsigned char*)zf;

    // CSR requires no check that destination is active.
    auto vexpr0 = op.fusionop( expr::value<simd::ty<VID,1>,expr::vk_src>(),
			       expr::value<simd::ty<VID,1>,expr::vk_dst>(),
			       expr::value<simd::ty<EID,1>,expr::vk_edge>() );
    auto uexpr0 = op.update( expr::value<simd::ty<VID,1>,expr::vk_dst>() );

    // Append.
    // TODO: set_mask provides short-circuit operation; check it.
    // auto vexpr1 = expr::set_mask( vexpr0, uexpr0 );
    auto vexpr1 = vexpr0 && uexpr0;

    // Rewrite local variables
    auto l_cache = expr::extract_local_vars( vexpr1, expr::cache<>() );
    auto vexpr2 = expr::rewrite_caches<expr::vk_zero>( vexpr1, l_cache );

    // Match vector lengths and move masks
    auto vexpr = expr::rewrite_mask_main( vexpr2 );

    auto env = expr::eval::create_execution_environment_op(
	op, l_cache, GA.getWeights() ? GA.getWeights()->get() : nullptr );

    auto mi = expr::create_value_map_new<1>();
    auto c = cache_create( env, l_cache, mi );

    const VID *edge = GA.getEdges();

    std::vector<VID> unprocessed;
    
    while( auto * buffer = work_queues.steal( self_id ) ) {
	auto I = buffer->edge_begin( idx );
	auto E = buffer->edge_end( idx );
	for( ; I != E; ++I ) {
	    VID v;
	    EID seid;
	    std::tie( v, seid ) = *I;

	    // Sequential code, so dump in sequential locations in outEdges[]
	    auto src = simd::template create_scalar<simd::ty<VID,1>>( v );

	    // EID seid = eid_retriever.get_edge_eid( v, j );
	    VID sdst = edge[seid];
	    auto dst = simd::template create_scalar<simd::ty<VID,1>>( sdst );
	    auto eid = simd::template create_scalar<simd::ty<EID,1>>( seid );
	    auto m = expr::create_value_map_new<1>(
		expr::create_entry<expr::vk_dst>( dst ),
		expr::create_entry<expr::vk_edge>( eid ),
		expr::create_entry<expr::vk_src>( src ) );

	    // Set evaluator to use atomics
	    // TODO: can drop atomics in case of benign races.

	    // Query whether to process destination vertex immediately.
	    // _0: report in output frontier
	    // _1: execute immediately (add to work list)
	    // _1s: no action taken
	    auto immediate_val = env.template evaluate<true>( c, m, vexpr );
	    int immediate = immediate_val.value().data();

	    if( immediate > 0 ) {
		// Process now. Differentiate idempotent operators
		// to avoid the repeated processing book-keeping.
		if constexpr ( no_duplicate_processing
			       || no_reporting_processed ) {
		    // A vertex may first be unprocessed,
		    // then on a later edge become immediately ready.
		    // Record differently.
		    // TODO (error): situation may arise where the vertex
		    // needs to be pushed on the work-queue multiple times
		    // because it improves (again) its status
		    // (e.g. distance in DSSSP). In KCore, this would not
		    // happen because the bucket is discrete and one unit
		    // wide. Repeated processing would be detrimental.
		    unsigned char flg = 2;
		    unsigned char val = uczf[sdst];
		    // if( ( val & flg ) == 0 ) {
		    unsigned char old
			= __sync_fetch_and_or( &uczf[sdst], flg );
		    if constexpr ( no_duplicate_processing ) {
			if( ( old & flg ) == 0 )
			    work_queues.push( self_id, sdst );
		    } else {
			work_queues.push( self_id, sdst );
		    }
		} else {
		    // A vertex may be processed multiple times, by
		    // multiple threads, possibly concurrently.
		    // Multiple threads may insert the vertex in their
		    // active list, then concurrent processing may occur.
		    // Alternatively, the operator would not enable
		    // a vertex for processing more than once (e.g.,
		    // count_down, setif).
		    work_queues.push( self_id, sdst );
		    assert( v != sdst );
		}
	    } else if( immediate == 0 ) {
		// Include in unprocessed list, if not already done so
		if constexpr ( no_duplicate_reporting
			       || no_reporting_processed ) {
		    unsigned char flg = 1;
		    unsigned char val = uczf[sdst];
		    if( ( val & flg ) == 0 ) {
			unsigned char old
			    = __sync_fetch_and_or( &uczf[sdst], flg );
			if constexpr ( no_reporting_processed
				       && no_duplicate_reporting ) {
			    if( ( old & (unsigned char)3 ) == 0 )
				unprocessed.push_back( sdst );
			} else if constexpr ( no_reporting_processed ) {
			    if( ( old & (unsigned char)2 ) == 0 )
				unprocessed.push_back( sdst );
			} else { // no_duplicate_reporting
			    if( ( old & (unsigned char)1 ) == 0 )
				unprocessed.push_back( sdst );
			}
		    }
		} else {
		    unprocessed.push_back( sdst );
		    assert( v != sdst );
		}
	    }
	}
	work_queues.finished( self_id, buffer );
    }

    return unprocessed;
}

template<typename config, typename EIDRetriever, typename Operator>
static __attribute__((noinline)) frontier csr_sparse_with_f_fusion_stealing(
    config & cfg,
    const GraphCSx & GA,
    const EIDRetriever & eid_retriever,
    const partitioner & part,
    frontier & old_frontier,
    Operator op ) {

    if( old_frontier.nActiveVertices() == 0 )
	return frontier::empty();

    // TODO: Address initial inefficiency when the initial frontier is
    // very small, as it will leave many threads idle initially.
    assert( cfg.is_parallel() && "this will execute in parallel" );

#if 0
    // Instantiate relax operation only to check if it requires strict exclusion
    // of replica's of the activated vertices in the immediate or postponed
    // vertex lists.
    auto vexpr0 = op.relax( expr::value<simd::ty<VID,1>,expr::vk_src>(),
			   expr::value<simd::ty<VID,1>,expr::vk_dst>(),
			   expr::value<simd::ty<EID,1>,expr::vk_edge>() );
    auto uexpr0 = op.update( expr::value<simd::ty<VID,1>,expr::vk_dst>() );
    auto fexpr = op.fusionop( expr::value<simd::ty<VID,1>,expr::vk_src>(),
			      expr::value<simd::ty<VID,1>,expr::vk_dst>(),
			      expr::value<simd::ty<EID,1>,expr::vk_edge>() );
    auto vexpr = vexpr0 && uexpr0;
#endif
    // Avoiding duplicates:
    //          processed unprocessed fexpr-idempot vexpr-idempot single-trigger
    // DSSSP    false     true        true          true          false
    // BF       true(f)   true        true          true          false
    //   + frontier unique occurences (preferred, not necessary?)
    //   + can process same vertex multiple times
    //   + if vertex processed and later not -> include in frontier
    //     (no infection from processed => unprocessed)
    // BFSLVL   false(t)  false(t)    true          true          false
    //   + frontier unique occurences
    //   + can process same vertex multiple times
    //   + if vertex previously processed, can still report (won't occur, no
    //     need to check and stop)
    // CC       true(f)   true(f)     true; cst     true          false
    //   + frontier unique occurences
    //   + can process same vertex multiple times
    //   + if vertex previously processed, can still report (won't occur)
    // Should be false/false: (??)
    // KCore(mod) false   true        false         false         true
    //   + vertex can be processed multiple times
    //   + no duplicates reported in frontier
    //   + if processed, then K has bottomed out and not reported
    // GC       true      true        false         false         true
    //   + vertex can be processed multiple times
    //   + at the end, frontier should be empty
    //   + if processed, then K has bottomed out and not reported
    constexpr api::fusion_flags flags = op.get_fusion_flags();
    constexpr bool no_duplicate_processing
		  = api::is_set( flags,
				 api::fusion_flags::no_duplicate_processing );
    constexpr bool no_duplicate_reporting
		  = api::is_set( flags,
				 api::fusion_flags::no_duplicate_reporting );
    constexpr bool no_reporting_processed
		  = api::is_set( flags,
				 api::fusion_flags::no_reporting_processed );

    constexpr bool need_strong_checking
		  = no_duplicate_processing
		  || no_duplicate_reporting
		  || no_reporting_processed;
    bool * zf = nullptr;
    if constexpr ( need_strong_checking )
	zf = GA.get_flags( part );

    // Split the list of active vertices over the number of threads, and
    // have each thread run through multiple iterations until complete.
    VID m = old_frontier.nActiveVertices();
    VID * s = old_frontier.getSparse();

    // VID num_threads = m > 1024 ? graptor_num_threads() : 1;
    VID num_threads = graptor_num_threads();
    assert( num_threads > 1 && "fusion requires multi-threaded execution" );
    work_stealing<VID,no_duplicate_processing> work_queues( num_threads, GA );

    std::vector<VID> * F = new std::vector<VID>[num_threads]();

    // Partition the work. Note this is a vertex-balanced parallelisation
    parallel_loop( (uint32_t)0, num_threads, 1, [&]( uint32_t t ) {
	VID from = ( uint64_t(t) * uint64_t(m) ) / uint64_t(num_threads);
	VID to = std::min( uint64_t(m),
			   ( uint64_t(t + 1) * uint64_t(m) )
			   / uint64_t(num_threads) );
	work_queues.create_buffer( t, &s[from], to-from, GA.getIndex() );
	F[t] =
	    csr_sparse_with_f_seq_fusion_stealing<
		no_duplicate_reporting,
	    no_duplicate_processing,
	    no_reporting_processed>(
		cfg, GA, part, work_queues, t, zf, op );
    } );

    // Restore zero frontier to all zeros; do not access old frontiers copied
    // in as we did not set their flag in the loop above
    if constexpr ( need_strong_checking ) {
	if constexpr ( no_duplicate_reporting ) { // flags set when reporting
	    // Vertices already processed leave info in zf[] that is not
	    // easily cleared. So we need to traverse the whole array.
	    parallel_loop( (uint32_t)0, num_threads, 1, [&]( uint32_t t ) {
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
		    VID v = outEdges[k];
		    if( !( uczf[v] & (unsigned char)2 ) )
			outEdges[nv_new++] = v;
		    uczf[v] = 0;
		}
		F[t].resize( nv_new );
	    } );
	}

	VID n = GA.numVertices();
	map_vertexL( part, [&]( VID v ) { zf[v] = false; } );
/*
	uint64_t * zf64 = reinterpret_cast<uint64_t *>( zf );
	VID n64 = n / 8;
	parallel_for( VID v=0; v < n64; ++v )
	    zf64[v] = 0;
	for( VID v=n64*8; v < n; ++v )
	    zf[v] = false;
*/

	if constexpr ( no_duplicate_processing ) { // flags set when processing
// TODO: || no_reporting_processed
	    work_queues.shift_processed();
	    parallel_loop( (uint32_t)0, num_threads, 1, [&]( uint32_t t ) {
		while( auto * buffer = work_queues.steal( t ) ) {
		    auto I = buffer->vertex_begin();
		    auto E = buffer->vertex_end();
		    for( ; I != E; ++I ) {
			VID v = *I;
			zf[v] = false;
		    }
		    buffer->destroy();
		}
	    } );
	}
/*
	for( VID v=0; v < n; ++v ) {
	    if( zf[v] != 0 )
		std::cerr << "zf[" << v << "]=" << (unsigned int)zf[v] << "\n";
	}
*/
    } else {
	// Vertices already processed leave no info in zf[]
/*
	parallel_loop( (uint32_t)0, num_threads, 1, [&]( uint32_t t ) {
	    VID outEdgeCount = F[t].size();
	    VID * outEdges = &F[t][0];
	    for( VID k=0; k < outEdgeCount; ++k )
		zf[outEdges[k]] = false;
	} );
*/
    }

    // Tally all activated vertices
    VID nactv = 0;
    VID * inspt = new VID[num_threads];
    for( uint32_t t=0; t < num_threads; ++t ) {
	inspt[t] = nactv;
	nactv += F[t].size();
    }

    // Merge all the resultant frontiers and count out-edges
    // TODO: only possible if no duplicates in unprocessed list
    frontier merged = frontier::sparse( GA.numVertices(), nactv );
    s = merged.getSparse();
    EID * ecnt = new EID[num_threads]();
    parallel_loop( (uint32_t)0, num_threads, 1, [&]( uint32_t t ) {
	// if constexpr ( avoid_unprocessed_duplicates ) {
	    VID * out = &s[inspt[t]];
	    EID nedg = 0;
	    const EID * const index = GA.getIndex();
	    for( auto I=F[t].begin(), E=F[t].end(); I != E; ++I ) {
		VID v = *I;
		*out++ = v;
		nedg += index[v+1] - index[v];
	    }
	    ecnt[t] = nedg;
	// } else {
	    // std::copy( F[t].begin(), F[t].end(), &s[inspt[t]] );
	// }
    } );

    EID nacte = ecnt[0];
    for( uint32_t t=1; t < num_threads; ++t )
	nacte += ecnt[t];

    // Cleanup. Deletes contents of vectors also.
    delete[] ecnt;
    delete[] inspt;
    delete[] F;

    // We likely don't need an accurate count of active edges,
    // except in some circumstances where the next iteration would be dense.
    merged.setActiveCounts( nactv, nacte );
    return merged;
}

#endif // GRAPTOR_DSL_EMAP_FUSION_H
