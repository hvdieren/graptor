// -*- c++ -*-
#ifndef GRAPTOR_COMB_REORDER_KCORE_H
#define GRAPTOR_COMB_REORDER_KCORE_H

#define NOBENCH
#define MD_ORDERING 0
#define VARIANT 11
#define FUSION 1
#include "../bench/KC_bucket.C"
#undef FUSION
#undef VARIANT
#undef MD_ORDERING
#undef NOBENCH

template<int sorting_order>
std::tuple<GraphCSx,VID,std::vector<VID>>
reorder_kcore( const GraphCSx & G,
	       mm::buffer<VID> & order,
	       mm::buffer<VID> & rev_order,
	       mm::buffer<VID> & remap_coreness,
	       commandLine & options,
	       VID prune_th,
	       VID dmax_v = ~(VID)0 ) {
    timer tm;
    tm.start();
    
    // Number of partitions is tunable. A fairly large number is helpful
    // to help load balancing.
    VID npart = options.getOptionLongValue( "-c", 256 );
    GraphCSRAdaptor GA( G, npart );
    KCv<GraphCSRAdaptor> kcore( GA, options );

    // TODO: after coreness computation, could further trim down
    //       the set of vertices to retain based on their coreness
    //       and how it compares to the best known clique.
    //       However, coreness computation remains fairly expensives for
    //       some graphs.
    kcore.run( prune_th );
    auto & coreness = kcore.getCoreness();
    std::cout << "Calculating coreness time: " << tm.next() << "\n";
    std::cout << "coreness=" << kcore.getLargestCore() << "\n";

    VID n = G.numVertices();
    VID pn = n;
    EID pm = G.numEdges();

    bool early_pruning = prune_th != ~(VID)0;
    if( early_pruning ) {
	// Look for a clique larger than this size, requires this many
	// neighbours.
	// TODO: duplicated in graph construction.
	pn = 0;
	pm = 0;
	const EID * const idx = G.getIndex();
	const VID * const edg = G.getEdges();
	for( VID v=0; v < n; ++v ) {
	    if( G.getDegree( v ) >= prune_th ) {
		++pn;
		for( EID j=idx[v], je=idx[v+1]; j != je; ++j ) {
		    if( G.getDegree( edg[j] ) >= prune_th )
			++pm;
		}
	    }
	}

	std::cout << "pruned vertex range: " << pn << "\n";
	std::cout << "Pruning vertex/edge count time: " << tm.next() << "\n";
    }

    if( dmax_v == ~(VID)0 )
	dmax_v = G.findHighestDegreeVertex();
    VID dmax = G.getDegree( dmax_v );

    VID K = kcore.getLargestCore();
    std::vector<VID> histo;
    if constexpr ( sorting_order == 0 ) {
	// Increasing degree
	if( early_pruning )
	    sort_order_pruned( n, dmax, G.getDegree(), prune_th,
			       order.get(), rev_order.get(), false );
	else
	    sort_order( n, dmax, G.getDegree(), order.get(), rev_order.get(),
			false );
    } else if constexpr ( sorting_order == 1 ) {
	// Decreasing degree
	if( early_pruning )
	    sort_order_pruned( n, dmax, G.getDegree(), prune_th,
			       order.get(), rev_order.get(), true );
	else
	    sort_order( n, dmax, G.getDegree(), order.get(), rev_order.get(),
			true );
    } else if constexpr ( sorting_order == 2 ) {
	// Increasing degeneracy
	if( early_pruning )
	    sort_order_pruned( n, K, coreness.get_ptr(), prune_th,
			       order.get(), rev_order.get(), false );
	else
	    sort_order( n, K, coreness, order.get(), rev_order.get(), false );
    } else if constexpr ( sorting_order == 3 ) {
	// Decreasing degeneracy
	if( early_pruning )
	    sort_order_pruned( n, K, coreness.get_ptr(), prune_th,
			       order.get(), rev_order.get(), true );
	else
	    sort_order( n, K, coreness, order.get(), rev_order.get(), true );
    } else if constexpr ( sorting_order == 4 ) {
	histo.resize( K+2 );
	if( early_pruning )
	    sort_order_ties_pruned( n, K, G.getDegree(), coreness.get_ptr(),
				    prune_th,
				    order.get(), rev_order.get(), &histo[0],
				    false, false );
	else
	    sort_order_ties( n, K, G.getDegree(), coreness,
			     order.get(), rev_order.get(), &histo[0],
			     false, false );
    } else if constexpr ( sorting_order == 5 ) {
	histo.resize( K+2 );
	// Sort by increasing coreness, and decreasing degree per coreness.
	// This place high-coreness vertices to "the right", resulting in
	// right-hand neighbourhoods containing only higher-degeneracy vertices.
	// The sorting by decreasing degree ensures that high-degree vertices
	// are visited first, for instance, in computation of the pivot.
	if( early_pruning )
	    sort_order_ties_pruned( n, K, G.getDegree(), coreness.get_ptr(),
				    prune_th,
				    order.get(), rev_order.get(), &histo[0],
				    false, true );
	else
	    sort_order_ties( n, K, G.getDegree(), coreness,
			     order.get(), rev_order.get(), &histo[0],
			     false, true );
    } else if constexpr ( sorting_order == 6 ) {
	histo.resize( K+2 );
	if( early_pruning )
	    sort_order_ties_pruned( n, K, G.getDegree(), coreness.get_ptr(),
				    prune_th,
				    order.get(), rev_order.get(), &histo[0],
				    true, false );
	else
	    sort_order_ties( n, K, G.getDegree(), coreness,
			     order.get(), rev_order.get(), &histo[0],
			     true, false );
    } else if constexpr ( sorting_order == 7 ) {
	histo.resize( K+2 );
	if( early_pruning )
	    sort_order_ties_pruned( n, K, G.getDegree(), coreness.get_ptr(),
				    prune_th,
				    order.get(), rev_order.get(), &histo[0],
				    true, true );
	else
	    sort_order_ties( n, K, G.getDegree(), coreness,
			     order.get(), rev_order.get(), &histo[0],
			     true, true );
    } else {
	static_assert( 0 <= sorting_order && sorting_order <= 5,
		       "sorting_order must be in range [0,5]" );
    }
    std::cout << "Determining sort order time: " << tm.next() << "\n";

    n = pn; // From here on

    // Only pruned vertex range (if applicable)
    parallel_loop( (VID)0, pn, [&]( VID v ) {
	remap_coreness[v] = coreness[order[v]];
    } );
    std::cout << "Remapping coreness data: " << tm.next() << "\n";

    // Remap graph.
    // TODO: remap+hash in one step
    RemapVertex<VID> remapper( order.get(), rev_order.get() );
    GraphCSx R( pn, pm, -1, true, false );
    R.import_select( G, remapper );
    std::cout << "Remapping graph: " << tm.next() << "\n";

    return { R, kcore.getLargestCore(), histo };
}

#endif // GRAPTOR_COMB_REORDER_KCORE_H
