// -*- c++ -*-
#ifndef GRAPTOR_GRAPH_CONTRACT_CLIQUE_H
#define GRAPTOR_GRAPH_CONTRACT_CLIQUE_H

#include <algorithm>
#include "graptor/graph/contract/vertex_set.h"

namespace contract {
    
namespace clique {

template<typename VID, typename EID>
void append_neighbours(
    vertex_set<VID> & s,
    const GraphCSx & G,
    const edge_cover<VID,EID> & ecov,
    VID v,
    VID filter,
    VID min_size ) {
    const EID * const index = G.getIndex();
    const VID * const edges = G.getEdges();
    const VID * B = &edges[index[v]];
    const VID * E = &edges[index[v+1]];
    s.reserve( s.size() + ( B - E ) );
    // sorted neighbour lists, so resolve *I > filter through lower_bound
    for( const VID * I=std::lower_bound( B, E, filter ); I != E; ++I ) {
	VID u = *I;
	if( index[u+1] - index[u] >= min_size
	    && !ecov.is_covered( u, v )
	    && detail::intersection_size(
		&edges[index[u]], (VID)(index[u+1]-index[u]),
		&edges[index[v]], (VID)(index[v+1]-index[v]) ) + 2 >= min_size )
	    s.push( u );
    }
}

// This method assumes both lists are sorted
template<typename VID>
vertex_set<VID>
intersect_neighbours(
    const vertex_set<VID> & s, const GraphCSx & G, VID v ) {
    vertex_set<VID> i;
    i.resize( s.size() );
    const VID * r = &G.getEdges()[G.getIndex()[v]];
    VID n = s.intersect( r, G.getDegree(v), i.begin() );
    i.resize( n );
    return i;
}

template<typename VID>
VID
count_intersect_neighbours(
    const vertex_set<VID> & s,
    const GraphCSx & G,
    VID v ) {
    const VID * I = &G.getEdges()[G.getIndex()[v]];
    const VID * E = &G.getEdges()[G.getIndex()[v+1]];
    return s.intersection_size( I, std::distance( I, E ) );
}

template<typename VID>
std::pair<VID,VID> get_pivot_from_set(
    const GraphCSx & G,
    const vertex_set<VID> & S ) {

    const EID * const index = G.getIndex();
    const VID * const edges = G.getEdges();

    VID max_i = 0;
    VID best_i = ~(VID)0;

    for( VID I=0, E=S.size(); I != E; ++I ) {
	VID v = S.get( I );

	// Skip if cannot improve because i <= degree of v
	VID deg = index[v+1] - index[v];
	if( deg > max_i ) {
	    VID i = S.intersection_size( &edges[index[v]], deg );
	    if( i > max_i || ~best_i == 0) {
		max_i = i;
		best_i = I;
	    }
	}
    }

    return std::make_pair( best_i, max_i );
}

// TODO: how is the pivot affected by the fact that certain edges have
//       been covered and are no longer of interest?
template<typename VID>
VID get_pivot(
    const GraphCSx & G,
    const vertex_set<VID> & P, const vertex_set<VID> & X ) {

    assert( P.size() > 0 || X.size() > 0 );

    auto Pi = get_pivot_from_set( G, P );
    auto Xi = get_pivot_from_set( G, X );

    VID pivot = Pi.second > Xi.second ? Pi.first : Xi.first;
    if( ~pivot == 0 )
	pivot = P.get( 0 );

    assert( ~pivot != 0 );

    return pivot;
}


template<typename VID, typename EID, typename Fn>
bool BronKerboschPivot(
    const GraphCSx & G,
    edge_cover<VID,EID> & ecov,
    VID size,
    vertex_set<VID> & R, vertex_set<VID> & P, vertex_set<VID> & X,
    VID depth, Fn record_clique ) {

    if( R.size() + P.size() < size )
	return false;
    
    if( P.size() == 0 ) {
	// Maximal clique?
	// Require minimal clique size
	if( X.size() == 0 && R.size() >= size ) {
	    // If we reach here, all edges in the clique R have been reserved
	    // for this clique and it is safe to record it.
	    record_clique( R );
	    return true;
	}
	// Clique not accepted - release edges
	return false;
    }

    // Select pivot
    // VID u = get_pivot( G, P, X );

/*
    std::cerr << "Iter depth " << depth << ", minsize=" << size << ": R={";
    for( auto I=R.begin(), E=R.end(); I != E; ++I )
	std::cerr << ' ' << *I;
    std::cerr << " } P={";
    // for( auto I=P.begin(), E=P.end(); I != E; ++I )
    // std::cerr << ' ' << *I;
    std::cerr << " #" << P.size();
    std::cerr << " } ";
    // std::cerr << "pivot=" << u;
    std::cerr << "\n";
*/

    // Iterate
    for( VID I=0, E=P.size(); I != E; ++I ) {
	VID v = P.get( I );
	// if( G.hasNeighbor( u, v ) )
	// continue;

	// Acquire all edges (u,v) where u is an element of R
	// If some edge already covered, then vertex v cannot be part of this
	// clique. Drop the vertex from consideration and move on with other
	// vertices.
	if( !ecov.cover_edges( v, R ) )
	    continue;
	
	R.push( v );
	vertex_set<VID> Pv = intersect_neighbours( P, G, v );
	vertex_set<VID> Xv = intersect_neighbours( X, G, v );

	// Filter Pv to ensure that every vertex in Pv has at least |R|-1
	// common neighbours with every vertex in R.
	// Rationale: in a clique of size R+1, every 2 vertices have R-1 common
	// neighbours
	const EID * const index = G.getIndex();
	const VID * const edges = G.getEdges();
	VID num_rm = 0;
	for( auto I=Pv.begin(), E=Pv.end(); I != E; ++I ) {
	    bool success = true;
	    if( index[1+*I]-index[*I] >= size ) { // trivial?
		// for( auto J=R.begin(), F=R.end(); J != F; ++J ) {
		// Only check latest addition
		    if( detail::intersection_size(
			    &edges[index[*I]], (VID)(index[1+*I]-index[*I]),
			    &edges[index[v]], (VID)(index[1+v]-index[v]) )
			< size-1 ) {
			success = false;
			break;
		    }
		    // }
	    } else
		success = false;
	    if( !success ) {
		// std::cerr << *I << " has insufficient common neighbours\n";
		++num_rm;
		Pv.remove( *I );
		--I;
		E = Pv.end();
	    }
	}
	// std::cerr << "removed " << num_rm
	// << " vertices with too few common neighbours\n";
	
	if( BronKerboschPivot( G, ecov, size, R, Pv, Xv, depth+1, record_clique )
	    && depth > 1 ) {
	    // Clique found and accepted; rewind to top level (depth == 1)
	    R.pop();

	    return true;
	} else {
	    // Undo addition of v, release edges.
	    R.pop();
	    ecov.release_edges( v, R );

	    // v has been considered. Move from P to X.
	    P.remove( v );
	    --I; // to account for removed element
	    --E;
	    X.push( v );
	}
    }

    return false;
}

template<typename VID, typename EID, typename Fn>
void find_cliques( const GraphCSx & G,
		   edge_cover<VID,EID> & ecov,
		   VID v,
		   VID size,
		   Fn record_clique ) {
    // TODO: iterate in order of decreasing degree...

    // TODO: could check for remaining incident edges through extending ecov
    //       with degree array, and update it
    if( G.getDegree()[v] >= size  ) {
	vertex_set<VID> R, P, X;

	// R = { v }
	R.push( v );
	// P = { u : (u,v) is an edge that has not been allocated yet }
	append_neighbours( P, G, ecov, v, v, size );

	// TODO: only require X elements that are neighbours of v?
	//       That would restrict the choice of pivot at the top level.
	// X.reserve( v );
	// for( VID x=0; x < v; ++x )
	// X.push( x );
	{
	    const EID * const index = G.getIndex();
	    const VID * const edges = G.getEdges();
	    const VID * B = &edges[index[v]];
	    const VID * E = &edges[index[v+1]];
	    E = std::lower_bound( B, E, v );
	    X.push( B, E );
	}

/*
	std::cerr << "find_cliques: v=" << v << " deg=" << G.getDegree()[v]
		  << " #P=" << P.size() << " minsize=" << size
		  << "\n";
*/
	
	bool fnd = BronKerboschPivot( G, ecov, size, R, P, X, (VID)1, record_clique );
	if( !fnd ) {
	    return;
	}
	// Otherwise, found a clique, try again (TODO)
    }
}

template<typename VID, typename EID, typename Fn>
void find_cliques_of_size( const GraphCSx & G,
			   edge_cover<VID,EID> & ecov,
			   VID size,
			   Fn record_clique ) {
    VID n = G.numVertices();

    std::cerr << ">> Searching cliques of size " << size
	      << " vertices=" << n << "\n";

    // TODO: parallel loop
    for( VID v=0; v < n; ++v )
	find_cliques( G, ecov, v, size, record_clique );
}

template<typename VID, typename EID, typename Fn>
void find_all_cliques( const GraphCSx & G,
		       edge_cover<VID,EID> & ecov,
		       Fn record_clique ) {
    find_cliques_of_size( G, ecov, (VID)256, record_clique );
    // find_cliques_of_size( G, ecov, (VID)64, record_clique );
    // find_cliques_of_size( G, ecov, (VID)16, record_clique );
    // find_cliques_of_size( G, ecov, (VID)8, record_clique );
    // find_cliques_of_size( G, ecov, (VID)4, record_clique );
}

#if 0
template<typename VID, typename EID, typename Fn>
bool is_clique( const GraphCSx & G,
		vertex_set<VID> & s,
		edge_cover<VID,EID> & ecov ) {
    // Check if every vertex in s is connected to every other vertex, and
    // can be selected through the cover.
}

template<typename VID, typename EID, typename Fn>
bool find_common_clique( const GraphCSx & G,
			 edge_cover<VID,EID> & ecov,
			 VID u, VID v,
			 Fn record_clique ) {
    const EID * const index = G.getIndex();
    const VID * const edges = G.getEdges();

    // Build set of common neighbours
    vertex_set<VID> ngh( index[u+1] - index[u] );
    detail::try_cover cnd( ecov, v, w );
    detail::append_list_cond<VID,decltype(cnd)> ins( ngh.begin(), cnd );
    detail::intersect( &edges[index[u]], &edges[index[u+1]],
		       &edges[index[v]], &edges[index[v+1]], ins );

    // Check for clique
    bool success = ngh.size() >= 8;
    if( success )
	reduce_to_clique( G, ngh, ecov );
    
    // Try and reserve all edges, if clique is large enough
    vertex_set<VID> rm( ngh.size() );
    if( success ) {
	// Reserve all edges; if failing a vertex, remove all edges incident
	// to this vertex. rm is the set of vertices that form the clique that
	// was ultimately reserved.
	rm = reserve_all_edges_or_remove( ngh, ecov );
    }

    // Update as we may have lost vertices when checking edges
    success = ngh.size() >= 8;

    // Check clique is large enough to be of interest
    if( !success ) {
	// Clique too small. Release and give up.
	for( auto I=ngh.begin(), E=ngh.end(); I != E; ++I ) {
	    ecov.release( u, *I );
	    ecov.release( v, *I );
	}

	return false;
    }

}
#endif

} // namespace clique

} // namespace contract

#endif // GRAPTOR_GRAPH_CONTRACT_CLIQUE_H
