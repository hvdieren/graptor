// -*- C++ -*-
#ifndef GRAPHGRIND_GRAPH_PARTITIONING_H
#define GRAPHGRIND_GRAPH_PARTITIONING_H

#include "graptor/partitioner.h"
#include "graptor/graph/gtraits.h"
#include "graptor/graph/CGraphCSx.h"

template<typename vertex>
inline void partitionBalanceDestinations( const wholeGraph<vertex> & WG,
					  partitioner &part ) {
    int p = part.get_num_partitions();
    VID * size = part.as_array();

    // Simple partitioning: every partition has same number of destinations
    int done = 0;
    for( int i=0; i < p-1; ++i ) {
	size[i] = WG.n / p;
	done += size[i];
    }
    size[p-1] = WG.n - done;

    // Aggregate values
    part.compute_starts();
}

template<typename vertex>
inline void
partitionBalanceEdges( const wholeGraph<vertex> & WG, partitioner &part ) {
    int p = part.get_num_partitions();
    VID * size = part.as_array();

    EID edges[p];
    for( int i=0; i < p; ++i )
	edges[i] = 0;

    // Calculate number of vertices in each partition
    EID avgdeg = WG.m / p;
    int cur = 0;
    size[0] = 0;
    for( VID v=0; v < WG.n; ++v ) {
	edges[cur] += WG.V[v].getInDegree();
	size[cur] += 1;
	if( edges[cur] >= avgdeg && cur < p-1 ) {
	    ++cur;
	    size[cur] = 0;
	}
    }
    assert( cur+1 == p );

    // Aggregate values
    part.compute_starts();
}

inline void
partitionBalanceDestinations( const GraphCSx & WG, partitioner &part ) {
    int p = part.get_num_partitions();
    VID * size = part.as_array();

    // Simple partitioning: every partition has same number of destinations
    int done = 0;
    for( int i=0; i < p-1; ++i ) {
	size[i] = WG.numVertices() / p;
	done += size[i];
    }
    size[p-1] = WG.numVertices() - done;

    // Aggregate values
    part.compute_starts();
}

/*
void partitionBalanceEdges( const GraphCSx & WG,
			    const VID * origID,
			    partitioner &part,
			    unsigned short maxVL ) {
    int p = part.get_num_partitions();
    VID * size = part.as_array();

    VID n = WG.numVertices();
    EID m = WG.numEdges();
    const EID * idx = WG.getIndex();

    EID edges[p];
    for( int i=0; i < p; ++i )
	edges[i] = 0;

    // Calculate number of vertices in each partition
    EID avgdeg = m / p;
    EID m_remain = m;
    int cur = 0;
    size[0] = 0;
    for( VID vv=0; vv < n; vv += maxVL ) {
	EID vve = 0;
	for( VID v=vv; v < vv+maxVL; ++v ) {
	    VID w = origID[v];
	    vve += idx[w+1] - idx[w];
	}
	// If with part of the edges we already exceed the threshold,
	// we will move these edges to the next partition.
	bool prior = false;
	if( edges[cur] + vve/2 >= avgdeg && cur < p-1 ) {
	    prior = ( edges[cur] == 0 );
	    // Recalculate the threshold as the first partitions may have
	    // taken an inproportionate number of edges by necessity.
	    // If we keep the threshold unmodified, we may not be able to
	    // fill all partitions.
	    m_remain -= ( prior ? edges[cur] + vve : edges[cur] );
	    avgdeg = m_remain / (p - cur);
	    ++cur;
	    size[cur] = 0;
	}
	if( prior ) {
	    edges[cur-1] += vve;
	    size[cur-1] += maxVL;
	} else {
	    edges[cur] += vve;
	    size[cur] += maxVL;
	}
    }
    assert( cur+1 == p );

    // Correct for working with multiples of vector length
    if( n % maxVL != 0 )
	size[cur] -= maxVL - ( n % maxVL );

    // Aggregate values
    part.compute_starts();

    // Safety check - all vertices accounted for
    assert( part.start_of(p) == n );
}
*/

template<typename GraphTy, typename GetDegree>
inline void partitionBalanceEdges( const GraphTy & WG,
				   GetDegree get_degree,
				   partitioner &part,
				   unsigned int roundup ) {
    using traits = gtraits<GraphTy>;
    using VID = typename traits::VID;
    using EID = typename traits::EID;
    using PID = typename traits::PID;

    static_assert( std::is_same<VID,typename partitioner::VID>::value,
		   "essential types are not matched" );
    
    int p = part.get_num_partitions();
    VID * size = part.as_array();

    VID n = traits::numVertices( WG );
    EID m = traits::numEdges( WG );

    EID edges[p];
    for( int i=0; i < p; ++i ) {
	edges[i] = 0;
	size[i] = 0;
    }

    // Calculate number of vertices in each partition
    EID avgdeg = m / EID(p);
    EID m_remain = m;
    int cur = 0;
/*
    std::cerr << "initial m_remain=" << m_remain
	      << " avgdeg=" << avgdeg << "\n";
*/
    for( VID vv=0; vv < n; vv += roundup ) {
	EID vve = 0;
	for( VID v=vv; v < vv+roundup; ++v )
	    vve += v < n ? get_degree( v ) : 0;
	// If with part of the edges we already exceed the threshold,
	// we will move these edges to the next partition.
	if( edges[cur] + vve/2 >= avgdeg && cur < p-1 ) {
	    // Recalculate the threshold as the first partitions may have
	    // taken an inproportionate number of edges by necessity.
	    // If we keep the threshold unmodified, we may not be able to
	    // fill all partitions.
	    assert( edges[cur] <= m_remain );
	    m_remain -= edges[cur];
	    avgdeg = m_remain / EID(p - cur - 1);
/*
	    std::cerr << "p=" << cur << " nv=" << size[cur]
		      << " ne=" << edges[cur]
		      << " m_remain=" << m_remain
		      << " avgdeg=" << avgdeg
		      << " switch-vve=" << vve
		      << "\n";
*/
	    ++cur;
	    size[cur] = 0;
	    edges[cur] = 0;
	}
	edges[cur] += vve;
	size[cur] += roundup;
    }
    assert( cur+1 == p );
    assert( m_remain == edges[cur] );

    // Correct for working with multiples of vector length
    if( n % roundup != 0 )
	size[cur] -= roundup - ( n % roundup );

/*
    std::cerr << "p=" << cur << " nv=" << size[cur]
	      << " ne=" << edges[cur] << " [final]\n";
    std::cerr << "n=" << n << " m=" << m << "\n";
*/

    EID tote = 0;
    for( int i=0; i < p; ++i )
	tote += edges[i];
    assert( tote == m );

    // Aggregate values
    part.compute_starts();

    // Safety check - all vertices accounted for
    assert( part.start_of(p) == n );
}

inline void partitionBalanceEdges( const GraphCSx & Gcsc, partitioner &part ) {
#if 1
    partitionBalanceEdges( Gcsc, gtraits_getoutdegree<GraphCSx>( Gcsc ),
			   part, 1 );
#else
    int p = part.get_num_partitions();
    VID * size = part.as_array();

    using GTraits = gtraits<GraphCSx>;

    EID edges[p];
    for( int i=0; i < p; ++i )
	edges[i] = 0;

    // Calculate number of vertices in each partition
    EID avgdeg = Gcsc.numEdges() / p;
    int cur = 0;
    size[0] = 0;
    for( VID v=0; v < Gcsc.numVertices(); ++v ) {
	edges[cur] += Gcsc.getDegree(v);
	size[cur] += 1;
	if( edges[cur] >= avgdeg && cur < p-1 ) {
	    ++cur;
	    size[cur] = 0;
	}
    }
    assert( cur+1 == p );

    // Aggregate values
    part.compute_starts();
#endif
}



#endif // GRAPHGRIND_GRAPH_PARTITIONING_H
