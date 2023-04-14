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
    // TODO: for case of roundup == 1, use lower_bound to find next partition
    //       boundary
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

template<typename VID, typename EID, typename PID>
void partitionBalanceEdges_bisect(
    VID n, EID m, const EID * const index, partitioner &part ) {
    // using traits = gtraits<GraphCSx>;
    // using VID = typename traits::VID;
    // using EID = typename traits::EID;
    // using PID = typename traits::PID;

    static_assert( std::is_same<VID,typename partitioner::VID>::value,
		   "essential types are not matched" );
    
    int P = part.get_num_partitions();
    VID * size = part.as_array();

    // const EID * index = WG.getIndex();

    // VID n = traits::numVertices( WG );
    // EID m = traits::numEdges( WG );

    // Calculate number of vertices in each partition
    EID m_done = 0;
    VID v_done = 0;
    for( unsigned p=0; p < P; ++p ) {
	EID avgdeg = ( m - m_done ) / EID(P - p);
	const EID * bnd = std::upper_bound( &index[v_done], &index[n],
					    m_done + avgdeg );
	size[p] = bnd - &index[v_done];

	v_done += size[p];
	m_done = *bnd;
    }

    assert( m_done == m );
    assert( v_done == n );
    
    // Aggregate values
    part.compute_starts();

    // Safety check - all vertices accounted for
    assert( part.start_of(P) == n );
}

inline void partitionBalanceEdges_bisect( const GraphCSx & WG,
					  partitioner &part ) {
    using traits = gtraits<GraphCSx>;
    using VID = typename traits::VID;
    using EID = typename traits::EID;
    using PID = typename traits::PID;

    return partitionBalanceEdges_bisect<VID,EID,PID>( 
	traits::numVertices( WG ),
	traits::numEdges( WG ),
	WG.getIndex(),
	part );
}


template<typename lVID, typename lEID>
inline void partitionBalanceEdges_pow2_rec(
    lVID lo, lVID hi, unsigned plo, unsigned phi,
    lVID * size, const lEID * index ) {

    if( phi > plo+1 ) {
	lEID tgt = ( index[lo] + index[hi] ) / 2;
	const lEID * half = std::lower_bound( &index[lo], &index[hi], tgt );

	lVID mid = half - &index[0];
	size[(phi+plo)/2] = mid;

	partitionBalanceEdges_pow2_rec( lo, mid, plo, (phi+plo)/2, size, index );
	partitionBalanceEdges_pow2_rec( mid, hi, (phi+plo)/2, phi, size, index );
    }
}

template<typename GraphTy, typename GetDegree>
inline void partitionBalanceEdges_pow2( const GraphTy & WG,
					GetDegree get_degree,
					partitioner &part,
					unsigned int roundup ) {
    assert( roundup == 1 );
    
    using traits = gtraits<GraphTy>;
    using VID = typename traits::VID;
    using EID = typename traits::EID;
    using PID = typename traits::PID;

    static_assert( std::is_same<VID,typename partitioner::VID>::value,
		   "essential types are not matched" );
    
    int p = part.get_num_partitions();
    VID * size = part.as_array();

    assert( (p & (p-1)) == 0 && "p must be power of 2" );

    VID n = traits::numVertices( WG );
    EID m = traits::numEdges( WG );

    EID edges[p];
    for( int i=0; i < p; ++i ) {
	edges[i] = 0;
	size[i] = 0;
    }

    // Cumulative histogram of edge counts...
    const EID * index = WG.getIndex();

    partitionBalanceEdges_pow2_rec( (VID)0, n, 0, p, size, index );

    // undo cumulative data to fit with partitioner::compute_starts()
    size[p] = n;
    VID cur = 0;
    for( int i=0; i < p; ++i ) {
	size[i] = size[i+1] - cur;
	edges[i] = index[size[i+1]] - index[cur];
	cur = size[i+1];
    }

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
/*
    int p = part.get_num_partitions();
    if( (p & (p-1)) == 0 )
	partitionBalanceEdges_pow2(
	    Gcsc, gtraits_getoutdegree<GraphCSx>( Gcsc ), part, 1 );
    else
*/
	partitionBalanceEdges_bisect( Gcsc, part );
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
