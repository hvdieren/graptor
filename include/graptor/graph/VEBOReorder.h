// -*- C++ -*-
#ifndef GRAPHGRIND_GRAPH_VEBOREORDER_H
#define GRAPHGRIND_GRAPH_VEBOREORDER_H

#include <queue>

#include "graptor/mm.h"
#include "graptor/graph/remap.h"
#include "graptor/graph/partitioning.h"
#include "graptor/graph/CGraphCSx.h"

#ifndef VEBO_DISABLE
#define VEBO_DISABLE 0
#endif // VEBO_DISABLE

#ifndef VEBO_FORCE
#define VEBO_FORCE 0
#endif // VEBO_FORCE

template<typename lVID, typename lEID>
class VEBOReorderState {
protected:
    // In w = reverse[v], v is the new vertex ID, w is the old one
    // In v = remap[w], v is the new vertex ID, w is the old one
    mmap_ptr<lVID> remap;
    mmap_ptr<lVID> reverse;

public:
    VEBOReorderState() { }

    const lVID *getRemap() const { return remap.get(); }
    const lVID *getReverse() const { return reverse.get(); }

    std::pair<const lVID *,const lVID *> maps() const {
	// In w = first[v], v is the new vertex ID, w is the old one
	// In v = second[w], v is the new vertex ID, w is the old one
	return std::make_pair( reverse.get(), remap.get() );
    }

    RemapVertex<lVID> remapper() const {
	return RemapVertex<lVID>( reverse.get(), remap.get() );
    }

    lVID originalID( lVID v ) const { return reverse.get()[v]; }
    lVID remapID( lVID v ) const { return remap.get()[v]; }

    void del() {
	remap.del( "VEBOReorder - remap" );
	reverse.del( "VEBOReorder - reverse" );
    }
};

template<typename lVID, typename lEID>
class ReorderDegreeSort : public VEBOReorderState<VID,EID> {
public:
    ReorderDegreeSort() { }

    ReorderDegreeSort( const GraphCSx &csc, unsigned short maxVL = 1 ) {
	VID n = csc.numVertices();  // number of vertices
	VID next = n;               // number of elements, multiple of maxVL
	if( next % maxVL != 0 )
	    next += maxVL - ( next % maxVL );

	remap.allocate( next, numa_allocation_interleaved() );
	reverse.allocate( next, numa_allocation_interleaved() );

	std::cerr << "ReorderDegreeSort: vertices: " << n
		  << " extended to: " << next << "\n";

	// Do degree sorting, use remap as scratch array
	degree_sort( csc.getIndex(), n, remap, reverse );

	// Extend range. Keep additional vertices at the end of the iteration
	// range so they can be easily skipped during processing.
	for( VID v=n; v < next; ++v )
	    reverse[v] = v;
	
	// Invert reverse array.
	std::fill( remap.get(), &remap.get()[next], ~(VID)0 );
	parallel_loop( (VID)0, next, [&]( VID v ) {
	    assert( remap.get()[reverse.get()[v]] == ~(VID)0 );
	    remap.get()[reverse.get()[v]] = v;
	} );
	
	std::cerr << "ReorderDegreeSort: done\n";
    }

    static void degree_sort( const lEID * index, lVID n,
			     lVID * histo, lVID * reorder ) {
	lVID dmax = 0;

	std::fill( &histo[0], &histo[n], lVID(0) );

	// Count vertices per degree and build list
	for( lVID v=0; v < n; ++v ) {
	    lVID deg = index[v+1] - index[v];
	    histo[deg]++;
	    if( dmax < deg )
		dmax = deg;
	}

	// Insertion positions per degree
	lVID d = dmax-1;
	lVID h = histo[dmax];
	histo[dmax] = 0;
	do {
	    lVID tmp = histo[d];
	    histo[d] = h;
	    h += tmp;
	} while( d-- > 0 );

	assert( h == n && "mismatch between counted vertices and actual" );

	// Place in reorder array
	for( lVID v=0; v < n; ++v ) {
	    lVID d = index[v+1] - index[v];
	    reorder[histo[d]++] = v;
	}

	// Check
	assert( histo[0] == n && "placement mismatch" );
    }
};

class VEBOReorder : public VEBOReorderState<VID,EID> {
    // In w = reverse[v], v is the new vertex ID, w is the old one
    // In v = remap[w], v is the new vertex ID, w is the old one

public:
    VEBOReorder() { }
    VEBOReorder( const GraphCSx &csc, partitioner & part )
	: VEBOReorder( csc, part, 1, false, 1 ) { }

    VEBOReorder( const GraphCSx &csc, partitioner & part,
		 unsigned short maxVL, bool interleave,
		 unsigned short pmul ) {
	int P = part.get_num_partitions();
	assert( P % maxVL == 0 && "maxVL must divide P" );
	
	VID n = csc.numVertices();

	VID max_v = csc.findHighestDegreeVertex(); // parallel
	VID max_deg = csc.getDegree( max_v );
#if VEBO_DISABLE
	if( true )
#elif VEBO_FORCE
	if( false )
#else
	if( max_deg < n / (128*1024) )
#endif
	{
	    // Looks like a graph with low degrees, probably not worth to
	    // reorder. Better decision-making procedure would be to sample
	    // the degree distribution.
	    std::cerr << "VEBO: looks like non-power-law graph: max-degree: "
		      << max_deg << " vertices: " << n << "\n";

	    remap.allocate( n, numa_allocation_interleaved() );

	    // Initialise partitioner
	    partitionBalanceEdges( csc, gtraits_getoutdegree<GraphCSx>( csc ),
				   part, maxVL );
		
	    reverse.allocate( n, numa_allocation_interleaved() );
	    map_partitionL( part, [&]( int p ) { 
		    VID s = part.start_of( p );
		    VID e = part.start_of( p+1 );
		    for( VID v=s; v < e; ++v ) {
			remap[v] = v;
			reverse[v] = v;
		    }
		} );
	} else {
	    // Looks like a  power-law graph
	    VID nwpad = n + P * pmul; // upper bound
	    // if( nwpad % pmul )
	    // nwpad += ( VID(pmul) - nwpad ) % pmul;
		
	    std::cerr << "VEBO: looks like power-law graph: max-degree: "
		      << max_deg << " vertices: " << n
		      << " provisionally extended to: " << nwpad << "\n";

	    remap.allocate( nwpad, numa_allocation_interleaved() );

	    // Calculate mapping
	    vebo( csc, part, maxVL, interleave, pmul );
	    nwpad = part.get_vertex_range();

	    // Calculate auxiliary reverse mapping
	    // reverse.Interleave_allocate( n );
	    timer tm;
	    tm.start();
	    reverse.allocate( nwpad, numa_allocation_interleaved() );
	    std::fill( reverse.get(), &reverse.get()[nwpad], ~(VID)0 );
	    parallel_loop( (VID)0, nwpad, [&]( VID v ) {
		assert( reverse.get()[remap.get()[v]] == ~(VID)0 );
		reverse.get()[remap.get()[v]] = v;
	    } );
	    std::cerr << "VEBO: reversing map: " << tm.next() << "\n";
	}

	// TODO: correctness checking
	/*
	for( VID v=0; v < n; ++v ) {
	    assert( v == reverse.get()[remap.get()[v]] );
	    assert( v == remap.get()[reverse.get()[v]] );
	}
	*/
    }

private:
    template<typename T>
    struct pstate_cmp {
	pstate_cmp( T * v ) : m_v( v ) { }
	bool operator() ( VID x, VID y ) const {
	    // Simple comparator, leaves positioning of partitions with few
	    // vertices up to the heap implementation, typically at
	    // p = 2**k - 2
	    // return m_v[x] > m_v[y];
	    // Ensures least-vertex partitions are at the low numbers
	    // return m_v[x] > m_v[y] || ( m_v[x] == m_v[y] && x > y );
	    // Spread least-vertex partitions around over the range of
	    // all partition numbers. They end up in partitions with the
	    // most number of trailing zeros
	    if( m_v[x] > m_v[y] )
		return true;
	    else if( m_v[x] < m_v[y] )
		return false;
	    else {
		unsigned int rx = bit_reverse( x );
		unsigned int ry = bit_reverse( y );
		return rx > ry;
	    }
	}
	T * m_v;
    };
    
    struct pstate {
	VID n;
	VID *u;
	EID *w;
	int pmax;
	int popen;
	EID Delta;
	std::priority_queue<EID,std::vector<EID>,pstate_cmp<EID>> w_heap;
	std::priority_queue<VID,std::vector<VID>,pstate_cmp<VID>> u_heap;

	pstate( VID n_, VID *u_, EID *w_ )
	    : n( n_ ), u( u_ ), w( w_ ), pmax( 0 ), popen( -1 ), Delta( 0 ),
	      w_heap( pstate_cmp<EID>( w ) ), u_heap( pstate_cmp<VID>( u ) ) {
	    // Assume that initially all w[i] are zero

	    // Don't initialise the u_heap until convert_zdeg is called
	    for( VID v=0; v < n; ++v )
		w_heap.push( v );
	}

	template<bool reinsert=true>
	int place( VID d ) {
	    if( d > 0 )
		return place_nzdeg<reinsert>( d );
	    else
		return place_zdeg<reinsert>();
	}

	int place_packed( VID d, int pmul ) {
	    if( popen >= 0 ) {
		if( u[popen] % pmul != 0 ) {
		    place_at( d, popen );
		    return popen;
		} else {
		    // Reinsert, then search any partition
		    close( d );
		}
	    }

	    popen = place<false>( d );
	    return popen;
	}

	void close( VID d ) {
	    if( popen >= 0 ) {
		if( d > 0 )
		    w_heap.push( popen );
		else
		    u_heap.push( popen );
		popen = -1;
	    }
	}

	int place_at( VID d, int p ) {
	    if( d > 0 )
		return place_at_nzdeg( d, p );
	    else
		return place_at_zdeg( p );
	}

	
	int place_at_nzdeg( VID d, int p ) {
	    // p is not currently in the heap
	    // do not update Delta as it will not impact on our decisions
	    // until the partition is closed with a multiple of pmul
	    w[p] += d;
	    u[p]++;

	    // Track highest loaded partition
	    if( w[p] > w[pmax] )
		pmax = p;

	    return p;
	}

	int place_at_zdeg( int p ) {
	    // p is not currently in the heap
	    // do not update Delta as it will not impact on our decisions
	    // until the partition is closed with a multiple of pmul
	    u[p]++;

	    // Track highest loaded partition
	    if( u[p] > u[pmax] )
		pmax = p;

	    return p;
	}


	// Place a vertex with degree d, assuming d > 0
	template<bool reinsert>
	int place_nzdeg( VID d ) {
	    // Get the least-loaded partition and remove it from the heap
	    int pmin = w_heap.top();
	    w_heap.pop();

	    w[pmin] += d;
	    u[pmin]++;

	    // Track highest loaded partition
	    if( w[pmin] > w[pmax] )
		pmax = pmin;

	    // Re-insert into heap
	    if constexpr ( reinsert )
		w_heap.push( pmin );

	    // Track Delta
	    Delta = w[pmax] - w[w_heap.top()];

	    return pmin;
	}

	void convert_zdeg() {
	    // Now start tracking delta
	    pmax = 0;
	    for( int i=0; i < n; ++i ) {
		u_heap.push( i );
		if( u[pmax] < u[i] )
		    pmax = i;
	    }
	    Delta = u[pmax] - u[u_heap.top()];
	}

	// Place a vertex with degree d, assuming d == 0
	template<bool reinsert>
	int place_zdeg() {
	    int pmin = u_heap.top();
	    u_heap.pop();

	    // no need to update w[], vertex has zero degree
	    u[pmin]++;

	    // Track highest loaded partition
	    if( u[pmin] > u[pmax] )
		pmax = pmin;

	    // Re-insert into heap
	    if constexpr ( reinsert )
		u_heap.push( pmin );

	    // Track delta
	    Delta = u[pmax] - u[u_heap.top()];

	    return pmin;
	}
    };

    VID place_vertices_simd( int P, int gP, int vP,
			     VID *first, VID *next, VID *a, int d,
			     VID *u, VID *du ) {
	VID v = first[d];
	VID vlast = v;
	for( int gp=0; gp < gP; ++gp ) {
	    // Maximise locality within groups of lanes, i.e., aim to
	    // have lanes accessed by one vector operation access vertices
	    // that had close vertex IDs in the original graph
	    bool done = false;
	    while( !done ) {
		VID old_v = v;
		for( int p=gp*vP; p < (gp+1)*vP; ++p ) {
		    if( u[p] > du[p] ) {
			a[v] = p;
			vlast = v;
			v = next[v];
			du[p]++;
		    }
		}
		if( v == old_v ) // check if progress made
		    done = true;
	    }
	}
	assert( v == ~(VID)0 );
	return vlast;
    }
    VID place_vertices( int P, VID *first, VID *next, VID *a, int d,
			VID *u, VID *du, int pmul ) {
	// currently processing vertices of degree d
	// du is the number of vertices per partition before starting degree d
	// u is the number of vertices per partition after degree d
	VID v = first[d];
	VID vlast = v;
	for( int p=0; p < P; ++p ) {
	    int kp = u[p] - du[p]; // number to place
	    for( int i=0; i < kp; ++i ) {
		a[v] = p;
		vlast = v;
		v = next[v];
	    }
	}
	assert( v == ~(VID)0 );
	return vlast;
    }
    void place_vertices_linked( int P, VID *first, VID *next,
				VID *plast, int d,
				VID *u, VID *du ) {
	// currently processing vertices of degree d
	// du is the number of vertices per partition before starting degree d
	// u is the number of vertices per partition after degree d
	VID v = first[d];
	VID vlast = v;
	for( int p=0; p < P; ++p ) {
	    int kp = u[p] - du[p]; // number to place
	    if( kp == 0 )
		continue;
	    
	    // Vertices are already on a linked list. Need to walk that list
	    // and link sections of it onto the per-partition list. However,
	    // all internal links in the section remain as they are.
	    VID plast_p = plast[p];
	    next[plast_p] = v;
	    for( int i=0; i < kp; ++i ) {
		// next[plast[p]] = v;
		plast_p = v;
		v = next[v];
	    }
	    plast[p] = plast_p;
	}
	assert( v == ~(VID)0 );
    }

    void vebo( const GraphCSx &csc, partitioner & part, unsigned short maxVL,
	       bool intlv, unsigned short pmul ) {
	if( intlv == false && maxVL == 1 ) {
	    vebo_graptor<true>( csc, part, pmul );
	    return;
	}
	
	timer tm;
	tm.start();
	
	std::cerr << "VEBO: start\n";
	
	VID n = csc.numVertices();         // Number of vertices
	int P = part.get_num_partitions(); // Total partitions
	int gP = P / maxVL;                // Groups of vector lanes
	int vP = maxVL;                    // Vector-lane partitions

	assert( gP * maxVL == P );

	std::cerr << "VEBO: parameters: maxVL=" << maxVL
		  << " intlv=" << intlv << " pmul=" << pmul
		  << " n=" << n << " P=" << P
		  << " gP=" << gP << " vP=" << vP << "\n";

	// VID nwpad = n;
	// if( nwpad % pmul )
	// nwpad += ( VID(pmul) - nwpad ) % pmul;
	VID max_nwpad = n + P * pmul; // upper bound on padding

	// 1. Build chains of vertices with the same degree
	mmap_ptr<VID> mm_first, mm_next, mm_histo, mm_alloc;
	mm_first.allocate( n, numa_allocation_interleaved() );
	mm_alloc.allocate( max_nwpad, numa_allocation_interleaved() );
	mm_next.allocate( n, numa_allocation_interleaved() );
	mm_histo.allocate( n, numa_allocation_interleaved() );
	VID * first = mm_first.get();
	VID * last = mm_alloc.get();
	VID * next = mm_next.get();
	VID * histo = mm_histo.get(); // assume histo is zero-initialised
	parallel_loop( (VID)0, n, [&]( VID v ) {
	    histo[v] = 0;
	    first[v] = 0;
	    next[v] = ~(VID)0;
	} );

	std::cerr << "VEBO: setup: " << tm.next() << "\n";

	const EID * idx = csc.getIndex();
	EID dmax = 0;
	for( VID v=0; v < n; ++v ) {
	    VID d = idx[v+1] - idx[v];

	    if( __builtin_expect( histo[d] == 0, 0 ) ) { // first occurence of degree
		first[d] = last[d] = v; // initialise 'linked list' for degree
		histo[d] = 1;
		if( dmax < d ) // track maximum degree seen
		    dmax = d;
	    } else {
		next[last[d]] = v; // add vertex to tail of list
		last[d] = v;
		++histo[d];
	    }
	}

	std::cerr << "VEBO: binning: " << tm.next() << "\n";

	// 2. Place vertices
	EID *w = new EID[P];
	VID *u = new VID[P];
	VID *du = new VID[P];
	std::fill( &w[0], &w[P], EID(0) );
	std::fill( &u[0], &u[P], VID(0) );

	pstate load( P, u, w );

	VID *a = last; // reuse array

	if( P == 1 ) {
	    u[0] = n;
	    w[0] = csc.numEdges();

	    // Link all vertex lists together
	    VID d = dmax;
	    VID vlast = ~(VID)0;
	    do {
		// Place vertices with degree d
		VID k = histo[d];
		if( k == 0 )
		    continue; // no vertices with degree d

		// Trick: chain all vertex lists together
		if( vlast != ~(VID)0 )
		    next[vlast] = first[d];
		vlast = last[d];
	    } while( d-- > 0 ); // loops over d == 0 as well

	    std::fill( &a[0], &a[n], 0 ); // important: a aliases last
	} else { // P > 1
	    // Identify the lowest degree for which
	    // ( #vertices at degree <= d ) <= P*(pmul-1)
	    // From d_crit to lower degrees, we need to try to obtain
	    // partitions with a multiple of pmul vertices. We need to
	    // start doing this from d_crit onwards, and it will be possible
	    // to obtain this, except perhaps for the final partition.
	    VID nv = 0;
	    VID d_crit = dmax;
	    for( VID d=0; d <= dmax; ++d ) {
		nv += histo[d];
		if( nv >= P * ( pmul - 1 ) ) {
		    d_crit = d;
		    break;
		}
	    }
	    std::cerr << "VEBO: critical d: " << d_crit << "\n";

	    // Assign vertices to partitions
	    VID d = dmax;
	    VID vlast = ~(VID)0;
	    do {
		// Place vertices with degree d
		VID k = histo[d];
		if( k == 0 )
		    continue; // no vertices with degree d

		// Trick: chain all vertex lists together
		if( vlast != ~(VID)0 )
		    next[vlast] = first[d];

		// Transition from balancing edge counts to vertex counts
		if( d == 0 )
		    load.convert_zdeg();

		// Infer how many vertices of degree d to place in each
		// partition.
		// We first determine how many vertices to place, then we choose
		// where vertices are placed.
		std::copy( &u[0], &u[P], &du[0] );

		for( VID j=0; j < k; ++j ) {
		    // place vertex in least-loaded partition, while tracking
		    // statistics on the spread of the loads
		    load.place( d );

		    // TODO: an alternative idea is to place a large number
		    //       of vertices at once if load.Delta > d. The idea
		    //       is that if there are at least K * P vertices left
		    //       for some K > 0, then surely at least K vertices
		    //       will be placed in the least-loaded partition.
		    //       As such, we can safely place K vertices at once.
		    if( load.Delta <= d && ((k-j-1) % P) == 0 ) {
			++j;
			// Remaining vertices are distributed equally over all
			// partitions
			// The scale-free property implies that most of the
			// vertices are handled by this loop
			if( j < k ) {
			    int r = (k-j) / P;
			    for( int p=0; p < P; ++p ) {
				u[p] += r;
				w[p] += r * d;
			    }
			}
			break;
		    }
		}

		// Rebalance such that partitions hold a multiple of pmul
		// vertices.
		// Assumes P > 1.
		if( pmul > 1 && d <= d_crit ) {
		    VID mv = std::min( u[0] % pmul, u[0] - du[0] );
		    u[0] -= mv;
		    w[0] -= mv * d;
		    for( unsigned p=1; p < P; ++p ) {
			u[p] += mv;
			w[p] += mv * d;
			mv = std::min( u[p] % pmul, u[p] - du[p] );
			u[p] -= mv;
			w[p] -= mv * d;
		    }
		    u[P-1] += mv;
		    w[P-1] += mv * d;
		}

		// Now determine which vertices are placed in each partition.
		// We aim to retain spatial locality by placing vertices in
		// the order they appear in the linked lists
		if( !intlv /* maxVL == 1 */ )
		    vlast = place_vertices( P, first, next, a, d, u, du, pmul );
		else
		    vlast = place_vertices_simd( P, gP, vP, first, next, a,
						 d, u, du );
	    } while( d-- > 0 ); // loops over d == 0 as well
	}

	// Assumption: u[p] = |{v: a[v] == p}|
/*
	{
	    std::fill( &du[0], &du[P], (VID)0 );
	    VID v = first[dmax];
	    for( VID i=0; i < n; ++i, v=next[v] ) {
		assert( ~v != 0 );
		du[a[v]]++;
	    }
	    for( unsigned p=0; p < P; ++p )
		assert( du[p] == u[p] );
	}
*/

	std::cerr << "VEBO: placement: " << tm.next() << "\n";

	// 3. Relabel vertices
	VID *s = du; // reuse array
	VID nwpad = n;
	if( pmul > 1 && P > 1 ) {
#if 0
	    // Shift vertices through to next partition to have multiple of pmul
	    s[0] = 0;
	    VID mask = pmul-1;
	    assert( (pmul & (pmul - 1)) == 0 );
	    if( u[0] > pmul ) {
		VID carry = u[0] & mask;
		assert( carry == 0 );
		u[0] -= carry;
		u[1] += carry;
		part.inuse_as_array()[0] = u[0];
	    } else {
		// at least a multiple of pmul
		nwpad += pmul - u[0];
		part.inuse_as_array()[0] = u[0];
		u[0] = pmul;
	    }
	    assert( u[0] > 0 );
	    part.as_array()[0] = u[0];
	    for( int p=1; p < P; ++p ) {
		if( p < P-1 ) {
		    if( u[p] > pmul ) {
			VID carry = u[p] & mask;
			assert( carry == 0 );
			u[p] -= carry;
			u[p+1] += carry;
			part.inuse_as_array()[p] = u[p];
		    } else {
			nwpad += pmul - u[p];
			part.inuse_as_array()[p] = u[p];
			u[p] = pmul;
		    }
		} else {
		    VID padding = -u[p] & mask;
		    nwpad += padding;
		    part.inuse_as_array()[p] = u[p];
		    u[p] += padding;
		}
		assert( u[p] > 0 );
		part.as_array()[p] = u[p];
		assert( (u[p] % pmul == 0) || p == P-1 );
		s[p] = s[p-1] + u[p-1];
	    }
	    assert( s[P-1] + u[P-1] >= n && s[P-1] + u[P-1] < max_nwpad );
	    assert( s[P-1] + u[P-1] == nwpad );

	    // Aggregate values
	    part.as_array()[P] = nwpad;
	    part.compute_starts_inuse(); // same as s[]
#else
	    // Already shifted vertices through. Calculate and record
	    // vertices/partition (including padding) and inuse-vertices
	    // (not including padding)
	    VID mask = pmul-1;
	    assert( (pmul & (pmul - 1)) == 0 );
	    s[0] = 0;
	    for( int p=0; p < P; ++p ) {
		if( p < P-1 ) {
		    if( u[p] >= pmul ) {
			part.inuse_as_array()[p] = u[p];
		    } else {
			// at least a multiple of pmul -> add padding
			nwpad += pmul - u[p];
			part.inuse_as_array()[p] = u[p];
			u[p] = pmul;
		    }
		} else {
		    VID padding = -u[p] & mask;
		    nwpad += padding;
		    part.inuse_as_array()[p] = u[p];
		    u[p] += padding;
		}
		assert( u[p] > 0 );
		part.as_array()[p] = u[p];
		assert( (u[p] % pmul == 0) || p == P-1 );
		if( p > 0 )
		    s[p] = s[p-1] + u[p-1];
	    }
	    assert( s[P-1] + u[P-1] >= n && s[P-1] + u[P-1] < max_nwpad );
	    assert( s[P-1] + u[P-1] == nwpad );

	    // Aggregate values
	    part.as_array()[P] = nwpad;
	    part.compute_starts_inuse(); // same as s[]
#endif
	} else {
	    s[0] = 0;
	    part.as_array()[0] = u[0];
	    for( int p=1; p < P; ++p ) {
		part.as_array()[p] = u[p];
		part.inuse_as_array()[p] = u[p];
		s[p] = s[p-1] + u[p-1];
	    }
	    if( nwpad % pmul ) {
		VID extra = ( VID(pmul) - nwpad ) % pmul;
		nwpad += extra;
		part.as_array()[P-1] += extra;
	    }

	    std::cerr << "VEBO: calculated nwpad: " << nwpad << "\n";

	    // Aggregate values
	    part.as_array()[P] = nwpad;
	    part.inuse_as_array()[0] = u[0];
	    part.compute_starts_inuse(); // same as s[]
	}

	// nwpad = part.start_of( P );
	// if( nwpad % pmul )
	// nwpad += ( VID(pmul) - nwpad ) % pmul;
	// part.appendv( nwpad - part.start_of( P ) );

	assert( part.start_of(P) == nwpad );
	assert( n <= nwpad );

	std::cerr << "VEBO: assignment to partitions: " << tm.next() << "\n";

	// Now create remapping array
	// Could parallelise this if previously we wired up linked list
	// per partition, in which case we could handle partitions in parallel
	if( intlv ) {
	    // The interleaved placement tries to put subsequent destination
	    // IDs on distinct vector lanes (blocks of vertices assigned to
	    // each partition are not interleaved). The benefit of this is
	    // locality, but possibly we could aim to turn gather/scatter
	    // operations into normal vector read operations.

	    // Leaves s[p] unmodified where p % maxVL == 0
	    // This should vectorise well
	    for( int p=0; p < P; ++p )
		u[p] = s[p-(p%maxVL)] + (p%maxVL);
	    s = u; // rename

	    VID mvfull[gP];
	    for( int gp=0; gp < gP; ++gp ) {
		VID mvf = part.as_array()[gp*maxVL];
		for( int p=gp*maxVL+1; p < (gp+1)*maxVL; ++p ) {
		    VID mv = part.as_array()[p];
		    if( mvf > mv )
			mvf = mv;
		}
		mvfull[gp] = mvf;
	    }

	    VID mvmax[gP];
	    for( int gp=0; gp < gP; ++gp ) {
		mvmax[gp] = s[gp*maxVL] + maxVL * mvfull[gp];
	    }
	    
	    // Alt for interleaved placement of vector lanes:
	    // remap_p[v] = s[a[v]];
	    // s[a[v]] += maxVL;
	    // but not exact if any of the lanes has a different number of
	    // vertices. Solution could be to reorder partitions such that the
	    // ones with more vertices occur first in every group of maxVL,
	    // could also sort them across all P partitions.
	    // Likely they turn out this way already, but need to account
	    // for situation where delta > 1.
	    // Remaining difficulty is to communicate which vertices appear in
	    // each partition in that case between here and the construction of
	    // the COO later on. Looks like we need to calculate msimd and mfull
	    // per group of maxVL already here, then assume interleaved IDs
	    // until mfull and something custom between mfull and msimd...
	    // The latter is important to put the vertices in the correct vector
	    // lane unless if they are zero in-degree vertices, in which case
	    // they are never used as a destination!
	    VID v = first[dmax];
	    VID *remap_p = remap.get();
	    for( VID i=0; i < n; ++i, v=next[v] ) {
		if( s[a[v]] < mvmax[a[v]/maxVL] ) {
		    remap_p[v] = s[a[v]];
		    s[a[v]] += maxVL;
		    if( s[a[v]] >= mvmax[a[v]/maxVL] ) {
			// as in the decoding function
			VID lane0 = a[v] - (a[v] % maxVL);
			VID r = 0;
			for( int vp=lane0; vp < a[v]; ++vp ) {
			    EID xtra = part.as_array()[vp] - mvfull[lane0/maxVL];
			    r += xtra;
			}
			s[a[v]] = mvmax[a[v]/maxVL] + r;
		    }
		} else {
		    remap_p[v] = s[a[v]]++;
		}
	    }
	    for( VID v=n; v < nwpad; ++v )
		remap_p[v] = v;
	} else if( pmul > 1 ) {

	    // Assumption: u[p] = |{v: a[v] == p}|
	    /*
	    {
		VID * du = new unsigned[P+1];
		std::fill( &du[0], &du[P], (VID)0 );
		// for( VID v=0; v < n; ++v ) {
		VID v = first[dmax];
		for( VID i=0; i < n; ++i, v=next[v] ) {
		    du[a[v]]++;
		}
		for( unsigned p=0; p < P; ++p ) {
		    assert( du[p] == u[p] );
		}
		delete[] du;
	    }
	    */

	    // Place vertices to partition in order of decreasing degree.
	    // Balance partitions out if a partition is full. This should be
	    // infrequent as it can happen only near the end of allocation.
#if 0
	    VID v = first[dmax];
	    VID *remap_p = remap.get();
	    for( VID i=0; i < n; ++i, v=next[v] ) {
		assert( v != ~(VID)0 );
		assert( a[v] != ~(VID)0 );
		// Move vertices over to next partition if current partition
		// is full. Partitions may have been shrunk so their vertex
		// count is a multiple of maxVL.
		while( s[a[v]] == part.start_of( a[v]+1 ) ) {
		    assert( 0 && "should be avoided" );
		    // Override allocation to have multiples of pmul / partition
		    std::cerr << "v=" << v << " overflows partition " << a[v]
			      << " move to next\n";
		    a[v]++;
		    // assert( idx[v+1] - idx[v] < 2 ); -- typically even 0
		}
		remap_p[v] = s[a[v]]++;
		a[v] = ~(VID)0;
	    }
#else
	    VID v = first[dmax];
	    VID *remap_p = remap.get();
	    for( VID i=0; i < n; ++i, v=next[v] ) {
		// assert( v != ~(VID)0 );
		// assert( a[v] != ~(VID)0 );
		// Move vertices over to next partition if current partition
		// is full. Partitions may have been shrunk so their vertex
		// count is a multiple of maxVL.
		// assert( s[a[v]] != part.start_of( a[v]+1 ) );
		remap_p[v] = s[a[v]]++;
		a[v] = ~(VID)0;
	    }
#endif

	    VID vpad = n;
	    for( VID p=0; p < P; ++p ) {
		while( s[p] != part.start_of( p+1 ) )
		    remap_p[vpad++] = s[p]++;
	    }
	    assert( vpad == part.get_vertex_range() );

	    for( VID p=0; p < P; ++p ) {
		assert( s[p] == part.start_of( p+1 )
			&& "Need to fill all partitions" );
	    }

	    // Correctness test
	    /*
	    for( VID v=0; v < n; ++v ) {
		VID w = remap_p[v];
		assert( a[v] == part.part_of(w) );
	    }
	    */
	} else {
	    VID v = first[dmax];
	    VID *remap_p = remap.get();
	    for( VID i=0; i < n; ++i, v=next[v] )
		remap_p[v] = s[a[v]]++;
	    for( VID v=n; v < nwpad; ++v )
		remap_p[v] = v;
	}

	std::cerr << "VEBO: remapping array: " << tm.next() << "\n";

	delete[] w;
	delete[] u;
	delete[] du; // frees also s

	mm_first.del( "VEBOReorder - mm_first" );
	mm_next.del( "VEBOReorder - mm_next" );
	mm_histo.del( "VEBOReorder - mm_histo" );
	mm_alloc.del( "VEBOReorder - mm_alloc" );

	std::cerr << "VEBO: done.\n";
    }

    // This method specialises VEBO to the cases of GraphGrind and Graptor.
    // packed == true: new version with better packing of high-degree vertices.
    template<bool packed>
    void vebo_graptor( const GraphCSx &csc, partitioner & part,
		       unsigned short pmul ) {
	timer tm;
	tm.start();
	
	std::cerr << "VEBO: start\n";
	
	VID n = csc.numVertices();         // Number of vertices
	int P = part.get_num_partitions(); // Total partitions

	std::cerr << "VEBO (GraphGrind+Graptor version): parameters: pmul="
		  << pmul << " n=" << n << " P=" << P << "\n";

	VID max_nwpad = n + P * pmul; // upper bound on padding

	// 1. Build chains of vertices with the same degree
	mmap_ptr<VID> mm_first, mm_next, mm_histo, mm_alloc;
	mm_first.allocate( n, numa_allocation_interleaved() );
	mm_alloc.allocate( max_nwpad, numa_allocation_interleaved() );
	mm_next.allocate( n + P, numa_allocation_interleaved() );
	mm_histo.allocate( n, numa_allocation_interleaved() );
	VID * first = mm_first.get();
	VID * last = mm_alloc.get();
	VID * next = mm_next.get();
	VID * histo = mm_histo.get(); // assume histo is zero-initialised
	parallel_loop( (VID)0, n, [&]( VID v ) {
	    histo[v] = 0;
	    first[v] = 0;
	    next[v] = ~(VID)0;
	} );

	std::cerr << "VEBO: setup: " << tm.next() << "\n";

	// Create bins of same-degree vertices. Organise them in a linked list
	// following their original order in the graph (i.e., the linked lists
	// are sorted by increasing degree). The sort order is important to
	// maintain locality that is present in the graph; otherwise would be
	// easy to parallelise.
	const EID * idx = csc.getIndex();
	EID dmax = 0;
	for( VID v=0; v < n; ++v ) {
	    VID d = idx[v+1] - idx[v];

	    if( histo[d] == 0 ) { // first occurence of degree
		first[d] = last[d] = v; // initialise 'linked list' for degree
		histo[d] = 1;
		if( dmax < d ) // track maximum degree seen
		    dmax = d;
	    } else {
		next[last[d]] = v; // add vertex to tail of list
		last[d] = v;
		++histo[d];
	    }
	}

	std::cerr << "VEBO: binning: " << tm.next() << "\n";

	// 2. Assign vertices to partitions. Create per-partition linked lists
	//    of vertices in the process.
	VID * pfirst = &next[n];
	VID * plast = new VID[P];
	EID *w = new EID[P];
	VID *u = new VID[P];
	VID *du = new VID[P];
	std::fill( &pfirst[0], &pfirst[P], ~VID(0) );
	std::iota( &plast[0], &plast[P], n );
	std::fill( &w[0], &w[P], EID(0) );
	std::fill( &u[0], &u[P], VID(0) );

	pstate load( P, u, w );

	if( P == 1 ) {
	    u[0] = n;
	    w[0] = csc.numEdges();

	    assert( 0 && "Not yet updated" );

	    // Link all vertex lists together
	    VID d = dmax;
	    VID vlast = ~(VID)0;
	    do {
		// Place vertices with degree d
		VID k = histo[d];
		if( k == 0 )
		    continue; // no vertices with degree d

		// Trick: chain all vertex lists together
		if( vlast != ~(VID)0 )
		    next[vlast] = first[d];
		vlast = last[d];
	    } while( d-- > 0 ); // loops over d == 0 as well
	} else { // P > 1
	    // Identify the lowest degree for which
	    // ( #vertices at degree <= d ) <= P*(pmul-1)
	    // From d_crit to lower degrees, we need to try to obtain
	    // partitions with a multiple of pmul vertices. We need to
	    // start doing this from d_crit onwards, and it will be possible
	    // to obtain this, except perhaps for the final partition.
	    VID nv = 0;
	    VID d_crit;
	    if( pmul > 1 ) {
		d_crit = dmax;
		for( VID d=0; d <= dmax; ++d ) {
		    nv += histo[d];
		    if( nv >= P * ( pmul - 1 ) ) {
			d_crit = d;
			break;
		    }
		}
	    } else
		d_crit = 0;
	    std::cerr << "VEBO: critical d: " << d_crit << "\n";

	    // Assign vertices to partitions
	    VID d = dmax;
	    do {
		// Place vertices with degree d
		VID k = histo[d];
		if( k == 0 )
		    continue; // no vertices with degree d

		// Transition from balancing edge counts to vertex counts
		if( d == 0 )
		    load.convert_zdeg();

		// Infer how many vertices of degree d to place in each
		// partition.
		// We first determine how many vertices to place, then we choose
		// where vertices are placed.
		std::copy( &u[0], &u[P], &du[0] );

		VID j=0;
		while( j < k ) {
		    // place vertex in least-loaded partition,
		    // while tracking statistics on the spread of the loads
		    if constexpr ( packed )
			load.place_packed( d, pmul );
		    else
			load.place( d );
		    ++j;

		    // TODO: an alternative idea is to place a large number
		    //       of vertices at once if load.Delta > d. The idea
		    //       is that if there are at least K * P vertices left
		    //       for some K > 0, then surely at least K vertices
		    //       will be placed in the least-loaded partition.
		    //       As such, we can safely place K vertices at once.
		    if( load.Delta <= d && ((k-j) % P) == 0 ) {
			// Remaining vertices are distributed equally over all
			// partitions
			// The scale-free property implies that most of the
			// vertices are handled by this loop
			if( j < k ) {
			    int r = (k-j) / P;
			    bool cont = false;
			    if( pmul > 1 && ( r % pmul ) != 0 ) {
				r -= r % pmul;
				cont = true;
			    }
			    for( VID p=0; p < P; ++p ) {
				u[p] += r;
				w[p] += r * d;
			    }
			    j += r * P;
			    if( !cont )
				break;
			} else
			    break;
		    }
		}

		// Rebalance such that partitions hold a multiple of pmul
		// vertices.
		// Assumes P > 1 and pmul > 1.
		if( pmul > 1 && d <= d_crit ) {
		    VID mv = std::min( u[0] % pmul, u[0] - du[0] );
		    u[0] -= mv;
		    w[0] -= mv * d;
		    for( unsigned p=1; p < P-1; ++p ) {
			u[p] += mv;
			w[p] += mv * d;
			mv = std::min( u[p] % pmul, u[p] - du[p] );
			u[p] -= mv;
			w[p] -= mv * d;
		    }
		    u[P-1] += mv;
		    w[P-1] += mv * d;
		}

		// Now determine which vertices are placed in each partition.
		// We aim to retain spatial locality by placing vertices in
		// the order they appear in the linked lists
		place_vertices_linked( P, first, next, plast, d, u, du );
	    } while( d-- > 0 ); // loops over d == 0 as well
	}

	std::cerr << "VEBO: placement (sequential, heap";
	if constexpr( packed )
	    std::cerr << ", packed@" << pmul;
	std::cerr << "): " << tm.next() << "\n";

	// 3. Determine partition sizes
	VID *s = du; // reuse array
	VID nwpad = n;
	if( pmul > 1 && P > 1 ) {
	    // Already shifted vertices through. Calculate and record
	    // vertices/partition (including padding) and inuse-vertices
	    // (not including padding)
	    VID mask = pmul-1;
	    assert( (pmul & (pmul - 1)) == 0 );
	    s[0] = 0;
	    for( int p=0; p < P; ++p ) {
		if( p < P-1 ) {
		    if( u[p] >= pmul ) {
			part.inuse_as_array()[p] = u[p];
		    } else if( u[p] == 0 ) {
			part.inuse_as_array()[p] = 0;
		    } else {
			// at least a multiple of pmul -> add padding
			nwpad += pmul - u[p];
			part.inuse_as_array()[p] = u[p];
			u[p] = pmul;
		    }
		} else {
		    VID padding = -u[p] & mask;
		    nwpad += padding;
		    part.inuse_as_array()[p] = u[p];
		    u[p] += padding;
		}
		// assert( u[p] > 0 );
		part.as_array()[p] = u[p];
		assert( (u[p] % pmul == 0) || p == P-1 );
		if( p > 0 )
		    s[p] = s[p-1] + u[p-1];
	    }
	    assert( s[P-1] + u[P-1] >= n && s[P-1] + u[P-1] < max_nwpad );
	    assert( s[P-1] + u[P-1] == nwpad );

	    // Aggregate values
	    part.as_array()[P] = nwpad;
	    part.compute_starts_inuse(); // same as s[]
	} else if( pmul == 1 && P > 1 ) {
	    s[0] = 0;
	    part.as_array()[0] = u[0];
	    for( int p=1; p < P; ++p ) {
		part.as_array()[p] = u[p];
		part.inuse_as_array()[p] = u[p];
		s[p] = s[p-1] + u[p-1];
	    }

	    std::cerr << "VEBO: calculated nwpad: " << nwpad << "\n";

	    // Aggregate values
	    part.as_array()[P] = nwpad;
	    part.inuse_as_array()[0] = u[0];
	    part.compute_starts_inuse(); // same as s[]
	} else {
	    s[0] = 0;
	    part.as_array()[0] = u[0];
	    for( int p=1; p < P; ++p ) {
		part.as_array()[p] = u[p];
		s[p] = s[p-1] + u[p-1];
	    }
	    if( nwpad % pmul ) {
		VID extra = ( VID(pmul) - nwpad ) % pmul;
		nwpad += extra;
		part.as_array()[P-1] += extra;
	    }

	    std::cerr << "VEBO: calculated nwpad: " << nwpad << "\n";

	    // Aggregate values
	    part.as_array()[P] = nwpad;
	    part.inuse_as_array()[0] = u[0];
	    part.compute_starts_inuse(); // same as s[]
	}

	assert( part.start_of(P) == nwpad );
	assert( n <= nwpad );

	std::cerr << "VEBO: partition boundaries: " << tm.next() << "\n";

	// Now create remapping array
	// Can parallelise this as previously we wired up linked list
	// per partition, and we can handle partitions in parallel.
	// Place vertices to partition in order of decreasing degree.
	VID *remap_p = remap.get();
	parallel_loop( (VID)0, (VID)P, [&]( VID p ) {
	    VID seq = s[p];
	    if( plast[p] != n+p ) {
		for( VID v=pfirst[p]; v != plast[p]; v=next[v] )
		    remap_p[v] = seq++;
		remap_p[plast[p]] = seq++;
	    }
	    s[p] = seq;
	} );

	// Note: if pmul == 1, then padding vertices appear strictly
	//       in the final partition only.
	VID vpad = n;
	for( VID p=0; p < P; ++p ) {
	    while( s[p] != part.start_of( p+1 ) )
		remap_p[vpad++] = s[p]++;
	}
	assert( vpad == part.get_vertex_range() );

	// Correctness test
	for( VID p=0; p < P; ++p ) {
	    assert( s[p] == part.start_of( p+1 )
		    && "Need to fill all partitions" );
	}

	std::cerr << "VEBO: remapping array: " << tm.next() << "\n";

	delete[] plast;
	delete[] w;
	delete[] u;
	delete[] du; // frees also s

	mm_first.del( "VEBOReorder - mm_first" );
	mm_next.del( "VEBOReorder - mm_next" );
	mm_histo.del( "VEBOReorder - mm_histo" );
	mm_alloc.del( "VEBOReorder - mm_alloc" );

	std::cerr << "VEBO: done.\n";
    }
};

template<typename lVID, typename lEID>
class VEBOReorderSIMD : public VEBOReorderState<VID,EID> {
public:
    using VID = lVID;
    using EID = lEID;
    
private:
    // In w = reverse[v], v is the new vertex ID, w is the old one
    // In v = remap[w], v is the new vertex ID, w is the old one

public:
    VEBOReorderSIMD() { }

    VEBOReorderSIMD( const GraphCSx &csc, partitioner & part,
		     unsigned short maxVL, bool interleave,
		     unsigned short roundup )
	: VEBOReorderSIMD( csc, part, maxVL, interleave, roundup,
#if VEBO_DISABLE
			   false
#elif VEBO_FORCE
			   true
#else
			   csc.max_degree() >= csc.numVertices() / (128*1024)
#endif
	    ) { }

    VEBOReorderSIMD( const GraphCSx &csc, partitioner & part,
		     unsigned short maxVL, bool interleave,
		     unsigned short roundup, bool enable_VEBO ) {
	assert( roundup % maxVL == 0 && "Roundup must be multiple of maxVL" );
	int P = maxVL * part.get_num_partitions();
	
	std::cerr << "VEBOReorderSIMD: npart=" << part.get_num_partitions() << '\n';

	VID n = csc.numVertices();
	// remap.Interleave_allocate( n );
	VID max_deg = csc.max_degree();

	if( !enable_VEBO ) {
	    VID next = n;
	    if( next % roundup != 0 )
		next += roundup - ( next % roundup );
	    
	    // Looks like a graph with low degrees, probably not worth to
	    // reorder. Better decision-making procedure would be to sample
	    // the degree distribution.
	    std::cerr << "VEBO: looks like non-power-law graph: max-degree: "
		      << max_deg << " vertices: " << n << "\n";
	    // Initialise partitioner - using n vertices, m edges
	    // partitionBalanceEdges( csc, part );
	    partitionBalanceEdges( csc, gtraits_getoutdegree<GraphCSx>( csc ),
				   part, roundup );
	    part.compute_starts();
	    // Adjust partitioner to append next-n vertices with 0 edges in
	    // final partition - note: we balance edges, so no impact on balance
	    part.appendv( next-n );
		
	    reverse.allocate( next, numa_allocation_interleaved() );
	    remap.allocate( next, numa_allocation_interleaved() );
	    map_partitionL( part, [&]( int p ) { 
		    VID s = part.start_of( p );
		    VID e = part.start_of( p+1 );
		    for( VID v=s; v < e; ++v ) {
			remap[v] = v;
			reverse[v] = v;
		    }
		} );
	} else {
	    // Looks like a  power-law graph
	    std::cerr << "VEBO: looks like power-law graph: max-degree: "
		      << max_deg << " vertices: " << n << "\n";

	    // Calculate mapping
	    vebo( csc, part, maxVL, interleave, roundup );
	    VID nwpad = part.get_num_elements();

	    // Calculate auxiliary reverse mapping
	    // reverse.Interleave_allocate( nwpad );
	    reverse.allocate( nwpad, numa_allocation_interleaved() );
	    std::fill( reverse.get(), &reverse.get()[nwpad], ~(VID)0 );
	    parallel_loop( (VID)0, nwpad, [&]( VID v ) {
		if( remap.get()[v] != ~(VID)0 ) {
		    assert( reverse.get()[remap.get()[v]] == ~(VID)0 );
		    reverse.get()[remap.get()[v]] = v;
		}
	    } );
	}

	// TODO: correctness checking
	/*
	for( VID v=0; v < n; ++v ) {
	    assert( v == reverse.get()[remap.get()[v]] );
	    assert( v == remap.get()[reverse.get()[v]] );
	}
	*/
    }

private:
    struct pstate {
	VID n;
	VID *u;
	EID *w;
	int pmax;
	EID Delta;

	pstate( VID n_, VID *u_, EID *w_ )
	    : n( n_ ), u( u_ ), w( w_ ), pmax( 0 ), Delta( 0 ) {
	    // Assume that initially all w[i] are zero
	}

	int place( VID d ) {
	    return d > 0 ? place_nzdeg( d ) : place_zdeg();
	}

	// Place a vertex with degree d, assuming d > 0
	int place_nzdeg( VID d ) {
	    int pmin = 0, pmin2 = 1;
	    if( w[pmin] > w[pmin2] )
		std::swap( pmin, pmin2 );
	    for( int i=2; i < n; ++i ) {
		if( w[pmin] > w[i] ) {
		    pmin2 = pmin;
		    pmin = i;
		} else if( w[pmin2] >= w[i] )
		    pmin2 = i;
	    }
	    // pmin is the least-loaded partition, pmin2 is
	    // the second least-loaded partition, possibly
	    // equally loaded to pmin.
	    assert( pmin != pmin2 );

	    w[pmin] += d;
	    u[pmin]++;

	    // Track highest loaded partition
	    if( w[pmin] > w[pmax] )
		pmax = pmin;

	    // Track Delta
	    Delta = w[pmax] - w[pmin2];

	    return pmin;
	}

	void convert_zdeg() {
	    // Now start tracking delta
	    int pmin = 0;
	    pmax = 0;
	    for( int i=1; i < n; ++i ) {
		if( u[pmin] > u[i] )
		    pmin = i;
		if( u[pmax] < u[i] )
		    pmax = i;
	    }
	    Delta = u[pmax] - u[pmin];
	}

	// Place a vertex with degree d, assuming d == 0
	int place_zdeg() {
	    int pmin = 0, pmin2 = 1;
	    if( w[pmin] > w[pmin2] )
		std::swap( pmin, pmin2 );
	    for( int i=2; i < n; ++i ) {
		if( u[pmin] > u[i] ) {
		    pmin2 = pmin;
		    pmin = i;
		} else if( u[pmin] == u[i] )
		    pmin2 = i;
	    }
	    // pmin is the least-loaded partition, pmin2 is
	    // the second least-loaded partition, possibly
	    // equally loaded to pmin.
	    assert( pmin != pmin2 );

	    // no need to update w[], vertex has zero degree
	    u[pmin]++;

	    // Track highest loaded partition
	    if( u[pmin] > u[pmax] )
		pmax = pmin;

	    // Track delta
	    Delta = u[pmax] - u[pmin2];

	    return pmin;
	}
    };

    VID place_vertices_simd( int P, int gP, int vP,
			     VID *first, VID *next, VID *a, int d,
			     VID *u, VID *du ) {
	VID v = first[d];
	VID vlast = v;
	for( int gp=0; gp < gP; ++gp ) {
	    // Maximise locality within groups of lanes, i.e., aim to
	    // have lanes accessed by one vector operation access vertices
	    // that had close vertex IDs in the original graph
	    bool done = false;
	    while( !done ) {
		VID old_v = v;
		for( int p=gp*vP; p < (gp+1)*vP; ++p ) {
		    if( u[p] > du[p] ) {
			a[v] = p;
			vlast = v;
			v = next[v];
			du[p]++;
		    }
		}
		if( v == old_v ) // check if progress made
		    done = true;
	    }
	}
	assert( v == 0 );
	for( int p=0; p < P; ++p ) {
	    assert( u[p] == du[p] );
	}
	return vlast;
    }
    VID place_vertices( int P, VID *first, VID *next, VID *a, int d,
			VID *u, VID *du ) {
	// currently processing vertices of degree d
	// du is the number of vertices per partition before starting degree d
	// u is the number of vertices per partition after degree d
	VID v = first[d];
	VID vlast = v;
	for( int p=0; p < P; ++p ) {
	    int kp = u[p] - du[p]; // number to place
	    for( int i=0; i < kp; ++i ) {
		a[v] = p;
		vlast = v;
		v = next[v];
	    }
	}
	assert( v == 0 );
	return vlast;
    }

    void vebo( const GraphCSx &csc, partitioner & part, unsigned short maxVL,
	       bool intlv, unsigned short roundup ) {
	std::cerr << "VEBO: start\n";
	
	VID n = csc.numVertices();          // Number of vertices
	int gP = part.get_num_partitions(); // Groups of vector lanes
	int vP = maxVL;                     // Vector-lane partitions
	int P = gP * maxVL;                 // Total partitions

	// 1. Build chains of vertices with the same degree
	mmap_ptr<VID> mm_first, mm_next, mm_histo, mm_alloc;
	mm_first.allocate( n, numa_allocation_interleaved() );
	mm_alloc.allocate( n, numa_allocation_interleaved() );
	mm_next.allocate( n, numa_allocation_interleaved() );
	mm_histo.allocate( n, numa_allocation_interleaved() );
	VID * first = mm_first.get();
	VID * last = mm_alloc.get();
	VID * next = mm_next.get();
	VID * histo = mm_histo.get(); // assume histo is zero-initialised
	parallel_loop( (VID)0, n, [&]( VID v ) {
	    histo[v] = 0;
	} );

	const EID * idx = csc.getIndex();
	EID dmax = 0;
	for( VID v=0; v < n; ++v ) {
	    VID d = idx[v+1] - idx[v];

	    if( __builtin_expect( histo[d] == 0, 0 ) ) { // first occurence of degree
		first[d] = last[d] = v; // initialise 'linked list' for degree
		histo[d] = 1;
		if( dmax < d ) // track maximum degree seen
		    dmax = d;
	    } else {
		next[last[d]] = v; // add vertex to tail of list
		last[d] = v;
		++histo[d];
	    }
	}

	// 2. Place vertices
	EID *w = new EID[P];
	VID *u = new VID[P];
	VID *du = new VID[P];
	std::fill( &w[0], &w[P], EID(0) );
	std::fill( &u[0], &u[P], VID(0) );

	pstate load( P, u, w );

	VID *a = last; // reuse array

	if( P == 1 ) {
	    u[0] = n;
	    w[0] = csc.numEdges();

	    // Link all vertex lists together
	    VID d = dmax;
	    VID vlast = ~(VID)0;
	    do {
		// Place vertices with degree d
		VID k = histo[d];
		if( k == 0 )
		    continue; // no vertices with degree d

		// Trick: chain all vertex lists together
		if( vlast != ~(VID)0 )
		    next[vlast] = first[d];
		vlast = last[d];
	    } while( d-- > 0 ); // loops over d == 0 as well

	    std::fill( &a[0], &a[n], 0 ); // important: a aliases last
	} else { // P > 1
	    VID d = dmax;
	    VID vlast = ~(VID)0;
	    do {
		// Place vertices with degree d
		VID k = histo[d];
		if( k == 0 )
		    continue; // no vertices with degree d

		// Trick: chain all vertex lists together
		if( vlast != ~(VID)0 )
		    next[vlast] = first[d];

		// Transition from balancing edge counts to vertex counts
		if( d == 0 )
		    load.convert_zdeg();

		// Infer how many vertices of degree d to place in each
		// partition.
		// We first determine how many vertices to place, then we choose
		// where vertices are placed.
		std::copy( &u[0], &u[P], &du[0] );

		VID j = 0;
		for( ; j < k; ++j ) {
		    // place vertex in least-loaded partition, while tracking
		    // statistics on the spread of the loads
		    load.place( d );

		    // TODO: an alternative idea is to place a large number
		    //       of vertices at once if load.Delta > d. The idea
		    //       is that if there are at least K * P vertices left
		    //       for some K > 0, then surely at least K vertices
		    //       will be placed in the least-loaded partition.
		    //       As such, we can safely place K vertices at once.
		    if( load.Delta <= d && ((k-j-1) % P) == 0 ) {
			++j;
			break;
		    }
		}

		// Remaining vertices are distributed equally over all
		// partitions
		// The scale-free property implies that most of the vertices
		// are handled by this loop
		if( j < k ) {
		    int r = (k-j) / P;
		    for( int p=0; p < P; ++p ) {
			u[p] += r;
			w[p] += r * d;
		    }
		}

		// Now determine which vertices are placed in each partition.
		// We aim to retain spatial locality by placing vertices in
		// the order they appear in the linked lists
		if( !intlv /* maxVL == 1 */ )
		    vlast = place_vertices( P, first, next, a, d, u, du );
		else
		    vlast = place_vertices_simd( P, gP, vP, first, next, a,
						 d, u, du );
		// vlast = last[d]; -- but last reused for a
	    } while( d-- > 0 ); // loops over d == 0 as well
	}

	// 3. Relabel vertices
	// u: number of vertices in each partition
	// s: starting positions of each partition
	VID *s = du; // reuse array
	VID *lgmax = new VID[gP];
	s[0] = 0;
	part.as_array()[0] = 0;
	VID npad = 0;
	for( int gp=0; gp < gP; ++gp ) {
	    VID gmax = 0;
	    // VID gmin = u[gp*vP];
	    for( int p=gp*vP; p < (gp+1)*vP; ++p ) {
		VID tmax = u[p];
		if( tmax % roundup )
		    tmax += roundup - ( tmax % roundup );
		if( gmax < tmax )
		    gmax = tmax;
		// VID tmin = u[p];
		// if( tmin % roundup )
		// tmin -= tmin % roundup;
		// if( gmin < tmin )
		// gmin = tmin;
	    }
	    // gmax: the number of vectors required to contain all vertices
	    // gmin: the number of vectors where all lanes are in use (src)
	    assert( ( gmax * maxVL ) % roundup == 0 );
	    lgmax[gp] = gmax;
	    s[(gp+1)*maxVL] = s[gp*maxVL] + gmax * maxVL;

	    VID nlpad = 0;
	    for( int p=gp*vP; p < (gp+1)*vP; ++p ) {
		// part.inuse_as_array()[p] = u[p];
		// part.as_array()[p] = gmax;
		nlpad += gmax - u[p];
		if( p > gp * vP ) // stagger
		    s[p] = s[gp*maxVL] + (p - gp*vP);
		// TODO: initialise s[p] as s[gp*vP] + (p - gp*vP) - staggered
		// partitioner should have only partitions per partition, not
		// per vector lane
		// if( p > 0 )
		    // s[p] = s[p-1] + part.as_array()[p-1];
		// else
		    // s[p] = 0;
		assert( s[p] % maxVL == p % maxVL );
	    }
	    part.as_array()[gp] = gmax * maxVL; // count
	    part.inuse_as_array()[gp] = gmax * maxVL - nlpad;
	    npad += nlpad;
	}
	part.as_array()[gP] = n + npad;
	// part.as_array()[P] = n + npad;
	// part.inuse_as_array()[P] = 0;

	// Aggregate values.
	part.compute_starts_inuse(); // same as s[]

	std::cerr << "check: " << part.start_of(P) << " n: " << n << " npad: " << npad << "\n";
	assert( part.start_of(gP) == n + npad );

	// Now create remapping array
	// remap.Interleave_allocate( n + npad );
	remap.allocate( n + npad, numa_allocation_interleaved() );
	// Could parallelise this if previously we wired up linked list
	// per partition, in which case we could handle partitions in parallel
	// Note: padding has been introduced, so constructing inverse needs care
	VID v = first[dmax];
	VID *remap_p = remap.get();
	for( VID i=0; i < n; ++i, v=next[v] ) {
	    remap_p[v] = s[a[v]];
	    s[a[v]] += maxVL; // interleaved allocation
	    assert( remap_p[v] < n + npad );
	}

	for( VID v=n; v < n + npad; ++v )
	    remap_p[v] = v;

	VID dummy = n;
	for( int p=0; p < P; ++p ) {
	    assert( s[p] % maxVL == p % maxVL );
	    int gp = p / maxVL;
	    VID tgt = part.start_of( gp+1 ) + ( p % maxVL );
	    assert( ( tgt - s[p] ) % maxVL == 0 );
	    VID k = ( tgt - s[p] ) / maxVL;
	    // int k = part.as_array()[p] - part.inuse_as_array()[p];
	    for( VID i=0; i < k; ++i ) {
		remap_p[dummy++] = s[p];
		s[p] += maxVL;
	    }
	    assert( s[p] == part.start_of(gp+1) + ( p % maxVL ) );
	    assert( s[p] % roundup == 0 );
	}

	delete[] lgmax;
	delete[] w;
	delete[] u;
	delete[] du; // frees also s

	mm_first.del( "VEBOReorderSIMD - mm_first" );
	mm_next.del( "VEBOReorderSIMD - mm_first" );
	mm_histo.del( "VEBOReorderSIMD - mm_first" );

	std::cerr << "VEBO: done.\n";
    }
};

template<typename lVID, typename lEID>
class VEBOReorderIdempotent {
public:
    using VID = lVID;
    using EID = lEID;
    
public:
    VEBOReorderIdempotent( const GraphCSx &csc, partitioner & part ) {
	// TODO: need to fill out partitioner
	partitionBalanceEdges( csc, part ); 
    }
    VEBOReorderIdempotent() { }

    RemapVertexIdempotent<VID> remapper() const {
	return RemapVertexIdempotent<VID>();
    }

    VID originalID( VID v ) const { return v; }
    VID remapID( VID v ) const { return v; }

    void del() { }
};

#endif // GRAPHGRIND_GRAPH_VEBOREORDER_H
