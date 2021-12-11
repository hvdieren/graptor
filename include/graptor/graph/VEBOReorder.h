// -*- C++ -*-
#ifndef GRAPHGRIND_GRAPH_VEBOREORDER_H
#define GRAPHGRIND_GRAPH_VEBOREORDER_H

#include "graptor/mm.h"
#include "graptor/graph/remap.h"
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
    mmap_ptr<lVID> m_alloc;
    lEID * w;                // Number of edges per partition

public:
    VEBOReorderState() : w( nullptr ) { }   // Private

    const lVID *getRemap() const { return remap.get(); }
    const lVID *getReverse() const { return reverse.get(); }
    const lVID *getAlloc() const { return m_alloc.get(); }
    const lEID *getW() const { return w; }

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
	m_alloc.del( "VEBOReorder - m_alloc" );
	if( w )
	    delete[] w;
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
	m_alloc.allocate( next, numa_allocation_interleaved() );

	std::cerr << "ReorderDegreeSort: vertices: " << n
		  << " extended to: " << next << "\n";

	// Do degree sorting, use remap as scratch array
	degree_sort( csc.getIndex(), n, remap, reverse );

	// Extend range. Keep additional vertices at the end of the iteration
	// range so they can be easily skipped during processing.
	for( VID v=n; v < next; ++v )
	    reverse[v] = v;
	
	// Invert reverse array. Set m_alloc.
	std::fill( remap.get(), &remap.get()[next], ~(VID)0 );
	parallel_for( VID v=0; v < next; ++v ) {
	    assert( remap.get()[reverse.get()[v]] == ~(VID)0 );
	    remap.get()[reverse.get()[v]] = v;
	}
	
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
    // mmap_ptr<VID> remap;
    // mmap_ptr<VID> reverse;
    // mmap_ptr<VID> m_alloc;
    // EID * w;                // Number of edges per partition

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

	VID max_deg = csc.max_degree();
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
	    // partitionBalanceEdges( csc, part );
	    partitionBalanceEdges( csc, gtraits_getoutdegree<GraphCSx>( csc ),
				   part, maxVL );
		
	    //reverse.Interleave_allocate( n );
	    //m_alloc.Interleave_allocate( n );
	    reverse.allocate( n, numa_allocation_interleaved() );
	    m_alloc.allocate( n, numa_allocation_interleaved() );
	    map_partitionL( part, [&]( int p ) { 
		    VID s = part.start_of( p );
		    VID e = part.start_of( p+1 );
		    for( VID v=s; v < e; ++v ) {
			remap[v] = v;
			reverse[v] = v;
			m_alloc[v] = p;
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
	    parallel_for( VID v=0; v < nwpad; ++v ) {
		assert( reverse.get()[remap.get()[v]] == ~(VID)0 );
		reverse.get()[remap.get()[v]] = v;
	    }
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

/*
*/

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
#if 0
	    if( d == 0 && pmul != 1 && p < P-1 ) {
		// Special case:
		// Ensure vertices per partition is multiple of pmul
		// This is required for applying VEBO to SlimSell/CSxSIMD
		// while retaining aligned vector load/store to destination
		// arrays.
		int ndiff = u[p] % pmul;
		if( ndiff != 0 ) {
		    // Try to get vertices from later partitions
		    if( kp < ndiff ) {
			int mv = pmul - kp - 1; // move this many vertices to p
			for( int q=p+1; q < P; ++q ) {
			    int kq = u[q] - du[q]; // number to place
			    if( kq >= mv ) {
				u[q] -= mv;
				u[p] += mv;
				mv -= mv;
				ndiff = u[p] % pmul;
				break;
			    } else {
				u[q] -= kq;
				u[p] += kq;
				mv -= kq;
				ndiff = u[p] % pmul;
			    }
			}
			kp = u[p] - du[p];
		    }

		    assert( kp >= ndiff && "Need movable vertices" );
		    u[p] -= ndiff; // move some vertices to next partition
		    kp -= ndiff;
		    u[p+1] += ndiff;
		}
	    }
#endif
	    for( int i=0; i < kp; ++i ) {
		a[v] = p;
		vlast = v;
		v = next[v];
	    }
	}
	assert( v == ~(VID)0 );
	return vlast;
    }

    void vebo( const GraphCSx &csc, partitioner & part, unsigned short maxVL,
	       bool intlv, unsigned short pmul ) {
	if( pmul > 1 && intlv == false && maxVL == 1 ) {
	    vebo_graptor( csc, part, pmul );
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
	mmap_ptr<VID> mm_first, mm_next, mm_histo;
	mm_first.allocate( n, numa_allocation_interleaved() );
	m_alloc.allocate( max_nwpad, numa_allocation_interleaved() );
	mm_next.allocate( n, numa_allocation_interleaved() );
	mm_histo.allocate( n, numa_allocation_interleaved() );
	VID * first = mm_first.get();
	VID * last = m_alloc.get();
	VID * next = mm_next.get();
	VID * histo = mm_histo.get(); // assume histo is zero-initialised
	parallel_for( VID v=0; v < n; ++v ) {
	    histo[v] = 0;
	    first[v] = 0;
	    next[v] = ~(VID)0;
	}

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
	w = new EID[P];
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

	delete[] u;
	delete[] du; // frees also s

	mm_first.del( "VEBOReorder - mm_first" );
	mm_next.del( "VEBOReorder - mm_next" );
	mm_histo.del( "VEBOReorder - mm_histo" );

	std::cerr << "VEBO: done.\n";
    }

    void vebo_graptor( const GraphCSx &csc, partitioner & part,
		       unsigned short pmul ) {
	timer tm;
	tm.start();
	
	std::cerr << "VEBO: start\n";
	
	VID n = csc.numVertices();         // Number of vertices
	int P = part.get_num_partitions(); // Total partitions

	std::cerr << "VEBO: parameters (Graptor version): pmul=" << pmul
		  << " n=" << n << " P=" << P << "\n";

	VID max_nwpad = n + P * pmul; // upper bound on padding

	// 1. Build chains of vertices with the same degree
	mmap_ptr<VID> mm_first, mm_next, mm_histo;
	mm_first.allocate( n, numa_allocation_interleaved() );
	m_alloc.allocate( max_nwpad, numa_allocation_interleaved() );
	mm_next.allocate( n, numa_allocation_interleaved() );
	mm_histo.allocate( n, numa_allocation_interleaved() );
	VID * first = mm_first.get();
	VID * last = m_alloc.get();
	VID * next = mm_next.get();
	VID * histo = mm_histo.get(); // assume histo is zero-initialised
	parallel_for( VID v=0; v < n; ++v ) {
	    histo[v] = 0;
	    first[v] = 0;
	    next[v] = ~(VID)0;
	}

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
	w = new EID[P];
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
		vlast = place_vertices( P, first, next, a, d, u, du, pmul );
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
	if( P > 1 ) {
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

	std::cerr << "VEBO: assignment to partitions: " << tm.next() << "\n";

	// Now create remapping array
	// Could parallelise this if previously we wired up linked list
	// per partition, in which case we could handle partitions in parallel

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
	    // count is a multiple of pmul.
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
	    // count is a multiple of pmul.
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
	
	std::cerr << "VEBO: remapping array: " << tm.next() << "\n";

	delete[] u;
	delete[] du; // frees also s

	mm_first.del( "VEBOReorder - mm_first" );
	mm_next.del( "VEBOReorder - mm_next" );
	mm_histo.del( "VEBOReorder - mm_histo" );

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
    // mmap_ptr<VID> remap;
    // mmap_ptr<VID> reverse;
    // mmap_ptr<VID> m_alloc;
    // EID * w;                // Number of edges per partition

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
		
	    // reverse.Interleave_allocate( next );
	    // m_alloc.Interleave_allocate( next );
	    // remap.Interleave_allocate( next );
	    reverse.allocate( next, numa_allocation_interleaved() );
	    m_alloc.allocate( next, numa_allocation_interleaved() );
	    remap.allocate( next, numa_allocation_interleaved() );
	    map_partitionL( part, [&]( int p ) { 
		    VID s = part.start_of( p );
		    VID e = part.start_of( p+1 );
		    for( VID v=s; v < e; ++v ) {
			remap[v] = v;
			reverse[v] = v;
			m_alloc[v] = p;
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
	    parallel_for( VID v=0; v < nwpad; ++v ) {
		if( remap.get()[v] != ~(VID)0 ) {
		    assert( reverse.get()[remap.get()[v]] == ~(VID)0 );
		    reverse.get()[remap.get()[v]] = v;
		}
	    }
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
	mmap_ptr<VID> mm_first, mm_next, mm_histo;
	// mm_first.Interleave_allocate( n );
	// m_alloc.Interleave_allocate( n );
	// mm_next.Interleave_allocate( n );
	// mm_histo.Interleave_allocate( n );
	mm_first.allocate( n, numa_allocation_interleaved() );
	m_alloc.allocate( n, numa_allocation_interleaved() );
	mm_next.allocate( n, numa_allocation_interleaved() );
	mm_histo.allocate( n, numa_allocation_interleaved() );
	VID * first = mm_first.get();
	VID * last = m_alloc.get();
	VID * next = mm_next.get();
	VID * histo = mm_histo.get(); // assume histo is zero-initialised
	parallel_for( VID v=0; v < n; ++v )
	    histo[v] = 0;

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
	w = new EID[P];
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
