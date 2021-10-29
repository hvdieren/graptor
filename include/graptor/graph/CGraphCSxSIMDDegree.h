// -*- c++ -*-
#ifndef GRAPTOR_GRAPH_CGRAPHCSXSIMDDEGREE_H
#define GRAPTOR_GRAPH_CGRAPHCSXSIMDDEGREE_H

// TODO:
// - consider expanding the range of degree's that can be encoded by forcing
//   it to be a multiple of 2, 3, ... Price: need some padding vectors with
//   all lanes disabled.
//   Expense will be incurred mostly for low-degree vertices.

namespace detail {

template<typename cVID, unsigned short DegreeBits>
struct CheckBits {
    CheckBits( unsigned short maxVL_ )
	: maxVL( maxVL_ ),
	  mask( ~( (VID(1)<<(sizeof(cVID)*8-((DegreeBits+maxVL-1)/maxVL)))-1 ) ),
	  overflows( 0 ) { }

    void record( VID seq, EID pos, VID value, VID deg ) {
	if( ( value & mask ) != 0 )
	    ++overflows;
    }
    void invalid( VID seq, EID pos, VID deg ) { }

    bool success() const { return overflows == 0; }

private:
    unsigned short maxVL;
    VID mask;
    EID overflows;
};

template<typename cVID, unsigned short DegreeBits>
struct EncodeBits {
    EncodeBits( unsigned short maxVL_, cVID * edges_ )
	: maxVL( maxVL_ ),
	  dbpl( std::max( 1, (DegreeBits+maxVL-1)/maxVL ) ),
	  vmask( (VID(1)<<(sizeof(cVID)*8-dbpl))-1 ),
	  dmax( VID(1) << DegreeBits ),
	  overflows( 0 ),
	  edges( edges_ ) { }

    void record( VID seq, EID pos, VID value, VID deg ) {
	cVID r = value & vmask;
	if( VID( r ) != value )
	    ++overflows;

	record_unchecked( seq, pos, r, deg );
    }

    void record_unchecked( VID seq, EID pos, cVID r, VID deg ) {
	assert( deg > 0 );
	if( (seq % (dmax-1)) == 0 ) {
	    cVID d = 0;
	    VID lane = pos % maxVL;
	    if( deg - seq > dmax-1 ) {
		d = ( dmax - 1 ) >> ( lane * dbpl );
		d &= ( cVID(1) << dbpl ) - 1;
	    } else {
		// Rescale degree from [1..dmax] to [0..dmax-1]
		d = (deg - seq - 1) >> (lane * dbpl);
		d &= ( cVID(1) << dbpl ) - 1;
	    }
	    r |= d << ( sizeof(cVID) * 8 - dbpl );
	}

	edges[pos] = r;
    }
    void invalid( VID seq, EID pos, VID deg ) {
	record_unchecked( seq, pos, cVID(vmask), deg );
    }

    bool success() const { return overflows == 0; }

private:
    unsigned short maxVL, dbpl;
    VID vmask, dmax;
    EID overflows;
    VID * edges;
};

} // namespace detail

// Assuming VEBO was applied, consider delta-encoding of the degree/index
// Starting points should be recoverable only at the start of
// partitions, i.e., index only needed for first vertex, remainder
// can be delta-degree
class GraphCSxSIMDDegree {
    VID n, nv;
    EID m, mv;
    unsigned short maxVL;
    mmap_ptr<VID> edges;

    static constexpr unsigned short DegreeBits = 12; // 16;

public:
    GraphCSxSIMDDegree() { }
    void del() {
	edges.del();
    }
    void import( const GraphCSx & Gcsc,
		 VID lo, VID hi, unsigned short maxVL_,
		 std::pair<const VID *, const VID *> remap,
		 int allocation ) {
	maxVL = maxVL_;

	assert( lo % maxVL == 0 );
	// assert( hi % maxVL == 0 ); -- not so in final partition

	const EID * idx = Gcsc.getIndex();
	const VID * edg = Gcsc.getEdges();

	// Calculate dimensions of SIMD representation
	n = Gcsc.numVertices();
	m = 0;
	for( nv=lo; nv < hi; nv++ ) {
	    VID r = remap.first[nv];
	    m += idx[r+1] - idx[r];
	}

	nv = ( ( hi - lo + maxVL - 1 ) / maxVL ) * maxVL;
	assert( nv >= (hi-lo) && nv < (hi-lo) + maxVL );

	mv = 0;
	for( VID v=lo; v < hi; v += maxVL ) {
	    VID r = remap.first[v];
	    mv += maxVL * ( idx[r+1] - idx[r] );
	}
	assert( mv >= m );

	// Check that indices are short enough to encode degree within
	// graph.
	{
	    detail::CheckBits<VID,DegreeBits> checker( maxVL );
	    import_phase( Gcsc, lo, hi, maxVL, mv, remap, checker );
	    assert( checker.success() );
	}
	    
	// Allocate data structures
	if( allocation == -1 )
	    allocateInterleaved();
	else
	    allocateLocal( allocation );

	// Now that we know that the degree bits are available, encode degrees.
	{
	    detail::EncodeBits<VID,DegreeBits> encoder( maxVL, edges.get() );
	    import_phase( Gcsc, lo, hi, maxVL, mv, remap, encoder );
	    assert( encoder.success() );
	}
    }

private:
    template<typename Functor>
    static void import_phase( const GraphCSx & Gcsc,
			      VID lo, VID hi, unsigned short maxVL,
			      EID mv,
			      std::pair<const VID *, const VID *> remap,
			      Functor & fnc ) {
	// Calculate dimensions of SIMD representation
	VID n = Gcsc.numVertices();
	const EID * idx = Gcsc.getIndex();
	const VID * edg = Gcsc.getEdges();
	VID nv = maxVL * ( ( hi - lo + maxVL - 1 ) / maxVL );
	assert( nv >= (hi-lo) && nv < (hi-lo) + maxVL );
	    
	// In w = remap.first[v], v is the new vertex ID, w is the old one
	// In v = remap.second[w], v is the new vertex ID, w is the old one

	VID maxdeg = idx[remap.first[0]+1] - idx[remap.first[0]];
	VID * buf = new VID[maxdeg];

	EID nxt = 0;
	for( VID v=lo; v < lo+nv; v += maxVL ) {
	    VID r = remap.first[v];
	    VID deg = idx[r+1] - idx[r];

	    if( deg == 0 ) {
		// std::cerr << "Stop zero-deg at " << v << "\n";
		assert( nxt == mv );
		break;
	    }
	    
	    for( unsigned short l=0; l < maxVL; ++l ) { // vector lane
		VID vv = v + l;
		EID lnxt = nxt + l;
		if( vv < n ) {
		    // Construct a list of sorted remapped sources
		    VID ww = remap.first[vv];
		    VID ldeg = idx[ww+1] - idx[ww];
		    assert( ldeg <= deg );
		    assert( ldeg <= maxdeg );
		    for( VID j=0; j < ldeg; ++j )
			buf[j] = remap.second[edg[idx[ww]+j]];
		    std::sort( &buf[0], &buf[ldeg] );

		    for( VID j=0; j < ldeg; ++j ) {
			// edges[lnxt] = buf[j];
			fnc.record( j, lnxt, buf[j], deg );
			lnxt += maxVL;
		    } 
		    for( VID j=ldeg; j < deg; ++j ) {
			// edges[lnxt] = ~(VID)0;
			fnc.invalid( j, lnxt, deg );
			lnxt += maxVL;
		    }
		} else {
		    VID ww = ~(VID)0;
		    for( VID j=0; j < deg; ++j ) {
			// edges[lnxt] = ~(VID)0;
			fnc.invalid( j, lnxt, deg );
			lnxt += maxVL;
		    } 
		}
	    }
	    nxt += maxVL * deg;
	}
	assert( nxt == mv );
	delete[] buf;
    }

public:
    VID *getEdges() { return edges.get(); }
    const VID *getEdges() const { return edges.get(); }

    VID numVertices() const { return n; }
    EID numEdges() const { return m; }

    VID numSIMDVertices() const { return nv; }
    EID numSIMDEdges() const { return mv; }

    unsigned short getMaxVL() const { return maxVL; }
    unsigned short getDegreeBits() const { return (DegreeBits+maxVL-1)/maxVL; }
    unsigned short getDegreeShift() const { return sizeof(VID)*8 - getDegreeBits(); }

private:
    void allocateInterleaved() {
	assert( nv % maxVL == 0 );
	edges.allocate( mv, sizeof(VID)*maxVL, numa_allocation_interleaved() );
    }
    void allocateLocal( int numa_node ) {
	assert( nv % maxVL == 0 );
	edges.allocate( mv, sizeof(VID)*maxVL, numa_allocation_local( numa_node ) );
    }
};


#endif // GRAPTOR_GRAPH_CGRAPHCSXSIMDDEGREE_H
