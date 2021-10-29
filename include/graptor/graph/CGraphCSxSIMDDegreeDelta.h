// -*- c++ -*-
#ifndef GRAPTOR_GRAPH_CGRAPHCSXSIMDDEGREEDELTA_H
#define GRAPTOR_GRAPH_CGRAPHCSXSIMDDEGREEDELTA_H

namespace detail {

template<typename cVID, unsigned short DegreeBits>
struct CheckBitsDelta {
    static constexpr bool verbose = false;
    CheckBitsDelta( unsigned short maxVL_ )
	: maxVL( maxVL_ ),
	  dbpl( std::max( 1, (DegreeBits+maxVL-1)/maxVL ) ),
	  mask( ~( (VID(1)<<(sizeof(cVID)*8-dbpl))-1 ) ),
	  overflows( 0 ) { }

    void record( VID seq, EID pos, VID value, VID abs_value, VID deg ) {
	if( ( value & mask ) != 0 )
	    ++overflows;
/*
	VID dmax = ( VID(1) << 16 );
	{
	    if( seq < 4 ) { // 4 vectors in first block
		; // no overflows
	    } else if( (seq % (4*(dmax-1))) != 0 ) {
		if( ( value & ~((VID(1)<<22)-1) ) != 0 )
		    ++overflows;
	    } else {
		if( ( value & ~((VID(1)<<21)-1) ) != 0 )
		    ++overflows;
	    }
	}
*/
    }
    void invalid( VID seq, EID pos, VID deg ) { }

    bool success() const { return overflows == 0; }
    EID nOverflows() const { return overflows; }

private:
    unsigned short maxVL, dbpl;
    VID mask;
    EID overflows;
};

template<typename cVID, unsigned short DegreeBits>
struct EncodeBitsDelta {
    static constexpr bool verbose = true;
    EncodeBitsDelta( unsigned short maxVL_, cVID * edges_ )
	: maxVL( maxVL_ ),
	  dbpl( std::max( 1, (DegreeBits+maxVL-1)/maxVL ) ),
	  vmask( (VID(1)<<(sizeof(cVID)*8-dbpl))-1 ),
	  dmax( VID(1) << (dbpl*maxVL) ),
	  overflows( 0 ),
	  edges( edges_ ) { }

    void record( VID seq, EID pos, VID value, VID abs_value, VID deg ) {
	cVID r = value & vmask;
	if( VID( r ) != value )
	    ++overflows;

	record_unchecked( seq, pos, value, abs_value, deg );
    }

    void record_unchecked( VID seq, EID pos, VID value, VID abs_value, VID deg ) {
	cVID r = value;
	assert( deg > 0 );
	if( deg == 1 || deg == 2 ) {
	    edges[pos] = abs_value;
	    return;
	}
#if 1
	if( (seq % (3*(dmax-1))) == 0 ) {
	    cVID d = 0;
	    VID lane = pos % maxVL;
	    if( deg - seq > 3*(dmax-1) ) {
		d = ( dmax - 1 ) >> ( lane * dbpl );
		d &= ( cVID(1) << dbpl ) - 1;
	    } else {
		// Rescale degree from [1..3*dmax] to [0..dmax-1]
		VID deg3 = (deg - seq - 1) / 3;
		d = deg3 >> (lane * dbpl);
		d &= ( cVID(1) << dbpl ) - 1;
	    }
	    r = value & vmask;
	    r |= d << ( sizeof(cVID) * 8 - dbpl );
	}
#else
	if( (seq % (3*dmax-9)) == 0 ) {
	    cVID d = 0;
	    VID lane = pos % maxVL;
	    if( deg - seq > 3*dmax-9 ) {
		d = ( dmax - 1 ) >> ( lane * dbpl );
		d &= ( cVID(1) << dbpl ) - 1;
	    } else if( deg > 3 ) {
		// Rescale degree from [4..3*dmax-8] to [0..dmax-1]
		VID deg3 = ( (deg - 3 - seq - 1) / 3 ) + 4;
		d = deg3 >> (lane * dbpl);
		d &= ( cVID(1) << dbpl ) - 1;
	    } else {
		d = ( deg - seq - 1 ) >> (lane * dbpl);
		d &= ( cVID(1) << dbpl ) - 1;
	    }
	    r = value & vmask;
	    r |= d << ( sizeof(cVID) * 8 - dbpl );
	}
#endif

	edges[pos] = r;
    }
    void invalid( VID seq, EID pos, VID deg ) {
	record_unchecked( seq, pos, ~VID(0), ~VID(0), deg );
    }

    bool success() const { return overflows == 0; }
    EID nOverflows() const { return overflows; }

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
class GraphCSxSIMDDegreeDelta {
    VID n, nv;
    EID m, mv, mv1, mv2;
    unsigned short maxVL;
    mmap_ptr<VID> edges;

    static constexpr unsigned short DegreeBits = 12; // 16;

public:
    GraphCSxSIMDDegreeDelta() { }
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

	mv = mv1 = mv2 = 0;
	for( VID v=lo; v < hi; v += maxVL ) {
	    VID r = remap.first[v];
	    VID deg = idx[r+1] - idx[r];
	    if( deg == 2 )
		mv2 += 2 * maxVL;
	    else if( deg == 1 )
		mv1 += maxVL;
	    else {
#if 1
		if( deg != 0 && ( (deg-1) % 3 ) != 0 )
		    deg += 3 - ( (deg-1) % 3 );
#else
		if( deg > 3 && ( (deg-1) % 3 ) != 0 )
		    deg += 3 - ( (deg-1) % 3 );
#endif
	    }
	    mv += maxVL * deg;
	}
	assert( mv >= m );
	assert( mv1 <= mv );
	assert( mv2 <= mv );

	// Check that indices are short enough to encode degree within
	// graph.
	{
	    detail::CheckBitsDelta<VID,DegreeBits> checker( maxVL );
	    import_phase( Gcsc, lo, hi, maxVL, mv, remap, checker );
	    std::cerr << "overflows: " << checker.nOverflows() << "\n";
	    // assert( checker.success() );
	}
	    
	// Allocate data structures
	if( allocation == -1 )
	    allocateInterleaved();
	else
	    allocateLocal( allocation );

	// Now that we know that the degree bits are available, encode degrees.
	{
	    detail::EncodeBitsDelta<VID,DegreeBits> encoder( maxVL, edges.get() );
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
	    // Degree 1, 2 are not degree-encoded
	    if( deg > 2 ) {
#if 1
		// 1, 2, 3 -> 4
		// 4, 5, 6 -> 7
		// etc
		// 1...3*dmax -> 0...(dmax-1)
		if( ( (deg-1) % 3 ) != 0 )
		    deg += 3 - ( (deg-1) % 3 );
#else
		// 1, 2, 3 -> 1, 2, 3
		// 4, 5, 6 -> 7
		// etc
		// 1...??? -> 0...(dmax-1) = 0...2 + 3...(dmax-1)
		// 1, 2, 3 ||  3*1+1(=4) ... 3*(dmax-3)+1=3*dmax-8
		// 1, 2, 3, 4 -> themselves
		// 5, 6, 7 -> 7
		// 3*(dmax-3) -> 3*(dmax-3)+3-2 = 3*dmax-8
		// etc
		if( deg > 3 ) {
		    if( ( (deg-1) % 3 ) != 0 )
			deg += 3 - ( (deg-1) % 3 );
	    }
#endif
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

		    if( Functor::verbose && false ) {
			std::cerr << "encode v=" << v << " l=" << l
				  << " lnxt=" << lnxt
				  << " deg=" << deg
				  << " ldeg=" << ldeg << "\n";
		    }

		    for( VID j=0; j < ldeg; ++j ) {
			assert( j == 0 || buf[j] > buf[j-1] );
			// edges[lnxt] = buf[j];
			if( Functor::verbose && false )
			    std::cerr << buf[j] << ' ' << (v+l) << '\n';
			if( j > 0 )
			    fnc.record( j, lnxt, buf[j] - buf[j-1] - 1, buf[j], deg );
			else
			    fnc.record( j, lnxt, buf[j], buf[j], deg );
			lnxt += maxVL;
		    } 
		    for( VID j=ldeg; j < deg; ++j ) {
			if( Functor::verbose && false )
			    std::cerr << (~(VID)0) << ' ' << (v+l) << '\n';
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
    VID numSIMDEdgesGT1() const { return mv - mv1; }
    VID numSIMDEdgesGT2() const { return mv - mv2 - mv1; }

    EID numInactive() const {
	EID nia = 0;
	VID mask = ( VID(1) << getDegreeShift() ) - 1;
	for( EID s=0; s < mv; s++ ) {
	    if( (edges[s] & mask) == mask )
		++nia;
	}
	return nia;
    }
    float fracInactive() const { return float(numInactive()) / mv; }

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


#endif // GRAPTOR_GRAPH_CGRAPHCSXSIMDDEGREEDELTA_H
