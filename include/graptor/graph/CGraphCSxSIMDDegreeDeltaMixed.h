// -*- c++ -*-
#ifndef GRAPTOR_GRAPH_CGRAPHCSXSIMDDEGREEDELTAMIXED_H
#define GRAPTOR_GRAPH_CGRAPHCSXSIMDDEGREEDELTAMIXED_H

namespace detail {

template<typename fVID, typename hVID, unsigned short DegreeBits>
struct EncDeltaSize {
    EncDeltaSize( unsigned short maxVL_ )
	: maxVL( maxVL_ ),
	  dbpl( std::max( 1, (DegreeBits+maxVL-1)/maxVL ) ),
	  vmask( (fVID(1)<<(sizeof(hVID)*8-dbpl))-1 ),
	  dmax( fVID(1) << (dbpl*maxVL-1) ), // loose 1 bit for end-of-run
	  overflows( 0 ), size( 0 ) { }

    void set_degree( char *where, fVID deg, bool last_for_v ) { }
    void set_degree( char *where, fVID deg ) { }
    void set_fullwidth( char *where, bool wfull ) { }
    void set_overshoot( char *where ) { }
    
    void record_short( char * &where, hVID delta ) {
	size += sizeof(hVID);
    }

    void record_wide( char * &where, fVID delta ) {
	size += sizeof(fVID);
    }

    bool success() const { return overflows == 0; }
    EID nOverflows() const { return overflows; }
    EID nBytes() const { return size; }
    void alignSize() {
	if( size % (sizeof(fVID) * maxVL) != 0 )
	    size += (sizeof(fVID) * maxVL) - ( size % (sizeof(fVID) * maxVL) );
    }

public:
    const unsigned short maxVL, dbpl;
    const fVID vmask, dmax;
    EID overflows;
    EID size;
};

template<typename fVID, unsigned short DegreeBits>
struct EncDstParSize {
    EncDstParSize( unsigned short maxVL_ )
	: maxVL( maxVL_ ),
	  dbpl( std::max( 1, (DegreeBits+maxVL-1)/maxVL ) ),
	  vmask( (fVID(1)<<(sizeof(fVID)*8-dbpl))-1 ),
	  dmax( fVID(1) << (dbpl*maxVL) ),
	  overflows( 0 ),
	  size( 0 ) { }

    void record( fVID seq, EID pos, fVID value, fVID abs_value, fVID deg ) {
	fVID r = value & vmask;
	if( VID( r ) != value )
	    ++overflows;
	size += sizeof(fVID);
    }

    void invalid( fVID seq, EID pos, fVID deg ) {
	size += sizeof(fVID);
    }

    bool success() const { return overflows == 0; }
    EID nOverflows() const { return overflows; }
    EID nBytes() const { return size; }

public:
    const unsigned short maxVL, dbpl;
    const fVID vmask, dmax;
    EID overflows, size;
};


static VID determineDeltaDegree( VID deg ) {
    if( deg > 2 && ( (deg-1) % 3 ) != 0 )
	deg += 3 - ( (deg-1) % 3 );
    return deg;
}

template<typename fVID, typename hVID, unsigned short DegreeBits>
struct EncDelta {
    EncDelta( unsigned short maxVL_ )
	: maxVL( maxVL_ ),
	  dbpl( std::max( 1, (DegreeBits+maxVL-1)/maxVL ) ),
	  vmask( (fVID(1)<<(sizeof(fVID)*8-dbpl))-1 ),
	  dmax( fVID(1) << (dbpl*maxVL-1) ), // loose 1 bit end of run indicator
	  overflows( 0 ) { }

    void alignSize() { }

    void set_degree( char *where, VID deg, bool last_for_v ) {
	// deg has been divided by maxVL
	VID *w = reinterpret_cast<VID *>(where);
	VID enc = deg > (dmax - 1) ? (dmax - 1) : (deg - 1);
	enc <<= 1;
	if( last_for_v )
	    enc |= 1;
	for( unsigned short l=0; l < maxVL; ++l ) {
	    VID d = enc >> ( l * dbpl );
	    d &= ( VID(1) << dbpl ) - 1;
	    w[l] |= d << ( sizeof(VID) * 8 - dbpl );
	}
    }
    
    void record_short( char * &where, hVID delta ) {
	*reinterpret_cast<hVID*>(where) = delta;
	where += sizeof(hVID);
    }
    void record_wide( char * &where, fVID delta ) {
	*reinterpret_cast<fVID*>(where) = delta;
	where += sizeof(fVID);
    }

    bool success() const { return overflows == 0; }
    EID nOverflows() const { return overflows; }

public:
    const unsigned short maxVL, dbpl;
    const fVID vmask, dmax;
    EID overflows;
};

template<typename fVID, typename hVID, unsigned short DegreeBits>
struct EncDeltaf {
    EncDeltaf( unsigned short maxVL_ )
	: maxVL( maxVL_ ),
	  dbpl( std::max( 1, (DegreeBits+maxVL-1)/maxVL ) ),
	  vmask( (fVID(1)<<(sizeof(fVID)*8-dbpl))-1 ),
	  dmax( fVID(1) << (dbpl*maxVL-1) ), // loose 1 bit end of run indicator
	  overflows( 0 ) { }

    void alignSize() { }

    void set_degree( char *where, VID deg ) {
	// deg has been divided by maxVL
	VID *w = reinterpret_cast<VID *>(where);
	VID enc = deg > (dmax - 1) ? (dmax - 1) : (deg - 1);
	// enc <<= 1; // space for the full-width bit
	enc = ( (enc & ~1) << 1 ) | ( enc & 1 );
	for( unsigned short l=0; l < maxVL; ++l ) {
	    VID d = enc >> ( l * dbpl );
	    d &= ( VID(1) << dbpl ) - 1;
	    w[l] |= d << ( sizeof(VID) * 8 - dbpl );
	}
    }

    void set_fullwidth( char *where, bool wfull ) {
	// wfull here indicates whether the SIMD word we are updating
	// is at full width. We always set a 1 bit
	if( wfull ) {
	    fVID *w = reinterpret_cast<fVID *>(where);
	    fVID d = 1;
	    // *w |= d << ( sizeof(fVID) * 8 - dbpl );
	    *w |= d << ( sizeof(fVID) * 8 - 1 );
	} else {
	    hVID *w = reinterpret_cast<hVID *>(where);
	    hVID d = 1;
	    // *w |= d << ( sizeof(hVID) * 8 - 1 );
	    w[1] |= d << ( sizeof(hVID) * 8 - 1 );
	}
    }
    void set_overshoot( char *where ) {
	// wfull here indicates whether the SIMD word we are updating
	// is at full width. We always set a 1 bit. This 1 bit must be
	// recoverable regardless of whether we read the first half word
	// of a full-width word, its second half word, or any random
	// half word.
	// If full-word:
	// + if bit is set, then second half-word will be read
	// + if bit is not set, then first half-word will be read
	// + if random half-word is read, bit won't be set.

	// Set the highest bit to 1 in lane 0. This bit is unused
	// in the half-words and will always be zero. It will normally
	// always be zero in full words that are not the first in
	// a degree-group.
	fVID *w = reinterpret_cast<fVID *>(where);
	fVID d = 1;
	*w |= d << ( sizeof(fVID) * 8 - 1 );
    }
    
    void record_short( char * &where, hVID delta ) {
	*reinterpret_cast<hVID*>(where) = delta;
	where += sizeof(hVID);
    }
    void record_wide( char * &where, fVID delta ) {
	*reinterpret_cast<fVID*>(where) = delta;
	where += sizeof(fVID);
    }

    bool success() const { return overflows == 0; }
    EID nOverflows() const { return overflows; }

public:
    const unsigned short maxVL, dbpl;
    const fVID vmask, dmax;
    EID overflows;
};

template<typename fVID, typename hVID, unsigned short DegreeBits>
struct EncDstPar {
    EncDstPar( unsigned short maxVL_, VID * edges_ )
	: maxVL( maxVL_ ),
	  dbpl( std::max( 1, (DegreeBits+maxVL-1)/maxVL ) ),
	  vmask( (fVID(1)<<(sizeof(fVID)*8-dbpl))-1 ),
	  dmax( fVID(1) << (dbpl*maxVL-1) ), // loose 1 bit end of run indicator
	  overflows( 0 ),
	  edges( edges_ ) { }

    void alignSize() { }

    void record( fVID seq, EID pos, fVID value, fVID abs_value, fVID deg ) {
	fVID r = value;
	assert( deg > 0 );
	if( deg == 1 || deg == 2 ) {
	    edges[pos] = abs_value;
	    return;
	}
	if( (seq % (3*(dmax-1))) == 0 ) {
	    fVID d = 0;
	    VID lane = pos % maxVL;
	    if( deg - seq > 3*(dmax-1) ) {
		d = ( dmax - 1 ) >> ( lane * dbpl );
		d &= ( fVID(1) << dbpl ) - 1;
	    } else {
		// Rescale degree from [1..3*dmax] to [0..dmax-1]
		VID deg3 = (deg - seq - 1) / 3;
		d = deg3 >> (lane * dbpl);
		d &= ( fVID(1) << dbpl ) - 1;
	    }
	    r = value & vmask;
	    r |= d << ( sizeof(fVID) * 8 - dbpl );
	}

	edges[pos] = r;
    }
    void invalid( VID seq, EID pos, VID deg ) {
	record( seq, pos, ~VID(0), ~VID(0), deg );
    }

    bool success() const { return overflows == 0; }
    EID nOverflows() const { return overflows; }

public:
    const unsigned short maxVL, dbpl;
    const fVID vmask, dmax;
    EID overflows;
    fVID * const edges;
};

} // namespace detail

/**
 * Assuming VEBO was applied, consider delta-encoding of the degree/index
 * Starting points should be recoverable only at the start of
 * partitions, i.e., index only needed for first vertex, remainder
 * can be delta-degree
 */
class GraphCSxSIMDDegreeDeltaMixed {
    using hVID = typename int_type_of_size<sizeof(VID)/2>::type;

    VID n;	//!< Number of vertices
    VID nv;	//!< Number of SIMD groupds of vertices (n rounded up to maxVL)
    EID m;	//!< Number of edges
    EID mv;	/**< Number of SIMD groups of edges worth sizeof(VID)*VL bytes
		   The invariant mv == mvd1 + mvdpar + mv1 + mv2 should hold */
    EID mvd1;	/**< Number of SIMD groups of edges worth sizeof(VID)*VL bytes
		   All edges in a SIMD group share the same destination. This
		   encoding is beneficial for high-degree vertices */
    EID mvdpar;	/**< Number of SIMD groups encoded such that each vector lane
		   holds sources for a subsequent destination. The degree is
		   encoded in the SIMD group and is quantized. Every SIMD group
		   has size sizeof(VID) * maxVL. This encoding is most
		   beneficial for lower-degree vertices. It is not applied for
		   SIMD groups where all vertices have degree 2 or less. */
    EID mv1;	//!< Number of SIMD groups as in mvdpar but with degree 2.
    EID mv2;	//!< Number of SIMD groups as in mvdpar but with degree 1.
    EID nbytes, dpar_off;
    VID vslim;
    unsigned short maxVL;
    mmap_ptr<VID> edges;

    static constexpr unsigned short DegreeBits = 12; // 16;

public:
    GraphCSxSIMDDegreeDeltaMixed() { }
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
	// Vertices
	n = Gcsc.numVertices();
	nv = ( ( hi - lo + maxVL - 1 ) / maxVL ) * maxVL;
	assert( nv >= (hi-lo) && nv < (hi-lo) + maxVL );

	// Total edges
	m = 0;
	for( nv=lo; nv < hi; nv++ ) {
	    VID r = remap.first[nv];
	    m += idx[r+1] - idx[r];
	}

	// Cut-off point for d1 vs dpar
	const VID threshold = maxVL;

	// Figure out cut-off vertex
	mv = mvd1 = mvdpar = mv1 = mv2 = 0;
	vslim = lo+nv;
	for( VID v=lo; v < hi; v++ ) {
	    // Traverse vertices in order of decreasing degree
	    VID r = remap.first[v];
	    VID deg = idx[r+1] - idx[r];
	    if( deg < threshold ) { // threshold - evaluate
		vslim = std::max( lo, v - ( v % maxVL ) );
		break;
	    }
	}

	// Count number of required edges to store graph
	{
	    detail::EncDeltaSize<VID,hVID,DegreeBits> size_d1( maxVL );
	    detail::EncDstParSize<VID,DegreeBits> size_dpar( maxVL );
	    fmt_d1_dpar( Gcsc, lo, hi, vslim, maxVL,
			 remap, nullptr, nullptr, size_d1, size_dpar );
	    dpar_off = size_d1.nBytes() / sizeof(VID);
	    nbytes = size_d1.nBytes() + size_dpar.nBytes();
	    mvd1 = size_d1.nBytes() / sizeof(VID);
	    assert( mvd1 * sizeof(VID) == size_d1.nBytes() );
	    mvdpar = size_dpar.nBytes() / sizeof(VID);
	    assert( mvdpar * sizeof(VID) == size_dpar.nBytes() );
	    mv = mvd1 + mvdpar;
	}

	// TODO: traverse backwards
	for( VID v=vslim; v < hi; v += maxVL ) {
	    VID r = remap.first[v];
	    VID deg = idx[r+1] - idx[r];
	    if( deg == 2 )
		mv2 += 2 * maxVL;
	    else if( deg == 1 )
		mv1 += maxVL;
	}
	mvdpar -= mv1 + mv2;
	assert( mv1 <= mv );
	assert( mv2 <= mv );

	// Allocate data structures
	if( allocation == -1 )
	    allocateInterleaved();
	else
	    allocateLocal( allocation );

	// Now that we know that the degree bits are available, encode degrees.
/*
	{
	    detail::EncodeDeltaMixed<VID,DegreeBits> encoder( maxVL, edges.get() );
	    import_phase( Gcsc, lo, hi, vslim, threshold, maxVL, mv, remap, edges.get(), encoder );
	    assert( encoder.success() );
	}
*/
	{
	    detail::EncDelta<VID,hVID,DegreeBits> enc_d1( maxVL );
	    detail::EncDeltaf<VID,hVID,DegreeBits> enc_d1f( maxVL );
	    detail::EncDstPar<VID,hVID,DegreeBits> enc_dpar( maxVL, &edges[dpar_off] );
	    fmt_d1_dpar( Gcsc, lo, hi, vslim, maxVL,
			 remap, edges.get(), &edges[dpar_off], enc_d1f, enc_dpar );
	}
    }

private:
    template<typename Functor1, typename FunctorD>
    void fmt_d1_dpar( const GraphCSx & Gcsc,
		      VID lo, VID hi, VID vslim,
		      unsigned short maxVL,
		      std::pair<const VID *, const VID *> remap,
		      VID * edges_d1, VID * edges_dpar,
		      Functor1 & fnc_d1, FunctorD & fnc_dpar ) {
	// d1_delta( Gcsc, lo, vslim, maxVL, remap, edges_d1, fnc_d1 );
	d1_deltaf( Gcsc, lo, vslim, maxVL, remap, edges_d1, fnc_d1 );
	assert( fnc_d1.success() );

	dst_par( Gcsc, vslim, hi, maxVL, remap, edges_dpar, fnc_dpar );
	assert( fnc_dpar.success() );
    }
    
    // Import vectorized data where all SIMD lanes have same destination
    // Degrees up to a maximum size are stored in the first SIMD word of
    // a degree group, allowing to jump over a large number of vectors when
    // the destination vertex is inactive.
    // Applies delta-compression where delta values are stored at half width.
    // If delta values do not fit in half-width words, then new degree groups
    // are started with may result in many small degree groups.
    template<typename Functor>
    static EID d1_delta( const GraphCSx & Gcsc,
			 VID lo, VID hi,
			 unsigned short maxVL,
			 std::pair<const VID *, const VID *> remap,
			 VID * edges,
			 Functor & fnc ) {
	// Calculate dimensions of SIMD representation
	VID n = Gcsc.numVertices();
	const EID * idx = Gcsc.getIndex();
	const VID * edg = Gcsc.getEdges();
	    
	// In w = remap.first[v], v is the new vertex ID, w is the old one
	// In v = remap.second[w], v is the new vertex ID, w is the old one

	VID maxdeg = idx[remap.first[0]+1] - idx[remap.first[0]];
	VID * buf = new VID[maxdeg];

	// hVID mask is applied to store delta's. Highest bit is left unused
	// to minimise computation needed to restore 0xff...ff bitmask.
	// Current solution is that highest bit is 1 only if bit pattern is
	// 0xffff, and conversion can be done by sign extension.
	VID mask = (VID(1)<<(sizeof(hVID)*8-2))-1;
	hVID hmask = (hVID(1)<<(sizeof(hVID)*8))-1;

	// All vector lanes map to same destination
	EID nxt = 0;
	char * where = reinterpret_cast<char *>(edges);
	char * curw = 0;
	for( VID v=lo; v < hi; v++ ) {
	    VID r = remap.first[v];
	    VID ldeg = idx[r+1] - idx[r];
	    VID deg = ldeg + maxVL - ( ((ldeg-1) % maxVL) + 1 );

	    if( ldeg == 0 ) {
		break;
	    }

	    for( VID j=0; j < ldeg; ++j )
		buf[j] = remap.second[edg[idx[r]+j]];
	    std::sort( &buf[0], &buf[ldeg] );

	    // Proceed in groups of vector width
	    VID curdeg = 0;
	    for( VID j=0; j < deg; j += maxVL ) {
		// Determine if this vector should be full width
		bool full_width = ( ((j-curdeg)/maxVL) % (fnc.dmax-1) ) == 0;
		if( !full_width ) {
		    for( unsigned short l=0; l < std::min(ldeg-j,VID(maxVL)); ++l ) {
			VID val = j+l < ldeg ? buf[j+l] : ~(VID)0;
			VID delta = j >= maxVL ? val - buf[j+l-maxVL] - maxVL : val;
			if( ( delta & mask ) != delta ) {
			    full_width = true;
			    break;
			}
		    }
		}

		if( full_width ) {
		    // Write degree into previous full width vector
		    if( j != curdeg )
			fnc.set_degree( curw, (j - curdeg) / maxVL, false );
		    curdeg = j;
		    curw = where;
		    
		    for( unsigned short l=0; l < maxVL; ++l ) {
			VID val = j+l < ldeg ? buf[j+l] : ( ~VID(0) >> 2 );
			VID delta = j + l < ldeg && j >= maxVL
			    ? val - buf[j+l-maxVL] - maxVL : val;
			fnc.record_wide( where, delta );
		    }
		} else {
		    for( unsigned short l=0; l < maxVL; ++l ) {
			VID val = j+l < ldeg ? buf[j+l] : ( ~VID(0) >> 2 );
			hVID delta = j+l < ldeg
					   ? hVID(val - buf[j+l-maxVL] - maxVL)
					   : hmask;
			fnc.record_short( where, delta & hmask );
		    }
		}
	    }
	    fnc.set_degree( curw, (deg - curdeg) / maxVL, true );

	    nxt += deg;
	}

	delete[] buf;

	// Ensure vectors in dpar slice are aligned
	fnc.alignSize();
	
	return nxt; // for debugging
    }

    // Import vectorized data where all SIMD lanes have same destination
    // Degrees up to a maximum size are stored in the first SIMD word of
    // a degree group, allowing to jump over a large number of vectors when
    // the destination vertex is inactive.
    // Applies delta-compression where delta values are stored at half width.
    // Each SIMD word contains one bit to indicate whether the next SIMD word
    // is at full width or half width. The first word in a degree group stores
    // absolute values and so is stored at full width.
    template<typename Functor>
    static EID d1_deltaf( const GraphCSx & Gcsc,
			  VID lo, VID hi,
			  unsigned short maxVL,
			  std::pair<const VID *, const VID *> remap,
			  VID * edges,
			  Functor & fnc ) {
	// Calculate dimensions of SIMD representation
	VID n = Gcsc.numVertices();
	const EID * idx = Gcsc.getIndex();
	const VID * edg = Gcsc.getEdges();
	    
	// In w = remap.first[v], v is the new vertex ID, w is the old one
	// In v = remap.second[w], v is the new vertex ID, w is the old one

	VID maxdeg = idx[remap.first[0]+1] - idx[remap.first[0]];
	VID * buf = new VID[maxdeg];

	// hVID mask is applied to store delta's. Highest bit is left unused
	// to minimise computation needed to restore 0xff...ff bitmask.
	// Current solution is that highest bit is 1 only if bit pattern is
	// 0xffff, and conversion can be done by sign extension.
	VID mask = (VID(1)<<(sizeof(hVID)*8-2))-1;
	hVID hmask = (hVID(1)<<(sizeof(hVID)*8))-1;

	// All vector lanes map to same destination
	EID nxt = 0;
	char * where = reinterpret_cast<char *>(edges);
	char * curw = 0, * prevw = 0;
	bool prevf = true;
	for( VID v=lo; v < hi; v++ ) {
	    VID r = remap.first[v];
	    VID ldeg = idx[r+1] - idx[r];
	    VID deg = ldeg + maxVL - ( ((ldeg-1) % maxVL) + 1 );

	    if( ldeg == 0 ) {
		break;
	    }

	    for( VID j=0; j < ldeg; ++j )
		buf[j] = remap.second[edg[idx[r]+j]];
	    std::sort( &buf[0], &buf[ldeg] );

	    // Proceed in groups of vector width
	    VID curdeg = 0;
	    VID nwords = fnc.dmax-1;
	    for( VID j=0; j < deg; j += maxVL ) {
		// Determine if this vector should be full width
		// bool new_group = ( ((j-curdeg)/maxVL) % (fnc.dmax-1) ) == 0;
		bool new_group = ( nwords >= (fnc.dmax-1) );
		bool full_width = new_group;
		if( !full_width ) {
		    for( unsigned short l=0; l < std::min(ldeg-j,VID(maxVL)); ++l ) {
			VID val = j+l < ldeg ? buf[j+l] : ~(VID)0;
			VID delta = j >= maxVL ? val - buf[j+l-maxVL] - maxVL : val;
			if( ( delta & mask ) != delta ) {
			    full_width = true;
			    break;
			}
		    }
		}

		char * ww = where;
		if( full_width ) {
		    if( new_group ) {
			// Write degree into previous full width vector
			if( j != curdeg )
			    fnc.set_degree( curw, nwords );
			curdeg = j;
			curw = where;
			nwords = 0;
		    }
		    nwords += 2;
		    
		    for( unsigned short l=0; l < maxVL; ++l ) {
			VID val = j+l < ldeg ? buf[j+l] : ( ~VID(0) >> 2 );
			VID delta = j + l < ldeg && j >= maxVL
			    ? val - buf[j+l-maxVL] - maxVL : val;
			fnc.record_wide( where, delta );
		    }
		} else {
		    for( unsigned short l=0; l < maxVL; ++l ) {
			VID val = j+l < ldeg ? buf[j+l] : ( ~VID(0) >> 2 );
			hVID delta = j+l < ldeg
					   ? hVID(val - buf[j+l-maxVL] - maxVL)
					   : hmask;
			fnc.record_short( where, delta & hmask );
		    }
		    nwords += 1;
		}
		if( full_width && j > 0 )
		    fnc.set_fullwidth( prevw, prevf );
		prevw = ww;
		prevf = full_width;

// need to check that this bit is in the right position and can always be
//     retrieved correctly, regardless of whether we read the first half word,
//     or the last half word of the final full-width word
// make this a different function - set_overshoot to allow setting the bit
//     in a different location, without affecting the decoding of degree
// because this word will not have a degree encoded
		if( nwords > fnc.dmax-1 && prevf && j > 0 )
		    fnc.set_overshoot( prevw );
	    }
	    fnc.set_degree( curw, nwords );

	    // Set bit in final word of degree group to indicate if it is
	    // a half-word or a full-word straggling across the dmax-1
	    // boundary.
	    if( nwords > fnc.dmax-1 && prevf )
		fnc.set_overshoot( prevw );

	    nxt += deg;
	}

	delete[] buf;

	// Ensure vectors in dpar slice are aligned
	fnc.alignSize();
	
	return nxt; // for debugging
    }

    template<typename Functor>
    static EID dst_par( const GraphCSx & Gcsc,
			VID lo, VID hi,
			unsigned short maxVL,
			std::pair<const VID *, const VID *> remap,
			VID * edges,
			Functor & fnc ) {
	// Calculate dimensions of SIMD representation
	VID n = Gcsc.numVertices();
	const EID * idx = Gcsc.getIndex();
	const VID * edg = Gcsc.getEdges();
	VID nv = maxVL * ( ( hi - lo + maxVL - 1 ) / maxVL );
	assert( nv >= (hi-lo) && nv < (hi-lo) + maxVL );
	    
	// In w = remap.first[v], v is the new vertex ID, w is the old one
	// In v = remap.second[w], v is the new vertex ID, w is the old one

	VID maxdeg = idx[remap.first[lo]+1] - idx[remap.first[lo]];
	VID * buf = new VID[maxdeg];

	// All vector lanes map to same destination
	EID nxt = 0;
	char * where = reinterpret_cast<char *>(edges);

	// Each vector lane is a subsequent destination
	for( VID v=lo; v < lo+nv; v += maxVL ) {
	    VID r = remap.first[v];
	    VID deg = idx[r+1] - idx[r];

	    if( deg == 0 ) {
		break;
	    }
	    // Degree 1, 2 are not degree-encoded
	    if( deg > 2 )
		deg = detail::determineDeltaDegree( deg );
	    
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
			assert( j == 0 || buf[j] > buf[j-1] );
			if( j > 0 )
			    fnc.record( j, lnxt, buf[j] - buf[j-1] - 1, buf[j], deg );
			else
			    fnc.record( j, lnxt, buf[j], buf[j], deg );
			lnxt += maxVL;
		    } 
		    for( VID j=ldeg; j < deg; ++j ) {
			fnc.invalid( j, lnxt, deg );
			lnxt += maxVL;
		    }
		} else {
		    VID ww = ~(VID)0;
		    for( VID j=0; j < deg; ++j ) {
			fnc.invalid( j, lnxt, deg );
			lnxt += maxVL;
		    } 
		}
	    }
	    nxt += maxVL * deg;
	}
	delete[] buf;
	return nxt;
    }

    template<typename Functor>
    static void import_phase( const GraphCSx & Gcsc,
			      VID lo, VID hi, VID vslim, VID threshold,
			      unsigned short maxVL,
			      EID mv,
			      std::pair<const VID *, const VID *> remap,
			      VID * edges,
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

	VID mask = (VID(1)<<(sizeof(VID)*8/2-1))-1;

	// All vector lanes map to same destination
	EID nxt = 0;
	char * where = reinterpret_cast<char *>(edges);
	for( VID v=lo; v < vslim; v++ ) {
	    VID r = remap.first[v];
	    VID ldeg = idx[r+1] - idx[r];
	    VID deg = ldeg + maxVL - ( ((ldeg-1) % maxVL) + 1 );
	    assert( ldeg >= threshold );

	    if( ldeg == 0 ) {
		assert( nxt == mv );
		break;
	    }

	    for( VID j=0; j < ldeg; ++j )
		buf[j] = remap.second[edg[idx[r]+j]];
	    std::sort( &buf[0], &buf[ldeg] );

	    // Proceed in groups of vector width
	    char * curw = 0;
	    VID curdeg = 0;
	    for( VID j=0; j < deg; j += maxVL ) {
		// Determine if this vector should be full width
		bool full_width = ( ((j-curdeg)/maxVL) % (fnc.dmax-1) ) == 0;
		if( !full_width ) {
		    for( unsigned short l=0; l < std::min(ldeg-j,VID(maxVL)); ++l ) {
			VID val = j+l < ldeg ? buf[j+l] : ~(VID)0;
			VID delta = j >= maxVL ? val - buf[j+l-maxVL] - maxVL : val;
			if( ( delta & mask ) != delta ) {
			    full_width = true;
			    break;
			}
		    }
		}

		if( full_width ) {
		    // if( v < 5 )
		    // std::cerr << "full width at " << curdeg << " where " << (void*)where << " curw " << (void *)curw << " for v=" << v << " deg " << (j-curdeg) << "\n";
		    // Write degree into previous full width vector
		    if( j != curdeg )
			fnc.set_degree( curw, (j - curdeg) / maxVL, false );
		    curdeg = j;
		    curw = where;
		    
		    for( unsigned short l=0; l < maxVL; ++l ) {
			VID val = j+l < ldeg ? buf[j+l] : ( ~VID(0) >> 2 );
			VID delta = j + l < ldeg && j >= maxVL ? val - buf[j+l-maxVL] - maxVL : val;
			fnc.record_wide( where, delta );
		    }
		} else {
		    for( unsigned short l=0; l < maxVL; ++l ) {
			VID val = j+l < ldeg ? buf[j+l] : ( ~VID(0) >> 2 );
			VID delta = j+l < ldeg ? val - buf[j+l-maxVL] - maxVL : ( ~VID(0) >> 2 );
			fnc.record_short( where, delta & mask );
		    }
		}
	    }
	    // if( v < 5 )
	    // std::cerr << "full width at " << curdeg << " where " << (void*)where << " curw " << (void *)curw << " for v=" << v << " deg " << (deg-curdeg) << " final\n";
	    fnc.set_degree( curw, (deg - curdeg) / maxVL, true );

	    nxt += deg;
	}

	// Each vector lane is a subsequent destination
	for( VID v=vslim; v < lo+nv; v += maxVL ) {
	    VID r = remap.first[v];
	    VID deg = idx[r+1] - idx[r];

	    if( deg == 0 ) {
		assert( nxt == mv );
		break;
	    }
	    // Degree 1, 2 are not degree-encoded
	    if( deg > 2 ) {
		// In this case use SlimSell style
		deg = detail::determineDeltaDegree( deg );
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
			assert( j == 0 || buf[j] > buf[j-1] );
			// edges[lnxt] = buf[j];
			if( j > 0 )
			    fnc.record( j, lnxt, buf[j] - buf[j-1] - 1, buf[j], deg );
			else
			    fnc.record( j, lnxt, buf[j], buf[j], deg );
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

    VID getSlimVertex() const { return vslim; }

    VID numSIMDVertices() const { return nv; }
    EID numSIMDEdges() const { return mv; }
    EID numSIMDEdgesDelta1() const { return mvd1; }
    EID numSIMDEdgesDeltaPar() const { return mvdpar; }
    EID numSIMDEdgesDeg1() const { return mv1; }
    EID numSIMDEdgesDeg2() const { return mv2; }

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
	EID esize = sizeof(VID) * maxVL;
	assert( nbytes % esize == 0 );
	edges.allocate( nbytes/sizeof(VID), esize, numa_allocation_interleaved() );
    }
    void allocateLocal( int numa_node ) {
	EID esize = sizeof(VID) * maxVL;
	assert( nbytes % esize == 0 );
	edges.allocate( nbytes/sizeof(VID), esize, numa_allocation_local( numa_node ) );
    }
};


#endif // GRAPTOR_GRAPH_CGRAPHCSXSIMDDEGREEDELTAMIXED_H
