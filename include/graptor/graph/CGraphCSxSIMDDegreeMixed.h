// -*- c++ -*-
#ifndef GRAPTOR_GRAPH_CGRAPHCSXSIMDDEGREEMIXED_H
#define GRAPTOR_GRAPH_CGRAPHCSXSIMDDEGREEMIXED_H

#include "graptor/graph/GraptorDef.h"

namespace {

template<unsigned short Bits>
unsigned short BitsPerLane( unsigned short VL ) {
    return std::max( 1, (Bits+VL-1)/VL );
}

} // namespace anonymous

// This encoding assumes a COO/Grazelle-like main loop iterating over all
// vectors. Values are not cached and a reduction is performed once for
// every vector (once every VL edges). This is inefficient.
//
// We can also use this encoding for a CSR-like main loop iterating over
// destinations and a nested loop iterating over vectors w/ sources for
// that destination.
// Values are cached and a reduction is performed only once per destination.
template<graptor_mode_t Mode,
	 typename fVID, typename fEID, unsigned short DegreeBits>
class GraptorVReduce {
public:
    GraptorVReduce( unsigned short maxVL_ )
	: maxVL( maxVL_ ),
	  dbpl( BitsPerLane<DegreeBits>( maxVL ) ),
	  vmask( (fVID(1)<<(sizeof(fVID)*8-dbpl))-1 ),
	  dmax( fVID(1) << (dbpl*maxVL) ),
	  overflows( 0 ), size( 0 ), ninv( 0 ), edges( nullptr ) { }

    template<bool AnalyseMode, typename Remapper>
    fEID process( const GraphCSx & Gcsc,
		  fVID lo, fVID hi,
		  Remapper remap,
		  fVID * edges_,
		  fEID * starts_,
		  float * weights_ ) {
	// We haven't written the code yet to fill out the weights
	assert( !weights_ && "NYI" );
	
	// Record edges for auxiliary function
	edges = edges_;
	// assert( false && "Initialisation of starts for EIDRetriever NYI" );

	// Process relevant edges and either calculate storage requirement
	// or construct graph representation.
	fVID n = Gcsc.numVertices();
	const fEID * idx = Gcsc.getIndex();
	const fVID * edg = Gcsc.getEdges();
	    
	fVID maxdeg = gtraits_getmaxoutdegree<GraphCSx>(Gcsc).getMaxOutDegree();
	fVID * buf = new fVID[maxdeg];

	// All vector lanes map to same destination
	fEID nxt = 0;
	for( fVID v=lo; v < hi; v++ ) {
	    fVID r = remap.origID(v);
	    fVID ldeg = idx[r+1] - idx[r];
	    fVID deg = ldeg + maxVL - ( ((ldeg-1) % maxVL) + 1 );

	    // Assumes vertices are degree-sorted.
	    if( ldeg == 0 )
		break;

	    // Collect all remapped vertices and sort them by increasing ID
	    for( fVID j=0; j < ldeg; ++j )
		buf[j] = remap.remapID(edg[idx[r]+j]);
	    std::sort( &buf[0], &buf[ldeg] );

	    // Proceed in groups of vector width
	    for( fVID j=0; j < deg; j += maxVL ) {
		for( unsigned short l=0; l < maxVL; ++l ) {
		    fVID val = j+l < ldeg ? buf[j+l] : vmask;
		    record<AnalyseMode>( nxt + j + l, j, l, val, deg );
		}
	    }
	    nxt += deg;
	}

	delete[] buf;

	return nxt;
    }

    bool success() const { return overflows == 0; }
    fEID nOverflows() const { return overflows; }
    fEID nBytes() const { return size; }
    fEID nInvalid() const { return ninv; }

private:
    template<bool AnalyseMode>
    void record( fEID pos, fVID seq, fVID lane, fVID value, fVID deg ) {
	if( AnalyseMode ) {
	    size += sizeof(fVID);
	    if( (value & vmask) == vmask )
		ninv++;
	    if( value & ~vmask )
		overflows++;
	} else {
	    fVID d;
	    if constexpr ( GraptorConfig<Mode>::is_cached ) {
		// Only record degree information in first vector of a run
		if( (seq % (GRAPTOR_DEGREE_MULTIPLIER*(dmax-1))) != 0 ) {
		    edges[pos] = value;
		    return;
		}

		fVID deg3 = (deg/maxVL - seq/maxVL - 1) / GRAPTOR_DEGREE_MULTIPLIER;
		if( deg3 > (dmax-1) )
		    deg3 = dmax-1;
		d = deg3 >> (lane * dbpl);
		d &= ( fVID(1) << dbpl ) - 1;
	    } else {
		d = 0;
/*
	    if( seq + maxVL == deg ) {
		d = 1 >> ( lane * dbpl ); // TODO: introduce skip bits later
		d &= ( fVID(1) << dbpl ) - 1;
		} else */ {
		fVID rdeg = (deg/maxVL - seq/maxVL - 1) / GRAPTOR_DEGREE_MULTIPLIER;
		fVID deg3 = rdeg;
		if( deg3 > (dmax/2-1) )
		    deg3 = dmax/2-1;
		deg3 = ( deg3 << 1 ); // | 0;
		if( rdeg <= (dmax/2-1) )
		    deg3 |= 1;
		d = deg3 >> (lane * dbpl);
		d &= ( fVID(1) << dbpl ) - 1;
		}
	    }
	    
	    value &= vmask;
	    value |= d << ( sizeof(fVID) * 8 - dbpl );
	    edges[pos] = value;
	}
    }

private:
    const unsigned short maxVL, dbpl;
    const fVID vmask, dmax;
    fEID overflows;
    fEID size;
    fEID ninv;
    fVID * edges;
};

template<graptor_mode_t Mode,
	 typename fVID, typename fEID, unsigned short DegreeBits>
class GraptorDataPar {
public:
    GraptorDataPar( unsigned short maxVL_ )
	: maxVL( maxVL_ ),
	  dbpl( BitsPerLane<DegreeBits>( maxVL ) ),
	  vmask( (fVID(1)<<(sizeof(fVID)*8-dbpl))-1 ),
	  dmax( fVID(1) << (dbpl*maxVL) ),
	  overflows( 0 ),
	  size( 0 ), ninv( 0 ),
	  edges( nullptr ), starts( nullptr ), weights( nullptr ) { }

    template<bool AnalyseMode, typename Remapper>
    fEID process( const GraphCSx & Gcsc,
		  fVID lo, fVID hi,
		  Remapper remap,
		  VID * edges_,
		  EID * starts_,
		  float * weights_ ) {
	// Record edges for auxiliary function
	edges = edges_;
	starts = starts_;
	weights = weights_;

	// Process relevant edges and either calculate storage requirement
	// or construct graph representation.
	fVID n = Gcsc.numVertices();
	const fEID * idx = Gcsc.getIndex();
	const fVID * edg = Gcsc.getEdges();
	const float * Gw = Gcsc.getWeights() ? Gcsc.getWeights()->get() : nullptr;
	fVID nv = maxVL * ( ( hi - lo + maxVL - 1 ) / maxVL );
	assert( nv >= (hi-lo) && nv < (hi-lo) + maxVL );
	    
	fVID maxdeg = gtraits_getmaxoutdegree<GraphCSx>(Gcsc).getMaxOutDegree();
	fVID * buf = new fVID[maxdeg];
	float * wuf = weights ? new float[maxdeg] : nullptr;

	// All vector lanes map to same destination
	fEID nxt = 0;

	// Each vector lane is a subsequent destination
	for( fVID v=lo; v < lo+nv; v += maxVL ) {
	    fVID r = remap.origID(v);
	    fVID deg = idx[r+1] - idx[r];
	    if constexpr ( !AnalyseMode )
		starts[(v-lo)/maxVL] = nxt;

	    // Assumes vertices are degree-sorted
	    if( deg == 0 )
		break;
	    
	    // Degree 1, 2 are not degree-encoded
	    deg = determineDegree( deg );

	    // std::cerr << "enc: v=" << v << " deg=" << deg << "\n";
	    
	    for( unsigned short l=0; l < maxVL; ++l ) { // vector lane
		fVID vv = v + l;
		fEID lnxt = nxt + l;
		// Construct a list of sorted remapped sources
		fVID ww = remap.origID(vv);
		if( ww < n ) { // An original vertex?
		    fVID ldeg = ww < n ? idx[ww+1] - idx[ww] : 0;
		    assert( ldeg <= deg );
		    assert( ldeg <= maxdeg );
		    for( fVID j=0; j < ldeg; ++j ) {
			buf[j] = remap.remapID(edg[idx[ww]+j]);
			if( wuf )
			    wuf[j] = Gw[idx[ww]+j];
		    }
		    if( wuf )
			paired_sort( &buf[0], &buf[ldeg], &wuf[0] );
		    else
			std::sort( &buf[0], &buf[ldeg] );

		    for( fVID j=0; j < ldeg; ++j ) {
			assert( j == 0 || buf[j] > buf[j-1] );
			record<AnalyseMode>( j, lnxt, buf[j], v,
					     wuf ? wuf[j] : 0, deg );
			lnxt += maxVL;
		    } 
		    for( fVID j=ldeg; j < deg; ++j ) {
			record<AnalyseMode>( j, lnxt, vmask, v, 0, deg );
			lnxt += maxVL;
		    }
		} else {
		    fVID ww = ~(VID)0;
		    for( fVID j=0; j < deg; ++j ) {
			record<AnalyseMode>( j, lnxt, vmask, v, 0, deg );
			lnxt += maxVL;
		    } 
		}
	    }
	    nxt += maxVL * deg;
	}
	if( wuf )
	    delete[] wuf;
	delete[] buf;
	return nxt;
    }

    bool success() const { return overflows == 0; }
    fEID nOverflows() const { return overflows; }
    fEID nBytes() const { return size; }
    fEID nInvalid() const { return ninv; }

private:
    template<bool AnalyseMode>
    void record( fVID seq, fEID pos, fVID value, fVID dest, float w, fVID deg ) {
	if constexpr ( AnalyseMode ) {
	    size += sizeof(fVID);
	    if( (value & vmask) == vmask )
		ninv++;
	    if( value & ~vmask )
		overflows++;
	} else {
	    if constexpr ( GraptorConfig<Mode>::is_cached ) {
		// Only record degree information in first vector of a word
		if( (seq % (GRAPTOR_DEGREE_MULTIPLIER*(dmax/2-1))) != 0 ) {
		    edges[pos] = value;
		    if( weights )
			weights[pos] = w;
		    return;
		}
	    }

	    fVID d = 0;
	    fVID lane = pos % maxVL;
	    if( seq + 1 == deg ) {
#if GRAPTOR_STOP_BIT_HIGH
		fVID stop = dmax/2;
#else
		fVID stop = 1;
#endif
		d = stop >> ( lane * dbpl ); // TODO: introduce skip bits later
		d &= ( fVID(1) << dbpl ) - 1;
	    } else {
		fVID deg3 = (deg - seq - 1) / GRAPTOR_DEGREE_MULTIPLIER;
		if( deg3 > (dmax/2-1) )
		    deg3 = dmax/2-1;
#if GRAPTOR_STOP_BIT_HIGH
		// nothing
#else
		deg3 = ( deg3 << 1 ) | 0;
#endif
		d = deg3 >> (lane * dbpl);
		d &= ( fVID(1) << dbpl ) - 1;
	    }
	    value &= vmask;
	    value |= d << ( sizeof(fVID) * 8 - dbpl );
	    edges[pos] = value;
	    if( weights )
		weights[pos] = w;
	}
    }

    static VID determineDegree( VID deg ) {
#if GRAPTOR_DEG12
	if( deg > 2 && ( (deg-1) % GRAPTOR_DEGREE_MULTIPLIER ) != 0 )
	    deg += GRAPTOR_DEGREE_MULTIPLIER - ( (deg-1) % GRAPTOR_DEGREE_MULTIPLIER );
#else
	if( ( (deg-1) % GRAPTOR_DEGREE_MULTIPLIER ) != 0 )
	    deg += GRAPTOR_DEGREE_MULTIPLIER - ( (deg-1) % GRAPTOR_DEGREE_MULTIPLIER );
#endif
	return deg;
    }

public:
    const unsigned short maxVL, dbpl;
    const fVID vmask, dmax;
    fEID overflows, size, ninv;
    fVID * edges;
    fEID * starts;
    float * weights;
};

/**
 * Assuming VEBO was applied, consider delta-encoding of the degree/index
 * Starting points should be recoverable only at the start of
 * partitions, i.e., index only needed for first vertex, remainder
 * can be delta-degree
 */
template<graptor_mode_t Mode>
class GraphCSxSIMDDegreeMixed {
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
    EID ninvd1; //!< Number of unused SIMD lanes in mvd1
    EID ninvdpar; //!< Number of unused SIMD lanes in mvdpar
    EID nbytes;
    VID vslim;
    unsigned short maxVL;
    mmap_ptr<VID> edges;
    mmap_ptr<EID> starts;
    mm::buffer<float> * weights;

    static constexpr unsigned short DegreeBits = GRAPTOR_DEGREE_BITS;

public:
    GraphCSxSIMDDegreeMixed() {
    }
    void del() {
	edges.del();
	starts.del();
	if( weights )
	    weights->del();
    }
    template<typename Remapper>
    void import( const GraphCSx & Gcsc,
		 VID lo, VID hi, VID xxx_unused, unsigned short maxVL_,
		 Remapper remap,
		 int allocation ) {
	weights = nullptr;
	maxVL = maxVL_;

#if GRAPTOR_EXTRACT_OPT
	assert( maxVL == GRAPTOR_DEGREE_BITS && "restriction" );
#endif // GRAPTOR_EXTRACT_OPT

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
	for( VID v=lo; v < hi; v++ ) {
	    VID r = remap.origID(v);
	    m += idx[r+1] - idx[r];
	}
	if constexpr ( !GraptorConfig<Mode>::is_datapar ) {
	    // #if (GRAPTOR & GRAPTOR_WITH_SELL) == 0
	    #if GRAPTOR_DEG12 == 1
	    // #error "GRAPTOR_DEG12 requires SELL"
	    assert( true && "GRAPTOR_DEG12 requires SELL" );
	    #endif
	    // #endif
	}

	// Figure out cut-off vertex
	mv = mvd1 = mvdpar = mv1 = mv2 = 0;
	if constexpr ( !GraptorConfig<Mode>::is_datapar ) {
	    // #if (GRAPTOR & GRAPTOR_WITH_SELL) == 0
	    vslim = hi;
	    for( VID v=hi; v > lo; v-- ) {
		VID r = remap.origID(v-1); // loop index shifted by -1
		VID deg = idx[r+1] - idx[r];
		if( deg > 0 ) {
		    vslim = v; // upper bound, deg[vslim] == 0
		    break;
		}
	    }
	} else if constexpr ( GraptorConfig<Mode>::is_datapar ) {
	    // #elif (GRAPTOR & GRAPTOR_WITH_REDUCE) == 0
	    vslim = lo;
	} else {
	    // #else
	    // Cut-off point for d1 vs dpar
	    const VID threshold = GRAPTOR_THRESHOLD_MULTIPLIER * maxVL;

	    vslim = lo+nv;
	    for( VID v=lo; v < hi; v++ ) { // TODO: take steps of maxVL
		// Traverse vertices in order of decreasing degree
		VID r = remap.origID(v);
		VID deg = idx[r+1] - idx[r];
		if( deg < threshold ) { // threshold - evaluate
		    vslim = std::max( lo, v - ( v % maxVL ) );
		    //std::cerr << "set vslim C=" << vslim << "\n";
		    break;
		}
	    }
	    // #endif // GRAPTOR_BY_VERTEX
	}

	// Count number of required edges to store graph
	{
	    GraptorVReduce<Mode,VID,EID,DegreeBits> size_d1( maxVL );
	    GraptorDataPar<Mode,VID,EID,DegreeBits> size_dpar( maxVL );
	    fmt_d1_dpar<true>( Gcsc, lo, hi, vslim, maxVL,
			       remap, nullptr, nullptr, nullptr, nullptr,
			       nullptr, nullptr, size_d1, size_dpar );
	    nbytes = size_d1.nBytes() + size_dpar.nBytes();
	    mvd1 = size_d1.nBytes() / sizeof(VID);
	    assert( mvd1 * sizeof(VID) == size_d1.nBytes() );
	    mvdpar = size_dpar.nBytes() / sizeof(VID);
	    assert( mvdpar * sizeof(VID) == size_dpar.nBytes() );
	    mv = mvd1 + mvdpar;

	    ninvd1 = size_d1.nInvalid();
	    ninvdpar = size_dpar.nInvalid();
	}

#if GRAPTOR_DEG12
	// Traverse backwards for efficiency. Based on assumption that
	// vertices are degree-sorted.
	for( VID v=lo+nv; v > lo; v -= maxVL ) {
	    VID r = remap.origID(v-maxVL);
	    VID deg = idx[r+1] - idx[r];
	    if( deg == 2 )
		mv2 += 2 * maxVL;
	    else if( deg == 1 )
		mv1 += maxVL;
	    else if( deg > 2 ) {
		if constexpr ( !GraptorConfig<Mode>::is_datapar ) {
		    // #if (GRAPTOR & GRAPTOR_WITH_REDUCE) != 0 && (GRAPTOR & GRAPTOR_WITH_SELL) == 0
		    vslim = v; // adjust vslim
		    // std::cerr << "adjust vslim=" << vslim << "\n";
		    // #endif
		}
		break;
	    }
	}

	if constexpr ( GraptorConfig<Mode>::is_datapar ) {
	    // #if (GRAPTOR & GRAPTOR_WITH_SELL) != 0
	    mvdpar -= mv1 + mv2;
	    // #endif
	}
	assert( mv1 <= mv );
	assert( mv2 <= mv );
#endif

	if constexpr ( !GraptorConfig<Mode>::is_datapar ) {
	    // #if (GRAPTOR & GRAPTOR_WITH_SELL) == 0
	    assert( mvdpar == 0 );
	    // #endif
	}

	// std::cerr << "mv=" << mv
		  // << "\nvslim=" << vslim
		  // << "\nmv1=" << mv1
		  // << "\nmv2=" << mv2
		  // << "\nmvd1=" << mvd1
		  // << "\nmvdpar=" << mvdpar
		  // << "\nninvd1=" << ninvd1
		  // << "\nninvdpar=" << ninvdpar
		  // << "\n";

	// Allocate data structures
	if( allocation == -1 ) {
	    allocateInterleaved();
	    if( Gcsc.getWeights() != nullptr )
		weights = new mm::buffer<float>(
		    mv, numa_allocation_interleaved() );
	} else {
	    allocateLocal( allocation );
	    if( Gcsc.getWeights() != nullptr )
		weights = new mm::buffer<float>(
		    mv, numa_allocation_local( allocation ) );
	}

	// Now that we know that the degree bits are available, encode degrees.
	{
	    GraptorVReduce<Mode,VID,EID,DegreeBits> enc_d1( maxVL );
	    GraptorDataPar<Mode,VID,EID,DegreeBits> enc_dpar( maxVL );
	    
	    float * w_d1 = nullptr, * w_dpar = nullptr;
	    if( weights ) {
		w_d1 = weights->get();
		w_dpar = &w_d1[mvd1];
	    }

	    fmt_d1_dpar<false>( Gcsc, lo, hi, vslim, maxVL,
				remap, edges.get(), &edges[mvd1],
				starts.get(), &starts[(vslim-lo)/maxVL],
				w_d1, w_dpar,
				enc_d1, enc_dpar );
	}
    }

private:
    template<bool EstimateMode, typename Functor1, typename FunctorD,
	     typename Remapper>
    void fmt_d1_dpar( const GraphCSx & Gcsc,
		      VID lo, VID hi, VID vslim,
		      unsigned short maxVL,
		      Remapper remap,
		      VID * edges_d1, VID * edges_dpar,
		      EID * starts_d1, EID * starts_dpar,
		      float * w_d1, float * w_dpar,
		      Functor1 & fnc_d1, FunctorD & fnc_dpar ) {
// #if (GRAPTOR & GRAPTOR_WITH_REDUCE) != 0
	if constexpr ( !GraptorConfig<Mode>::is_datapar ) {
	// d1_value( Gcsc, lo, vslim, maxVL, remap, edges_d1, fnc_d1 );
	    fnc_d1.template process<EstimateMode>( Gcsc, lo, vslim, remap, edges_d1, starts_d1, w_d1 );
	    assert( fnc_d1.success() );
	}
// #endif

// #if (GRAPTOR & GRAPTOR_WITH_SELL) != 0
	if constexpr ( GraptorConfig<Mode>::is_datapar ) {
	    fnc_dpar.template process<EstimateMode>( 
		Gcsc, vslim, hi, remap, edges_dpar, starts_dpar, w_dpar );
	    assert( fnc_dpar.success() );
	}
// #endif
    }
    
public:
    VID *getEdges() { return edges.get(); }
    const VID *getEdges() const { return edges.get(); }
    const EID *getStarts() const { return starts.get(); }
    float * getWeights() { return weights ? weights->get() : nullptr; }
    const float * getWeights() const { return weights ? weights->get() : nullptr; }

    VID numVertices() const { return n; }
    EID numEdges() const { return m; }

    VID getSlimVertex() const { return vslim; }

    VID numSIMDVertices() const { return nv; }
    EID numSIMDEdges() const { return mv; }
    EID numSIMDEdgesDelta1() const { return mvd1; }
    EID numSIMDEdgesDeltaPar() const { return mvdpar; }
    EID numSIMDEdgesDeg1() const { return mv1; }
    EID numSIMDEdgesDeg2() const { return mv2; }

    EID numSIMDEdgesInvDelta1() const { return ninvd1; }
    EID numSIMDEdgesInvDeltaPar() const { return ninvdpar; }

    unsigned short getMaxVL() const { return maxVL; }
    unsigned short getDegreeBits() const { return (DegreeBits+maxVL-1)/maxVL; }
    unsigned short getDegreeShift() const { return sizeof(VID)*8 - getDegreeBits(); }

private:
    void allocateInterleaved() {
	EID esize = sizeof(VID) * maxVL;
	assert( nbytes % esize == 0 );
	edges.allocate( nbytes/sizeof(VID), esize, numa_allocation_interleaved() );
	starts.allocate( nv/maxVL, numa_allocation_interleaved() );
    }
    void allocateLocal( int numa_node ) {
	EID esize = sizeof(VID) * maxVL;
	assert( nbytes % esize == 0 );
	edges.allocate( nbytes/sizeof(VID), esize, numa_allocation_local( numa_node ) );
	starts.allocate( nv/maxVL, numa_allocation_local( numa_node ) );
    }
};


#endif // GRAPTOR_GRAPH_CGRAPHCSXSIMDDEGREEMIXED_H
