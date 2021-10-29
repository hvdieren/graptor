// -*- c++ -*-
#ifndef GRAPTOR_GRAPH_CGRAPHCSRSIMDDEGREEMIXED_H
#define GRAPTOR_GRAPH_CGRAPHCSRSIMDDEGREEMIXED_H

#include <cstdlib>

#include "graptor/mm.h"
#include "graptor/graph/gtraits.h"
#include "graptor/graph/GraphCSx.h"

#define GRAPTOR_WITH_REDUCE 1
#define GRAPTOR_WITH_SELL 2
#define GRAPTOR_MIXED 3

#ifndef GRAPTOR_THRESHOLD_MULTIPLIER
#define GRAPTOR_THRESHOLD_MULTIPLIER 1
#endif // GRAPTOR_THRESHOLD_MULTIPLIER

#ifndef GRAPTOR_DEGREE_BITS
#define GRAPTOR_DEGREE_BITS 16
#endif // GRAPTOR_DEGREE_BITS

#ifndef GRAPTOR_SKIP_BITS
#if GRAPTOR_CSR_INDIR
#define GRAPTOR_SKIP_BITS 0
#else
#define GRAPTOR_SKIP_BITS 0
#endif
#endif // GRAPTOR_SKIP_BITS

#ifndef GRAPTOR_DEGREE_MULTIPLIER
#define GRAPTOR_DEGREE_MULTIPLIER 3
#endif // GRAPTOR_DEGREE_MULTIPLIER

// Problem: DEG12 does not work easily as common targets will cause allocation
//          of additional vectors, causing the degree to decrease
//          non-monotonically
#ifndef GRAPTOR_DEG12
#define GRAPTOR_DEG12 0
#endif // GRAPTOR_DEG12

template<typename fVID>
struct SortByDecrDegStable {
    SortByDecrDegStable( fVID * degp_ ) : degp( degp_ ) { }
    bool operator () ( fVID l, fVID r ) const {
	return degp[l] == degp[r] ? l < r : degp[l] > degp[r];
    }
private:
    fVID * degp;
};


// GraptorVReduce is a special case of this with nelm == G.n
// Also necessary to change the direction of the argument graph
template<typename fVID, typename fEID, unsigned short DegreeBits>
class GraptorVPush {
public:
    GraptorVPush( unsigned short maxVL_ )
	: maxVL( maxVL_ ),
	  dbpl( BitsPerLane<DegreeBits>( maxVL ) ),
	  vmask( (fVID(1)<<(sizeof(fVID)*8-dbpl))-1 ),
	  dmax( fVID(1) << (dbpl*maxVL) ),
	  overflows( 0 ), size( 0 ), ninv( 0 ), edges( nullptr ) { }

    template<bool AnalyseMode, typename Remapper>
    fEID process( const GraphCSx & Gcsr,
		  fVID lo, fVID hi, fVID nelm,
		  unsigned int p,
		  Remapper remap,
		  fVID * edges_ ) {
	// Record edges for auxiliary function
	edges = edges_;

	// Process relevant edges and either calculate storage requirement
	// or construct graph representation.
	fVID n = nelm;
	fVID norig = Gcsr.numVertices();
	assert( norig <= n );
	const fEID * idx = Gcsr.getIndex();
	const fVID * edg = Gcsr.getEdges();
	    
	fVID maxdeg = gtraits_getmaxoutdegree<GraphCSx>(Gcsr).getMaxOutDegree();
	fVID * buf = new fVID[maxdeg]; // pessimistically sized

	// All vector lanes map to same destination
	fEID nxt = 0;
	for( fVID v=0; v < n; v++ ) { // ERROR - missing sources
	    // for( fVID v=0; v < n; v++ ) { // TODO: introduce skip values
	    fVID r = remap.origID(v);
	    fVID mldeg = r < norig ? idx[r+1] - idx[r] : 0;

	    fVID ldeg = 0;
	    for( fVID j=0; j < mldeg; ++j ) {
		fVID w = remap.remapID( edg[idx[r]+j] );
		if( lo <= w && w < hi )
		    buf[ldeg++] = w;
	    }
	    std::sort( &buf[0], &buf[ldeg] );
	    assert( ldeg <= maxdeg );

	    fVID deg = ldeg + maxVL - ( ((ldeg-1) % maxVL) + 1 );
	    if( deg == 0 ) // record a vector with all lanes disabled for consistency
		deg = maxVL;

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

    template<bool AnalyseMode>
    fEID process( const GraphCSx & Gcsr,
		  const partitioner & part,
		  unsigned int p,
#if GRAPTOR_CSR_INDIR == 1
		  fVID * redir,
		  fVID redir_nnz,
#endif
		  fVID * edges_,
		  fEID * starts_ ) {
	// Record edges for auxiliary function
	edges = edges_;
	// starts = starts_;

	// Process relevant edges and either calculate storage requirement
	// or construct graph representation.
	const fVID nelm = part.get_num_elements();
	const fVID n = nelm;
	const fVID norig = Gcsr.numVertices();
	const fVID lo = part.start_of( p );
	const fVID hi = part.end_of( p );
	assert( norig <= n );
	const fEID * idx = Gcsr.getIndex();
	const fVID * edg = Gcsr.getEdges();
	    
	// TODO: This can be optimised such that the values are immediately
	//       written to the final data structure instead of writing to buf.
	//       A second pass then updates only the degree bits. The second
	//       pass is necessary as we don't know the degree at the start.
	//       However, we do know there is sufficient space in the buffer.
	fVID maxdeg = gtraits_getmaxoutdegree<GraphCSx>(Gcsr).getMaxOutDegree();
	fVID * buf = new fVID[maxdeg]; // pessimistically sized

#if GRAPTOR_CSR_INDIR == 1
	fVID vmax = redir_nnz;
#else
	fVID vmax = n;
#endif

	// All vector lanes map to same destination
	fEID nxt = 0;
	for( fVID v=0; v < vmax; v++ ) {
	    // for( fVID v=0; v < n; v++ ) { // TODO: introduce skip values
#if GRAPTOR_CSR_INDIR == 1
	    if( redir[v] == n )
		break;
	    fVID r = redir[v];
#else
	    fVID r = v;
#endif
	    fVID mldeg = idx[r+1] - idx[r];

	    fVID ldeg = 0;
	    for( fVID j=0; j < mldeg; ++j ) {
		fVID w = edg[idx[r]+j];
		if( lo <= w && w < hi )
		    buf[ldeg++] = w;
	    }
	    // std::sort( &buf[0], &buf[ldeg] ); -- redundant
	    assert( ldeg <= maxdeg );

	    fVID deg = ldeg + maxVL - ( ((ldeg-1) % maxVL) + 1 );
#if GRAPTOR_CSR_INDIR == 0
	    if( deg == 0 ) // record a vector with all lanes disabled for consistency
		deg = maxVL;
#else
	    if( ldeg == 0 ) // skip zero-degree vertices
		continue;
#endif

	    // Proceed in groups of vector width
	    for( fVID j=0; j < deg; j += maxVL ) {
		for( unsigned short l=0; l < maxVL; ++l ) {
		    if( j+l < ldeg )
			record<AnalyseMode>( nxt + j + l, j, l, buf[j+l], deg );
		    else {
			record<AnalyseMode>( nxt + j + l, j, l, vmask, deg );
			++ninv;
		    }
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
#if GRAPTOR_CACHED == 1
	    // Only record degree information in first vector of a run
	    if( (seq % (GRAPTOR_DEGREE_MULTIPLIER*(dmax-1))) == 0 ) {
		fVID deg3 = (deg/maxVL - seq/maxVL - 1) / GRAPTOR_DEGREE_MULTIPLIER;
		if( deg3 > (dmax/2-1) )
		    deg3 = dmax/2-1;
		fVID d = deg3 >> (lane * dbpl);
		d &= ( fVID(1) << dbpl ) - 1;

		value &= vmask;
		value |= d << ( sizeof(fVID) * 8 - dbpl );
	    }
	    edges[pos] = value;
#else
	    fVID deg3 = (deg/maxVL - seq/maxVL - 1) / GRAPTOR_DEGREE_MULTIPLIER;
	    if( deg3 > (dmax-1) )
		deg3 = dmax-1;
	    fVID d = deg3 >> (lane * dbpl);
	    d &= ( fVID(1) << dbpl ) - 1;

	    value &= vmask;
	    value |= d << ( sizeof(fVID) * 8 - dbpl );
	    edges[pos] = value;
#endif
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

template<typename fVID, typename fEID, unsigned short DegreeBits>
class GraptorCSRDataPar {
public:
    GraptorCSRDataPar( unsigned short maxVL_ )
	: maxVL( maxVL_ ),
	  dbpl( BitsPerLane<DegreeBits>( maxVL ) ),
	  vmask( (fVID(1)<<(sizeof(fVID)*8-dbpl))-1 ),
	  dmax( fVID(1) << (dbpl*maxVL) ),
	  overflows( 0 ),
	  size( 0 ), ninv( 0 ), edges( nullptr ) { }

    template<bool AnalyseMode, typename Remapper>
    fEID process( const GraphCSx & Gcsc,
		  unsigned int p,
		  fVID vslim, fVID nelm,
		  Remapper remap,
		  fVID * edges_,
		  fEID * starts_ ,
		  const fVID * redir, fVID redir_nnz,
		  const partitioner & part ) {
	// Calculate dimensions of SIMD representation
	// Record edges for auxiliary function
	edges = edges_;
	starts = starts_;

	// Process relevant edges and either calculate storage requirement
	// or construct graph representation.
	fVID n = nelm;
	fVID norig = Gcsc.numVertices();
	fVID lo = part.start_of( p );
	fVID hi = part.end_of( p );
	const fEID * idx = Gcsc.getIndex();
	const fVID * edg = Gcsc.getEdges();
	fVID nv = maxVL * ( ( hi - lo + maxVL - 1 ) / maxVL );
	assert( nv >= (hi-lo) && nv < (hi-lo) + maxVL );
	    
	fVID maxdeg = gtraits_getmaxoutdegree<GraphCSx>(Gcsc).getMaxOutDegree();
	fVID * buf = new fVID[maxdeg*maxVL*maxVL]; // worst-case storage
	fVID * lbuf = new fVID[maxVL];

	// All vector lanes map to same destination
	fEID nxt = 0;

	unsigned int npart = part.get_num_partitions();

	fVID vmax = redir_nnz;
	if( vmax % maxVL ) // roundup
	    vmax += maxVL - ( vmax % maxVL );

	// Each vector lane is a subsequent destination
	for( fVID vi=0; vi < vmax; vi += maxVL ) {
	    const fVID * va = &redir[vi];
	    if constexpr ( !AnalyseMode )
		starts[/*(vi-lo)*/vi/maxVL] = nxt;
	    
	    // Calculate the maximum degree required to store this batch
	    // of edges originating from the next maxVL vertices and pointing
	    // to the current partition (range lo - hi)
	    fVID deg = 0;
	    // std::fill( &pcount[0], &pcount[npart], (EID)0 );
	    for( unsigned short l=0; l < maxVL; ++l ) { // vector lane
		fVID r = vi+l < redir_nnz ? remap.origID(va[l]) : norig;
		fVID mdeg = r < norig ? idx[r+1] - idx[r] : 0;
		lbuf[l] = 0;
		fEID lpos = fEID(maxdeg) * fEID(maxVL) * fEID(l);
		for( fVID j=0; j < mdeg; ++j ) {
		    fVID w = remap.remapID(edg[idx[r]+j]);
		    if( lo <= w && w < hi ) {
			buf[lpos+lbuf[l]] = w;
			++lbuf[l];
		    }
		}
		assert( lbuf[l] <= maxdeg );
		if( deg < lbuf[l] )
		    deg = lbuf[l];

		std::sort( &buf[lpos], &buf[lpos+lbuf[l]] );
	    }

	    assert( deg <= maxdeg*maxVL ); // TODO: this may be violated! implications on buffer sizes?

	    if( deg == 0 )
		deg = 1;

	    // std::cerr << "va[0]=" << va[0] << " deg=" << deg << " nxt=" << nxt << "\n";
	    
	    // The degree encoding may round up, take that into account
	    // and fill 1 or 2 extra slots also with invalid vertices
	    for( unsigned short l=0; l < maxVL; ++l ) { // vector lane
		fEID lpos = fEID(maxdeg) * fEID(maxVL) * fEID(l);
		std::fill( &buf[lpos+lbuf[l]], &buf[lpos+deg], ~VID(0));
		lbuf[l] = deg;
	    }

	    for( unsigned short l=0; l < maxVL; ++l ) { // vector lane
		fVID vv = va[l];
		fEID lnxt = nxt + l;
		if( vv < n ) {
		    fEID lpos = EID(maxdeg) * EID(maxVL) * EID(l);
		    fVID ninv = 0;
		    for( fVID j=0; j < deg; ++j ) {
			assert( lnxt-nxt < deg * maxVL );

			// std::cerr << "filling slot j=" << j << " l=" << l << "\n";
			bool found = false;
			for( fVID k=lpos+j; k < lpos+lbuf[l]; ++k ) {
			    bool allowed = true;
			    if( buf[k] != ~VID(0) ) {
				for( unsigned short ll=0; ll < l; ++ll ) {
				    fEID llpos = fEID(maxdeg) * fEID(maxVL) * fEID(ll);
				    if( buf[llpos+j] != ~VID(0)
					&& buf[llpos+j] == buf[k] ) {
					allowed = false;
					break;
				    }
				}
			    }
			    if( allowed ) {
				found = true;
				fVID tmp = buf[lpos+j];
				buf[lpos+j] = buf[k];
				buf[k] = tmp;
				// std::cerr << "        found k=" << k
				// << " take=" << take << "\n";
				break;
			    }
			}
			if( !found ) {
			    // std::cerr << " bump up deg to " << deg
			    // << " for " << buf[lpos+j] << "\n";
			    for( int i=0; i < GRAPTOR_DEGREE_MULTIPLIER; ++i ) {
				for( unsigned short ll=0; ll < maxVL; ++ll ) {
				    fEID llpos = EID(maxdeg) * EID(maxVL) * EID(ll);
				    fEID llnxt = nxt + ll + lbuf[ll] * maxVL;
				    if( ll < l )
					// fnc.invalid( deg, llnxt, deg+1 );
					record<AnalyseMode>( deg, llnxt, vmask, deg+1 );
				    buf[llpos+lbuf[ll]] = ~VID(0);
				    ++lbuf[ll];
				}
				if( i == 0 ) {
				    buf[lpos+lbuf[l]-1] = buf[lpos+j];
				    buf[lpos+j] = ~VID(0);
				}
				++deg;
				assert( deg <= maxdeg * maxVL );
			    }
			}

			if( buf[lpos+j] != ~(fVID)0 ) {
			    // fnc.record( j, lnxt, buf[lpos+j], deg );
			    record<AnalyseMode>( j, lnxt, buf[lpos+j], deg );
			} else {
			    // fnc.invalid( j, lnxt, deg );
			    record<AnalyseMode>( j, lnxt, vmask, deg );
			    ++ninv;
			    assert( lbuf[l] <= deg );
			}
			lnxt += maxVL;
		    }
		} else {
		    fVID ww = ~(fVID)0;
		    for( fVID j=0; j < deg; ++j ) {
			// fnc.invalid( j, lnxt, deg );
			record<AnalyseMode>( j, lnxt, vmask, deg );
			lnxt += maxVL;
		    } 
		}
	    }
	    fVID next_vi = vi + maxVL; // find_next_v( v, p, maxVL, Gcsc, part, n, remap );
	    if( edges && false ) {
		for( unsigned short l=0; l < maxVL; ++l ) { // vector lane
		    fVID r = remap.origID(va[l]);
		    fVID mdeg = r < norig ? idx[r+1] - idx[r] : 0;
		    for( fVID j=0; j < mdeg; ++j ) {
			fVID w = remap.remapID(edg[idx[r]+j]);
			if( lo <= w && w < hi ) {
			    bool found = false;
			    for( fVID k=0; k < deg; ++k ) {
				if( edges[nxt+l+k*maxVL] == w ) {
				    found = true;
				    break;
				}
			    }
			    assert( found && "checking correctness" );
			}
		    }
		}
	    }

	    // Round-up degree to next required multiple
	    {
		// Degree 1, 2 are not degree-encoded
		fVID ndeg = determineDegree( deg );
	
		for( int i=0; i < (ndeg - deg); ++i ) {
		    for( unsigned short ll=0; ll < maxVL; ++ll ) {
			fEID llnxt = nxt + ll + (lbuf[ll] + i) * maxVL;
			// fnc.invalid( deg, llnxt, deg+1 );
			record<AnalyseMode>( deg, llnxt, vmask, deg+1 );
		    }
		}

		deg = ndeg;
	    }

	    if( edges ) {
		// std::cerr << "Set degree=" << deg << "\n";
		for( fVID j=0; j < deg; ++j ) {
		    for( unsigned short l=0; l < maxVL; ++l ) // vector lane
			set_degreep<AnalyseMode>( j, nxt + l + j * maxVL, deg );
		}
	    }
	    nxt += maxVL * deg;
	}

	delete[] buf;
	delete[] lbuf;
	return nxt;
    }

    bool success() const { return overflows == 0; }
    fEID nOverflows() const { return overflows; }
    fEID nBytes() const { return size; }
    fEID nInvalid() const { return ninv; }

public:
    template<typename Remapper>
    std::pair<mmap_ptr<VID>,VID>
    reorder( const GraphCSx & Gcsc,
	     unsigned int p,
	     unsigned short roundup,
	     fVID nelm,
	     Remapper remap,
	     const partitioner & part,
	     int allocation ) {
	fVID n = nelm;
	fVID norig = Gcsc.numVertices();
	fVID lo = part.start_of( p );
	fVID hi = part.end_of( p );
	const fEID * idx = Gcsc.getIndex();
	const fVID * edg = Gcsc.getEdges();

	// In w = remap.first[v], v is the new vertex ID, w is the old one
	// In v = remap.second[w], v is the new vertex ID, w is the old one

	fVID * degp = new fVID[n];
	fVID * imap = new fVID[n];

	parallel_for( fVID v=0; v < n; ++v ) {
	    // Calculate the degree of each vertex within this partition
	    fVID r = remap.origID( v );
	    fVID mdeg = r < norig ? idx[r+1] - idx[r] : 0;
	    fVID deg = 0;
	    for( fVID j=0; j < mdeg; ++j ) {
		fVID w = remap.remapID( edg[idx[r]+j] );
		if( lo <= w && w < hi )
		    ++deg;
	    }
	    degp[v] = deg;

	    // Construct a remapping array
	    imap[v] = v;
	}
	// TODO: Can we be more intelligent here? Look at common destinations?
	//       Or uncommon destinations to avoid padding?
	//       Sort lexicographically for tied degrees?
	std::sort( &imap[0], &imap[n], SortByDecrDegStable<fVID>( degp ) );

	// Calculate number of remapped vertices (iteration index range for
	// edgemap; inuse for the remapped partitioner)
	fVID nnz = 0;
	for( fVID v=0; v < n; ++v ) {
	    if( degp[v] > 0 )
		++nnz;
	}
	delete[] degp;

	// Make sure that there is backing space for a vector load
	fVID nnz_alc = nnz;
	if( nnz_alc % roundup )
	    nnz_alc += roundup - ( nnz_alc % roundup );

	// Trim the allocation
	// TODO: use mmap_ptr and can we extend ifc to trim using munmap?
	mmap_ptr<VID> rmap;
	if( allocation == -1 )
	    rmap.allocate( nnz_alc, numa_allocation_interleaved() );
	else
	    rmap.allocate( nnz_alc, numa_allocation_local( allocation ) );

	parallel_for( fVID v=0; v < nnz; ++v )
	    rmap[v] = imap[v];
	for( fVID v=nnz; v < nnz_alc; ++v ) // low numbers
	    rmap[v] = n;
	
	delete[] imap;

	return std::make_pair( rmap, nnz );
    }

private:
    template<bool AnalyseMode>
    void record( fVID seq, fEID pos, fVID value, fVID deg ) {
	if( AnalyseMode ) {
	    size += sizeof(fVID);
	    if( (value & vmask) == vmask )
		ninv++;
	    if( value & ~vmask )
		overflows++;
	} else {
	    edges[pos] = value;
	}
    }

    template<bool AnalyseMode>
    void set_degreep( fVID seq, fEID pos, fVID deg ) {
	assert( deg > 0 );
#if GRAPTOR_DEG12
	if( deg == 1 || deg == 2 ) {
	    return;
	}
#endif
#if GRAPTOR_CACHED == 1
	    // Only record degree information in first vector of a word
	if( (seq % (GRAPTOR_DEGREE_MULTIPLIER*(dmax/2))) != 0 )
	    return;
#endif
	    
	fVID r = edges[pos];

	fVID lane = pos % maxVL;
	fVID deg3 = (deg - seq - 1) / GRAPTOR_DEGREE_MULTIPLIER;
	if( deg3 > (dmax/2-1) )
	    deg3 = (dmax/2-1) << 1;
	else
	    deg3 = (deg3 << 1) | 1;
	fVID d = deg3 >> (lane * dbpl);
	d &= ( fVID(1) << dbpl ) - 1;

	r &= vmask;
	r |= d << ( sizeof(fVID) * 8 - dbpl );

	edges[pos] = r;
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
};

/**
 * Assuming VEBO was applied, consider delta-encoding of the degree/index
 * Starting points should be recoverable only at the start of
 * partitions, i.e., index only needed for first vertex, remainder
 * can be delta-degree
 */
class GraphCSRSIMDDegreeMixed {
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
#if GRAPTOR_CSR_INDIR == 1
    mmap_ptr<VID> redir; //!< Map implicit indices to vertices
    VID redir_nnz; //!< Number of elements in redirection array
#endif
    unsigned short maxVL;
    mmap_ptr<VID> edges;
    mmap_ptr<EID> starts;

    static constexpr unsigned short DegreeBits = GRAPTOR_DEGREE_BITS;
    static constexpr unsigned short SkipBits = GRAPTOR_SKIP_BITS;
    static constexpr unsigned short DegreeSkipBits = DegreeBits + SkipBits;

public:
    GraphCSRSIMDDegreeMixed() { }
    void del() {
	edges.del();
#if GRAPTOR_CSR_INDIR == 1
	redir.del();
#endif
	starts.del();
    }
    template<typename Remapper>
    void import( const GraphCSx & Gcsc,
		 const partitioner & part,
		 unsigned int p,
		 unsigned short maxVL_,
		 // std::pair<const VID *, const VID *> remap,
		 Remapper remap,
		 int allocation ) {
	maxVL = maxVL_;

#if GRAPTOR_EXTRACT_OPT
	assert( maxVL == GRAPTOR_DEGREE_BITS && "restriction" );
#endif // GRAPTOR_CSC

	VID lo = part.start_of(p);
	VID hi = part.end_of(p);
	VID nelm = part.get_num_elements();

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
	// TODO - only works because graph is symmetric
	//        FIX is probably to work from CSR rather than from CSC
	//        as we are interpreting the CSC as if it were a CSR
	// for( VID v=lo; v < hi; v++ ) {
	for( VID v=lo; v < hi; v++ ) {
	    // VID r = remap.first[v];
	    VID r = remap.origID( v );
	    if( r < n )
		m += idx[r+1] - idx[r];
	}
	assert( m <= Gcsc.numEdges() );
#if (GRAPTOR & GRAPTOR_WITH_SELL) == 0
#if GRAPTOR_DEG12 == 1
#error "GRAPTOR_DEG12 requires SELL"
#endif
#endif

	// Figure out cut-off vertex
	mv = mvd1 = mvdpar = mv1 = mv2 = 0;
#if (GRAPTOR & GRAPTOR_WITH_SELL) == 0
	vslim = hi;
	for( VID v=hi; v > lo; v-- ) {
	    // VID r = remap.first[v-1]; // loop index shifted by -1
	    VID r = remap.origID( v-1 ); // loop index shifted by -1
	    VID deg = idx[r+1] - idx[r];
	    if( deg > 0 ) {
		vslim = v; // upper bound, deg[vslim] == 0
		break;
	    }
	}
#elif (GRAPTOR & GRAPTOR_WITH_REDUCE) == 0
	vslim = lo;
#else
	// Cut-off point for d1 vs dpar
	const VID threshold = GRAPTOR_THRESHOLD_MULTIPLIER * maxVL;

	vslim = lo+nv;
	for( VID v=lo; v < hi; v++ ) { // TODO: take steps of maxVL
	    // Traverse vertices in order of decreasing degree
	    // VID r = remap.first[v];
	    VID r = remap.origID( v );
	    VID deg = idx[r+1] - idx[r];
	    if( deg < threshold ) { // threshold - evaluate
		vslim = std::max( lo, v - ( v % maxVL ) );
		//std::cerr << "set vslim C=" << vslim << "\n";
		break;
	    }
	}
#endif // GRAPTOR_BY_VERTEX

	// Round up vslim to next multiple of maxVL
	if( vslim % maxVL != 0 )
	    vslim += maxVL - ( vslim % maxVL );

#if GRAPTOR_CSR_INDIR == 1
	{
	    GraptorCSRDataPar<VID,EID,DegreeBits> enc_dpar( maxVL );
	    std::pair<mmap_ptr<VID>, VID> pair
		= enc_dpar.reorder( Gcsc, p, maxVL, nelm, remap, part, allocation );
	    redir = pair.first;
	    redir_nnz = pair.second;

	    // std::cerr << "p=" << p << " nnz=" << redir_nnz << "\n";
	}
#endif

	// Count number of required edges to store graph
	{
	    GraptorVPush<VID,EID,DegreeBits> size_d1( maxVL );
	    GraptorCSRDataPar<VID,EID,DegreeBits> size_dpar( maxVL );
	    fmt_d1_dpar<true>( Gcsc, p, lo, hi, nelm, vslim, maxVL,
			       remap, nullptr, nullptr, size_d1, size_dpar,
			       starts.get(), &starts[(vslim-lo)/maxVL],
#if GRAPTOR_CSR_INDIR == 1
			       redir, redir_nnz,
#endif
			       part );
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
	    // VID r = remap.first[v-maxVL];
	    VID r = remap.origID( v-maxVL );
	    VID deg = idx[r+1] - idx[r];
	    if( deg == 2 )
		mv2 += 2 * maxVL;
	    else if( deg == 1 )
		mv1 += maxVL;
	    else if( deg > 2 ) {
#if (GRAPTOR & GRAPTOR_WITH_REDUCE) != 0 && (GRAPTOR & GRAPTOR_WITH_SELL) == 0
		vslim = v; // adjust vslim
		// std::cerr << "adjust vslim=" << vslim << "\n";
#endif
		break;
	    }
	}
#if (GRAPTOR & GRAPTOR_WITH_SELL) != 0
	mvdpar -= mv1 + mv2;
#endif
	assert( mv1 <= mv );
	assert( mv2 <= mv );
#endif

#if (GRAPTOR & GRAPTOR_WITH_SELL) == 0
	assert( mvdpar == 0 );
#endif

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
	if( allocation == -1 )
	    allocateInterleaved();
	else
	    allocateLocal( allocation );

	// Now that we know that the degree bits are available, encode degrees.
	{
	    GraptorVPush<VID,EID,DegreeBits> enc_d1( maxVL );
	    GraptorCSRDataPar<VID,EID,DegreeBits> enc_dpar( maxVL );
	    fmt_d1_dpar<false>( Gcsc, p, lo, hi, nelm, vslim, maxVL,
				remap, edges.get(), &edges[mvd1], enc_d1, enc_dpar,
				starts.get(), &starts[(vslim-lo)/maxVL],
#if GRAPTOR_CSR_INDIR == 1
				redir, redir_nnz,
#endif
				part );
	}
    }

#if GRAPTOR_CSC == 0 && GRAPTOR == 1
    // VPush only. SIMD edge groups/partition and NNZ/partition
    // These statistics are correct also in case of GRAPTOR_CSR_INDIR
    static std::pair<const EID *,const VID *>
    space_from_csr( const GraphCSx & Gcsr,
		    const partitioner & part,
		    unsigned short maxVL ) {
	// Calculate the number of edges, counted by full SIMD groups,
	// required to store each partition of the graph.
	const unsigned int np = part.get_num_partitions();
	const VID n = part.get_num_elements();

	EID * count = new EID[np];
	std::fill( &count[0], &count[np], EID(0) );

	VID * nnz = new VID[np];
	std::fill( &nnz[0], &nnz[np], VID(0) );

	const EID * idx = Gcsr.getIndex();
	const VID * edg = Gcsr.getEdges();

	parallel_for( VID v=0; v < n; ++v ) {
	    VID deg = idx[v+1] - idx[v];

#if GRAPTOR_CSR_INDIR == 0
	    if( deg <= maxVL ) { // special case, easily predicted impact
		// one word per zero-degree vertex
		for( unsigned int p=0; p < np; ++p )
		    __sync_fetch_and_add( &count[p], 1 );
	    }
#else
	    if( deg == 0 ) { // special case, nothing to do
	    } else if( deg == 1 ) { // special case
		// only one partition takes vertex
		unsigned int p = part.part_of( edg[idx[v]] );
		__sync_fetch_and_add( &count[p], 1 );
		__sync_fetch_and_add( &nnz[p], 1 );
	    } else if( deg == 2 ) { // special case
		unsigned int p0 = part.part_of( edg[idx[v]] );
		unsigned int p1 = part.part_of( edg[idx[v]+1] );
		if( p0 == p1 ) { // only one partition takes vertex
		    __sync_fetch_and_add( &count[p0], 1 );
		    __sync_fetch_and_add( &nnz[p0], 1 );
		} else { // each partition contains vertex
		    __sync_fetch_and_add( &count[p0], 1 );
		    __sync_fetch_and_add( &nnz[p0], 1 );
		    __sync_fetch_and_add( &count[p1], 1 );
		    __sync_fetch_and_add( &nnz[p1], 1 );
		}
	    }
#endif
	    else {
		VID * vcnt = new VID[np]; // each entry limited by degree, fits VID
		std::fill( &vcnt[0], &vcnt[np], VID(0) );
		for( VID d=0; d < deg; ++d ) {
		    unsigned int p = part.part_of( edg[idx[v]+d] );
		    vcnt[p]++;
		}
		for( unsigned int p=0; p < np; ++p ) {
		    VID c = vcnt[p];
#if GRAPTOR_CSR_INDIR == 1
		    if( c > 0 )
			__sync_fetch_and_add( &nnz[p], 1 );
#else
		    if( c == 0 ) // at least one SIMD word/vertex
			c = 1;
#endif
		    c = ( c + maxVL - 1 ) / maxVL;
		    __sync_fetch_and_add( &count[p], c );
		}
		delete[] vcnt;
	    }
	}

#if GRAPTOR_CSR_INDIR == 0
	for( unsigned int p=0; p < np; ++p ) {
	    nnz[p] = n;
	    assert( count[p] >= n ); // at least one SIMD word/vertex
	}
#endif

	for( unsigned int p=0; p < np; ++p ) {
	    if( nnz[p] % maxVL != 0 )
		nnz[p] += maxVL - ( nnz[p] % maxVL );
	}

	return std::make_pair( count, nnz );
    }

    template<typename Allocation>
    void import_csr( const GraphCSx & Gcsr,
		     const partitioner & part,
		     unsigned int p,
		     unsigned short maxVL_,
		     EID mv_, // pre-calculated
		     VID redir_nnz_, // pre-calculated
		     Allocation allocation ) {
	maxVL = maxVL_;

	VID lo = part.start_of(p);
	VID hi = part.end_of(p);
	VID nelm = part.get_num_elements();

	assert( lo % maxVL == 0 );
	assert( hi % maxVL == 0 || p == part.get_num_partitions()-1 );

	const EID * idx = Gcsr.getIndex();
	const VID * edg = Gcsr.getEdges();

	// Calculate dimensions of SIMD representation
	// Vertices
	n = Gcsr.numVertices();
	nv = ( ( n + maxVL - 1 ) / maxVL ) * maxVL;
	assert( nv >= n && nv < n + maxVL );

	// Total edges
	m = 0;
	// TODO - only works because graph is symmetric
	//        FIX is probably to work from CSC rather than from CSR
	//        as we are interpreting the CSR as if it were a CSC
	for( VID v=lo; v < hi; v++ ) {
	    VID r = v;
	    if( r < n )
		m += idx[r+1] - idx[r];
	}
	assert( m <= Gcsr.numEdges() );
#if (GRAPTOR & GRAPTOR_WITH_SELL) == 0
#if GRAPTOR_DEG12 == 1
#error "GRAPTOR_DEG12 requires SELL"
#endif
#endif

	// Figure out cut-off vertex
	// mv is pre-calculated
	mv = mvd1 = mv_ * maxVL;
	mvdpar = mv1 = mv2 = 0;

	// TODO: not required ...?
	vslim = n;
	// Round up vslim to next multiple of maxVL
	if( vslim % maxVL != 0 )
	    vslim += maxVL - ( vslim % maxVL );

	// We pre-calculated the sizes:
	nbytes = sizeof(VID) * mv;
#if GRAPTOR_CSR_INDIR == 1
	redir_nnz = redir_nnz_;
#endif
	allocate( allocation );

#if GRAPTOR_CSR_INDIR == 1
	reorder( Gcsr, part, p, nelm );
#endif

	// std::cerr << "mv=" << mv
		  // << "\nvslim=" << vslim
		  // << "\nmv1=" << mv1
		  // << "\nmv2=" << mv2
		  // << "\nmvd1=" << mvd1
		  // << "\nmvdpar=" << mvdpar
		  // << "\nninvd1=" << ninvd1
		  // << "\nninvdpar=" << ninvdpar
		  // << "\n";

	// Encode...
	{
	    GraptorVPush<VID,EID,DegreeBits> enc_d1( maxVL );
	    EID ne = enc_d1.template process<false>(
		Gcsr, part, p,
#if GRAPTOR_CSR_INDIR == 1
		redir.get(), redir_nnz,
#endif
		edges, starts );
	    assert( enc_d1.success() );
	    assert( ne == mvd1 );
/*
	    GraptorCSRDataPar<VID,EID,DegreeBits> enc_dpar( maxVL );
	    fmt_d1_dpar<false>( Gcsr, p, lo, hi, nelm, vslim, maxVL,
				remap, edges.get(), &edges[mvd1], enc_d1, enc_dpar,
#if GRAPTOR_CSR_INDIR == 1
				redir, redir_nnz,
#endif
				part );
*/

	    ninvd1 = enc_d1.nInvalid();
	    ninvdpar = 0;

	}
    }

#if GRAPTOR_CSR_INDIR == 1
    void reorder( const GraphCSx & Gcsr,
		  const partitioner & part,
		  unsigned int p,
		  VID nelm ) {
	VID n = nelm;
	VID norig = Gcsr.numVertices();
	VID lo = part.start_of( p );
	VID hi = part.end_of( p );
	const EID * idx = Gcsr.getIndex();
	const VID * edg = Gcsr.getEdges();

	// In w = remap.first[v], v is the new vertex ID, w is the old one
	// In v = remap.second[w], v is the new vertex ID, w is the old one

	VID * degp = new VID[n+1];
	VID * imap = redir.get();

	VID ins = 0;
	for( VID v=0; v < n; ++v ) {
	    // Calculate the degree of each vertex within this partition
	    VID r = v;
	    VID mdeg = idx[r+1] - idx[r];
	    VID deg = 0;
	    for( VID j=0; j < mdeg; ++j ) {
		VID w = edg[idx[r]+j];
		if( lo <= w && w < hi )
		    ++deg;
	    }
	    degp[v] = deg;

	    // Construct a remapping array
	    if( deg > 0 ) {
		imap[ins++] = v;
		assert( ins <= redir_nnz );
	    }
	}
	// Redir_nnz is rounded up up to multiple of maxVL
	assert( ins <= redir_nnz  && ins + maxVL > redir_nnz );
	std::sort( &imap[0], &imap[ins], SortByDecrDegStable<VID>( degp ) );

	for( VID i=ins; i < redir_nnz; ++i )
	    imap[i] = n;

	// Cleanup
	delete[] degp;
    }
#endif
#endif


private:
    template<bool EstimateMode, typename Functor1, typename FunctorD,
	     typename Remapper>
    void fmt_d1_dpar( const GraphCSx & Gcsc,
		      unsigned int p,
		      VID lo, VID hi, VID nelm, VID vslim,
		      unsigned short maxVL,
		      Remapper remap,
		      VID * edges_d1, VID * edges_dpar,
		      Functor1 & fnc_d1, FunctorD & fnc_dpar,
		      EID * starts_d1, EID * starts_dpar,
#if GRAPTOR_CSR_INDIR == 1
		      const VID * redir, VID redir_nnz,
#endif
		      const partitioner & part ) {
	// TODO: lo-vslim is not the range of source; sources
	// are 0-vslim/vslim-n and destinations are defined by partition
	// start/end
#if (GRAPTOR & GRAPTOR_WITH_REDUCE) != 0
	fnc_d1.template process<EstimateMode>( Gcsc, lo, vslim, nelm, p, remap, edges_d1, starts_d1 );
	assert( fnc_d1.success() );
#endif

#if (GRAPTOR & GRAPTOR_WITH_SELL) != 0
#if GRAPTOR_CSR_INDIR != 1 && GRAPTOR_CSC == 0
#error "GRAPTOR_CSR_INDIR currently required"
#elif GRAPTOR_CSR_INDIR == 1
	fnc_dpar.template process<EstimateMode>(
	    Gcsc, p, vslim, nelm, remap, edges_dpar, starts_dpar,
	    redir, redir_nnz, part );
	assert( fnc_dpar.success() );
#endif
#endif
    }
    
public:
    VID *getEdges() { return edges.get(); }
    const VID *getEdges() const { return edges.get(); }
    const EID *getStarts() const { return starts.get(); }

    VID numVertices() const { return n; }
    EID numEdges() const { return m; }

    VID getSlimVertex() const { return vslim; }

#if GRAPTOR_CSR_INDIR == 1
    const VID *getRedirP() const { return redir.get(); }
    VID getRedirNNZ() const { return redir_nnz; }
#endif

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
    unsigned short getSkipBits() const { return (SkipBits+maxVL-1)/maxVL; }

    unsigned short getDegreeSkipBits() const { return getDegreeBits() + getSkipBits(); }
    unsigned short getDegreeSkipShift() const { return sizeof(VID)*8 - getDegreeSkipBits(); }

private:
    template<typename AllocationType>
    void allocate( AllocationType alc ) {
	EID esize = sizeof(VID) * maxVL;
	assert( nbytes % esize == 0 );
	edges.allocate( nbytes/sizeof(VID), esize, alc );
#if GRAPTOR_CSR_INDIR
	redir.allocate( redir_nnz, esize, alc );
#endif
    }
    
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


#endif // GRAPTOR_GRAPH_CGRAPHCSRSIMDDEGREEMIXED_H
