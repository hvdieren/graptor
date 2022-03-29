// -*- c++ -*-
#ifndef GRAPTOR_GRAPH_GRAPTORDATAPARPUSH_H
#define GRAPTOR_GRAPH_GRAPTORDATAPARPUSH_H

#include "graptor/partitioner.h"
#include "graptor/graph/GraphCSx.h"
#include "graptor/graph/GraptorDef.h"
#include "graptor/graph/CGraphCSxSIMDDegreeMixed.h"

/*======================================================================
 * An optimized version of the construction of the Graptor data-parallel
 * push data structure.
 *======================================================================*/

template<typename WeightT_, graptor_mode_t Mode_>
class GraptorDataParPushPartition {
public:
    using WeightT = WeightT_;
    static constexpr graptor_mode_t Mode = Mode_;

private:
    VID n;	//!< Number of vertices
    VID nv;	//!< Number of SIMD groups of vertices (n rounded up to maxVL)
    VID vlo;    //!< First vertex of slab
    EID mv;	//!< Number of SIMD groups of edges worth sizeof(VID)*VL bytes
    EID mpad;   //!< Number of padding edge entries
    unsigned short maxVL;
    mm::buffer<VID> edges;
    WeightT * weights;
    mm::buffer<VID> redir;   //!< indirection array
    VID redir_nnz; //!< length of indirection array

    static constexpr unsigned short DegreeBits = GRAPTOR_DEGREE_BITS;
    static constexpr unsigned short SkipBits = GRAPTOR_SKIP_BITS;
    static constexpr unsigned short DegreeSkipBits = DegreeBits + SkipBits;

public:
    GraptorDataParPushPartition() : weights( nullptr ) { }
    
    GraptorDataParPushPartition( VID n_, VID vlo_, EID mv_,
				 unsigned short maxVL_,
				 unsigned numa_node,
				 WeightT * const weights_,
				 mm::buffer<VID> redir_,
				 VID redir_nnz_ )
	: n( n_ ), nv( roundup_multiple_pow2( n_, (VID)maxVL_ ) ),
	  vlo( vlo_ ), mv( mv_ * maxVL_ ), mpad( 0 ), maxVL( maxVL_ ),
	  edges( mv_ * maxVL_, numa_allocation_local( numa_node ) ),
	  weights( weights_ ),
	  redir( redir_ ), redir_nnz( redir_nnz_ ) {
    }

    GraptorDataParPushPartition(
	const GraphCSx & rcsr,	//!< Remapped CSR/CSC
	partitioner & part,	//!< Partitioning of destinations
	unsigned short maxVL_, 	//!< Maximum vector length supported
	unsigned p		//!< Partition number
	) : n( rcsr.numVertices() ),
	    nv( roundup_multiple_pow2( n, (VID)maxVL_ ) ),
	    vlo( part.start_of( p ) ), mpad( 0 ), maxVL( maxVL_ ) {
	// 0. Short-hands
	const unsigned P = part.get_num_partitions();
	const VID n = rcsr.numVertices();
	unsigned numa_node = part.numa_node_of( p );

	// 1. Build histogram of frequency of vertex being source.
	//    Do this fast based on symmetry of remapped CSR
	mm::buffer<VID> ctrs( n, numa_allocation_local( numa_node ) );
	std::fill( &ctrs[0], &ctrs[n], VID(0) );

	// This assumes that rcsr is symmetric
	assert( rcsr.isSymmetric() && "requirement" );
	const EID * r_idx = rcsr.getIndex();
	const VID * r_edg = rcsr.getEdges();
	// Note: edge-starts have not been determined yet in partitioner part
	VID vs = part.start_of( p );
	VID ve = part.start_of( p+1 );
	EID es = r_idx[vs];
	EID ee = r_idx[ve];
	for( EID e=es; e < ee; ++e )
	    ctrs[r_edg[e]]++;

	// 2. Count non-zero elements.
	VID nnz = 0;
	for( VID v=0; v < n; ++v )
	    if( ctrs[v] != 0 )
		++nnz;

	// Round up to next multiple of maxVL
	VID nnz_alc = nnz;
	if( nnz_alc % VID(maxVL) )
	    nnz_alc += maxVL - ( nnz_alc % VID(maxVL) );

	// 3. Build list of non-zero elements.
	new ( &redir ) mm::buffer<VID>(
	    nnz_alc, numa_allocation_local( numa_node ) );
	redir_nnz = nnz_alc;
	VID k = 0;
	for( VID v=0; v < n; ++v ) {
	    if( ctrs[v] != 0 ) {
		redir[k++] = v;
		if( k == nnz )
		    break;
	    }
	}

	// 4. Sort list of non-zero elements
	//    TODO: set padding lanes to ~(VID)0 instead of n
	std::sort( &redir[0], &redir[nnz],
		   SortByDecrDegStable<VID>( ctrs.get() ) );
	std::fill( &redir[nnz], &redir[nnz_alc], n );

	// 5. Allocate edge and weight lists
	EID e_est = ( ee - es ) * 2; // Estimated number of edges
	new ( &edges ) mm::buffer<VID>(
	    e_est, numa_allocation_local( numa_node ) );
	WeightT * r_wght = nullptr;
	weights = nullptr;
	if( rcsr.getWeights() != nullptr ) {
	    r_wght = rcsr.getWeights()->get();
	    weights = new WeightT[e_est];
	}
	mv = e_est;

	// 6. Build representation
	static constexpr unsigned short DegreeBits = GRAPTOR_DEGREE_BITS;
	const VID dbpl = BitsPerLane<DegreeBits>( maxVL );
	const VID dmax = VID(1) << (dbpl*VID(maxVL));
	const VID vmask = (VID(1)<<(sizeof(VID)*8-dbpl))-1;

	assert( GRAPTOR_DEGREE_MULTIPLIER == 1 && "restriction" );

	VID lo = part.start_of( p );
	VID hi = part.end_of( p );
	EID pos = 0;
	for( VID i=0; i < nnz; i += VID(maxVL) ) { // for all relevant vertices
	    EID group_d = 0;
	    for( VID l=0; l < VID(maxVL); ++l ) { // iterate by maxVL vertices
		VID v = redir[i+l];

		// Handle padding edges
		if( v >= n ) {
		    for( EID d=0; d < group_d; ++d )
			edges[pos+EID(maxVL)*d+l] = ~(VID)0;
		    continue;
		}

		// Place edges. This iterates rcsr in the push direction
		EID start_d = 0;
		VID deg = r_idx[v+1] - r_idx[v];
		VID k_left = ctrs[v];
		for( VID k=0; k < deg && k_left > 0; ++k ) {
		    VID w = r_edg[r_idx[v]+k];
		    assert( ~w != 0 );
		    EID d = start_d;
		    // Note: If we assume that the neighbour lists in rcsr are
		    // sorted, then we can break the k-loop early when w >= hi
		    if( lo <= w && w < hi ) { // check neighbour is in partition
			// One fewer to complete
			--k_left;
			
			// Increase group_d
			if( d >= group_d ) {
			    assert( d == group_d );
			    for( VID j=0; j < VID(maxVL); ++j )
				edges[pos+EID(maxVL)*group_d+j] = ~(VID)0;
			    group_d++;
			    // Grow storage, reload variables
			    if( pos + EID(maxVL) * group_d > mv )
				grow( 4 * ( nnz - i ), numa_node );
			}
			
			// Find slot for w at degree rank d
			bool found;
			do {
			    found = false;
			    if( edges[pos+EID(maxVL)*d+l] != ~(VID)0 ) { // occupied
				// Position in use, no need to consider again
				if( start_d == d )
				    start_d++;
				found = true;
			    } else {
				for( VID j=0; j < l; ++j )
				    if( edges[pos+EID(maxVL)*d+j] == w ) {
					found = true;
					break;
				    }
			    }
			    if( !found )
				break;
			    // We have a conflict, so fill slot with invalid
			    // code and move up a position.
			    // p_edg[pos+EID(maxVL)*d+l] = ~(VID)0; // redundant
			    ++d;
			    // If we require more than group_d slots to store
			    // the edges for the current group of maxVL sources
			    // then extend the number and initialise all new
			    // slots with an invalid code.
			    if( d >= group_d ) {
				assert( d == group_d );
				for( VID j=0; j < VID(maxVL); ++j )
				    edges[pos+EID(maxVL)*group_d+j] = ~(VID)0;
				group_d++;
				// Grow storage, reload variables
				if( pos + EID(maxVL) * group_d > mv )
				    grow( 4 * ( nnz - i ), numa_node );
			    }
			} while( true );

			// We have now found the right d to store the edge in.
			edges[pos+EID(maxVL)*d+l] = w;
			if( weights != nullptr )
			    weights[pos+EID(maxVL)*d+l] = r_wght[r_idx[v]+k];

			// This position has been filled in, so set
			// helper pointer to next slot
			if( start_d == d )
			    start_d++;
		    } // if included in current partition
		}
	    }

	    // Now group_d is fixed and permanent.
	    // Fill out the degree metadata.
	    for( EID d=0; d < group_d; ++d ) {
		if constexpr ( GraptorConfig<Mode>::is_cached ) {
		    if( (d % (dmax/2)) != 0 )
			continue;
		}

		VID deg3 = VID(group_d) - VID(d) - 1;
		if( deg3 > (dmax/2-1) )
		    deg3 = (dmax/2-1) << 1;
		else
		    deg3 = (deg3 << 1) | 1;
		for( VID l=0; l < VID(maxVL); ++l ) {
		    VID de = deg3 >> (l * dbpl);
		    de &= ( VID(1) << dbpl ) - 1;

		    VID r = edges[pos+EID(maxVL)*d+l];
		    r &= vmask;
		    r |= de << ( sizeof(VID) * 8 - dbpl );
		    edges[pos+EID(maxVL)*d+l] = r;
		}
	    }
	    
	    // Advance position to write into edge list
	    pos += EID(maxVL) * group_d;
	    assert( pos <= mv );
	}

	mv = pos; // trim access to array
	mpad = mv - ( ee - es );

	// 7. Cleanup
	ctrs.del();
    }
	
	    
    void del() {
	edges.del();
    }

    void grow( EID slots, unsigned numa_node ) {
	slots = ( slots + 1023 ) & 1023; // round up to multiple of 1k slots
	EID new_mv = mv + slots;
	mm::buffer<VID> new_edges( new_mv, numa_allocation_local( numa_node ) );
	std::copy( &edges.get()[0], &edges.get()[mv], &new_edges.get()[0] );
	edges.del();
	edges = new_edges;

	if( weights ) {
	    WeightT * new_weights = new WeightT[new_mv];
	    std::copy( &weights[0], &weights[mv], &new_weights[0] );
	    delete[] weights;
	    weights = new_weights;
	}

	mv = new_mv;
    }

public:
    VID *getEdges() { return edges.get(); }
    const VID *getEdges() const { return edges.get(); }
    bool hasWeights() const { return weights != nullptr; }
    float * getWeights() { return weights; }
    const float * getWeights() const { return weights; }

    void moveWeights( float * p ) {
	std::copy( &weights[0], &weights[mv], p );
	delete[] weights;
	weights = p;
    }

    // VID numVertices() const { return n; }
    // EID numEdges() const { return m; }

    VID getSlimVertex() const { return vlo; }

    VID numSIMDVertices() const { return redir_nnz; }
    EID numSIMDEdges() const { return mv; }
    void setNumSIMDEdges( EID mv_new ) { mv = mv_new; }
    EID numPaddingEdges() const { return mpad; }
    void setNumPaddingEdges( EID mpad_ ) { mpad = mpad_; }

    // Compatibility with GraphCSxSIMDDegreeMixed
    EID numSIMDEdgesDelta1() const { return 0; }
    EID numSIMDEdgesDeltaPar() const { return numSIMDEdges(); }

    const VID *getRedirP() const { return redir.get(); }
    VID getRedirNNZ() const { return redir_nnz; }

    unsigned short getMaxVL() const { return maxVL; }
    unsigned short getDegreeBits() const { return (DegreeBits+maxVL-1)/maxVL; }
    unsigned short getDegreeShift() const {
	return sizeof(VID)*8 - getDegreeBits();
    }
    unsigned short getSkipBits() const { return (SkipBits+maxVL-1)/maxVL; }

    unsigned short getDegreeSkipBits() const { return getDegreeBits() + getSkipBits(); }
    unsigned short getDegreeSkipShift() const { return sizeof(VID)*8 - getDegreeSkipBits(); }
};


template<typename WeightT, graptor_mode_t Mode>
struct GraptorDataParPushBuilder {
    GraptorDataParPushPartition<WeightT,Mode> * slabs;
    mm::buffer<WeightT> weights;

public:
    template<typename Remapper>
    void
    build( const GraphCSx & rcsr,	//!< Remapped CSR
	   const GraphCSx & csc,	//!< Original CSC
	   partitioner & part,		//!< Partitioning of destinations
	   unsigned short maxVL, 	//!< Maximum vector length supported
	   const Remapper & remap	//!< Remapping vertices
	) {
	if( rcsr.isSymmetric() )
	    build_undirected( rcsr, part, maxVL );
	else
	    build_directed( rcsr, part, maxVL );
    }
	
    void
    build_undirected(
	const GraphCSx & rcsr,	//!< Remapped CSR
	partitioner & part,		//!< Partitioning of destinations
	unsigned short maxVL 	//!< Maximum vector length supported
	) {
	timer tm;
	tm.start();
	
	// 0. Short-hands
	const unsigned P = part.get_num_partitions();
	const VID n = rcsr.numVertices();

	// 1. Allocate slabs
	slabs = new GraptorDataParPushPartition<WeightT,Mode>[P];

	// 2. Fill them out
	map_partition( part, [&]( unsigned p ) {
	    new ( &slabs[p] )
		GraptorDataParPushPartition<WeightT,Mode>( rcsr, part, maxVL, p );
	} );

	std::cerr << "Graptor: build partitions: " << tm.next() << "\n";

	// 3. Complete edge partitions.
	// Edge partitions have not been initialised yet
	EID * counts = part.edge_starts();
	EID mw = 0;
	for( unsigned p=0; p < P; ++p ) {
	    counts[p] = mw;
	    mw += slabs[p].numSIMDEdges(); // Adjusted for real value
	}
	counts[P] = mw;

	// 4. Allocate weights array
	if( rcsr.getWeights() != nullptr )
	    new (&weights) mm::buffer<WeightT>(
		mw, numa_allocation_edge_partitioned( part ) );

	// 3. Copy weights and allocate old storage. Oversized edge slabs
	//    will remain.
	if( rcsr.getWeights() != nullptr )
	    map_partition( part, [&]( unsigned p ) {
		EID e = part.edge_start_of( p );
		slabs[p].moveWeights( &weights.get()[e] );
	    } );
	std::cerr << "Graptor: align weights: " << tm.next() << "\n";
    }
    
    void
    build_directed(
	const GraphCSx & rcsr,	//!< Remapped CSR
	partitioner & part,		//!< Partitioning of destinations
	unsigned short maxVL 	//!< Maximum vector length supported
	) {
	timer tm;
	tm.start();
	
	// 0. Short-hands
	const unsigned P = part.get_num_partitions();
	const VID n = rcsr.numVertices();

	// 1. Calculate indirection arrays and estimate edge counts
	//    In case of symmetric rcsr, we can calculate real edges more
	//    quickly, but we cannot determine the indirection array (degrees)
	//    and cannot count padding edges.
	VID * p_redir_nnz = new VID[P];
	mm::buffer<VID> * p_redir = new mm::buffer<VID>[P];
	EID * p_edge_est = new EID[P];
	map_partition( part, [&]( unsigned p ) {
	    std::tie( p_redir[p], p_redir_nnz[p], p_edge_est[p] )
		= determine_indirection( rcsr, part, maxVL, p );
	} );
	std::cerr << "Graptor: per-partition indirection: "
		  << tm.next() << "\n";

	// 2. Allocate slabs. This is a temporary allocation as the estimated
	//    edge counts may be too small.
	//    We also allocate temporary space for the weights, which will be
	//    copied to a compact array once the edge counts are accurate.
	slabs = new GraptorDataParPushPartition<WeightT,Mode>[P];
	for( unsigned p=0; p < P; ++p ) {
	    new ( &slabs[p] )
		GraptorDataParPushPartition<WeightT,Mode>(
		    part.end_of( p ) - part.start_of( p ),
		    part.start_of( p ), p_edge_est[p], maxVL,
		    part.numa_node_of( p ),
		    new WeightT[p_edge_est[p] * EID(maxVL)], // temporary storage
		    p_redir[p],
		    p_redir_nnz[p] );
	}
	std::cerr << "Graptor: slabs allocation: " << tm.next() << "\n";

	// 3. Fill out edges and weights.
	map_partition( part, [&]( unsigned p ) {
	    set_edges( rcsr, part, maxVL, p, slabs[p] );
	} );
	std::cerr << "Graptor: copy edges and weights: " << tm.next() << "\n";

	// 5. Complete edge partitions.
	// Edge partitions have not been initialised yet
	EID * counts = part.edge_starts();
	EID mw = 0;
	for( unsigned p=0; p < P; ++p ) {
	    counts[p] = mw;
	    mw += slabs[p].numSIMDEdges(); // Adjusted for real value
	}
	counts[P] = mw;

	// 5. Allocate weights array
	if( rcsr.getWeights() != nullptr )
	    new (&weights) mm::buffer<WeightT>(
		mw, numa_allocation_edge_partitioned( part ) );

	// 3. Copy weights and allocate old storage. Oversized edge slabs
	//    will remain.
	map_partition( part, [&]( unsigned p ) {
	    EID e = part.edge_start_of( p );
	    slabs[p].moveWeights( &weights.get()[e] );
	} );
	std::cerr << "Graptor: align weights: " << tm.next() << "\n";

	// 4. Clean up.
	delete[] p_edge_est;
	delete[] p_redir;
	delete[] p_redir_nnz;
	std::cerr << "Graptor: cleanup: " << tm.next() << "\n";
    }

    std::tuple<mm::buffer<VID>, VID, EID>
    determine_indirection(
	const GraphCSx & rcsr,
	const partitioner & part,
	unsigned short maxVL,
	unsigned p ) {
	VID n = rcsr.numVertices();
	VID lo = part.start_of( p );
	VID hi = part.end_of( p );
	const EID * idx = rcsr.getIndex();
	const VID * edg = rcsr.getEdges();

	VID * degp = new VID[n];
	VID * imap = new VID[n];

	VID nnz = 0;
	for( VID v=0; v < n; ++v ) {
	    // Calculate the degree of each vertex within this partition
	    VID mdeg = idx[v+1] - idx[v];
	    VID deg = 0;
	    for( VID j=0; j < mdeg; ++j ) {
		VID w = edg[idx[v]+j];
		if( lo <= w && w < hi )
		    ++deg;
	    }
	    degp[v] = deg;

	    // Construct a remapping array
	    if( deg )
		imap[nnz++] = v;
	}
	// TODO: Can we be more intelligent here? Look at common destinations?
	//       Or uncommon destinations to avoid padding?
	//       Sort lexicographically for tied degrees?
	std::sort( &imap[0], &imap[nnz], SortByDecrDegStable<VID>( degp ) );

	// Calculate an estimated number of required edges; based on sort order
	EID n_edg = 0;
	for( VID v=0; v < nnz; v += maxVL ) {
	    VID w = imap[v];
	    VID d = degp[w];
	    n_edg += EID(d); // counted in groups of maxVL
	    n_edg += 2;      // allow some growth for conflicts
	}

	delete[] degp;

	// Make sure that there is backing space for a vector load
	VID nnz_alc = nnz;
	if( nnz_alc % VID(maxVL) )
	    nnz_alc += maxVL - ( nnz_alc % VID(maxVL) );

	// Trim the allocation
	// TODO: use mmap_ptr and can we extend ifc to trim using munmap?
	mm::buffer<VID> rmap( nnz_alc,
			      numa_allocation_local( part.numa_node_of( p ) ),
			      "indirection array Graptor Push" );
	std::copy( &imap[0], &imap[nnz], rmap.get() );
	for( VID v=nnz; v < nnz_alc; ++v ) // low numbers
	    rmap[v] = n;
	
	delete[] imap;

	return std::make_tuple( rmap, nnz_alc, n_edg );
    }

    void set_edges(
	const GraphCSx & rcsr,
	const partitioner & part,
	unsigned short maxVL,
	unsigned p,
	GraptorDataParPushPartition<WeightT,Mode> & slab ) {
	// This code starts of with speculatively allocated slabs for the edges
	// and separately the weights. If these are too small, a re-allocation
	// needs to occur.
	VID n = rcsr.numVertices();
	VID lo = part.start_of( p );
	VID hi = part.end_of( p );
	const EID * idx = rcsr.getIndex();
	const VID * edg = rcsr.getEdges();
	VID * redir = const_cast<VID *>( slab.getRedirP() );
	VID nnz = slab.getRedirNNZ();
	EID mv = slab.numSIMDEdges();
	VID * p_edg = slab.getEdges();
	WeightT * p_weights = slab.getWeights();
	const WeightT * weights = nullptr;
	if( rcsr.getWeights() != nullptr )
	    weights = rcsr.getWeights()->get();

	static constexpr unsigned short DegreeBits = GRAPTOR_DEGREE_BITS;
	const VID dbpl = BitsPerLane<DegreeBits>( maxVL );
	const VID dmax = VID(1) << (dbpl*VID(maxVL));
	const VID vmask = (VID(1)<<(sizeof(VID)*8-dbpl))-1;

	assert( GRAPTOR_DEGREE_MULTIPLIER == 1 && "restriction" );

	bool degree_one_phase = false;

	EID e_real = 0;
	EID pos = 0;
	for( VID i=0; i < nnz; i += VID(maxVL) ) { // for all relevant vertices
	    EID group_d = 0;
	    for( VID l=0; l < VID(maxVL); ++l ) { // iterate by maxVL vertices
		VID v = redir[i+l];
		if( v >= n ) {
		    // From here on, we look at padding vertices to tally up
		    // to nnz that is a multiple of maxVL. Zero out all
		    // corresponding edge slots in the edge array.
/*
		    for( EID d=0; d < group_d; ++d )
			p_edg[pos+EID(maxVL)*d+l] = ~(VID)0;
*/
		    // Go to next vertex (will also obey v >= n)
		    continue;
		}

		// If we are dealing with vertices with only degree 1,
		// we may find many of them linking to the same vertex, making
		// us unable to fill vectors beyond one. Rather than spreading
		// destinations out over vectors, actively search for a vertex
		// that fits. Note that this could be performed also for
		// higher-degree vertices; it would just be more complex.
		if( degree_one_phase && l > 0 ) {
		    // Find a vertex with a compatible neighbour
		    for( VID ii=i+l; ii < nnz; ++ii ) {
			VID vv = redir[ii];
			if( vv >= n )
			    break;
			// Find the neighbour
			VID u = ~(VID)0;
			VID deg = idx[v+1] - idx[v];
			for( VID k=0; k < deg; ++k ) {
			    VID w = edg[idx[v]+k];
			    if( lo <= w && w < hi ) {
				u = w;
				break;
			    }
			}
			assert( u != ~(VID)0 );
			// Check that the neighbour is compatible
			// We may have already split over multiple vectors
			// We only need one vector where the vertex u can
			// be placed.
			bool found = false;
			for( VID d=0; d < group_d; ++d ) {
			    found = false;
			    for( VID j=0; j < l; ++j )
				if( p_edg[pos+EID(maxVL)*d+j] == u ) {
				    found = true;
				    break;
				}
			    if( !found )
				break;
			}
			// We have found a good vertex. Use it next.
			if( !found ) {
			    if( ii != i+l ) {
				std::swap( redir[ii], redir[i+l] );
				v = redir[i+l];
			    }
			    break;
			}
		    }
		    // If arrive here, either we have a good vertex in
		    // position i+l/v that has a compatible neighbour,
		    // or we failed to find one. Either way, proceed as
		    // normal and place it, or spread out vectors.
		}

		// Place the vertex
		VID start_d = 0;
		VID deg = idx[v+1] - idx[v];
		for( VID k=0; k < deg; ++k ) { // iterate over all neighbours
		    VID w = edg[idx[v]+k];
		    assert( ~w != 0 );
		    EID d = start_d;
		    // Note: If we assume that the neighbour lists in rcsr are
		    // sorted, then we can break the k-loop early when w >= hi
		    if( lo <= w && w < hi ) { // check neighbour is in partition
			// We found an edge to insert
			++e_real;
			
			// Increase group_d
			if( d >= group_d ) {
			    assert( d == group_d );
			    for( VID j=0; j < VID(maxVL); ++j )
				p_edg[pos+EID(maxVL)*group_d+j] = ~(VID)0;
			    group_d++;
			    // Grow storage, reload variables
			    if( pos + EID(maxVL) * group_d > mv ) {
				slab.grow( 4 * ( nnz - i ),
					   part.numa_node_of(p) );
				mv = slab.numSIMDEdges();
				p_edg = slab.getEdges();
				p_weights = slab.getWeights();
			    }
			    assert( pos + EID(maxVL) * group_d <= mv );
			}
			
			// Find slot for w at degree rank d
			bool found;
			do {
			    found = false;
			    if( p_edg[pos+EID(maxVL)*d+l] != ~(VID)0 ) { // occupied
				// Position in use, no need to consider again
				if( start_d == d )
				    start_d++;
				found = true;
			    } else {
				for( VID j=0; j < l; ++j )
				    if( p_edg[pos+EID(maxVL)*d+j] == w ) {
					found = true;
					break;
				    }
			    }
			    if( !found )
				break;
			    // We have a conflict, so fill slot with invalid
			    // code and move up a position.
			    // p_edg[pos+EID(maxVL)*d+l] = ~(VID)0; // redundant
			    ++d;
			    // If we require more than group_d slots to store
			    // the edges for the current group of maxVL sources
			    // then extend the number and initialise all new
			    // slots with an invalid code.
			    if( d >= group_d ) {
				assert( d == group_d );
				for( VID j=0; j < VID(maxVL); ++j )
				    p_edg[pos+EID(maxVL)*group_d+j] = ~(VID)0;
				group_d++;
				// Grow storage, reload variables
				if( pos + EID(maxVL) * group_d > mv ) {
				    slab.grow( 4 * ( nnz - i ),
					       part.numa_node_of(p) );
				    mv = slab.numSIMDEdges();
				    p_edg = slab.getEdges();
				    p_weights = slab.getWeights();
				}
				assert( pos + EID(maxVL) * group_d <= mv );
			    }
			} while( true );

			// We have now found the right d to store the edge in.
			p_edg[pos+EID(maxVL)*d+l] = w;
			if( weights != nullptr )
			    p_weights[pos+EID(maxVL)*d+l] = weights[idx[v]+k];

			// This position has been filled in, so set
			// helper pointer to next slot
			if( start_d == d )
			    start_d++;
		    } // if included in current partition
		} // for k (neighbours)
		if( l == 0 && group_d == 1 )
		    degree_one_phase = true;
/*
		assert( d <= group_d );
		// zero out remaining slots allocated for prior higher-degree
		// vertices
		while( d < group_d ) {
		    p_edg[pos+EID(maxVL)*d+l] = ~(VID)0;
		    ++d;
		}
*/
	    } // for maxVL vertices

	    // Now group_d is fixed and permanent.
	    // Fill out the degree metadata.
	    for( EID d=0; d < group_d; ++d ) {
		if constexpr ( GraptorConfig<Mode>::is_cached ) {
		    if( (d % (dmax/2)) != 0 )
			continue;
		}

		VID deg3 = VID(group_d) - VID(d) - 1;
		if( deg3 > (dmax/2-1) )
		    deg3 = (dmax/2-1) << 1;
		else
		    deg3 = (deg3 << 1) | 1;
		for( VID l=0; l < VID(maxVL); ++l ) {
		    VID de = deg3 >> (l * dbpl);
		    de &= ( VID(1) << dbpl ) - 1;

		    VID r = p_edg[pos+EID(maxVL)*d+l];
		    r &= vmask;
		    r |= de << ( sizeof(VID) * 8 - dbpl );
		    p_edg[pos+EID(maxVL)*d+l] = r;
		}
	    }
	    
	    // Advance position to write into edge list
	    pos += EID(maxVL) * group_d;
	    assert( pos <= mv );
	} // for every maxVL-th vertex

	// Now adjust number of edges
	slab.setNumPaddingEdges( pos - e_real );
	slab.setNumSIMDEdges( pos );
    }
};

// Override
template<>
class GraphCSRSIMDDegreeMixed<GRAPTOR_MODE_MACRO(0,0,0)>
    : public GraptorDataParPushPartition<float,GRAPTOR_MODE_MACRO(0,0,0)> {
};

template<>
class GraphCSRSIMDDegreeMixed<GRAPTOR_MODE_MACRO(1,0,0)>
    : public GraptorDataParPushPartition<float,GRAPTOR_MODE_MACRO(1,0,0)> {
};


#endif // GRAPTOR_GRAPH_GRAPTORDATAPARPUSH_H
