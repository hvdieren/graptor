// -*- c++ -*-
#ifndef GRAPTOR_GRAPH_GRAPTORDATAPARPULL_H
#define GRAPTOR_GRAPH_GRAPTORDATAPARPULL_H

#include "graptor/partitioner.h"
#include "graptor/graph/GraphCSx.h"
#include "graptor/graph/GraptorDef.h"
#include "graptor/graph/CGraphCSxSIMDDegreeMixed.h"

/*======================================================================
 * An optimized version of the construction of the Graptor data-parallel
 * pull data structure.
Notes for new Graptor Pull Data-Parallel

Steps to construct:
1. Calculate slab size per partition
   1.a. if CSR* is symmetric, calculate slab size as sum of degree of each
   	VL-th vertex
   1.b. if CSR* is directed, options:
   	1.b.i. Work from CSC and use remap array (O(V))
	1.b.ii. Work from CSR* and scan edges (O(E))
	In both cases, only look at VL-th edges

	(does not apply)
	Need to know only sum of degrees, so need only one counter per partition
	Can store as PxP counters: privatized, no atomics

2. Allocate slabs

3. Write vertices in place (transpose from CSR* = relabeled CSR)
   3.a. if CSR* is symmetric, then construct from CSR*, no sorting required (step 4)
   	Need to write padding locations correctly.
	Otherwise pretty much straightforward copy (expanded in space)
	Or better to construct as zip from VL streams
	   (for high-degree vertices)?
   3.b. if CSR* is directed, then write in correct place immediately
   	Sort neighbour lists (incl weights)
	Note: poor locality, spaced apart, one element per cache block!

4. Fill in degree codes. Does require to know per-vertex in-degrees! (see 1.b)

Conclusion: should be extremely efficient for undirected graphs!
 *======================================================================*/

template<typename WeightT_>
class GraptorDataParPullPartition {
public:
    using WeightT = WeightT_;

private:
    VID n;	//!< Number of vertices
    VID nv;	//!< Number of SIMD groups of vertices (n rounded up to maxVL)
    VID vlo;    //!< First vertex of slab
    EID mv;	//!< Number of SIMD groups of edges worth sizeof(VID)*VL bytes
    unsigned short maxVL;
    mm::buffer<VID> edges;
    WeightT * const weights;

    static constexpr unsigned short DegreeBits = GRAPTOR_DEGREE_BITS;

public:
    GraptorDataParPullPartition() : weights( nullptr ) { }
    
    GraptorDataParPullPartition( VID n_, VID vlo_, EID mv_,
				 unsigned short maxVL_,
				 unsigned numa_node,
				 WeightT * const weights_ )
	: n( n_ ), nv( n_ + ( (-n_) % maxVL_ ) ), vlo( vlo_ ),
	  mv( mv_ * maxVL_ ),
	  maxVL( maxVL_ ),
	  edges( mv_ * maxVL_, numa_allocation_local( numa_node ) ),
	  weights( weights_ ) {
    }
    void del() {
	edges.del();
    }

public:
    VID *getEdges() { return edges.get(); }
    const VID *getEdges() const { return edges.get(); }
    bool hasWeights() const { return weights != nullptr; }
    float * getWeights() { return weights; }
    const float * getWeights() const { return weights; }

    VID numVertices() const { return n; }
// EID numEdges() const { return m; }

    VID getSlimVertex() const { return vlo; }

    VID numSIMDVertices() const { return nv; }
    EID numSIMDEdges() const { return mv; }

    // Compatibility with GraphCSxSIMDDegreeMixed
    EID numSIMDEdgesDelta1() const { return 0; }
    EID numSIMDEdgesDeltaPar() const { return numSIMDEdges(); }

    unsigned short getMaxVL() const { return maxVL; }
    unsigned short getDegreeBits() const { return (DegreeBits+maxVL-1)/maxVL; }
    unsigned short getDegreeShift() const {
	return sizeof(VID)*8 - getDegreeBits();
    }
};


template<typename WeightT, graptor_mode_t Mode>
struct GraptorDataParPullBuilder {
    GraptorDataParPullPartition<WeightT> * slabs;
    mm::buffer<WeightT> weights;

public:
    void
    build( const GraphCSx & rcsr,	//!< Remapped CSR
	   partitioner & part,		//!< Partitioning of destinations
	   unsigned short maxVL 	//!< Maximum vector length supported
	) {
	if( rcsr.isSymmetric() )
	    build_symmetric( rcsr, part, maxVL );
	else
	    build_asymmetric( rcsr, part, maxVL );
    }
	   
private:
    void
    build_symmetric( const GraphCSx & rcsr,	//!< Remapped CSR
		     partitioner & part,	//!< Partitioning destinations
		     unsigned short maxVL 	//!< Maximum vector length
	) {
	// 0. Short-hands
	const unsigned P = part.get_num_partitions();
	const VID n = rcsr.numVertices();

	// 1. Calculate size of each partition as the sum of degrees of
	//    every maxVL-th vertex.
	EID * e_per_p = edges_per_partition_from_csr( rcsr, part, maxVL );

	// Edge partitions have not been initialised yet
	EID * counts = part.edge_starts();
	EID mw = 0;
	for( unsigned p=0; p < P; ++p ) {
	    counts[p] = mw * (EID)maxVL;
	    mw += e_per_p[p];
	}
	counts[P] = mw * (EID)maxVL;

	// 2. Allocate partition slabs
	// 2.a. Allocate weights array
	if( rcsr.getWeights() != nullptr )
	    new (&weights) mm::buffer<WeightT>(
		mw * maxVL, numa_allocation_edge_partitioned( part ) );

	// 2.b. Allocate slabs
	slabs = new GraptorDataParPullPartition<WeightT>[P];
	for( unsigned p=0; p < P; ++p ) {
	    new ( &slabs[p] )
		GraptorDataParPullPartition<WeightT>(
		    n, part.start_of( p ), e_per_p[p], maxVL,
		    part.numa_node_of( p ),
		    &weights[part.edge_start_of( p )] );
	}

	// 3. Fill out edges and weights. Assume that edges are sorted
	//    in rcsr, so we only need to copy.
	map_partition( part, [&]( unsigned p ) {
	    const VID * redges = rcsr.getEdges();
	    const EID * rindex = rcsr.getIndex();
	    const WeightT * rweights = rcsr.getWeights()
		? rcsr.getWeights()->get() : nullptr;

	    VID * pedges = slabs[p].getEdges();
	    WeightT * pweights = slabs[p].getWeights();

	    constexpr unsigned short DegreeBits = GRAPTOR_DEGREE_BITS;
	    const unsigned short dbpl = BitsPerLane<DegreeBits>( maxVL );
	    const VID absent = (VID(1)<<(sizeof(VID)*8-dbpl)) - 1;
	    const VID dmax = VID(1) << (dbpl*maxVL);

	    const VID vs = part.start_of( p );
	    const VID ve = part.end_of( p );
	    EID pe = 0;
	    for( VID v=vs; v < ve; v += maxVL ) {
		EID deg = rindex[v+1] - rindex[v];
		for( EID d=0; d < deg; ++d ) {
		    // Determine encoded bits
		    VID enc;
		    if( GraptorConfig<Mode>::is_cached 
			&& (d % (GRAPTOR_DEGREE_MULTIPLIER*(dmax/2-1))) != 0 )
			enc = 0;
		    else {
			if( d+1 == deg ) {
#if GRAPTOR_STOP_BIT_HIGH
			    enc = dmax/2;
#else
			    enc = 1;
#endif
			} else {
			    VID deg3 = (deg - d - 1) / GRAPTOR_DEGREE_MULTIPLIER;
			    if( deg3 > (dmax/2-1) )
				deg3 = dmax/2-1;
#if GRAPTOR_STOP_BIT_HIGH
			    enc = deg3;
#else
			    enc = deg3 << 1;
#endif
			}
		    }

		    // Set source vertices and add encoded bits
		    for( unsigned l=0; l < maxVL; ++l ) {
			VID b = enc & ( ( VID(1) << dbpl ) - 1 );
			enc = enc >> dbpl;

			if( rindex[v+l]+d < rindex[v+l+1] ) {
			    VID u = redges[rindex[v+l]+d];
			    u &= absent;
			    u |= b << ( sizeof(VID) * 8 - dbpl );
			    pedges[pe] = u;

			    if( rweights )
				pweights[pe] = rweights[rindex[v+l]+d];
			} else {
			    // No need to set a weight in this case
			    VID u = absent;
			    u |= b << ( sizeof(VID) * 8 - dbpl );
			    pedges[pe] = u;
			}

			++pe;
		    }
		}
	    }
	} );

	// 4. Clean up.
	delete[] e_per_p;
    }

    EID *
    edges_per_partition_from_csr(
	const GraphCSx & rcsr,		//!< Remapped CSR
	const partitioner & part,	//!< Partitioning of destinations
	unsigned short maxVL 		//!< Maximum vector length supported
	) {
	// Calculate size of each partition as the sum of degrees of
	// every maxVL-th vertex.
	const unsigned P = part.get_num_partitions();
	const VID n = rcsr.numVertices();
	const VID * degree = rcsr.getDegree();
	EID * e_per_p = new EID[P];
	map_partition( part, [&]( unsigned p ) {
	    VID vs = part.start_of( p );
	    VID ve = part.end_of( p );
	    EID ne = 0;
	    for( VID v=vs; v < ve; v += maxVL )
		ne += (EID)degree[v];
	    e_per_p[p] = ne;
	} );
	return e_per_p;
    }

    void
    build_asymmetric( const GraphCSx & rcsr,	//!< Remapped CSR
		      partitioner & part,	//!< Partitioning destinations
		      unsigned short maxVL 	//!< Maximum vector length
	) {
	assert( 0 && "NYI" );
    }
};

// Override
template<>
class GraphCSxSIMDDegreeMixed<GRAPTOR_MODE_MACRO(1,0,1)>
    : public GraptorDataParPullPartition<float> {
};

template<>
class GraphCSxSIMDDegreeMixed<GRAPTOR_MODE_MACRO(1,0,0)>
    : public GraptorDataParPullPartition<float> {
};


#endif // GRAPTOR_GRAPH_GRAPTORDATAPARPULL_H
