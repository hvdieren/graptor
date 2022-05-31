// -*- C++ -*-
#ifndef GRAPHGRIND_PARTITIONER_H
#define GRAPHGRIND_PARTITIONER_H

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <sched.h>
#include <errno.h>
#include <cstring>
#include <utility>
#include <algorithm>

#include "graptor/utils.h"

template<typename lVID, typename lEID, typename lPID>
class partitioner_template
{
public:
    using VID = lVID;
    using EID = lEID;
    using PID = lPID;

public:
    // Deep copy semantics: every copy gets a new array
    partitioner_template() : num_partitions( 0 ), num_per_node( 0 ),
			     counts( nullptr ), starts( nullptr ), 
			     nzcounts( nullptr ), inuse( nullptr ),
			     pstarts( nullptr ), estarts( nullptr ) { }
    partitioner_template( VID n, VID e ) : num_partitions( n )
    {
        counts = new VID [num_partitions+1];
        starts = new VID [num_partitions+1];
        nzcounts = new VID [num_partitions+1];
        inuse = new VID [num_partitions+1];
	pstarts = new PID [num_numa_node+1];
	estarts = new EID [num_partitions+1];
        counts[num_partitions] = e;
        num_per_node = num_partitions/num_numa_node;
	// assert( num_per_node * num_numa_node == num_partitions
	// && "number of partitions must be multiple of NUMA nodes" );
	numa_setup();
    }
    partitioner_template( const partitioner_template & p )
	: num_partitions( p.num_partitions )
    {
        counts = new VID [num_partitions+1];
        starts = new VID [num_partitions+1];
        nzcounts = new VID [num_partitions+1];
        inuse = new VID [num_partitions+1];
	pstarts = new PID [num_numa_node+1];
	estarts = new EID [num_partitions+1];
        std::copy( &p.counts[0], &p.counts[num_partitions+1], counts );
        std::copy( &p.starts[0], &p.starts[num_partitions+1], starts );
        std::copy( &p.nzcounts[0], &p.nzcounts[num_partitions+1], nzcounts );
        std::copy( &p.inuse[0], &p.inuse[num_partitions+1], inuse );
        std::copy( &p.estarts[0], &p.estarts[num_partitions+1], estarts );
        num_per_node = num_partitions/num_numa_node;
	// assert( num_per_node * num_numa_node == num_partitions
	// && "number of partitions must be multiple of NUMA nodes" );
	numa_setup();
    }
    partitioner_template scale( VID s ) const {
	partitioner_template p( num_partitions, counts[num_partitions] * s );
	for( PID i=0; i < num_partitions+1; ++i ) {
	    p.counts[i] = counts[i] * s;
	    p.starts[i] = starts[i] * s;
	    p.nzcounts[i] = nzcounts[i] * s;
	    p.inuse[i] = inuse[i] * s;
	    p.estarts[i] = estarts[i] * s;
	}
	return p;
    }
    partitioner_template contract( VID s ) const {
	for( PID i=0; i < num_partitions; ++i ) {
	    assert( ( counts[i] % s ) == 0 );
	    assert( ( starts[i] % s ) == 0 );
	}
	partitioner_template p( num_partitions,
				( counts[num_partitions] + s - 1 ) / s );
	for( PID i=0; i < num_partitions+1; ++i ) {
	    p.counts[i] = counts[i] / s;
	    p.starts[i] = starts[i] / s;
	    p.nzcounts[i] = ( nzcounts[i] + s - 1 ) / s;
	    p.inuse[i] = ( inuse[i] + s - 1 ) / s;
	    p.estarts[i] = ( estarts[i] + s - 1 ) / s;
	}
	return p;
    }
    partitioner_template contract_widen( VID s ) const {
	partitioner_template p( num_partitions,
				( counts[num_partitions] + s - 1 ) / s );
	for( PID i=0; i < num_partitions+1; ++i ) {
	    p.counts[i] = counts[i] / s;
	    p.starts[i] = starts[i] / s;
	    p.nzcounts[i] = ( nzcounts[i] + s - 1 ) / s;
	    p.inuse[i] = ( inuse[i] + s - 1 ) / s;
	}
	return p;
    }
    static partitioner_template
    vertex_balanced( VID np, VID elm, int roundup = 1 ) {
	partitioner_template p( np, elm );

	VID remaining = elm;
	for( PID i=0; i < p.num_partitions; ++i ) {
	    VID avg_cnt = remaining / ( p.num_partitions - i );
	    VID cnt_i = avg_cnt;
	    if( cnt_i % roundup != 0 )
		cnt_i += roundup - ( cnt_i % roundup );
	    if( cnt_i > remaining )
		cnt_i = remaining;
	    remaining -= cnt_i;
	    
	    p.counts[i] = cnt_i;
	    p.starts[i] = i == 0 ? 0 : p.starts[i-1] + p.counts[i-1];
	    p.nzcounts[i] = cnt_i;
	    p.inuse[i] = cnt_i;
	    p.estarts[i] = 0;
	}

	p.starts[p.num_partitions] = elm;
	p.nzcounts[p.num_partitions] = elm;
	p.inuse[p.num_partitions] = elm;
	p.estarts[p.num_partitions] = 0;

	return p;
    }
    const partitioner_template & operator = ( const partitioner_template & p )
    {
        if( counts )
            delete [] counts;
        if( starts )
            delete [] starts;
        if( nzcounts )
            delete [] nzcounts;
        if( inuse )
            delete [] inuse;
        if( pstarts )
            delete [] pstarts;
        if( estarts )
            delete [] estarts;
        num_partitions = p.num_partitions;
        num_per_node = num_partitions/num_numa_node;
	// assert( num_per_node * num_numa_node == num_partitions
	// && "number of partitions must be multiple of NUMA nodes" );
        counts = new VID [num_partitions+1];
        starts = new VID [num_partitions+1];
        nzcounts = new VID [num_partitions+1];
        inuse = new VID [num_partitions+1];
	pstarts = new PID [num_numa_node+1];
	estarts = new EID [num_partitions+1];
        std::copy( &p.counts[0], &p.counts[num_partitions+1], counts );
        std::copy( &p.starts[0], &p.starts[num_partitions+1], starts );
        std::copy( &p.nzcounts[0], &p.nzcounts[num_partitions+1], nzcounts );
        std::copy( &p.inuse[0], &p.inuse[num_partitions+1], inuse );
        std::copy( &p.pstarts[0], &p.pstarts[num_numa_node+1], pstarts );
        std::copy( &p.estarts[0], &p.estarts[num_partitions+1], estarts );
        return *this;
    }
    ~partitioner_template()
    {
        if( counts )
            delete [] counts;
        if( starts )
            delete [] starts;
        if( nzcounts )
            delete [] nzcounts;
        if( inuse )
            delete [] inuse;
        if( pstarts )
            delete [] pstarts;
	if( estarts )
	    delete[] estarts;
    }
    // For easy interfacing with partitionByDegree()
    VID * as_array() { return counts; }
    const VID * as_array() const { return counts; }
    VID * inuse_as_array() { return inuse; }
    const VID * inuse_as_array() const { return inuse; }
    EID * edge_starts() { return estarts; }
    const EID * edge_starts() const { return estarts; }

    PID numa_start_of( PID numa_node ) const   { return pstarts[numa_node]; }
    PID numa_end_of( PID numa_node ) const     { return pstarts[numa_node+1]; }
    [[deprecated("try not to use this")]]
    PID get_num_per_node_partitions() const { return num_per_node; }
    PID numa_node_of( PID p ) const {
	return bisect_starts<VID>( p, num_numa_node, pstarts );
    }

    PID get_num_partitions() const	    { return num_partitions; }
    VID get_num_elements() const	    { return starts[num_partitions]; }
    VID get_num_vertices() const	    { return inuse[num_partitions]; }
    VID get_num_padding() const {
	return get_num_elements() - get_num_vertices();
    }

    // These methods query the number of allocated vertices rather than
    // real vertices
    VID get_vertex_range() const	    { return counts[num_partitions]; }
    VID get_size( PID i ) const		    { return counts[i]; }

    // Allocate edges
    EID get_edge_range() const	    	    { return estarts[num_partitions]; }

    // Get the start/end number of vertices in a partition
    VID start_of( PID i ) const 	    { return starts[i]; }
    VID end_of( PID i ) const		    { return starts[i] + inuse[i]; }

    // Get the start/end number of edges in a partition
    EID edge_start_of( PID i ) const 	    { return estarts[i]; }
    EID edge_end_of( PID i ) const	    { return estarts[i+1]; }

    // Get the start/end number of vertices in a partition assuming vertex
    // balanced partitioning
    VID start_of_vbal( PID i ) const 	    {
	const VID P = get_num_partitions();
	VID vs = ((VID)i) * ( ( get_vertex_range() + (P-1) ) / P );
	if( vs > get_vertex_range() ) vs = get_vertex_range();
	return vs;
    }
    VID end_of_vbal( PID i ) const	    {
	return start_of_vbal( i+1 );
    }

    // Either vertices or edges
    template<bool byV>
    auto start_of( PID i ) const {
	if constexpr ( byV )
	    return start_of( i );
	else
	    return edge_start_of( i );
    }

    // Get the partition number of a vertex
    PID part_of( VID v ) const {
	return bisect_starts<EID>( v, num_partitions, starts );
#if 0
	// Optimized for the case where all partitions have about the same
	// number of vertices (load balanced for vertices)
	// Could do binary search after initial bisection.
	PID est = PID( EID(v) * EID(num_partitions) / EID(get_num_elements()) );
	if( start_of( est ) <= v ) {
	    do {
		if( start_of( est ) <= v && v < start_of( est+1 ) )
		    return est;
	    } while( ++est < get_num_partitions() );
	} else {
	    while( est-- > 0 ) {
		if( start_of( est ) <= v && v < start_of( est+1 ) )
		    return est;
	    }
	}
	assert( 0 && "Failed to find partition of vertex" );
#endif
    }

    // Get the start number of each partition; non-zero count and inuse vertices
    void compute_starts() //  PID multiple = 0 )
    {
        VID startID = 0;
	VID totinuse = 0;
        for( PID i=0; i <= num_partitions; i++ )
        {
	    inuse[i] = counts[i];
	    if( i < num_partitions )
		totinuse += inuse[i];

#if 0
	    // If required, ensure that the number of vertices in each
	    // partition is a multiple of @a multiple by inserting padding
	    if( multiple != 0 && counts[i] % multiple != 0 ) {
		VID move = multiple - ( counts[i] % multiple );
		counts[i] += move;
		if( i < num_partitions )
		    counts[i+1] -= move;
	    }
#endif

            starts[i] = startID;
            startID += counts[i];

	    nzcounts[i] = inuse[i]; // TODO
        }
	inuse[num_partitions] = totinuse;
	assert( totinuse <= counts[num_partitions] );
    }
    // Get the start number of each partition; non-zero count and inuse vertices
    void compute_starts_inuse() {
        VID startID = 0;
	VID totinuse = 0;
        for( PID i=0; i < num_partitions; i++ )
        {
	    totinuse += inuse[i];
	    // inuse[i] = counts[i]; // temp - TODO
            starts[i] = startID;
            startID += counts[i];

	    nzcounts[i] = inuse[i]; // TODO
        }
	starts[num_partitions] = startID;
	inuse[num_partitions] = totinuse;
	nzcounts[num_partitions] = totinuse;
    }

    // Append vertices in the end of the final partition
    // It is assumed that compute_starts*() has been called.
    void appendv( VID next ) {
	PID p = num_partitions - 1;
	counts[p] += next;
	counts[p+1] += next;
	starts[p+1] += next;
    }

private:
    // Get the partition number of a vertex
    template<typename W, typename T, typename U>
    static T bisect_starts( U v, T parts, U * starts ) {
	// Optimized for the case where all partitions have about the same
	// number of vertices (load balanced for vertices)
	// Could do binary search after initial bisection.
	T est = T( W(v) * W(parts) / W(starts[parts]) );
	if( starts[est] <= v ) {
	    do {
		if( starts[est] <= v && v < starts[est+1] )
		    return est;
	    } while( ++est < parts );
	} else {
	    while( est-- > 0 ) {
		if( starts[est] <= v && v < starts[est+1] )
		    return est;
	    }
	}
	assert( 0 && "Failed to bisect starts array" );
    }

    void numa_setup() {
	PID avg = ( num_partitions + num_numa_node - 1 ) / num_numa_node;
	PID run = 0;
	for( PID n=0; n < num_numa_node; ++n ) {
	    pstarts[n] = run;
	    run += avg;
	    if( run >= num_partitions )
		run = num_partitions;
	}
	pstarts[num_numa_node] = num_partitions;
    }
    
private:
    PID num_partitions;	//!< number of partitions
    PID num_per_node;	//!< number of partitions per NUMA node
    VID * counts;	//!< number of vertices per partition
    VID * starts;	//!< first vertex index of each partition
    VID * nzcounts;	//!< number of vertices/partition with non-zero degree
    VID * inuse;	//!< number of vertices/partition in use
    PID * pstarts;	//!< start partition IDs per NUMA node
    EID * estarts;      //!< start indices of edge counts
};
using partitioner = partitioner_template<VID, EID, unsigned int>;

// Serially map over partitions
template<typename Fn>
void map_partition_serialL( const partitioner & part, Fn fn ) {
    VID _np = part.get_num_partitions();
    for( VID vname=0; vname < _np; ++vname ) {
	fn( vname );
    }
}

// Serially map over vertices
template<typename Fn>
void map_vertex_serialL( const partitioner & part, Fn fn ) {
    using VID = typename partitioner::VID;
    unsigned int np = part.get_num_partitions();
    for( unsigned int p=0; p < np; ++p ) {
	VID s = part.start_of( p );
	VID e = part.end_of( p );
	for( VID v=s; v < e; ++v )
	    fn( v );
    }
}

// Serially map over edges
template<typename Fn>
void map_edge_serialL( const partitioner & part, Fn fn ) {
    using EID = typename partitioner::EID;
    unsigned int np = part.get_num_partitions();
    for( unsigned int p=0; p < np; ++p ) {
	EID ps = part.edge_start_of( p );
	EID pe = part.edge_end_of( p );
	for( EID e=ps; e < pe; ++e )
	    fn( e );
    }
}

#endif // GRAPHGRIND_PARTITIONER_H
