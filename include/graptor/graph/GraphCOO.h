// -*- C++ -*-
#ifndef GRAPTOR_GRAPH_GRAPHCOO_H
#define GRAPTOR_GRAPH_GRAPHCOO_H

#include "graptor/mm.h"
#include "graptor/mm/mm.h"

class GraphCOO {
    VID n;
    EID m;
    mmap_ptr<VID> src;
    mmap_ptr<VID> dst;
    mm::buffer<float> * weights;

public:
    GraphCOO() { }
    GraphCOO( VID n_, EID m_, int allocation, bool weights = false )
	: n( n_ ), m( m_ ), weights( nullptr ) {
	if( allocation == -1 )
	    allocateInterleaved( weights );
	else
	    allocateLocal( allocation, weights );
    }
    template<typename vertex>
    GraphCOO( const wholeGraph<vertex> & WG,
	      const partitioner & part, int p )
	: n( WG.numVertices() ), weights( nullptr ) {
	// Range of destination vertices to include
	VID rangeLow = part.start_of(p);
	VID rangeHi = part.start_of(p+1);
	
	// Short-hands
	vertex *V = WG.V.get();

	EID num_edges = 0;
        for( VID i=rangeLow; i < rangeHi; i++ )
	    num_edges += V[i].getInDegree();

	m = num_edges;
	allocateLocal( part.numa_node_of( p ) );
	
        EID k = 0;
        for( VID i=rangeLow; i < rangeHi; i++ ) {
            for( VID j=0; j < V[i].getInDegree(); ++j ) {
                VID d = V[i].getInNeighbor( j );
		src[k] = d;
		dst[k] = i;
#ifdef WEIGHTED
		wgh[k] = V[i].getInWeight( j );
#endif
		k++;
	    }
	}
	assert( k == num_edges );

#if EDGES_HILBERT
	hilbert_sort();
#else
	CSR_sort();
#endif
    }
    GraphCOO( const GraphCSx & Gcsr, int allocation )
	: n( Gcsr.numVertices() ), m( Gcsr.numEdges() ), weights( nullptr ) {
	if( allocation == -1 )
	    allocateInterleaved( Gcsr.getWeights() != nullptr );
	else
	    allocateLocal( allocation, Gcsr.getWeights() != nullptr );
	
        EID k = 0;
        for( VID i=0; i < n; i++ ) {
	    VID deg = Gcsr.getDegree(i);
	    const VID * ngh = &Gcsr.getEdges()[Gcsr.getIndex()[i]];
	    const float * w
		= Gcsr.getWeights()
		? &Gcsr.getWeights()->get()[Gcsr.getIndex()[i]]
		: nullptr;
            for( VID j=0; j < deg; ++j ) {
		VID d = ngh[j];
		src[k] = i;
		dst[k] = d;
#ifdef WEIGHTED
		wgh[k] = V[i].getInWeight( j );
#endif
		if( weights )
		    weights->get()[k] = w[j];
		k++;
	    }
	}
	assert( k == m );

#if EDGES_HILBERT
	hilbert_sort();
#else
	CSR_sort();
#endif
    }
	
    GraphCOO( const GraphCSx & Gcsc,
	      const partitioner & part, int p )
	: n( Gcsc.numVertices() ), weights( nullptr ) {
	// Range of destination vertices to include
	VID rangeLow = part.start_of(p);
	VID rangeHi = part.start_of(p+1);
	
	EID num_edges = 0;
        for( VID i=rangeLow; i < rangeHi; i++ )
	    num_edges += Gcsc.getDegree(i);

	m = num_edges;
	allocateLocal( part.numa_node_of( p ), Gcsc.getWeights() != nullptr );
	
	// Temporary info - outdegree of inverted edges - later insertion pos
	// to aid insertion in order sorted by source
	mmap_ptr<EID> pos( n, numa_allocation_interleaved() );
	std::fill( &pos[0], &pos[n], EID(0) );

        for( VID i=rangeLow; i < rangeHi; i++ ) {
	    VID deg = Gcsc.getDegree(i);
	    const VID * ngh = &Gcsc.getEdges()[Gcsc.getIndex()[i]];
            for( VID j=0; j < deg; ++j ) {
		VID d = ngh[j];
		++pos[d];
	    }
	}

	EID s = 0;
	for( VID v=0; v < n; ++v ) {
	    EID tmp = pos[v];
	    pos[v] = s;
	    s += tmp;
	}
	assert( s == num_edges );
	
        for( VID i=rangeLow; i < rangeHi; i++ ) {
	    VID deg = Gcsc.getDegree(i);
	    const VID * ngh = &Gcsc.getEdges()[Gcsc.getIndex()[i]];
	    const float * w
		= Gcsc.getWeights()
		? &Gcsc.getWeights()->get()[Gcsc.getIndex()[i]]
		: nullptr;
            for( VID j=0; j < deg; ++j ) {
		VID d = ngh[j];
		EID k = pos[d]++;
		src[k] = d;
		dst[k] = i;
#ifdef WEIGHTED
		wgh[k] = V[i].getInWeight( j );
#endif
		if( weights )
		    weights->get()[k] = w[j];
		k++;
	    }
	}

	pos.del();
    }


    void del() {
	src.del();
	dst.del();
	if( weights ) {
	    weights->del();
	    delete weights;
	}
    }

    void CSR_sort() {
	//assert( 0 && "NYI" );
    }

    void transpose() {
	std::swap( src, dst );
    }

public:
    VID *getSrc() { return src.get(); }
    const VID *getSrc() const { return src.get(); }
    VID *getDst() { return dst.get(); }
    const VID *getDst() const { return dst.get(); }
    float *getWeights() {
	return weights ? weights->get() : nullptr;
    }
    const float *getWeights() const {
	return weights ? weights->get() : nullptr;
    }

    void setEdge( EID e, VID s, VID d ) {
	assert( 0 <= e && e < m );
	src[e] = s;
	dst[e] = d;
    }
    void setEdge( EID e, VID s, VID d, float w ) {
	assert( 0 <= e && e < m );
	src[e] = s;
	dst[e] = d;
	weights->get()[e] = w;
    }

    VID numVertices() const { return n; }
    EID numEdges() const { return m; }

private:
    // Align assuming AVX-512
    void allocateInterleaved( bool aw = false ) {
	// src.Interleave_allocate( m, 64 );
	// dst.Interleave_allocate( m, 64 );
	src.allocate( m, 64, numa_allocation_interleaved() );
	dst.allocate( m, 64, numa_allocation_interleaved() );
	if( aw )
	    weights = new mm::buffer<float>(
		m, numa_allocation_interleaved(), "CSx weights" );
	else
	    weights = nullptr;
    }
    void allocateLocal( int numa_node, bool aw = false ) {
	// src.local_allocate( m, 64, numa_node );
	// dst.local_allocate( m, 64, numa_node );
	src.allocate( m, 64, numa_allocation_local( numa_node ) );
	dst.allocate( m, 64, numa_allocation_local( numa_node ) );
	if( aw )
	    weights = new mm::buffer<float>(
		m, numa_allocation_local( numa_node ), "CSx weights" );
	else
	    weights = nullptr;
    }
};

class GraphCOOIntlv {
    VID n;
    EID m;
    bool trp;
    mmap_ptr<VID> edges;

public:
    GraphCOOIntlv() { }
    GraphCOOIntlv( VID n_, EID m_, int allocation )
	: n( n_ ), m( m_ ), trp( false ) {
	if( allocation == -1 )
	    allocateInterleaved();
	else
	    allocateLocal( allocation );
    }
    GraphCOOIntlv( const GraphCSx & Gcsr, int allocation )
	: n( Gcsr.numVertices() ), m( Gcsr.numEdges() ), trp( false ) {
	if( allocation == -1 )
	    allocateInterleaved();
	else
	    allocateLocal( allocation );
	
        EID k = 0;
        for( VID i=0; i < n; i++ ) {
	    VID deg = Gcsr.getDegree(i);
	    const VID * ngh = &Gcsr.getEdges()[Gcsr.getIndex()[i]];
            for( VID j=0; j < deg; ++j ) {
		VID d = ngh[j];
		edges[k*2] = i;
		edges[k*2+1] = d;
#ifdef WEIGHTED
		wgh[k] = V[i].getInWeight( j );
#endif
		k++;
	    }
	}
	assert( k == m );

#if EDGES_HILBERT
	hilbert_sort();
#else
	CSR_sort();
#endif
    }
	
    GraphCOOIntlv( const GraphCSx & Gcsc,
		   const partitioner & part, int p )
	: n( Gcsc.numVertices() ), trp( false ) {
	// Range of destination vertices to include
	VID rangeLow = part.start_of(p);
	VID rangeHi = part.start_of(p+1);
	
	EID num_edges = 0;
        for( VID i=rangeLow; i < rangeHi; i++ )
	    num_edges += Gcsc.getDegree(i);

	m = num_edges;
	allocateLocal( part.numa_node_of( p ) );
	
	// Temporary info - outdegree of inverted edges - later insertion pos
	// to aid insertion in order sorted by source
	mmap_ptr<EID> pos( n, numa_allocation_interleaved() );
	std::fill( &pos[0], &pos[n], EID(0) );

        for( VID i=rangeLow; i < rangeHi; i++ ) {
	    VID deg = Gcsc.getDegree(i);
	    const VID * ngh = &Gcsc.getEdges()[Gcsc.getIndex()[i]];
            for( VID j=0; j < deg; ++j ) {
		VID d = ngh[j];
		++pos[d];
	    }
	}

	EID s = 0;
	for( VID v=0; v < n; ++v ) {
	    EID tmp = pos[v];
	    pos[v] = s;
	    s += tmp;
	}
	assert( s == num_edges );
	
        for( VID i=rangeLow; i < rangeHi; i++ ) {
	    VID deg = Gcsc.getDegree(i);
	    const VID * ngh = &Gcsc.getEdges()[Gcsc.getIndex()[i]];
            for( VID j=0; j < deg; ++j ) {
		VID d = ngh[j];
		EID k = pos[d]++;
		edges[2*k] = d;   // src
		edges[2*k+1] = i; // dst
#ifdef WEIGHTED
		wgh[k] = V[i].getInWeight( j );
#endif
		k++;
	    }
	}

	pos.del();
    }

    void del() {
	edges.del();
    }

    void CSR_sort() {
	//assert( 0 && "NYI" );
    }

    void transpose() { trp = !trp; }
    bool is_transposed() const { return trp; }

public:
    void setEdge( EID e, VID src, VID dst ) {
	assert( !trp );
	edges[2*e] = src;
	edges[2*e+1] = dst;
    }
    const VID * getEdge( EID e ) const {
	return &edges[2*e];
    }

    const VID * getEdges() const { return edges.get(); }
    VID * getEdges() { return edges.get(); }

    VID numVertices() const { return n; }
    EID numEdges() const { return m; }

private:
    // Align assuming AVX-512
    void allocateInterleaved() {
	// edges.Interleave_allocate( 2*m, 64 );
	edges.allocate( 2*m, 64, numa_allocation_interleaved() );
    }
    void allocateLocal( int numa_node ) {
	// edges.local_allocate( 2*m, 64, numa_node );
	edges.allocate( 2*m, 64, numa_allocation_local( numa_node ) );
    }
};

#endif // GRAPTOR_GRAPH_GRAPHCOO_H
