// -*- C++ -*-
#ifndef GRAPHGRIND_CGRAPH_H
#define GRAPHGRIND_CGRAPH_H

#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <algorithm>

#include "graptor/mm.h"
#include "graptor/frontier.h"

constexpr unsigned short VLUpperBound = ~(unsigned short)0;

#include "graptor/graph/EIDRemapper.h"
#include "graptor/graph/VEBOReorder.h"
#include "graptor/graph/partitioning.h"

#include "graptor/graph/CGraphCSx.h"
#include "graptor/graph/GraphCOO.h"
#include "graptor/graph/GraphGG.h"
#include "graptor/graph/GraphGGVEBO.h"
#include "graptor/graph/GraphCSx.h"
#include "graptor/graph/GraphCSR.h"


// TODO:
// - General:
//     * Are neighbour lists sorted?
//     * Remove/disable debugging code
// - For VEBO/SIMD:
//     * Consider impact of striving for longer vectors, even if not
//       executed that way. Benefits to storage size (index array CSC-like).
//       Are there benefits to COO?

template<typename GraphType>
inline bool isLowDegreeGraph( const GraphType & G ) {
    return isLowDegreeGraph( G.getCSR() );
}

template<>
inline bool isLowDegreeGraph<GraphCSx>( const GraphCSx & G ) {
    VID max_v = G.findHighestDegreeVertex(); // parallel
    VID max_deg = G.getDegree( max_v );
    return max_deg < G.numVertices() / (128*1024);
}

class GraphCSxSlice {
    EID m;
    VID n;
    VID lo, hi;
    mmap_ptr<EID> index;
    mmap_ptr<VID> edges;

public:
    GraphCSxSlice() { }
    GraphCSxSlice( VID n_, VID lo_, VID hi_, EID m_, int allocation )
	: m( m_ ), n( n_ ), lo( lo_ ), hi( hi_ ) {
	if( allocation == -1 )
	    allocateInterleaved();
	else
	    allocateLocal( allocation );
    }
    void del() {
	index.del();
	edges.del();
    }
    template<typename vertex>
    void import( const wholeGraph<vertex> & WG,
		 std::pair<const VID *, const VID *> remap ) {
	// Short-hands
	vertex *V = WG.V.get();

        EID nxt = 0;
        for( VID s=lo; s < hi; s++ ) {
	    VID sr = remap.first[s];
	    index[s-lo] = nxt;
	    VID deg = V[sr].getOutDegree();
	    for( VID i=0; i < deg; i++ ) {
		VID d = remap.second[V[sr].getOutNeighbor(i)];
		edges[nxt++] = d;
	    }
	    std::sort( &edges[index[s-lo]], &edges[nxt] );
	}
	assert( nxt == m );
	index[hi-lo] = nxt;
    }

public:
    EID *getIndex() { return index.get(); }
    const EID *getIndex() const { return index.get(); }
    VID *getEdges() { return edges.get(); }
    const VID *getEdges() const { return edges.get(); }

    VID numVertices() const { return n; }
    EID numEdges() const { return m; }

    VID getDegree( VID v ) const { return index[v+1] - index[v]; }

    VID getLowVertex() const { return lo; }
    VID getHighVertex() const { return hi; }

private:
    void allocateInterleaved() {
	index.allocate( hi-lo+1, numa_allocation_interleaved() );
	edges.allocate( m, numa_allocation_interleaved() );
    }
    void allocateLocal( int numa_node ) {
	index.allocate( hi-lo+1, numa_allocation_local( numa_node ) );
	edges.allocate( m, numa_allocation_local( numa_node ) );
    }
};

class GraphCSR : public GraphCSx {
public:
    GraphCSR( const GraphCSx & WG, int allocation )
	: GraphCSx( WG, allocation ), part( 1, WG.numVertices() ) {
	part.as_array()[0] = numVertices(); // all vertices in first partition
	part.as_array()[1] = numVertices(); // total number
	part.compute_starts();
    }

    template<typename vertex>
    GraphCSR( const wholeGraph<vertex> & WG, int allocation )
	: GraphCSx( WG, allocation ), part( 1, WG.numVertices() ) {
	part.as_array()[0] = numVertices(); // all vertices in first partition
	part.as_array()[1] = numVertices(); // total number
	part.compute_starts();
    }

    const partitioner & get_partitioner() const { return part; }

    const GraphCSx & getCSR() const { return *this; }

    VID getOutDegree( VID v ) const { return GraphCSx::getDegree( v ); }

    VID originalID( VID v ) const { return v; }
    VID remapID( VID v ) const { return v; }

    void transpose() { assert( 0 && "Not supported by GraphCSR" ); }
    bool transposed() const { return false; }

private:
    partitioner part;
};

class GraphCCSx {
    VID n;
    VID nzn;
    EID m;
    mmap_ptr<EID> index;
    mmap_ptr<VID> vid;
    mmap_ptr<VID> edges;

public:
    GraphCCSx() { }
    GraphCCSx( VID n_, VID nzn_, EID m_, int allocation )
	: n( n_ ), nzn( nzn_ ), m( m_ ) {
	if( allocation == -1 )
	    allocateInterleaved();
	else
	    allocateLocal( allocation );
    }
    void del() {
	index.del();
	vid.del();
	edges.del();
    }
    template<typename vertex>
    GraphCCSx( const wholeGraph<vertex> & WG, int allocation )
	: n( WG.n ), nzn( countNZDeg( WG ) ), m( WG.m ) {
	if( allocation == -1 )
	    allocateInterleaved();
	else
	    allocateLocal( allocation );
	import( WG );
    }
    template<typename vertex>
    void import( const wholeGraph<vertex> & WG ) {
	assert( n == WG.n && m == WG.m );
	assert( nzn == countNZDeg( WG ) );

	const vertex * V = WG.V.get();
	EID nxt = 0;
	VID vnxt = 0;
	for( VID v=0; v < n; ++v ) {
	    if( V[v].getOutDegree() > 0 ) {
		vid[vnxt] = v;
		index[vnxt] = nxt;
		for( VID j=0; j < V[v].getOutDegree(); ++j )
		    edges[nxt++] = V[v].getOutNeighbor(j);
	    }
	}
	assert( vnxt == nzn );
	index[nzn] = m;
    }

    template<typename vertex>
    GraphCCSx( const wholeGraph<vertex> & WG, int allocation,
	       std::pair<const VID *, const VID *> remap )
	: n( WG.n ), nzn( countNZDeg( WG ) ), m( WG.m ) {
	if( allocation == -1 )
	    allocateInterleaved();
	else
	    allocateLocal( allocation );
	import( WG, remap );
    }
    template<typename vertex>
    void import( const wholeGraph<vertex> & WG,
		 std::pair<const VID *, const VID *> remap ) {
	// In w = remap.first[v], v is the new vertex ID, w is the old one
	// In v = remap.second[w], v is the new vertex ID, w is the old one
	assert( n == WG.n && m == WG.m );

	const vertex * V = WG.V.get();
	EID nxt = 0;
	VID vnxt = 0;
	for( VID v=0; v < n; ++v ) {
	    VID w = remap.first[v];
	    if( V[w].getOutDegree() > 0 ) {
		vid[vnxt] = v;
		index[vnxt++] = nxt;
		for( VID j=0; j < V[w].getOutDegree(); ++j )
		    edges[nxt++] = remap.second[V[w].getOutNeighbor(j)];
		std::sort( &edges[index[vnxt-1]], &edges[nxt] );
	    }
	}
	assert( nxt == m );
	assert( vnxt == nzn );
	index[nzn] = m;
    }

    template<typename vertex>
    GraphCCSx( const wholeGraph<vertex> & WG,
	       const partitioner & part, int p,
	       std::pair<const VID *, const VID *> remap )
	: n( WG.numVertices() ) {
	// Range of destination vertices to include
	VID rangeLow = part.start_of(p);
	VID rangeHi = part.start_of(p+1);
	
	// Short-hands
	vertex *V = WG.V.get();

	// In w = remap.first[v], v is the new vertex ID, w is the old one
	// In v = remap.second[w], v is the new vertex ID, w is the old one

	// Visit all sources and count how many destinations are in the
	// range rangeLow - rangeHi after remapping.
	nzn = 0;
	m = 0;
        for( VID s=0; s < n; s++ ) {
	    VID deg = V[s].getOutDegree();
	    bool match = false;
	    for( VID i=0; i < deg; i++ ) {
		VID d = V[s].getOutNeighbor(i);
		VID dd = remap.second[d];
		if( rangeLow <= dd && dd < rangeHi ) {
		    match = true;
		    m++;
		}
	    }
	    if( match )
		++nzn;
	}

	allocateLocal( part.numa_node_of( p ) );
	
        EID nxt = 0;
	VID vnxt = 0;
        for( VID s=0; s < n; s++ ) { // traverse in order of increasing new ID
	    VID ss = remap.first[s];
	    VID deg = V[ss].getOutDegree();
	    index[vnxt] = nxt;
	    vid[vnxt] = s;
	    bool match = false;
	    for( VID i=0; i < deg; i++ ) {
		VID d = V[ss].getOutNeighbor(i);
		VID dd = remap.second[d];
		if( rangeLow <= dd && dd < rangeHi ) {
		    match = true;
		    edges[nxt++] = dd;
		}
	    }
	    if( match ) {
		std::sort( &edges[index[vnxt]], &edges[nxt] );
		vnxt++;
	    }
	}
	assert( nxt == m );
	assert( vnxt == nzn );
	index[vnxt] = nxt;
    }

    GraphCCSx( const GraphCSx & WG,
	       const partitioner & part, int p,
	       const RemapVertexIdempotent<VID> & )
	: GraphCCSx( WG, part, p, (int)part.numa_node_of( p ) ) { }

    GraphCCSx( const GraphCSx & WG,
	       const partitioner & part, int p, int allocation )
	: n( WG.numVertices() ) {
	// Range of destination vertices to include
	VID rangeLow = part.start_of(p);
	VID rangeHi = part.start_of(p+1);
	
	// Short-hands
	const EID * WG_idx = WG.getIndex();
	const VID * WG_edge = WG.getEdges();

	// Visit all sources and count how many destinations are in the
	// range rangeLow - rangeHi after remapping.
	nzn = 0;
	m = 0;
        for( VID s=0; s < n; s++ ) {
	    VID deg = WG_idx[s+1] - WG_idx[s];
	    bool match = false;
	    for( VID i=0; i < deg; i++ ) {
		VID d = WG_edge[WG_idx[s]+i];
		if( rangeLow <= d && d < rangeHi ) {
		    match = true;
		    m++;
		}
	    }
	    if( match )
		++nzn;
	}

	if( allocation == -1 )
	    allocateInterleaved();
	else
	    allocateLocal( allocation );
	
        EID nxt = 0;
	VID vnxt = 0;
        for( VID s=0; s < n; s++ ) { // traverse in order of increasing new ID
	    VID ss = s;
	    VID deg = WG_idx[ss+1] - WG_idx[ss];
	    index[vnxt] = nxt;
	    vid[vnxt] = s;
	    bool match = false;
	    for( VID i=0; i < deg; i++ ) {
		VID d = WG_edge[WG_idx[ss]+i];
		VID dd = d;
		if( rangeLow <= dd && dd < rangeHi ) {
		    match = true;
		    edges[nxt++] = dd;
		}
	    }
	    if( match ) {
		std::sort( &edges[index[vnxt]], &edges[nxt] );
		vnxt++;
	    }
	}
	assert( nxt == m );
	assert( vnxt == nzn );
	index[vnxt] = nxt;
    }

    GraphCCSx( const GraphCSx & WG,
	       const partitioner & part, int p,
	       std::pair<const VID *, const VID *> remap,
	       CSxEIDRemapper<VID,EID> * eid_remap )
	: n( WG.numVertices() ) {
	// Range of destination vertices to include
	VID rangeLow = part.start_of(p);
	VID rangeHi = part.start_of(p+1);
	
	// Short-hands
	const EID * WG_idx = WG.getIndex();
	const VID * WG_edge = WG.getEdges();

	// In w = remap.first[v], v is the new vertex ID, w is the old one
	// In v = remap.second[w], v is the new vertex ID, w is the old one

	// Visit all sources and count how many destinations are in the
	// range rangeLow - rangeHi after remapping.
	nzn = 0;
	m = 0;
        for( VID s=0; s < n; s++ ) {
	    VID deg = WG_idx[s+1] - WG_idx[s];
	    bool match = false;
	    for( VID i=0; i < deg; i++ ) {
		VID d = WG_edge[WG_idx[s]+i];
		VID dd = remap.second[d];
		if( rangeLow <= dd && dd < rangeHi ) {
		    match = true;
		    m++;
		}
	    }
	    if( match )
		++nzn;
	}

	allocateLocal( part.numa_node_of( p ) );
	
        EID nxt = 0;
	VID vnxt = 0;
        for( VID s=0; s < n; s++ ) { // traverse in order of increasing new ID
	    VID ss = remap.first[s];
	    VID deg = WG_idx[ss+1] - WG_idx[ss];
	    index[vnxt] = nxt;
	    vid[vnxt] = s;
	    bool match = false;
	    for( VID i=0; i < deg; i++ ) {
		VID d = WG_edge[WG_idx[ss]+i];
		VID dd = remap.second[d];
		if( rangeLow <= dd && dd < rangeHi ) {
		    match = true;
		    if( eid_remap )
			eid_remap->set( s, dd, nxt );
		    edges[nxt++] = dd;
		}
	    }
	    if( match ) {
		std::sort( &edges[index[vnxt]], &edges[nxt] );
		vnxt++;
	    }
	}
	assert( nxt == m );
	assert( vnxt == nzn );
	index[vnxt] = nxt;
    }
    GraphCCSx( const GraphCSx & WG,
	       const partitioner & part, int p,
	       std::pair<const VID *, const VID *> remap,
	       CSxEIDRemapper<VID,EID> & eid_remap )
	: GraphCCSx( WG, part, p, remap, &eid_remap ) { }
    GraphCCSx( const GraphCSx & WG,
	       const partitioner & part, int p,
	       std::pair<const VID *, const VID *> remap )
	: GraphCCSx( WG, part, p, remap, nullptr ) { }

    GraphCCSx( const GraphCSx & WG, const GraphCSx & csc,
	       const partitioner & part, int p,
	       const RemapVertexIdempotent<VID> & )
	: GraphCCSx( WG, csc, part, p, (int)part.numa_node_of( p ) ) { }
    
    GraphCCSx( const GraphCSx & WG, const GraphCSx & csc,
	       const partitioner & part, int p, int allocation )
	: n( WG.numVertices() ) {
	// Range of destination vertices to include
	VID rangeLow = part.start_of(p);
	VID rangeHi = part.end_of(p);
	
	// Short-hands
	const EID * WG_idx = WG.getIndex();
	const VID * WG_edge = WG.getEdges();
	const EID * csc_idx = csc.getIndex();
	const VID * csc_edge = csc.getEdges();

	mmap_ptr<VID> present( n, numa_allocation_local(part.numa_node_of(p)) );
	mmap_ptr<VID> tmp( n, numa_allocation_local(part.numa_node_of(p)) );

	for( VID vv=0; vv < n; vv++ ) {
	    present[vv] = 0;
	    tmp[vv] = 0;
	}

	// Analyse graph
	nzn = 0;
	m = 0;

	// Lazily copied and minimal adjustments. Room for optimization!!!
	for( VID vv=rangeLow; vv < rangeHi; vv++ ) {
	    VID v = vv;
	    VID deg = csc_idx[v+1] - csc_idx[v];
	    m += deg;

	    for( VID i=0; i < deg; i++ ) {
		VID d = csc_edge[csc_idx[v]+i];
		VID dd = d;
		if( present[dd] == 0 )
		    ++nzn;
		++present[dd];
	    }
	}
	    
	allocateLocal( part.numa_node_of( p ) );

	VID vnxt = 0;
	VID sum = 0;
	for( VID v=0; v < n; ++v ) {
	    if( present[v] != 0 ) {
		VID deg = present[v];
		tmp[vnxt] = sum;
		index[vnxt] = sum;
		vid[vnxt] = v;
		sum += deg;
		present[v] = vnxt++;
	    }
	}
	tmp[vnxt] = m;
	index[vnxt] = m;

	assert( sum == m );
	assert( vnxt == nzn );

	// traverse in order of increasing new ID
        for( VID s=rangeLow; s < rangeHi; s++ ) {
	    VID ss = s;
	    VID deg = csc_idx[ss+1] - csc_idx[ss];
	    bool match = false;
	    for( VID i=0; i < deg; i++ ) {
		VID d = csc_edge[csc_idx[ss]+i];
		VID dd = d;
		VID cdd = present[dd];
		edges[tmp[cdd]++] = s;
	    }
	}
        for( VID s=0; s < nzn; s++ ) { // traverse in order of increasing new ID
	    assert( tmp[s] == index[s+1] );
	    std::sort( &edges[index[s]], &edges[index[s]+1] );
	}
	tmp.del();
	present.del();
    }
	
    GraphCCSx( const GraphCSx & WG, const GraphCSx & csc,
	       const partitioner & part, int p,
	       std::pair<const VID *, const VID *> remap,
	       CSxEIDRemapper<VID,EID> * eid_remap = nullptr )
	: n( WG.numVertices() ) {
	// Range of destination vertices to include
	VID rangeLow = part.start_of(p);
	VID rangeHi = part.end_of(p);

	// TODO: This should be optimised by generating all partitions in
	//       one sweep.

	// Short-hands
	const EID * WG_idx = WG.getIndex();
	const VID * WG_edge = WG.getEdges();
	const EID * csc_idx = csc.getIndex();
	const VID * csc_edge = csc.getEdges();

	// In w = remap.first[v], v is the new vertex ID, w is the old one
	// In v = remap.second[w], v is the new vertex ID, w is the old one

	mmap_ptr<VID> present( n, numa_allocation_local(part.numa_node_of(p)) );
	mmap_ptr<VID> tmp( n, numa_allocation_local(part.numa_node_of(p)) );

	// Analyse graph
	nzn = 0;
	m = 0;

	for( VID vv=rangeLow; vv < rangeHi; vv++ ) {
	    VID v = remap.first[vv];
	    VID deg = csc_idx[v+1] - csc_idx[v];
	    m += deg;

	    for( VID i=0; i < deg; i++ ) {
		VID d = csc_edge[csc_idx[v]+i];
		VID dd = remap.second[d];
		if( present[dd] == 0 )
		    ++nzn;
		++present[dd];
	    }
	}

	allocateLocal( part.numa_node_of( p ) );

	VID vnxt = 0;
	VID sum = 0;
	for( VID v=0; v < n; ++v ) {
	    if( present[v] != 0 ) {
		VID deg = present[v];
		tmp[vnxt] = sum;
		index[vnxt] = sum;
		vid[vnxt] = v;
		sum += deg;
		present[v] = vnxt++;
	    }
	}
	tmp[vnxt] = m;
	index[vnxt] = m;

	assert( sum == m );
	assert( vnxt == nzn );

	// traverse in order of increasing new ID
        for( VID s=rangeLow; s < rangeHi; s++ ) {
	    VID ss = remap.first[s];
	    VID deg = csc_idx[ss+1] - csc_idx[ss];
	    bool match = false;
	    for( VID i=0; i < deg; i++ ) {
		VID d = csc_edge[csc_idx[ss]+i];
		VID dd = remap.second[d];
		VID cdd = present[dd];
		edges[tmp[cdd]++] = s;
	    }
	}

        for( VID z=0; z < nzn; z++ ) { // traverse in order of increasing new ID
	    VID s = vid[z];
	    assert( tmp[z] == index[z+1] );
	    std::sort( &edges[index[z]], &edges[index[z+1]] );
	    if( eid_remap ) {
		for( EID j=index[z]; j < index[z+1]; ++j )
		    eid_remap->set( s, edges[j], j );
	    }
	}

	tmp.del();
	present.del();
    }

    GraphCCSx( const GraphCSx & WG, const GraphCSx & csc,
	       const partitioner & part, int p,
	       std::pair<const VID *, const VID *> remap,
	       CSxEIDRemapper<VID,EID> & eid_remap )
	: GraphCCSx( WG, csc, part, p, remap, &eid_remap ) { }

    template<typename vertex>
    GraphCCSx( const wholeGraph<vertex> & WG,
	       const partitioner & part, int p, int allocation )
	: n( WG.numVertices() ) {
	// Range of destination vertices to include
	VID rangeLow = part.start_of(p);
	VID rangeHi = part.start_of(p+1);
	
	// Short-hands
	vertex *V = WG.V.get();

	nzn = 0;
        for( VID s=0; s < n; s++ ) {
	    VID deg = V[s].getOutDegree();
	    bool match = false;
	    for( VID i=0; i < deg; i++ ) {
		VID d = V[s].getOutNeighbor(i);
		if( rangeLow <= d && d < rangeHi ) {
		    match = true;
		    break;
		}
	    }
	    if( match )
		++nzn;
	}

	m = 0;
        for( VID i=rangeLow; i < rangeHi; i++ )
	    m += V[i].getInDegree();

	if( allocation == -1 )
	    allocateInterleaved();
	else
	    allocateLocal( allocation );
	
        EID nxt = 0;
	VID vnxt = 0;
        for( VID s=0; s < n; s++ ) {
	    index[vnxt] = nxt;
	    vid[vnxt] = nxt;
	    VID deg = V[s].getOutDegree();
	    bool match = false;
	    for( VID i=0; i < deg; i++ ) {
		VID d = V[s].getOutNeighbor(i);
		if( rangeLow <= d && d < rangeHi ) {
		    match = true;
		    edges[nxt++] = d;
		}
	    }
	    if( match )
		vnxt++;
	}
	assert( nxt == m );
	assert( vnxt == nzn );
	index[vnxt] = nxt;
    }

public:
    class edge_iterator {
    public:
	edge_iterator( VID zid, EID eid, const GraphCCSx * graph )
	    : m_zid( zid ), m_eid( eid ), m_graph( graph ) { }
	edge_iterator( const edge_iterator & it )
	    : m_zid( it.m_zid ), m_eid( it.m_eid ), m_graph( it.m_graph ) { }

	edge_iterator & operator = ( const edge_iterator & it ) {
	    m_zid = it.m_zid;
	    m_eid = it.m_eid;
	    m_graph = it.m_graph;
	    return *this;
	}

	std::pair<VID,VID> operator * () const {
	    return std::make_pair( m_graph->getVertexID()[m_zid],
				   m_graph->getEdges()[m_eid] );
	}

	edge_iterator & operator ++ () {
	    ++m_eid;
	    if( m_eid == m_graph->getIndex()[m_zid+1] )
		++m_zid;
	    return *this;
	}

	edge_iterator operator ++ ( int ) {
	    edge_iterator cp( *this );
	    ++*this;
	    return cp;
	}

	bool operator == ( edge_iterator it ) const {
	    return m_zid == it.m_zid && m_eid == it.m_eid
		&& m_graph == it.m_graph;
	}
	bool operator != ( edge_iterator it ) const {
	    return !( *this == it );
	}

    private:
	VID m_zid;
	EID m_eid;
	const GraphCCSx * m_graph;
    };

    edge_iterator edge_begin() const {
	return edge_iterator( 0, 0, this );
    }
    edge_iterator edge_end() const {
	return edge_iterator( nzn, m, this );
    }

public:
    class vertex_iterator {
    public:
	vertex_iterator( VID zid, const GraphCCSx * graph )
	    : m_zid( zid ), m_graph( graph ) { }
	vertex_iterator( const vertex_iterator & it )
	    : m_zid( it.m_zid ), m_graph( it.m_graph ) { }

	vertex_iterator & operator = ( const vertex_iterator & it ) {
	    m_zid = it.m_zid;
	    m_graph = it.m_graph;
	    return *this;
	}

	VertexInfo operator * () const {
	    VID idx = m_graph->getIndex()[m_zid];
	    return VertexInfo( m_graph->getVertexID()[m_zid],
			       m_graph->getIndex()[m_zid+1] - idx,
			       &m_graph->getEdges()[idx] );
	}

	vertex_iterator & operator ++ () {
	    ++m_zid;
	    return *this;
	}

	vertex_iterator operator ++ ( int ) {
	    vertex_iterator cp( *this );
	    ++*this;
	    return cp;
	}

	bool operator == ( vertex_iterator it ) const {
	    return m_zid == it.m_zid && m_graph == it.m_graph;
	}
	bool operator != ( vertex_iterator it ) const {
	    return !( *this == it );
	}

    private:
	VID m_zid;
	const GraphCCSx * m_graph;
    };

    vertex_iterator vertex_begin() const {
	return vertex_iterator( 0, this );
    }
    vertex_iterator vertex_end() const {
	return vertex_iterator( nzn, this );
    }

public:
    EID *getIndex() { return index.get(); }
    const EID *getIndex() const { return index.get(); }
    VID *getVertexID() { return vid.get(); }
    const VID *getVertexID() const { return vid.get(); }
    VID *getEdges() { return edges.get(); }
    const VID *getEdges() const { return edges.get(); }

    VID numVertices() const { return n; }
    VID numNZDegVertices() const { return nzn; }
    EID numEdges() const { return m; }

    // This is harder to get; need to be conscious about calling this.
    // VID getDegree( VID v ) const { }

private:
    void allocateInterleaved() {
	// index.Interleave_allocate( nzn+1 );
	// vid.Interleave_allocate( nzn );
	// edges.Interleave_allocate( m );
	index.allocate( nzn+1, numa_allocation_interleaved() );
	vid.allocate( nzn, numa_allocation_interleaved() );
	edges.allocate( m, numa_allocation_interleaved() );
    }
    void allocateLocal( int numa_node ) {
	// index.local_allocate( nzn+1, numa_node );
	// vid.local_allocate( nzn, numa_node );
	// edges.local_allocate( m, numa_node );
	index.allocate( nzn+1, numa_allocation_local( numa_node ) );
	vid.allocate( nzn, numa_allocation_local( numa_node ) );
	edges.allocate( m, numa_allocation_local( numa_node ) );
    }
    template<typename vertex>
    static VID countNZDeg( const wholeGraph<vertex> & WG ) {
	VID nzn = 0;
	VID n = WG.n;
	const vertex * V = WG.V.get();
	for( VID v=0; v < n; ++v ) {
	    if( V[v].getOutDegree() > 0 )
		++nzn;
	}
	return nzn;
    }
};

class GraphCCSxParts {
    GraphCCSx m_ccsx;
    VID * m_startnz;
    EID * m_starte;
    partitioner m_part;

public:
    GraphCCSxParts() { }
    GraphCCSxParts( const GraphCSx & WG, const GraphCSx & csc,
		    const partitioner & part, int rd_part,
		    const RemapVertexIdempotent<VID> & r )
	: m_ccsx( WG, csc, part, rd_part, r ),
	  m_part( part ) {
	unsigned int npart = part.get_num_partitions();
	m_startnz = new VID[npart+1];
	m_starte = new EID[npart+1];

	unsigned int p = 0;
	unsigned int plim = part.start_of( p+1 );
	const EID * idx = m_ccsx.getIndex();
	const VID * vid = m_ccsx.getVertexID();
	VID nnz = m_ccsx.numNZDegVertices();
	m_startnz[(p+npart-1-rd_part) % npart] = 0;
	m_starte[(p+npart-1-rd_part) % npart] = 0;
	for( VID i=0; i < nnz; ++i ) {
	    VID u = vid[i];
	    while( u >= plim ) {
		++p;

		m_startnz[(p+npart-1-rd_part) % npart] = i;
		m_starte[(p+npart-1-rd_part) % npart] = idx[i];


		plim = part.start_of( p+1 );
	    }
	}
	while( p < npart ) {
	    ++p;
	    m_startnz[(p+npart-1-rd_part) % npart] = nnz;
	    m_starte[(p+npart-1-rd_part) % npart] = m_ccsx.numEdges();
	}

	m_startnz[npart] = m_startnz[0] == 0 ? nnz : m_startnz[0];
	m_starte[npart] = m_starte[0] == 0 ? m_ccsx.numEdges() : m_starte[0];
    }

    void del() {
	m_ccsx.del();
	delete[] m_startnz;
	delete[] m_starte;
    }

    const partitioner & get_partitioner() const { return m_part; }

    EID *getIndex() { return m_ccsx.getIndex(); }
    const EID *getIndex() const { return m_ccsx.getIndex(); }
    VID *getVertexID() { return m_ccsx.getVertexID(); }
    const VID *getVertexID() const { return m_ccsx.getVertexID(); }
    VID *getEdges() { return m_ccsx.getEdges(); }
    const VID *getEdges() const { return m_ccsx.getEdges(); }

    VID numVertices() const { return m_ccsx.numVertices(); }
    VID numNZDegVertices() const { return m_ccsx.numNZDegVertices(); }
    EID numEdges() const { return m_ccsx.numEdges(); }

    VID * getPartStartNZ() { return m_startnz; }
    const VID * getPartStartNZ() const { return m_startnz; }
    EID * getPartStartEdge() { return m_starte; }
    const EID * getPartStartEdge() const { return m_starte; }
};

class GraphPartCSR {
    GraphCSx csr; // only for calculating statistics (to be removed)
    GraphCSx * csrp;
    partitioner part;
    
public:
    template<typename vertex>
    GraphPartCSR( const wholeGraph<vertex> & WG, int npart,
		  bool balance_vertices )
	: csr( WG, -1 ),
	  csrp( new GraphCSx[npart] ),
	  part( npart, WG.numVertices() ) {
	// Decide what partition each vertex goes into
	if( balance_vertices )
	    partitionBalanceDestinations( WG, part ); 
	else
	    partitionBalanceEdges( WG, part ); 

	// Create COO partitions in parallel
	map_partitionL( part, [&]( int p ) {
		GraphCSx & pcsr = csrp[p];
		int node = part.numa_node_of( p );
		new ( &pcsr ) GraphCSx( WG, part, p, node );
	    } );
    }

    void del() {
	csr.del();
	for( int p=0; p < part.get_num_partitions(); ++p )
	    csrp[p].del();
	delete[] csrp;
	csrp = nullptr;
    }
public:
    void fragmentation() const {
	std::cerr << "GraphPartCSR:\ntotal-size: todo\n";
    }
    
public:
    VID numVertices() const { return csr.numVertices(); }
    EID numEdges() const { return csr.numEdges(); }

    const partitioner & get_partitioner() const { return part; }

    const GraphCSx & getCSR() const { return csr; }
    const GraphCSx & getPartition( int p ) const { return csrp[p]; }

    VID getOutDegree( VID v ) const { return getCSR().getDegree( v ); }

    VID originalID( VID v ) const { return v; }
    VID remapID( VID v ) const { return v; }

    void transpose() { assert( 0 && "Not supported by GraphPartCSR" ); }
    bool transposed() const { return false; }

    // Push-style
    static constexpr bool getRndRd() { return false; }
    static constexpr bool getRndWr() { return true; }
};


class GraphVEBOPartCSR {
    GraphCSx csr; // only for calculating statistics (to be removed)
    GraphCSx * csrp;
    partitioner part;
    VEBOReorder remap;
    
public:
    template<typename vertex>
    GraphVEBOPartCSR( const wholeGraph<vertex> & WG, int npart )
	: csr( WG.n, WG.m, -1 ),
	  csrp( new GraphCSx[npart] ),
	  part( npart, WG.numVertices() ) {

	std::cerr << "VEBOPartCSR: "
		  << " n=" << WG.numVertices()
		  << " e=" << WG.numEdges()
		  << "\n";

	// We require a temporary CSC
	GraphCSx csc( WG.n, WG.m, -1 );
	if( std::is_same<vertex,symmetricVertex>::value )
	    csc.import( WG );
	else {
	    wholeGraph<vertex> * WGc = const_cast<wholeGraph<vertex> *>( &WG );
	    WGc->transpose();
	    csc.import( WG );
	    WGc->transpose();
	}

	// Calculate remapping table
	remap = VEBOReorder( csc, part );

	// Setup CSR
	csr.import( WG, remap.maps() );

	// Create COO partitions in parallel
	map_partitionL( part, [&]( int p ){
		GraphCSx & pcsr = csrp[p];
		new ( &pcsr ) GraphCSx( WG, part, p, remap.maps() );

		std::cerr << "VEBOPartCSR part " << p
			  << " s=" << part.start_of(p)
			  << " e=" << part.end_of(p)
			  << " nv=" << pcsr.numVertices()
			  << " ne=" << pcsr.numEdges()
			  << "\n";
	    } );

	// Cleanup temporary data structures
	csc.del();
    }
    GraphVEBOPartCSR( const GraphCSx & WG, int npart )
	: csr( WG.numVertices(), WG.numEdges(), -1 ),
	  csrp( new GraphCSx[npart] ),
	  part( npart, WG.numVertices() ) {

	std::cerr << "VEBOPartCSR: "
		  << " n=" << WG.numVertices()
		  << " e=" << WG.numEdges()
		  << "\n";

	// Setup temporary CSC, try to be space-efficient
	std::cerr << "Transposing CSR...\n";
	GraphCSx csc( WG.numVertices(), WG.numEdges(), -1 );
	const GraphCSx * csc_tmp_ptr = &csc;
	if( WG.isSymmetric() )
	    csc_tmp_ptr = &WG;
	else
	    csc.import_transpose( WG );
	const GraphCSx & csc_tmp = *csc_tmp_ptr;

	// Calculate remapping table
	remap = VEBOReorder( csc_tmp, part );

	// Setup unpartitioned CSR
	csr.import( WG, remap.maps() );

	// Create CSR partitions in parallel
	map_partitionL( part, [&]( int p ){
		GraphCSx & pcsr = csrp[p];
		new ( &pcsr ) GraphCSx( WG, part, p, remap.maps() );

		std::cerr << "VEBOPartCSR part " << p
			  << " s=" << part.start_of(p)
			  << " e=" << part.end_of(p)
			  << " nv=" << pcsr.numVertices()
			  << " ne=" << pcsr.numEdges()
			  << "\n";
	    } );

	// Cleanup temporary data structures
	csc.del();
    }

    void del() {
	remap.del();
	csr.del();
	for( int p=0; p < part.get_num_partitions(); ++p )
	    csrp[p].del();
	delete[] csrp;
	csrp = nullptr;
    }

    void fragmentation() const {
	std::cerr << "GraphVEBOPartCSR:\ntotal-size: todo\n";
    }

public:
    VID numVertices() const { return csr.numVertices(); }
    EID numEdges() const { return csr.numEdges(); }

    const partitioner & get_partitioner() const { return part; }
    auto get_remapper() const { return remap.remapper(); }

    const GraphCSx & getCSR() const { return csr; }
    const GraphCSx & getPartition( int p ) const { return csrp[p]; }

    VID getOutDegree( VID v ) const { return getCSR().getDegree( v ); }

    VID originalID( VID v ) const { return remap.originalID( v ); }
    VID remapID( VID v ) const { return remap.remapID( v ); }

    void transpose() { assert( 0 && "Not supported by GraphPartCSR" ); }
    bool transposed() const { return false; }

    static constexpr unsigned short getMaxVLCOO() { return 1; }
    static constexpr unsigned short getMaxVLCSC() { return 1; }
    static constexpr unsigned short getMaxVLCSR() { return 1; }
    static constexpr unsigned short getVLCOOBound() { return 1; }
    static constexpr unsigned short getVLCSCBound() { return 1; }

    // Push-style
    static constexpr bool getRndRd() { return false; }
    static constexpr bool getRndWr() { return true; }
};

class GraphVEBOPartCCSR {
public:
    using EIDRetriever = CSxEIDRetriever<VID,EID>;

private:
    GraphCSx csr; // only for calculating statistics (to be removed)
    GraphCCSx * csrp;
    partitioner part;
    VEBOReorder remap;
    EIDRetriever eid_retriever;
    
public:
    using PID = unsigned int;
    static constexpr bool is_ccsr = true;

public:
    template<typename vertex>
    GraphVEBOPartCCSR( const wholeGraph<vertex> & WG, int npart )
	: csr( WG.n, WG.m, -1, WG.isSymmetric() ),
	  csrp( new GraphCCSx[npart] ),
	  part( npart, WG.numVertices() ) {

#if OWNER_READS
	assert( WG.isSymmetric() && "OWNER_READS requires symmetric graphs" );
#endif

	// We require a temporary CSC
	GraphCSx csc( WG.n, WG.m, -1 );
	if( std::is_same<vertex,symmetricVertex>::value )
	    csc.import( WG );
	else {
	    wholeGraph<vertex> * WGc = const_cast<wholeGraph<vertex> *>( &WG );
	    WGc->transpose();
	    csc.import( WG );
	    WGc->transpose();
	}

	// Calculate remapping table
	remap = VEBOReorder( csc, part );

	// Setup CSR
	csr.import( WG, remap.maps() );

	// Setup EID remapper based on remapped (padded) vertices
	CSxEIDRemapper<VID,EID> eid_remapper( csr );

	// Create COO partitions in parallel
	map_partitionL( part, [&]( int p ){
		GraphCCSx & pcsr = csrp[p];
		new ( &pcsr ) GraphCCSx( WG, part, p, remap.maps(),
					 eid_remapper );
	    } );

	eid_remapper.finalize( part );
	eid_retriever = eid_remapper.create_retriever();

	// Cleanup temporary data structures
	csc.del();
    }

    GraphVEBOPartCCSR( const GraphCSx & WG, int npart )
	: csr( WG.numVertices(), WG.numEdges(), -1, WG.isSymmetric() ),
	  csrp( new GraphCCSx[npart] ),
	  part( npart, WG.numVertices() ) {

#if OWNER_READS
	assert( WG.isSymmetric() && "OWNER_READS requires symmetric graphs" );
#endif

	std::cerr << "VEBOPartCCSR: "
		  << " n=" << WG.numVertices()
		  << " e=" << WG.numEdges()
		  << "\n";

	// Setup temporary CSC, try to be space-efficient
	timer tm;
	tm.start();
	GraphCSx csc( WG.numVertices(), WG.numEdges(), -1 );
	const GraphCSx * csc_tmp_ptr = &csc;
	if( WG.isSymmetric() )
	    csc_tmp_ptr = &WG;
	else
	    csc.import_transpose( WG );
	const GraphCSx & csc_tmp = *csc_tmp_ptr;

	std::cerr << "Transposing CSR: " << tm.next() << "\n";

	// Calculate remapping table
	remap = VEBOReorder( csc_tmp, part );
	std::cerr << "VEBOReorder: " << tm.next() << "\n";

	// Setup CSR
	csr.import( WG, remap.maps() );
	std::cerr << "Remap CSR (sparse): " << tm.next() << "\n";

	// Setup EID remapper based on remapped (padded) vertices
	CSxEIDRemapper<VID,EID> eid_remapper( csr );
	std::cerr << "EID Remapper setup: " << tm.next() << "\n";

	// Create CCSR partitions in parallel
	map_partitionL( part, [&]( int p ){
		GraphCCSx & pcsr = csrp[p];
		new ( &pcsr ) GraphCCSx( WG, csc_tmp, part, p, remap.maps(),
					 eid_remapper );
					 
	    } );
	std::cerr << "Create CCSR partitions: " << tm.next() << "\n";

	// Skip....
	// eid_remapper.finalize( part );
	std::cerr
	    << "************************************************************\n"
	    << "WARNING: Skipping EID remapper finalization - too expensive\n"
	    << "************************************************************\n";
	eid_retriever = eid_remapper.create_retriever();
	std::cerr << "EID Remapper finalise: " << tm.next() << "\n";

	EID tot_edges = 0;
	for( int p=0; p < part.get_num_partitions(); ++p ) {
	    GraphCCSx & pcsr = csrp[p];
	    std::cerr << "VEBOPartCCSR part " << p
		      << " s=" << part.start_of(p)
		      << " e=" << part.end_of(p)
		      << " nv=" << pcsr.numVertices()
		      << " ne=" << pcsr.numEdges()
		      << "\n";
	    tot_edges += pcsr.numEdges();
	}
	assert( tot_edges == WG.numEdges() );
	assert( tot_edges == csr.numEdges() );
	assert( tot_edges == numEdges() );

	// Cleanup temporary data structures
	csc.del();

	// Check correctness
	// graph_compare( *this, csr );
    }

    void del() {
	remap.del();
	csr.del();
	for( int p=0; p < part.get_num_partitions(); ++p )
	    csrp[p].del();
	delete[] csrp;
	csrp = nullptr;
    }

    void fragmentation() const {
	std::cerr << "GraphVEBOPartCCSR:\ntotal-size: todo\n";
    }

public:
    class edge_iterator {
    public:
	edge_iterator( PID pid, PID npart, GraphCCSx::edge_iterator pit,
		       GraphCCSx * parts )
	    : m_pid( pid ), m_npart( npart ), m_pit( pit ), m_parts( parts )
	    { }
	edge_iterator( const edge_iterator & it )
	    : m_pid( it.m_pid ), m_npart( it.m_npart ), m_pit( it.m_pit ),
	      m_parts( it.m_parts ) { }

	edge_iterator & operator = ( const edge_iterator & it ) {
	    m_pid = it.m_pid;
	    m_npart = it.m_npart;
	    m_pit = it.m_pit;
	    m_parts = it.m_parts;
	    return *this;
	}

	std::pair<VID,VID> operator * () const {
	    return *m_pit;
	}

	edge_iterator & operator ++ () {
	    ++m_pit;
	    if( m_pit == m_parts[m_pid].edge_end() ) {
		if( m_pid+1 < m_npart ) {
		    ++m_pid;
		    m_pit = m_parts[m_pid].edge_begin();
		}
	    }
	    return *this;
	}

	edge_iterator operator ++ ( int ) {
	    edge_iterator cp( *this );
	    ++*this;
	    return cp;
	}

	bool operator == ( edge_iterator it ) const {
	    return m_pid == it.m_pid && m_npart == it.m_npart
		&& m_pit == it.m_pit && m_parts == it.m_parts;
	}
	bool operator != ( edge_iterator it ) const {
	    return !( *this == it );
	}

    private:
	PID m_pid;
	PID m_npart;
	GraphCCSx::edge_iterator m_pit;
	GraphCCSx * m_parts;
    };

    edge_iterator edge_begin() const {
	return edge_iterator( 0, part.get_num_partitions(),
			      csrp[0].edge_begin(), csrp );
    }
    edge_iterator edge_end() const {
	return edge_iterator( part.get_num_partitions()-1,
			      part.get_num_partitions(),
			      csrp[part.get_num_partitions()-1].edge_end(),
			      csrp );
    }

public:
    GraphCCSx::vertex_iterator part_vertex_begin( const partitioner & part, PID p ) const {
	return csrp[p].vertex_begin();
    }
    GraphCCSx::vertex_iterator part_vertex_end( const partitioner & part, PID p ) const {
	return csrp[p].vertex_end();
    }

public:
    VID numVertices() const { return csr.numVertices(); }
    EID numEdges() const { return csr.numEdges(); }

    bool isSymmetric() const { return csr.isSymmetric(); }

    const partitioner & get_partitioner() const { return part; }
    const EIDRetriever & get_eid_retriever() const { return eid_retriever; }

    auto get_remapper() const { return remap.remapper(); }

    const GraphCSx & getCSR() const { return csr; }
    const GraphCCSx & getPartition( int p ) const { return csrp[p]; }

    VID originalID( VID v ) const { return remap.originalID( v ); }
    VID remapID( VID v ) const { return remap.remapID( v ); }

    VID getOutDegree( VID v ) const { return getCSR().getDegree( v ); }
    const VID * getOutDegree() const { return getCSR().getDegree(); }

    void transpose() { assert( 0 && "Not supported by GraphVEBOPartCCSR" ); }
    bool transposed() const { return false; }

    static constexpr unsigned short getMaxVLCOO() { return 1; }
    static constexpr unsigned short getMaxVLCSC() { return 1; }
    static constexpr unsigned short getVLCOOBound() { return 1; }
    static constexpr unsigned short getVLCSCBound() { return 1; }
    static constexpr unsigned short getPullVLBound() { return 1; }
    static constexpr unsigned short getPushVLBound() { return 1; }
    static constexpr unsigned short getIRegVLBound() { return 1; }

    // Push-style
    static constexpr bool getRndRd() { return false; }
    static constexpr bool getRndWr() { return true; }

    graph_traversal_kind select_traversal(
	bool fsrc_strong,
	bool fdst_strong,
	bool adst_strong,
	bool record_strong,
	frontier F,
	bool is_sparse ) const {
#if OWNER_READS
	return graph_traversal_kind::gt_pull;
#else
	return graph_traversal_kind::gt_push;
#endif
    }

    static constexpr bool is_privatized( graph_traversal_kind gtk ) {
	return gtk == graph_traversal_kind::gt_push;
    }
};

// This class should become a template for Graph(VEBO)Part{CSR,CCSR}
template<typename ReorderTy, typename GraphPartTy>
class GraphPart_tmpl {
public:
    using reorder_type = ReorderTy;
    using graph_part_type = GraphPartTy;
    using PID = unsigned int;
    
public:
    GraphPart_tmpl( const GraphCSx & WG, int npart )
	: csr( WG.numVertices(), WG.numEdges(), -1 ),
	  csrp( new graph_part_type[npart] ),
	  part( npart, WG.numVertices() ) {

	std::cerr << "GraphPart_tmpl: "
		  << " n=" << WG.numVertices()
		  << " e=" << WG.numEdges()
		  << "\n";

	// Setup temporary CSC, try to be space-efficient
	timer tm;
	tm.start();
	GraphCSx csc( WG.numVertices(), WG.numEdges(), -1 );
	const GraphCSx * csc_tmp_ptr = &csc;
	if( WG.isSymmetric() )
	    csc_tmp_ptr = &WG;
	else
	    csc.import_transpose( WG );
	const GraphCSx & csc_tmp = *csc_tmp_ptr;

	std::cerr << "Transposing CSR: " << tm.next() << "\n";

	// Calculate remapping table
	remap = reorder_type( csc_tmp, part );
	std::cerr << "Reorder: " << tm.next() << "\n";

	// Setup CSR
	csr.import( WG, remap.remapper() );
	std::cerr << "Remap CSR (sparse): " << tm.next() << "\n";

	// Create CCSR partitions in parallel
	map_partitionL( part, [&]( int p ){
		graph_part_type & pcsr = csrp[p];
		new ( &pcsr ) graph_part_type( WG, csc_tmp, part, p, remap.remapper() );
					 
	    } );
	std::cerr << "Create CCSR partitions: " << tm.next() << "\n";

	EID tot_edges = 0;
	for( int p=0; p < part.get_num_partitions(); ++p ) {
	    graph_part_type & pcsr = csrp[p];
	    std::cerr << "VEBOPartCCSR part " << p
		      << " s=" << part.start_of(p)
		      << " e=" << part.end_of(p)
		      << " nv=" << pcsr.numVertices()
		      << " ne=" << pcsr.numEdges();
	    if constexpr ( std::is_same<graph_part_type,GraphCCSx>::value )
		  std::cerr << " nnz=" << pcsr.numNZDegVertices();
	    std::cerr << "\n";
	    tot_edges += pcsr.numEdges();
	}
	assert( tot_edges == WG.numEdges() );
	assert( tot_edges == csr.numEdges() );
	assert( tot_edges == numEdges() );

	// Cleanup temporary data structures
	csc.del();

	// Check correctness
	// graph_compare( *this, csr );
    }

    void del() {
	remap.del();
	csr.del();
	for( int p=0; p < part.get_num_partitions(); ++p )
	    csrp[p].del();
	delete[] csrp;
	csrp = nullptr;
    }

    void fragmentation() const {
	std::cerr << "GraphPart_tmpl:\ntotal-size: todo\n";
    }

    class edge_iterator {
    public:
	edge_iterator( PID pid, PID npart, GraphCCSx::edge_iterator pit,
		       graph_part_type * parts )
	    : m_pid( pid ), m_npart( npart ), m_pit( pit ), m_parts( parts )
	    { }
	edge_iterator( const edge_iterator & it )
	    : m_pid( it.m_pid ), m_npart( it.m_npart ), m_pit( it.m_pit ),
	      m_parts( it.m_parts ) { }

	edge_iterator & operator = ( const edge_iterator & it ) {
	    m_pid = it.m_pid;
	    m_npart = it.m_npart;
	    m_pit = it.m_pit;
	    m_parts = it.m_parts;
	    return *this;
	}

	std::pair<VID,VID> operator * () const {
	    return *m_pit;
	}

	edge_iterator & operator ++ () {
	    ++m_pit;
	    if( m_pit == m_parts[m_pid].edge_end() ) {
		if( m_pid+1 < m_npart ) {
		    ++m_pid;
		    m_pit = m_parts[m_pid].edge_begin();
		}
	    }
	    return *this;
	}

	edge_iterator operator ++ ( int ) {
	    edge_iterator cp( *this );
	    ++*this;
	    return cp;
	}

	bool operator == ( edge_iterator it ) const {
	    return m_pid == it.m_pid && m_npart == it.m_npart
		&& m_pit == it.m_pit && m_parts == it.m_parts;
	}
	bool operator != ( edge_iterator it ) const {
	    return !( *this == it );
	}

    private:
	PID m_pid;
	PID m_npart;
	typename graph_part_type::edge_iterator m_pit;
	graph_part_type * m_parts;
    };

    edge_iterator edge_begin() const {
	return edge_iterator( 0, part.get_num_partitions(),
			      csrp[0].edge_begin(), csrp );
    }
    edge_iterator edge_end() const {
	return edge_iterator( part.get_num_partitions()-1,
			      part.get_num_partitions(),
			      csrp[part.get_num_partitions()-1].edge_end(),
			      csrp );
    }


public:
    VID numVertices() const { return csr.numVertices(); }
    EID numEdges() const { return csr.numEdges(); }

    const partitioner & get_partitioner() const { return part; }

    auto get_remapper() const { return remap.remapper(); }

    const GraphCSx & getCSR() const { return csr; }
    const graph_part_type & getPartition( int p ) const { return csrp[p]; }

    VID originalID( VID v ) const { return remap.originalID( v ); }
    VID remapID( VID v ) const { return remap.remapID( v ); }

    VID getOutDegree( VID v ) const { return getCSR().getDegree( v ); }

    void transpose() { assert( 0 && "Not supported by GraphVEBOPartCCSR" ); }
    bool transposed() const { return false; }

    static constexpr unsigned short getMaxVLCOO() { return 1; }
    static constexpr unsigned short getMaxVLCSC() { return 1; }
    static constexpr unsigned short getVLCOOBound() { return 1; }
    static constexpr unsigned short getVLCSCBound() { return 1; }

    // Push-style
    static constexpr bool getRndRd() { return false; }
    static constexpr bool getRndWr() { return true; }

private:
    GraphCSx csr; // only for calculating statistics (to be removed)
    graph_part_type * csrp;
    partitioner part;
    reorder_type remap;
};

class GraphCSxSIMD {
    VID n, nv;
    EID m, mv;
    unsigned short maxVL;
    mmap_ptr<EID> index;
    mmap_ptr<EID> mindex;
    mmap_ptr<VID> edges;

public:
    GraphCSxSIMD() { }
    void del() {
	mindex.del();
	index.del();
	edges.del();
    }
    template<typename vertex>
    void import( const wholeGraph<vertex> & WG,
		 VID lo, VID hi, unsigned short maxVL_,
		 std::pair<const VID *, const VID *> remap,
		 int allocation ) {
	// This is written to be a CSC-style import, i.e., WG is transposed,
	// lo/hi applied to 'sources'.
	// Calculate dimensions of SIMD representation
	maxVL = maxVL_;
	n = WG.numVertices();
	// m = WG.numEdges();
	m = 0;
	for( nv=lo; nv < hi; nv++ )
	    m += WG.V[remap.first[nv]].getOutDegree();
	mv = 0;
	for( nv=lo; nv < hi; nv += maxVL ) {
	    mv += maxVL * WG.V[remap.first[nv]].getOutDegree();
	}
	nv -= lo;
	assert( nv >= (hi-lo) && nv < (hi-lo) + maxVL );
	assert( mv >= m ); // && mv < m + (hi-lo) * maxVL );
	    
	// Allocate data structures
	if( allocation == -1 )
	    allocateInterleaved();
	else
	    allocateLocal( allocation );

	// In w = remap.first[v], v is the new vertex ID, w is the old one
	// In v = remap.second[w], v is the new vertex ID, w is the old one
	// assert( n == WG.n && m == WG.m ); -- obvious
	const vertex * V = WG.V.get();

	VID maxdeg = V[remap.first[0]].getOutDegree();
	VID * buf = new VID[maxdeg];

	EID nxt = 0;
	for( VID v=lo; v < lo+nv; v += maxVL ) {
	    VID deg = V[remap.first[v]].getOutDegree();
	    index[(v-lo)/maxVL] = std::min( nxt, mv ); // nxt > mv only when deg=0
	    VID mlodeg = deg;
	    for( unsigned short l=0; l < maxVL; ++l ) { // vector lane
		VID vv = v + l;
		EID lnxt = nxt + l;
		if( vv < n ) {
		    VID ww = remap.first[vv];
		    VID ldeg = V[ww].getOutDegree();
		    assert( ldeg <= deg );
		    if( ldeg < mlodeg ) // track first SIMD group w/ mask
			mlodeg = ldeg;
		    // VID buf[ldeg];
		    assert( ldeg <= maxdeg );
		    for( VID j=0; j < ldeg; ++j )
			buf[j] = remap.second[V[ww].getOutNeighbor(j)];
		    std::sort( &buf[0], &buf[ldeg] );
		    for( VID j=0; j < ldeg; ++j ) {
			edges[lnxt] = buf[j];
			lnxt += maxVL;
		    } 
		    for( VID j=ldeg; j < deg; ++j ) {
			edges[lnxt] = ~(VID)0;
			lnxt += maxVL;
		    }
		} else {
		    mlodeg = 0;
		    VID ww = ~(VID)0;
		    for( VID j=0; j < deg; ++j ) {
			edges[lnxt] = ~(VID)0;
			lnxt += maxVL;
		    } 
		}
	    }
	    mindex[(v-lo)/maxVL] = index[(v-lo)/maxVL] + maxVL * mlodeg;
	    nxt += maxVL * deg;
	}
	assert( nxt == mv );
	// for( unsigned short l=0; l < maxVL; ++l )
	// index[nv-maxVL+l] = mv;
	index[nv/maxVL] = mv;
	mindex[nv/maxVL] = mv;
	delete[] buf;
    }
    void import( const GraphCSx & Gcsc,
		 VID lo, VID hi, unsigned short maxVL_,
		 std::pair<const VID *, const VID *> remap,
		 int allocation ) {
	// This is written to be a CSC-style import, i.e., WG is transposed,
	// lo/hi applied to 'sources'.
	// Calculate dimensions of SIMD representation
	maxVL = maxVL_;
	n = Gcsc.numVertices();
	const EID * idx = Gcsc.getIndex();
	const VID * edg = Gcsc.getEdges();
#if 0
	m = 0;
	for( nv=lo; nv < hi; nv++ ) {
	    VID r = remap.first[nv];
	    m += idx[r+1] - idx[r];
	}
	mv = 0;
	for( nv=lo; nv < hi; nv += maxVL ) {
	    VID r = remap.first[nv];
	    mv += maxVL * ( idx[r+1] - idx[r] );
	}
#endif
	m = 0;
	mv = 0;
	for( nv=lo; nv < hi; nv += maxVL ) {
	    VID deg = 0;

	    VID r = remap.first[nv];
	    VID d0 = idx[r+1] - idx[r];
	    m += d0;
	    mv += maxVL * d0;

	    for( unsigned short l=1; l < maxVL; ++l ) {
		VID r = remap.first[nv+l];
		VID d = r < n ? idx[r+1] - idx[r] : 0;
		m += d;
		assert( d <= d0 );
	    }
	}
	nv -= lo;
	assert( nv >= (hi-lo) && nv < (hi-lo) + maxVL );
	assert( mv >= m ); // && mv < m + (hi-lo) * maxVL );
	    
	// Allocate data structures
	if( allocation == -1 )
	    allocateInterleaved();
	else
	    allocateLocal( allocation );

	// In w = remap.first[v], v is the new vertex ID, w is the old one
	// In v = remap.second[w], v is the new vertex ID, w is the old one
	// assert( n == WG.n && m == WG.m ); -- obvious

	VID maxdeg = idx[remap.first[0]+1] - idx[remap.first[0]];
	VID * buf = new VID[maxdeg];

	EID nxt = 0;
	for( VID v=lo; v < lo+nv; v += maxVL ) {
	    VID r = remap.first[v];
	    VID deg = r < n ? idx[r+1] - idx[r] : 0;
	    index[(v-lo)/maxVL] = std::min( nxt, mv ); // nxt > mv only when deg=0
	    VID mlodeg = deg;
	    for( unsigned short l=0; l < maxVL; ++l ) { // vector lane
		VID vv = v + l;
		EID lnxt = nxt + l;
		if( vv < n ) {
		    VID ww = remap.first[vv];
		    VID ldeg = ww < n ? idx[ww+1] - idx[ww] : 0;
		    assert( ldeg <= deg );
		    if( ldeg <= mlodeg ) // track first SIMD group w/ mask
			mlodeg = ldeg;
		    // VID buf[ldeg];
		    assert( ldeg <= maxdeg );
		    for( VID j=0; j < ldeg; ++j )
			buf[j] = remap.second[edg[idx[ww]+j]];
		    std::sort( &buf[0], &buf[ldeg] );
		    for( VID j=0; j < ldeg; ++j ) {
			edges[lnxt] = buf[j];
			lnxt += maxVL;
		    } 
		    for( VID j=ldeg; j < deg; ++j ) {
			edges[lnxt] = ~(VID)0;
			lnxt += maxVL;
		    }
		} else {
		    mlodeg = 0;
		    VID ww = ~(VID)0;
		    for( VID j=0; j < deg; ++j ) {
			edges[lnxt] = ~(VID)0;
			lnxt += maxVL;
		    } 
		}
	    }
	    mindex[(v-lo)/maxVL] = index[(v-lo)/maxVL] + maxVL * mlodeg;
	    nxt += maxVL * deg;
	}
	assert( nxt == mv );
	// for( unsigned short l=0; l < maxVL; ++l )
	// index[nv-maxVL+l] = mv;
	index[nv/maxVL] = mv;
	mindex[nv/maxVL] = mv;
	delete[] buf;
    }

    template<typename Remapper>
    void validate( const GraphCSx & Gcsr,
		   VID s, VID e,
		   const Remapper & remapper ) {
	EID ne = 0;
	for( VID vv=s; vv < e; vv += maxVL ) {
	    for( unsigned short l=0; l < maxVL; ++l ) {
		VID v = vv + l;
		VID Gv = remapper.origID( v );
		for( EID i=index[(v-s)/16]+l; i < index[((v-s)/16)+1]+l; i += maxVL ) {
		    VID u = edges[i];
		    if( u != ~(VID)0 ) {
			VID Gu = remapper.origID( u );
			assert( Gcsr.hasEdge( Gu, Gv ) );
			++ne;
		    }
		}
	    }
	}
	assert( ne == m );
    }

public:
    EID *getIndex() { return index.get(); }
    const EID *getIndex() const { return index.get(); }
    const EID *getMaskIndex() const { return mindex.get(); }
    VID *getEdges() { return edges.get(); }
    const VID *getEdges() const { return edges.get(); }

    VID numVertices() const { return n; }
    EID numEdges() const { return m; }

    VID numSIMDVertices() const { return nv; }
    EID numSIMDEdges() const { return mv; }

    unsigned short getMaxVL() const { return maxVL; }

private:
    void allocateInterleaved() {
	assert( nv % maxVL == 0 );
	// mindex.Interleave_allocate( nv/maxVL+1, sizeof(VID)*maxVL );
	// index.Interleave_allocate( nv/maxVL+1, sizeof(VID)*maxVL );
	// edges.Interleave_allocate( mv, sizeof(VID)*maxVL );
	mindex.allocate( nv/maxVL+1, sizeof(VID)*maxVL, numa_allocation_interleaved() );
	index.allocate( nv/maxVL+1, sizeof(VID)*maxVL, numa_allocation_interleaved() );
	edges.allocate( mv, sizeof(VID)*maxVL, numa_allocation_interleaved() );
    }
    void allocateLocal( int numa_node ) {
	assert( nv % maxVL == 0 );
	// mindex.local_allocate( nv/maxVL+1, sizeof(VID)*maxVL, numa_node );
	// index.local_allocate( nv/maxVL+1, sizeof(VID)*maxVL, numa_node );
	// edges.local_allocate( mv, sizeof(VID)*maxVL, numa_node );
	mindex.allocate( nv/maxVL+1, sizeof(VID)*maxVL, numa_allocation_local( numa_node ) );
	index.allocate( nv/maxVL+1, sizeof(VID)*maxVL, numa_allocation_local( numa_node ) );
	edges.allocate( mv, sizeof(VID)*maxVL, numa_allocation_local( numa_node ) );
    }
};


template<bool doVEBO = true>
class GraphVEBOSlimSell_template {
    GraphCSxSIMD * csc;
    GraphCSx csr; // used only for calculating active vertices -- TODO
    VEBOReorderState<VID,EID> remap;
    partitioner part;
    unsigned short maxVL;

    static constexpr VID rndup( VID nv, unsigned short VL ) {
	return doVEBO ? nv : ( nv % VL ? nv + VL - ( nv % VL ) : nv );
    }

public:
    template<class vertex>
    GraphVEBOSlimSell_template( const wholeGraph<vertex> & WG,
				int npart, unsigned short maxVL_ )
	: csc( new GraphCSxSIMD[npart] ),
	  csr( WG.n, WG.m, -1 ),
	  part( npart, WG.numVertices() ),
	  maxVL( maxVL_ ) {
	// Setup temporary CSC
	wholeGraph<vertex> * WGc = const_cast<wholeGraph<vertex> *>( &WG );
	WGc->transpose();
	GraphCSx csc_tmp( WG, -1 );

	// Calculate remapping table. Do not use the feature to interleave
	// subsequent destinations over per-lane partitions.
	if( doVEBO ) {
	    remap = VEBOReorder( csc_tmp, part, 1, false, maxVL );
	} else {
	    partitioner vebo_part( 1, WG.numVertices() );
	    remap = VEBOReorder( csc_tmp, vebo_part, 1, false, 1 );

	    // Perform partitioning by destination. Ensure every partition
	    // starts at a vertex ID that is a multiple of maxVL to allow
	    // aligned vector load/store
	    partitionBalanceEdges( csc_tmp, remap.getReverse(), part, maxVL );
	}

	// Setup CSC partitions
	// This is inefficient. It repeatedly scans WG.
	map_partitionL( part, [&]( int p ) {
		new (&csc[p]) GraphCSxSIMD();
		csc[p].import( WG, part.start_of(p), part.end_of(p),
			       maxVL, remap.maps(), part.numa_node_of( p ) );
	    } );
	WGc->transpose();

	// Setup CSR
	csr.import( WG, remap.maps() );

	// Clean up intermediates
	csc_tmp.del();
    }

    GraphVEBOSlimSell_template( const GraphCSx & Gcsr,
				int npart, unsigned short maxVL_ )
	: csc( new GraphCSxSIMD[npart] ),
	  // csr( rndup( Gcsr.numVertices(), maxVL_ ), Gcsr.numEdges(), -1 ),
	  part( npart, Gcsr.numVertices() ),
	  maxVL( maxVL_ ) {

	// Setup temporary CSC, try to be space-efficient
	std::cerr << "Transposing CSR...\n";
	GraphCSx csc_tmp( Gcsr.numVertices(), Gcsr.numEdges(), -1 );
	if( Gcsr.isSymmetric() )
	    csc_tmp.import( Gcsr );
	else
	    csc_tmp.import_transpose( Gcsr );

	if( doVEBO ) {
	    // Calculate remapping table. Do not use the feature to interleave
	    // subsequent destinations over per-lane partitions.
	    remap = VEBOReorder( csc_tmp, part, 1, false, maxVL );

	    // Setup CSR
	    csr = GraphCSx( part.get_num_elements(), Gcsr.numEdges(), -1 );
	    csr.import_expand( Gcsr, part, remap.remapper() ); // remap.maps() );
	} else {
	    // Calculate remapping table. Do not use the feature to interleave
	    // subsequent destinations over per-lane partitions.
	    remap = ReorderDegreeSort<VID,EID>( csc_tmp, maxVL );

	    // Perform partitioning by destination. Ensure every partition
	    // starts at a vertex ID that is a multiple of maxVL to allow
	    // aligned vector load/store
	    // partitionBalanceEdges( csc_tmp, remap.getReverse(), part, maxVL );
	    partitionBalanceEdges( csc_tmp, gtraits_getoutdegree_remap<GraphCSx>( csc_tmp, remap.getReverse() ), part, maxVL );

	    // Add in space holder vertices
	    part.appendv( ( VID(maxVL) - csc_tmp.numVertices() ) % maxVL );

	    // Setup CSR
	    csr = GraphCSx( part.get_num_elements(), Gcsr.numEdges(), -1 );
	    csr.import_expand( Gcsr, part, remap.remapper() ); // remap.maps() );
	}

	// Setup CSC partitions
	map_partitionL( part, [&]( int p ) {
		new (&csc[p]) GraphCSxSIMD();
		csc[p].import( csc_tmp, part.start_of(p), part.end_of(p),
			       maxVL, remap.maps(),
			       part.numa_node_of( p ) );
	    } );

	assert( part.get_num_elements() % maxVL == 0
		&& "number of elements must be multiple of maxVL" );

	// validate( Gcsr, get_remapper() );

	// Clean up intermediates
	csc_tmp.del();
    }

    template<typename Remapper>
    void validate( const GraphCSx & Gcsr, const Remapper & remapper ) {
	map_partitionL( get_partitioner(), 
			[&]( int p ) {
			    csc[p].validate( Gcsr,
					     get_partitioner().start_of(p),
					     get_partitioner().start_of(p+1),
					     remapper );
			} );
    }

    void del() {
	csr.del();
	for( int p=0; p < part.get_num_partitions(); ++p )
	    csc[p].del();
	delete[] csc;
	csc = nullptr;
	remap.del();
    }

public:
    void fragmentation() const {
	EID simde = 0;
	VID simdv = 0;
	int np = get_partitioner().get_num_partitions();
	for( int p=0; p < np; ++p ) {
	    simdv += getCSC(p).numSIMDVertices(); 
	    simde += getCSC(p).numSIMDEdges(); 
	}
	double bloat
	    = double(simde*sizeof(VID)+(simdv+getMaxVL())*sizeof(EID))
	    / double(numEdges()*sizeof(VID)+(numVertices()+1)*sizeof(EID));
	std::cerr << "GraphVEBOSlimSell<doVEBO=" << doVEBO
		  << ">:\nvertices: " << numVertices()
		  << "\nedges: " << numEdges()
		  << "\nalloc-vertices: " << simdv
		  << "\nalloc-edges: " << simde
		  << "\ntotal-size: " << (simde*sizeof(VID)+simdv*sizeof(EID))
		  << "\nbloat: " << bloat << "\n";
    }
public:
    VID numVertices() const { return csr.numVertices(); }
    EID numEdges() const { return csr.numEdges(); }

    bool transposed() const { return false; }
    void transpose() { assert( 0 && "Not supported" ); }

    const GraphCSx & getCSR() const { return csr; }
    const GraphCSxSIMD & getCSC( int p ) const { return csc[p]; }
    const GraphVEBOSlimSell_template<doVEBO> & getCSC() const { return *this; }

    const partitioner & get_partitioner() const { return part; }

    VID originalID( VID v ) const { return remap.originalID( v ); }
    VID remapID( VID v ) const { return remap.remapID( v ); }

    auto get_remapper() const { return remap.remapper(); }

    unsigned short getMaxVL() const { return maxVL; }
    static constexpr unsigned short getMaxVLCSC() { return VLUpperBound; }
    static constexpr unsigned short getMaxVLCOO() { return VLUpperBound; }
    static constexpr unsigned short getVLCOOBound() { return VLUpperBound; }
    static constexpr unsigned short getVLCSCBound() { return VLUpperBound; }

    VID getOutDegree( VID v ) const { return getCSR().getDegree( v ); }

    // Pull-based
    static constexpr bool getRndRd() { return true; }
    static constexpr bool getRndWr() { return false; }
};

using GraphSlimSell = GraphVEBOSlimSell_template<false>;
using GraphVEBOSlimSell = GraphVEBOSlimSell_template<true>;

#include "graptor/graph/CGraphCSxSIMDDegree.h"
#include "graptor/graph/CGraphVEBOGraptor.h"

#endif // GRAPHGRIND_CGRAPH_H
