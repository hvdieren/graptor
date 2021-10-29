// -*- C++ -*-
#ifndef GRAPHGRIND_GRAPH_GRAPHCSXWB_H
#define GRAPHGRIND_GRAPH_GRAPHCSXWB_H

#include "graptor/itraits.h"
#include "graptor/mm.h"
#include "graptor/graph/gtraits.h"
#include "graptor/graph/GraphCSx.h"
#include "graptor/reuse.h"

template<typename VID, typename EID>
struct GraphCSxWB_decoder {
    static VID decode_neighbour( VID c ) {
	// return c & ~(VID(1)<<(sizeof(VID)*8-1));
	// return c >> 1;
	return c;
    }
    static bool decode_valid( VID c ) {
	// return c >> (sizeof(VID)*8-1);
	// return c & 1;
	return false;
    }
    template<bool f>
    static VID encode( VID c ) {
	if constexpr ( f ) {
	    return ( c << 1 ) | 1;
	} else {
	    return c << 1;
	}
    }
    static VID encode( VID c, bool f ) {
	return f ? encode<true>( c ) : encode<false>( c );
    }
};

template<typename VID, typename EID>
struct getVDegWB {
    getVDegWB( const EID * idx_ ) : idx( idx_ ) { }
    std::pair<VID,VID> operator() ( VID v ) {
	return std::make_pair(
	    (VID)( idx[v+1] - idx[v] ),
	    GraphCSxWB_decoder<VID,EID>::decode_neighbour( v ) );
    }
private:
    const EID * idx;
};

class GraphCSxWB {
    GraphCSx csx;

public:
    GraphCSxWB() { }
    GraphCSxWB( const std::string & infile, int allocation = -1,
		bool _symmetric = false ) 
	: csx( infile, allocation, _symmetric ) {
	set_writeback();
     }
    GraphCSxWB( const std::string & infile, const numa_allocation & allocation,
	      bool _symmetric )
	: csx( infile, allocation, _symmetric ) {
	set_writeback();
    }
    GraphCSxWB( VID n_, EID m_, int allocation )
	: csx( n_, m_, allocation ) {
	set_writeback();
     }
    GraphCSxWB( const GraphCSx & WG, int allocation )
	: csx( WG, allocation ) {
	set_writeback();
    }
    GraphCSxWB( const GraphCSx & WG,
		const partitioner & part, int p,
		std::pair<const VID *, const VID *> remap )
	: csx( WG, part, p, remap ) {
	set_writeback();
    }

    void del() {
	csx.del();
    }

public:
    void fragmentation() const { csx.fragmentation(); }

    VID max_degree() const {
	return csx.max_degree();
    }

    VID findHighestDegreeVertex() const {
	using T = std::pair<VID,VID>;
	VID n = numVertices();
	T s = sequence::reduce<T>( (VID)0, n, sequence::argmaxF<T>(),
				   getVDegWB<VID,EID>(getIndex()) );
	return s.second;
    }
    
public:
    EID *getIndex() { return csx.getIndex(); }
    const EID *getIndex() const { return csx.getIndex(); }
    VID *getEdges() { return csx.getEdges(); }
    const VID *getEdges() const { return csx.getEdges(); }
    VID *getDegree() { return csx.getDegree(); }
    const VID *getDegree() const { return csx.getDegree(); }
    bool * getFlags() { return m_flags.get(); }
    const bool * getFlags() const { return m_flags.get(); }

    VID numVertices() const { return csx.numVertices(); }
    EID numEdges() const { return csx.numEdges(); }

    VID getDegree( VID v ) const { return csx.getDegree( v ); }
    VID getNeighbor( VID v, VID pos ) const {
	return GraphCSxWB_decoder<VID,EID>::decode_neighbour(
	    csx.getNeighbor( v, pos ) );
    }

    VID getMaxDegreeVertex() const { return csx.getMaxDegreeVertex(); }
    void setMaxDegreeVertex( VID v ) { csx.setMaxDegreeVertex( v ); }

    bool isSymmetric() const { return csx.isSymmetric(); }

    bool hasEdge( VID s, VID d ) const {
	bool ret = false;
	const EID * index = getIndex();
	const VID * edges = getEdges();
	for( EID e=index[s]; e < index[s+1]; ++e )
	    if( GraphCSxWB_decoder<VID,EID>::decode_neighbour( edges[e] ) == d )
		return true;
	return false;
    }

private:
    void set_writeback() {
	// using encoder = GraphCSxWB_decoder<VID,EID>;

	constexpr VID K = 8; // 8 vertex properties per cache block
	constexpr VID mask = (K<<1) - 1;

	VID * edges = getEdges();
	const EID * index = getIndex();
	VID n = numVertices();
	EID m = numEdges();
	mmap_ptr<EID> last_use( (n+K-1)/K, numa_allocation_interleaved() );

	m_flags.allocate( m, numa_allocation_interleaved() );

	VID bit = VID(1) << (sizeof(VID)*8-1);

	constexpr EID C = EID(1925) * EID(1024 * 1024)
	    / ( 8 * sizeof(EID) * 100 );

	std::cerr << "GraphCSxWB: K=" << K << " C=" << C << "\n";

	// Initialise next use
	parallel_for( VID v=0; v < (n+K-1)/K; ++v )
	    last_use[v] = ~EID(0);

	EID nsete = 0, nsetv = 0;

	RDLogger<VID> reuse( (n+K-1) / K, m ); 

	// Edges are initially not encoded. Depending on the encoding, that
	// may be problematic, i.e., if v != (v,false)
	// for( EID e=0; e < m; ++e )
	// edges[e] = encoder::encode<false>( edges[e] );
	
	for( EID e=0; e < m; ++e ) {
	    // Take into account multiple properties will be mapped to same
	    // cache block. Depends on property byte width...
	    // VID v = encoder::decode_neighbour( edges[e] ) / K;
	    VID v = edges[e] / K;

	    // If access is considered cache miss, could remove all blocks
	    // that have not been referenced since from splay tree to keep
	    // its size small. Current block has to be in splay tree because
	    // it has just been touched.
	    std::pair<size_t,size_t> r = reuse.access( v );
	    size_t dist = r.first;
	    size_t prev = r.second;

	    // std::cerr << "v=" << v << " dist=" << dist << " prev=" << prev << "\n";
	    
	    // set bit if next use is far away
	    if( ~dist != 0 && dist > C ) {
		// std::cerr << "set edge at " << prev << " e=" << (edges[prev]/K) << "\n";
		// edges[prev] = encoder::encode<true>(
		// encoder::decode_neighbour( edges[prev] ) );
		m_flags[prev] = true;
		++nsete;
	    }
	    last_use[v] = e;
	}

	for( VID v=0; v < (n+1)/K; ++v ) {
	    if( last_use[v] != ~EID(0) ) {
		// edges[last_use[v]] = encoder::encode<true>(
		// encoder::decode_neighbour( edges[last_use[v]] ) );
		EID e = last_use[v];
		m_flags[e] = true;
		++nsetv;
	    }
	}

	EID nfl = 0;
	for( EID e=0; e < m; ++e ) {
	    unsigned f = m_flags[e];
	    if( f )
		++nfl;
	}

	EID nv0 = 0;
	for( VID v=0; v < n; ++v ) {
	    EID l = index[v+1];
	    bool any = false;
	    for( EID k=index[v]; k < l; ++k ) {
		if( m_flags[k] ) {
		    any = true;
		    break;
		}
	    }
	    if( !any ) {
		*reinterpret_cast<unsigned char *>( &m_flags[index[v]] ) |= 2;
		++nv0;
	    }
	}

	last_use.del();

	std::cerr << "GraphCSxWB: writeback marked edges: reuse: " << nsete
		  << " last use: " << nsetv
		  << " vertices w/o flags: " << nv0
		  << " fraction edges: " << ((double)(nsete+nsetv)/(double)m)
		  << " checked: " << nfl
		  << "\n";
    }

private:
    mmap_ptr<bool> m_flags;
};

// Obtaining the degree of a vertex
template<>
struct gtraits_getoutdegree<GraphCSxWB> {
    using traits = gtraits<GraphCSxWB>;
    using VID = typename traits::VID;

    gtraits_getoutdegree( const GraphCSxWB & G_ ) : G( G_ ) { }
    
    VID operator() ( VID v ) {
	return G.getDegree( v );
    }

private:
    const GraphCSxWB & G;
};

// Obtaining the vertex with maximum degree
template<>
struct gtraits_getmaxoutdegree<GraphCSxWB> {
    using traits = gtraits<GraphCSxWB>;
    using VID = typename traits::VID;

    gtraits_getmaxoutdegree( const GraphCSxWB & G_ ) : G( G_ ) { }
    
    VID getMaxOutDegreeVertex() const { return G.getMaxDegreeVertex(); }
    VID getMaxOutDegree() const { return G.getDegree( G.getMaxDegreeVertex() ); }

private:
    const GraphCSxWB & G;
};

#endif // GRAPHGRIND_GRAPH_GRAPHCSXWB_H
