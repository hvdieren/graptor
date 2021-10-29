// -*- C++ -*-
#ifndef GRAPHGRIND_GRAPH_GTRAITS_H
#define GRAPHGRIND_GRAPH_GTRAITS_H

#include "graptor/itraits.h"
#include "graptor/partitioner.h"

// Extract the type VID from GraphTy if present, else use default VID
template<typename GraphTy, typename Enable = void>
struct gtraits_vid {
    using type = VID;
};

template<typename GraphTy>
struct gtraits_vid<GraphTy,
		   typename std::enable_if<
		       !std::is_void<typename GraphTy::VID>::value>::type> {
    using type = typename GraphTy::VID;
};

// Extract the type EID from GraphTy if present, else use default EID
template<typename GraphTy, typename Enable = void>
struct gtraits_eid {
    using type = EID;
};

template<typename GraphTy>
struct gtraits_eid<GraphTy,
		   typename std::enable_if<
		       !std::is_void<typename GraphTy::EID>::value>::type> {
    using type = typename GraphTy::EID;
};

// Extract the type PID from GraphTy if present, else use default PID
template<typename GraphTy, typename Enable = void>
struct gtraits_pid {
    using type = unsigned int;
};

template<typename GraphTy>
struct gtraits_pid<GraphTy,
		   typename std::enable_if<
		       !std::is_void<typename GraphTy::PID>::value>::type> {
    using type = typename GraphTy::PID;
};

// Uniform way of accessing content of a graph
template<typename GraphTy>
struct gtraits {
    using VID = typename gtraits_vid<GraphTy>::type;
    using EID = typename gtraits_eid<GraphTy>::type;
    using PID = typename gtraits_pid<GraphTy>::type;

    static VID numVertices( const GraphTy & G ) { return G.numVertices(); }
    static EID numEdges( const GraphTy & G ) { return G.numEdges(); }
    static const partitioner & getPartitioner( const GraphTy & G ) {
	return G.get_partitioner();
    }
    static auto getRemapper( const GraphTy & G ) { return G.get_remapper(); }

    template<unsigned short W>
    static auto createValidFrontier( const GraphTy & G ) {
	return getRemapper( G ).template createValidFrontier<W>( G );
    }
};

// Obtaining the degree of a vertex
template<typename GraphTy>
struct gtraits_getoutdegree {
    using traits = gtraits<GraphTy>;
    using VID = typename traits::VID;

    gtraits_getoutdegree( const GraphTy & G_ ) : G( G_ ) { }
    
    VID operator() ( VID v ) {
	return G.getOutDegree( v );
    }

private:
    const GraphTy & G;
};

template<typename GraphTy>
struct gtraits_getoutdegree_remap {
    using traits = gtraits<GraphTy>;
    using VID = typename traits::VID;

    gtraits_getoutdegree_remap( const GraphTy & G_, const VID * origID_ )
	: getoutdegree( G_ ), origID( origID_ ) { }
    
    VID operator() ( VID v ) {
	return getoutdegree( origID[v] );
    }

private:
    gtraits_getoutdegree<GraphTy> getoutdegree;
    const VID * origID;
};

// Obtaining the vertex with maximum degree
template<typename GraphTy>
struct gtraits_getmaxoutdegree {
    using traits = gtraits<GraphTy>;
    using VID = typename traits::VID;

    gtraits_getmaxoutdegree( const GraphTy & G_ ) : G( G_ ) { }
    
    VID getMaxOutDegreeVertex() const { return G.getMaxOutDegreeVertex(); }
    VID getMaxOutDegree() const { return G.getMaxOutDegree(); }

private:
    const GraphTy & G;
};

template<typename GraphTy1, typename GraphTy2>
void graph_compare( const GraphTy1 & G, const GraphTy2 & H ) {
    using iterator = typename GraphTy1::edge_iterator;

    assert( G.numEdges() == H.numEdges() );
    assert( G.numVertices() == H.numVertices() );

    for( iterator I=G.edge_begin(), E=G.edge_end(); I != E; ++I ) {
	std::pair<VID,VID> e = *I;
	assert( H.hasEdge( e.first, e.second ) );
    }
}

#endif // GRAPHGRIND_GRAPH_GTRAITS_H
