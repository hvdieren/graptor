// -*- C++ -*-
#ifndef GRAPHGRIND_GRAPH_REMAP_H
#define GRAPHGRIND_GRAPH_REMAP_H

// std::pair
#include <utility>

#include "graptor/partitioner.h"
#include "graptor/frontier.h"
#include "graptor/graph/gtraits.h"

/*
 * It is ok performance-wise to pass the RemapVertex types by value.
 * The classes do not own the allocated memory that they hold pointers to.
 */

template<typename lVID>
class RemapVertex {
public:
    using VID = lVID;
    
    RemapVertex( const VID * origID, const VID * remapID )
	: m_origID( origID ), m_remapID( remapID ) { }

    static constexpr bool is_idempotent() { return false; }
    
    VID origID( VID v ) const { return m_origID[v]; }
    VID remapID( VID v ) const { return m_remapID[v]; }

    VID * getOrigIDPtr() const { return const_cast<VID *>( m_origID ); }

    template<unsigned short W, typename GraphTy>
    frontier createValidFrontier( const GraphTy & G ) {
	using traits = gtraits<GraphTy>;
	const partitioner & part = traits::getPartitioner( G );
	frontier f = frontier::template dense<W>( part );
	logical<W> *d = f.getDenseL<W>();
	VID norig = part.get_num_vertices();
	unsigned int np = part.get_num_partitions();
	EID * nacte = new EID[np];
	VID * nactv = new VID[np];
	map_partitionL( part, [&]( unsigned int p ) {
		VID s = part.start_of( p );
		VID e = part.end_of( p );
		EID nactep = 0;
		VID nactvp = 0;
		for( VID v=s; v < e; ++v ) {
		    nactvp++;
		    nactep += G.getOutDegree( v );
		    d[v] = logical<W>::get_val( origID( v ) < norig );
		}
		nactv[p] = nactvp;
		nacte[p] = nactep;
	    } );

	VID nav = 0;
	EID nae = 0;
	for( unsigned int p=0; p < np; ++p ) {
	    nav += nactv[p];
	    nae += nacte[p];
	}
	
	assert( norig == nav );
	    
	f.setActiveCounts( nav, nae );
	delete[] nactv;
	delete[] nacte;
	return f;
    }

private:
    const VID * m_origID;
    const VID * m_remapID;
};

template<typename lVID>
class RemapVertexIdempotent {
public:
    using VID = lVID;
    
    RemapVertexIdempotent() { }

    static constexpr bool is_idempotent() { return true; }
    
    VID origID( VID v ) const { return v; }
    VID remapID( VID v ) const { return v; }

    template<unsigned int W, typename GraphTy>
    static frontier createValidFrontier( const GraphTy & G ) {
	using traits = gtraits<GraphTy>;
	const partitioner & part = traits::getPartitioner( G );
	return frontier::all_true( traits::numVertices( G ),
				   traits::numEdges( G ) );
    }
};

#endif // GRAPHGRIND_GRAPH_REMAP_H
