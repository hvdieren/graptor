// -*- c++ -*-

#ifndef GRAPHGRIND_DSL_VMAPUTILS_H
#define GRAPHGRIND_DSL_VMAPUTILS_H

#include <cstdlib>
#include <typeinfo>
#include <mutex>

#include "graptor/frontier.h"

//Note: this is the optimized version of vertexMap which does not
//perform a filter
template <class F>
void vertexMap(const partitioner &part, frontier V, F add)
{
    // const int perNode = part.get_num_per_node_partitions();
    if( V.allTrue() || V.getType() == frontier_type::ft_unbacked ) {
	// In the case of an unbacked frontier, we can only copy over all data
	// loop(j,part,perNode,add(j));
	map_vertexL( part, [&]( VID v ) { add(v); } );
    } else if( V.hasDense() ) {
	if( V.getDenseB() ) {
	    bool * d = V.getDenseB();
	    // loop(j,part,perNode,if (d[j]) add(j));
	    map_vertexL( part, [&]( VID v ) { if (d[v]) add(v); } );
	} else if( V.getDenseL<8>() ) {
	    logical<8> * d = V.getDenseL<8>();
	    // loop(j,part,perNode,if (d[j]) add(j));
	    map_vertexL( part, [&]( VID v ) { if (d[v]) add(v); } );
	} else if( V.getDenseL<4>() ) {
	    logical<4> * d = V.getDenseL<4>();
	    // loop(j,part,perNode,if (d[j]) add(j));
	    map_vertexL( part, [&]( VID v ) { if (d[v]) add(v); } );
	} else if( V.template getDense<frontier_type::ft_bit>() ) {
	    unsigned char * b = V.template getDense<frontier_type::ft_bit>();
	    map_vertexL( part, [&]( VID v ) {
		    if( (b[v/8] >> (v%8)) & 1 ) add(v);
		} );
	} else
	    assert( 0 && "NYI" );
    } else {
	VID * s = V.getSparse();
	parallel_loop( (VID)0, V.nActiveVertices(), [&]( VID i ) {
            add(s[i]);
	} );
    }
}

template <class F>
void vertexMap(const partitioner &part, F add)
{
    // const int perNode = part.get_num_per_node_partitions();
    // loop(j,part,perNode,add(j));
    map_vertexL( part, [&]( VID v ) { add(v); } );
}

//Note: this is the version of vertexMap in which only a subset of the
//input partitioned_vertices is returned

template <typename vertex, class F>
frontier vertexFilter(const partitioned_graph<vertex> & GA, frontier V, F filter) {
    frontier f = V.filter( GA.get_partitioner(), filter );
    f.calculateActiveCounts( GA, f.nActiveVertices() );
    return f;
}

template<typename GraphType, class F>
frontier vertexFilter(const GraphType & GA, frontier V, F filter)
{
    frontier f = V.filter( GA.get_partitioner(), filter );
    if( f.getType() == frontier_type::ft_sparse )
	f.calculateActiveCounts( GA.getCSR(), GA.get_partitioner(),
				 f.nActiveVertices() );
    else
	f.calculateActiveCounts( GA );
    return f;
}

namespace detail {
// Copy operation
template<typename T>
struct CopyArray
{
    T* to;
    const T* from;
    CopyArray( T* _to, const T* _from ) : to( _to ), from( _from ) { }
    inline void operator () ( VID i ) {
	to[i] = from[i];
    }
};

} // namespace detail

// Arrays src and dst differ only in the elements listed in the frontier F.
// Restore equality (i.e., copy src to dst) but do it sparingly by utilising
// the frontier.
template <typename T>
void maintain_copies( const partitioner & part, frontier & F, 
		      T * dst, const T * src ) {
    vertexMap( part, F, detail::CopyArray<T>( dst, src ) );
}

template <typename T>
void maintain_copies( const partitioner & part,
		      T * dst, const T * src ) {
    vertexMap( part, detail::CopyArray<T>( dst, src ) );
}

template <typename T>
void maintain_copies( const partitioner & part, frontier & F, 
		      mmap_ptr<T> & dst, const mmap_ptr<T> & src ) {
    maintain_copies( part, F, dst.get(), src.get() );
}

template <typename T>
void maintain_copies( const partitioner & part,
		      mmap_ptr<T> & dst, const mmap_ptr<T> & src ) {
    maintain_copies( part, dst.get(), src.get() );
}

#endif // GRAPHGRIND_DSL_VMAPUTILS_H
