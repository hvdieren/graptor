// -*- C++ -*-
#ifndef GRAPHGRIND_MM_BUFFER_H
#define GRAPHGRIND_MM_BUFFER_H

/***********************************************************************
 * Represents a memory buffer
 ***********************************************************************/

#include "graptor/mm/methods.h"

namespace mm {

template<typename T>
class buffer {
public:
    using value_type = T;
    
public:
    buffer() { }
    buffer( size_t elements, numa_allocation_partitioned am,
	    const char * reason = nullptr )
	: buffer( am, reason ) {
	assert( elements == am.get_partitioner().get_vertex_range()
		&& "presumed allocation of vertex property" );
    }
    buffer( numa_allocation_partitioned am,
	    const char * reason = nullptr )
	: alc( methods::template allocate_part<value_type,true>(
		   am.get_partitioner(), reason ) ) { }
    buffer( size_t elements, numa_allocation_edge_partitioned am,
	    const char * reason = nullptr )
	: buffer( am, reason ) {
	assert( elements == am.get_partitioner().get_edge_range()
		&& "presumed allocation of edge property" );
    }
    buffer( numa_allocation_edge_partitioned am,
	    const char * reason = nullptr )
	: alc( methods::template allocate_part<value_type,false>(
		   am.get_partitioner(), reason ) ) { }
    buffer( size_t elements, numa_allocation_interleaved am,
	    const char * reason = nullptr )
	: alc( methods::template allocate_intlv<value_type>(
		   elements, reason ) ) { }
    buffer( size_t elements, int fd, off_t off, const numa_allocation & am,
	    const char * reason = nullptr )
	: alc( methods::template map_file<value_type>(
		   elements, fd, off, am, reason ) ) { }
    buffer( size_t elements, numa_allocation_local am,
	    const char * reason = nullptr )
	: alc( methods::template allocate_local<value_type>(
		   elements, am.node(), reason ) ) { }

    void del( const char * reason = nullptr ) {
	if( alc.ptr() )
	    methods::deallocate( alc );
    }
    
    value_type * get() const {
	return reinterpret_cast<value_type *>( alc.ptr() );
    }
private:
    allocation alc;
};
    
} // namespace mm

#endif // GRAPHGRIND_MM_BUFFER_H

