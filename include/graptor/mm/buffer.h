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
    buffer() : alc( 0 ) { }
    [[deprecated("poor constructor design")]]
    buffer( int ) : alc( 0 ) { } // zero-initialise - bad idea / missing alloc
    buffer( size_t elements, const numa_allocation & alloc,
	    const char * reason = nullptr ) {
	switch( alloc.get_kind() ) {
	case na_local:
	    new (this) buffer(
		elements,
		*static_cast<const numa_allocation_local*>( &alloc ) );
	    break;
	case na_interleaved:
	    new (this) buffer(
		elements,
		*static_cast<const numa_allocation_interleaved*>( &alloc ) );
	    break;
	case na_partitioned:
	    new (this) buffer(
		elements,
		*static_cast<const numa_allocation_partitioned*>( &alloc ) );
	    break;
	case na_edge_partitioned:
	    new (this) buffer(
		elements,
		*static_cast<const numa_allocation_edge_partitioned*>( &alloc ) );
	    break;
	case na_small:
	    new (this) buffer(
		elements,
		*static_cast<const numa_allocation_small*>( &alloc ) );
	    break;
	default:
	    UNREACHABLE_CASE_STATEMENT;
	}
    }
    buffer( size_t elements, numa_allocation_partitioned am,
	    const char * reason = nullptr )
	: buffer( am, reason ) {
	assert( ( elements == am.get_partitioner().get_vertex_range()
		  || elements == am.get_partitioner().get_vertex_range()+1 )
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
    buffer( size_t elements, numa_allocation_small am,
	    const char * reason = nullptr )
	: alc( methods::template allocate_small<value_type>(
		   elements, reason ) ) { }

    void del( const char * reason = nullptr ) {
	if( alc.ptr() ) {
	    methods::deallocate( alc );
	    alc.clear();
	}
    }
    
    value_type * get() const {
	return reinterpret_cast<value_type *>( alc.ptr() );
    }

    template<typename U>
    std::enable_if_t<std::is_integral_v<U>,value_type> & get_ref( U t ) {
	return get()[t];
    }
    template<typename U>
    const std::enable_if_t<std::is_integral_v<U>,value_type> &
    get_ref( U t ) const {
	return get()[t];
    }

    template<typename U>
    auto operator[] ( U t ) -> decltype( get_ref(t) ) {
	return get_ref( t );
    }
    template<typename U>
    auto operator[] ( U t ) const -> decltype( get_ref(t) ) {
	return get_ref( t );
    }

private:
    allocation alc;
};
    
} // namespace mm

#endif // GRAPHGRIND_MM_BUFFER_H

