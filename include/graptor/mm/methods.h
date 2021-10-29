// -*- C++ -*-
#ifndef GRAPHGRIND_MM_METHODS_H
#define GRAPHGRIND_MM_METHODS_H

#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <memory>

#if NUMA
#include <numa.h>
#include <numaif.h>
#endif

/***********************************************************************
 * Represents a memory allocation
 ***********************************************************************/

#include "graptor/partitioner.h"
#include "graptor/mm/allocation.h"

namespace mm {

struct config {
    static constexpr int FLAGS = (MAP_PRIVATE|MAP_ANON);
    static constexpr int PROTECTED = (PROT_WRITE|PROT_READ);
#if NUMA
    static constexpr int BIND_TO_NODE_FLAGS = (MPOL_BIND);
    static constexpr int BIND_INTERLEAVE_FLAGS = (MPOL_INTERLEAVE);
    static constexpr int BIND_LOCAL_FLAGS = (MPOL_BIND);
#else
    static constexpr int BIND_TO_NODE_FLAGS = 0;
    static constexpr int BIND_INTERLEAVE_FLAGS = 0;
    static constexpr int BIND_LOCAL_FLAGS = 0;
#endif

    static constexpr size_t HUGE_PAGE_SIZE = size_t(2) << 20; // 2 MiB
    static constexpr size_t SMALL_PAGE_SIZE = size_t(4) << 10; // 4 KiB

    static constexpr bool USE_POSIX_MEMALIGN = true;
    static constexpr bool USE_HUGE_PAGES = true;
#if NUMA
    static constexpr bool BIND_NUMA = true;
#else
    static constexpr bool BIND_NUMA = false;
#endif

    static constexpr size_t PAGE_SIZE
    = USE_HUGE_PAGES ? HUGE_PAGE_SIZE : SMALL_PAGE_SIZE;

    static constexpr size_t roundup_page_size( size_t size ) {
	return size % PAGE_SIZE
	    ? (((size + PAGE_SIZE - 1)/ PAGE_SIZE)) * PAGE_SIZE
	    : size;
    }

    static intptr_t round_page( intptr_t ptr, size_t & sz )
    {
	intptr_t page_size = PAGE_SIZE;

        intptr_t ret = (ptr + (PAGE_SIZE-1)) & ~intptr_t(PAGE_SIZE - 1);
        assert( (ret & (PAGE_SIZE - 1)) == 0 );

	intptr_t size = sz;
	size += (PAGE_SIZE-1) - ((ptr + PAGE_SIZE-1) & intptr_t(PAGE_SIZE -1));
        size = (size + PAGE_SIZE-1) & ~intptr_t(PAGE_SIZE - 1);
        assert( (size & (PAGE_SIZE - 1)) == 0 );
	sz = size;
        return ret;
    }
};

struct methods {
    // Allocate memory assuming size is a multiple of config::PAGE_SIZE
    static allocation aligned_allocate( size_t size ) {
	void * memp;
	if constexpr ( config::USE_POSIX_MEMALIGN ) {
	    int ret = posix_memalign( &memp, config::PAGE_SIZE, size );
	    if( ret != 0 ) { // failure
		std::cerr << __FILE__ << ':'
			  << __LINE__ << ':'
			  << __func__ << ": posix_memalign of size "
			  << size << " failed: " << strerror( ret )
			  << std::endl;
		abort();
	    }
	} else {
	    memp = mmap( 0, size, config::PROTECTED, config::FLAGS, 0, 0 );
	    if( memp == MAP_FAILED ) { // failure
		std::cerr << __FILE__ << ':'
			  << __LINE__ << ':'
			  << __func__ << ": mmap of size "
			  << size << " failed: " << strerror( errno )
			  << std::endl;
		abort();
	    }
	}
	intptr_t mem = reinterpret_cast<intptr_t>( memp );
	return allocation( mem, size );
    }

    template<typename T,    // type to allocate
	     bool byV,      // vertex property (true), or edge property (false)
	     typename lVID, // partitioner config - VID type
	     typename lEID, // partitioner config - EID type
	     typename lPID  // partitioner config - PID type
	     >
    static allocation
    allocate_part( const partitioner_template<lVID,lEID,lPID> &part,
		   const char * reason = nullptr ) {
	size_t size;
	if constexpr ( byV )
	    size = part.get_vertex_range() * sizeof(T);
	else
	    size = part.get_edge_range() * sizeof(T);
	size = config::roundup_page_size( size );

	allocation alc = aligned_allocate( size );

	MM_DEBUG_PART_ALLOC( part, sizeof(T),
			     reinterpret_cast<void *>( alc.ptr() ), reason );

	if constexpr ( config::BIND_NUMA )
	    numa_bind<T,byV>( alc, part );

	return alc;
    }

    template<typename T>    // type to allocate
    static allocation
    allocate_intlv( size_t elements,
		    const char * reason = nullptr ) {
	size_t size = config::roundup_page_size( elements * sizeof(T) );

	allocation alc = aligned_allocate( size );

	MM_DEBUG_INTLV_ALLOC( elements, sizeof(T), alc.ptr(), reason );

	if constexpr ( config::BIND_NUMA )
	    interleave_pages( alc );

	return alc;
    }

    template<typename T>    // type to allocate
    static allocation
    allocate_local( size_t elements,
		    int numa_node,
		    const char * reason = nullptr ) {
	size_t size = config::roundup_page_size( elements * sizeof(T) );

	allocation alc = aligned_allocate( size );

	MM_DEBUG_LOCAL_ALLOC( elements, sizeof(T), numa_node, alc.ptr(), reason );

	if constexpr ( config::BIND_NUMA )
	    bind_pages( reinterpret_cast<void*>(alc.ptr()), alc.size(),
			config::BIND_TO_NODE_FLAGS, numa_node );

	return alc;
    }

    static void deallocate( const allocation & alc,
			    const char * reason = nullptr ) {
	MM_DEBUG_DEL( reinterpret_cast<void *>( alc.ptr() ), reason );
	if constexpr ( config::USE_POSIX_MEMALIGN ) {
            free( reinterpret_cast<void *>( alc.ptr() ) );
	} else {
            int munmapres = munmap( reinterpret_cast<void *>( alc.ptr() ),
				    alc.size() );
            if( munmapres == -1 ) {
		std::cerr << __FILE__ << ':'
			  << __LINE__ << ':'
			  << __func__ << ": munmap of address "
			  << std::hex << alc.ptr() << std::dec
			  << " size " << alc.size() << " failed: "
			  << strerror( errno )
			  << std::endl;
                abort();
            }
        }
    }

private:
    template<typename T,    // type to allocate
	     bool byV,      // vertex property (true), or edge property (false)
	     typename lVID, // partitioner config - VID type
	     typename lEID, // partitioner config - EID type
	     typename lPID  // partitioner config - PID type
	     >
    static void
    numa_bind( const allocation & alc,
	       const partitioner_template<lVID,lEID,lPID> & part ) {
#if NUMA
        const int partNum = part.get_num_partitions();
        intptr_t pmem = alc.ptr();

        for( int p=0 ; p < num_numa_node; ++p ) {
	    lPID pe = part.numa_start_of( p+1 );
	    auto ppe = part.template start_of<byV>(pe);
	    intptr_t pmem_e = pmem + static_cast<intptr_t>( ppe * sizeof(T) );
	    size_t size = pmem_e - pmem;
	    intptr_t pmem_er = config::round_page( pmem_e, size );
	    size = pmem_er - pmem;
	    bind_pages( reinterpret_cast<void*>(pmem), size,
			config::BIND_TO_NODE_FLAGS, p );
	    pmem +=size;
	}
#endif
    }

    static void
    bind_pages( void * mem, size_t size, int policy, int numa_node) {
#if NUMA
        struct bitmask *bmp;
        bmp = numa_allocate_nodemask();
        numa_bitmask_setbit( bmp, numa_node );
	dombind( mem, size, config::BIND_TO_NODE_FLAGS, bmp );
        numa_bitmask_free( bmp );
#endif
    }

    static void
    interleave_pages( const allocation & alc ) {
#if NUMA
        struct bitmask *bmp;
        bmp = numa_allocate_nodemask();
        numa_bitmask_setall( bmp );
	dombind( reinterpret_cast<void*>( alc.ptr() ), alc.size(),
		 config::BIND_INTERLEAVE_FLAGS, bmp );
        numa_bitmask_free( bmp );
#endif
    }

#if NUMA
    static void
    dombind( void *mem, size_t size, int policy, struct bitmask *bmp ) {
       long ret = mbind( mem, size, policy,
			 bmp ? bmp->maskp : NULL, bmp ? bmp->size : 0, 0 );
       if( ret < 0 ) {
	   // Survive after the error - performance is affected but normally
	   // speaking not accuracy.
	   std::cerr << __FILE__ << ':'
		     << __LINE__ << ':'
		     << __func__ << ": mbind of address "
		     << std::hex << mem << std::dec
		     << " size " << size << " failed: " << strerror( errno )
		     << std::endl;
       }
    }
#endif
};
    
} // namespace mm

#endif // GRAPHGRIND_MM_METHODS_H
