// -*- C++ -*-
#ifndef GRAPHGRIND_MM_H
#define GRAPHGRIND_MM_H

#if defined(DMALLOC)
#include <dmalloc.h>
#endif

/*
 * TODO:
 * - when using huge pages and allocating small arrays, it would be good to
 *   revert to small pages for those small arrays.
 */

#include <stdlib.h>
// #include "parallel.h"
#include "graptor/legacy/gettime.h"
#include "graptor/partitioner.h"
#include <assert.h>
#include <unistd.h>
#include <errno.h>
#include <string>
#include <algorithm>
#include <sys/mman.h>
#include <memory> // std::align
#if NUMA
#include <numa.h>
#include <numaif.h>
#endif
#include <vector>
//#define MAP_HUGETLB 0x40000
//#define FLAGS (MAP_PRIVATE|MAP_ANON|MAP_HUGETLB)
#define MM_FLAGS (MAP_PRIVATE|MAP_ANON)
#define MM_PROTECTED (PROT_WRITE|PROT_READ)
/*
#define bflags (MPOL_MF_MOVE)
*/
#define mflag (MPOL_BIND)
// Linux only supports two megabyte pages
//Align totalSize 1 << 21 2M 1G 1024*2048=2097152;
constexpr intptr_t page_size = intptr_t(1)<<21;
//Align page for part_allocation
constexpr intptr_t small_size = intptr_t(4)<<10;
using namespace std;
static double mmap_alloc=0;
static double del_time=0;

#include "graptor/itraits.h"
#include "graptor/customfp.h"

/***********************************************************************
 * Debugging support
 ***********************************************************************/
#if GRAPTOR_MM_DEBUG

#include <mutex>
#include <set>

#define MM_DEBUG_INTLV_ALLOC(n,e,mem,s)				\
    _mm_dbg_intlv_alc(__FILE__,__LINE__,(n),(e),(mem),(s))
#define MM_DEBUG_PART_ALLOC(part,e,mem,s)			\
    _mm_dbg_part_alc(__FILE__,__LINE__,(part),(e),(mem),(s))
#define MM_DEBUG_LOCAL_ALLOC(n,e,node,mem,s)			\
    _mm_dbg_local_alc(__FILE__,__LINE__,(n),(e),(node),(mem),(s))
#define MM_DEBUG_DEL(mem,s) 	 	 	\
    _mm_dbg_del(__FILE__,__LINE__,(mem),(s))

namespace {

static std::recursive_mutex _mm_dbg_mux;
static std::set<void *> _mm_dbg_current;

static void
_mm_dbg_alc( const char * file, unsigned lineno, const char * s,
	     const char * amethod, void * mem ) {
    std::unique_lock<std::recursive_mutex> guard( _mm_dbg_mux );
    std::cerr << "MM_DEBUG:" << file << ':' << lineno
	      << ": @=" << mem << " allocate-" << amethod << " reason: ";
    if( s )
	std::cerr << s;
    else
	std::cerr << "none";
    if( _mm_dbg_current.find( mem ) != _mm_dbg_current.end() )
	std:: cerr << " [ALREADY PRESENT]";
    else
	_mm_dbg_current.insert( mem );

    std::cerr << " (" << _mm_dbg_current.size() << " allocations)";
}

static void
_mm_dbg_part_alc( const char * file, unsigned lineno,
		  const partitioner & part, size_t e, void * mem,
		  const char * s ) {
    std::unique_lock<std::recursive_mutex> guard( _mm_dbg_mux );
    size_t size = part.get_num_elements() * e;
    _mm_dbg_alc( file, lineno, s, "part", mem );
    std::cerr << " PART #elements=" << part.get_num_elements()
	      << " #partitions=" << part.get_num_partitions()
	      << " total size=" << pretty(size) << std::endl;
}

static void
_mm_dbg_local_alc( const char * file, unsigned lineno,
		   size_t n, size_t e, size_t node, void * mem,
		   const char * s ) {
    std::unique_lock<std::recursive_mutex> guard( _mm_dbg_mux );
    size_t size = n * e;
    _mm_dbg_alc( file, lineno, s, "local", mem );
    std::cerr << " LOCAL #elements=" << n << " node=" << node
	      << " total size=" << pretty(size) << std::endl;
}
static void
_mm_dbg_intlv_alc( const char * file, unsigned lineno,
		   size_t n, size_t e, void * mem, const char * s ) {
    std::unique_lock<std::recursive_mutex> guard( _mm_dbg_mux );
    size_t size = n * e;
    _mm_dbg_alc( file, lineno, s, "intlv", mem );
    std::cerr << " INTLV #elements=" << n
	      << " total size=" << pretty(size) << std::endl;
}

static void
_mm_dbg_del( const char * file, unsigned lineno,
	     void * mem, const char * s ) {
    std::unique_lock<std::recursive_mutex> guard( _mm_dbg_mux );
    std::cerr << "MM_DEBUG:" << file << ':' << lineno
	      << ": @=" << mem << " free reason: ";
    if( s )
	std::cerr << s;
    else
	std::cerr << "none";

    auto e = _mm_dbg_current.find( mem );
    if( e == _mm_dbg_current.end() )
	std:: cerr << " [NOT PRESENT]";
    else
	_mm_dbg_current.erase( e );

    std::cerr << " (" << _mm_dbg_current.size() << " allocations)";

    std::cerr << "\n";
}

} // namespace anonymous

#else // GRAPTOR_MM_DEBUG

#define MM_DEBUG_LOCAL_ALLOC(n,e,node,mem,s)
#define MM_DEBUG_PART_ALLOC(part,e,mem,s)
#define MM_DEBUG_INTLV_ALLOC(n,e,mem,s)
#define MM_DEBUG_DEL(mem,s)

#endif // GRAPTOR_MM_DEBUG

/***********************************************************************
 * Portability
 ***********************************************************************/
#if defined(__clang__) && __clang_major__ < 12
// Clang 3.9.0 on a gcc 4.8.5 system does not recognize the std::align function
namespace std {
void *align( std::size_t alignment,
	     std::size_t size,
	     void *&ptr,
	     std::size_t&space ) {
    char * p = reinterpret_cast<char *>( ptr );
    intptr_t off = intptr_t(p) & (intptr_t(alignment)-1);
    if( off != 0 ) {
	if( size + alignment - off > space )
	    return nullptr;
	p += alignment - off;
	ptr = reinterpret_cast<void *>( p );
    }
    return ptr;
}
}
#endif // __clang__

// Alignment requirement for AVX-512
#define MIN_ALIGN 64

enum numa_allocation_kind {
    na_local,
    na_interleaved,
    na_partitioned,
    na_edge_partitioned
};

class numa_allocation {
    numa_allocation_kind kind;

protected:
    numa_allocation( numa_allocation_kind k ) : kind( k ) { }

public:
    numa_allocation_kind get_kind() const { return kind; }
};

class numa_allocation_local : public numa_allocation {
    int numa_node;

public:
    numa_allocation_local( int numa_node_ )
	: numa_allocation( na_local ), numa_node( numa_node_ ) { }

    int node() const { return numa_node; }
};

class numa_allocation_interleaved : public numa_allocation {
public:
    numa_allocation_interleaved() : numa_allocation( na_interleaved ) { }
};

class numa_allocation_partitioned : public numa_allocation {
    const partitioner & part;

public:
    numa_allocation_partitioned( const partitioner & part_ )
	: numa_allocation( na_partitioned ), part( part_ ) { }
    const partitioner & get_partitioner() const { return part; }
};

class numa_allocation_edge_partitioned : public numa_allocation {
    const partitioner & part;

public:
    numa_allocation_edge_partitioned( const partitioner & part_ )
	: numa_allocation( na_edge_partitioned ), part( part_ ) { }
    const partitioner & get_partitioner() const { return part; }
};

#if NUMA

/***********************************************************************
 * Definition of mmap_ptr for systems with NUMA
 ***********************************************************************/
template <typename T>
class mmap_ptr
{
    size_t totalSize;
    void * mem;
public:
    using type = T;
    
    mmap_ptr():mem(0),totalSize(0) {}

    // New interface, more streamlined
    mmap_ptr( numa_allocation_partitioned alloc ) : mmap_ptr() {
        part_allocate<true>( alloc.get_partitioner() );
    }
    mmap_ptr( numa_allocation_edge_partitioned alloc ) : mmap_ptr() {
        part_allocate<false>( alloc.get_partitioner() );
    }
    // For uniformity
    mmap_ptr( size_t elements, numa_allocation_partitioned alloc ) : mmap_ptr() {
	assert( elements == alloc.get_partitioner().get_vertex_range() );
        part_allocate<true>( alloc.get_partitioner() );
    }
    mmap_ptr( size_t elements, numa_allocation_interleaved alloc ) : mmap_ptr() {
        alloc_interleaved( elements );
    }
    mmap_ptr( size_t elements, numa_allocation_local alloc ) : mmap_ptr() {
        local_allocate( elements, alloc.node() );
    }
    mmap_ptr( size_t elements, const numa_allocation & alloc ) : mmap_ptr() {
	allocate( elements, alloc );
/*
	switch( alloc.get_kind() ) {
	case na_local:
	    local_allocate( elements,
			    *static_cast<numa_allocation_local*>( &alloc ) );
	    break;
	case na_interleaved:
	    Interleave_allocate( elements );
	    break;
	case na_partitioned:
	    part_allocate( elements,
			   *static_cast<numa_allocation_partitioned*>( &alloc ) );
	    break;
	}
*/
    }

    // Old interface
    mmap_ptr(const partitioner & part) : mmap_ptr()
    {
        // Constructor intended for frontiers
        // and algorithm-specific vertex arrays
        part_allocate<true>(part);
    }
    mmap_ptr(size_t elements) : mmap_ptr()
    {
        // Constructor intended for whole graph's edge array. 
        // It does a page-by-page
        // interleaved allocation
        alloc_interleaved( elements );
    }

    mmap_ptr(size_t elements, size_t numa_node) : mmap_ptr()   // NUMA-local allocation
    {
        // Constructor intended for partitioned graphs.
        local_allocate(elements,numa_node);
    }

    void allocate( size_t elements, const numa_allocation & alloc ) {
	switch( alloc.get_kind() ) {
	case na_local:
	    local_allocate( elements,
			    *static_cast<const numa_allocation_local*>( &alloc ) );
	    break;
	case na_interleaved:
	    alloc_interleaved( elements );
	    break;
	case na_partitioned:
	    part_allocate<true>( elements,
			   *static_cast<const numa_allocation_partitioned*>( &alloc ) );
	case na_edge_partitioned:
	    part_allocate<false>( elements,
			   *static_cast<const numa_allocation_edge_partitioned*>( &alloc ) );
	    break;
	}
    }

    void allocate( size_t elements, size_t align, const numa_allocation & alloc ) {
	switch( alloc.get_kind() ) {
	case na_local:
	    // Note: assume alignment less than page size.
	    local_allocate( elements,
			    *static_cast<const numa_allocation_local*>( &alloc ) );
	    break;
	case na_interleaved:
	    alloc_interleaved( elements, align );
	    break;
	case na_partitioned:
	    // Note: need to ensure alignment through partitioner
	    part_allocate<true>( elements,
				 *static_cast<const numa_allocation_partitioned*>( &alloc ) );
	    break;
	}
    }

    void allocate( const numa_allocation_partitioned &alloc ) {
	part_allocate<true>( alloc.get_partitioner() );
    }

    template<bool byV>
    void part_allocate(size_t elements,
		       const numa_allocation_partitioned & alloc) {
	static_assert( byV == true, "expectation" );
	part_allocate<true>( elements, alloc.get_partitioner() );
    }	

    template<bool byV>
    void part_allocate(size_t elements,
		       const numa_allocation_edge_partitioned & alloc) {
	static_assert( byV == false, "expectation" );
	part_allocate<false>( elements, alloc.get_partitioner() );
    }	

    template<bool byV,typename lVID, typename lEID, typename lPID>
    void part_allocate(const partitioner_template<lVID,lEID,lPID> &part) {
	if constexpr ( byV )
	    part_allocate<byV>( part.get_vertex_range(), part );
	else
	    part_allocate<byV>( part.get_edge_range(), part );
    }

    template<bool byV,typename lVID, typename lEID, typename lPID>
    void part_allocate(size_t elements,
		       const partitioner_template<lVID,lEID,lPID> &part)
    {
        //timer part_alloc;
       // part_alloc.start();
        if( totalSize !=0 || mem != 0 )
        {
            cerr<<"partitioner already allocated"<<'\n';
            abort();
        }
        // totalSize = part.get_vertex_range()*sizeof(T);
        totalSize = elements*sizeof(T);
        if((totalSize % page_size) !=0)
        {
           totalSize = (((totalSize+page_size-1)/ page_size)) * page_size;
	}
	int err;
	size_t pgsz;
#if defined(DMALLOC)
	pgsz = page_size;
	mem = malloc( totalSize+page_size-1 );
	err = errno;
#elif HUGE_PAGES
	pgsz = page_size;
        // mem = mmap( 0, totalSize, MM_PROTECTED, MM_FLAGS | MAP_HUGETLB, 0, 0);
	if( (err = posix_memalign( &mem, page_size, totalSize )) < 0 )
	    mem = NULL;
#else
	pgsz = small_size;
        mem = mmap( 0, totalSize, MM_PROTECTED, MM_FLAGS, 0, 0);
	err = errno;
#endif
        if( mem == (void *)-1 || mem ==(void *)0 )
        {
            std::cerr << "part mmap failed: " << strerror(err) << ", size " << totalSize << '\n';
            abort();
        }

	// std::cerr << "mmap_ptr::part_allocate: " << mem << " size " << totalSize << "\n";

	MM_DEBUG_PART_ALLOC( part, sizeof(T), mem, "none" );

        // cout << "mmap: mem=" << mem << " size=" << totalSize << "\n";
	//For frontier and algorithm data array, if not use 
	//NUMA aware allocation, not use mbind, only mmap       
        const int partNum = part.get_num_partitions();
        // const int perNode = part.get_num_per_node_partitions();
        intptr_t pmem = reinterpret_cast<intptr_t>(mem);

        for( int p=0 ; p < num_numa_node; ++p ) {
            // for( int i = perNode*p; i < perNode*(p+1); ++i ) { 
                //This function use too many time during huge array 
                //to do special allocation use the mbind()
	    // size_t size = part.get_size(perNode*p)*sizeof(T);
	    // intptr_t pmem_e = intptr_t(mem) + part.start_of(perNode*(p+1))*sizeof(T);
	    unsigned pe = part.numa_start_of( p+1 );
	    intptr_t pmem_e = intptr_t(mem) + part.template start_of<byV>(pe)*sizeof(T);
	    size_t size = pmem_e - pmem;
                intptr_t pmem_er = round_page(pmem_e,pgsz,size);
		// size = std::min( pmem_rounded + size, intptr_t(mem) + totalSize ) - pmem_rounded;
		size = pmem_er - pmem;
                bind_pages(reinterpret_cast<void*>(pmem),size,mflag,p);
/*
		std::cerr << "bind partition " << perNode*p
			  << " pmem=" << pmem
			  << " pmem_er=" << pmem_er
			  << " size=" << size
			  << " idx=" << part.start_of(perNode*p)
			  << "-" << part.start_of(perNode*(p+1)) << "\n";
*/
#if 0
                cout << "mem=" << mem << " rnd=" << (void*)pmem_rounded
				 << " size=" <<size
				 << " totalSize=" <<totalSize
				 << " Page_Size=" <<page_size
				 << " end of range= " << (void*)((char*)mem+totalSize)
				 << "\n";
#endif
                pmem +=size;
		// }
        }
       //mmap_alloc+=part_alloc.next();
    }

    [[deprecated("replace by interface with allocator class to support changes to partitioner class")]]
    void Interleave_allocate(size_t elements, size_t align = 0) {
	alloc_interleaved( elements, align );
    }

    private:
    void alloc_interleaved(size_t elements, size_t align = 0)
    {
	// assert( 0 );
        if( totalSize!=0 ||mem != 0 )
        {
            cerr<<"Interleave already allocated\n";
            abort();
        }

	if( elements == 0 ) {
	    totalSize = 0;
	    mem = nullptr;
	    return;
	}
	
        totalSize = elements*sizeof(T);
        if((totalSize % page_size) !=0)
        {
           totalSize = (((totalSize+page_size-1)/ page_size)) * page_size;
	}
	int err;
#if defined(DMALLOC)
	mem = malloc( totalSize+page_size-1 );
	err = errno;
#elif HUGE_PAGES
        // mem = mmap( 0, totalSize, MM_PROTECTED, MM_FLAGS | MAP_HUGETLB, 0, 0);
	if( (err = posix_memalign( &mem, page_size, totalSize )) < 0 )
	    mem = NULL;
#else
        mem = mmap( 0, totalSize, MM_PROTECTED, MM_FLAGS, 0, 0);
	err = errno;
#endif
        if( mem == (void *)-1|| mem==(void*)0)
        {
            std::cerr << "numa interleave mmap failed: " << strerror(err) << ", size " << totalSize << '\n';
            exit(1);
        }
	MM_DEBUG_INTLV_ALLOC( elements, sizeof(T), mem, "none" );

        interleave_pages(mem,totalSize);
    }

    void local_allocate(size_t elements, size_t align, int numa_node) {
	local_allocate( elements, numa_node );
    }
    void local_allocate(size_t elements, const numa_allocation_local & alloc) {
	local_allocate( elements, alloc.node() );
    }	
    void local_allocate(size_t elements, int numa_node)
    {
        if( totalSize!=0 ||mem != 0 )
        {
            cerr<<"local NUMA already allocated\n";
            abort();
        }
        totalSize = elements*sizeof(T);
        if((totalSize % page_size) !=0)
        {
           totalSize = (((totalSize+page_size-1)/ page_size)) * page_size;

        }
	int err;
#if defined(DMALLOC)
	mem = malloc( totalSize+page_size-1 );
	err = errno;
#elif HUGE_PAGES
	if( (err = posix_memalign( &mem, page_size, totalSize )) < 0 )
	    mem = NULL;
#else
        mem = mmap( NULL, totalSize, MM_PROTECTED, MM_FLAGS, 0, 0);
	err = errno;
#endif
        if( mem == (void *)-1|| mem==(void*)0)
        {
            std::cerr << "numa-node mmap failed: " << strerror(err) << ", size " << totalSize << '\n';
            exit(1);
        }
	MM_DEBUG_LOCAL_ALLOC( elements, sizeof(T), numa_node, mem, "none" );
        bind_pages(mem,totalSize,mflag,numa_node);
    }

private:
    intptr_t round_page(intptr_t ptr, intptr_t pagesize, size_t & sz)
    {
        intptr_t ret = (ptr + (pagesize-1)) & ~intptr_t(pagesize - 1);
        assert( (ret & (pagesize - 1)) == 0 );

	intptr_t size = sz;
	size += (pagesize-1) - ((ptr + pagesize-1) & intptr_t(pagesize -1));
        size = (size + pagesize-1) & ~intptr_t(pagesize - 1);
        assert( (size & (pagesize - 1)) == 0 );
	sz = size;
        return ret;
    }

    void bind_pages(void * mem, size_t size,int policy, int numa_node)
    {
        struct bitmask *bmp;
        bmp = numa_allocate_nodemask();
        numa_bitmask_setbit(bmp, numa_node);
        if (mem == (void *)-1)
            mem = NULL;
        else
            dombind(mem, size, policy, bmp);
        numa_bitmask_free(bmp);
    }

    void interleave_pages(void * mem, size_t size)
    {
        struct bitmask *bmp;
        bmp = numa_allocate_nodemask();
        numa_bitmask_setall(bmp);
        if (mem == (void *)-1)
            mem = NULL;
        else
            dombind(mem, size, MPOL_INTERLEAVE, bmp);
        numa_bitmask_free(bmp);
    }

    static void dombind(void *mem, size_t size, int policy, struct bitmask *bmp)
    {
       if(mbind(mem, size, policy, bmp ? bmp->maskp : NULL, bmp ? bmp->size : 0, 0 )<0)
            std::cerr << "mbind failed: " << strerror(errno) 
		      << " address " << (size_t*)mem
		      << ", size " << size << '\n';
    }

public:
    void clear() {
	mem = nullptr;
	totalSize = 0;
    }
    void del( const char * msg = nullptr ) {
        timer del;
        del.start();
        if( mem )
        {
	    MM_DEBUG_DEL( mem, msg );
#if HUGE_PAGES
            free( mem );
#else
            int munmapres = munmap( mem, totalSize );
            if(munmapres == -1)
            {
                cerr<<"munmap failed "<<errno<<" "<<strerror(errno)
                    <<" address "<<mem
                    <<" and size "<<totalSize<<endl;
                abort();
            }
#endif
        }
        mem = 0;
        totalSize=0;
        del_time+=del.next();
    }

    operator T * ()
    {
        return get();
    }
    operator const T * () const
    {
        return get();
    }

    operator bool () const
    {
        return mem != 0;
    }

    size_t get_bytes() const
    {
        return totalSize;
    }

    size_t get_length() const { return totalSize/sizeof(T); }

    T *get() const
    {
#if DMALLOC
	intptr_t ret = (intptr_t)mem;
        if((ret % page_size) !=0)
	    ret += page_size - ( ret % page_size );
        return reinterpret_cast<T *>( ret );
#else
        return reinterpret_cast<T *>( mem );
#endif
    }
};

#else //NUMA

/***********************************************************************
 * Definition of mmap_ptr for systems without NUMA
 ***********************************************************************/
template <typename T>
class mmap_ptr
{
    size_t totalSize;
    T *mem;
    void *allocated;
public:
    using type = T;

    mmap_ptr():mem(nullptr),totalSize(0),allocated(nullptr) {}

    // New interface, more streamlined
    mmap_ptr( numa_allocation_partitioned alloc ) : mmap_ptr() {
	generic_allocate( alloc.get_partitioner().get_vertex_range(),
			  MIN_ALIGN );
	MM_DEBUG_INTLV_ALLOC( alloc.get_partitioner().get_vertex_range(),
			      sizeof(T), mem, "none" );
    }
    mmap_ptr( numa_allocation_edge_partitioned alloc ) : mmap_ptr() {
	generic_allocate( alloc.get_partitioner().get_edge_range(),
			  MIN_ALIGN );
	MM_DEBUG_INTLV_ALLOC( alloc.get_partitioner().get_edge_range(),
			      sizeof(T), mem, "none" );
    }
    // For uniformity
    mmap_ptr( size_t elements, numa_allocation_partitioned alloc ) : mmap_ptr() {
	assert( elements == alloc.get_partitioner().get_vertex_range() );
	generic_allocate( alloc.get_partitioner().get_vertex_range(),
			  MIN_ALIGN );
	MM_DEBUG_INTLV_ALLOC( elements, sizeof(T), mem, "none" );
    }
    mmap_ptr( size_t elements, numa_allocation_interleaved alloc ) : mmap_ptr() {
	generic_allocate( elements, MIN_ALIGN );
	MM_DEBUG_INTLV_ALLOC( elements, sizeof(T), mem, "none" );
    }
    mmap_ptr( size_t elements, numa_allocation_local alloc ) : mmap_ptr() {
	generic_allocate( elements, MIN_ALIGN );
	MM_DEBUG_INTLV_ALLOC( elements, sizeof(T), mem, "none" );
    }
    mmap_ptr( size_t elements, const numa_allocation & alloc ) : mmap_ptr() {
	generic_allocate( elements, MIN_ALIGN );
	MM_DEBUG_INTLV_ALLOC( elements, sizeof(T), mem, "none" );
    }
    mmap_ptr( size_t elements, int fd, off_t off,
	      const numa_allocation & alloc ) : mmap_ptr() {
	generic_allocate( elements, fd, off );
	MM_DEBUG_INTLV_ALLOC( elements, sizeof(T), mem, "none" );
    }
    // Default is interleaved allocation
    mmap_ptr( size_t elements )
	: mmap_ptr( elements, numa_allocation_interleaved() ) { }
    void allocate( const numa_allocation_partitioned & alloc ) {
	generic_allocate( alloc.get_partitioner().get_vertex_range(),
			  MIN_ALIGN );
	MM_DEBUG_INTLV_ALLOC( alloc.get_partitioner().get_vertex_range(),
			      sizeof(T), mem, "none" );
    }
    void allocate( const numa_allocation_edge_partitioned & alloc ) {
	generic_allocate( alloc.get_partitioner().get_edge_range(),
			  MIN_ALIGN );
	MM_DEBUG_INTLV_ALLOC( alloc.get_partitioner().get_edge_range(),
			      sizeof(T), mem, "none" );
    }
    void allocate( size_t elements, int fd, off_t off,
		   const numa_allocation & alloc ) {
	generic_allocate( elements, fd, off );
	MM_DEBUG_INTLV_ALLOC( elements, sizeof(T), mem, "none" );
    }
    void allocate( size_t elements, const numa_allocation & alloc ) {
	generic_allocate( elements, MIN_ALIGN );
	MM_DEBUG_INTLV_ALLOC( elements, sizeof(T), mem, "none" );
    }
    void allocate( size_t elements, size_t align,
		   const numa_allocation & alloc ) {
	generic_allocate( elements, align );
	MM_DEBUG_INTLV_ALLOC( elements, sizeof(T), mem, "none" );
    }

    // Old interface
    [[deprecated("old interface to mmap_ptr; should no longer be used")]]
    mmap_ptr(const partitioner & part)
    {
        // Constructor intended for frontiers
        // and algorithm-specific vertex arrays
        part_allocate( part, MIN_ALIGN );
    }
    [[deprecated("old interface to mmap_ptr; should no longer be used")]]
    mmap_ptr(size_t elements, size_t numa_node)   // NUMA-local allocation
    {
        // Constructor intended for partitioned graphs.
        local_allocate( elements, MIN_ALIGN, numa_node );
    }

    [[deprecated("old interface to mmap_ptr; should no longer be used")]]
    void part_allocate( const partitioner &part ) {
	generic_allocate( part.get_vertex_range(), MIN_ALIGN );
    }

    [[deprecated("old interface to mmap_ptr; should no longer be used")]]
    void Interleave_allocate( size_t elements, size_t align = MIN_ALIGN ) {
	generic_allocate( elements, align );
    }

    [[deprecated("old interface to mmap_ptr; should no longer be used")]]
    void local_allocate(size_t elements, int numa_node) {
	generic_allocate( elements, MIN_ALIGN );
    }

    [[deprecated("old interface to mmap_ptr; should no longer be used")]]
    void part_allocate( const partitioner &part, size_t align ) {
	generic_allocate( part.get_vertex_range(), align );
    }

    [[deprecated("old interface to mmap_ptr; should no longer be used")]]
    void local_allocate( size_t elements, size_t align, int numa_node ) {
	generic_allocate( elements, align );
    }

private:
    void generic_allocate( size_t elements ) {
        totalSize = mm_get_total_bytes<T>::get( elements );
	if( totalSize == 0 ) {
	    allocated = mem = nullptr;
	} else {
	    if( (totalSize % page_size) != 0 )
		totalSize = (((totalSize+page_size-1)/ page_size)) * page_size;
	    int err;
#if defined(DMALLOC)
	    mem = malloc( totalSize+page_size-1 );
	    err = errno;
#elif HUGE_PAGES
	    if( (err = posix_memalign( &mem, page_size, totalSize )) < 0 )
		mem = NULL;
#else
	    mem = mmap( 0, totalSize, MM_PROTECTED, MM_FLAGS ,0, 0);
	    err = errno;
#endif
	    if( mem == (void *)-1 || mem ==(void *)0 ) {
		std::cerr << "numa-node mmap failed: " << strerror(errno)
			  << ", size " << totalSize << '\n';
		exit( 1 );
	    }
	    allocated = reinterpret_cast<void *>( mem );
	}
    }
    void generic_allocate( size_t elements, size_t align ) {
        totalSize = mm_get_total_bytes<T>::get( elements );
	if( totalSize == 0 ) {
	    allocated = mem = nullptr;
	} else {
	    size_t space = totalSize + align;
	    if( (space % page_size) != 0 )
		space = (((space+page_size-1)/ page_size)) * page_size;
	    int err;
#if defined(DMALLOC)
	    allocated = (T *)malloc( totalSize+page_size-1 );
	    err = errno;
#elif HUGE_PAGES
	    if( (err = posix_memalign( &allocated, page_size, space )) < 0 )
		allocated = NULL;
#else
	    allocated = mmap( 0, space, MM_PROTECTED, MM_FLAGS ,0, 0);
	    err = errno;
#endif
	    if( allocated == (void *)-1 || allocated == (void *)0 ) {
		std::cerr << "numa-node mmap failed: " << strerror(errno)
			  << ", size " << space << '\n';
		abort();
	    }
	    void * ptr = allocated;
	    if( std::align( align, totalSize, ptr, space ) ) {
		mem = reinterpret_cast<T*>( ptr );
		assert( (size_t(mem) & (align-1)) == 0 );
	    } else {
		assert( 0 );
	    }
	    totalSize = space;
	}
	// std::cerr << "alloc " << allocated << "\n";
    }

    void generic_allocate( size_t elements, int fd, off_t off ) {
	off_t diff = 0;
        totalSize = mm_get_total_bytes<T>::get( elements );
	if( (off & off_t(small_size-1)) != 0 ) {
	    diff = off & off_t(small_size-1);
	    totalSize += diff;
	    off -= diff;
	    assert( (off & off_t(small_size-1)) == 0 );
	}
	int err;
	allocated = mmap( 0, totalSize, PROT_READ, MAP_SHARED, fd, off );
	err = errno;
	if( allocated == (void *)-1 || allocated == (void *)0 ) {
	    std::cerr << "file mmap failed: " << strerror(errno)
		      << ", size " << totalSize << '\n';
	    abort();
	}
	mem = reinterpret_cast<T *>(reinterpret_cast<char *>(allocated) + diff);
    }

public:
    void clear() {
	mem = nullptr;
	totalSize = 0;
	allocated = nullptr;
    }
    void del( const char * msg = nullptr )
    {
	// std::cerr << "dealloc " << allocated << "\n";
        if( allocated ) {
	    MM_DEBUG_DEL( mem, msg );
#if defined(DMALLOC) || HUGE_PAGES
            free( allocated );
#else
            int munmapres = munmap( allocated, totalSize );
            if(munmapres == -1)
            {
		std::cerr << "munmap failed " << strerror(errno)
			  << " address " << mem
			  << " and size " << totalSize << endl;
                abort();
	    }
#endif
	}
        mem = nullptr;
	allocated = nullptr;
        totalSize = 0;
    }

    operator T * ()
    {
        return get();
    }
    operator const T * () const
    {
        return get();
    }

    operator bool () const
    {
        return mem != 0;
    }
    size_t get_bytes() const
    {
        return totalSize;
    }
    size_t get_length() const {
	static_assert( !is_customfp_v<T>,
		       "length calculation incorrect for customfp" );
	return totalSize/sizeof(T);
    }
    T *get() const
    {
        return reinterpret_cast<T *>( mem );
    }
};

#endif // NUMA

#endif // GRAPHGRIND_MM_H
