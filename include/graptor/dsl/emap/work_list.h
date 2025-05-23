// -*- c++ -*-
#ifndef GRAPTOR_DSL_EMAP_WORK_LIST_H
#define GRAPTOR_DSL_EMAP_WORK_LIST_H

#include <sched.h>
#include <unistd.h>
#include <cstdint>
#include <atomic>

#include "graptor/dsl/emap/edgechunk.h"

#define MANUALLY_ALIGNED_ARRAYS 1
#define DEBUG_WORK_LIST 0

#ifndef FUSION_BLOCK_SIZE
#define FUSION_BLOCK_SIZE 2048
#endif

#ifndef FUSION_EDGE_FACTOR
#define FUSION_EDGE_FACTOR 2
#endif

#ifndef FUSION_EDGE_BALANCE
#define FUSION_EDGE_BALANCE 1
#endif

#ifndef FUSION_DROP_ZDEG
#define FUSION_DROP_ZDEG 1
#endif

#ifndef FUSION_EARLY_PUSH
#define FUSION_EARLY_PUSH 0
#endif

#if DEBUG_WORK_LIST
namespace {
    static std::atomic<size_t> n_alloc_full = 0;
    static std::atomic<size_t> n_alloc_half = 0;
    static std::atomic<size_t> n_destroy = 0;
}
#endif


/************************************************************************
 * Auxiliary for tagged pointers.
 * https://stackoverflow.com/questions/4825400/cmpxchg16b-correct
 ************************************************************************/
namespace types {
    struct uint128_t {
        uint64_t lo;
        uint64_t hi;
    } __attribute__ (( __aligned__( 16 ) ));
}

inline bool cas( volatile types::uint128_t * src,
		 types::uint128_t cmp,
		 types::uint128_t with ) {
    // cmp can be by reference so the caller's value is updated on failure.

    // suggestion: use __sync_bool_compare_and_swap and compile with -mcx16
    // instead of inline asm
    bool result;
    __asm__ __volatile__
    (
        "lock cmpxchg16b %1\n\t"
        "setz %0"       // on gcc6 and later, use a flag output constraint instead
        : "=q" ( result )
        , "+m" ( *src )
        , "+d" ( cmp.hi )
        , "+a" ( cmp.lo )
        : "c" ( with.hi )
        , "b" ( with.lo )
        : "cc", "memory" // compile-time memory barrier.  Omit if you want memory_order_relaxed compile-time ordering.
    );
    return result;
}

/************************************************************************
 * A work_list that organisations work items in a linked list of array
 * chunks.
 *
 * Memory is allocated in such a way that linked list node and array
 * buffer are allocated in a single memory allocation step.
 *
 * @param <T> the type of the work items
 * @param <block_target_> the number of work items (vertices) per array buffer
 * @param <block_threshold_> threshold on edges for spilling into new block
 ************************************************************************/
template<typename T,
	 EID block_target_ = FUSION_BLOCK_SIZE,
	 EID block_threshold_ = FUSION_BLOCK_SIZE>
class work_list {
public:
    using type = T;

    static_assert( sizeof(T) == sizeof(VID) );

#if FUSION_EDGE_BALANCE
    using buffer_type = edge_buffer<VID,EID>;
#else
    using buffer_type = vertex_buffer<VID,EID>;
#endif

    static constexpr EID mm_block =
	block_target_ - ( sizeof(intptr_t) + sizeof(buffer_type) ) / sizeof(T);
    // mm_edge_block only used with FUSION_EDGE_BALANCE == 1
    static constexpr EID mm_edge_block = mm_block * FUSION_EDGE_FACTOR;
    static constexpr EID mm_threshold = block_threshold_;
    static_assert( mm_block > 0, "Need non-zero space to store information" );

    class list_node {
	template<typename, EID, EID>
	friend class work_list;

    private:
	list_node( type * elms )
	    : m_next( nullptr ), m_buf( elms, mm_block ) { }
	list_node( type * elms, VID fill, const EID * const idx )
	    : m_next( nullptr ), m_buf( elms, fill, fill, idx ) { }
#if FUSION_EDGE_BALANCE
	list_node( type * buf, const edge_partition<VID,EID> & ep )
	    : m_next( nullptr ), m_buf( buf, ep ) { }
#endif
	~list_node() = delete;

	bool is_pre_init() const {
	    const uint8_t * p = reinterpret_cast<const uint8_t *>( this );
	    const type * b = reinterpret_cast<const type *>(
		&p[sizeof(list_node)] );
	    return b != m_buf.get_vertices();
	}

	static list_node * create() {
	    constexpr size_t size = sizeof(list_node) + mm_block * sizeof(type);
	    static_assert( size == sizeof(type) * block_target_ );
	    uint8_t * p = new uint8_t[size];
	    list_node * l = reinterpret_cast<list_node *>( p );
	    type * b = reinterpret_cast<type *>( &p[sizeof(list_node)] );
	    new ( l ) list_node( b );

#if DEBUG_WORK_LIST
	    n_alloc_full++;
#endif
	    
	    return l;
	}
	static list_node * create( type * buf, size_t fill,
				   const EID * idx ) {
	    size_t size = sizeof(list_node);
	    uint8_t * p = new uint8_t[size];
	    list_node * l = reinterpret_cast<list_node *>( p );
	    new ( l ) list_node( buf, fill, idx );

#if DEBUG_WORK_LIST
	    n_alloc_half++;
#endif

	    return l;
	}
#if FUSION_EDGE_BALANCE
	static list_node * create( type * buf,
				   const edge_partition<VID,EID> & ep ) {
	    size_t size = sizeof(list_node);
	    uint8_t * p = new uint8_t[size];
	    list_node * l = reinterpret_cast<list_node *>( p );
	    new ( l ) list_node( buf, ep );

#if DEBUG_WORK_LIST
	    n_alloc_half++;
#endif

	    return l;
	}
#endif
	static void destroy( list_node * l ) {
	    // The initial (pre-initialised) buffers may be larger than the
	    // chunk size.
	    assert( l->m_buf.size() <= mm_block || l->is_pre_init() );
#if DEBUG_WORK_LIST
	    n_destroy++;
#endif
	    delete[] reinterpret_cast<uint8_t *>( l );
	}

    public:
	void destroy() { return destroy( this ); }

	static void debug() {
#if DEBUG_WORK_LIST
	    size_t check = ( (size_t)n_alloc_full
			     + (size_t)n_alloc_half
			     - (size_t)n_destroy );
	    std::cerr << "n_alloc_full: " << (size_t)n_alloc_full
		      << " n_alloc_half: " << (size_t)n_alloc_half
		      << " n_destroy: " << (size_t)n_destroy
		      << " check: " << check
		      << "\n";
	    assert( check == 0 );
	    n_alloc_full = 0;
	    n_alloc_half = 0;
	    n_destroy = 0;
#endif
	}
	
	auto vertex_begin() const {
	    return m_buf.vertex_begin();
	}
	auto vertex_end() const {
	    return m_buf.vertex_end();
	}
	auto edge_begin( const EID * idx ) const {
	    return m_buf.edge_begin( idx );
	}
	auto edge_end( const EID * idx ) const {
	    return m_buf.edge_end( idx );
	}

	void push_back( const T & value, const EID * const idx ) {
	    m_buf.push_back( value, idx );
	    assert( m_buf.size() <= mm_block );
	}

	void close( const EID * idx ) {
	    m_buf.close( idx );
	}

	bool has_space( VID v, EID e ) const {
	    return m_buf.has_space( v, e, mm_edge_block );
	} 
	bool is_empty() const {
	    return m_buf.is_empty();
	}
	VID size() const {
	    return m_buf.size();
	}
#if FUSION_EDGE_BALANCE
	EID esize() const {
	    return m_buf.esize();
	}
#endif
	void set( const edge_partition<VID,EID> & ep,
		  const VID * const v = nullptr ) {
	    m_buf.set( ep, v );
	}

	list_node * get_next() { return m_next; }
	const list_node * get_next() const { return m_next; }
	void set_next( list_node * p ) { m_next = p; }

    protected:
	list_node * m_next;
	buffer_type m_buf;
    };

public:
    work_list() : m_head_tag( { 0, 0 } ) { }

    static list_node * create_list_node() {
	return list_node::create();
    };
    static list_node * create_list_node( type * buf, size_t fill,
					 const EID * idx ) {
	return list_node::create( buf, fill, idx );
    };
#if FUSION_EDGE_BALANCE
    static list_node * create_list_node( type * buf,
					 const edge_partition<VID,EID> & ep ) {
	return list_node::create( buf, ep );
    };
#endif
    
    void setup( list_node * n ) {
	m_head_tag.lo = reinterpret_cast<uint64_t>( n );
    }
    
    list_node * pop() {
	types::uint128_t old;
	types::uint128_t upd;
	list_node * old_ptr;
	do {
	    old = m_head_tag;
	    old_ptr = reinterpret_cast<list_node *>( old.lo ); 
	    if( old_ptr == nullptr )
		return nullptr;
	    upd.lo = reinterpret_cast<uint64_t>( old_ptr->m_next );
	    upd.hi = old.hi + 1;
	} while( !cas( &m_head_tag, old, upd ) );
	return reinterpret_cast<list_node *>( old.lo );
    }
    void push( list_node * n ) {
	if( n->is_empty() ) {
	    n->destroy();
	    return;
	}

	types::uint128_t old, upd;
	do {
	    old = m_head_tag;
	    n->m_next = reinterpret_cast<list_node *>( old.lo );
	    upd.lo = reinterpret_cast<uint64_t>( n );
	    upd.hi = old.hi + 1;
	} while( !cas( &m_head_tag, old, upd ) );
    }
    bool is_empty() const {
	return m_head_tag.lo == 0;
    }

private:
    types::uint128_t m_head_tag;
    char m_padding[64-sizeof(types::uint128_t)];
} __attribute__ (( __aligned__( 16 ) ));

/************************************************************************
 * A work stealing structure, based on one work_list per thread.
 *
 * @param <T> the type of the work items
 * @param <retain_chunks> remember all chunks active at the end
 ************************************************************************/
template<typename T, bool retain_chunks>
class work_stealing {
public:
    using type = T;
    using queue_type = work_list<T>;
    using buffer_type = typename queue_type::list_node;

    work_stealing( unsigned threads, const GraphCSx & G )
	: m_threads( threads ), m_working( m_threads ), m_processed( nullptr ),
	  m_G( G ) {
#if MANUALLY_ALIGNED_ARRAYS
	size_t space = sizeof(queue_type)*m_threads+16;
	m_queues_alloc = new char[space];
	void * ptr = reinterpret_cast<void *>( m_queues_alloc );
	m_queues = 
	    reinterpret_cast<queue_type *>(
		std::align( 16, sizeof(queue_type)*m_threads, ptr, space ) );
	for( unsigned t=0; t < m_threads; ++t )
	    new ( &m_queues[t] ) queue_type();
#else
	m_queues = new queue_type[m_threads]();
#endif
	// Check alignment to 16 bytes
	assert( ( reinterpret_cast<intptr_t>( &m_queues[0] ) & intptr_t(0xf) ) == 0 );
	assert( ( reinterpret_cast<intptr_t>( &m_queues[1] ) & intptr_t(0xf) ) == 0 );

	m_active = new buffer_type *[m_threads];
	for( unsigned t=0; t < m_threads; ++t )
	    m_active[t] = nullptr;

	if constexpr ( retain_chunks ) {
	    m_processed = new buffer_type *[m_threads];
	    for( unsigned t=0; t < m_threads; ++t )
		m_processed[t] = nullptr;
	}
    }

    ~work_stealing() {
	for( unsigned t=0; t < m_threads; ++t ) {
	    assert( m_queues[t].is_empty() );
#if MANUALLY_ALIGNED_ARRAYS
	    m_queues[t].~queue_type(); // actually default and empty
#endif
	    if( m_active[t] ) {
		assert( m_active[t]->is_empty() );
		m_active[t]->destroy();
	    }
	    if constexpr ( retain_chunks ) {
		buffer_type * buf = m_processed[t];
		while( buf != nullptr ) {
		    buffer_type * nxt = buf->get_next();
		    buf->destroy();
		    buf = nxt;
		}
	    }
	}
	delete[] m_active;
	if constexpr ( retain_chunks )
	    delete[] m_processed;
#if MANUALLY_ALIGNED_ARRAYS
	delete[] m_queues_alloc;
#else
	delete[] m_queues;
#endif
	buffer_type::debug();
    }

    void finished( unsigned self_id, buffer_type * buf ) {
	if constexpr ( retain_chunks ) {
	    buf->set_next( m_processed[self_id] );
	    m_processed[self_id] = buf;
	} else
	    buf->destroy();
    }
    
    void push( unsigned self_id, type value ) {
	const EID * const idx = m_G.getIndex();
	const EID degree = idx[value+1] - idx[value];
#if FUSION_DROP_ZDEG
	const bool do_push = degree != 0;
#else
	const bool do_push = true;
#endif
	if( do_push )
	    reserve( self_id, 1, degree )->push_back( value, m_G.getIndex() );
    }

    void push_safe( unsigned self_id, type value ) {
	m_active[self_id]->push_back( value );
	assert( 0 && "needs checking" );
    }

    buffer_type * reserve( unsigned self_id, unsigned num, EID edges ) {
	assert( num < queue_type::mm_block
		&& "can reserve at most queue_type::mm_block elements" );
	buffer_type * buf = m_active[self_id];
	// If we don't have a buffer, or the buffer has insufficient space,
	// or if the amount of parallelism is low, then push out the buffer
	// and start a new one.
	if( !buf )
	    m_active[self_id] = buf = queue_type::create_list_node();
	else if( !buf->is_empty() ) {
	    if( !buf->has_space( num, edges )
#if FUSION_EARLY_PUSH
		|| 4*m_working.load() < 3*m_threads && m_threads > 1
#endif
		) {
		push_buffer( buf, self_id );
		m_active[self_id] = buf = queue_type::create_list_node();
	    }
	}
	return buf;
    }

    void create_buffer( unsigned self_id, type * buf, size_t fill,
			const EID * idx ) {
	if( fill > 0 )
	    push_buffer( queue_type::create_list_node( buf, fill, idx ),
			 self_id );
    }

    buffer_type * steal( unsigned self_id ) {
	// First try our own queue
	if( buffer_type * buf = m_queues[self_id].pop() )
	    return buf;

	// Before stealing, first consider our own active buffer
	if( m_active[self_id] && !m_active[self_id]->is_empty() ) {
	    buffer_type * buf = m_active[self_id];
	    m_active[self_id] = queue_type::create_list_node();
	    // buf->close( m_G.getIndex() );
	    buf = close_and_split( buf, self_id );
	    return buf;
	}

	// Note we are stealing, for termination detection
	--m_working;

	// Randomly select any buffer.
	// Sleeping protocol based of Parlay, which cites
	// https://raw.githubusercontent.com/ogiroux/atomic_wait/master/include/atomic_wait
	long history = 10;
	while( m_working.load() != 0 ) {
	    unsigned id = rand() % m_threads;
	    if( buffer_type * buf = m_queues[id].pop() ) {
		++m_working;
		return buf;
	    }

	    // If failing to steal, sleep increasing amounts
	    sched_yield();
/*
            usleep( history >> 2 );
            history += history >> 2;
            if( history > (1 << 10) )
                history = 1 << 10;
*/
	}
	return nullptr;
    }

    void shift_processed() {
	for( unsigned t=0; t < m_threads; ++t ) {
	    m_queues[t].setup( m_processed[t] );
	    m_processed[t] = nullptr;
	}
	m_working = m_threads;
    }

private:
    void push_buffer( buffer_type * buf, unsigned self_id ) {
	// buf->close( m_G.getIndex() );
	buf = close_and_split( buf, self_id );
	m_queues[self_id].push( buf );
    }

    buffer_type * close_and_split( buffer_type * buf, unsigned self_id ) {
	const EID * const idx = m_G.getIndex();

	// Close buffer (book-keeping)
	buf->close( idx );

#if FUSION_EDGE_BALANCE == 1
	// Check if it is worthwhile to split the buffer.
	if( m_working.load() >= m_threads || m_threads == 1 )
	    return buf;

	// For every part, push it to the queue. Return one part that
	// has not been pushed.
	VID m = buf->size();
	EID mm = buf->esize();
	VID mm_parts = std::min( VID( m_threads ) * 4,
				 VID( ( mm + queue_type::mm_edge_block - 1 )
				      / queue_type::mm_edge_block ) );

	// If there are no edges, just return the buffer
	// If there are few parts, don't split
	if( mm == 0 || mm_parts == 1 )
	    return buf;

	// std::cerr << "m=" << m << " mm=" << mm << " parts=" << mm_parts << "\n";

	// Calculate info on the degrees of the vertices and total number of
	// edges
	EID * degree = new EID[2*m+1];
	const VID * const s = buf->vertex_begin();
/* called from parallel threads, m should be relatively small...
	parallel_loop( (VID)0, (VID)m, [&]( auto i ) {
	    degree[i] = idx[s[i]+1] - idx[s[i]];
	} );
*/
	for( VID i=0; i < m; ++i )
	    degree[i] = idx[s[i]+1] - idx[s[i]];

	EID* voffsets = &degree[m];
	EID mm2 = voffsets[m] = sequence::plusScan( degree, voffsets, m );
	assert( mm == mm2 );

	// Create buffers with balanced number of edges, cutting high-degree
	// vertices over multiple buffers.
	partition_edge_list<queue_type::mm_threshold,VID,EID>(
	    s, m, voffsets, idx, mm, mm_parts,
	    [&]( VID p, VID from, VID to, EID fstart, EID lend, EID offset ) {
		edge_partition<VID,EID> ep( from, to, fstart, lend, offset );
		if( p == 0 )
		    buf->set( ep );
		else {
		    // Cast away const-ness, but won't be modified
		    buffer_type * sbuf = queue_type::create_list_node();
		    
		    sbuf->set( ep, s );
		    // const_cast<VID *>( s ), ep );
		    m_queues[self_id].push( sbuf );
		}
	    } );

	delete[] degree;
#endif

	return buf;
    }


private:
    unsigned m_threads;
    std::atomic<unsigned> m_working;
#if MANUALLY_ALIGNED_ARRAYS
    char * m_queues_alloc;
#endif
    queue_type * m_queues;
    buffer_type ** m_active;
    buffer_type ** m_processed;
    const GraphCSx & m_G;
};


#endif // GRAPTOR_DSL_EMAP_WORK_LIST_H

