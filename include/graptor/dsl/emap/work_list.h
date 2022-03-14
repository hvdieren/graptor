// -*- c++ -*-
#ifndef GRAPTOR_DSL_EMAP_WORK_LIST_H
#define GRAPTOR_DSL_EMAP_WORK_LIST_H

#include <cstdint>
#include <atomic>

#define MANUALLY_ALIGNED_ARRAYS 1
#define DEBUG_WORK_LIST 0

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

    // suggestion: use __sync_bool_compare_and_swap and compile with -mcx16 instead of inline asm
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
 * @param <CHUNK_> the number of work items per array buffer
 ************************************************************************/
template<typename T, unsigned CHUNK_ = 4096>
class work_list {
public:
    using type = T;
    static constexpr size_t CHUNK = CHUNK_;

    class list_node {
	template<typename T_, unsigned CHUNK__>
	friend class work_list;

    private:
	list_node( type * buf, size_t fill )
	    : m_next( nullptr ), m_buf( buf ),
	      m_fill( fill ) { }
	~list_node() = delete;

	static list_node * create() {
	    size_t size = sizeof(list_node) + CHUNK * sizeof(type);
	    uint8_t * p = new uint8_t[size];
	    list_node * l = reinterpret_cast<list_node *>( p );
	    type * b = reinterpret_cast<type *>( &p[sizeof(list_node)] );
	    new ( l ) list_node( b, 0 );

#if DEBUG_WORK_LIST
	    n_alloc_full++;
#endif
	    
	    return l;
	}
	static list_node * create( type * buf, size_t fill ) {
	    size_t size = sizeof(list_node);
	    uint8_t * p = new uint8_t[size];
	    list_node * l = reinterpret_cast<list_node *>( p );
	    new ( l ) list_node( buf, fill );

#if DEBUG_WORK_LIST
	    n_alloc_half++;
#endif

	    return l;
	}
	static void destroy( list_node * l ) {
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
	
	const T * begin() const { return m_buf; }
	const T * end() const { return &m_buf[m_fill]; }

	void push_back( T & value ) {
	    m_buf[m_fill++] = value;
	    // assert( m_fill <= CHUNK );
	}

	list_node * get_next() { return m_next; }
	const list_node * get_next() const { return m_next; }
	void set_next( list_node * p ) { m_next = p; }

	bool has_space( size_t n ) const {
	    return m_fill + n <= CHUNK;
	    // assert( m_fill <= CHUNK );
	} 
	bool is_empty() const { return m_fill == 0; }

    protected:
	list_node * m_next;
	type * m_buf;
	size_t m_fill;
    };

public:
    work_list() : m_head_tag( { 0, 0 } ) { }

    static list_node * create_list_node() {
	return list_node::create();
    };
    static list_node * create_list_node( type * buf, size_t fill ) {
	return list_node::create( buf, fill );
    };
    
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
 * @param <CHUNK_> the number of work items per array buffer
 ************************************************************************/
template<typename T, unsigned CHUNK_ = 4096>
class work_stealing {
public:
    using type = T;
    using queue_type = work_list<T,CHUNK_>;
    using buffer_type = typename queue_type::list_node;
    static constexpr size_t CHUNK = CHUNK_;

    work_stealing() : m_threads( graptor_num_threads() ),
		      m_working( m_threads ) {
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
	}
	delete[] m_active;
#if MANUALLY_ALIGNED_ARRAYS
	delete[] m_queues_alloc;
#else
	delete[] m_queues;
#endif
	buffer_type::debug();
    }

    void push( unsigned self_id, type value ) {
	reserve( self_id, 1 )->push_back( value );
    }

    void push_safe( unsigned self_id, type value ) {
	m_active[self_id]->push_back( value );
    }

    buffer_type * reserve( unsigned self_id, unsigned num ) {
	assert( num < CHUNK && "can reserve at most CHUNK elements" );
	buffer_type * buf = m_active[self_id];
	// If we don't have a buffer, or the buffer has insufficient space,
	// or if the amount of parallelism is low, then push out the buffer
	// and start a new one.
	if( !buf || !buf->has_space( num )
	    || ( 4*m_working.load() < 3*m_threads && m_threads > 1
		 && !buf->is_empty() ) ) {
	    if( buf )
		push_buffer( buf, self_id );
	    m_active[self_id] = buf = queue_type::create_list_node();
	}
	return buf;
    }

    void create_buffer( unsigned self_id, type * buf, size_t fill ) {
	if( fill > 0 )
	    push_buffer( queue_type::create_list_node( buf, fill ), self_id );
    }

    buffer_type * steal( unsigned self_id ) {
	// First try our own queue
	if( buffer_type * buf = m_queues[self_id].pop() )
	    return buf;

	// Before stealing, first consider our own active buffer
	if( m_active[self_id] && !m_active[self_id]->is_empty() ) {
	    buffer_type * buf = m_active[self_id];
	    m_active[self_id] = queue_type::create_list_node();
	    return buf;
	}

	// Note we are stealing, for termination detection
	--m_working;

	// Randomly select any buffer.
	while( m_working.load() != 0 ) {
	    unsigned id = rand() % m_threads;
	    if( buffer_type * buf = m_queues[id].pop() ) {
		++m_working;
		return buf;
	    }
	}
	return nullptr;
    }

private:
    void push_buffer( buffer_type * buf, unsigned self_id ) {
	m_queues[self_id].push( buf );
    }


private:
    unsigned m_threads;
    std::atomic<unsigned> m_working;
#if MANUALLY_ALIGNED_ARRAYS
    char * m_queues_alloc;
#endif
    queue_type * m_queues;
    buffer_type ** m_active;
};


#endif // GRAPTOR_DSL_EMAP_WORK_LIST_H

