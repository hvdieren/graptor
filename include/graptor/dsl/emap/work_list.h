// -*- c++ -*-
#ifndef GRAPTOR_DSL_EMAP_WORK_LIST_H
#define GRAPTOR_DSL_EMAP_WORK_LIST_H

#include <atomic>

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
	    return l;
	}
	static list_node * create( type * buf, size_t fill ) {
	    size_t size = sizeof(list_node);
	    uint8_t * p = new uint8_t[size];
	    list_node * l = reinterpret_cast<list_node *>( p );
	    new ( l ) list_node( buf, fill );
	    return l;
	}
	static void destroy( list_node * l ) {
	    delete[] reinterpret_cast<uint8_t *>( l );
	}

    public:
	void destroy() { return destroy( this ); }
	
	const T * begin() const { return m_buf; }
	const T * end() const { return &m_buf[m_fill]; }

	void push_back( T & value ) { m_buf[m_fill++] = value; }

	list_node * get_next() { return m_next; }
	const list_node * get_next() const { return m_next; }
	void set_next( list_node * p ) { m_next = p; }

	bool has_space( size_t n ) const { return m_fill + n <= CHUNK; } 
	bool is_empty() const { return m_fill == 0; }

    protected:
	list_node * m_next;
	type * m_buf;
	size_t m_fill;
    };

public:
    work_list() : m_head( nullptr ) { }

    static list_node * create_list_node() {
	return list_node::create();
    };
    static list_node * create_list_node( type * buf, size_t fill ) {
	return list_node::create( buf, fill );
    };
    
    list_node * pop() {
	list_node * old;
	do {
	    old = m_head.load();
	    if( old == nullptr )
		return nullptr;
	} while( !m_head.compare_exchange_weak( old, old->get_next() ) );
	return old;
    }
    void push( list_node * n ) {
	if( n->is_empty() ) {
	    n->destroy();
	    return;
	}

	list_node * old;
	do {
	    old = m_head.load();
	    n->set_next( old );
	} while( !m_head.compare_exchange_weak( old, n ) );
    }

private:
    std::atomic<list_node *> m_head;
};

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
	m_queues = new queue_type[m_threads]();
	m_active = new buffer_type *[m_threads];
	for( unsigned t=0; t < m_threads; ++t )
	    m_active[t] = nullptr;
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
	if( !buf || !buf->has_space( num ) ) {
	    if( buf )
		push_buffer( buf, self_id );
	    m_active[self_id] = buf = queue_type::create_list_node();
	}
	return buf;
    }

    void create_buffer( unsigned self_id, type * buf, size_t fill ) {
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
	if( m_working.fetch_add( -1 ) == 0 )
	    return nullptr;

	// Randomly select any buffer.
	while( true ) {
	    unsigned id = rand() % m_threads;
	    if( buffer_type * buf = m_queues[id].pop() ) {
		++m_working;
		return buf;
	    }

	    if( m_working.load() == 0 )
		return nullptr;
	}
    }

private:
    void push_buffer( buffer_type * buf, unsigned self_id ) {
	m_queues[self_id].push( buf );
    }


private:
    unsigned m_threads;
    std::atomic<unsigned> m_working;
    queue_type * m_queues;
    buffer_type ** m_active;
};


#endif // GRAPTOR_DSL_EMAP_WORK_LIST_H

