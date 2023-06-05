// -*- c++ -*-
#ifndef GRAPTOR_GRAPH_SIMPLE_HASH_TABLE_H
#define GRAPTOR_GRAPH_SIMPLE_HASH_TABLE_H

#include <vector>
#include <algorithm>
#include <ostream>

#include "graptor/graph/simple/conditional_iterator.h"

namespace graptor {

namespace graph {
    
template<typename T, typename Hash = std::hash<T>>
class hash_table {
public:
    typedef T type;
    typedef Hash hash_type;
    typedef size_t size_type;
    typedef type & reference;
    typedef const type & const_reference;

    static constexpr type invalid_element = ~type(0);

public:
    explicit hash_table()
	: m_elements( 0 ),
	  m_size( 16 ),
	  m_table( new type[16] ) {
	assert( ( m_size & ( m_size - 1 ) ) == 0 && "size must be power of 2" );
	clear();
    }
    hash_table( hash_table && ) = delete;
    hash_table( const hash_table & ) = delete;
    hash_table & operator = ( const hash_table & ) = delete;

    ~hash_table() {
	delete[] m_table;
    }

    void clear() {
	m_elements = 0;
	std::fill( m_table, m_table+m_size, invalid_element );
    }

    size_type size() const { return m_elements; }
    bool empty() const { return size() == 0; }

    auto begin() const { return m_table; }
    auto end() const { return m_table+m_size; }

/*
    auto begin() const {
	return graptor::graph::make_conditional_iterator(
	    m_table, [&]( const type * && it ) {
		return it != m_table+m_size && *it != invalid_element;
	    } );
    }
    auto end() const {
	return graptor::graph::make_conditional_iterator(
	    m_table+m_size, [&]( const type * && it ) {
		return it != m_table+m_size && *it != invalid_element;
	    } );
    }
*/

    bool insert( type value ) {
	size_type index = m_hash( value ) & ( m_size - 1 );
	while( m_table[index] != invalid_element && m_table[index] != value )
	    index = ( index + 1 ) & ( m_size - 1 );
	if( m_table[index] == value ) {
	    return false;
	} else {
	    if( (m_elements+1) >= ( m_size >> 1 ) ) {
		// Rehash
		size_type old_size = m_size << 1;
		type * old_table = new type[old_size];
		using std::swap;
		swap( old_size, m_size );
		swap( old_table, m_table );
		clear(); // sets m_elements=0; will be reset when rehashing
		for( size_type i=0; i < old_size; ++i )
		    if( old_table[i] != invalid_element )
			insert( old_table[i] );
		delete[] old_table;
		return insert( value );
	    } else {
		++m_elements;
		m_table[index] = value;
		return true;
	    }
	}
    }

    template<typename It>
    void insert( It && I, It && E ) {
	while( I != E )
	    insert( *I++ );
    }

    bool contains( type value ) const {
	size_type index = m_hash( value ) & ( m_size - 1 );
	while( m_table[index] != invalid_element && m_table[index] != value )
	    index = ( index + 1 ) & ( m_size - 1 );
	return m_table[index] == value;
    }

    template<typename Fn>
    void for_each( Fn && fn ) const {
	for( size_type i=0; i < m_size; ++i )
	    if( m_table[i] != invalid_element )
		fn( m_table[i] );
    }

    template<typename It, typename Ot>
    Ot intersect( It I, It E, Ot O ) const {
	while( I != E ) {
	    VID v = *I;
	    if( contains( v ) ) {
		*O++ = v;
	    }
	    ++I;
	}
	return O;
    }

    template<typename It>
    size_t intersect_size( It I, It E, size_t exceed ) const {
	size_t sz = 0;
	size_t todo = std::distance( I, E );
	while( I != E ) {
	    if( sz + todo <= exceed )
		return 0;

	    VID v = *I;
	    if( contains( v ) )
		++sz;

	    ++I;
	    --todo;
	}
	return sz;
    }
	    
private:
    size_type m_elements;
    size_type m_size;
    type * m_table;
    hash_type m_hash;
};


/*
template<typename VID>
ostream & operator << ( ostream & os, const hash_table<VID> & s ) {
    os << "{ #" << s.size() << ": ";
    for( auto I=s.begin(), E=s.end(); I != E; ++I )
	os << ' ' << *I;
    os << " }";
    return os;
}
*/

} // namespace graph

} // namespace graptor

#endif // GRAPTOR_GRAPH_SIMPLE_HASH_TABLE_H
