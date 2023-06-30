// -*- c++ -*-
#ifndef GRAPTOR_GRAPH_SIMPLE_HASHED_SET_H
#define GRAPTOR_GRAPH_SIMPLE_HASHED_SET_H

#include <vector>
#include <algorithm>
#include <ostream>

#include "graptor/container/conditional_iterator.h"

namespace graptor {

namespace graph {
    
template<typename T, typename Hash = std::hash<T>>
class hashed_set {
public:
    typedef T type;
    typedef Hash hash_type;
    typedef size_t size_type;
    typedef type & reference;
    typedef const type & const_reference;

    static constexpr type invalid_element = ~type(0);

public:
    explicit hashed_set( type * space, size_type num_elements,
			 size_type maximum_elements )
	: m_elements( num_elements ),
	  m_size( maximum_elements ),
	  m_table( space ) {
	assert( ( m_size & ( m_size - 1 ) ) == 0 && "size must be power of 2" );
    }

    void clear() {
	std::fill( m_table, m_table+m_size, invalid_element );
    }

    size_type size() const { return m_elements; }
    bool empty() const { return size() == 0; }

    auto begin() const {
	return graptor::make_conditional_iterator(
	    m_table, [&]( const type * && it ) {
		return it != m_table+m_size && *it != invalid_element;
	    } );
    }
    auto end() const {
	return graptor::make_conditional_iterator(
	    m_table+m_size, [&]( const type * && it ) {
		return it != m_table+m_size && *it != invalid_element;
	    } );
    }

    bool insert( type value ) {
	size_type index = m_hash( value ) & ( m_size - 1 );
	while( m_table[index] != invalid_element && m_table[index] != value )
	    index = ( index + 1 ) & ( m_size - 1 );
	if( m_table[index] == value ) {
	    return false;
	} else {
	    ++m_elements;
	    assert( m_elements < m_size && "hash table should never get full" );
	    m_table[index] = value;
	    return true;
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


template<typename VID>
ostream & operator << ( ostream & os, const hashed_set<VID> & s ) {
    os << "{ #" << s.size() << ": ";
    for( auto I=s.begin(), E=s.end(); I != E; ++I )
	os << ' ' << *I;
    os << " }";
    return os;
}

} // namespace graph

} // namespace graptor

#endif // GRAPTOR_GRAPH_SIMPLE_HASHED_SET_H
