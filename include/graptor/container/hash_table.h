// -*- c++ -*-
#ifndef GRAPTOR_CONTAINER_HASH_TABLE_H
#define GRAPTOR_CONTAINER_HASH_TABLE_H

#include <type_traits>
#include <algorithm>
#include <utility>
#include <ostream>

#include "graptor/container/bitset.h"
#include "graptor/container/hash_fn.h"
#include "graptor/container/hash_set.h"
#include "graptor/container/conditional_iterator.h"

/*!=====================================================================*
 * TODO:
 * + ensure that high-degree vertices are closer to their intended position
 *   than low-degree vertices; or insert high-degree vertices first.
 *   The latter is insufficient due to resizing.
 *======================================================================*/

namespace graptor {

template<typename K, typename V, typename Hash = rand_hash<K>>
class hash_table {
public:
    typedef K key_type;
    typedef V value_type;
    typedef std::pair<key_type,value_type> type;
    typedef Hash hash_type;
    typedef uint32_t size_type;
    typedef type & reference;
    typedef const type & const_reference;

    static constexpr key_type invalid_element = ~key_type(0);

public:
    explicit hash_table()
	: m_elements( 0 ),
	  m_log_size( 4 ),
	  m_keys( new key_type[16] ),
	  m_values( new value_type[16] ),
	  m_hash( 4 ),
	  pre_allocated( false ) {
	clear();
    }
    explicit hash_table( size_t expected_elms )
	: m_elements( 0 ),
	  m_log_size( required_log_size( expected_elms ) ),
	  m_keys( new key_type[1<<m_log_size] ),
	  m_values( new value_type[1<<m_log_size] ),
	  m_hash( m_log_size ),
	  pre_allocated( false ) {
	clear();
    }
#if 0
    explicit hash_table( type * storage, size_type num_elements,
			 size_type log_size )
	: m_elements( num_elements ),
	  m_log_size( log_size ),
	  m_table( storage ),
	  m_hash( log_size ),
	  pre_allocated( true ) {
	clear();
    }
    explicit hash_table( type * storage, size_type num_elements,
			 size_type log_size, const hash_type & hash )
	: m_elements( num_elements ),
	  m_log_size( log_size ),
	  m_table( storage ),
	  m_hash( hash ),
	  pre_allocated( true ) {
	clear();
    }
#endif
    hash_table( hash_table && ) = delete;
    hash_table( const hash_table & ) = delete;
    hash_table & operator = ( const hash_table & ) = delete;

    ~hash_table() {
	if( !pre_allocated ) {
	    delete[] m_values;
	    delete[] m_keys;
	}
    }

    void clear() {
	m_elements = 0;
	std::fill( m_keys, m_keys+capacity(), invalid_element );
    }

    size_type size() const { return m_elements; }
    size_type capacity() const { return size_type(1) << m_log_size; }
    bool empty() const { return size() == 0; }

    const key_type * get_table() const { return m_keys; }

    auto begin() const { return m_keys; }
    auto end() const { return m_keys+capacity(); }

    static size_type required_log_size( size_type num_elements ) {
	// Maintain fill factor of 50% at most
	// ilog2( num_elements-1 ) + 1 -> next higher power of two
	// but ensure #elms+1 does not exceed 50%
	// + 1: fill factor not exceeding 50%
	return rt_ilog2( num_elements ) + 2;
    }

    bool insert( type v ) {
	return insert( v.first, v.second );
    }
    
    bool insert( key_type key, value_type value ) {
	size_type index = m_hash( key ) & ( capacity() - 1 );
	while( m_keys[index] != invalid_element && m_keys[index] != key )
	    index = ( index + 1 ) & ( capacity() - 1 );
	if( m_keys[index] == key ) {
	    m_values[index] = value;
	    return false;
	} else {
	    if( (m_elements+1) >= ( capacity() >> 1 ) ) {
		assert( !pre_allocated
			&& "Cannot resize if not owning the storage" );
		// Rehash
		size_type old_log_size = m_log_size + 1;
		key_type * old_keys = new key_type[size_type(1)<<old_log_size];
		value_type * old_values = new value_type[size_type(1)<<old_log_size];
		using std::swap;
		swap( old_log_size, m_log_size );
		swap( old_keys, m_keys );
		swap( old_values, m_values );
		clear(); // sets m_elements=0; will be reset when rehashing
		size_type old_size = size_type(1) << old_log_size;
		m_hash.resize( m_log_size );
		for( size_type i=0; i < old_size; ++i )
		    if( old_keys[i] != invalid_element )
			insert( old_keys[i], old_values[i] );
		delete[] old_keys;
		delete[] old_values;
		return insert( key, value );
	    } else {
		++m_elements;
		m_keys[index] = key;
		m_values[index] = value;
		return true;
	    }
	}
    }

    value_type contains( key_type key ) const {
	size_type index = m_hash( key ) & ( capacity() - 1 );
	while( m_keys[index] != invalid_element && m_keys[index] != key )
	    index = ( index + 1 ) & ( capacity() - 1 );
	return m_keys[index] == key ? m_values[index] : ~value_type(0);
    }

    bool contains( key_type key, value_type & ret_val ) const {
	size_type index = m_hash( key ) & ( capacity() - 1 );
	while( m_keys[index] != invalid_element && m_keys[index] != key )
	    index = ( index + 1 ) & ( capacity() - 1 ); 
	if( m_keys[index] == key ) {
	    ret_val = m_values[index];
	    return true;
	} else
	    return false;
    }

    template<typename U, unsigned short VL, typename MT>
    auto multi_contains(
	typename vector_type_traits_vl<U,VL>::type index, MT ) const {
	return multi_lookup<U,VL>( index, MT() ).first;
    }

    template<typename U, unsigned short VL, typename MT = target::mt_mask>
    auto
    multi_lookup( typename vector_type_traits_vl<U,VL>::type index, MT ) const {
	static_assert( sizeof( U ) >= sizeof( value_type ) );
	using tr = vector_type_traits_vl<U,VL>;
	using vtype = typename tr::type;
#if __AVX512F__
	using mtr = typename tr::mask_traits;
	using mkind = target::mt_mask;
#else
	using mtr = typename tr::vmask_traits;
	using mkind = target::mt_vmask;
#endif
	using mtype = typename mtr::type;

	const vtype ones = tr::setone();
	const vtype vinv = ones; // invalid_element == -1
	const vtype hmask = tr::srli( ones, tr::B - m_log_size );
	vtype v_ins = tr::setzero();

	size_type s_ins = 0;

	const vtype v = index; // Assuming &*(I+1) == (&*I)+1
	const vtype h = m_hash.template vectorized<VL>( v );
	vtype vidx = tr::bitwise_and( h, hmask );
	vtype e = tr::gather( m_keys, vidx );
	const mtype imi = tr::cmpne( e, vinv, mkind() );
	const mtype imv = tr::cmpne( e, v, mkind() );
	mtype imiv = mtr::logical_and( imi, imv );
	// Some lanes are neither empty nor the requested value and
	// need further probes.
	while( !mtr::is_zero( imiv ) ) {
	    // TODO: hash table load is targeted below 50%. Due to the
	    // birthday paradox, there will be frequent collisions still
	    // and it may be useful to make multiple vectorized probes
	    // before resorting to scalar execution. We could even count
	    // active lanes to decide.
	    vtype nvidx = tr::add( vidx, tr::setoneval() );
	    nvidx = tr::bitwise_and( nvidx, hmask );
	    vidx = tr::blend( imiv, vidx, nvidx );
	    // set default value for gather to prior e such that it acts
	    // as a blend.
	    e = tr::gather( e, m_keys, nvidx, imiv ); // use nvidx for ILP
	    const mtype imi = tr::cmpne( e, vinv, mkind() );
	    imiv = tr::cmpne( imi, e, v, mkind() );
	}
	// Return success if found, which equals imv inverted
	// It just takes one cycle to recompute, same as invert. The code
	// below is most compact to achieve the correct return type.
	// return std::make_pair( vidx, tr::cmpeq( e, v, MT() ) );
	vtype vals = tr::gather( tr::setone(), m_values, vidx,
				 tr::cmpeq( e, v, mkind() ) );
	auto mask = tr::cmpeq( e, v, MT() );
	return std::make_pair( mask, vals );
    }

private:
    size_type m_elements;
    size_type m_log_size;
    key_type * m_keys;
    value_type * m_values;
    hash_type m_hash;
    bool pre_allocated;
};

template<typename T, typename Hash>
ostream & operator << ( ostream & os, const hash_table<T,Hash> & s ) {
    os << "{ #" << s.size() << ": ";
    for( auto I=s.begin(), E=s.end(); I != E; ++I )
	os << ' ' << I->first << ':' << I->second;
    os << " }";
    return os;
}

} // namespace graptor

#endif // GRAPTOR_CONTAINER_HASH_TABLE_H
