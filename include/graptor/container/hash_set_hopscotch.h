// -*- c++ -*-
#ifndef GRAPTOR_CONTAINER_HASH_SET_HOPSCOTCH_H
#define GRAPTOR_CONTAINER_HASH_SET_HOPSCOTCH_H

#include <type_traits>
#include <algorithm>
#include <ostream>

#include "graptor/container/hash_fn.h"

/*!=====================================================================*
 * TODO:
 * + ensure that high-degree vertices are closer to their intended position
 *   than low-degree vertices; or insert high-degree vertices first.
 *   The latter is insufficient due to resizing.
 *======================================================================*/

/*!=====================================================================*
 * The array that stores the hash table is of size (1<<m_log_size)+2*H-1.
 * The first (1<<m_log_size) elements are the primary storage for the hash
 * table. The next H-1 elements are overflow area for elements incurring
 * hash collisions in the final H buckets of the primary storage.
 * The next H buckets are allocated and set to empty to allow vector access
 * to any valid bucket.
 *======================================================================*/

namespace graptor {

template<typename T, typename Hash = rand_hash<T>>
class hash_set_hopscotch {
public:
    using type = T;
    using hash_type = Hash;
    using size_type = uint32_t;
    using reference = type &;
    using const_reference = const type &;

#if __AVX512F__
    static constexpr size_type H = 64 / sizeof( type );
#else // assuming AVX2
    static constexpr size_type H = 32 / sizeof( type );
#endif

    using tr = vector_type_traits_vl<type,H>;
    using vtype = typename tr::type;

    static constexpr type invalid_element = ~type(0);

public:
    explicit hash_set_hopscotch( size_t expected_elms = 0 )
	: m_elements( 0 ),
	  m_log_size( required_log_size( expected_elms ) ),
	  m_table( new type[(1<<m_log_size)+2*H-1] ),
	  m_hash( m_log_size ),
	  m_pre_allocated( false ) {
	clear();
    }
    template<typename It>
    explicit hash_set_hopscotch( It begin, It end )
	: hash_set_hopscotch( std::distance( begin, end ) ) {
	for( It i=begin; i != end; ++i )
	    insert( *i );
    }
    hash_set_hopscotch( hash_set_hopscotch && ) = delete;
    hash_set_hopscotch( const hash_set_hopscotch & ) = delete;
    hash_set_hopscotch & operator = ( const hash_set_hopscotch & ) = delete;

    ~hash_set_hopscotch() {
	if( !m_pre_allocated )
	    delete[] m_table;
    }

    void clear() {
	m_elements = 0;
	std::fill( m_table, m_table+capacity()+2*H-1, invalid_element );
    }

    size_type size() const { return m_elements; }
    size_type capacity() const { return size_type(1) << m_log_size; }
    bool empty() const { return size() == 0; }

    const type * get_table() const { return m_table; }

    auto begin() const { return m_table; }
    auto end() const { return m_table+capacity()+H; }

    static size_type required_log_size( size_type num_elements ) {
	// Maintain fill factor of 50% at most
	// Make sure at least one size of H elements is included
	// (adding 2 to the log will make the table at least 4H elements).
	return rt_ilog2( std::max( H, num_elements ) ) + 2;
    }

    bool insert( type value ) {
	size_type home_index = m_hash( value ) & ( capacity() - 1 );
	size_type index = home_index;
	vtype v = tr::set1( value );
	vtype vinv = tr::setone();

	// We don't need wrap-around in this, as we allocated H extra positions.
	vtype data = tr::loadu( &m_table[index] );

	// Try to find the element.
	if( !tr::is_zero( tr::cmpeq( data, v, target::mt_vmask() ) ) )
	    return false; // do not insert if already present

	// Try to find an empty slot in the next H buckets.
	typename tr::mask_type e = tr::cmpeq( data, vinv, target::mt_mask() );
	if( e != 0 ) {
	    size_type lane = tr::mask_traits::tzcnt( e );
	    m_table[index + lane] = value;
	    ++m_elements;
	    return true;
	}

	// Find first empty bucket.
	// Note: an additional H buckets have been allocated and initialized
	// to empty to support vector access to the usable buckets.
	for( ; index < capacity()+H-1; index += H ) {
	    vtype data = tr::loadu( &m_table[index] );
	    e = tr::cmpeq( data, vinv, target::mt_mask() );
	    if( e != 0 )
		break;
	}

	// Found an empty bucket, move it closer to home_index
	if( e != 0 ) {
	    size_type lane = tr::mask_traits::tzcnt( e );
	    size_type free_index = index + lane;

	    // Do the hopscotch trick
	    index = hopscotch_move( value, home_index, free_index );

	    // Only if we found a moveable element...
	    if( ~index != size_type(0) ) {
		assert( index - home_index < H );
		assert( m_table[index] == invalid_element );

		m_table[index] = value;
		++m_elements;
		return true;
	    }
	}

	// Could not identify a free bucket close enough to the home_index.
	// Resize and retry.
	assert( !m_pre_allocated && "Cannot resize if not owning the storage" );

	// Rehash
	size_type old_log_size = m_log_size + 1;
	type * old_table = new type[(size_type(1)<<old_log_size) + 2*H-1];
	using std::swap;
	swap( old_log_size, m_log_size );
	swap( old_table, m_table );
	clear(); // sets m_elements=0; will be reset when rehashing

	size_type old_size = size_type(1) << old_log_size;
	m_hash.resize( m_log_size );
	for( size_type i=0; i < old_size+H; ++i )
	    if( old_table[i] != invalid_element )
		insert( old_table[i] );
	delete[] old_table;

	// Retry insertion. Hope for tail recursion optimisation.
	return insert( value );
    }

    template<typename It>
    void insert( It && I, It && E ) {
	while( I != E )
	    insert( *I++ );
    }

    bool contains( type value ) const {
	size_type index = m_hash( value ) & ( capacity() - 1 );
	vtype v = tr::set1( value );

	// We don't need wrap-around in this, as we allocated H extra positions.
	vtype data = tr::loadu( &m_table[index] );

	// Check for presence of value.
	return !mtr::is_zero( tr::cmpeq( data, v, mkind() ) );
    }

    template<typename U, unsigned short VL, typename MT>
    std::conditional_t<std::is_same_v<MT,target::mt_mask>,
	typename vector_type_traits_vl<U,VL>::mask_type,
	typename vector_type_traits_vl<U,VL>::vmask_type>
    multi_contains( typename vector_type_traits_vl<U,VL>::type
			 v, MT ) const {
	static_assert( sizeof( U ) == sizeof( type ) );
	using tr = vector_type_traits_vl<type,VL>;
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
	const vtype one = tr::setoneval();
	const vtype hmask = tr::srli( ones, tr::B - m_log_size );

	vtype hval = m_hash.template vectorized<VL>( v );
	vtype vidx = tr::bitwise_and( hval, hmask );

	mtype notfound = mtr::setone();

	for( size_type h=0; h < H && !mtr::is_zero( notfound ); ++h ) {
	    // This code is independent of the value returned by gather
	    // for inactive lanes. Inactive lanes have already been matched,
	    // and the bitwise_and will keep them so regardless of the value
	    // returned by gather for that lane.
	    vtype e = tr::gather( m_table, vidx, notfound );
	    vidx = tr::add( vidx, one );
	    mtype fnd = tr::cmpne( e, v, mkind() );
	    notfound = mtr::bitwise_and( fnd, notfound );
	}

	mtype found = tr::bitwise_invert( notfound );
	if constexpr ( std::is_same_v<MT,mkind> )
	    return found;
	else if constexpr ( std::is_same_v<MT,target::mt_mask> )
	    return tr::asmask( found );
	else
	    return tr::asvector( found );
    }

    template<typename Fn>
    void for_each( Fn && fn ) const {
	for( size_type i=0; i < capacity()+H; ++i )
	    if( m_table[i] != invalid_element )
		fn( m_table[i] );
    }

private:
    size_type
    hopscotch_move( type v, size_type home_index, size_type free_index ) {
	while( free_index - home_index >= H ) {
	    bool fnd = false;
	    for( size_type b=0; b < H-1; ++b ) {
		if( free_index >= H - 1 + b ) {
		    size_type idx = free_index - ( H - 1 ) + b;
		    size_type h = m_hash( m_table[idx] ) & ( capacity() - 1 );
		    // Element is movable
		    if( ( free_index - h ) < H ) {
			m_table[free_index] = m_table[idx];
			m_table[idx] = invalid_element; // redundant
			free_index = idx;
			fnd = true;
			break;
		    }
		}
	    }
	    if( !fnd )
		return ~size_type(0);
	}

	return free_index;
    }

private:
    size_type m_elements;
    size_type m_log_size;
    type * m_table;
    hash_type m_hash;
    bool m_pre_allocated;
};

} // namespace graptor

#endif // GRAPTOR_CONTAINER_HASH_SET_HOPSCOTCH_H
