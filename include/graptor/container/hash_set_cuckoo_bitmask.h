// -*- c++ -*-
#ifndef GRAPTOR_CONTAINER_HASH_SET_CUCKOO_BITMASK_H
#define GRAPTOR_CONTAINER_HASH_SET_CUCKOO_BITMASK_H

#include <type_traits>
#include <algorithm>
#include <ostream>
#include <mutex>

#include "graptor/container/hash_fn.h"

#ifdef LOAD_FACTOR
#define HASH_SET_CUCKOO_LOAD_FACTOR LOAD_FACTOR
#else
#define HASH_SET_CUCKOO_LOAD_FACTOR 1
#endif

namespace graptor {

template<typename T, typename Hash = hash_fn_pair<T,rand_hash<T>,rand_hash<T>>>
class hash_set_cuckoo_bitmask {
public:
    using type = T;
    using hash_type = Hash;
    using size_type = uint32_t;
    using reference = type &;
    using const_reference = const type &;

    static constexpr type invalid_element = ~type(0);
    static constexpr size_type max_attempts = 8;

    static constexpr size_type log_group_size = 4;
    static constexpr type group_mask = ( type(1) << log_group_size ) - 1;
    using group_t = uint16_t;

    static_assert( 8*sizeof(group_t) == size_type(1) << log_group_size,
		   "log_group_size must be log of size of group_t" );
    static_assert( sizeof(group_t) <= sizeof(type),
		   "data that is gathered must have a width not exceeding "
		   "that of a query element" );

public:
    explicit hash_set_cuckoo_bitmask()
	: m_elements( 0 ),
	  m_log_size( 0 ),
	  m_table( nullptr ),
	  m_group( nullptr ),
	  m_hash( m_log_size-1 ) { }
    explicit hash_set_cuckoo_bitmask( size_t expected_elms )
	: m_elements( 0 ),
	  m_log_size( required_log_size( expected_elms ) ),
	  m_table( new type[(1<<m_log_size)] ),
	  m_group( new group_t[(1<<m_log_size)] ),
	  m_hash( m_log_size-1 ) {
	clear();
    }
    template<typename It>
    explicit hash_set_cuckoo_bitmask( It begin, It end )
	: hash_set_cuckoo_bitmask( std::distance( begin, end ) ) {
	insert( begin, end );
    }
    hash_set_cuckoo_bitmask( hash_set_cuckoo_bitmask && ) = delete;
    hash_set_cuckoo_bitmask( const hash_set_cuckoo_bitmask & ) = delete;
    hash_set_cuckoo_bitmask & operator = ( const hash_set_cuckoo_bitmask & ) =
	delete;

    ~hash_set_cuckoo_bitmask() {
	if( m_table != nullptr )
	    delete[] m_table;
	if( m_group != nullptr )
	    delete[] m_group;
    }

    void clear() {
	if( is_initialised() ) {
	    m_elements = 0;
	    std::fill( m_table, m_table+capacity(), invalid_element );
	    // Could potentially skip initialisation as group information has
	    // no relevance when table element is invalid_element.
	    std::fill( m_group, m_group+capacity(), group_t(0) );
	}
    }

    size_type size() const { return m_elements; }
    size_type capacity() const {
	return m_log_size == 0 ? size_type(0) : size_type(1) << m_log_size;
    }
    bool empty() const { return size() == 0; }

    const type * get_table() const { return m_table; }

    auto begin() const { return m_table; }
    auto end() const { return m_table+capacity(); }

    static size_type required_log_size( size_type num_elements ) {
	// Maintain fill factor of 50% at most
	return rt_ilog2( num_elements ) + HASH_SET_CUCKOO_LOAD_FACTOR;
    }

    // Assumes insert is called sequentially
    bool insert_prev( type value, type prev ) {
	create_if_uninitialised();

	if( contains( value ) )
	    return false;

	return insert_value( value );
    }

    bool insert_next( type value, type next ) {
	create_if_uninitialised();

	if( contains( value ) )
	    return false;

	return insert_value( value );
    }

private:
    bool insert_value( type value ) {
	type off = value & group_mask;
	type gvalue = value >> log_group_size;
	group_t group = group_t(1) << off;

	size_type index = locate( value );
	if( index != ~size_type(0) ) {
	    assert( ( m_group[index] & group ) == 0 );

	    m_group[index] |= group;
	    ++m_elements;
	    return true;
	}

	return insert_gvalue( gvalue, group );
    }

    bool insert_gvalue( type gvalue, group_t group ) {
	using std::swap;
	
	for( size_type attempts=0; attempts < max_attempts; ++attempts ) {
	    size_type index1 = m_hash.fn1( gvalue ) & ( capacity()/2 - 1 );
	    swap( m_table[index1], gvalue );
	    swap( m_group[index1], group );
	    if( gvalue == invalid_element ) {
		++m_elements;
		return true;
	    }

	    size_type index2 = m_hash.fn2( gvalue ) & ( capacity()/2 - 1 );
	    swap( m_table[capacity()/2+index2], gvalue );
	    swap( m_group[capacity()/2+index2], group );
	    if( gvalue == invalid_element ) {
		++m_elements;
		return true;
	    }
	}

	// Could not identify a free bucket close enough to the home_index.
	// Or, the displacement window contains too many collisions on the
	// same home index.
	// Resize and retry.
	size_type old_log_size = m_log_size + 1;
	type * old_table = new type[(size_type(1)<<old_log_size)];
	group_t * old_group = new group_t[(size_type(1)<<old_log_size)];
	using std::swap;
	swap( old_log_size, m_log_size );
	swap( old_table, m_table );
	swap( old_group, m_group );
	clear(); // sets m_elements=0; will be reset when rehashing
	m_hash.resize( m_log_size-1 );

	size_type old_size = size_type(1) << old_log_size;
	for( size_type i=0; i < old_size; ++i )
	    if( old_table[i] != invalid_element )
		insert_gvalue( old_table[i], group );
	delete[] old_table;

	// Retry insertion. Hope for tail recursion optimisation.
	return insert_gvalue( gvalue, group );
    }

    size_type locate( type value ) const {
	if( empty() ) // also catches uninitialised case
	    return ~size_type(0);

	type gvalue = value >> log_group_size;

	size_type index1 = m_hash.fn1( gvalue ) & ( capacity()/2 - 1 );
	if( m_table[index1] == gvalue )
	    return index1;

	size_type index2 = m_hash.fn2( gvalue ) & ( capacity()/2 - 1 );
	index2 += capacity()/2;
	if( m_table[index2] == gvalue )
	    return index2;

	return ~size_type(0);
    }

public:
    bool contains( type value ) const {
	size_type index = locate( value );
	type off = value & group_mask;
	group_t grp = group_t(1) << off;
	return index != ~size_type(0) && ( m_group[index] & grp ) != 0;
    }

    template<typename U, unsigned short VL, typename MT>
    std::conditional_t<std::is_same_v<MT,target::mt_mask>,
	typename vector_type_traits_vl<U,VL>::mask_type,
	typename vector_type_traits_vl<U,VL>::vmask_type>
    __attribute__((noinline))
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

	if( empty() ) { // also catches uninitialised case
	    if constexpr ( std::is_same_v<MT,target::mt_vmask> )
		return tr::setzero();
	    else
		return tr::mask_traits::setzero();
	}

	const vtype ones = tr::setone();
	const vtype hmask = tr::srli( ones, tr::B - m_log_size + 1 );
	const vtype gmask = tr::srli( ones, tr::B - log_group_size );

	vtype vg = tr::srli( v, log_group_size );
	vtype hval1 = m_hash.template vectorized1<VL>( vg );
	vtype index1 = tr::bitwise_and( hval1, hmask );
	vtype probe1 = tr::gather( m_table, index1 );

	vtype hval2 = m_hash.template vectorized2<VL>( vg );
	vtype index2 = tr::bitwise_and( hval2, hmask );
	vtype probe2 = tr::gather( m_table+capacity()/2, index2 );

	// The top bits in each lane of group are contaminated, but we will
	// mask them out using a bitwise_and below
	vtype group1 =
	    tr::template gather_w<sizeof(group_t)>(
		reinterpret_cast<const type *>( m_group ), index1 );
	vtype group2 =
	    tr::template gather_w<sizeof(group_t)>(
		reinterpret_cast<const type *>( m_group+capacity()/2 ),
		index2 );

	vtype voff = tr::bitwise_and( v, gmask );
	vtype vpos = tr::sllv( tr::setoneval(), voff );

	vtype bits1 = tr::bitwise_and( group1, vpos );
	vtype bits2 = tr::bitwise_and( group2, vpos );
	mtype set1 = tr::cmpne( bits1, tr::setzero(), mkind() );
	mtype set2 = tr::cmpne( bits2, tr::setzero(), mkind() );

	mtype fnd1 = tr::cmpeq( set1, probe1, vg, mkind() );
	mtype fnd2 = tr::cmpeq( set2, probe2, vg, mkind() );
	mtype fnd = mtr::logical_or( fnd1, fnd2 );

	if constexpr ( std::is_same_v<MT,mkind> )
	    return fnd;
	else if constexpr ( std::is_same_v<MT,target::mt_mask> )
	    return tr::asmask( fnd );
	else
	    return tr::asvector( fnd );
    }

    std::mutex & get_lock() {
	return m_mux;
    }

    void create_if_uninitialised( size_t elements = 32 ) {
	if( !is_initialised() ) {
	    m_elements = elements;
	    m_log_size = required_log_size( m_elements );
	    m_table = new type[(1<<m_log_size)];
	    m_group = new group_t[(1<<m_log_size)];
	    m_hash.resize( m_log_size-1 );
	    clear();
	}
    }

private:
    bool is_initialised() const { return m_log_size != 0; }

private:
    size_type m_elements;
    size_type m_log_size;
    type * m_table;
    group_t * m_group;
    hash_type m_hash;
    std::mutex m_mux;
};

} // namespace graptor

#endif // GRAPTOR_CONTAINER_HASH_SET_CUCKOO_BITMASK_H
