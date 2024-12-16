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

    using delta_t = uint32_t;

    static constexpr type invalid_element = ~type(0);
    static constexpr delta_t max_delta = std::numeric_limits<delta_t>::max();
    static constexpr size_type max_attempts = 8;
    static constexpr size_type log_group_size = 4;
    static constexpr type group_mask = ( type(1) << log_group_size ) - 1;
    using group_t = uint16_t;
    using index_t = uint32_t;
    static constexpr index_t invalid_index = ~index_t(0);

    static_assert( 8*sizeof(group_t) == size_type(1) << log_group_size,
		   "log_group_size must be log of size of group_t" );
    static_assert( sizeof(group_t) <= sizeof(type),
		   "data that is gathered must have a width not exceeding "
		   "that of a query element" );
    static_assert( sizeof(index_t) <= sizeof(type),
		   "data that is gathered must have a width not exceeding "
		   "that of a query element" );

public:
    explicit hash_set_cuckoo_bitmask()
	: m_elements( 0 ),
	  m_log_size( 0 ),
	  m_table( nullptr ),
	  // m_delta( nullptr ),
	  m_group( nullptr ),
	  m_hash( m_log_size ) { }
    explicit hash_set_cuckoo_bitmask( size_t expected_elms )
	: m_elements( 0 ),
	  m_log_size( required_log_size( expected_elms ) ),
	  m_table( new index_t[(1<<m_log_size)*2] ),
	  // m_delta( new delta_t[(1<<m_log_size)] ),
	  m_group( new group_t[(1<<m_log_size)*2] ),
	  m_hash( m_log_size ) {
	m_sequence.reserve( expected_elms );
	clear();
    }
    template<typename It>
    explicit hash_set_cuckoo_bitmask( It begin, It end )
	: hash_set_cuckoo_bitmask( std::distance( begin, end ) ) {
	insert( begin, end );
    }
    hash_set_cuckoo_bitmask( hash_set_cuckoo_bitmask && ) = delete;
    hash_set_cuckoo_bitmask( const hash_set_cuckoo_bitmask & ) = delete;
    hash_set_cuckoo_bitmask & operator = ( const hash_set_cuckoo_bitmask & ) = delete;

    ~hash_set_cuckoo_bitmask() {
	if( m_table != nullptr )
	    delete[] m_table;
	// if( m_delta != nullptr )
	    // delete[] m_delta;
	if( m_group != nullptr )
	    delete[] m_group;
    }

    void clear() {
	if( is_initialised() ) {
	    m_elements = 0;
	    std::fill( m_table, m_table+capacity()*2, invalid_index );
	    // std::fill( m_delta, m_delta+capacity(), delta_t(0) );
	    std::fill( m_group, m_group+capacity()*2, group_t(0) );
	}
    }

    size_type size() const { return m_elements; }
    size_type capacity() const {
	return m_log_size == 0 ? size_type(0) : size_type(1) << m_log_size;
    }
    bool empty() const { return size() == 0; }

    // const type * get_table() const { return m_table; }

    auto begin() const { return m_sequence.begin(); }
    auto end() const { return m_sequence.end(); }

    static size_type required_log_size( size_type num_elements ) {
	// Maintain fill factor of 50% at most
	return rt_ilog2( num_elements ) + HASH_SET_CUCKOO_LOAD_FACTOR;
    }

    // Assumes insert is called sequentially
    bool insert_prev( type value, type prev ) {
	create_if_uninitialised();

#if 0
	if( prev != invalid_element ) {
	    // Figure out where we put the value.
	    prev >>= log_group_size;
	    size_type index1 = m_hash.fn1( prev ) & ( capacity() - 1 );
	    size_type index2 = m_hash.fn2( prev ) & ( capacity() - 1 );
	    if( m_sequence[m_table[index1]] == prev )
		m_delta[index1] = std::max( m_delta[index1], value );
	    else if( m_sequence[m_table[index2]] == prev )
		m_delta[index2] = std::max( m_delta[index2], value );
	}
#endif

	if( contains( value ) )
	    return false;

	/*
	size_type index = locate( value );

	if( index != ~size_type(0) ) {
	    type off = value & group_mask;
	    group_t grp = group_t(1) << off;
	    if( ( m_group[index] & grp ) != 0 )
		return false;
	    else {
		m_sequence.push_back( value );
		m_group[index] |= grp;
		return true;
	    }
	}
	*/

	m_sequence.push_back( value );

	return insert_value( m_sequence.size()-1, value );
    }

    bool insert_next( type value, type next ) {
	create_if_uninitialised();

/*
	if( contains( value ) )
	    return false;

	m_sequence.push_back( value );

	return insert_value( m_sequence.size()-1, value );
*/
	return insert_prev( value, 0 );
    }

private:
    bool insert_value( index_t sindex, type value ) {
	int ret = insert_value_aux( sindex, value );

	if( ret >= 0 )
	    return (bool)ret;
	
	while( true ) {
	    // Resize and reconstruct
	    delete[] m_table;
	    delete[] m_group;
	    ++m_log_size;
	    m_table = new type[(size_type(1)<<m_log_size)*2];
	    m_group = new group_t[(size_type(1)<<m_log_size)*2];
	    clear(); // sets m_elements=0; will be reset when rehashing
	    m_hash.resize( m_log_size );

	    assert( m_log_size < 8*sizeof(size_type) );

	    bool all_good = true;
	    for( size_type i=0; i < m_sequence.size(); ++i ) {
		if( insert_value_aux( i, m_sequence[i] ) < 0 ) {
		    all_good = false;
		    break;
		}
	    }
	    if( all_good )
		break;
	}

	assert( m_elements == m_sequence.size() );

	return true;
    }

    int insert_value_aux( index_t sindex, type value ) {
	using std::swap;
	
	type off = value & group_mask;
	type gvalue = value >> log_group_size;
	group_t group = group_t(1) << off;


	size_type index = locate( value );
	if( index != ~size_type(0) ) {
	    assert( ( m_group[index] & group ) == 0 );

	    m_group[index] |= group;
	    ++m_elements;
	    return 1;
	}

	for( size_type attempts=0; attempts < max_attempts; ++attempts ) {
	    size_type index1 = m_hash.fn1( gvalue ) & ( capacity() - 1 );
	    if( m_table[index1] == invalid_index ) {
		m_table[index1] = sindex;
		m_group[index1] = group;
		++m_elements;
		return 1;
	    } else {
		swap( m_table[index1], sindex );
		swap( m_group[index1], group );
		value = m_sequence[sindex];
		gvalue = value >> log_group_size;
	    }

	    size_type index2 = m_hash.fn2( gvalue ) & ( capacity() - 1 );
	    if( m_table[capacity()+index2] == invalid_index ) {
		m_table[capacity()+index2] = sindex;
		m_group[capacity()+index2] = group;
		++m_elements;
		return 1;
	    } else {
		swap( m_table[capacity()+index2], sindex );
		swap( m_group[capacity()+index2], group );
		value = m_sequence[sindex];
		gvalue = value >> log_group_size;
	    }
	}

	return -1;

#if 0
	// Could not identify a free bucket.
	// Resize and retry.
	size_type old_log_size = m_log_size + 1;
	type * old_table = new type[(size_type(1)<<old_log_size)*2];
    // delta_t * old_delta = new delta_t[(size_type(1)<<old_log_size)];
	group_t * old_group = new group_t[(size_type(1)<<old_log_size)*2];
	using std::swap;
	swap( old_log_size, m_log_size );
	swap( old_table, m_table );
    // swap( old_delta, m_delta );
	swap( old_group, m_group );
	size_type old_elements = m_elements;
	clear(); // sets m_elements=0; will be reset when rehashing

	assert( m_log_size < 8*sizeof(size_type) );

	size_type old_size = ( size_type(1) << old_log_size ) * 2;
	m_hash.resize( m_log_size );
	for( size_type i=0; i < old_size; ++i ) {
	    if( old_table[i] != invalid_index ) {
		for( size_type i=0; i < (size_type(1)<<log_group_size); ++i )
		    if( ( m_group[i] >> i ) & 1 )
			insert_value( ( old_table[i] << log_group_size ) | type(i) );
pfff
	    }
	}
	delete[] old_table;
    // delete[] old_delta;
	delete[] old_group;

	// Retry insertion. Hope for tail recursion optimisation.
	// assert( m_elements == sindex );
	return insert_value( sindex, value );
#endif
    }

public:

#if 0
    template<typename It>
    void insert( It && I, It && E ) {
	create_if_uninitialised();
	
	while( I != E )
	    insert( *I++ );
    }
#endif

    size_type locate( type value ) const {
	if( empty() ) // also catches uninitialised case
	    return ~size_type(0);

	value >>= log_group_size;

	size_type index1 = m_hash.fn1( value ) & ( capacity() - 1 );
	size_type idx1 = m_table[index1];
	if( idx1 < m_elements
	    && value == ( m_sequence[idx1] >> log_group_size ) )
	    return index1;

	size_type index2 = m_hash.fn2( value ) & ( capacity() - 1 );
	size_type idx2 = m_table[capacity()+index2];
	if( idx2 < m_elements
	    && value == ( m_sequence[idx2] >> log_group_size ) )
	    return capacity()+index2;

	return ~size_type(0);
    }

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
	const vtype hmask = tr::srli( ones, tr::B - m_log_size );
	const vtype gmask = tr::srli( ones, tr::B - log_group_size );
	const vtype cap2 = tr::slli( tr::setoneval(), m_log_size );

	vtype vg = tr::srli( v, log_group_size );
	vtype hval1 = m_hash.template vectorized1<VL>( vg );
	vtype index1 = tr::bitwise_and( hval1, hmask );
	vtype probe1 = tr::gather( m_table, index1 );

	vtype hval2 = m_hash.template vectorized2<VL>( vg );
	vtype index2a = tr::bitwise_and( hval2, hmask );
	vtype index2 = tr::add( index2, cap2 );
	vtype probe2 = tr::gather( m_table, index2 );

	// The top bits in each lane of group are contaminated, but we will
	// mask them out using a bitwise_and below
	vtype group1 =
	    tr::template gather_w<sizeof(group_t)>(
		reinterpret_cast<const type *>( m_group ), index1 );
	vtype group2 =
	    tr::template gather_w<sizeof(group_t)>(
		reinterpret_cast<const type *>( m_group ), index2 );

	vtype voff = tr::bitwise_and( v, gmask );
	vtype vpos = tr::sllv( tr::setoneval(), voff );

	vtype ve = tr::set1( m_elements );
	vtype bits1 = tr::bitwise_and( group1, vpos );
	vtype bits2 = tr::bitwise_and( group2, vpos );
	mtype set1 = tr::cmpne( bits1, tr::setzero(), mkind() );
	mtype set2 = tr::cmpne( bits2, tr::setzero(), mkind() );
	mtype valid1 = tr::cmplt( set1, probe1, ve, mkind() );
	mtype valid2 = tr::cmplt( set2, probe2, ve, mkind() );

	vtype seq1 = tr::gather( &m_sequence[0], probe1, valid1 );
	vtype seq2 = tr::gather( &m_sequence[0], probe2, valid2 );
	vtype seq1g = tr::srli( seq1, log_group_size );
	vtype seq2g = tr::srli( seq2, log_group_size );
	mtype present1 = tr::cmpeq( valid1, seq1g, vg, mkind() );
	mtype present2 = tr::cmpeq( valid2, seq2g, vg, mkind() );
	mtype present = mtr::logical_or( present1, present2 );

	if constexpr ( std::is_same_v<MT,mkind> )
	    return present;
	else if constexpr ( std::is_same_v<MT,target::mt_mask> )
	    return tr::asmask( present );
	else
	    return tr::asvector( present );
    }

#if 0
    template<typename U, unsigned short VL, typename MT>
    std::pair<
	std::conditional_t<std::is_same_v<MT,target::mt_mask>,
			   typename vector_type_traits_vl<U,VL>::mask_type,
			   typename vector_type_traits_vl<U,VL>::vmask_type>,
	type>
    __attribute__((noinline))
    multi_contains_next( typename vector_type_traits_vl<U,VL>::type
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
		return std::make_pair( tr::setzero(), (type)0 );
	    else
		return std::make_pair( tr::mask_traits::setzero(), (type)0 );
	}

	const vtype ones = tr::setone();
	const vtype hmask = tr::srli( ones, tr::B - m_log_size );

	vtype hval1 = m_hash.template vectorized1<VL>( v );
	vtype index1 = tr::bitwise_and( hval1, hmask );
	vtype probe1 = tr::gather( m_table, index1 );

	vtype hval2 = m_hash.template vectorized2<VL>( v );
	vtype index2 = tr::bitwise_and( hval2, hmask );
	vtype probe2 = tr::gather( m_table+capacity(), index2 );

	mtype fnd1 = tr::cmpeq( probe1, v, mkind() );
	mtype fnd2 = tr::cmpeq( probe2, v, mkind() );
	vtype probe = tr::blend( fnd2, probe1, probe2 );
	mtype mfnd = mtr::logical_or( fnd1, fnd2 );

	uint32_t fnd = mfnd;
	type nxt = 0;
	uint32_t last;
	bool z;
	asm( "\n\t bsr %[fnd], %[last]"
	     : [last] "=r"(last), "=@ccz"(z)
	     : [fnd] "mr"(fnd)
	     : "cc" );
	if( !z ) {
	    vtype index = tr::blend( fnd2, index1, index2 );
	    nxt = m_delta[tr::lane( index, last )];
	}

	if constexpr ( std::is_same_v<MT,mkind> )
	    return std::make_pair( mfnd, nxt );
	else if constexpr ( std::is_same_v<MT,target::mt_mask> )
	    return std::make_pair( tr::asmask( mfnd ), nxt );
	else
	    return std::make_pair( tr::asvector( mfnd ), nxt );
    }
#endif

/*
    template<typename Fn>
    void for_each( Fn && fn ) const {
	if( empty() ) // also catches uninitialised case
	    return;
	
	for( size_type i=0; i < capacity(); ++i )
	    if( m_table[i] != invalid_element )
		fn( m_table[i] );
    }
*/

    std::mutex & get_lock() {
	return m_mux;
    }

    void create_if_uninitialised( size_t elements = 32 ) {
	if( !is_initialised() ) {
	    m_elements = 0;
	    m_log_size = required_log_size( m_elements );
	    m_table = new type[capacity()*2];
	    // m_delta = new delta_t[(1<<m_log_size)];
	    m_group = new group_t[(1<<m_log_size)*2];
	    m_sequence.reserve( elements );
	    m_hash.resize( m_log_size );
	    clear();
	}
    }

private:
    bool is_initialised() const { return m_log_size != 0; }

private:
    std::vector<type> m_sequence;
    size_type m_elements;
    size_type m_log_size;
    index_t * m_table;
    // delta_t * m_delta;
    group_t * m_group;
    hash_type m_hash;
    std::mutex m_mux;
};

} // namespace graptor

#endif // GRAPTOR_CONTAINER_HASH_SET_CUCKOO_BITMASK_H
