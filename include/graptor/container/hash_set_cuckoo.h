// -*- c++ -*-
#ifndef GRAPTOR_CONTAINER_HASH_SET_CUCKOO_H
#define GRAPTOR_CONTAINER_HASH_SET_CUCKOO_H

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
class hash_set_cuckoo {
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

public:
    explicit hash_set_cuckoo()
	: m_elements( 0 ),
	  m_log_size( 0 ),
	  m_table( nullptr ),
	  m_delta( nullptr ),
	  m_hash( m_log_size-1 ) { }
    explicit hash_set_cuckoo( size_t expected_elms )
	: m_elements( 0 ),
	  m_log_size( required_log_size( expected_elms ) ),
	  m_table( new type[(1<<m_log_size)] ),
	  m_delta( new delta_t[(1<<m_log_size)] ),
	  m_hash( m_log_size-1 ) {
	clear();
    }
    template<typename It>
    explicit hash_set_cuckoo( It begin, It end )
	: hash_set_cuckoo( std::distance( begin, end ) ) {
	insert( begin, end );
    }
    hash_set_cuckoo( hash_set_cuckoo && ) = delete;
    hash_set_cuckoo( const hash_set_cuckoo & ) = delete;
    hash_set_cuckoo & operator = ( const hash_set_cuckoo & ) = delete;

    ~hash_set_cuckoo() {
	if( m_table != nullptr )
	    delete[] m_table;
	if( m_delta != nullptr )
	    delete[] m_delta;
    }

    void clear() {
	if( is_initialised() ) {
	    m_elements = 0;
	    std::fill( m_table, m_table+capacity(), invalid_element );
	    std::fill( m_delta, m_delta+capacity(), delta_t(0) );
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

	if( prev != invalid_element ) {
	    // Figure out where we put the value.
	    size_type index1 = m_hash.fn1( prev ) & ( capacity()/2 - 1 );
	    size_type index2 = m_hash.fn2( prev ) & ( capacity()/2 - 1 );
	    if( m_table[index1] == prev )
		m_delta[index1] = value;
	    else if( m_table[capacity()/2+index2] == prev )
		m_delta[capacity()/2+index2] = value;
	}

	return insert_value( value, 0 );
    }

    bool insert_next( type value, type next ) {
	create_if_uninitialised();

	return insert_value( value, next );
    }

    bool insert_value( type value, type next ) {
	using std::swap;
	
	if( contains( value ) )
	    return false;

	type delta = next; // ~size_type(0);

	for( size_type attempts=0; attempts < max_attempts; ++attempts ) {
	    size_type index1 = m_hash.fn1( value ) & ( capacity()/2 - 1 );
	    swap( m_table[index1], value );
	    swap( m_delta[index1], delta );
	    if( value == invalid_element ) {
		++m_elements;
		return true;
	    }

	    size_type index2 = m_hash.fn2( value ) & ( capacity()/2 - 1 );
	    swap( m_table[capacity()/2+index2], value );
	    swap( m_delta[capacity()/2+index2], delta );
	    if( value == invalid_element ) {
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
	delta_t * old_delta = new delta_t[(size_type(1)<<old_log_size)];
	using std::swap;
	swap( old_log_size, m_log_size );
	swap( old_table, m_table );
	swap( old_delta, m_delta );
	clear(); // sets m_elements=0; will be reset when rehashing

	size_type old_size = size_type(1) << old_log_size;
	m_hash.resize( m_log_size-1 );
	for( size_type i=0; i < old_size; ++i )
	    if( old_table[i] != invalid_element )
		insert_next( old_table[i], old_delta[i] );
	delete[] old_table;
	delete[] old_delta;

	// Retry insertion. Hope for tail recursion optimisation.
	return insert_value( value, next );
    }

    template<typename It>
    void insert( It && I, It && E ) {
	create_if_uninitialised();
	
	while( I != E )
	    insert( *I++ );
    }

    bool contains( type value ) const {
	if( empty() ) // also catches uninitialised case
	    return false;

	size_type index1 = m_hash.fn1( value ) & ( capacity()/2 - 1 );
	size_type index2 = m_hash.fn2( value ) & ( capacity()/2 - 1 );

	return m_table[index1] == value
	    || m_table[capacity()/2+index2] == value;
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

	const vtype one = tr::setoneval();
	const vtype mask2 = tr::slli( one, m_log_size-1 );
	const vtype hmask = tr::sub( mask2, one );

	vtype hval1 = m_hash.template vectorized1<VL>( v );
	vtype index1 = tr::bitwise_and( hval1, hmask );
	vtype probe1 = tr::gather( m_table, index1 );

	vtype hval2 = m_hash.template vectorized2<VL>( v );
	vtype index2 = tr::bitwise_and_or( hval2, hmask, mask2 );
	vtype probe2 = tr::gather( m_table, index2 );

	mtype fnd1 = tr::cmpeq( probe1, v, mkind() );
	mtype fnd2 = tr::cmpeq( probe2, v, mkind() );
	vtype probe = tr::blend( fnd2, probe1, probe2 );
	mtype fnd = mtr::logical_or( fnd1, fnd2 );

	if constexpr ( std::is_same_v<MT,mkind> )
	    return fnd;
	else if constexpr ( std::is_same_v<MT,target::mt_mask> )
	    return tr::asmask( fnd );
	else
	    return tr::asvector( fnd );
    }

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

	const vtype one = tr::setoneval();
	const vtype mask2 = tr::slli( one, m_log_size-1 );
	const vtype hmask = tr::sub( mask2, one );

	vtype hval1 = m_hash.template vectorized1<VL>( v );
	vtype index1 = tr::bitwise_and( hval1, hmask );
	vtype probe1 = tr::gather( m_table, index1 );

	vtype hval2 = m_hash.template vectorized2<VL>( v );
	vtype index2 = tr::bitwise_and_or( hval2, hmask, mask2 );
	vtype probe2 = tr::gather( m_table, index2 );

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


    template<typename Fn>
    void for_each( Fn && fn ) const {
	if( empty() ) // also catches uninitialised case
	    return;
	
	for( size_type i=0; i < capacity(); ++i )
	    if( m_table[i] != invalid_element )
		fn( m_table[i] );
    }

    std::mutex & get_lock() {
	return m_mux;
    }

    void create_if_uninitialised( size_t elements = 32 ) {
	if( !is_initialised() ) {
	    m_elements = elements;
	    m_log_size = required_log_size( m_elements );
	    m_table = new type[(1<<m_log_size)];
	    m_delta = new delta_t[(1<<m_log_size)];
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
    delta_t * m_delta;
    hash_type m_hash;
    std::mutex m_mux;
};

} // namespace graptor

#endif // GRAPTOR_CONTAINER_HASH_SET_CUCKOO_H
