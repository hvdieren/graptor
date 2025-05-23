// -*- c++ -*-
#ifndef GRAPTOR_CONTAINER_HASH_SET_HOPSCOTCH_H
#define GRAPTOR_CONTAINER_HASH_SET_HOPSCOTCH_H

#include <type_traits>
#include <algorithm>
#include <ostream>
#include <mutex>

#include "graptor/container/hash_fn.h"

#ifdef LOAD_FACTOR
#define HASH_SET_HOPSCOTCH_LOAD_FACTOR LOAD_FACTOR
#else
#define HASH_SET_HOPSCOTCH_LOAD_FACTOR 1
#endif

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
 *
 * Further ideas:
 * + Keep each linked list sorted.
 *   If so, in multi_contains, active should include cmplt( e, v, mkind() );
 *   However, linked lists are short, and conflicts rare (1/H! according
 *   to the hopscotch paper), so sorting will not provide much benefit.
 * + Add a short hash, similar to bloom filter, such that we can
 *   quickly check if the linked list might contain the item.
 *   As linked lists are never longer than H, a H or 2H bit bloom
 *   filter could work well. This would require a second hash function
 *   that is complementary to the main hash function such that it
 *   can distinguish values that hash to the same bucket.
 *   We currently take the lower bits of the hash function; could
 *   settle for taking the next log_2 H higher bits for the hash.
 *   Conclusion: bloom filter adds slight overhead and the extra filtering
 *   does not pay of performance wise, probably because linked lists are
 *   short to start with, and the filtering is only useful if it removes
 *   all lanes.
 *   The efficacy of this will also depend on whether a typical query
 *   finds the elements in the hash set or not. 
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

    // Decoupling the displacement distance from the collision count.
    // H = distance over which an element may be displaced from the home index
    // C = collision count within H, i.e., at most C out of H elements may have
    //     the same home index.
    // The reason for decoupling them is that H controls the normal Hopscotch
    // locality, whereas C determines the worst case number of gather operations
    // we need to perform per lookup.
#if __AVX512F__
    static constexpr size_type H = 64 / sizeof( type ) / 4;
#else // assuming AVX2
    static constexpr size_type H = 32 / sizeof( type );
#endif

    // static constexpr size_type C = std::min( size_type(3), H );
    static constexpr size_type C = std::min( size_type(1), H );

    using bitmask_t = int_type_of_size_t<std::max( size_type(1), H/8 )>;

    using delta_t = uint32_t;

    // tr is defined for scalar access to simultaneously inspect all elements
    // in the deplacement window
    using tr = vector_type_traits_vl<type,H>;
    using vtype = typename tr::type;
#if __AVX512F__
    using mtr = typename tr::mask_traits;
    using mkind = target::mt_mask;
#else
    using mtr = typename tr::vmask_traits;
    using mkind = target::mt_vmask;
#endif

    static constexpr type invalid_element = ~type(0);
    static constexpr delta_t max_delta = std::numeric_limits<delta_t>::max();

public:
    explicit hash_set_hopscotch()
	: m_elements( 0 ),
	  m_log_size( 0 ),
	  m_table( nullptr ),
	  m_bitmask( nullptr ),
	  m_delta( nullptr ),
	  m_prev( ~type(0) ),
	  m_hash( m_log_size ) { }
    explicit hash_set_hopscotch( size_t expected_elms )
	: m_elements( 0 ),
	  m_log_size( required_log_size( expected_elms ) ),
	  m_table( new type[(1<<m_log_size)+2*H-1] ),
	  m_bitmask( new bitmask_t[(1<<m_log_size)+3] ), // to allow reads of size 4
	  m_delta( new delta_t[(1<<m_log_size)+3] ), // to allow reads of size 4
	  m_prev( ~type(0) ),
	  m_hash( m_log_size ) {
	clear();
    }
    template<typename It>
    explicit hash_set_hopscotch( It begin, It end )
	: hash_set_hopscotch( std::distance( begin, end ) ) {
	insert( begin, end );
    }
    hash_set_hopscotch( hash_set_hopscotch && ) = delete;
    hash_set_hopscotch( const hash_set_hopscotch & ) = delete;
    hash_set_hopscotch & operator = ( const hash_set_hopscotch & ) = delete;

    ~hash_set_hopscotch() {
	if( m_table != nullptr )
	    delete[] m_table;
	if( m_bitmask != nullptr )
	    delete[] m_bitmask;
	if( m_delta != nullptr )
	    delete[] m_delta;
    }

    void clear() {
	if( is_initialised() ) {
	    m_elements = 0;
	    std::fill( m_table, m_table+capacity()+2*H-1, invalid_element );
	    std::fill( m_bitmask, m_bitmask+capacity()+3, bitmask_t(0) );
	    std::fill( m_delta, m_delta+capacity()+3, delta_t(0) );
	}
    }

    size_type size() const { return m_elements; }
    size_type capacity() const {
	return m_log_size == 0 ? size_type(0) : size_type(1) << m_log_size;
    }
    bool empty() const { return size() == 0; }

    const type * get_table() const { return m_table; }

    auto begin() const { return m_table; }
    auto end() const { return m_table+capacity()+H; }

    static size_type required_log_size( size_type num_elements ) {
	// Maintain fill factor of 50% at most
	// Make sure at least one size of H elements is included
	// (adding 2 to the log will make the table at least 4H elements).
	return rt_ilog2( std::max( H, num_elements ) )
	    + HASH_SET_HOPSCOTCH_LOAD_FACTOR;
    }

    // Assumes insert is called sequentially
    bool insert_prev( type value, type prev ) {
#if 0
	if( m_prev != ~(type)0 ) {
	    size_type index = m_hash( m_prev ) & ( capacity() - 1 );
	    vtype v = tr::set1( m_prev );

	    // We don't need wrap-around in this, as we allocated H
	    // extra positions.
	    vtype data = tr::loadu( &m_table[index] );

	    // Check for presence of value.
	    uint32_t mask = tr::cmpeq( data, v, target::mt_mask() );
	    assert( mask != 0 && "element was inserted" );
	    uint32_t off = _tzcnt_u32( mask );
	    // type d = value - m_prev;
	    // m_delta[index+off] = d > type(max_delta) ? max_delta : delta_t(d);
	    m_delta[index+off] = value;
	    m_prev = value;
	}
#endif

	return insert_value( value, prev );
    }

private:
    bool insert_value( type value, type prev ) {
	create_if_uninitialised();

	size_type home_index = m_hash( value ) & ( capacity() - 1 );
	size_type index = home_index;
	vtype v = tr::set1( value );
	vtype vinv = tr::setone();

	// We don't need wrap-around in this, as we allocated H extra positions.
	vtype data = tr::loadu( &m_table[index] );

	// Try to find the element.
	// In our use case, it will not normally be attempted to insert
	// an element already present. Still, we check for correctness reasons.
	if( !tr::is_zero( tr::cmpeq( data, v, target::mt_vmask() ) ) )
	    return false; // do not insert if already present

	// Try to find an empty slot in the next H buckets.
	typename tr::mask_type e = tr::cmpeq( data, vinv, target::mt_mask() );
	if( e != 0 ) {
	    size_type lane = tr::mask_traits::tzcnt( e );
	    index += lane;
	    if( set_bitmask( home_index, index ) ) {
		m_table[index] = value;
		++m_elements;
		return true;
	    }
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

		if( set_bitmask( home_index, index ) ) {
		    m_table[index] = value;
		    ++m_elements;
		    return true;
		}
	    }
	}

	// Could not identify a free bucket close enough to the home_index.
	// Or, the displacement window contains too many collisions on the
	// same home index.
	// Resize and retry.
	size_type old_log_size = m_log_size + 1;
	type * old_table = new type[(size_type(1)<<old_log_size) + 2*H-1];
	bitmask_t * old_bitmask = new bitmask_t[(size_type(1)<<old_log_size)+3];
	delta_t * old_delta = new delta_t[(size_type(1)<<old_log_size)+3];
	using std::swap;
	swap( old_log_size, m_log_size );
	swap( old_table, m_table );
	swap( old_bitmask, m_bitmask );
	swap( old_delta, m_delta );
	clear(); // sets m_elements=0; will be reset when rehashing

	size_type old_size = size_type(1) << old_log_size;
	m_hash.resize( m_log_size );
	for( size_type i=0; i < old_size+H; ++i )
	    if( old_table[i] != invalid_element )
		insert_prev( old_table[i], old_delta[i] );
	delete[] old_table;
	delete[] old_bitmask;
	delete[] old_delta;

	// Retry insertion. Hope for tail recursion optimisation.
	return insert_prev( value, prev );
    }

public:
    template<typename It>
    void insert( It && I, It && E ) {
	create_if_uninitialised();
	
	type prev = invalid_element;
	while( I != E ) {
	    type cur = *I;
	    insert( cur, prev );
	    prev = cur;
	    ++I;
	}
    }

    bool contains( type value ) const {
	if( empty() ) // also catches uninitialised case
	    return false;

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

	const vtype zero = tr::setzero();
	const vtype ones = tr::setone();
	const vtype hmask = tr::srli( ones, tr::B - m_log_size );
	const vtype hi = tr::srli( ones, 1 );

#if 0
	// This is a simplistic version that probes all H possible locations.
	// Retained for debugging purposes.
	vtype hval = m_hash.template vectorized<VL>( v );
	vtype home_index = tr::bitwise_and( hval, hmask );
	vtype vidx = home_index;

	mtype notfound = mtr::setone();

	for( size_type h=0; h < H; ++h ) {
	    vtype e = tr::gather( m_table+h, vidx );
	    notfound = mtr::logical_and( notfound, tr::cmpne( e, v, mkind() ) );
	}
#else
	vtype hval = m_hash.template vectorized<VL>( v );
	vtype home_index = tr::bitwise_and( hval, hmask );
	vtype vidx = home_index;

	vtype e = tr::gather( m_table, vidx );
	// Aligns bitmask to top such that sllv drops consumed bits
	vtype b = vget_bitmask<VL>( vidx );
	mtype notfound = tr::cmpne( e, v, mkind() );
	mtype active = tr::cmpne( notfound, b, zero, mkind() );

	b = tr::srli( b, 1 ); // helps to count +1 for the position of 1-bit

	if( !mtr::is_zero( active ) ) {
	    do {
		vtype off = tr::lzcnt( b );
		vidx = tr::add( vidx, off );
		e = tr::gather( m_table, vidx, active );
		b = tr::sllv( b, off );
		b = tr::bitwise_and( b, hi ); // disable 1-bit in top position
		notfound = tr::cmpne( notfound, e, v, mkind() );
	    } while( tr::cmpne( notfound, b, zero, target::mt_bool() ) );
	}
#endif

	if constexpr ( std::is_same_v<MT,mkind> )
	    return mtr::logical_invert( notfound );
	else if constexpr ( std::is_same_v<MT,target::mt_mask> )
	    return tr::mask_traits::logical_invert( tr::asmask( notfound ) );
	else
	    return tr::asvector( mtr::logical_invert( notfound ) );
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

	const vtype zero = tr::setzero();
	const vtype ones = tr::setone();
	const vtype hmask = tr::srli( ones, tr::B - m_log_size );
	const vtype hi = tr::srli( ones, 1 );

#if 0
	// This is a simplistic version that probes all H possible locations.
	// Retained for debugging purposes.
	vtype hval = m_hash.template vectorized<VL>( v );
	vtype home_index = tr::bitwise_and( hval, hmask );
	vtype vidx = home_index;

	mtype notfound = mtr::setone();

	for( size_type h=0; h < H; ++h ) {
	    vtype e = tr::gather( m_table+h, vidx );
	    notfound = mtr::logical_and( notfound, tr::cmpne( e, v, mkind() ) );
	}
#else
	vtype hval = m_hash.template vectorized<VL>( v );
	vtype home_index = tr::bitwise_and( hval, hmask );
	vtype vidx = home_index;

	vtype e = tr::gather( m_table, vidx );
	// Aligns bitmask to top such that sllv drops consumed bits
	vtype b = vget_bitmask<VL>( vidx );
	mtype notfound = tr::cmpne( e, v, mkind() );
	mtype active = tr::cmpne( notfound, b, zero, mkind() );

	b = tr::srli( b, 1 ); // helps to count +1 for the position of 1-bit

	// if( !mtr::is_zero( active ) ) {
	    // do {
	while( !mtr::is_zero( active ) ) {
		vtype off = tr::lzcnt( b );
		vidx = tr::add( vidx, active, vidx, off );
		e = tr::gather( m_table, vidx, active );
		b = tr::sllv( b, off );
		b = tr::bitwise_and( b, hi ); // disable 1-bit in top position
		notfound = tr::cmpne( notfound, e, v, mkind() );
		active = tr::cmpne( notfound, b, zero, mkind() );
		// } while( !mtr::is_zero( active ) );
		// } while( tr::cmpne( notfound, b, zero, target::mt_bool() ) );
	}
// why is active not updated??
#endif

	uint32_t fnd = /*mtr::asmask*/( mtr::logical_invert( notfound ) );
	type nxt = 0;
	// if( fnd ) {
	uint32_t last;
	bool z;
	asm( "bsr %[fnd], %[last] \n\t"
	     : [last] "=r"(last), "=@ccz"(z)
	     : [fnd] "mr" (fnd)
	     : "cc" );
	if( !z ) {
	// if( _BitScanReverse( &last, fnd ) ) {
	    // uint32_t last = 31 - _lzcnt_u32( fnd );
	    delta_t d = m_delta[tr::lane( vidx, last )];
	    // nxt = type(d) + tr::lane( v, last );
	    nxt = d;
	}

	if constexpr ( std::is_same_v<MT,mkind> )
	    return std::make_pair( mtr::logical_invert( notfound ), nxt );
	else if constexpr ( std::is_same_v<MT,target::mt_mask> )
	    return std::make_pair(
		tr::mask_traits::logical_invert( tr::asmask( notfound ) ),
		nxt );
	else
	    return std::make_pair(
		tr::asvector( mtr::logical_invert( notfound ) ),
		nxt );
    }


    template<typename Fn>
    void for_each( Fn && fn ) const {
	if( empty() ) // also catches uninitialised case
	    return;
	
	for( size_type i=0; i < capacity()+H; ++i )
	    if( m_table[i] != invalid_element )
		fn( m_table[i] );
    }

    std::mutex & get_lock() {
	return m_mux;
    }

    void create_if_uninitialised( size_t elements = H ) {
	if( !is_initialised() ) {
	    m_elements = elements;
	    m_log_size = required_log_size( m_elements );
	    m_table = new type[(1<<m_log_size)+2*H-1];
	    m_bitmask = new bitmask_t[(1<<m_log_size)+3];
	    m_delta = new delta_t[(1<<m_log_size)+3];
	    m_hash.resize( m_log_size );
	    clear();
	}
    }

private:
    bool is_initialised() const { return m_log_size != 0; }
    
    size_type
    hopscotch_move( type v, size_type home_index, size_type free_index ) {
	while( free_index - home_index >= H ) {
	    bool fnd = false;
	    for( size_type b=0; b < H-1; ++b ) {
		if( free_index >= H - 1 - b ) {
		    size_type idx = free_index - ( H - 1 ) + b;
		    size_type h = m_hash( m_table[idx] ) & ( capacity() - 1 );
		    // Element is movable
		    if( ( free_index - h ) < H ) {
			move_bitmask( h, idx, free_index );

			m_table[free_index] = m_table[idx];
			m_table[idx] = invalid_element; // redundant
			m_delta[free_index] = m_delta[idx];
			m_delta[idx] = 0;
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

    bool set_bitmask( size_type home_index, size_type index ) {
	if( index != home_index ) {
	    if( _popcnt32( (uint32_t)m_bitmask[home_index] ) < C ) {
		size_type pos = 8*sizeof(bitmask_t) - ( index - home_index );
		m_bitmask[home_index] |= bitmask_t(1) << pos;
		return true;
	    } else
		return false;
	}
	return true;
    }

    void move_bitmask( size_type home_index,
		       size_type erase_index,
		       size_type add_index ) {
	bitmask_t b = m_bitmask[home_index];
	size_type epos = 8*sizeof(bitmask_t) - ( erase_index - home_index );
	size_type apos = 8*sizeof(bitmask_t) - ( add_index - home_index );
	b &= ~( bitmask_t(1) << epos );
	b |= bitmask_t(1) << apos;
	m_bitmask[home_index] = b;
    }

    template<unsigned short VL>
    typename vector_type_traits_vl<type,VL>::type
    vget_bitmask( typename vector_type_traits_vl<type,VL>::type idx ) const {
	using tr = vector_type_traits_vl<type,VL>;
	using vtype = typename tr::type;

	vtype raw = tr::template gather_w<sizeof(bitmask_t)>(
	    reinterpret_cast<const type *>( m_bitmask ), idx );

	// Place bitmask top-aligned in lane
	vtype bitmask = tr::slli( raw, tr::B - 8*sizeof(bitmask_t) );

	return bitmask;
    }

private:
    size_type m_elements;
    size_type m_log_size;
    type * m_table;
    bitmask_t * m_bitmask;
    delta_t * m_delta;
    type m_prev;
    hash_type m_hash;
    std::mutex m_mux;
};

} // namespace graptor

#endif // GRAPTOR_CONTAINER_HASH_SET_HOPSCOTCH_H
