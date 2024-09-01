// -*- c++ -*-
#ifndef GRAPTOR_CONTAINER_HASH_SET_HOPSCOTCH_DELTA_H
#define GRAPTOR_CONTAINER_HASH_SET_HOPSCOTCH_DELTA_H

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
 *
 * Building the delta's incrementally is challenging due to linked list
 * insertions and the hopscotch movement of the empty bucket changing
 * the relative order of elements in the linked list. For our use case,
 * however, the hash table is built up once, then only queried. As such,
 * we calculate the delta's only after all elements have been inserted.
 *======================================================================*/

namespace graptor {

template<typename T, typename Hash = rand_hash<T>>
class hash_set_hopscotch_delta {
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

    //! Padding for delta vector.
    // A delta vector should be at least an SSE4 vector as a 1x8 vector
    // is implemented with SSE4 operations.
    static constexpr size_type DH = std::min( size_type(16), H );

    using tr = vector_type_traits_vl<type,H>;
    using vtype = typename tr::type;
#if __AVX512F__
    using mtr = typename tr::mask_traits;
    using mkind = target::mt_mask;
#else
    using mtr = typename tr::vmask_traits;
    using mkind = target::mt_vmask;
#endif

    // Vector access to delta's
    using dtr = vector_type_traits_vl<uint8_t,H>;
    using dvtype = typename dtr::type;

    static constexpr type invalid_element = ~type(0);

public:
    explicit hash_set_hopscotch_delta( size_t expected_elms = 0 )
	: m_elements( 0 ),
	  m_log_size( required_log_size( expected_elms ) ),
	  m_table( new type[(1<<m_log_size)+2*H-1] ),
	  m_delta( new uint8_t[(1<<m_log_size)+2*DH-1]+DH ),
	  m_hash( m_log_size ) {
	clear();
    }
    template<typename It>
    explicit hash_set_hopscotch_delta( It begin, It end )
	: hash_set_hopscotch_delta( std::distance( begin, end ) ) {
	for( It i=begin; i != end; ++i )
	    insert( *i );
    }
    hash_set_hopscotch_delta( hash_set_hopscotch_delta && ) = delete;
    hash_set_hopscotch_delta( const hash_set_hopscotch_delta & ) = delete;
    hash_set_hopscotch_delta & operator = ( const hash_set_hopscotch_delta & ) = delete;

    ~hash_set_hopscotch_delta() {
	delete[] m_table;
	delete[] ( m_delta - DH );
    }

    void clear() {
	m_elements = 0;
	std::fill( m_table, m_table+capacity()+2*H-1, invalid_element );
	std::fill( m_delta-DH, m_delta+capacity()+DH-1, uint8_t(0) );
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
	return rt_ilog2( std::max( H, num_elements ) );
    }

    bool insert( type value ) {
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

	// Locate the end of the linked list
	size_type tail_index = find_tail( index );

	// Try to find an empty slot in the next H buckets.
	typename tr::mask_type e = tr::cmpeq( data, vinv, target::mt_mask() );
	if( e != 0 ) {
	    size_type lane = tr::mask_traits::tzcnt( e );
	    index += lane;
	    m_table[index] = value;
	    // If home_index == tail_index and != index, set first delta
	    // on home_index; otherwise set next delta on tail_index.
/*
	    set_delta( tail_index, // which cell
		       home_index == tail_index, // first (true) or next (false)
		       index - tail_index ); // delta
*/
	    ++m_elements;
	    // delta_check();
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
	    index = hopscotch_move( value, home_index, tail_index, free_index );
	    // delta_check();

	    // Only if we found a moveable element...
	    if( ~index != size_type(0) ) {
		assert( index - home_index < H );
		assert( m_table[index] == invalid_element );

		m_table[index] = value;
/*
		if( tail_index < index )
		    set_delta( tail_index, // which cell
			       home_index == tail_index, // first (true) or next
			       index - tail_index ); // delta
		else
		    insert_in_list( home_index, home_index, index );
*/
		++m_elements;
		// delta_check();
		return true;
	    }
	}

	// Could not identify a free bucket close enough to the home_index.
	// Resize and retry.
	size_type old_log_size = m_log_size + 1;
	type * old_table = new type[(size_type(1)<<old_log_size) + 2*H-1];
	uint8_t * old_delta = new uint8_t[(size_type(1)<<old_log_size) + 2*DH-1] + DH;
	using std::swap;
	swap( old_log_size, m_log_size );
	swap( old_table, m_table );
	swap( old_delta, m_delta );
	clear(); // sets m_elements=0; will be reset when rehashing

	size_type old_size = size_type(1) << old_log_size;
	m_hash.resize( m_log_size );
	for( size_type i=0; i < old_size+H; ++i )
	    if( old_table[i] != invalid_element )
		insert( old_table[i] );
	delete[] old_table;
	delete[] ( old_delta - DH );

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

	const vtype zero = tr::setzero();
	const vtype ones = tr::setone();
	const vtype one = tr::setoneval();
	const vtype hmask = tr::srli( ones, tr::B - m_log_size );

#if 0
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
	vtype d = vget_delta<VL>( vidx, true, mtr::setone() );
	mtype notfound = tr::cmpne( e, v, mkind() );
	mtype active = tr::cmpne( notfound, d, zero, mkind() );

	while( !mtr::is_zero( active ) ) {
	    vidx = tr::add( vidx, d );
	    e = tr::gather( m_table, vidx, active );
	    d = vget_delta<VL>( vidx, false, active );
	    notfound = mtr::logical_and( notfound, tr::cmpne( e, v, mkind() ) );
	    active = tr::cmpne( notfound, d, zero, mkind() );
	}
#endif

	if constexpr ( std::is_same_v<MT,mkind> )
	    return mtr::logical_invert( notfound );
	else if constexpr ( std::is_same_v<MT,target::mt_mask> )
	    return tr::mask_traits::logical_invert( tr::asmask( notfound ) );
	else
	    return tr::asvector( mtr::logical_invert( notfound ) );
    }

    template<typename Fn>
    void for_each( Fn && fn ) const {
	for( size_type i=0; i < capacity()+H-1; ++i )
	    if( m_table[i] != invalid_element )
		fn( m_table[i] );
    }

    void finalise() { build_deltas(); }

private:
    void build_deltas() {
	std::array<size_type,H> tail;
	std::iota( &tail[0], &tail[H], 0 );
	
	for( size_type i=0; i < capacity()+H-1; ++i ) {
	    // The tail is a rolling buffer. Squash info on linked lists
	    // too far away to matter.
	    tail[i % H] = i;

	    // Get the element
	    type v = m_table[i];
	    if( v == invalid_element )
		continue;

	    // Calculate home index
	    size_type home_index = m_hash( v ) & ( capacity() - 1 );

	    // A delta is only needed for elements that are not on their
	    // home index.
	    if( home_index != i ) {
		size_type tail_index = tail[home_index % H];
		set_delta( tail_index,
			   tail_index == home_index,
			   i - tail_index );
	    }
	    
	    // Remember position of last element seen on this list
	    tail[home_index % H] = i;
	}

	// Debugging
	// delta_check();
    }

private:
    void delta_check() {
	// check all delta's are correct by checking for each element its
	// home and whether it is reachable from its home.
	for( size_type i=0; i < capacity()+H-1; ++i ) {
	    type v = m_table[i];
	    if( v == invalid_element )
		continue;

	    // std::cout << " check v=" << v << "\n";

	    size_type home_index = m_hash( v ) & ( capacity() - 1 );
	    if( m_table[home_index] == v )
		continue;

	    size_type delta = get_delta( home_index, true );
	    size_type index = home_index;
	    bool found = false;
	    do {
		// std::cout << "v=" << v << " index=" << index << " delta=" << delta << "\n";
		assert( index + delta > index );
		index += delta;
		assert( index < capacity()+H-1 );

		if( m_table[index] == v ) {
		    found = true;
		    break;
		}
		
		delta = get_delta( index, false );
	    } while( delta != 0 );
	    if( !found )
		assert( 0 && "Element not reachable in linked list" );
	}
    }
    
    size_type
    hopscotch_move( type v, size_type home_index, size_type tail_index,
		    size_type free_index ) {
	while( free_index - home_index >= H ) {
	    bool fnd = false;
	    for( size_type b=0; b < H-1; ++b ) {
		if( free_index >= H - 1 - b ) {
		    size_type idx = free_index - ( H - 1 ) + b;
		    size_type h = m_hash( m_table[idx] ) & ( capacity() - 1 );
		    // Element is movable
		    if( ( free_index - h ) < H ) {
/*
			// Unlink element at idx from its list
			size_type pred = find_predecessor( h, idx );
			size_type idx_delta = get_delta( idx, idx == h );
			size_type succ = idx + idx_delta;
			if( pred != idx ) {
			    size_type pred_delta =
				idx_delta != 0 ? succ - pred : 0;
			    set_delta( pred, pred == h, pred_delta );
			}

			// Element at idx is not longer in its linked list.
			if( idx == h ) {
			    // In this case, pred == idx == h, so we haven't
			    // updated the home link yet.
			    // If idx does not have a successor, then the
			    // first link was already 0.
			    // If it does have a successor, then its successor
			    // is given by idx_delta, and does not need to be
			    // updated
			    set_delta( idx, false, 0 );
			} else {
			    // Respect the first link of the list homed in idx
			    set_delta( idx, false, 0 );
			}

			// Reinsert into list
			insert_in_list( h, pred, free_index );
*/

			// Empty slot is not in any linked list, however,
			// a new list should be rooted here if idx == h.
			// set_deltas_moved( h, idx, idx_delta == 0 ? free_index - idx : std::min( idx_delta, free_index - idx ) );
			// if( h != idx && h == pred )
			// set_delta( h, true, free_index - h );

			// Alternative view
			// free_index and h are no more than H apart, hence
			// can load vector to look at delta's over this range.
			// Then we can get a mask of which elements belong to
			// our linked list, adding idx to it. Can do this based
			// on hash_fn. Better?
			// Then recalculate delta's from indices
			//
			// The element at position idx is moved to free_index
			// The problem with maintaining linked list are elements
			// between idx and free_index, which exist only if
			// next_delta(idx) != 0 OR h == idx and first_delta(idx) != 0
			// Can turn bitmask of our positions into a sequence
			// of delta's using repeated application of bsf

			m_table[free_index] = m_table[idx];
			m_table[idx] = invalid_element; // redundant
	    // delta_check();
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

    size_type find_tail( size_type idx ) const {
	size_type delta = get_delta( idx, true );
	while( delta != 0 ) {
	    idx += delta;
	    delta = get_delta( idx, false );
	}
	return idx;
    }

    size_type find_predecessor( size_type home_index, size_type idx ) const {
#if 1
	size_type home_delta = get_delta( home_index, true );
	if( home_index + home_delta == idx )
	    return home_index;

	// Note that the first lane receives H, which may not fit in a nibble.
	// This is however OK, it will never match an extracted nibble.
	// There is always one lane that cannot match, either the first lane
	// as done here, or the final lane (which is idx). An empty bucket
	// at idx, however, would have next delta equal to zero and look like
	// a match, which we need to avoid.
	const dvtype vH = dtr::set1( H );	// H,   H,   H, ...,   H
	const dvtype inc = dtr::set1inc0();	// 0,   1,   2, ..., H-1
	const dvtype dec = dtr::sub( vH, inc );	// H, H-1, H-2, ...,   1
	const dvtype msk = dtr::srli( dtr::setone(), 4 ); // 0xf repeated
	dvtype fn = dtr::loadu( &m_delta[idx] - H ); // unsigned ...
	dvtype lo = dtr::bitwise_and( fn, msk );
	dvtype eqlo = dtr::cmpeq( lo, dec, target::mt_vmask() );
	dvtype val = dtr::bitwise_and( eqlo, dec );
	// There is at most one non-zero lane, so reduction gives its value.
	// Alternative is to compare eqlo != 0 and take lzcnt of the mask.
	// Yet another approach is to xor lo with dec, compare to 0, and
	// take lzcnt of the resulting mask.
	size_type delta = dtr::reduce_add( val );
	return idx - delta;
#else
	size_type bmax = std::min( H, idx+1 );
	for( size_type b=1; b < bmax; ++b ) {
	    if( get_delta( idx - b, home_index == idx - b ) == b )
		return idx - b;
	}
	return idx;
#endif
    }

    void insert_in_list( size_type home_index,
			 size_type pred_index,
			 size_type pos ) {
	while( true ) {
	    size_type succ =
		pred_index + get_delta( pred_index, pred_index == home_index );
	    if( succ == pred_index ) {
		// succ is tail; just append
		set_delta( succ, succ == home_index, pos - succ );
		break;
	    } else if( succ > pos ) {
		// Create list pred_index -> pos -> succ
		set_delta( pred_index, pred_index == home_index,
			   pos - pred_index );
		set_delta( pos, false, succ - pos );
		break;
	    } else {
		pred_index = succ;
	    }
	}
    }

    size_type get_delta( size_type idx, bool get_first ) const {
	assert( idx < capacity()+H-1 );
	size_type shift = get_first ? 4 : 0;
	uint8_t d = ( m_delta[idx] >> shift ) & uint8_t(0xfu);
	assert( d < uint8_t(H) );
	return d;
    }

    void set_delta( size_type idx, bool set_first, size_type delta ) {
	assert( idx < capacity()+H-1 );
	assert( delta < H );
	size_type shift = set_first ? 4 : 0;
	uint8_t fld = uint8_t(delta) << shift;
	uint8_t mask = uint8_t(0xf0u) >> shift;
	m_delta[idx] = ( m_delta[idx] & mask ) | fld;
    }

    void set_deltas_moved( size_type home_index,
			   size_type index,
			   size_type delta ) {
	// Set deltas corresponding to moving item at index, with home index
	// home_index over a distance of delta.
	// If index is the home_index, then set first delta to delta and
	// clear next delta.
	// Otherwise, clear next delta.
	size_type d = home_index == index ? ( delta << 4 ) : 0;
	m_delta[index] = d;
    }

    template<unsigned short VL, typename M>
    typename vector_type_traits_vl<type,VL>::type
    vget_delta( typename vector_type_traits_vl<type,VL>::type idx,
		bool get_first, M active ) const {
	using tr = vector_type_traits_vl<type,VL>;
	using vtype = typename tr::type;

	vtype raw = tr::template gather_w<1>(
	    reinterpret_cast<const type *>( m_delta ), idx, active );

	if( get_first )
	    raw = tr::srli( raw, 4 );

	const vtype mask = tr::srli( tr::setone(), tr::B - 4 );
	vtype delta = tr::bitwise_and( raw, mask );

	return delta;
    }

private:
    size_type m_elements;
    size_type m_log_size;
    type * m_table;
    uint8_t * m_delta;
    hash_type m_hash;
};

template<typename HashSet>
struct is_hash_set_hopscotch_delta : public std::false_type { };

template<typename T, typename Hash>
struct is_hash_set_hopscotch_delta<hash_set_hopscotch_delta<T,Hash>>
    : public std::true_type { };

template<typename HashSet>
constexpr bool is_hash_set_hopscotch_delta_v =
    is_hash_set_hopscotch_delta<HashSet>::value;


} // namespace graptor

#endif // GRAPTOR_CONTAINER_HASH_SET_HOPSCOTCH_DELTA_H
