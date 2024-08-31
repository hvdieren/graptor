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

    // The metadata per linked list consists of one H-bit bitmask and
    // one H-bit Bloom filter. The Bloom filter is stored in the lower
    // part of the metadata.
    using metadata_t = int_type_of_size_t<2*H/8>;

    using bitmask_t = int_type_of_size_t<H/8>;
    using bloom_t = bitmask_t;

    static constexpr size_type bloom_bits = 3 + ilog2( sizeof(bloom_t) );
    static constexpr size_type bloom_mask = ( size_type(1) << bloom_bits ) - 1;

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

public:
    explicit hash_set_hopscotch( size_t expected_elms = 0 )
	: m_elements( 0 ),
	  m_log_size( required_log_size( expected_elms ) ),
	  m_table( new type[(1<<m_log_size)+2*H-1] ),
	  m_metadata( new metadata_t[1<<m_log_size] ),
	  m_hash( m_log_size + bloom_bits ) {
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
	delete[] m_table;
	delete[] m_metadata;
    }

    void clear() {
	m_elements = 0;
	std::fill( m_table, m_table+capacity()+2*H-1, invalid_element );
	std::fill( m_metadata, m_metadata+capacity(), metadata_t(0) );
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
	return rt_ilog2( std::max( H, num_elements ) ) + 1;
    }

    bool insert( type value ) {
	size_type hash = m_hash( value );
	size_type home_index = hash & ( capacity() - 1 );
	size_type bloom_hash = ( hash >> m_log_size ) & bloom_mask;
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
	    m_table[index] = value;
	    set_bitmask( home_index, index );
	    set_bloom( home_index, bloom_hash );
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
		set_bitmask( home_index, index );
		set_bloom( home_index, bloom_hash );
		++m_elements;
		return true;
	    }
	}

	// Could not identify a free bucket close enough to the home_index.
	// Resize and retry.
	size_type old_log_size = m_log_size + 1;
	type * old_table = new type[(size_type(1)<<old_log_size) + 2*H-1];
	metadata_t * old_metadata = new metadata_t[size_type(1)<<old_log_size];
	using std::swap;
	swap( old_log_size, m_log_size );
	swap( old_table, m_table );
	swap( old_metadata, m_metadata );
	clear(); // sets m_elements=0; will be reset when rehashing

	size_type old_size = size_type(1) << old_log_size;
	m_hash.resize( m_log_size + bloom_hash );
	for( size_type i=0; i < old_size+H; ++i )
	    if( old_table[i] != invalid_element )
		insert( old_table[i] );
	delete[] old_table;
	delete[] old_metadata;

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
	const vtype fmask = tr::srli( ones, tr::B - bloom_bits );
	const vtype bmask = tr::slli( ones, tr::B - 8*sizeof(bloom_t) );
	const vtype hi = tr::srli( ones, 1 );

	vtype hval = m_hash.template vectorized<VL>( v );
	vtype home_index = tr::bitwise_and( hval, hmask );
	vtype vidx = home_index;

	vtype e = tr::gather( m_table, vidx );
	mtype notfound = tr::cmpne( e, v, mkind() );

	vtype meta = vget_metadata<VL>( vidx );

	// Bloom indexing
	vtype fidx = tr::bitwise_and( tr::srli( hval, m_log_size ), fmask );
	vtype fsel = tr::sllv( one, fidx );
	// fmat = ( fsel | bmask ) & meta using ternary logic if available
	// fmat contains all bitmask bits and the selected bit from
	// the bloom filter
	vtype fmat = tr::bitwise_or_and( fsel, bmask, meta );
	
	// Both the bloom selection (active bit) and the bitmask
	// must be non-zero for the search to continue
	mtype active = tr::cmpne( notfound, fmat, zero, mkind() );

	// Aligns bitmask to top such that sllv drops consumed bits
	// Clear bloom filter bits
	vtype b = tr::bitwise_and( bmask, fmat ); // can use meta or fmat
	b = tr::srli( b, 1 ); // helps to count +1 for the position of 1-bit

	while( !mtr::is_zero( active ) ) {
	    vtype off = tr::lzcnt( b );
	    vidx = tr::add( vidx, off );
	    e = tr::gather( m_table, vidx, active );
	    b = tr::sllv( b, off );
	    b = tr::bitwise_and( b, hi ); // disable 1-bit in top position
	    notfound = tr::cmpne( notfound, e, v, mkind() );
	    active = tr::cmpne( notfound, b, zero, mkind() );
	}

	if constexpr ( std::is_same_v<MT,mkind> )
	    return mtr::logical_invert( notfound );
	else if constexpr ( std::is_same_v<MT,target::mt_mask> )
	    return tr::mask_traits::logical_invert( tr::asmask( notfound ) );
	else
	    return tr::asvector( mtr::logical_invert( notfound ) );
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
		if( free_index >= H - 1 - b ) {
		    size_type idx = free_index - ( H - 1 ) + b;
		    size_type h = m_hash( m_table[idx] ) & ( capacity() - 1 );
		    // Element is movable
		    if( ( free_index - h ) < H ) {
			move_bitmask( h, idx, free_index );

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

    void set_bitmask( size_type home_index, size_type index ) {
	if( index != home_index ) {
	    size_type pos = 8*sizeof(bitmask_t) - ( index - home_index );
	    pos += 8*sizeof(bloom_t);
	    m_metadata[home_index] |= metadata_t(1) << pos;
	}
    }

    void set_bloom( size_type home_index, size_type bloom_hash ) {
	assert( home_index < capacity() );
	size_type pos = ( bloom_hash & bloom_mask );
	m_metadata[home_index] |= metadata_t(1) << pos;
    }

    void move_bitmask( size_type home_index,
		       size_type erase_index,
		       size_type add_index ) {
	bitmask_t * bm =
	    reinterpret_cast<bitmask_t*>( &m_metadata[home_index] );
	bitmask_t b = bm[1]; // assuming sizeof(bloom_t) == sizeof(bitmask_t)
	size_type epos = 8*sizeof(bitmask_t) - ( erase_index - home_index );
	size_type apos = 8*sizeof(bitmask_t) - ( add_index - home_index );
	b &= ~( metadata_t(1) << epos );
	b |= metadata_t(1) << apos;
	bm[1] = b;
    }

#if 0
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

    template<unsigned short VL>
    typename vector_type_traits_vl<type,VL>::type
    vget_bloom( typename vector_type_traits_vl<type,VL>::type idx ) const {
	using tr = vector_type_traits_vl<type,VL>;
	using vtype = typename tr::type;

	vtype raw = tr::template gather_w<sizeof(bloom_t)>(
	    reinterpret_cast<const type *>( m_bloom ), idx );

	return raw;
    }
#endif

    template<unsigned short VL>
    typename vector_type_traits_vl<type,VL>::type
    vget_metadata( typename vector_type_traits_vl<type,VL>::type idx ) const {
	static_assert( sizeof(type) >= sizeof(metadata_t) );
	using tr = vector_type_traits_vl<type,VL>;
	using vtype = typename tr::type;

	vtype raw = tr::template gather_w<sizeof(metadata_t)>(
	    reinterpret_cast<const type *>( m_metadata ), idx );

	return raw;
    }

private:
    size_type m_elements;
    size_type m_log_size;
    type * m_table;
    metadata_t * m_metadata;
    hash_type m_hash;
};

} // namespace graptor

#endif // GRAPTOR_CONTAINER_HASH_SET_HOPSCOTCH_H
