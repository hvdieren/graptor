// -*- c++ -*-
#ifndef GRAPTOR_CONTAINER_BITSET_H
#define GRAPTOR_CONTAINER_BITSET_H

#include <iterator>
#include <type_traits>
#include <immintrin.h>

#include "graptor/target/vector.h"

/*======================================================================*
 * TODO:
 * + use compress instruction to turn bitmask into a vector or array
 *   in memory from which the indices can be fetched efficiently. Obviates
 *   the need of one tzcnt per set bit into a fixed number of compress/st
 *   operations. For a 256-bit vector, this would be 16 compress operations.
 *   This would be more efficient than tzcnt only if ~16 bits or more are
 *   set. (AVX512F+VL only)
 *
 * + consider iteration over all elements, then identify in/out. Could
 *   iterate over all vectors with a single bit set by:
 *   - keep vector inc = vector with 1 in active lane
 *   - add inc to current
 *   - if relevant lane overflows (compare to set of ones), then move one
 *     to next lane, reset current lane, move current lane index up, redo
 *   -> Require to replicate overflow status of the current lane onto the
 *      next lane. If we have a bitmask, multiply by 3.
 *   - sketch:
 *     add inc to current mask
 *     compare to overflow bit pattern, producing mask
 *     multiply mask by 3 (AVX512 may be better with shift and or)
 *     blend between (current mask+inc) and (1 in next lane, 0 in others)
 *======================================================================*/

template<unsigned Bits, typename Enable = void>
class bitset_iterator {
public:
    using type = std::conditional_t<Bits<=32,uint32_t,uint64_t>;
    static constexpr unsigned bits_per_lane = sizeof(type)*8;
    static constexpr unsigned short VL = Bits / bits_per_lane;
    using tr = vector_type_traits_vl<type,VL>;
    using bitset_type = typename tr::type;

    // iterator traits
    using iterator_category = std::input_iterator_tag;
    using value_type = unsigned;
    using difference_type = unsigned;
    using pointer = const unsigned *;
    using reference = unsigned;

public:
    // The end iterator has an empty bitset - that's when we're done!
    explicit bitset_iterator()
	: m_subset( 0 ), m_lane( VL ), m_off( 0 ) { }
    explicit bitset_iterator( bitset_type bitset )
	: m_lane( 0 ), m_off( 0 ) {
	tr::storeu( m_bitset, bitset );
	tr::storeu( m_mask, tr::setzero() );
	m_subset = m_bitset[0];
	// The invariant is that the bit at (m_lane,m_off) was originally set
	// but has now been erased in the subset. Note that we never modify
	// the bitset itself.
	++*this;
    }
    // We have 'optimized' the code to keep the vector in memory and to
    // load scalar elements through memory. It seems the compiler 'undoes'
    // this optimization.
    bitset_iterator& operator++() {
	while( m_subset == 0 ) [[unlikely]] {
	    // pop next lane and recalculate off
	    m_mask[m_lane] = 0;
	    ++m_lane;
	    if( m_lane == VL ) [[unlikely]] { // reached end iterator
		m_off = 0;
		return *this;
	    }
	    m_subset = m_bitset[m_lane];
	}

	// Record position of erased bit.
	m_off = tzcnt( m_subset );

	// Erase bit from subset
	type old_subset = m_subset;
	m_subset &= m_subset - 1;

	// Set bit in mask
	m_mask[m_lane] = m_subset ^ old_subset;

	return *this;
    }
    bitset_iterator operator++( int ) {
	bitset_iterator retval = *this; ++(*this); return retval;
    }
    // (In-)equality of iterators is determined by the position of the
    // iterators, not by the content of the set.
    bool operator == ( bitset_iterator other ) const {
	return m_lane == other.m_lane && m_off == other.m_off;
    }
    bool operator != ( bitset_iterator other ) const {
	return !( *this == other );
    }
    typename bitset_iterator::value_type operator*() const {
	return m_lane * bits_per_lane + m_off;
    }

    bitset_type get_mask() const {
	return tr::loadu( m_mask );
    }

private:
    static unsigned tzcnt( type s ) {
	if constexpr ( bits_per_lane == 64 )
	    return _tzcnt_u64( s );
	else
	    return _tzcnt_u32( s );
    }
    
private:
    type m_bitset[VL];
    type m_mask[VL];
    type m_subset;
    // Might be useful to recode (m_lane,m_off) in an unsigned m_pos
    // and use shift and mask to recover m_lane and m_off when necessary.
    unsigned m_lane;
    unsigned m_off;
};

// Variant with single tzcnt-able lane
template<unsigned Bits>
class bitset_iterator<Bits,std::enable_if_t<Bits <= 64>> {
public:
    using type = std::conditional_t<Bits<=32,uint32_t,uint64_t>;
    static constexpr unsigned short VL = 1;
    static_assert( Bits <= 8*sizeof(type) );
    using tr = vector_type_traits_vl<type,VL>;
    using bitset_type = typename tr::type;

    // iterator traits
    using iterator_category = std::input_iterator_tag;
    using value_type = unsigned;
    using difference_type = unsigned;
    using pointer = const unsigned *;
    using reference = unsigned;

public:
    // The end iterator has an empty bitset - that's when we're done!
    explicit bitset_iterator()
	: m_bitset( 0 ), m_off( 0 ) { }
    explicit bitset_iterator( bitset_type bitset )
	: m_bitset( bitset ), m_off( 0 ) {
	// The invariant is that the bit at m_off is set.
	// Find position of next bit
	m_off = tzcnt( m_bitset );
    }
    bitset_iterator& operator++() {
	// Erase bit from bitset (works also if m_bitset == 0)
	m_bitset &= m_bitset - 1;

	// Find and record position of next bit
	m_off = tzcnt( m_bitset );

	return *this;
    }
    bitset_iterator operator++( int ) {
	bitset_iterator retval = *this; ++(*this); return retval;
    }
    // (In-)equality of iterators is determined by the position of the
    // iterators, not by the content of the set.
    bool operator == ( bitset_iterator other ) const {
	return m_bitset == other.m_bitset;
    }
    bool operator != ( bitset_iterator other ) const {
	return !( *this == other );
    }
    typename bitset_iterator::value_type operator*() const {
	return m_off;
    }

    bitset_type get_mask() const {
	return bitset_type(1) << m_off;
    }

private:
    static unsigned tzcnt( type s ) {
	if constexpr ( sizeof(type) > 4 )
	    return _tzcnt_u64( s );
	else
	    return _tzcnt_u32( s );
    }
    
private:
    bitset_type m_bitset;
    value_type m_off;
};

template<unsigned Bits, typename Enable = void>
class bitset_reverse_iterator {
public:
    using type = std::conditional_t<Bits<=32,uint32_t,uint64_t>;
    static constexpr unsigned bits_per_lane = sizeof(type)*8;
    static constexpr unsigned short VL = Bits / bits_per_lane;
    using tr = vector_type_traits_vl<type,VL>;
    using bitset_type = typename tr::type;

    // iterator traits
    using iterator_category = std::input_iterator_tag;
    using value_type = unsigned;
    using difference_type = unsigned;
    using pointer = const unsigned *;
    using reference = unsigned;

public:
    // The end iterator has an empty bitset - that's when we're done!
    explicit bitset_reverse_iterator()
	: m_subset( 0 ), m_lane( 0 ), m_off( bits_per_lane ) { }
    explicit bitset_reverse_iterator( bitset_type bitset )
	: m_lane( VL-1 ), m_off( bits_per_lane ) {
	tr::storeu( m_bitset, bitset );
	tr::storeu( m_mask, tr::setzero() );
	m_subset = m_bitset[VL-1];
	// The invariant is that the bit at (m_lane,m_off) was originally set
	// but has now been erased in the subset. Note that we never modify
	// the bitset itself.
	++*this;
    }
    // We have 'optimized' the code to keep the vector in memory and to
    // load scalar elements through memory. It seems the compiler 'undoes'
    // this optimization.
    bitset_reverse_iterator& operator++() {
	while( m_subset == 0 ) [[unlikely]] {
	    // pop next lane and recalculate off
	    m_mask[m_lane] = 0;
	    if( m_lane == 0 ) [[unlikely]] { // reached end iterator
		m_lane = 0;
		m_off = bits_per_lane;
		return *this;
	    }
	    --m_lane;
	    m_subset = m_bitset[m_lane];
	}

	// Record position of erased bit.
	m_off = ilzcnt( m_subset );

	// Erase bit from subset
	type this_bit = type(1) << m_off;
	m_subset ^= this_bit;

	// Set bit in mask
	m_mask[m_lane] = this_bit;

	return *this;
    }
    bitset_reverse_iterator operator++( int ) {
	bitset_reverse_iterator retval = *this; ++(*this); return retval;
    }
    // (In-)equality of iterators is determined by the position of the
    // iterators, not by the content of the set.
    bool operator == ( bitset_reverse_iterator other ) const {
	return m_lane == other.m_lane && m_off == other.m_off;
    }
    bool operator != ( bitset_reverse_iterator other ) const {
	return !( *this == other );
    }
    typename bitset_reverse_iterator::value_type operator*() const {
	return m_lane * bits_per_lane + m_off;
    }

    bitset_type get_mask() const {
	return tr::loadu( m_mask );
    }

private:
    static unsigned ilzcnt( type s ) {
	// lzcnt returns operand width if s == 0, however, we never call
	// the method on s == 0.
	return 8*sizeof(type) - lzcnt( s ) - 1;
    }
    static unsigned lzcnt( type s ) {
	if constexpr ( bits_per_lane == 64 )
	    return _lzcnt_u64( s );
	else
	    return _lzcnt_u32( s );
    }
    
private:
    type m_bitset[VL];
    type m_mask[VL];
    type m_subset;
    // Might be useful to recode (m_lane,m_off) in an unsigned m_pos
    // and use shift and mask to recover m_lane and m_off when necessary.
    unsigned m_lane;
    unsigned m_off;
};

// Variant with single tzcnt-able lane
template<unsigned Bits>
class bitset_reverse_iterator<Bits,std::enable_if_t<Bits <= 64>> {
public:
    using type = std::conditional_t<Bits<=32,uint32_t,uint64_t>;
    static constexpr unsigned short VL = 1;
    static_assert( Bits <= 8*sizeof(type) );
    using tr = vector_type_traits_vl<type,VL>;
    using bitset_type = typename tr::type;

    // iterator traits
    using iterator_category = std::input_iterator_tag;
    using value_type = unsigned;
    using difference_type = unsigned;
    using pointer = const unsigned *;
    using reference = unsigned;

public:
    // The end iterator has an empty bitset - that's when we're done!
    explicit bitset_reverse_iterator()
	: m_bitset( 0 ), m_off( 0 ) { }
    explicit bitset_reverse_iterator( bitset_type bitset )
	: m_bitset( bitset ), m_off( 0 ) {
	// The invariant is that the bit at m_off is set.
	// Find position of next bit
	m_off = ilzcnt( m_bitset );
    }
    bitset_reverse_iterator& operator++() {
	// Erase bit from bitset (works also if m_bitset == 0)
	m_bitset ^= get_mask();

	// Find and record position of next bit
	m_off = ilzcnt( m_bitset );

	return *this;
    }
    bitset_reverse_iterator operator++( int ) {
	bitset_reverse_iterator retval = *this; ++(*this); return retval;
    }
    // (In-)equality of iterators is determined by the position of the
    // iterators, not by the content of the set.
    bool operator == ( bitset_reverse_iterator other ) const {
	return m_bitset == other.m_bitset;
    }
    bool operator != ( bitset_reverse_iterator other ) const {
	return !( *this == other );
    }
    typename bitset_reverse_iterator::value_type operator*() const {
	return m_off;
    }

    bitset_type get_mask() const {
	return bitset_type(1) << m_off;
    }

private:
    static unsigned ilzcnt( type s ) {
	// lzcnt returns operand width when s is zero, however, we don't
	// care about the value computed for a zero bitmask as termination
	// is determined by im_bitset becoming zero, irrespective of m_off
	return 8*sizeof(type) - lzcnt( s ) - 1;
    }
    static unsigned lzcnt( type s ) {
	if constexpr ( sizeof(type) > 4 )
	    return _lzcnt_u64( s );
	else
	    return _lzcnt_u32( s );
    }
    
private:
    bitset_type m_bitset;
    value_type m_off;
};


template<unsigned Bits>
class bitset {
public:
    using iterator = bitset_iterator<Bits>;
    using reverse_iterator = bitset_reverse_iterator<Bits>;
    using bitset_type = typename iterator::bitset_type;

public:
    explicit bitset( bitset_type bitset ) : m_bitset( bitset ) { }

    operator bitset_type () const { return m_bitset; }

    // Iterators are read-only, they cannot modify the bitset
    iterator begin() { return iterator( m_bitset ); }
    iterator begin() const { return iterator( m_bitset ); }
    iterator end() { return iterator(); }
    iterator end() const { return iterator(); }

    reverse_iterator rbegin() { return reverse_iterator( m_bitset ); }
    reverse_iterator rbegin() const { return reverse_iterator( m_bitset ); }
    reverse_iterator rend() { return reverse_iterator(); }
    reverse_iterator rend() const { return reverse_iterator(); }

    size_t size() const {
	return target::allpopcnt<
	    size_t,typename iterator::type,iterator::VL>::compute( m_bitset );
    }

private:
    bitset_type m_bitset;
};


#endif // GRAPTOR_CONTAINER_BITSET_H
