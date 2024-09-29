// -*- c++ -*-
/*!=====================================================================*
 * \file graptor/container/intersect.h
 * \brief Various intersection algorithms
 *======================================================================*/

#ifndef GRAPTOR_CONTAINER_INTERSECT_H
#define GRAPTOR_CONTAINER_INTERSECT_H

#ifndef INTERSECTION_ALGORITHM
#define INTERSECTION_ALGORITHM 1
#endif

#ifndef FILTER_INTERSECTION_ALGORITHM
#define FILTER_INTERSECTION_ALGORITHM 0
#endif

#ifndef MC_INTERSECTION_ALGORITHM
#define MC_INTERSECTION_ALGORITHM 0
#endif

#ifndef INTERSECTION_TRIM
#define INTERSECTION_TRIM 1
#endif

#ifndef INTERSECT_ONE_SIDED
#define INTERSECT_ONE_SIDED 1
#endif

#ifndef INTERSECT_GE_ABOVE
#define INTERSECT_GE_ABOVE 1
#endif

#ifndef ABLATION_DISABLE_ADV_INTERSECT
#define ABLATION_DISABLE_ADV_INTERSECT 0
#endif

#include <iterator>
#include <type_traits>
#include <immintrin.h>

#include "graptor/target/vector.h"
#include "graptor/container/dual_set.h"
#include "graptor/container/array_slice.h"

namespace graptor {

template<typename T>
struct is_fast_array : public std::false_type { };

template<typename T>
constexpr bool is_fast_array_v = is_fast_array<T>::value;


/*! Enumeration of set operation types
 */
enum set_operation {
    so_intersect = 0,			//!< intersection - list of elements
    so_intersect_xlat = 1,		//!< intersection - list + translate
    so_intersect_size = 2,		//!< intersection size
    so_intersect_size_gt_val = 3,	//!< intersection size > or abort
    so_intersect_size_ge = 4,		//!< intersection size >=
    so_intersect_gt = 5,		//!< list elements if size gt threshold
    so_N = 6 	 	 	 	//!< number of set operations
};

/*! Trait for multi-element (vectorized) collector
 *
 * \tparam C Collector type.
 */
template<typename C>
struct is_multi_collector {
    static constexpr bool value = requires( const C & c ) {
	c.template multi_record<true,typename C::type,8>(
	    static_cast<typename C::type *>( nullptr ),
	    vector_type_traits_vl<typename C::type,8>::setzero(),
	    vector_type_traits_vl<typename C::type,8>::mask_traits::setzero(),
	    size_t(0), size_t(0) );
    };
};

/*! Variable indicating that is_multi_collector trait is met
 *
 * \tparam C Collector type.
 */
template<typename C>
constexpr bool is_multi_collector_v = is_multi_collector<C>::value;

/*! Intersection set collector.
 *
 * All intersection sets are "collected" using a class for commonality as this
 * provides the richest interface adapted to various special cases, in
 * particular simultaneous remapping (translation) of elements as they are
 * produced.
 * When a pointer is passed to the intersection method, we wrap it in the
 * intersection_collector class to have this common interface.
 *
 * This class covers both so_intersect and so_intersect_xlat, as indicated
 * by the is_xlat template parameter.
 *
 * \tparam is_xlat true when values in the intersection are additionally
 *                 translated to a different range or namespace.
 * \tparam T the type of the values in the set
 */
template<bool is_xlat, typename T>
struct intersection_collector {
    //! The type of elements in this set
    using type = T;

    //! \brief Constructor for storing the intersection at a certain address.
    //
    // It is important that sufficient space is available, as it is not
    // checked during intersection. The minimum of the sizes of the sets being
    // intersected should suffice.
    //
    // \arg _pointer The contents of the intersection will be stored
    //               consecutively from this address.
    intersection_collector( type * _pointer ) : m_pointer( _pointer ) { }

    //! \brief Indicate reversal of LHS and RHS arguments in intersection
    //
    // \tparam LSet left hand side set type
    // \tparam RSet right hand side set type
    //
    // \arg lset left hand side set
    // \arg rset right hand side set
    template<typename LSet, typename RSet>
    void swap( LSet && lset, RSet && rset ) { }

    //! \brief Record one element of the intersection.
    //
    // \tparam rhs True if the right-hand side in the intersection is the same
    //             as supplied by the caller, i.e., arguments have not been
    //             swapped.
    // \tparam otype Type of the xlat argument.
    // \arg l Pointer to the common element in the left-hand collection
    // \arg r Pointer to the common element in the right-hand collection
    // \arg ins True when the value should be inserted
    // \return Boolean that is false when intersection should be aborted.
    template<bool rhs>
    bool record( const type * l, const type * r, bool ins ) {
	if( ins )
	    *m_pointer++ = *l;
	return true;
    }

    //! \brief Record one element of the intersection.
    //
    // \tparam rhs True if the right-hand side in the intersection is the same
    //             as supplied by the caller, i.e., arguments have not been
    //             swapped.
    // \tparam otype Type of the xlat argument.
    // \arg l Pointer to the common element in the left-hand collection
    // \arg value The value to be stored, possibly translated
    // \arg ins True when the value should be inserted
    // \return Boolean that is false when intersection should be aborted.
    template<bool rhs>
    bool record( const type * l, type value, bool ins ) {
	if( ins )
	    *m_pointer++ = value;
	return true;
    }

    //! \brief register remaining unprocessed elements.
    //
    // \tparam rhs True if right-hand side remained as requested.
    // \arg l Number of elemetns remaining on LHS.
    // \arg r Number of elemetns remaining on RHS.
    template<bool rhs>
    void remainder( size_t l, size_t r ) { }

    //! \brief Record multiple elements in the vector.
    //
    // The elements are captured in a vector of length VL. A mask of the same
    // length indicated which elements are valid.
    //
    // \tparam rhs Indicates of order of elements has been preserved.
    // \tparam U Type of elements in the set.
    // \tparam VL Length of the vector in elements.
    // \tparam M Type of the validity mask.
    // \arg p Pointer to values in left-hand side argument to intersection.
    // \arg value Vector of VL values or translated values to store.
    // \arg mask Vector of VL mask bits indicating valid values in value.
    // \arg vl Only lower vl lanes are set in mask; others are invalid.
    // \return Boolean that is false when intersection should be aborted.
    template<bool rhs, typename U, unsigned short VL, typename M>
    bool
    multi_record( const U * p,
		  typename vector_type_traits_vl<U,VL>::type value,
		  M mask,
		  size_t vl,
		  size_t vr ) {
	using tr = vector_type_traits_vl<U,VL>;

	m_pointer = tr::cstoreu_p( m_pointer, mask, value );

	return true;
    }

    //! \brief Returns value that results from this intersection.
    // \return The pointer to the first memory location after the content
    //         of the intersection.
    type * return_value() const { return m_pointer; }

    //! \brief Checks if intersection should be terminated.
    // \return Boolean that is false when intersection should be aborted.
    bool terminated() const { return false; }

private:
    type * m_pointer; //!< Pointer to the next location to store elements
};

/*! Collector type to support the set_operation::intersect_size operation.
 *
 * \tparam T type of elements in the sets.
 */
template<typename T>
struct intersection_size {
    //! The type of elements in this set
    using type = T;

    intersection_size() : m_size( 0 ) { }

    template<typename LSet, typename RSet>
    void swap( LSet && lset, RSet && rset ) { }

    template<bool rhs>
    bool record( const type * l, const type * r, bool ins ) {
	if( ins )
	    ++m_size;
	return true;
    }

    template<bool rhs>
    bool record( const type * l, type value, bool ins ) {
	if( ins )
	    ++m_size;
	return true;
    }

    template<bool rhs>
    void remainder( size_t l, size_t r ) { }

    template<bool rhs, typename U, unsigned short VL>
    bool
    multi_record( const U * p,
		  typename vector_type_traits_vl<U,VL>::type index,
		  typename vector_type_traits_vl<U,VL>::mask_type mask,
		  size_t vl,
		  size_t vr ) {
	using tr = vector_type_traits_vl<U,VL>;
	size_t present = tr::mask_traits::popcnt( mask );
	m_size += present;
	return true;
    }

    size_t return_value() const { return m_size; }

    bool terminated() const { return false; }

private:
    size_t m_size;
};

/*! Collector type to support the set_operation::intersect_size_gt_val
 *  operation.
 *
 * \tparam T type of elements in the sets.
 */
template<typename T>
struct intersection_size_gt_val {
    //! The type of elements in this set
    using type = T;

    intersection_size_gt_val( size_t min_arg_size, size_t threshold )
	: m_options( min_arg_size - threshold ),
	  m_threshold( threshold ),
	  m_terminated( min_arg_size <= threshold ) { }

    // This code currently only works with rhs == true
    // When swapping lhs/rhs, we need to change the initial options.
    template<typename LSet, typename RSet>
    intersection_size_gt_val( LSet && lset, RSet && rset, size_t threshold )
	: intersection_size_gt_val( lset.size(), // iterated set!
				    threshold ) { }

    template<typename LSet, typename RSet>
    void swap( LSet && lset, RSet && rset ) {
	m_options = lset.size() - m_threshold;
    }

    // Version called by merge
    template<bool rhs>
    bool record( const type * l, const type * r, bool ins ) {
	if( *l < *r ) { // lhs not present
	    if( --m_options <= 0 ) [[unlikely]] {
		m_terminated = true;
		return false;
	    }
	}
	return true;
    }

    // Version called by hash
    template<bool rhs>
    bool record( const type * l, type value, bool ins ) {
	if( !ins ) {
	    if( --m_options <= 0 ) [[unlikely]] {
		m_terminated = true;
		return false;
	    }
	}
	return true;
    }

    template<bool rhs>
    void remainder( size_t l, size_t r ) {
	// We could flag m_terminated if m_options drops below zero,
	// however, this is not necessary as a correct intersection size
	// will be returned.
	m_options -= l;
    }

    template<bool rhs, typename U, unsigned short VL>
    bool
    multi_record( const U * p,
		  typename vector_type_traits_vl<U,VL>::type index,
		  typename vector_type_traits_vl<U,VL>::mask_type mask,
		  size_t vl,
		  size_t vr ) {
	using tr = vector_type_traits_vl<U,VL>;
	size_t absent = vl - tr::mask_traits::popcnt( mask );
	m_options -= absent;
	if( m_options <= 0 ) {
	    m_terminated = true;
	    return false;
	} else
	    return true;
    }

    size_t return_value() const {
	return m_terminated ? 0 : m_options + m_threshold;
    }

    bool terminated() const { return m_terminated; }

private:
    std::make_signed_t<size_t> m_options;
    size_t m_threshold;
    bool m_terminated;
};

/*! Collector type to support the set_operation::intersect_exceed
 *  operation.
 *
 * \tparam T type of elements in the sets.
 */
template<typename T>
struct intersection_collector_gt {
    //! The type of elements in this set
    using type = T;

    intersection_collector_gt(
	type * pointer, size_t min_arg_size, size_t threshold )
	: m_pointer( pointer ),
	  m_pointer_start( pointer ),
	  m_options( min_arg_size - threshold ),
	  m_threshold( threshold ),
	  m_terminated( min_arg_size <= threshold ) { }

    // This code currently only works with rhs == true
    // When swapping lhs/rhs, we need to change the initial options.
    template<typename LSet, typename RSet>
    intersection_collector_gt( type * pointer, LSet && lset, RSet && rset,
			       size_t threshold )
	: intersection_collector_gt( pointer, lset.size(), // iterated set!
				     threshold ) { }

    template<typename LSet, typename RSet>
    void swap( LSet && lset, RSet && rset ) {
	m_options = lset.size() - m_threshold;
    }

    // Version called by merge
    template<bool rhs>
    bool record( const type * l, const type * r, bool ins ) {
	if( *l < *r ) { // lhs not present
	    if( --m_options <= 0 ) [[unlikely]] {
		m_terminated = true;
		return false;
	    }
	} else if( ins )
	    *m_pointer++ = *l;
	return true;
    }

    // Version called by hash
    template<bool rhs>
    bool record( const type * l, type value, bool ins ) {
	if( !ins ) {
	    if( --m_options <= 0 ) [[unlikely]] {
		m_terminated = true;
		return false;
	    }
	} else
	    *m_pointer++ = *l;
	return true;
    }

    template<bool rhs>
    void remainder( size_t l, size_t r ) {
	// We could flag m_terminated if m_options drops below zero,
	// however, this is not necessary as a correct intersection size
	// will be returned.
	m_options -= l;
	if( m_options <= 0 )
	    m_terminated = true;
    }

    template<bool rhs, typename U, unsigned short VL>
    bool
    multi_record( const U * p,
		  typename vector_type_traits_vl<U,VL>::type value,
		  typename vector_type_traits_vl<U,VL>::mask_type mask,
		  size_t vl,
		  size_t vr ) {
	using tr = vector_type_traits_vl<U,VL>;
	size_t absent = vl - tr::mask_traits::popcnt( mask );
	m_options -= absent;
	if( m_options <= 0 ) {
	    m_terminated = true;
	    return false;
	} else {
	    using tr = vector_type_traits_vl<U,VL>;
	    m_pointer = tr::cstoreu_p( m_pointer, mask, value );

	    return true;
	}
    }

    type * return_value() const {
	assert( m_terminated
		|| ( std::distance( m_pointer_start, m_pointer )
		     == m_options + m_threshold ) );
	return m_terminated ? m_pointer_start : m_pointer;
    }

    bool terminated() const { return m_terminated; }

private:
    type * m_pointer, * m_pointer_start;
    std::make_signed_t<size_t> m_options;
    size_t m_threshold;
    bool m_terminated;
};


/*! Collector type to support the set_operation::intersect_size_gt_val
 *  operation.
 *
 * \tparam T type of elements in the sets.
 */
template<typename T>
struct intersection_size_gt_val_two_sided {
    //! The type of elements in this set
    using type = T;

    intersection_size_gt_val_two_sided(
	size_t left_size,
	size_t right_size,
	size_t threshold )
	: m_left_options( left_size - threshold ),
	  m_right_options( right_size - threshold ),
	  m_threshold( threshold ),
	  m_terminated( left_size <= threshold || right_size <= threshold ) { }

    // This code currently only works with rhs == true
    // When swapping lhs/rhs, we need to change the initial options.
    template<typename LSet, typename RSet>
    intersection_size_gt_val_two_sided(
	LSet && lset, RSet && rset, size_t threshold )
	: intersection_size_gt_val_two_sided(
	    lset.size(), rset.size(), threshold ) { }

    template<typename LSet, typename RSet>
    void swap( LSet && lset, RSet && rset ) {
	m_left_options = lset.size() - m_threshold;
	m_right_options = rset.size() - m_threshold;
    }

    // Version called by merge
    template<bool rhs>
    bool record( const type * l, const type * r, bool ins ) {
	if( *l < *r ) { // lhs not present
	    if( --m_left_options <= 0 ) [[unlikely]] {
		m_terminated = true;
		return false;
	    }
	} else if( *l > *r ) { // rhs not present
	    if( --m_right_options <= 0 ) [[unlikely]] {
		m_terminated = true;
		return false;
	    }
	}
	return true;
    }

    // Version called by hash
    template<bool rhs>
    bool record( const type * l, type value, bool ins ) {
	if( !ins ) {
	    if( --m_left_options <= 0 ) [[unlikely]] {
		m_terminated = true;
		return false;
	    }
	}
	return true;
    }

    template<bool rhs>
    void remainder( size_t l, size_t r ) {
	// We could flag m_terminated if m_options drops below zero,
	// however, this is not necessary as a correct intersection size
	// will be returned.
	m_left_options -= l;
	m_right_options -= r;
    }

    template<bool rhs, typename U, unsigned short VL>
    bool
    multi_record( const U * p, // pointer to LHS sequential data
		  typename vector_type_traits_vl<U,VL>::type index, // LHS data
		  typename vector_type_traits_vl<U,VL>::mask_type mask, // LHS mask
		  size_t vl, size_t vr ) {
	using tr = vector_type_traits_vl<U,VL>;
	std::make_signed_t<size_t> num_matched =
	    tr::mask_traits::popcnt( mask );
	m_left_options -= std::make_signed_t<size_t>( vl ) - num_matched;
	m_right_options -= std::make_signed_t<size_t>( vr ) - num_matched;
	if( m_left_options <= 0 || m_right_options <= 0 ) {
	    m_terminated = true;
	    return false;
	}

	return true;
    }

    size_t return_value() const {
	return m_terminated ? 0 : m_left_options + m_threshold;
    }

    bool terminated() const { return m_terminated; }

private:
    std::make_signed_t<size_t> m_left_options, m_right_options;
    size_t m_threshold;
    bool m_terminated;
};

/*! Collector type to support the set_operation::intersect_size_ge
 *  operation.
 *
 * \tparam T type of elements in the sets.
 */
template<typename T>
struct intersection_size_ge {
    //! The type of elements in this set
    using type = T;

    intersection_size_ge(
	const type * const l_end,
	const type * const r_end,
	size_t left_size,
	size_t right_size,
	size_t threshold )
	: m_left_options( left_size - threshold ),
	  m_threshold( threshold ),
	  m_terminated( left_size < threshold || right_size < threshold ),
	  m_ge( false ),
	  m_left_end( l_end ) { }

    // This code currently only works with rhs == true
    // When swapping lhs/rhs, we need to change the initial options.
    template<typename LSet, typename RSet>
    intersection_size_ge(
	LSet && lset, RSet && rset, size_t threshold )
	: intersection_size_ge(
	    &*lset.end(), &*rset.end(), lset.size(), rset.size(), threshold ) { }

    template<typename LSet, typename RSet>
    void swap( LSet && lset, RSet && rset ) {
	m_left_options = lset.size() - m_threshold;
	m_left_end = &*lset.end();
    }

    // Version called by merge
    template<bool rhs>
    bool record( const type * l, const type * r, bool ins ) {
	if( *l < *r ) { // lhs not present
	    if( --m_left_options < 0 ) [[unlikely]] {
		m_terminated = true;
		m_ge = false;
		return false;
	    }
	}
#if INTERSECT_GE_ABOVE
	// > because current element not included
	else if( m_left_options > std::distance( l, m_left_end ) ) {
	    m_terminated = true;
	    m_ge = true;
	    return false;
	}
#endif
	return true;
    }

    // Version called by hash
    template<bool rhs>
    bool record( const type * l, type value, bool ins ) {
	if( !ins ) {
	    if( --m_left_options < 0 ) [[unlikely]] {
		m_terminated = true;
		m_ge = false;
		return false;
	    }
	}
#if INTERSECT_GE_ABOVE
	else if( m_left_options > std::distance( l, m_left_end ) ) {
	    m_terminated = true;
	    m_ge = true;
	    return false;
	}
#endif
	return true;
    }

    template<bool rhs>
    void remainder( size_t l, size_t r ) {
	// We could flag m_terminated if m_options drops below zero,
	// however, this is not necessary as a correct intersection size
	// will be returned.
	m_left_options -= l;
    }

    template<bool rhs, typename U, unsigned short VL>
    bool
    multi_record( const U * p, // pointer to LHS sequential data
		  typename vector_type_traits_vl<U,VL>::type index, // LHS data
		  typename vector_type_traits_vl<U,VL>::mask_type mask, // LHS mask
		  size_t vl, size_t vr ) {
	using tr = vector_type_traits_vl<U,VL>;
	std::make_signed_t<size_t> num_matched =
	    tr::mask_traits::popcnt( mask );
	m_left_options -= std::make_signed_t<size_t>( vl ) - num_matched;
	if( m_left_options < 0 ) {
	    m_terminated = true;
	    m_ge = false;
	    return false;
#if INTERSECT_GE_ABOVE
	} else if( m_left_options >= std::distance( p+vl, m_left_end ) ) {
	    m_terminated = true;
	    m_ge = true;
	    return false;
#endif
	}

	return true;
    }

    bool return_value() const {
	return m_terminated ? m_ge : m_left_options >= 0;
    }

    bool terminated() const { return m_terminated; }

private:
    std::make_signed_t<size_t> m_left_options;
    size_t m_threshold;
    bool m_terminated, m_ge;
    const type * m_left_end;
};

template<typename T>
struct intersection_size_ge_two_sided {
    //! The type of elements in this set
    using type = T;

    intersection_size_ge_two_sided(
	const type * const l_end,
	const type * const r_end,
	size_t left_size,
	size_t right_size,
	size_t threshold )
	: m_left_options( left_size - threshold ),
	  m_right_options( right_size - threshold ),
	  m_threshold( threshold ),
	  m_terminated( left_size < threshold || right_size < threshold ),
	  m_ge( false ),
	  m_left_end( l_end ),
	  m_right_end( r_end ) { }

    // This code currently only works with rhs == true
    // When swapping lhs/rhs, we need to change the initial options.
    template<typename LSet, typename RSet>
    intersection_size_ge_two_sided(
	LSet && lset, RSet && rset, size_t threshold )
	: intersection_size_ge_two_sided(
	    &*lset.end(), &*rset.end(), lset.size(), rset.size(), threshold ) {
    }

    template<typename LSet, typename RSet>
    void swap( LSet && lset, RSet && rset ) {
	m_left_options = lset.size() - m_threshold;
	m_right_options = rset.size() - m_threshold;
	m_left_end = &*lset.end();
	m_right_end = &*rset.end();
    }

    // Version called by merge
    template<bool rhs>
    bool record( const type * l, const type * r, bool ins ) {
	if( *l < *r ) { // lhs not present
	    if( --m_left_options < 0 ) [[unlikely]] {
		m_terminated = true;
		m_ge = false;
		return false;
	    }
	} else if( *l > *r ) { // rhs not present
	    if( --m_right_options < 0 ) [[unlikely]] {
		m_terminated = true;
		m_ge = false;
		return false;
	    }
	}
#if INTERSECT_GE_ABOVE
	else if( m_left_options > std::distance( l, m_left_end ) ) {
	    m_terminated = true;
	    m_ge = true;
	    return false;
	} else if( m_right_options > std::distance( r, m_right_end ) ) {
	    m_terminated = true;
	    m_ge = true;
	    return false;
	}
#endif
	return true;
    }

    // Version called by hash
    template<bool rhs>
    bool record( const type * l, type value, bool ins ) {
	if( !ins ) {
	    if( --m_left_options < 0 ) [[unlikely]] {
		m_terminated = true;
		m_ge = false;
		return false;
	    }
	}
#if INTERSECT_GE_ABOVE
	else if( m_left_options > std::distance( l, m_left_end ) ) {
	    m_terminated = true;
	    m_ge = true;
	    return false;
	}
#endif
	return true;
    }

    template<bool rhs>
    void remainder( size_t l, size_t r ) {
	// We could flag m_terminated if m_options drops below zero,
	// however, this is not necessary as a correct intersection size
	// will be returned.
	m_left_options -= l;
	m_right_options -= r;
    }

    template<bool rhs, typename U, unsigned short VL>
    bool
    multi_record( const U * p, // pointer to LHS sequential data
		  typename vector_type_traits_vl<U,VL>::type index, // LHS data
		  typename vector_type_traits_vl<U,VL>::mask_type mask, // LHS mask
		  size_t vl, size_t vr ) {
	using tr = vector_type_traits_vl<U,VL>;
	std::make_signed_t<size_t> num_matched =
	    tr::mask_traits::popcnt( mask );
	m_left_options -= std::make_signed_t<size_t>( vl ) - num_matched;
	m_right_options -= std::make_signed_t<size_t>( vr ) - num_matched;
	if( m_left_options < 0 || m_right_options < 0 ) {
	    m_terminated = true;
	    m_ge = false;
	    return false;
	} else if( m_left_options >= std::distance( p + vl, m_left_end ) ) {
	    m_terminated = true;
	    m_ge = true;
	    return false;
	}

	return true;
    }

    bool return_value() const {
	return m_terminated ? m_ge : m_left_options >= 0;
    }

    bool terminated() const { return m_terminated; }

private:
    std::make_signed_t<size_t> m_left_options, m_right_options;
    size_t m_threshold;
    bool m_terminated, m_ge;
    const type * m_left_end;
    const type * m_right_end;
};


/*! Traits class to recognise collectors for the
 *  set_operation::intersect_size_gt_val operation.
 *
 * \tparam S The collector type that is checked.
 */
template<typename S>
struct is_intersection_size_gt_val : public std::false_type { };

template<typename T>
struct is_intersection_size_gt_val<intersection_size_gt_val<T>>
    : public std::true_type { };

template<typename T>
struct is_intersection_size_gt_val<intersection_size_gt_val_two_sided<T>>
    : public std::true_type { };

/*! Variable to recognise collectors for the
 *  set_operation::intersect_size_gt_val operation.
 */
template<typename S>
constexpr bool is_intersection_size_gt_val_v =
    is_intersection_size_gt_val<S>::value;

/*! \section Intersection approaches
 */
struct hash_scalar;
struct hash_vector;
struct merge_scalar;
struct merge_vector;
struct merge_scalar_jump;
struct merge_vector_jump;

/*! Describes a set_operation and its arguments and allows to create
 *  a suitable collector
 *
 * This is the default version of the class which includes a collector
 *
 * \tparam op the set_operation to apply
 * \tparam Collector a pre-defined collector, if relevant to the operation
 */
template<set_operation op, typename Collector>
struct intersection_task {
    static constexpr set_operation operation = op;

    intersection_task( Collector & c ) : m_collector( c ) { }

    template<typename so_traits, typename LSet, typename RSet>
    auto & create_collector( LSet && lset, RSet && rset ) {
	return m_collector;
    }

    auto return_value_empty_set() { return m_collector.return_value(); }

private:
    Collector & m_collector;
};

/*! Specialisation for so_intersect_gt
 */
template<typename type>
struct intersection_task<so_intersect_gt,type> {
    static constexpr set_operation operation = so_intersect_gt;

    intersection_task( type * pointer, size_t threshold )
	: m_pointer( pointer ), m_threshold( threshold ) { }

    template<typename so_traits, typename LSet, typename RSet>
    auto create_collector( LSet && lset, RSet && rset ) {
	return intersection_collector_gt<typename std::decay_t<LSet>::type>(
	    m_pointer, lset, rset, m_threshold );
    }

    type * return_value_empty_set() { return m_pointer; }

private:
    type * m_pointer;
    size_t m_threshold;
};

/*! Specialisation for so_intersect_size
 */
template<>
struct intersection_task<so_intersect_size,void> {
    static constexpr set_operation operation = so_intersect_size;

    template<typename so_traits, typename LSet, typename RSet>
    auto create_collector( LSet && lset, RSet && rset ) {
	return intersection_size<typename std::decay_t<LSet>::type>();
    }

    size_t return_value_empty_set() { return 0; }
};

/*! Specialisation for so_intersect_size_gt_val
 */
template<>
struct intersection_task<so_intersect_size_gt_val,void> {
    static constexpr set_operation operation = so_intersect_size_gt_val;

    intersection_task( size_t threshold ) : m_threshold( threshold ) { }

    template<typename so_traits, typename LSet, typename RSet>
    auto create_collector( LSet && lset, RSet && rset ) {
	// Two-sided operation has no merit for hashed representations
	// as we cannot count missed opportunites in the RHS.
	// Although this may work differently for a dual representation where
	// we can cheaply relate the hashed data structure to the sequential
	// representation.
	if constexpr ( std::is_same_v<so_traits,hash_scalar>
		       || std::is_same_v<so_traits,hash_vector> ) {
	    return intersection_size_gt_val<
		typename std::decay_t<LSet>::type>( lset, rset, m_threshold );
	} else {
#if INTERSECT_ONE_SIDED
	    return intersection_size_gt_val<
		typename std::decay_t<LSet>::type>( lset, rset, m_threshold );
#else
	    return intersection_size_gt_val_two_sided<
		typename std::decay_t<LSet>::type>( lset, rset, m_threshold );
#endif
	}
    }

    size_t return_value_empty_set() { return 0; }

private:
    size_t m_threshold;
};

/*! Specialisation for so_intersect_size_ge
 */
template<>
struct intersection_task<so_intersect_size_ge,void> {
    static constexpr set_operation operation = so_intersect_size_ge;

    intersection_task( size_t threshold ) : m_threshold( threshold ) { }

    template<typename so_traits, typename LSet, typename RSet>
    auto create_collector( LSet && lset, RSet && rset ) {
	// Two-sided operation has no merit for hashed representations
	// as we cannot count missed opportunites in the RHS.
	// Although this may work differently for a dual representation where
	// we can cheaply relate the hashed data structure to the sequential
	// representation.
	if constexpr ( std::is_same_v<so_traits,hash_scalar>
		       || std::is_same_v<so_traits,hash_vector> ) {
	    return intersection_size_ge<
		typename std::decay_t<LSet>::type>( lset, rset, m_threshold );
	} else {
#if INTERSECT_ONE_SIDED
	    return intersection_size_ge<
		typename std::decay_t<LSet>::type>( lset, rset, m_threshold );
#else
	    return intersection_size_ge_two_sided<
		typename std::decay_t<LSet>::type>( lset, rset, m_threshold );
#endif
	}
    }

    bool return_value_empty_set() { return m_threshold == 0; }

private:
    size_t m_threshold;
};

template<set_operation op, typename Collector>
auto create_intersection_task( Collector & c ) {
    return intersection_task<op,Collector>( c );
}

template<typename so_traits>
struct set_operations {

    template<typename LSet, typename RSet, typename Collector>
    static
    auto
    intersect_ds( LSet && lset, RSet && rset, Collector & out ) {
	static_assert( std::is_same_v<
		       typename std::decay_t<LSet>::type,
		       typename std::decay_t<RSet>::type>,
		       "Sets must contain elements of the same type" );
	
	// Collector is of type pointer to element of set
	// Construct custom collector object to have unified code base
	// for storing intersection with pointer or with a custom class.
	if constexpr ( std::is_same_v<std::decay_t<Collector>,
		       std::add_pointer_t<typename std::decay_t<LSet>::type>> ) {
	    intersection_collector<false,typename std::decay_t<LSet>::type>
		cout( out );
	    auto task = create_intersection_task<so_intersect>( cout );
	    return apply( lset, rset, task );
	} else {
	    auto task = create_intersection_task<so_intersect>( out );
	    return apply( lset, rset, task );
	}
    }

    template<typename LSet, typename RSet, typename Collector>
    static
    auto
    intersect_gt_ds( LSet && lset, RSet && rset, size_t threshold,
		     Collector & out ) {
	static_assert( std::is_same_v<
		       typename std::decay_t<LSet>::type,
		       typename std::decay_t<RSet>::type>,
		       "Sets must contain elements of the same type" );

#if ABLATION_DISABLE_ADV_INTERSECT
	return intersect_ds( lset, rset, out );
#else
	// Collector is of type pointer to element of set
	// Construct custom collector object to have unified code base
	// for storing intersection with pointer or with a custom class.
	if constexpr ( std::is_same_v<std::decay_t<Collector>,
		       std::add_pointer_t<typename std::decay_t<LSet>::type>> ) {
	    using type = typename std::decay_t<LSet>::type;
	    intersection_task<so_intersect_gt,type> task( out, threshold );
	    return apply( lset, rset, task );
	} else {
	    // In the case of a custom collector, we cannot guarantee the
	    // ability to apply the threshold.
	    auto task = create_intersection_task<so_intersect>( out );
	    return apply( lset, rset, task );
	}
#endif
    }

    template<typename LSet, typename RSet, typename Collector>
    static
    auto
    intersect_xlat_ds( LSet && lset, RSet && rset, Collector & out ) {
	static_assert( std::is_same_v<
		       typename std::decay_t<LSet>::type,
		       typename std::decay_t<RSet>::type>,
		       "Sets must contain elements of the same type" );
	
	// Collector is of type pointer to element of set
	// Construct custom collector object to have unified code base
	// for storing intersection with pointer or with a custom class.
	if constexpr ( std::is_same_v<std::decay_t<Collector>,
		       std::add_pointer_t<typename std::decay_t<LSet>::type>> ) {
	    intersection_collector<true,typename std::decay_t<LSet>::type>
		cout( out );
	    auto task = create_intersection_task<so_intersect_xlat>( cout );
	    return apply( lset, rset, task );
	} else {
	    auto task = create_intersection_task<so_intersect_xlat>( out );
	    return apply( lset, rset, task );
	}
    }

    template<typename LSet, typename RSet>
    static
    size_t
    intersect_size_ds( LSet && lset, RSet && rset ) {
	static_assert( std::is_same_v<
		       typename std::decay_t<LSet>::type,
		       typename std::decay_t<RSet>::type>,
		       "Sets must contain elements of the same type" );
	
	intersection_task<so_intersect_size,void> task;
	return apply( lset, rset, task );
    }

    template<typename LSet, typename RSet>
    static
    size_t
    intersect_size_gt_val_ds( LSet && lset, RSet && rset, size_t threshold ) {
	static_assert( std::is_same_v<
		       typename std::decay_t<LSet>::type,
		       typename std::decay_t<RSet>::type>,
		       "Sets must contain elements of the same type" );

#if ABLATION_DISABLE_ADV_INTERSECT
	return intersect_size_ds( lset, rset );
#else
	intersection_task<so_intersect_size_gt_val,void> task( threshold );
	return apply( lset, rset, task );
#endif
    }

    template<typename LSet, typename RSet>
    static
    bool
    intersect_size_ge_ds( LSet && lset, RSet && rset, size_t threshold ) {
	static_assert( std::is_same_v<
		       typename std::decay_t<LSet>::type,
		       typename std::decay_t<RSet>::type>,
		       "Sets must contain elements of the same type" );

#if ABLATION_DISABLE_ADV_INTERSECT
	return intersect_size_ds( lset, rset ) >= threshold;
#else
	intersection_task<so_intersect_size_ge,void> task( threshold );
	return apply( lset, rset, task );
#endif
    }

    template<typename LSet, typename RSet, typename Task>
    static
    auto
    apply( LSet && lset, RSet && rset, Task & task ) {
	// Corner case
	if( lset.size() == 0 || rset.size() == 0 )
	    return task.return_value_empty_set();

#if INTERSECTION_TRIM == 0
	auto & tlset = lset;
#else
	// Trim front of range.
	// There is no benefit in trimming the end of the range for merge
	// intersections, although there is for hash intersection.
	auto tlset = rset.has_sequential()
	    ? lset.trim_front( rset.front() ) : lset;
#endif
	return so_traits::apply( tlset, rset, task );
    }
};

struct merge_scalar {

    static constexpr bool uses_hash = false;

    template<set_operation so, bool rhs,
	     typename LIt, typename RIt, typename Collector>
    static
    std::pair<LIt,RIt>
    intersect_task( LIt lb, LIt le, RIt rb, RIt re, Collector & out ) {
	static_assert( so != so_intersect_xlat,
		       "intersect-with-translate not supported in merge-based"
		       " intersection" );

	if( out.terminated() )
	    return std::make_pair( lb, rb );

	while( lb != le && rb != re ) {
	    LIt olb = lb;
	    LIt orb = rb;

	    // translation not supported
	    if( *lb == *rb ) {
		++lb;
		++rb;
	    } else if( *lb < *rb )
		++lb;
	    else
		++rb;

	    if( !out.template record<rhs>( olb, orb, *olb == *orb ) )
		break;
	}

	out.template remainder<rhs>( le - lb, re - rb );

	return std::make_pair( lb, rb );
    }

    template<set_operation so, bool rhs,
	     typename LSet, typename RSet, typename Collector>
    static
    void
    intersect_task( LSet && lset, RSet && rset, Collector & out ) {
	auto lb = lset.begin();
	auto le = lset.end();
	auto rb = rset.begin();
	auto re = rset.end();
	intersect_task<so,rhs>( lb, le, rb, re, out );
    }
    
    template<typename LSet, typename RSet, typename Task>
    static
    auto
    apply( LSet && lset, RSet && rset, Task & task ) {
	auto out = task.template create_collector<merge_scalar>( lset, rset );
	intersect_task<Task::operation,true>( lset, rset, out );
	return out.return_value();
    }

    template<typename It, typename Ot>
    static
    Ot intersect( It lb, It le, It rb, It re, Ot o ) {
	return intersect<false>( lb, le, rb, re, o );
    }

    template<bool send_lhs_ptr, typename It, typename Ot>
    static
    Ot intersect( It lb, It le, It rb, It re, Ot o ) {
	It l = lb;
	It r = rb;

	while( l != le && r != re ) {
	    if( *l == *r ) {
		if constexpr ( send_lhs_ptr )
		    o.push_back( l, r );
		else
		    *o++ = *l;
		++l;
		++r;
	    } else if( *l < *r )
		++l;
	    else
		++r;
	}

	return o;
    }

    template<typename It>
    static
    size_t intersect_size( It lb, It le, It rb, It re ) {
	It l = lb;
	It r = rb;
	size_t sz = 0;

	while( l != le && r != re ) {
	    if( *l == *r ) {
		++sz;
		++l;
		++r;
	    } else if( *l < *r )
		++l;
	    else
		++r;
	}

	return sz;
    }

    template<typename It>
    static
    size_t intersect_size_gt_val( It lb, It le, It rb, It re, size_t exceed ) {
	size_t ld = std::distance( lb, le );
	size_t rd = std::distance( rb, re );
	if( ld > rd ) {
	    std::swap( lb, rb );
	    std::swap( le, re );
	    std::swap( ld, rd );
	}

	It l = lb;
	It r = rb;
	size_t d = std::min( ld, rd );

	if( d < exceed )
	    return 0;

	std::make_signed_t<size_t> options = d - exceed;

	while( l != le && r != re ) {
	    if( *l == *r ) {
		++l;
		++r;
	    } else if( *l < *r ) {
		if( --options < 0 )
		    return 0;
		++l;
	    } else {
		++r;
	    }
	}

	return options + exceed - ( le - l );
    }

};

struct merge_scalar_jump {

    static constexpr bool uses_hash = false;

    template<set_operation so, bool rhs,
	     typename LIt, typename RIt, typename Collector>
    static
    std::pair<LIt,RIt>
    intersect_task( LIt lb, LIt le, RIt rb, RIt re, Collector & out ) {
	static_assert( so != so_intersect_xlat,
		       "intersect-with-translate not supported in merge-based"
		       " intersection" );

	if( out.terminated() )
	    return std::make_pair( lb, rb );

	while( lb != le && rb != re ) {
	    if( !out.template record<rhs>( lb, rb, *lb == *rb ) )
		break;
	    if( *lb == *rb ) {
		++lb;
		++rb;
	    } else if( *lb < *rb ) {
		++lb;
		LIt lj = jump( lb, le, *rb );
		out.template remainder<rhs>( std::distance( lb, lj ), 0 );
		lb = lj;
	    } else {
		++rb;
		RIt rj = jump( rb, re, *lb );
		out.template remainder<rhs>( 0, std::distance( rb, rj ) );
		rb = rj;
	    }
	}

	out.template remainder<rhs>( le - lb, re - rb );

	return std::make_pair( lb, rb );
    }

    template<set_operation so, bool rhs,
	     typename LSet, typename RSet, typename Collector>
    static
    void
    intersect_task( LSet && lset, RSet && rset, Collector & out ) {
	auto lb = lset.begin();
	auto le = lset.end();
	auto rb = rset.begin();
	auto re = rset.end();
	intersect_task<so,rhs>( lb, le, rb, re, out );
    }
    
    template<typename LSet, typename RSet, typename Task>
    static
    auto
    apply( LSet && lset, RSet && rset, Task & task ) {
	auto out =
	    task.template create_collector<merge_scalar_jump>( lset, rset );
	intersect_task<Task::operation,true>( lset, rset, out );
	return out.return_value();
    }


    //! Jump ahead, aiming to skip over many non-matching values
    //
    // The method differs from binary search for ref in the range [b,e)
    // as we are not interested in finding the value, we merely want to find
    // a position in the range that points to a value that is not larger than
    // ref and is as far in the range as possible.
    //
    // \tparam It Iterator type for sequence. Assumes random access iterator.
    // \tparam T Type of value resulting from dereferencing It
    //
    // \arg b Begin of range.
    // \arg e End of range.
    // \arg ref Reference value which is not to be exceeded.
    // \return Iterator in the range [b,e) such that the iterator points to a
    //         value that is not higher than ref, or the iterator e is returned
    //         if all values in [b,e) are less than ref.
    template<typename It, typename T>
    static
    It jump( It b, It e, T ref ) {
	if( ref <= *b )
	    return b;

	// Search for the furthest position in bounds and not higher than ref
	It i = b;
	size_t off = 1;
	while( i+off < e && *(i+off) < ref ) {
	    i += off;
	    off <<= 1;
	}
	return i;
    }

    //! Repeated jumping: reset step size when step size too large.
    template<typename It, typename T>
    static
    It jump_precise( It b, It e, T ref ) {
	// Search for the furthest position in bounds and not higher than ref
	size_t off = 1;
	while( b != e && ref > *b ) {
	    if( b+off >= e || *(b+off) > ref ) {
		off >>= 1;
		if( off == 0 )
		    break;
	    } else if( *(b+off) <= ref ) {
		b += off;
		off <<= 1;
	    }
	}
	return b;
    }

    //! Jump ahead with binary search to land on next value or just above.
    template<typename It, typename T>
    static
    It jump_binary_search( It b, It e, T ref ) {
	if( ref <= *b )
	    return b;

	// Search for the furthest position in bounds and not higher than ref
	It i = b;
	size_t off = 1;
	while( i+off < e && *(i+off) < ref ) {
	    i += off;
	    off <<= 1;
	}

	// Work with tighter bounds
	b = i;
	if( i+off < e )
	    e = i + off;

	if( ref <= *b )
	    return b;

	while( std::distance( b, e ) > 1 ) {
	    It mid = std::next( b, std::distance( b, e ) / 2 );
	    if( ref == *mid )
		return mid;
	    else if( ref < *mid ) {
		if( mid != b && ref >= *std::prev( mid, 1 ) )
		    return std::prev( mid, 1 );
		else
		    e = mid;
	    } else
		b = mid;
	}

	return b;
    }

    template<typename It, typename Ot>
    static
    Ot intersect( It lb, It le, It rb, It re, Ot o ) {
	return intersect<false>( lb, le, rb, re, o );
    }

    template<bool send_lhs_ptr, typename It, typename Ot>
    static
    Ot intersect( It lb, It le, It rb, It re, Ot o ) {
	It l = lb;
	It r = rb;

	while( l != le && r != re ) {
	    if( *l == *r ) {
		if constexpr ( send_lhs_ptr )
		    o.push_back( l, r );
		else
		    *o++ = *l;
		++l;
		++r;
	    } else if( *l < *r ) {
		++l;
		l = jump( l, le, *r );
	    } else {
		++r;
		r = jump( r, re, *l );
	    }
	}
	return o;
    }

    template<typename It>
    static
    size_t intersect_size( It lb, It le, It rb, It re ) {
	It l = lb;
	It r = rb;
	size_t sz = 0;

	while( l != le && r != re ) {
	    if( *l == *r ) {
		++sz;
		++l;
		++r;
	    } else if( *l < *r ) {
		++l;
		l = jump( l, le, *r );
	    } else {
		++r;
		r = jump( r, re, *l );
	    }
	}

	return sz;
    }

    template<typename It>
    static
    size_t intersect_size_gt_val( It lb, It le, It rb, It re, size_t exceed ) {
	size_t ld = std::distance( lb, le );
	size_t rd = std::distance( rb, re );
	if( ld > rd ) {
	    std::swap( lb, rb );
	    std::swap( le, re );
	    std::swap( ld, rd );
	}

	It l = lb;
	It r = rb;
	size_t d = std::min( ld, rd );

	if( d < exceed )
	    return 0;

	std::make_signed_t<size_t> options = d - exceed;

	while( l != le && r != re ) {
	    if( *l == *r ) {
		++l;
		++r;
	    } else if( *l < *r ) {
		++l;
		It lf = jump( l, le, *r );
		options -= std::distance( l, lf ) + 1;
		l = lf;
		if( options <= 0 )
		    return 0;
	    } else {
		++r;
		r = jump( r, re, *l );
	    }
	}

	return options + exceed - ( le - l );
    }

};

struct merge_vector_jump {
    template<unsigned short VL, typename T>
    static
    const T * jump( const T * b, const T * e, T ref ) {
	using tr = vector_type_traits_vl<T,VL>;
	using type = typename tr::type;
	using mask_type = typename tr::mask_type;

	if( ref <= *b )
	    return b;

	// Search for the furthest position in bounds and not higher than ref
	type step = tr::sllv( tr::setoneval(), tr::set1inc0() );
	size_t voff = size_t(1) << ( VL - 1 ); // same as step lane VL-1
	const type vref = tr::set1( ref );
	const T * i = b;
	while( i+voff < e ) {
	    type val = tr::gather( i, step );
	    mask_type fnd = tr::cmplt( val, vref, target::mt_mask() );
	    if( !tr::mask_traits::is_ones( fnd ) ) {
		if( tr::mask_traits::is_zero( fnd ) )
		    return i; // do not advance
		else {
		    // Lane is the first lane containing a value >= reference.
		    // Return the lane before it.
		    size_t lane = tr::mask_traits::find_first( fnd );
		    return i + tr::lane( step, lane-1 ); // remain before ref
		}
	    }
	    step = tr::slli( step, VL );
	    i += voff;
	    voff <<= VL;
	}
	voff >>= ( VL - 1 );
	while( i+voff < e && *(i+voff) < ref ) {
	    i += voff;
	    voff <<= 1;
	}
	return i;
    }

    template<set_operation so, typename T, unsigned short VL, bool rhs,
	     typename LIt, typename RIt, typename Collector>
    static
    std::pair<LIt,RIt>
    intersect_task_vl( LIt lb, LIt le, RIt rb, RIt re, Collector & out ) {
	static_assert( so != so_intersect_xlat,
		       "intersect-with-translate not supported in merge-based"
		       " intersection" );

	using tr = vector_type_traits_vl<T,VL>;
	using type = typename tr::type;
	using mask_type = typename tr::mask_type;

	if( out.terminated() )
	    return std::make_pair( lb, rb );

	while( lb+VL <= le && rb+VL <= re ) {
	    type vl = tr::loadu( lb );

	    mask_type ma = tr::intersect( vl, rb );

	    type lf = tr::set1( lb[VL-1] );
	    type rf = tr::set1( rb[VL-1] );

	    mask_type ladv = tr::cmple( vl, rf, target::mt_mask() );
	    mask_type radv = tr::cmple( tr::loadu( rb ), lf, target::mt_mask() );
	    size_t la = rt_ilog2( ((uint64_t)ladv) + 1 );
	    size_t ra = rt_ilog2( ((uint64_t)radv) + 1 );

	    lb += la;
	    rb += ra;

	    if( !out.template multi_record<rhs,T,VL>( lb-la, vl, ma, la, ra ) )
		break;

	    // If not a single element in the LHS vector was present on the RHS
	    // side, perhaps many more will follow. Try jumping.
	    if( la == VL ) {
		LIt ln = merge_scalar_jump::jump( lb, le, *rb );
		// LIt ln = jump<VL>( lb, le, *rb );
		out.template remainder<rhs>( std::distance( lb, ln ), 0 );
		lb = ln;
	    }
	    // Similar jumping for RHS without matches.
	    if( ra == VL ) {
		LIt rn = merge_scalar_jump::jump( rb, re, *lb );
		// LIt rn = jump<VL>( rb, re, *lb );
		out.template remainder<rhs>( 0, std::distance( rb, rn ) );
		rb = rn;
	    }
	}

	if( lb+VL == le && rb+VL == re ) // always false?
	    out.template remainder<rhs>( le - lb, re - rb );
	
	return std::make_pair( lb, rb );
    }

    template<set_operation so, bool rhs,
	     typename LSet, typename RSet, typename Collector>
    static
    void
    intersect_task( LSet && lset, RSet && rset, Collector & out ) {
	using T = typename std::decay_t<LSet>::type;
	
	auto lb = lset.begin();
	auto le = lset.end();
	auto rb = rset.begin();
	auto re = rset.end();
#if defined( __AVX512F__ )
	std::tie( lb, rb ) =
	    intersect_task_vl<so,T,64/sizeof(T),rhs>( lb, le, rb, re, out );
#elif defined( __AVX2__ )
	std::tie( lb, rb ) =
	    intersect_task_vl<so,T,32/sizeof(T),rhs>( lb, le, rb, re, out );
#endif
	// Why not jump? We know one of the lists is short (less than VL),
	// but the other may be very long.
	merge_scalar::intersect_task<so,rhs>( lb, le, rb, re, out );
    }
    
    template<typename LSet, typename RSet, typename Task>
    static
    auto
    apply( LSet && lset, RSet && rset, Task & task ) {
	auto out =
	    task.template create_collector<merge_vector_jump>( lset, rset );
	intersect_task<Task::operation,true>( lset, rset, out );
	return out.return_value();
    }
};

struct hash_scalar {

    static constexpr bool uses_hash = true;

    template<typename It, typename HT, typename B>
    static void
    intersect_xlat( It lb, It le, const HT & htable, B & build ) {
	while( lb != le ) {
	    VID v = *lb;
	    build.insert( lb, htable.contains( v ) );
	    ++lb;
	}
    }

    template<set_operation so, bool rhs,
	     typename LIt, typename RSet, typename Collector>
    static
    LIt
    intersect_task( LIt lb, LIt le, RSet && rset, Collector & out ) {
	if( out.terminated() )
	    return lb;

	// It is assumed that the caller has already determined
	// that the left set is the smaller one.
	while( lb != le ) {
	    VID v = *lb;
	    if constexpr ( so == so_intersect_xlat ) {
		auto rc = rset.lookup( v );
		// all 1s indicates invalid/absent value
		if( ~rc != 0 && !out.template record<rhs>( lb, rc, true ) )
		    break;
	    } else {
		auto rc = rset.contains( v );
		if( !out.template record<rhs>( lb, *lb, rc ) )
		    break;
	    }
	    ++lb;
	}

	return lb;
    }

    template<set_operation so, bool rhs,
	     typename LSet, typename RSet, typename Collector>
    static
    void
    intersect_task( LSet && lset, RSet && rset, Collector & out ) {
#if INTERSECTION_TRIM == 0
	auto & tlset = lset;
#else
	auto tlset = rset.has_sequential()
	    ? lset.trim_back( rset.back() ) : lset;
	out.swap( tlset, rset );
#endif
	auto lb = tlset.begin();
	auto le = tlset.end();
	intersect_task<so,rhs>( lb, le, rset, out );
    }

    template<typename LSet, typename RSet, typename Task>
    static
    auto
    apply( LSet && lset, RSet && rset, Task & task ) {
	static_assert( is_hash_set_v<std::decay_t<LSet>>
		       || is_hash_set_v<std::decay_t<RSet>>,
		       "at least one of arguments should be hash set" );
	if constexpr ( is_hash_set_v<std::decay_t<LSet>>
		       && is_hash_set_v<std::decay_t<RSet>> ) {
	    if( lset.size() <= rset.size() ) {
		auto out = task.template create_collector<hash_scalar>(
		    lset, rset );
		intersect_task<Task::operation,true>( lset, rset, out );
		return out.return_value();
	    } else {
		auto out = task.template create_collector<hash_scalar>(
		    rset, lset );
		intersect_task<Task::operation,false>( rset, lset, out );
		return out.return_value();
	    }
	} else if constexpr ( is_hash_set_v<std::decay_t<RSet>> ) {
	    auto out = task.template create_collector<hash_scalar>(
		lset, rset );
	    intersect_task<Task::operation,true>( lset, rset, out );
	    return out.return_value();
	} else {
#if INTERSECTION_TRIM == 0
	    auto & trset = rset;
#else
	    auto trset = lset.has_sequential()
		? rset.trim_front( lset.front() ) : rset;
#endif
	    auto out = task.template create_collector<hash_scalar>(
		trset, lset );
	    intersect_task<Task::operation,false>( trset, lset, out );
	    return out.return_value();
	}
    }
    
    template<typename It, typename HT, typename Ot>
    static
    Ot intersect( It lb, It le, const HT & htable, Ot o ) {
	while( lb != le ) {
	    VID v = *lb;
	    if( htable.contains( v ) )
		*o++ = v;
	    ++lb;
	}
	return o;
    }

    template<bool send_lhs_ptr, typename It, typename HT, typename Ot>
    static
    Ot intersect( It lb, It le, const HT & htable, Ot o ) {
	while( lb != le ) {
	    VID v = *lb;
	    if( htable.contains( v ) ) {
		if constexpr ( send_lhs_ptr )
		    o.push_back( lb );
		else
		    *o++ = v;
	    }
	    ++lb;
	}
	return o;
    }

    template<bool send_lhs_ptr, typename It, typename HT, typename Ot>
    static
    Ot intersect_invert( It lb, It le, const HT & htable, Ot o ) {
	while( lb != le ) {
	    VID v = *lb;
	    if( !htable.contains( v ) ) {
		if constexpr ( send_lhs_ptr )
		    o.push_back( lb );
		else
		    *o++ = v;
	    }
	    ++lb;
	}
	return o;
    }

    template<bool send_lhs_ptr, typename It, typename HT, typename Ot>
    static
    Ot intersect3( It lb, It le, const HT & htable1, const HT & htable2,
		   Ot o ) {
	while( lb != le ) {
	    VID v = *lb;
	    if( htable1.contains( v ) && htable2.contains( v ) ) {
		if constexpr ( send_lhs_ptr )
		    o.push_back( lb );
		else
		    *o++ = v;
	    }
	    ++lb;
	}
	return o;
    }

    template<bool send_lhs_ptr, typename It, typename HT, typename Ot>
    static
    Ot intersect3not( It lb, It le, const HT & htable1, const HT & htable2,
		   Ot o ) {
	while( lb != le ) {
	    VID v = *lb;
	    if( htable1.contains( v ) && !htable2.contains( v ) ) {
		if constexpr ( send_lhs_ptr )
		    o.push_back( lb );
		else
		    *o++ = v;
	    }
	    ++lb;
	}
	return o;
    }

    template<typename It, typename HT>
    static
    size_t intersect_size( It lb, It le, const HT & htable ) {
	size_t sz = 0;
	while( lb != le ) {
	    VID v = *lb;
	    if( htable.contains( v ) )
		++sz;
	    ++lb;
	}
	return sz;
    }

    template<typename It, typename HT>
    static
    size_t intersect_size_gt_val( It lb, It le, const HT & htable, size_t exceed ) {
	size_t d = std::distance( lb, le );

	if( d <= exceed )
	    return 0;

	std::make_signed_t<size_t> options = d - exceed;

	while( lb != le ) {
	    VID v = *lb;
	    if( !htable.contains( v ) ) [[likely]]
		if( --options <= 0 ) [[unlikely]]
		    return 0;
	    ++lb;
	}

	return options + exceed;
    }

};

struct hash_scalar_jump {
    template<set_operation so, bool rhs,
	     typename LIt, typename RSet, typename Collector>
    static
    LIt
    intersect_task( LIt lb, LIt le, RSet && rset, Collector & out ) {
	if( out.terminated() )
	    return lb;

	auto rb = rset.begin();
	auto re = rset.end();

	// It is assumed that the caller has already determined
	// that the left set is the smaller one.
	while( lb != le ) {
	    VID v = *lb;
	    if constexpr ( so == so_intersect_xlat ) {
		auto rc = rset.lookup( v );
		// all 1s indicates invalid/absent value
		if( ~rc != 0 && !out.template record<rhs>( lb, rc, true ) )
		    break;
		++lb;
	    } else {
		auto rc = rset.contains( v );
		if( !out.template record<rhs>( lb, *lb, rc ) )
		    break;

		++lb;

		auto rn = merge_scalar_jump::jump( rb, re, *(lb-1) );
		auto ln = lb;
		if( *rn > *lb )
		    ln = merge_scalar_jump::jump( lb, le, *rn );

		out.template remainder<rhs>(
		    std::distance( lb, ln ),
		    std::distance( rb, rn ) );

		rb = rn;
		lb = ln;
	    }
	}

	return lb;
    }

    template<set_operation so, bool rhs,
	     typename LSet, typename RSet, typename Collector>
    static
    void
    intersect_task( LSet && lset, RSet && rset, Collector & out ) {
	auto lb = lset.begin();
	auto le = lset.end();
	intersect_task<so,rhs>( lb, le, rset, out );
    }

    template<typename LSet, typename RSet, typename Task>
    static
    auto
    apply( LSet && lset, RSet && rset, Task & task ) {
	static_assert( is_hash_set_v<std::decay_t<LSet>>
		       || is_hash_set_v<std::decay_t<RSet>>,
		       "at least one of arguments should be hash set" );
	if constexpr ( is_hash_set_v<std::decay_t<LSet>>
		       && is_hash_set_v<std::decay_t<RSet>> ) {
	    if( lset.size() <= rset.size() ) {
		auto out = task.template create_collector<hash_scalar_jump>(
		    lset, rset );
		intersect_task<Task::operation,true>( lset, rset, out );
		return out.return_value();
	    } else {
		auto out = task.template create_collector<hash_scalar_jump>(
		    rset, lset );
		intersect_task<Task::operation,false>( rset, lset, out );
		return out.return_value();
	    }
	} else if constexpr ( is_hash_set_v<std::decay_t<RSet>> ) {
	    auto out = task.template create_collector<hash_scalar_jump>(
		lset, rset );
	    intersect_task<Task::operation,true>( lset, rset, out );
	    return out.return_value();
	} else {
	    auto out = task.template create_collector<hash_scalar_jump>(
		rset, lset );
	    intersect_task<Task::operation,false>( rset, lset, out );
	    return out.return_value();
	}
    }
};

#if 0
struct hash_wide {

    static constexpr bool uses_hash = true;

    template<typename It, typename HT, typename Ot>
    static
    Ot intersect( It lb, It le, const HT & htable, Ot o ) {
	while( lb != le ) {
	    VID v = *lb;
	    if( htable.template wide_contains<8>( v ) )
		*o++ = v;
	    ++lb;
	}
	return o;
    }

    template<bool send_lhs_ptr, typename It, typename HT, typename Ot>
    static
    Ot intersect( It lb, It le, const HT & htable, Ot o ) {
	while( lb != le ) {
	    VID v = *lb;
	    if( htable.template wide_contains<8>( v ) ) {
		if constexpr ( send_lhs_ptr )
		    o.push_back( lb );
		else
		    *o++ = v;
	    }
	    ++lb;
	}
	return o;
    }

    template<typename It, typename HT>
    static
    size_t intersect_size( It lb, It le, const HT & htable ) {
	size_t sz = 0;
	while( lb != le ) {
	    VID v = *lb;
	    if( htable.template wide_contains<8>( v ) )
		++sz;
	    ++lb;
	}
	return sz;
    }

    template<typename It, typename HT>
    static
    size_t intersect_size_gt_val( It lb, It le, const HT & htable, size_t exceed ) {
	size_t d = std::distance( lb, le );

	if( d <= exceed )
	    return 0;

	std::make_signed_t<size_t> options = d - exceed;

	while( lb != le ) {
	    VID v = *lb;
	    if( !htable.template wide_contains<8>( v ) ) [[likely]]
		if( --options <= 0 ) [[unlikely]]
		    return 0;
	    ++lb;
	}

	return options + exceed;
    }

};
#endif


struct hash_vector {

    static constexpr bool uses_hash = true;

private:
    template<unsigned VL, typename T, typename HT, typename B>
    static
    const T *
    // __attribute__((noinline))
    detail_intersect_xlat(
	const T * lb, const T * le, const HT & htable, B & build ) {
	using tr = vector_type_traits_vl<T,VL>;
	using type = typename tr::type;

	while( lb+VL <= le ) {
	    auto x = htable.template multi_contains<T,VL>(
		tr::loadu( lb ), target::mt_vmask() );
	    build.multi_insert( lb, x );
	    lb += VL;
	}

	return lb;
    }

   template<bool send_lhs_ptr,
	    unsigned VL, bool store, bool invert,
	    typename T, typename HT, typename Ot>
    static
    const T *
    // __attribute__((noinline))
    detail_intersect(
	const T * lb, const T * le, const HT & htable, Ot & out ) {
	using tr = vector_type_traits_vl<T,VL>;
	using type = typename tr::type;

	while( lb+VL <= le ) {
	    type v = tr::loadu( lb );
	    typename target::mask_type_traits<VL>::type m
		= htable.template multi_contains<T,VL>( v, target::mt_mask() );
	    if constexpr ( invert )
		m = target::mask_type_traits<VL>::logical_invert( m );
	    if constexpr ( send_lhs_ptr )
		out.template push_back<VL>( m, v, lb );
	    else {
		if constexpr ( store )
		    out = tr::cstoreu_p( out, m, v );
		else
		    out += _popcnt32( m );
	    }
	    lb += VL;
	}

	return lb;
    }

    template<bool send_lhs_ptr, bool inv,
	     unsigned VL, bool store, typename T, typename HT, typename Ot>
    static
    const T *
    detail_intersect3(
	const T * lb, const T * le, const HT & htable1,
	const HT & htable2, Ot & out ) {
	using tr = vector_type_traits_vl<T,VL>;
	using mtr = target::mask_type_traits<VL>;
	using type = typename tr::type;

	while( lb+VL <= le ) {
	    type v = tr::loadu( lb );
	    typename mtr::type m
		= htable1.template multi_contains<T,VL>( v, target::mt_mask() );
	    typename mtr::type m2
		= htable2.template multi_contains<T,VL>( v, target::mt_mask() );
	    if constexpr ( inv )
		m = mtr::logical_and( m2, m );
	    else
		m = mtr::logical_andnot( m2, m );
	    if constexpr ( send_lhs_ptr )
		out.template push_back<VL>( m, v, lb );
	    else {
		if constexpr ( store )
		    tr::cstoreu( out, m, v );
		out += _popcnt32( m );
	    }
	    lb += VL;
	}

	return lb;
    }

    template<unsigned VL, typename T, typename HT>
    static
    bool
    detail_intersect_size_gt_val(
	const T *& lb, const T * le, const HT & htable, size_t exceed,
	size_t & sz ) {
	using tr = vector_type_traits_vl<T,VL>;
	using type = typename tr::type;

	size_t d = std::distance( lb, le );

	if( d < exceed )
	    return false;

	std::make_signed_t<size_t> options = d - exceed;
	const T * l = lb;

	while( l+VL <= le ) {
	    type v = tr::loadu( l );
	    auto m = htable.template multi_contains<T,VL>(
		v, target::mt_mask() );
	    options -= VL - _popcnt32( m );
	    if( options < 0 ) {
		lb = l;
		return false;
	    }
	    l += VL;
	}

	lb = l;
	sz = options + exceed - ( le - l );
	return true;
    }

    template<bool inv, typename T, typename HT, typename Ot>
    static
    Ot detail_intersect3_vec( const T * lb, const T * le, const HT & htable1,
			      const HT & htable2, Ot out ) {
#if __AVX512F__
	lb = detail_intersect3<false,inv,64/sizeof(T),true>(
	    lb, le, htable1, htable2, out );
#endif
#if __AVX2__
	lb = detail_intersect3<false,inv,32/sizeof(T),true>(
	    lb, le, htable1, htable2, out );
#endif
	if constexpr ( inv )
	    out = graptor::hash_scalar::template intersect3not<false>(
		lb, le, htable1, htable2, out );
	else
	    out = graptor::hash_scalar::template intersect3<false>(
		lb, le, htable1, htable2, out );

	return out;
    } 

public:
    template<typename T>
    static const T * adjust_start( const T * start, const T * desired ) {
	constexpr size_t Bits = sizeof( T ) * 8;
#if __AVX512F__
	constexpr unsigned VL = 512/Bits;
#elif __AVX2__
	constexpr unsigned VL = 256/Bits;
#else
	constexpr unsigned VL = 1;
#endif
	return start + ( std::distance( start, desired ) & ~( VL - 1 ) );
    }
    
    template<typename T, typename HT, typename Ot>
    static
    Ot
    __attribute__((noinline))
    intersect( const T * lb, const T * le, const HT & htable, Ot out ) {
	return intersect<false>( lb, le, htable, out );
    } 

    template<typename T, typename HT, typename Ot>
    static
    Ot intersect3( const T * lb, const T * le, const HT & htable1,
		   const HT & htable2, Ot out ) {
	return detail_intersect3_vec<false>( lb, le, htable1, htable2, out );
    } 
    template<typename T, typename HT, typename Ot>
    static
    Ot intersect3not( const T * lb, const T * le, const HT & htable1,
		      const HT & htable2, Ot out ) {
	return detail_intersect3_vec<true>( lb, le, htable1, htable2, out );
    } 

    template<bool send_lhs_ptr, typename T, typename HT, typename Ot>
    static
    Ot intersect( const T * lb, const T * le, const HT & htable, Ot out ) {
#if __AVX512F__
	lb = detail_intersect<send_lhs_ptr,64/sizeof(T),true,false>( lb, le, htable, out );
#endif
#if __AVX2__
	lb = detail_intersect<send_lhs_ptr,32/sizeof(T),true,false>( lb, le, htable, out );
#endif
	out = graptor::hash_scalar::template intersect<send_lhs_ptr>(
	    lb, le, htable, out );

	return out;
    }

    template<bool send_lhs_ptr, typename T, typename HT, typename Ot>
    static
    Ot intersect_invert( const T * lb, const T * le, const HT & htable,
			 Ot out ) {
#if __AVX512F__
	lb = detail_intersect<send_lhs_ptr,64/sizeof(T),true,true>( lb, le, htable, out );
#endif
#if __AVX2__
	lb = detail_intersect<send_lhs_ptr,32/sizeof(T),true,true>( lb, le, htable, out );
#endif
	out = graptor::hash_scalar::template intersect_invert<send_lhs_ptr>(
	    lb, le, htable, out );

	return out;
    }

    template<typename T, typename HT>
    static
    size_t intersect_size( const T * lb, const T * le, const HT & htable ) {
	size_t sz = 0;
#if __AVX512F__
	lb = detail_intersect<false,64/sizeof(T),false,false>( lb, le, htable, sz );
#endif
#if __AVX2__
	lb = detail_intersect<false,32/sizeof(T),false,false>( lb, le, htable, sz );
#endif
	sz += hash_scalar::intersect_size( lb, le, htable );

	return sz;
    }

    template<typename T, typename HT>
    static
    size_t
    intersect_size_gt_val(
	const T * lb, const T * le, const HT & htable, size_t exceed ) {
	size_t sz0 = 0, sz1 = 0, sz2;

#if __AVX512F__
	if( !detail_intersect_size_gt_val<64/sizeof(T)>(
		lb, le, htable, exceed, sz0 ) )
	    return 0;
	if( sz0 > exceed )
	    return sz0 + hash_scalar::intersect_size( lb, le, htable );
	exceed -= sz0;
#endif
#if __AVX2__
	if( !detail_intersect_size_gt_val<32/sizeof(T)>(
		lb, le, htable, exceed, sz1 ) )
	    return 0;
	if( sz1 > exceed )
	    return sz0 + sz1 + hash_scalar::intersect_size( lb, le, htable );
	exceed -= sz1;
#endif
	sz2 = hash_scalar::intersect_size_gt_val( lb, le, htable, exceed );
	if( !sz2 )
	    return 0;

	return sz0 + sz1 + sz2;
    }

    template<set_operation so, typename T, unsigned VL, bool rhs,
	     typename LIt, typename RSet, typename Collector>
    static
    LIt
    intersect_task_vl( LIt lb, LIt le, RSet && rset, Collector & out ) {
	if( out.terminated() )
	    return lb;

	using tr = vector_type_traits_vl<T,VL>;
	using type = typename tr::type;

	while( lb+VL <= le ) {
	    // Load sequence of values from left-hand argument
	    type v = tr::loadu( lb );

	    if constexpr ( so == so_intersect_xlat ) {
		// Convert through hash table
		// Returns a pair of { present, translated }
		auto m = rset.template multi_lookup<T,VL>(
		    v, target::mt_mask() );

		// Record / count common values
		if( !out.template multi_record<rhs,T,VL>( lb, m.second, m.first, 0 ) )
		    break;
	    } else {
		// Check present in hash set. Returns a mask.
		auto m = rset.template multi_contains<T,VL>(
		    v, target::mt_mask() );

		// Record / count common values
		if( !out.template multi_record<rhs,T,VL>( lb, v, m, VL, 0 ) )
		    break;
	    }

	    lb += VL;
	}

	return lb;
    }

    template<set_operation so, bool rhs,
	     typename LSet, typename RSet, typename Collector>
    static
    void
    intersect_task( LSet && lset, RSet && rset, Collector & out ) {
	using T = typename std::decay_t<LSet>::type;

#if INTERSECTION_TRIM == 0
	auto & tlset = lset;
#else
	auto tlset = rset.has_sequential()
	    ? lset.trim_back( rset.back() ) : lset;
	out.swap( tlset, rset );
#endif

	auto lb = tlset.begin();
	auto le = tlset.end();

#if defined( __AVX512F__ )
	lb = intersect_task_vl<so,T,64/sizeof(T),rhs>( lb, le, rset, out );
#elif defined( __AVX2__ )
	lb = intersect_task_vl<so,T,32/sizeof(T),rhs>( lb, le, rset, out );
#endif
	hash_scalar::intersect_task<so,rhs>( lb, le, rset, out );
    }

    template<typename LSet, typename RSet, typename Task>
    static
    auto
    apply( LSet && lset, RSet && rset, Task & task ) {
	static_assert( is_multi_hash_set_v<std::decay_t<LSet>>
		       || is_multi_hash_set_v<std::decay_t<RSet>>,
		       "at least one of arguments should be hash set" );
	if constexpr ( is_multi_hash_set_v<std::decay_t<LSet>>
		       && is_multi_hash_set_v<std::decay_t<RSet>> ) {
	    if( lset.size() <= rset.size() ) {
		auto out = task.template create_collector<hash_vector>(
		    lset, rset );
		intersect_task<Task::operation,true>( lset, rset, out );
		return out.return_value();
	    } else {
		auto out = task.template create_collector<hash_vector>(
		    rset, lset );
		intersect_task<Task::operation,false>( rset, lset, out );
		return out.return_value();
	    }
	} else if constexpr ( is_multi_hash_set_v<std::decay_t<RSet>> ) {
	    auto out = task.template create_collector<hash_vector>(
		lset, rset );
	    intersect_task<Task::operation,true>( lset, rset, out );
	    return out.return_value();
	} else {
#if INTERSECTION_TRIM == 0
	    auto & trset = rset;
#else
	    auto trset = lset.has_sequential()
		? rset.trim_front( lset.front() ) : rset;
#endif

	    auto out = task.template create_collector<hash_vector>(
		trset, lset );
	    intersect_task<Task::operation,false>( trset, lset, out );
	    return out.return_value();
	}
    }
    
};

#if 0
struct hash_vector_jump {
    template<set_operation so, typename T, unsigned VL, bool rhs,
	     typename LIt, typename RSet, typename Collector>
    static
    LIt
    intersect_task_vl( LIt lb, LIt le, RSet && rset, Collector & out ) {
	if( out.terminated() )
	    return lb;

	using tr = vector_type_traits_vl<T,VL>;
	using type = typename tr::type;

	auto rb = rset.begin();
	auto re = rset.end();

	while( lb+VL <= le ) {
	    // Load sequence of values from left-hand argument
	    type v = tr::loadu( lb );

	    if constexpr ( so == so_intersect_xlat ) {
		// Convert through hash table
		// Returns a pair of { present, translated }
		auto m = rset.template multi_lookup<T,VL>(
		    v, target::mt_mask() );

		// Record / count common values
		if( !out.template multi_record<rhs,T,VL>( lb, m.second, m.first, 0 ) )
		    break;

		lb += VL;
	    } else {
		// Check present in hash set. Returns a mask.
		auto m = rset.template multi_contains<T,VL>(
		    v, target::mt_mask() );

		// Record / count common values
		if( !out.template multi_record<rhs,T,VL>( lb, v, m, VL, 0 ) )
		    break;

		lb += VL;
	    }
	}

	return lb;
    }

    template<set_operation so, bool rhs,
	     typename LSet, typename RSet, typename Collector>
    static
    void
    intersect_task( LSet && lset, RSet && rset, Collector & out ) {
	using T = typename std::decay_t<LSet>::type;

	auto lb = lset.begin();
	auto le = lset.end();

#if defined( __AVX512F__ )
	lb = intersect_task_vl<so,T,64/sizeof(T),rhs>( lb, le, rset, out );
#elif defined( __AVX2__ )
	lb = intersect_task_vl<so,T,32/sizeof(T),rhs>( lb, le, rset, out );
#endif
	hash_scalar::intersect_task<so,rhs>( lb, le, rset, out );
    }

    template<set_operation so,
	     typename LSet, typename RSet, typename Collector>
    static
    auto
    apply( LSet && lset, RSet && rset, Collector & out ) {
	static_assert( is_multi_hash_set_v<std::decay_t<LSet>>
		       || is_multi_hash_set_v<std::decay_t<RSet>>,
		       "at least one of arguments should be hash set" );
	// static_assert( is_multi_collector_v<std::decay_t<Collector>>,
		//        "collector must accept vectors of values" );
	if constexpr ( is_multi_hash_set_v<std::decay_t<LSet>>
		       && is_multi_hash_set_v<std::decay_t<RSet>> ) {
	    if( lset.size() <= rset.size() )
		return intersect_task<so,true>( lset, rset, out );
	    else {
		out.swap( rset, lset );
		return intersect_task<so,false>( rset, lset, out );
	    }
	} else if constexpr ( is_hash_set_v<std::decay_t<RSet>> ) {
	    return intersect_task<so,true>( lset, rset, out );
	} else {
	    out.swap( rset, lset );
	    return intersect_task<so,false>( rset, lset, out );
	}
    }
};
#endif


/*!=====================================================================*
 * TODO:
 * + Most vectors return no or few matches. Measure frequency of outcomes
 *   and adjust code accordingly.
 * + Search for "bulls" first and then solve smaller sub-problems (e.g.,
 *   intersecting up to the next bull, shorter vector, fewer comparisons),
 *   or specialise the computation to limit the number of rotations to
 *   consider while maintaining vector length.
 * + Consider how to move ahead in the stream (jumping?), but maintain
 *   invariant ladv, radv >= #matches and avoid counting duplicates.
 *   
 *======================================================================*/
struct merge_vector {

    static constexpr bool uses_hash = false;

    template<set_operation so, typename T, unsigned short VL, bool rhs,
	     typename LIt, typename RIt, typename Collector>
    static
    std::pair<LIt,RIt>
    intersect_task_vl( LIt lb, LIt le, RIt rb, RIt re, Collector & out ) {
	static_assert( so != so_intersect_xlat,
		       "intersect-with-translate not supported in merge-based"
		       " intersection" );

	using tr = vector_type_traits_vl<T,VL>;
	using type = typename tr::type;
	using mask_type = typename tr::mask_type;

	if( out.terminated() )
	    return std::make_pair( lb, rb );

	while( lb+VL <= le && rb+VL <= re ) {
	    type vl = tr::loadu( lb );

	    mask_type ma = tr::intersect( vl, rb );

	    type lf = tr::set1( lb[VL-1] );
	    type rf = tr::set1( rb[VL-1] );

	    mask_type ladv = tr::cmple( vl, rf, target::mt_mask() );
	    mask_type radv = tr::cmple( tr::loadu( rb ), lf, target::mt_mask() );
	    size_t la = rt_ilog2( ((uint64_t)ladv) + 1 );
	    size_t ra = rt_ilog2( ((uint64_t)radv) + 1 );

	    lb += la;
	    rb += ra;

	    if( !out.template multi_record<rhs,T,VL>( lb-la, vl, ma, la, ra ) )
		break;
	}

	if( lb+VL == le && rb+VL == re )
	    out.template remainder<rhs>( le - lb, re - rb );
	
	return std::make_pair( lb, rb );
    }

    template<set_operation so, bool rhs,
	     typename LSet, typename RSet, typename Collector>
    static
    void
    intersect_task( LSet && lset, RSet && rset, Collector & out ) {
	using T = typename std::decay_t<LSet>::type;
	
	auto lb = lset.begin();
	auto le = lset.end();
	auto rb = rset.begin();
	auto re = rset.end();
#if defined( __AVX512F__ )
	std::tie( lb, rb ) =
	    intersect_task_vl<so,T,64/sizeof(T),rhs>( lb, le, rb, re, out );
#elif defined( __AVX2__ )
	std::tie( lb, rb ) =
	    intersect_task_vl<so,T,32/sizeof(T),rhs>( lb, le, rb, re, out );
#endif
	merge_scalar::intersect_task<so,rhs>( lb, le, rb, re, out );
    }
    
    template<typename LSet, typename RSet, typename Task>
    static
    auto
    apply( LSet && lset, RSet && rset, Task & task ) {
	auto out = task.template create_collector<merge_vector>(
	    lset, rset );
	intersect_task<Task::operation,true>( lset, rset, out );
	return out.return_value();
    }

private:
    template<bool send_lhs_ptr,
	     unsigned VL, bool store, typename T, typename Inc>
    static
    void
    detail_intersect( const T *& lb, const T * le, const T *& rb, const T * re,
	       Inc & out ) {
	using tr = vector_type_traits_vl<T,VL>;
	using type = typename tr::type;
	using mask_type = typename tr::mask_type;

	while( lb+VL <= le && rb+VL <= re ) {
	    type vl = tr::loadu( lb );

	    mask_type ma = tr::intersect( vl, rb );

	    if constexpr ( send_lhs_ptr )
		out.template push_back<VL>( ma, vl, lb, rb );
	    else {
		if constexpr ( store )
		    tr::cstoreu( out, ma, vl );
		out += _popcnt32( ma );
	    }

	    type lf = tr::set1( lb[VL-1] );
	    type rf = tr::set1( rb[VL-1] );

	    mask_type ladv = tr::cmple( vl, rf, target::mt_mask() );
	    mask_type radv = tr::cmple( tr::loadu( rb ), lf, target::mt_mask() );

	    lb += rt_ilog2( ((uint64_t)ladv) + 1 );
	    rb += rt_ilog2( ((uint64_t)radv) + 1 );
	}
    }

    template<unsigned VL, typename T>
    static
    bool
    detail_intersect_size_gt_val(
	const T *& lb, const T *& le, const T *& rb, const T *& re,
	size_t exceed, size_t & sz ) {
	using tr = vector_type_traits_vl<T,VL>;
	using type = typename tr::type;
	using mask_type = typename tr::mask_type;

	size_t ld = std::distance( lb, le );
	size_t rd = std::distance( rb, re );
	if( ld > rd ) {
	    std::swap( lb, rb );
	    std::swap( le, re );
	    std::swap( ld, rd );
	}

	size_t d = ld; // equals std::min( ld, rd );

	if( d < exceed )
	    return false;

	std::make_signed_t<size_t> options = d - exceed;

	while( lb+VL <= le && rb+VL <= re ) {
	    type vl = tr::loadu( lb );

	    mask_type ma = tr::intersect( vl, rb );

	    type lf = tr::set1( lb[VL-1] );
	    type rf = tr::set1( rb[VL-1] );

	    mask_type ladv = tr::cmple( vl, rf, target::mt_mask() );
	    mask_type radv = tr::cmple( tr::loadu( rb ), lf, target::mt_mask() );

	    size_t la = rt_ilog2( ((uint64_t)ladv) + 1 );
	    // options += VL - la; // whichever elements not matched, put back options
	    lb += la;
	    rb += rt_ilog2( ((uint64_t)radv) + 1 );

	    options -= la - _popcnt32( ma );
	    if( options < 0 )
		return false;
	}

	sz = options + exceed - ( le - lb );
	return true;
    }

public:
    template<typename T>
    static
    T *
    intersect( const T* lb, const T* le, const T* rb, const T* re, T * out ) {
	return intersect<false>( lb, le, rb, re, out );
    }
    
    template<bool send_lhs_ptr, typename T, typename Out>
    static
    Out
    intersect( const T* lb, const T* le, const T* rb, const T* re, Out out ) {
#if defined( __AVX512F__ )
	detail_intersect<send_lhs_ptr,64/sizeof(T),true>( lb, le, rb, re, out );
#elif defined( __AVX2__ )
	detail_intersect<send_lhs_ptr,32/sizeof(T),true>( lb, le, rb, re, out );
#endif
	out = graptor::merge_scalar::template intersect<send_lhs_ptr>(
	    lb, le, rb, re, out );

	return out;
    }

    template<typename T>
    static
    size_t
    intersect_size( const T* lb, const T* le, const T* rb, const T* re ) {
	size_t sz = 0;
#if __AVX512F__
	detail_intersect<false,64/sizeof(T),false>( lb, le, rb, re, sz );
#endif
#if __AVX2__
	detail_intersect<false,32/sizeof(T),false>( lb, le, rb, re, sz );
#endif
	sz += graptor::merge_scalar::intersect_size( lb, le, rb, re );

	return sz;
    }

    template<typename T>
    static
    size_t
    intersect_size_gt_val( const T* lb, const T* le, const T* rb, const T* re,
			   size_t exceed ) {
	size_t sz0 = 0, sz1 = 0, sz2;

#if __AVX512F__
	if( !detail_intersect_size_gt_val<64/sizeof(T)>(
		lb, le, rb, re, exceed, sz0 ) )
	    return 0;
	if( sz0 > exceed )
	    return sz0 + merge_scalar::intersect_size( lb, le, rb, re );
	exceed -= sz0;
#endif
#if __AVX2__
	if( !detail_intersect_size_gt_val<32/sizeof(T)>(
		lb, le, rb, re, exceed, sz1 ) )
	    return 0;
	if( sz1 > exceed )
	    return sz0 + sz1 + merge_scalar::intersect_size( lb, le, rb, re );
	exceed -= sz1;
#endif
	sz2 = merge_scalar::intersect_size_gt_val( lb, le, rb, re, exceed );
	if( !sz2 )
	    return 0;

	return sz0 + sz1 + sz2;
    }
};

template<typename Underlying>
struct merge_partitioned {

    static constexpr bool uses_hash = false;

    template<typename T, typename I>
    static
    void
    prestudy( const T* lb, const T* le, T max_range, size_t levels, I* idx ) {
	assert( lb == le || *(le-1) < max_range );
	const T step = max_range >> levels;
	T cur = step;
	size_t off = 1;
	idx[0] = 0;

	for( const T * l=lb; l != le; ++l ) {
	    while( cur <= *l ) {
		idx[off++] = l - lb;
		cur += step;
	    }
	}
	while( off < (size_t(1)<<levels) )
	    idx[off++] = le - lb;

	idx[size_t(1)<<levels] = le - lb; // unconditionally overwrite
    }

    template<typename T, typename HT, typename I>
    static
    T *
    intersect( const T* lb, const T* le,
	       const T* rb, const T* re,
	       const HT & htable,
	       size_t levels,
	       size_t lo, size_t hi,
	       const I* lidx, // prestudy of lhs
	       const I* ridx, // prestudy of rhs
	       T * out ) {
       return intersect<false>( lb, le, rb, re, htable, levels,
				lo, hi, lidx, ridx, out );
   }

    template<bool send_lhs_ptr, typename T, typename HT, typename I, typename O>
    static
    O
    intersect( const T* lb, const T* le,
	       const T* rb, const T* re,
	       const HT & htable,
	       size_t levels,
	       size_t lo, size_t hi,
	       const I* lidx, // prestudy of lhs
	       const I* ridx, // prestudy of rhs
	       O out ) {
	if( lidx[lo] == lidx[hi] || ridx[lo] == ridx[hi] )
	    return out;
	else if( levels == 0 ) {
	    if constexpr ( std::is_same_v<Underlying,hash_scalar>
			   || std::is_same_v<Underlying,hash_vector> )
		return Underlying::template intersect<send_lhs_ptr>(
		    &lb[lidx[lo]], &lb[lidx[hi]], htable, out );
	    else
		return Underlying::template intersect<send_lhs_ptr>(
		    &lb[lidx[lo]], &lb[lidx[hi]],
		    &rb[ridx[lo]], &rb[ridx[hi]],
		    out );
	} else {
	    size_t mid = ( lo + hi ) / 2;
	    auto out1 = intersect<send_lhs_ptr>(
		lb, le, rb, re, htable, levels-1,
		lo, mid, lidx, ridx, out );
	    return intersect<send_lhs_ptr>(
		lb, le, rb, re, htable, levels-1,
		mid, hi, lidx, ridx, out1 );
	}
    }

    template<typename T, typename HT, typename I>
    static
    size_t
    intersect_size( const T* lb, const T* le,
		    const T* rb, const T* re,
		    const HT & htable,
		    size_t levels,
		    size_t lo, size_t hi,
		    const I* lidx, // prestudy of lhs
		    const I* ridx // prestudy of rhs
	) {
	if( lidx[lo] == lidx[hi] || ridx[lo] == ridx[hi] )
	    return 0;
	else if( levels == 0 ) {
	    if constexpr ( std::is_same_v<Underlying,hash_vector> )
		return Underlying::intersect_size(
		    &lb[lidx[lo]], &lb[lidx[hi]], htable );
	    else
		return Underlying::intersect_size(
		    &lb[lidx[lo]], &lb[lidx[hi]],
		    &rb[ridx[lo]], &rb[ridx[hi]] );
	} else {
	    size_t mid = ( lo + hi ) / 2;
	    size_t sz0 = intersect_size( lb, le, rb, re, htable, levels-1,
					 lo, mid, lidx, ridx );
	    size_t sz1 = intersect_size( lb, le, rb, re, htable, levels-1,
					 mid, hi, lidx, ridx );
	    return sz0 + sz1;
	}
    }

    template<typename T, typename HT, typename I>
    static
    size_t
    intersect_size_gt_val( const T* lb, const T* le,
			   const T* rb, const T* re,
			   const HT & htable,
			   size_t levels,
			   size_t lo, size_t hi,
			   const I* lidx, // prestudy of lhs
			   const I* ridx, // prestudy of rhs
			   size_t exceed ) {
	I ldiff = lidx[hi] - lidx[lo];
	I rdiff = ridx[hi] - ridx[lo];
	if( exceed > ldiff || ldiff == 0 || exceed > rdiff || rdiff == 0 )
	    return 0;
	else if( levels == 0 ) {
	    assert( lo+1 == hi );
	    if constexpr ( std::is_same_v<Underlying,hash_vector> )
		return Underlying::intersect_size_gt_val(
		    &lb[lidx[lo]], &lb[lidx[hi]], htable, exceed );
	    else
		return Underlying::intersect_size_gt_val(
		    &lb[lidx[lo]], &lb[lidx[hi]], &rb[ridx[lo]], &rb[ridx[hi]],
		    exceed );
	} else {
	    // TODO: a further optimisation would be to ponder which of the
	    // two branches to execute first, as it may affect the
	    // effectiveness of pruning (early return 0).
	    //
	    // The first branch must find at least exceed elements minus
	    // the number of elements that can be found from the second
	    // branch, which is the minimum of the two list sizes.
	    size_t mid = ( lo + hi ) / 2;
	    size_t ld2 = lidx[hi] - lidx[mid];
	    size_t rd2 = ridx[hi] - ridx[mid];
	    size_t d2 = std::min( ld2, rd2 );
	    size_t sz0, sz1;
	    // If the second branch lists are longer than the exceed
	    // threshold, then we can find everything in the second list
	    // and the threshold for the first branch is 0. In that case,
	    // be faster by avoiding exceed checks and simply determine the
	    // intersection size.
	    if( exceed > d2 ) {
		sz0 = intersect_size_gt_val(
		    lb, le, rb, re, htable, levels-1,
		    lo, mid, lidx, ridx, exceed-d2 );
		// If this branch fails to find at least exceed-d2 elements,
		// then it will be impossible to find exceed elements in
		// total as the second branch can only find d2 elements.
		if( sz0 < exceed-d2 )
		    return 0;
	    } else
		sz0 = intersect_size(
		    lb, le, rb, re, htable, levels-1,
		    lo, mid, lidx, ridx );
	    // The second branch needs to exceed the threshold, minus
	    // the elements found in the first branch. If the first branch
	    // has all the required elements, just complete the count correctly.
	    if( sz0 > exceed )
		sz1 = intersect_size( lb, le, rb, re, htable, levels-1,
				      mid, hi, lidx, ridx );
	    else
		sz1 = intersect_size_gt_val(
		    lb, le, rb, re, htable, levels-1,
		    mid, hi, lidx, ridx, exceed-sz0 );
	    return sz0 + sz1;
	}
    }
};

struct something {
    template<set_operation so, bool rhs,
	     typename LIt, typename RIt, typename Collector>
    static
    void
    intersect_task( LIt lb, LIt le, RIt rb, RIt re, Collector & out ) {
	static_assert( so != so_intersect_xlat,
		       "intersect-with-translate not supported in merge-based"
		       " intersection" );

	if( out.terminated() )
	    return;

	if( std::distance( lb, le ) <= 4 || std::distance( rb, re ) <= 4 ) {
	    merge_scalar_jump::intersect_task( lb, le, rb, re, out );
	    return;
	}

	LIt lm = std::next( lb, std::distance( lb, le ) / 2 );
	RIt rm = std::next( rb, std::distance( rb, re ) / 2 );

	// Splits three-ways
	// 1. intersect lb ... lm with rb ... rm
	intersect_task( lb, lm, rb, rm, out );

	// 2. intersect lm ... le with rm ... re
	intersect_task( lm, le, rm, re, out );
	
	if( *rm < *lm ) {
	    // 3. intersect equiv(rm) ... lm with rm ... equiv(lm)
	    intersect_task( lb, lm, rm, re, out );
	} else {
	    // 3. intersect lb ... equiv(rm) with equiv(lm) ... re
	    intersect_task( lm, le, rb, rm, out );
	}
    }

    template<set_operation so, bool rhs,
	     typename LSet, typename RSet, typename Collector>
    static
    void
    intersect_task( LSet && lset, RSet && rset, Collector & out ) {
	auto lb = lset.begin();
	auto le = lset.end();
	auto rb = rset.begin();
	auto re = rset.end();
	intersect_task<so,rhs>( lb, le, rb, re, out );
    }
    
    template<typename LSet, typename RSet, typename Task>
    static
    auto
    apply( LSet && lset, RSet && rset, Task & task ) {
	auto out = task.template create_collector<something>(
	    lset, rset );
	intersect_task<Task::operation,true>( lset, rset, out );
	return out.return_value();
    }

};

struct MC_intersect {
    template<typename LSet, typename RSet, typename Task>
    static
    auto
    apply( LSet && lset, RSet && rset, Task & task ) {
	// Corner case
	if( lset.size() == 0 || rset.size() == 0 )
	    return task.return_value_empty_set();

#if MC_INTERSECTION_ALGORITHM == 1
	if constexpr ( graptor::is_fast_array_v<std::decay_t<LSet>>
		       || graptor::is_fast_array_v<std::decay_t<RSet>> )
	    return hash_vector::apply( lset, rset, task );
	else if constexpr ( is_hash_set_v<std::decay_t<LSet>>
			      || is_hash_set_v<std::decay_t<RSet>> )
	    return hash_scalar::apply( lset, rset, task );
	else
	    return merge_vector_jump::apply( lset, rset, task );
#else
    if constexpr ( is_hash_set_v<std::decay_t<LSet>>
		   || is_hash_set_v<std::decay_t<RSet>> ) {
	if( lset.has_hash_set() || rset.has_hash_set() )
	    return hash_vector::apply( lset, rset, task );
	else
	    return merge_vector_jump::apply( lset, rset, task );
    } else
	return merge_vector_jump::apply( lset, rset, task );
#endif
    }
};

#if 0
struct adaptive_intersect {
    template<set_operation so,
	     typename LSet, typename RSet, typename Collector>
    static
    auto
    apply( LSet && lset, RSet && rset, Collector & out ) {
	// Corner case
	if( lset.size() == 0 || rset.size() == 0 ) {
	    out.template remainder<true>( lset.size(), rset.size() );
	    return;
	}

#if INTERSECTION_ALGORITHM >= 6
	// Trim ranges
	auto llo = *lset.begin();
	auto lhi = *std::prev( lset.end() );
	auto rlo = *rset.begin();
	auto rhi = *std::prev( rset.end() );
#endif

#if INTERSECTION_ALGORITHM == 0
	return merge_scalar::template apply<so>( lset, rset, out );
#elif INTERSECTION_ALGORITHM == 1
	return merge_vector::template apply<so>( lset, rset, out );
#elif INTERSECTION_ALGORITHM == 2
	return merge_scalar_jump::template apply<so>( lset, rset, out );
#elif INTERSECTION_ALGORITHM == 3
	return merge_vector_jump::template apply<so>( lset, rset, out );
#elif INTERSECTION_ALGORITHM == 4
	if constexpr ( is_hash_set_v<std::decay_t<LSet>>
		       || is_hash_set_v<std::decay_t<RSet>> )
	    return hash_scalar::template apply<so>( lset, rset, out );
	else
	    return merge_vector_jump::template apply<so>( lset, rset, out );
#elif INTERSECTION_ALGORITHM == 5
	if constexpr ( is_multi_hash_set_v<std::decay_t<LSet>>
		       || is_multi_hash_set_v<std::decay_t<RSet>> )
	    return hash_vector::template apply<so>( lset, rset, out );
	else
	    return merge_vector_jump::template apply<so>( lset, rset, out );
#else
	auto range = std::max( lhi, rhi ) - std::min( llo, rlo );

	int lcat = rt_ilog2( lset.size()/2 );
	int rcat = rt_ilog2( rset.size()/2 );

	bool do_hash = true;

	if constexpr ( so == so_intersect || so == so_intersect_xlat ) {
	    // do_hash = ( lcat + rcat >= 5 && std::abs( lcat - rcat ) > 1 ); // rule3
	    // do_hash = std::abs( lcat - rcat ) > 1; // rule2
	    // do_hash = std::abs( lcat - rcat ) > 0; // rule5
	    // do_hash = ( lcat != rcat || std::max( lcat, rcat ) >= 5 ); // rule6
	} else if constexpr ( so == so_intersect_size ) {
	    // do_hash = ( lcat != rcat );
	    // do_hash = ( lcat + rcat >= 5 && std::abs( lcat - rcat ) > 1 ); // rule3
	} else if constexpr ( so == so_intersect_size_gt_val ) {
#if INTERSECTION_ALGORITHM == 6
	    do_hash = std::abs( lcat - rcat ) > 1; // rule2
#elif INTERSECTION_ALGORITHM == 7
	    do_hash = std::min(lset.size(), rset.size()) > range/8; // rule8
#elif INTERSECTION_ALGORITHM == 8
	    do_hash = std::min(lset.size(), rset.size()) > range/8
		|| std::abs( lcat - rcat ) > 1; // rule2.8
#endif
	    // do_hash = ( lcat + rcat >= 5 || std::abs( lcat - rcat ) > 1 ); // rule1
	    // do_hash = ( lcat != rcat ); // ne
	    // do_hash = ( lcat + rcat >= 5 && std::abs( lcat - rcat ) > 1 ); // rule3
	    // do_hash = ( lcat + rcat >= 4 && std::abs( lcat - rcat ) > 1 ); // rule4
	    // do_hash = std::min(lset.size(), rset.size()) > range/2; // rule6
	    // do_hash = std::min(lset.size(), rset.size()) > range/4; // rule7
	    // do_hash = std::min(lset.size(), rset.size()) > range/8; // rule8
	}

	if( do_hash ) {
	    // Assume trimmed sets have same type as original sets
	    if constexpr ( is_hash_set_v<std::decay_t<LSet>>
			   || is_hash_set_v<std::decay_t<RSet>> )
		return hash_vector::template apply<so>( lset, rset, out );
	    else
		return merge_vector_jump::template apply<so>( lset, rset, out );
	}
	else
	    return merge_vector_jump::template apply<so>( lset, rset, out );
#endif
    }
};
#endif

#if 0
/*! The adaptive_intersect_filter class optimises the intersection algorithm
 *  for the filtering step. Filtering is applied on a sequential (non-hashed)
 *  list and an adjacency list. There is thus only one option for hashing.
 */
struct adaptive_intersect_filter {
    template<set_operation so,
	     typename LSet, typename RSet, typename Collector>
    static
    auto
    apply( LSet && lset, RSet && rset, Collector & out ) {
	// Corner case
	if( lset.size() == 0 || rset.size() == 0 ) {
	    out.template remainder<true>( lset.size(), rset.size() );
	    return;
	}

	if constexpr ( so != so_intersect_size_gt_val )
	    // This class is designed strictly for so_intersect_size_gt_val
	    return adaptive_intersect::template apply<so>( lset, rset, out );
	else {
	    static_assert( !is_hash_set_v<std::decay_t<LSet>>,
			   "expect LHS is a sequential list" );
	    static_assert( is_hash_set_v<std::decay_t<RSet>>,
			   "expect RHS is hashed" );
	    static_assert( is_multi_hash_set_v<std::decay_t<RSet>>,
			   "expect RHS is hashed and vectorisable" );

	    // Trim ranges
	    auto llo = *lset.begin();
	    auto lhi = *std::prev( lset.end() );
	    auto rlo = *rset.begin();
	    auto rhi = *std::prev( rset.end() );

	    auto tlset = lset.trim_range( rlo, rhi );
	    auto trset = rset.trim_range( llo, lhi );

	    out.swap( tlset, trset );

	    int lcat = rt_ilog2( tlset.size()/2 );
	    int rcat = rt_ilog2( trset.size()/2 );

	    llo = *tlset.begin();
	    lhi = *std::prev( tlset.end() );
	    rlo = *trset.begin();
	    rhi = *std::prev( trset.end() );

	    size_t lf = 10.0f * float(tlset.size()) / float(lhi - llo);
	    size_t rf = 10.0f * float(trset.size()) / float(rhi - rlo);

	    // If set sizes about equal, merge could work well.
	    // If LHS size much smaller than RHS size, hash would work well.
	    // If RHS has high density, chances of having good vector occupation
	    // are high.
	    if( lcat < rcat ) // || rf <= 2 )
		return hash_vector::template apply<so>( tlset, trset, out );
	    // else if( rcat+1 < lcat ) // better hash_vector
	    else if( rf == 0 )
		return merge_vector_jump::template apply<so>( tlset, trset, out );
	    else if( lf == 0 )
		return merge_scalar_jump::template apply<so>( tlset, trset, out );
	    else
		return merge_vector::template apply<so>( tlset, trset, out );
	}
    }
};
#endif


} // namespace graptor

#endif // GRAPTOR_CONTAINER_INTERSECT_H
