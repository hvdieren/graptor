// -*- c++ -*-
/*!=====================================================================*
 * \file graptor/container/intersect.h
 * \brief Various intersection algorithms
 *======================================================================*/

#ifndef GRAPTOR_CONTAINER_INTERSECT_H
#define GRAPTOR_CONTAINER_INTERSECT_H

#include <iterator>
#include <type_traits>
#include <immintrin.h>

#include "graptor/target/vector.h"
#include "graptor/container/dual_set.h"

namespace graptor {

/*! Enumeration of set operation types
 */
enum set_operation {
    so_intersect = 0,			//!< intersection - list of elements
    so_intersect_xlat = 1,		//!< intersection - list + translate
    so_intersect_size = 2,		//!< intersection size
    so_intersect_size_exceed = 3,	//!< intersection size > or abort
    so_intersect_size_ge = 4,		//!< intersection size >=
    so_N = 5 	 	 	 	//!< number of set operations
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
	    vector_type_traits_vl<typename C::type,8>::mask_traits::setzero() );
    };
};

/*! Variable indicating that \sa is_multi_collector trait is met
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
    // \return Boolean that is false when intersection should be aborted.
    template<bool rhs, typename U, unsigned short VL, typename M>
    bool
    multi_record( const U * p,
		  typename vector_type_traits_vl<U,VL>::type value,
		  M mask ) {
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

template<typename T>
struct intersection_size {
    //! The type of elements in this set
    using type = T;

    intersection_size() : m_size( 0 ) { }

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

    template<bool rhs, typename U, unsigned short VL>
    bool
    multi_record( const U * p,
		  typename vector_type_traits_vl<U,VL>::type index,
		  typename vector_type_traits_vl<U,VL>::mask_type mask ) {
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

template<typename T>
struct intersection_size_exceed {
    //! The type of elements in this set
    using type = T;

    intersection_size_exceed( size_t min_arg_size, size_t exceed )
	: m_options( min_arg_size - exceed ),
	  m_exceed( exceed ),
	  m_terminated( min_arg_size <= exceed ) { }

    // This code currently only works with rhs == true
    // When swapping lhs/rhs, we need to change the initial options.
    template<typename LSet, typename RSet>
    intersection_size_exceed( LSet && lset, RSet && rset, size_t exceed )
	: intersection_size_exceed( lset.size(), // iterated set!
				    // std::min( lset.size(), rset.size() ),
				    exceed ) { }

    template<bool rhs>
    bool record( const type * l, const type * r, bool ins ) {
	if( !ins ) {
	    if( --m_options <= 0 ) [[unlikely]] {
		m_terminated = true;
		return false;
	    }
	}
	return true;
    }

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

    template<bool rhs, typename U, unsigned short VL>
    bool
    multi_record( const U * p,
		  typename vector_type_traits_vl<U,VL>::type index,
		  typename vector_type_traits_vl<U,VL>::mask_type mask ) {
	using tr = vector_type_traits_vl<U,VL>;
	size_t absent = VL - tr::mask_traits::popcnt( mask );
	m_options -= absent;
	if( m_options <= 0 ) {
	    m_terminated = true;
	    return false;
	} else
	    return true;
    }

    size_t return_value() const {
	return m_terminated ? 0 : m_options + m_exceed;
    }

    bool terminated() const { return m_terminated; }

private:
    std::make_signed_t<size_t> m_options;
    size_t m_exceed;
    bool m_terminated;
};

template<typename S>
struct is_intersection_size_exceed : public std::false_type { };

template<typename T>
struct is_intersection_size_exceed<intersection_size_exceed<T>>
    : public std::true_type { };

template<typename S>
constexpr bool is_intersection_size_exceed_v =
    is_intersection_size_exceed<S>::value;

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
	    so_traits::template apply<so_intersect>( lset, rset, cout );
	    return cout.return_value();
	} else {
	    so_traits::template apply<so_intersect>( lset, rset, out );
	    return out;
	}
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
	    so_traits::template apply<so_intersect_xlat>( lset, rset, cout );
	    return cout.return_value();
	} else {
	    so_traits::template apply<so_intersect_xlat>( lset, rset, out );
	    return out;
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
	
	intersection_size<typename std::decay_t<LSet>::type> out;
	so_traits::template apply<so_intersect_size>( lset, rset, out );
	return out.return_value();
    }

    template<typename LSet, typename RSet>
    static
    size_t
    intersect_size_exceed_ds( LSet && lset, RSet && rset, size_t exceed ) {
	static_assert( std::is_same_v<
		       typename std::decay_t<LSet>::type,
		       typename std::decay_t<RSet>::type>,
		       "Sets must contain elements of the same type" );
	
	intersection_size_exceed<typename std::decay_t<LSet>::type>
	    out( lset, rset, exceed );
	so_traits::template apply<so_intersect_size_exceed>( lset, rset, out );
	return out.return_value();
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
	    // translation not supported
	    if( !out.template record<rhs>( lb, rb, *lb == *rb ) )
		break;
	    if( *lb == *rb ) {
		++lb;
		++rb;
	    } else if( *lb < *rb )
		++lb;
	    else
		++rb;
	}

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
    
    template<set_operation so,
	     typename LSet, typename RSet, typename Collector>
    static
    auto
    apply( LSet && lset, RSet && rset, Collector & out ) {
	return intersect_task<so,true>( lset, rset, out );
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
    size_t intersect_size_exceed( It lb, It le, It rb, It re, size_t exceed ) {
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

struct merge_jump {

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
		lb = jump( lb, le, *rb );
	    } else {
		++rb;
		rb = jump( rb, re, *lb );
	    }
	}

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
    
    template<set_operation so,
	     typename LSet, typename RSet, typename Collector>
    static
    auto
    apply( LSet && lset, RSet && rset, Collector & out ) {
	return intersect_task<so,true>( lset, rset, out );
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
    size_t intersect_size_exceed( It lb, It le, It rb, It re, size_t exceed ) {
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
	auto lb = lset.begin();
	auto le = lset.end();
	intersect_task<so,rhs>( lb, le, rset, out );
    }

    template<set_operation so,
	     typename LSet, typename RSet, typename Collector>
    static
    auto
    apply( LSet && lset, RSet && rset, Collector & out ) {
	// TODO: tighten begin/end of each set using jumping, from start
	//       and also from the end. This should give a tighter bound
	//       on the number of sequential values to lookup in the hash set
	//       and reduce work overall.
	static_assert( is_hash_set_v<std::decay_t<LSet>>
		       || is_hash_set_v<std::decay_t<RSet>>,
		       "at least one of arguments should be hash set" );
	if constexpr ( is_hash_set_v<std::decay_t<LSet>>
		       && is_hash_set_v<std::decay_t<RSet>>
		       && !is_intersection_size_exceed_v<Collector> ) {
	    if( lset.size() < rset.size() )
		return intersect_task<so,true>( lset, rset, out );
	    else
		return intersect_task<so,false>( rset, lset, out );
	} else if constexpr ( !is_hash_set_v<std::decay_t<LSet>> ) {
	    return intersect_task<so,true>( lset, rset, out );
	} else {
	    static_assert( !is_intersection_size_exceed_v<Collector>,
			   "RHS must be hashable in intersection_size_exceed" );
	    return intersect_task<so,false>( rset, lset, out );
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
    size_t intersect_size_exceed( It lb, It le, const HT & htable, size_t exceed ) {
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
    size_t intersect_size_exceed( It lb, It le, const HT & htable, size_t exceed ) {
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
    detail_intersect_size_exceed(
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
    intersect_size_exceed(
	const T * lb, const T * le, const HT & htable, size_t exceed ) {
	size_t sz0 = 0, sz1 = 0, sz2;

#if __AVX512F__
	if( !detail_intersect_size_exceed<64/sizeof(T)>(
		lb, le, htable, exceed, sz0 ) )
	    return 0;
	if( sz0 > exceed )
	    return sz0 + hash_scalar::intersect_size( lb, le, htable );
	exceed -= sz0;
#endif
#if __AVX2__
	if( !detail_intersect_size_exceed<32/sizeof(T)>(
		lb, le, htable, exceed, sz1 ) )
	    return 0;
	if( sz1 > exceed )
	    return sz0 + sz1 + hash_scalar::intersect_size( lb, le, htable );
	exceed -= sz1;
#endif
	sz2 = hash_scalar::intersect_size_exceed( lb, le, htable, exceed );
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
		if( !out.template multi_record<rhs,T,VL>( lb, m.second, m.first ) )
		    break;
	    } else {
		// Check present in hash set. Returns a mask.
		auto m = rset.template multi_contains<T,VL>(
		    v, target::mt_mask() );

		// Record / count common values
		if( !out.template multi_record<rhs,T,VL>( lb, v, m ) )
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

	auto lb = lset.begin();
	auto le = lset.end();

#if __AVX512F__
	lb = intersect_task_vl<so,T,64/sizeof(T),rhs>( lb, le, rset, out );
#endif
#if __AVX2__
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
		       && is_multi_hash_set_v<std::decay_t<RSet>>
		       && !is_intersection_size_exceed_v<Collector> ) {
	    if( lset.size() < rset.size() )
		return intersect_task<so,true>( lset, rset, out );
	    else
		return intersect_task<so,false>( rset, lset, out );
	} else if constexpr ( !is_hash_set_v<std::decay_t<LSet>> ) {
	    return intersect_task<so,true>( lset, rset, out );
	} else {
	    static_assert( !is_intersection_size_exceed_v<Collector>,
			   "RHS must be hashable in intersection_size_exceed" );
	    return intersect_task<so,false>( rset, lset, out );
	}
    }
    
};

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

	    if( !out.template multi_record<rhs,T,VL>( lb, vl, ma ) )
		break;

	    type lf = tr::set1( lb[VL-1] );
	    type rf = tr::set1( rb[VL-1] );

	    mask_type ladv = tr::cmple( vl, rf, target::mt_mask() );
	    mask_type radv = tr::cmple( tr::loadu( rb ), lf, target::mt_mask() );

	    lb += rt_ilog2( ((uint64_t)ladv) + 1 );
	    rb += rt_ilog2( ((uint64_t)radv) + 1 );
	}

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
#if __AVX512F__
	std::tie( lb, rb ) =
	    intersect_task_vl<so,T,64/sizeof(T),rhs>( lb, le, rb, re, out );
#endif
#if __AVX2__
	std::tie( lb, rb ) =
	    intersect_task_vl<so,T,32/sizeof(T),rhs>( lb, le, rb, re, out );
#endif
	merge_scalar::intersect_task<so,rhs>( lb, le, rb, re, out );
    }
    
    template<set_operation so,
	     typename LSet, typename RSet, typename Collector>
    static
    auto
    apply( LSet && lset, RSet && rset, Collector & out ) {
	return intersect_task<so,true>( lset, rset, out );
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
    detail_intersect_size_exceed(
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
#if __AVX512F__
	detail_intersect<send_lhs_ptr,64/sizeof(T),true>( lb, le, rb, re, out );
#endif
#if __AVX2__
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
    intersect_size_exceed( const T* lb, const T* le, const T* rb, const T* re,
			   size_t exceed ) {
	size_t sz0 = 0, sz1 = 0, sz2;

#if __AVX512F__
	if( !detail_intersect_size_exceed<64/sizeof(T)>(
		lb, le, rb, re, exceed, sz0 ) )
	    return 0;
	if( sz0 > exceed )
	    return sz0 + merge_scalar::intersect_size( lb, le, rb, re );
	exceed -= sz0;
#endif
#if __AVX2__
	if( !detail_intersect_size_exceed<32/sizeof(T)>(
		lb, le, rb, re, exceed, sz1 ) )
	    return 0;
	if( sz1 > exceed )
	    return sz0 + sz1 + merge_scalar::intersect_size( lb, le, rb, re );
	exceed -= sz1;
#endif
	sz2 = merge_scalar::intersect_size_exceed( lb, le, rb, re, exceed );
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
    intersect_size_exceed( const T* lb, const T* le,
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
		return Underlying::intersect_size_exceed(
		    &lb[lidx[lo]], &lb[lidx[hi]], htable, exceed );
	    else
		return Underlying::intersect_size_exceed(
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
		sz0 = intersect_size_exceed(
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
		sz1 = intersect_size_exceed(
		    lb, le, rb, re, htable, levels-1,
		    mid, hi, lidx, ridx, exceed-sz0 );
	    return sz0 + sz1;
	}
    }
};

} // namespace graptor

#endif // GRAPTOR_CONTAINER_INTERSECT_H
