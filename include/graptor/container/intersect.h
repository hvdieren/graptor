// -*- c++ -*-
#ifndef GRAPTOR_CONTAINER_INTERSECT_H
#define GRAPTOR_CONTAINER_INTERSECT_H

#include <iterator>
#include <type_traits>
#include <immintrin.h>

#include "graptor/target/vector.h"

/*======================================================================*
 *======================================================================*/

namespace graptor {

struct merge_scalar {

    template<typename It, typename Ot>
    static
    Ot intersect( It lb, It le, It rb, It re, Ot o ) {
	It l = lb;
	It r = rb;

	while( l != le && r != re ) {
	    if( *l == *r ) {
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
	size_t sz = 0;
	size_t d = std::min( ld, rd );

	if( d < exceed )
	    return 0;

	size_t options = d - exceed;

	while( l != le && r != re ) {
	    if( *l == *r ) {
		++sz;
		++l;
		++r;
	    } else if( *l < *r ) {
		if( --options == 0 )
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

    template<typename It, typename T>
    static
    It jump( It b, It e, T ref ) {
	if( ref <= *b )
	    return b;

	// Search for the furthest position in bounds and below ref
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
	It l = lb;
	It r = rb;

	while( l != le && r != re ) {
	    if( *l == *r ) {
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

	if( d < exceed )
	    return 0;

	std::make_signed_t<size_t> options = d - exceed;

	while( lb != le ) {
	    VID v = *lb;
	    if( !htable.contains( v ) )
		if( --options == 0 )
		    return 0;
	    ++lb;
	}

	return options + exceed;
    }

};

struct hash_vector {

private:
    template<unsigned VL, bool store, typename T, typename HT, typename Ot>
    static
    const T *
    detail_intersect(
	const T * lb, const T * le, const HT & htable, Ot & out ) {
	using tr = vector_type_traits_vl<T,VL>;
	using type = typename tr::type;

	while( lb+VL <= le ) {
	    type v = tr::loadu( lb );
	    typename target::mask_type_traits<VL>::type m
		= htable.template multi_contains<T,VL>( v, target::mt_mask() );
	    if constexpr ( store )
		tr::cstoreu( out, m, v );
	    out += _popcnt32( m );
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
	    auto m = htable.template multi_contains<T,VL>( v, target::mt_mask() );
	    options -= VL - _popcnt32( m );
	    if( options <= 0 ) {
		lb = l;
		return false;
	    }
	    l += VL;
	}

	lb = l;
	sz = options + exceed - ( le - l );
	return true;
    }

public:
    template<typename T, typename HT>
    static
    T * intersect( const T * lb, const T * le, const HT & htable, T * out ) {
#if __AVX512F__
	lb = detail_intersect<64/sizeof(T),true>( lb, le, htable, out );
#endif
#if __AVX2__
	lb = detail_intersect<32/sizeof(T),true>( lb, le, htable, out );
#endif
	out = graptor::hash_scalar::intersect( lb, le, htable, out );

	return out;
    }

    template<typename T, typename HT>
    static
    size_t intersect_size( const T * lb, const T * le, const HT & htable ) {
	size_t sz = 0;
#if __AVX512F__
	lb = detail_intersect<64/sizeof(T),false>( lb, le, htable, sz );
#endif
#if __AVX2__
	lb = detail_intersect<32/sizeof(T),false>( lb, le, htable, sz );
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

};

struct merge_vector {

private:
    template<unsigned VL, bool store, typename T, typename Inc>
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

	    if constexpr ( store )
		tr::cstoreu( out, ma, vl );
	    out += _popcnt32( ma );

	    type lf = tr::set1( lb[VL-1] );
	    type rf = tr::set1( rb[VL-1] );

	    mask_type ladv = tr::cmple( vl, rf, target::mt_mask() );
	    mask_type radv = tr::cmple( tr::loadu( rb ), lf, target::mt_mask() );

	    lb += ilog2( ((uint64_t)ladv) + 1 );
	    rb += ilog2( ((uint64_t)radv) + 1 );
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

	size_t d = std::min( ld, rd );

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

	    size_t la = ilog2( ((uint64_t)ladv) + 1 );
	    // options += VL - la; // whichever elements not matched, put back options
	    lb += la;
	    rb += ilog2( ((uint64_t)radv) + 1 );

	    options -= la - _popcnt32( ma );
	    if( options <= 0 )
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
#if __AVX512F__
	detail_intersect<64/sizeof(T),true>( lb, le, rb, re, out );
#endif
#if __AVX2__
	detail_intersect<32/sizeof(T),true>( lb, le, rb, re, out );
#endif
	out = graptor::merge_scalar::intersect( lb, le, rb, re, out );

	return out;
    }

    template<typename T>
    static
    size_t
    intersect_size( const T* lb, const T* le, const T* rb, const T* re ) {
	size_t sz = 0;
#if __AVX512F__
	detail_intersect<64/sizeof(T),false>( lb, le, rb, re, sz );
#endif
#if __AVX2__
	detail_intersect<32/sizeof(T),false>( lb, le, rb, re, sz );
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
	    while( cur <= *l ) { // TODO: <= ??
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
	if( lidx[lo] == lidx[hi] || ridx[lo] == ridx[hi] )
	    return out;
	else if( levels == 0 ) {
	    if constexpr ( std::is_same_v<Underlying,hash_vector> )
		return Underlying::intersect( &lb[lidx[lo]], &lb[lidx[hi]],
					      htable,
					      out );
	    else
		return Underlying::intersect( &lb[lidx[lo]], &lb[lidx[hi]],
					      &rb[ridx[lo]], &rb[ridx[hi]],
					      out );
	} else {
	    size_t mid = ( lo + hi ) / 2;
	    T * out1 = intersect( lb, le, rb, re, htable, levels-1,
				  lo, mid, lidx, ridx, out );
	    return intersect( lb, le, rb, re, htable, levels-1,
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
