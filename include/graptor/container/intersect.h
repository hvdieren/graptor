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

namespace merge_scalar {

template<typename It, typename Ot>
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

} // namespace merge_scalar

namespace merge_jump {

template<typename It, typename T>
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
size_t intersect_size_exceed( It lb, It le, It rb, It re, size_t exceed ) {
    size_t ld = std::distance( lb, le );
    size_t rd = std::distance( rb, re );
    if( ld < rd ) {
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

} // namespace merge_jump

namespace hash_scalar {

template<typename It, typename HT, typename Ot>
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

} // namespace hash_scalar

namespace hash_vector {

namespace detail {

template<unsigned VL, bool store, typename T, typename HT, typename Ot>
const T *
intersect( const T * lb, const T * le, const HT & htable, Ot & out ) {
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
bool
intersect_size_exceed(
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

} // namespace detail

template<typename T, typename HT>
T * intersect( const T * lb, const T * le, const HT & htable, T * out ) {
#if __AVX512F__
    lb = detail::intersect<64/sizeof(T),true>( lb, le, htable, out );
#endif
#if __AVX2__
    lb = detail::intersect<32/sizeof(T),true>( lb, le, htable, out );
#endif
    out = graptor::hash_scalar::intersect( lb, le, htable, out );

    return out;
}

template<typename T, typename HT>
size_t intersect_size( const T * lb, const T * le, const HT & htable ) {
    size_t sz = 0;
#if __AVX512F__
    lb = detail::intersect<64/sizeof(T),false>( lb, le, htable, sz );
#endif
#if __AVX2__
    lb = detail::intersect<32/sizeof(T),false>( lb, le, htable, sz );
#endif
    sz += hash_scalar::intersect_size( lb, le, htable );

    return sz;
}

template<typename T, typename HT>
size_t
intersect_size_exceed(
    const T * lb, const T * le, const HT & htable, size_t exceed ) {
    size_t sz0 = 0, sz1 = 0, sz2;

#if __AVX512F__
    if( !detail::intersect_size_exceed<64/sizeof(T)>(
	    lb, le, htable, exceed, sz0 ) )
	return 0;
    exceed -= sz0;
#endif
#if __AVX2__
    if( !detail::intersect_size_exceed<32/sizeof(T)>(
	    lb, le, htable, exceed, sz1 ) )
	return 0;
    exceed -= sz1;
#endif
    sz2 = hash_scalar::intersect_size_exceed( lb, le, htable, exceed );
    if( !sz2 )
	return 0;

    return sz0 + sz1 + sz2;
}

} // namespace hash_vector

namespace merge_vector {

namespace detail {

template<unsigned VL, bool store, typename T, typename Inc>
void
intersect( const T *& lb, const T * le, const T *& rb, const T * re,
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
bool
intersect_size_exceed(
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

	options -= VL - _popcnt32( ma );
	if( options <= 0 )
	    return false;

	type lf = tr::set1( lb[VL-1] );
	type rf = tr::set1( rb[VL-1] );

	mask_type ladv = tr::cmple( vl, rf, target::mt_mask() );
	mask_type radv = tr::cmple( tr::loadu( rb ), lf, target::mt_mask() );

	size_t la = ilog2( ((uint64_t)ladv) + 1 );
	options += VL - la; // whichever elements not matched, put back options
	lb += la;
	rb += ilog2( ((uint64_t)radv) + 1 );
    }

    sz = options + exceed - ( le - lb );
    return true;
}

} // namespace detail

template<typename T>
T *
intersect( const T* lb, const T* le, const T* rb, const T* re, T * out ) {
#if __AVX512F__
    detail::intersect<64/sizeof(T),true>( lb, le, rb, re, out );
#endif
#if __AVX2__
    detail::intersect<32/sizeof(T),true>( lb, le, rb, re, out );
#endif
    out = graptor::merge_scalar::intersect( lb, le, rb, re, out );

    return out;
}
    
template<typename T>
size_t
intersect_size( const T* lb, const T* le, const T* rb, const T* re ) {
    size_t sz = 0;
#if __AVX512F__
    detail::intersect<64/sizeof(T),false>( lb, le, rb, re, sz );
#endif
#if __AVX2__
    detail::intersect<32/sizeof(T),false>( lb, le, rb, re, sz );
#endif
    sz += graptor::merge_scalar::intersect_size( lb, le, rb, re );

    return sz;
}

template<typename T>
size_t
intersect_size_exceed( const T* lb, const T* le, const T* rb, const T* re,
		       size_t exceed ) {
    size_t sz0 = 0, sz1 = 0, sz2;

#if __AVX512F__
    if( !detail::intersect_size_exceed<64/sizeof(T)>(
	    lb, le, rb, re, exceed, sz0 ) )
	return 0;
    exceed -= sz0;
#endif
#if __AVX2__
    if( !detail::intersect_size_exceed<32/sizeof(T)>(
	    lb, le, rb, re, exceed, sz1 ) )
	return 0;
    exceed -= sz1;
#endif
    sz2 = merge_scalar::intersect_size_exceed( lb, le, rb, re, exceed );
    if( !sz2 )
	return 0;

    return sz0 + sz1 + sz2;
}

} // namespace merge_vector


} // namespace graptor

#endif // GRAPTOR_CONTAINER_INTERSECT_H
