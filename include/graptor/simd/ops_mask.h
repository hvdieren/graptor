// -*- C++ -*-
#ifndef GRAPTOR_SIMD_OPS_MASK_H
#define GRAPTOR_SIMD_OPS_MASK_H

#include "graptor/simd/mask.h"
#include "graptor/simd/vector.h"
#include "graptor/simd/mask_ref.h"
#include "graptor/simd/vector_ref.h"
#include "graptor/simd/ops2.h"

namespace simd {

template<unsigned short VL>
nomask<VL> operator && ( nomask<VL>, nomask<VL> ) {
    return nomask<VL>();
}

template<unsigned short VL>
nomask<VL> operator || ( nomask<VL>, nomask<VL> ) {
    return nomask<VL>();
}

namespace detail { // within namespace simd

template<typename Tr>
auto operator || ( simd::detail::mask_impl<Tr> l,
		   simd::detail::mask_impl<Tr> r ) {
    return simd::detail::mask_impl<Tr>( 
	simd::detail::mask_impl<Tr>::traits::logical_or( l.get(), r.get() ) );
}

template<typename Tr>
auto operator && ( mask_impl<Tr> l,
		   mask_impl<Tr> r ) {
    return mask_impl<Tr>( 
	mask_impl<Tr>::traits::logical_and( l.get(), r.get() ) );
}

template<typename Tr>
mask_impl<Tr> operator && ( mask_impl<Tr> l, nomask<Tr::VL> ) {
    return l;
}

template<typename Tr>
mask_impl<Tr> operator && ( nomask<Tr::VL>, mask_impl<Tr> r ) {
    return r;
}

template<unsigned short W, unsigned short VL>
auto operator && ( mask_impl<mask_bit_traits<VL>> l,
		   mask_impl<mask_logical_traits<W,VL>> r ) {
    using Tr = mask_preferred_traits_width<W,VL>;
    return mask_impl<Tr>(
	l.template convert<Tr>() && r.template convert<Tr>() );
}

inline auto
operator && ( mask_impl<mask_bool_traits> l,
	      mask_impl<mask_bit_logical_traits<2,1>> r ) {
    return mask_impl<mask_bool_traits>( 
	l.get() && ( r.get() > 1 ) ); // only top bit matters
}

// This shouldn't be necessary, or occur during evaluation of ASTs:
// mask_bit_traits<1> should not be used
inline auto
operator && ( mask_impl<mask_bit_traits<1>> l,
	      mask_impl<mask_bool_traits> r ) {
    return mask_impl<mask_bool_traits>( 
	mask_bool_traits::traits::logical_and(
	    l.get(),
	    mask_bit_traits<1>::traits::cmpne(
		r.get(),
		mask_bit_traits<1>::traits::setzero(),
		target::mt_bool() ) ) );
}

template<typename Tr>
simd::detail::mask_impl<Tr> operator ~ ( simd::detail::mask_impl<Tr> a ) {
    return simd::detail::mask_impl<Tr>( simd::detail::mask_impl<Tr>::traits::logical_invert( a.get() ) );
}

template<typename Tr>
simd::detail::mask_impl<Tr> operator ! ( simd::detail::mask_impl<Tr> a ) {
    return simd::detail::mask_impl<Tr>( simd::detail::mask_impl<Tr>::traits::logical_invert( a.get() ) );
}

template<typename Tr>
auto operator != ( mask_impl<Tr> l, mask_impl<Tr> r ) {
    return mask_impl<Tr>( mask_impl<Tr>::traits::cmpne(
			      l.get(), r.get(), typename Tr::tag_type() ) );
}

template<typename Tr>
auto operator == ( mask_impl<Tr> l, mask_impl<Tr> r ) {
    return mask_impl<Tr>( mask_impl<Tr>::traits::cmpeq(
			      l.get(), r.get(), typename Tr::tag_type() ) );
}

} // namespace detail
} // namespace simd

template<typename Tr1, typename Tr2>
auto iif( simd::detail::mask_impl<Tr1> sel,
	  simd::detail::mask_impl<Tr2> l,
	  simd::detail::mask_impl<Tr2> r ) {
    return simd::detail::mask_impl<Tr2>(
	simd::detail::mask_impl<Tr2>::traits::blendm(
	    sel.data(), l.data(), r.data() ) );
}

template<typename Tr>
auto reduce_logicalor( simd::detail::mask_impl<Tr> m ) {
    using Tr1 = typename Tr::template rebindVL<1>::type;
    return simd::detail::mask_impl<Tr1>(
	simd::detail::mask_impl<Tr>::traits::reduce_logicalor( m.get() ) );
}

template<typename Tr>
auto reduce_logicaland( simd::detail::mask_impl<Tr> m ) {
    using Tr1 = typename Tr::template rebindVL<1>::type;
    return simd::detail::mask_impl<Tr1>(
	simd::detail::mask_impl<Tr>::traits::reduce_logicaland( m.get() ) );
}

template<typename Tr>
auto reduce_bitwiseor( simd::detail::mask_impl<Tr> v ) {
    using Tr1 = typename Tr::template rebindVL<1>::type;
    return simd::detail::mask_impl<Tr1>(
	simd::detail::mask_impl<Tr>::traits::reduce_bitwiseor( v.data() ) );
}

#endif // GRAPTOR_SIMD_OPS_MASK_H
