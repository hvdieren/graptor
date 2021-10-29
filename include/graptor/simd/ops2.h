// -*- C++ -*-
#ifndef GRAPTOR_SIMD_OPS2_H
#define GRAPTOR_SIMD_OPS2_H

#include "graptor/simd/mask.h"
#include "graptor/simd/vector.h"
#include "graptor/simd/mask_ref.h"
#include "graptor/simd/vector_ref.h"

namespace simd {

namespace detail { // within namespace simd

template<typename Tr>
simd::detail::vector_impl<Tr> &
operator += ( simd::detail::vector_impl<Tr> & l,
	      simd::detail::vector_impl<Tr> r ) {
    l.set( simd::detail::vector_impl<Tr>::traits::add( l.data(), r.data() ) );
    return l;
}

template<typename Tr, typename Cr>
simd::detail::vector_impl<Tr>
operator >> ( simd::detail::vector_impl<Tr> l,
	      simd::detail::vector_impl<Cr> r ) {
    if( r.get_layout() == lo_constant )
	return simd::detail::vector_impl<Tr>(
	    simd::detail::vector_impl<Tr>::traits::srl( l.data(), r.at(0) ) );
    else
	return simd::detail::vector_impl<Tr>(
	    simd::detail::vector_impl<Tr>::traits::srlv( l.data(), r.data() ) );
}

template<typename Tr, typename Cr>
simd::detail::vector_impl<Tr>
operator << ( simd::detail::vector_impl<Tr> l,
	      simd::detail::vector_impl<Cr> r ) {
    if( r.get_layout() == lo_constant )
	return simd::detail::vector_impl<Tr>(
	    simd::detail::vector_impl<Tr>::traits::sll( l.data(), r.at(0) ) );
    else
	return simd::detail::vector_impl<Tr>(
	    simd::detail::vector_impl<Tr>::traits::sllv( l.data(), r.data() ) );
}

template<typename Tr>
simd::detail::vector_impl<Tr> operator ~ ( simd::detail::vector_impl<Tr> a ) {
    return simd::detail::vector_impl<Tr>(
	simd::detail::vector_impl<Tr>::traits::bitwise_invert( a.data() ) );
}

template<typename T, unsigned short VL>
simd::vector<T,VL> abs( simd::vector<T,VL> a ) {
    return simd::vector<T, VL>( simd::vector<T,VL>::traits::abs( a.data() ) );
}

template<typename T, unsigned short VL>
auto operator & ( simd::vector<T,VL> l, simd::vector<T,VL> r ) {
    return simd::vector<T,VL>::make_vector( simd::vector<T,VL>::traits::bitwise_and( l.data(), r.data() ) );
}

template<typename Tr>
auto operator | ( simd::detail::vector_impl<Tr> l, simd::detail::vector_impl<Tr> r ) {
    return simd::detail::vector_impl<Tr>::make_vector(
	simd::detail::vector_impl<Tr>::traits::bitwise_or( l.data(), r.data() )
	);
}

template<typename Tr>
auto operator ^ ( simd::detail::vector_impl<Tr> l, simd::detail::vector_impl<Tr> r ) {
    return simd::detail::vector_impl<Tr>::make_vector(
	simd::detail::vector_impl<Tr>::traits::bitwise_xor( l.data(), r.data() )
	);
}

template<typename Tr>
__attribute__((always_inline))
inline auto operator + ( simd::detail::vector_impl<Tr> l, simd::detail::vector_impl<Tr> r ) {
    return simd::detail::vector_impl<Tr>::make_vector(
	simd::detail::vector_impl<Tr>::traits::add( l.data(), r.data() ),
	strongest( l.get_layout(), r.get_layout() ) );
}

template<typename Tr>
__attribute__((always_inline))
inline auto operator - ( vector_impl<Tr> l, vector_impl<Tr> r ) {
    return vector_impl<Tr>::make_vector(
	vector_impl<Tr>::traits::sub( l.data(), r.data() ),
	strongest( l.get_layout(), r.get_layout() ) );
}

template<typename Tr>
__attribute__((always_inline))
inline auto operator * ( vector_impl<Tr> l, vector_impl<Tr> r ) {
    return vector_impl<Tr>::make_vector(
	vector_impl<Tr>::traits::mul( l.data(), r.data() ),
	strongest( l.get_layout(), r.get_layout() ) );
}

template<typename Tr>
auto operator / ( vector_impl<Tr> l, vector_impl<Tr> r ) {
    return vector_impl<Tr>::make_vector(
	vector_impl<Tr>::traits::div( l.data(), r.data() ),
	strongest( l.get_layout(), r.get_layout() ) );
}

} // namespace detail
} // namespace simd

/*
template<typename T, unsigned short VL>
auto add( simd::vector<T,VL> s, typename simd::vector<T,VL>::simd_mask_type m,
	  simd::vector<T,VL> l, simd::vector<T,VL> r ) {
    return simd::vector<T,VL>::make_vector(
	simd::vector<T,VL>::traits::add( s.data(), m.mask(), l.data(), r.data() ),
	strongest( l.get_layout(), r.get_layout() ) );
}
*/

template<typename Tr, typename MTr>
auto add( simd::detail::vector_impl<Tr> s,
	  simd::detail::mask_impl<MTr> m,
	  simd::detail::vector_impl<Tr> l,
	  simd::detail::vector_impl<Tr> r,
	  typename std::enable_if<simd::matchVLtt<Tr,MTr>::value>::type *
	  = nullptr ) {
    using VTy = typename simd::detail::vector_impl<Tr>;
    return VTy::make_vector(
	VTy::traits::add( s.data(), m.get(), l.data(), r.data() ),
	strongest( l.get_layout(), r.get_layout() ) );
}

/*
template<typename T>
auto add( simd::vector<T,1> s, simd::mask<sizeof(T),1> m,
	  simd::vector<T,1> l, simd::vector<T,1> r ) {
    return simd::vector<T,1>::make_vector(
	simd::vector<T,1>::traits::add( s.data(), !!m.mask(), l.data(), r.data() ),
	strongest( l.get_layout(), r.get_layout() ) );
}
*/

template<typename T, unsigned short VL>
auto mul( simd::vector<T,VL> s, typename simd::vector<T,VL>::simd_mask_type m,
	  simd::vector<T,VL> l, simd::vector<T,VL> r ) {
    return simd::vector<T,VL>::make_vector(
	simd::vector<T,VL>::traits::mul( s.data(), m.get(), l.data(), r.data() ),
	strongest( l.get_layout(), r.get_layout() ) );
}

namespace simd {
namespace detail {

template<typename Tr>
auto operator != ( vector_impl<Tr> l, vector_impl<Tr> r ) {
    return vector_impl<Tr>::make_mask(
	vector_impl<Tr>::traits::cmpne(
	    l.data(), r.data(), typename vector_impl<Tr>::tag_type() ) );
}

template<typename Tr>
auto operator == ( vector_impl<Tr> l, vector_impl<Tr> r ) {
    return vector_impl<Tr>::make_mask(
	vector_impl<Tr>::traits::cmpeq(
	    l.data(), r.data(), typename vector_impl<Tr>::tag_type() ) );
}

template<typename Tr>
auto operator >= ( vector_impl<Tr> l, vector_impl<Tr> r ) {
    return vector_impl<Tr>::make_mask(
	vector_impl<Tr>::traits::cmpge(
	    l.data(), r.data(), typename vector_impl<Tr>::tag_type() ) );
}

template<typename Tr>
auto operator > ( vector_impl<Tr> l, vector_impl<Tr> r ) {
    return vector_impl<Tr>::make_mask(
	vector_impl<Tr>::traits::cmpgt(
	    l.data(), r.data(), typename vector_impl<Tr>::tag_type() ) );
}

template<typename Tr>
auto operator < ( vector_impl<Tr> l, vector_impl<Tr> r ) {
    return vector_impl<Tr>::make_mask(
	vector_impl<Tr>::traits::cmplt(
	    l.data(), r.data(), typename vector_impl<Tr>::tag_type() ) );
}

/*
template<typename Tr, typename I, typename Enc, bool NT>
auto operator < ( vector_ref_impl<Tr,I,Enc,NT> l, vector_impl<Tr> r ) {
    return vector_impl<Tr>::make_mask(
	vector_impl<Tr>::traits::cmplt(
	    l.data(), r.data(), typename vector_impl<Tr>::tag_type() ) );
}

template<typename Tr, typename I, typename Enc, bool NT>
auto operator < ( vector_impl<Tr> l, vector_ref_impl<Tr,I,Enc,NT> r ) {
    return vector_impl<Tr>::make_mask(
	vector_impl<Tr>::traits::cmplt(
	    l.data(), r.data(), typename vector_impl<Tr>::tag_type() ) );
}
*/

#if 0 //repeat
template<typename Tr, typename I1, typename I2, typename Enc1, typename Enc2,
	 bool NT1, bool NT2>
auto operator < ( vector_ref_impl<Tr,I1,Enc1,NT1> l,
		  vector_ref_impl<Tr,I2,Enc2,NT2> r ) {
    return vector_impl<Tr>::make_mask(
	vector_impl<Tr>::traits::cmplt(
	    l.data(), r.data(), typename vector_impl<Tr>::tag_type() ) );
}
#endif

} // namespace detail
} // namespace simd

template<typename T, unsigned short W, unsigned short VL>
auto iif( simd::vector<logical<W>,VL> sel,
	  simd::vector<T,VL> l,
	  simd::vector<T,VL> r,
	  typename std::enable_if<sizeof(T)==W>::type * = nullptr ) {
    return simd::vector<T,VL>(
	simd::vector<T,VL>::traits::blendm( sel.data(), l.data(), r.data() ) );
}

template<typename T, typename U, unsigned short W, unsigned short VL,
	 typename Enc, bool NT>
simd::vector<T,VL> iif( simd::vector<logical<W>,VL> sel,
			simd::vector_ref<T,U,VL,Enc,NT> l,
			simd::vector<T,VL> r,
			typename std::enable_if<sizeof(T)==W>::type * = nullptr ) {
    return simd::vector<T,VL>(
	simd::vector<T,VL>::traits::blend( sel.data(), l.data(), r.data() ) );
}

template<typename T, typename U, unsigned short W, unsigned short VL,
	 typename Enc, bool NT>
simd::vector<T,VL> iif( simd::vector<logical<W>,VL> sel,
			simd::vector<T,VL> l,
			simd::vector_ref<T,U,VL,Enc,NT> r,
			typename std::enable_if<sizeof(T)==W>::type * = nullptr ) {
    return simd::vector<typename simd::vector<T,VL>::mask_type, VL>(
	simd::vector<T,VL>::traits::blend( sel.data(), l.data(), r.data() ) );
}

template<typename Tr, typename MTr>
simd::detail::vector_impl<Tr>
iif( simd::detail::mask_impl<MTr> sel,
     simd::detail::vector_impl<Tr> l,
     simd::detail::vector_impl<Tr> r ) {
    return simd::detail::vector_impl<Tr>(
	simd::detail::vector_impl<Tr>::traits::blendm(
	    sel.get(), l.data(), r.data() ) );
}

template<typename Tr>
simd::detail::vector_impl<Tr> iif( bool cond,
				   simd::detail::vector_impl<Tr> l,
				   simd::detail::vector_impl<Tr> r ) {
    using Ty = typename simd::detail::vector_impl<Tr>;
    int cui = cond;
    typename Ty::traits::mask_type m = -cui; // 0 -> 0; 1 -> 111...
    return Ty( Ty::traits::blendm( m, l.data(), r.data() ) );
}

template<typename T, unsigned short VL>
auto min( simd::vector<T,VL> a, simd::vector<T,VL> b ) {
    return simd::vector<T,VL>(
	simd::vector<T,VL>::traits::min( a.data(), b.data() ) );
}

#if 0
template<typename T, unsigned short VL>
auto reduce_min( simd::vector<T,VL> v ) {
    return simd::vector<T,1>(
	simd::vector<T,VL>::traits::reduce_min( v.data() ) );
}

template<typename T, unsigned short VL>
auto reduce_min( simd::vector<T,VL> v, simd::mask<sizeof(T),VL> m ) {
    return simd::vector<T,1>(
	simd::vector<T,VL>::traits::reduce_min( v.data(), m.get() ) );
}

template<typename Tr>
auto reduce_max( simd::detail::vector_impl<Tr> v ) {
    using Tr1 = typename Tr::template rebindVL<1>::type;
    return simd::detail::vector_impl<Tr1>(
	simd::detail::vector_impl<Tr>::traits::reduce_max( v.data() ) );
}

template<typename Tr>
auto reduce_max( simd::detail::vector_impl<Tr> v, simd::nomask<Tr::VL> ) {
    return reduce_max( v );
}

template<typename T, unsigned short VL>
auto reduce_max( simd::vector<T,VL> v, simd::mask<sizeof(T),VL> m ) {
    return simd::vector<T,1>(
	simd::vector<T,VL>::traits::reduce_max( v.data(), m.get() ) );
}

template<typename Tr>
auto reduce_logicalor( simd::detail::vector_impl<Tr> v ) {
    using Tr1 = typename Tr::template rebindVL<1>::type;
    return simd::detail::vector_impl<Tr1>(
	simd::detail::vector_impl<Tr>::traits::reduce_logicalor( v.data() ) );
}

template<typename Tr, typename MTr>
auto reduce_logicalor( simd::detail::vector_impl<Tr> v,
		       simd::detail::mask_impl<MTr> m,
		       typename std::enable_if<simd::matchVLtt<Tr,MTr>::value
		       >::type * = nullptr ) {
    using Tr1 = typename Tr::template rebindVL<1>::type;
    return simd::detail::vector_impl<Tr1>(
	simd::detail::vector_impl<Tr>::traits::reduce_logicalor(
	    v.data(), m.get() ) );
}

template<unsigned short W, unsigned short VL>
auto reduce_logicalor( simd::vector<logical<W>,VL> v, simd::nomask<VL> ) {
    return simd::vector<logical<W>,1>(
	simd::vector<logical<W>,VL>::traits::reduce_logicalor( v.data() ) );
}

template<typename Tr>
auto reduce_bitwiseor( simd::detail::vector_impl<Tr> v ) {
    using Tr1 = typename Tr::template rebindVL<1>::type;
    return simd::detail::vector_impl<Tr1>(
	simd::detail::vector_impl<Tr>::traits::reduce_bitwiseor( v.data() ) );
}

template<typename VTr, typename MTr>
auto reduce_bitwiseor( simd::detail::vector_impl<VTr> v,
		       simd::detail::mask_impl<MTr> m ) {
    using VTr1 = typename VTr::template rebindVL<1>::type;
    return simd::detail::vector_impl<VTr1>(
	simd::detail::mask_impl<VTr>::traits::reduce_bitwiseor(
	    v.data(), m.get() ) );
}


template<typename T, unsigned short VL>
__attribute__((always_inline))
static inline
auto reduce_add( simd::vector<T,VL> v ) {
    return simd::vector<T,1>(
	simd::vector<T,VL>::traits::reduce_add( v.data() ) );
}

/*
template<typename T, unsigned short VL>
auto reduce_add( simd::vector<T,VL> v, simd::vector<logical<sizeof(T)>,VL> m ) {
    return simd::vector<T,1>(
	simd::vector<T,VL>::traits::reduce_add( v.data(), m.data() ) );
}
*/

template<typename T, unsigned short VL>
auto reduce_add( simd::vector<T,VL> v, simd::mask<sizeof(T),VL> m ) {
    return simd::vector<T,1>(
	simd::vector<T,VL>::traits::reduce_add( v.data(), m.get() ) );
}

template<typename T, unsigned short VL>
auto reduce_mul( simd::vector<T,VL> v ) {
    return simd::vector<T,1>(
	simd::vector<T,VL>::traits::reduce_mul( v.data() ) );
}

template<typename T, unsigned short VL>
auto reduce_mul( simd::vector<T,VL> v, simd::vector<logical<sizeof(T)>,VL> m ) {
    return simd::vector<T,1>(
	simd::vector<T,VL>::traits::reduce_mul( v.data(), m.data() ) );
}

template<typename VTr, typename MTr>
auto blend( simd::detail::mask_impl<MTr> m,
	    simd::detail::vector_impl<VTr> l,
	    simd::detail::vector_impl<VTr> r,
	    typename std::enable_if<simd::matchVLtt<VTr,MTr>::value>::type *
	    = nullptr ) {
    return simd::detail::vector_impl<VTr>::make_vector(
	simd::detail::vector_impl<VTr>::traits::blendm(
	    m.get(), l.data(), r.data() ) );
}


template<typename T, unsigned short VL>
auto reduce_setif( simd::vector<T,VL> v ) {
    return simd::vector<T,1>(
	simd::vector<T,VL>::traits::reduce_setif( v.data() ) );
}

template<typename T, unsigned short VL>
auto reduce_setif( simd::vector<T,VL> v, simd::vector<logical<sizeof(T)>,VL> m ) {
    return simd::vector<T,1>(
	simd::vector<T,VL>::traits::reduce_setif( v.data(), m.data() ) );
}

template<typename Tr, typename MTr>
auto reduce_setif( simd::detail::vector_impl<Tr> v,
		   simd::detail::mask_impl<MTr> m ) {
    using Tr1 = typename Tr::template rebindVL<1>::type;
    return simd::detail::vector_impl<Tr1>(
	simd::detail::vector_impl<Tr>::traits::reduce_setif( v.data(), m.data() ) );
}
#endif

namespace simd {

namespace detail {

#if 0 // repeated between ops.h and ops2.h
template<class Tr, typename I, typename Enc, bool NT_>
typename vector_ref_impl<Tr,I,Enc,NT_>::simd_mask_type
vector_ref_impl<Tr,I,Enc,NT_>::bor_assign(
    typename vector_ref_impl<Tr,I,Enc,NT_>::vector_type r,
    typename vector_ref_impl<Tr,I,Enc,NT_>::simd_mask_type m ) {
#if ALT_BOR_ASSIGN
    auto a = load( m ); // load data
    auto upd = a | r;
    store( upd, m ); // store data
    // Ensures that return value is stronger than m
    return (upd != a).asmask() & m;
#else
    auto a = load( m ); // load data
    auto mod = ~a & r;
    store( a | r, m ); // store data
    auto v = mod;
    auto z = vector_type::zero_val();
    // Ensures that return value is stronger than m
    return (v != z).asmask() & m;
#endif
}

template<class Tr, typename I, typename Enc, bool NT_>
typename vector_ref_impl<Tr,I,Enc,NT_>::simd_mask_type
vector_ref_impl<Tr,I,Enc,NT_>::bor_assign(
    typename vector_ref_impl<Tr,I,Enc,NT_>::vector_type r ) {
#if ALT_BOR_ASSIGN
    auto a = load(); // load data
    auto upd = a | r;
    store( upd ); // store data
    return (upd != a).asmask();
#else
    auto a = load(); // load data
    auto mod = ~a & r;
    store( a | r ); // store data
    auto v = mod;
    auto z = vector_type::zero_val();
    return (v != z).asmask();
#endif
}


template<class Tr, typename I, typename Enc, bool NT_>
typename vector_ref_impl<Tr,I,Enc,NT_>::simd_mask_type
vector_ref_impl<Tr,I,Enc,NT_>::lor_assign(
    typename vector_ref_impl<Tr,I,Enc,NT_>::vector_type r,
    typename vector_ref_impl<Tr,I,Enc,NT_>::simd_mask_type m ) {
    auto a = load( m ); // load data
    auto mod = ~a & r;
    store( a | r, m ); // store data
    auto v = mod;
    // Ensures that return value is stronger than m
    // return (v & m).asmask();
    return v.asmask() & m;
}

template<class Tr, typename I, typename Enc, bool NT_>
typename vector_ref_impl<Tr,I,Enc,NT_>::simd_mask_type
vector_ref_impl<Tr,I,Enc,NT_>::lor_assign(
    typename vector_ref_impl<Tr,I,Enc,NT_>::vector_type r ) {
    auto a = load(); // load data
    auto mod = ~a & r;
    store( a | r ); // store data
    return mod.asmask();
}

template<class Tr, typename I, typename Enc, bool NT_>
typename vector_ref_impl<Tr,I,Enc,NT_>::simd_mask_type
vector_ref_impl<Tr,I,Enc,NT_>::lor_assign(
    typename vector_ref_impl<Tr,I,Enc,NT_>::vector_type r,
    simd::nomask<vector_ref_impl<Tr,I,Enc,NT_>::VL> ) {
    return this->lor_assign( r );
}
#endif

} // namespace detail

} // namespace simd

#endif // GRAPTOR_SIMD_OPS2_H
