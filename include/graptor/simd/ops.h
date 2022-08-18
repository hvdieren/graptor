// -*- C++ -*-
#ifndef GRAPTOR_SIMD_OPS_H
#define GRAPTOR_SIMD_OPS_H

#include "graptor/simd/mask.h"
#include "graptor/simd/vector.h"
#include "graptor/simd/mask_ref.h"
#include "graptor/simd/vector_ref.h"
#include "graptor/simd/ops_mask.h"
#include "graptor/simd/ops2.h"

namespace simd {

namespace detail { // within namespace simd

template<typename Tr, layout_t Layout1, layout_t Layout2>
std::enable_if_t<Layout1 == _arith_add(Layout1,Layout2),
		 simd::detail::vec<Tr,Layout1>> &
operator += ( simd::detail::vec<Tr,Layout1> & l,
	      simd::detail::vec<Tr,Layout2> r ) {
    l.set_unsafe( l + r );
    return l;
}

template<typename Tr, typename Cr, layout_t Layout1, layout_t Layout2>
auto
operator >> ( simd::detail::vec<Tr,Layout1> l,
	      simd::detail::vec<Cr,Layout2> r ) {
    if constexpr ( Layout2 == lo_constant )
	return simd::detail::vec<Tr,_arith_cst(Layout1)>(
	    Tr::traits::srl( l.data(), r.at(0) ) );
    else
	return simd::detail::vec<Tr,lo_unknown>(
	    Tr::traits::srlv( l.data(), r.data() ) );
}

template<typename Tr, typename Cr, layout_t Layout1, layout_t Layout2>
auto sra( simd::detail::vec<Tr,Layout1> l,
	  simd::detail::vec<Cr,Layout2> r ) {
    if constexpr ( Layout2 == lo_constant )
	return simd::detail::vec<Tr,_arith_cst(Layout1)>(
	    Tr::traits::sra( l.data(), r.at(0) ) );
    else
	return simd::detail::vec<Tr,lo_unknown>(
	    Tr::traits::srav( l.data(), r.data() ) );
}

template<typename Tr, typename Cr, layout_t Layout1, layout_t Layout2>
auto
operator << ( simd::detail::vec<Tr,Layout1> l,
	      simd::detail::vec<Cr,Layout2> r ) {
    if constexpr ( Layout2 == lo_constant )
	return simd::detail::vec<Tr,_arith_cst(Layout1)>(
	    Tr::traits::sll( l.data(), r.at(0) ) );
    else
	return simd::detail::vec<Tr,lo_unknown>(
	    Tr::traits::sllv( l.data(), r.data() ) );
}

template<unsigned short Shift, typename Tr, layout_t Layout>
auto slli( simd::detail::vec<Tr,Layout> l ) {
    return simd::detail::vec<Tr,_arith_cst(Layout)>(
	Tr::traits::slli( l.data(), Shift ) );
}

template<unsigned short Shift, typename Tr, layout_t Layout>
auto srli( simd::detail::vec<Tr,Layout> l ) {
    return simd::detail::vec<Tr,_arith_cst(Layout)>(
	Tr::traits::srli( l.data(), Shift ) );
}

template<unsigned short Shift, typename Tr, layout_t Layout>
auto srai( simd::detail::vec<Tr,Layout> l ) {
    return simd::detail::vec<Tr,_arith_cst(Layout)>(
	Tr::traits::srai( l.data(), Shift ) );
}

template<typename Tr, layout_t Layout>
auto operator ~ ( simd::vec<Tr,Layout> a ) {
    return simd::detail::vec<Tr,_arith_cst(Layout)>(
	Tr::traits::bitwise_invert( a.data() ) );
}

template<typename Tr, layout_t Layout>
auto abs( simd::detail::vec<Tr,Layout> a ) {
    return simd::detail::vec<Tr,_arith_cst(Layout)>(
	Tr::traits::abs( a.data() ) );
}

template<typename Tr, layout_t Layout1, layout_t Layout2>
auto operator & ( simd::detail::vec<Tr,Layout1> l,
		  simd::detail::vec<Tr,Layout2> r ) {
    return simd::detail::vec<Tr,_arith_cst(Layout1,Layout2)>(
	Tr::traits::bitwise_and( l.data(), r.data() ) );
}

template<typename Tr, layout_t Layout1, layout_t Layout2>
auto operator | ( simd::detail::vec<Tr,Layout1> l,
		  simd::detail::vec<Tr,Layout2> r ) {
    return simd::detail::vec<Tr,_arith_cst(Layout1,Layout2)>(
	Tr::traits::bitwise_or( l.data(), r.data() ) );
}

template<typename Tr, layout_t Layout1, layout_t Layout2>
auto operator ^ ( simd::detail::vec<Tr,Layout1> l,
		  simd::detail::vec<Tr,Layout2> r ) {
    return simd::detail::vec<Tr,_arith_cst(Layout1,Layout2)>(
	Tr::traits::bitwise_xor( l.data(), r.data() ) );
}

template<typename Tr, layout_t Layout1, layout_t Layout2>
auto operator + ( simd::detail::vec<Tr,Layout1> l,
		  simd::detail::vec<Tr,Layout2> r ) {
    return simd::detail::vec<Tr,_arith_add(Layout1,Layout2)>(
	Tr::traits::add( l.data(), r.data() ) );
}

template<typename Tr, layout_t Layout1, layout_t Layout2>
auto operator - ( simd::detail::vec<Tr,Layout1> l,
		  simd::detail::vec<Tr,Layout2> r ) {
    return simd::detail::vec<Tr,_arith_add(Layout1,Layout2)>(
	Tr::traits::sub( l.data(), r.data() ) );
}

template<typename Tr, layout_t Layout1, layout_t Layout2>
auto operator * ( simd::detail::vec<Tr,Layout1> l,
		  simd::detail::vec<Tr,Layout2> r ) {
    return simd::detail::vec<Tr,_arith_cst(Layout1,Layout2)>(
	Tr::traits::mul( l.data(), r.data() ) );
}

template<typename Tr, layout_t Layout1, layout_t Layout2>
auto mulhi( simd::detail::vec<Tr,Layout1> l,
	    simd::detail::vec<Tr,Layout2> r ) {
    return simd::detail::vec<Tr,_arith_cst(Layout1,Layout2)>(
	Tr::traits::mulhi( l.data(), r.data() ) );
}

template<typename Tr, layout_t Layout1, layout_t Layout2>
auto operator / ( simd::detail::vec<Tr,Layout1> l,
		  simd::detail::vec<Tr,Layout2> r ) {
    return simd::detail::vec<Tr,_arith_cst(Layout1,Layout2)>(
	Tr::traits::div( l.data(), r.data() ) );
}

template<typename Tr, layout_t Layout1, layout_t Layout2>
auto operator % ( simd::detail::vec<Tr,Layout1> l,
		  simd::detail::vec<Tr,Layout2> r ) {
    return simd::detail::vec<Tr,_arith_cst(Layout1,Layout2)>(
	Tr::traits::mod( l.data(), r.data() ) );
}

} // namespace detail
} // namespace simd

template<typename Tr, typename MTr, simd::layout_t Layout1,
	 simd::layout_t Layout2, simd::layout_t Layout3>
auto add( simd::detail::vec<Tr,Layout1> s,
	  simd::detail::mask_impl<MTr> m,
	  simd::detail::vec<Tr,Layout2> l,
	  simd::detail::vec<Tr,Layout3> r,
	  typename std::enable_if<simd::matchVLtt<Tr,MTr>::value>::type *
	  = nullptr ) {
    return simd::detail::vec<Tr,simd::lo_unknown>(
	Tr::traits::add( s.data(), m.get(), l.data(), r.data() ) );
}

template<typename Tr, typename MTr, simd::layout_t Layout1,
	 simd::layout_t Layout2, simd::layout_t Layout3>
auto mul( simd::detail::vec<Tr,Layout1> s,
	  simd::detail::mask_impl<MTr> m,
	  simd::detail::vec<Tr,Layout2> l,
	  simd::detail::vec<Tr,Layout3> r,
	  typename std::enable_if<simd::matchVLtt<Tr,MTr>::value>::type *
	  = nullptr ) {
    return simd::detail::vec<Tr,simd::lo_unknown>(
	Tr::traits::mul( s.data(), m.get(), l.data(), r.data() ) );
}

namespace simd {

template<typename Tr, layout_t Layout1, layout_t Layout2>
auto min( detail::vec<Tr,Layout1> a,
	  detail::vec<Tr,Layout2> b ) {
    return detail::vec<Tr,lo_unknown>(
	Tr::traits::min( a.data(), b.data() ) );
}

template<typename Tr, layout_t Layout1, layout_t Layout2>
auto max( detail::vec<Tr,Layout1> a,
	  detail::vec<Tr,Layout2> b ) {
    return detail::vec<Tr,lo_unknown>(
	Tr::traits::max( a.data(), b.data() ) );
}

namespace detail {

template<typename Tr, layout_t Layout1, layout_t Layout2>
auto operator != ( vec<Tr,Layout1> l, vec<Tr,Layout2> r ) {
    return vec<Tr,Layout1>::make_mask(
	Tr::traits::cmpne( l.data(), r.data(), typename Tr::tag_type() ) );
}

template<typename Tr, layout_t Layout1, layout_t Layout2>
auto operator == ( const vec<Tr,Layout1> & l, const vec<Tr,Layout2> & r ) {
    return vec<Tr,Layout1>::make_mask(
	Tr::traits::cmpeq( l.data(), r.data(), typename Tr::tag_type() ) );
}

template<typename Tr, layout_t Layout1, layout_t Layout2>
auto operator == ( vec<Tr,Layout1> && l, vec<Tr,Layout2> && r ) {
    return vec<Tr,Layout1>::make_mask(
	Tr::traits::cmpeq( std::move( l.data() ), std::move( r.data() ),
			   typename Tr::tag_type() ) );
}

template<typename Tr, layout_t Layout1, layout_t Layout2>
auto operator > ( vec<Tr,Layout1> l, vec<Tr,Layout2> r ) {
    return vec<Tr,Layout1>::make_mask(
	Tr::traits::cmpgt( l.data(), r.data(), typename Tr::tag_type() ) );
}

template<typename Tr, layout_t Layout1>
auto operator > ( vec<Tr,Layout1> l, vector_impl<Tr> r ) {
    return vec<Tr,Layout1>::make_mask(
	Tr::traits::cmpgt( l.data(), r.data(), typename Tr::tag_type() ) );
}

template<typename Tr, layout_t Layout1, layout_t Layout2>
auto operator >= ( vec<Tr,Layout1> l, vec<Tr,Layout2> r ) {
    return vec<Tr,Layout1>::make_mask(
	Tr::traits::cmpge( l.data(), r.data(), typename Tr::tag_type() ) );
}

template<typename Tr, layout_t Layout1, layout_t Layout2>
auto operator < ( vec<Tr,Layout1> l, vec<Tr,Layout2> r ) {
    return vec<Tr,Layout1>::make_mask(
	Tr::traits::cmplt( l.data(), r.data(), typename Tr::tag_type() ) );
}

template<typename Tr, layout_t Layout1, layout_t Layout2>
auto operator <= ( vec<Tr,Layout1> l, vec<Tr,Layout2> r ) {
    return vec<Tr,Layout1>::make_mask(
	Tr::traits::cmple( l.data(), r.data(), typename Tr::tag_type() ) );
}

template<typename Tr, typename I, typename Enc, bool NT, layout_t Layout1,
	 layout_t Layout2>
auto operator < ( vector_ref_impl<Tr,I,Enc,NT,Layout1> l, vec<Tr,Layout2> r ) {
    return vector_impl<Tr>::make_mask(
	Tr::traits::cmplt( l.data(), r.data(), typename Tr::tag_type() ) );
}

template<typename Tr, typename I, typename Enc, bool NT, layout_t Layout1,
	 layout_t Layout2>
auto operator < ( vec<Tr,Layout1> l, vector_ref_impl<Tr,I,Enc,NT,Layout2> r ) {
    return vector_impl<Tr>::make_mask(
	Tr::traits::cmplt( l.data(), r.data(), typename Tr::tag_type() ) );
}

template<typename Tr, typename I1, typename I2, typename Enc1, typename Enc2,
	 bool NT1, bool NT2, layout_t Layout1, layout_t Layout2>
auto operator < ( vector_ref_impl<Tr,I1,Enc1,NT1,Layout1> l,
		  vector_ref_impl<Tr,I2,Enc2,NT2,Layout2> r ) {
    return vector_impl<Tr>::make_mask(
	Tr::traits::cmplt( l.data(), r.data(), typename Tr::tag_type() ) );
}

} // namespace detail
} // namespace simd

template<typename Tr, typename MTr, simd::layout_t Layout1,
	 simd::layout_t Layout2, simd::layout_t Layout3>
auto iif( simd::detail::vec<MTr,Layout1> sel,
	  simd::detail::vec<Tr,Layout2> l,
	  simd::detail::vec<Tr,Layout3> r,
	  std::enable_if_t< is_logical_v<typename Tr::member_type>
	  && Tr::W==MTr::W> * = nullptr ) {
    return simd::detail::vec<Tr,_arith_cst(Layout2,Layout3)>(
	Tr::traits::blendm( sel.data(), l.data(), r.data() ) );
}

template<typename Tr, typename MTr, simd::layout_t Layout1,
	 simd::layout_t Layout2>
auto iif( simd::detail::mask_impl<MTr> sel,
	  simd::detail::vec<Tr,Layout1> l,
	  simd::detail::vec<Tr,Layout2> r ) {
    return simd::detail::vec<Tr,_arith_cst(Layout1,Layout2)>(
	Tr::traits::blendm( sel.get(), l.data(), r.data() ) );
}

/* suspect -- check before enabling: should be vmask_type instead mask_type?
 *         -- applies only to VL==1 due to lack of set1()?
template<typename Tr, layout_t Layout1, layout_t Layout2>
auto iif( bool sel,
	  simd::detail::vec<Tr,Layout1> l,
	  simd::detail::vec<Tr,Layout2> r ) {
    int cui = sel;
    typename Ty::traits::mask_type m = -cui; // 0 -> 0; 1 -> 111...
    return simd::detail::vec<Tr,_arith_cst(Layout1,Layout2)>(
	Tr::traits::blendm( sel.get(), l.data(), r.data() ) );
}
*/

template<typename VTr, typename MTr,
	 simd::layout_t Layout1, simd::layout_t Layout2>
auto blend( simd::detail::mask_impl<MTr> m,
	    simd::detail::vec<VTr,Layout1> l,
	    simd::detail::vec<VTr,Layout2> r,
	    typename std::enable_if<simd::matchVLtt<VTr,MTr>::value>::type *
	    = nullptr ) {
    static_assert( !std::is_class_v<VTr>,
		   "check where this code is used, is redundant to iif" );
    return simd::detail::vec<VTr,simd::lo_unknown>(
	VTr::traits::blendm( m.get(), l.data(), r.data() ) );
}

#define REDUCTION_OP min
#include "graptor/simd/reduction.h"
#undef REDUCTION_OP

#define REDUCTION_OP max
#include "graptor/simd/reduction.h"
#undef REDUCTION_OP

#define REDUCTION_OP logicalor
#include "graptor/simd/reduction.h"
#undef REDUCTION_OP

#define REDUCTION_OP bitwiseor
#include "graptor/simd/reduction.h"
#undef REDUCTION_OP

#define REDUCTION_OP add
#include "graptor/simd/reduction.h"
#undef REDUCTION_OP

#define REDUCTION_OP mul
#include "graptor/simd/reduction.h"
#undef REDUCTION_OP

#define REDUCTION_OP setif
#include "graptor/simd/reduction.h"
#undef REDUCTION_OP

namespace simd {

namespace detail {

template<class Tr, typename I, typename Enc, bool NT_,layout_t Layout>
template<layout_t Layout_>
typename vector_ref_impl<Tr,I,Enc,NT_,Layout>::simd_mask_type
vector_ref_impl<Tr,I,Enc,NT_,Layout>::bor_assign(
    vec<Tr,Layout_> r,
    typename vector_ref_impl<Tr,I,Enc,NT_,Layout>::simd_mask_type m ) {
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

template<class Tr, typename I, typename Enc, bool NT_, layout_t Layout>
template< layout_t Layout_>
typename vector_ref_impl<Tr,I,Enc,NT_,Layout>::simd_mask_type
vector_ref_impl<Tr,I,Enc,NT_,Layout>::bor_assign(
    vec<Tr,Layout_> r ) {
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

template<class Tr, typename I, typename Enc, bool NT_, layout_t Layout>
typename vector_ref_impl<Tr,I,Enc,NT_,Layout>::simd_mask_type
vector_ref_impl<Tr,I,Enc,NT_,Layout>::bor_assign(
    mask_impl<Tr> r ) {
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

template<class Tr, typename I, typename Enc, bool NT_,layout_t Layout>
template<layout_t Layout_>
typename vector_ref_impl<Tr,I,Enc,NT_,Layout>::simd_mask_type
vector_ref_impl<Tr,I,Enc,NT_,Layout>::band_assign(
    vec<Tr,Layout_> r,
    typename vector_ref_impl<Tr,I,Enc,NT_,Layout>::simd_mask_type m ) {
#if ALT_BOR_ASSIGN
    auto a = load( m ); // load data
    auto upd = a & r;
    store( upd, m ); // store data
    // Ensures that return value is stronger than m
    return (upd != a).asmask() & m;
#else
    auto a = load( m ); // load data
    auto mod = a & ~r;
    store( a & r, m ); // store data
    auto v = mod;
    auto z = vector_type::zero_val();
    // Ensures that return value is stronger than m
    return (v != z).asmask() & m;
#endif
}

template<class Tr, typename I, typename Enc, bool NT_, layout_t Layout>
template< layout_t Layout_>
typename vector_ref_impl<Tr,I,Enc,NT_,Layout>::simd_mask_type
vector_ref_impl<Tr,I,Enc,NT_,Layout>::band_assign(
    vec<Tr,Layout_> r ) {
#if ALT_BOR_ASSIGN
    auto a = load(); // load data
    auto upd = a & r;
    store( upd ); // store data
    return (upd != a).asmask();
#else
    auto a = load(); // load data
    auto mod = a & ~r;
    store( a & r ); // store data
    auto v = mod;
    auto z = vector_type::zero_val();
    return (v != z).asmask();
#endif
}

template<class Tr, typename I, typename Enc, bool NT_, layout_t Layout>
typename vector_ref_impl<Tr,I,Enc,NT_,Layout>::simd_mask_type
vector_ref_impl<Tr,I,Enc,NT_,Layout>::band_assign(
    mask_impl<Tr> r ) {
#if ALT_BOR_ASSIGN
    auto a = load(); // load data
    auto upd = a & r;
    store( upd ); // store data
    return (upd != a).asmask();
#else
    auto a = load(); // load data
    auto mod = a & ~r;
    store( a & r ); // store data
    auto v = mod;
    auto z = vector_type::zero_val();
    return (v != z).asmask();
#endif
}


template<class Tr, typename I, typename Enc, bool NT_, layout_t Layout>
template< layout_t Layout_>
typename vector_ref_impl<Tr,I,Enc,NT_,Layout>::simd_mask_type
vector_ref_impl<Tr,I,Enc,NT_,Layout>::lor_assign(
    vec<Tr,Layout_> r,
    typename vector_ref_impl<Tr,I,Enc,NT_,Layout>::simd_mask_type m ) {
    auto a = load( m ); // load data
    auto mod = ~a & r;
    store( a | r, m ); // store data
    auto v = mod;
    // Ensures that return value is stronger than m
    // return (v & m).asmask();
    return v.asmask() & m;
}

template<class Tr, typename I, typename Enc, bool NT_, layout_t Layout>
template< layout_t Layout_>
typename vector_ref_impl<Tr,I,Enc,NT_,Layout>::simd_mask_type
vector_ref_impl<Tr,I,Enc,NT_,Layout>::lor_assign(
    vec<Tr,Layout_> r ) {
    auto a = load(); // load data
    auto mod = ~a & r;
    store( a | r ); // store data
    return mod.asmask();
}

template<class Tr, typename I, typename Enc, bool NT_, layout_t Layout>
template< layout_t Layout_>
typename vector_ref_impl<Tr,I,Enc,NT_,Layout>::simd_mask_type
vector_ref_impl<Tr,I,Enc,NT_,Layout>::lor_assign(
    vec<Tr,Layout_> r,
    simd::nomask<vector_ref_impl<Tr,I,Enc,NT_,Layout>::VL> ) {
    return this->lor_assign( r );
}

template<class Tr, typename I, typename Enc, bool NT_, layout_t Layout>
template< layout_t Layout_>
typename vector_ref_impl<Tr,I,Enc,NT_,Layout>::simd_mask_type
vector_ref_impl<Tr,I,Enc,NT_,Layout>::land_assign(
    vec<Tr,Layout_> r,
    typename vector_ref_impl<Tr,I,Enc,NT_,Layout>::simd_mask_type m ) {
    auto a = load( m ); // load data
    auto mod = a & ~r;
    store( a & r, m ); // store data
    auto v = mod;
    // Ensures that return value is stronger than m
    // return (v & m).asmask();
    return v.asmask() & m;
}

template<class Tr, typename I, typename Enc, bool NT_, layout_t Layout>
template< layout_t Layout_>
typename vector_ref_impl<Tr,I,Enc,NT_,Layout>::simd_mask_type
vector_ref_impl<Tr,I,Enc,NT_,Layout>::land_assign(
    vec<Tr,Layout_> r ) {
    auto a = load(); // load data
    auto mod = a & ~r;
    store( a & r ); // store data
    return mod.asmask();
}

template<class Tr, typename I, typename Enc, bool NT_, layout_t Layout>
template< layout_t Layout_>
typename vector_ref_impl<Tr,I,Enc,NT_,Layout>::simd_mask_type
vector_ref_impl<Tr,I,Enc,NT_,Layout>::land_assign(
    vec<Tr,Layout_> r,
    simd::nomask<vector_ref_impl<Tr,I,Enc,NT_,Layout>::VL> ) {
    return this->land_assign( r );
}

} // namespace detail

} // namespace simd

#endif // GRAPTOR_SIMD_OPS_H
