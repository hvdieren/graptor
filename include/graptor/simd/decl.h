// -*- c++ -*-

#ifndef GRAPTOR_SIMD_DECL_H
#define GRAPTOR_SIMD_DECL_H

#include <type_traits>
#include "graptor/itraits.h"
#include "graptor/target/decl.h"
#include "graptor/target/vector.h"

/***********************************************************************
 * pre-declarations
 ***********************************************************************/

namespace simd {

/***********************************************************************
 * layout_t: tracking the constitution of vectors of indices
 ***********************************************************************/
enum layout_t {
    lo_unknown = 0,
    lo_constant = 1,
    lo_linear = 2,
    lo_linalgn = 3, // linear and aligned; if not linear, don't care about align
    lo_variable = 4
};

inline layout_t strongest( layout_t l, layout_t r ) {
    switch( l ) {
    case lo_unknown: return lo_unknown;
    case lo_constant: return r == lo_linalgn ? lo_linear : r;
    case lo_linear: return r == lo_unknown ? lo_unknown : lo_linear;
    case lo_linalgn: return r == lo_unknown ? lo_unknown : lo_linear;
    case lo_variable:
    default: UNREACHABLE_CASE_STATEMENT;
    }
}

inline bool maintains_left( layout_t l, layout_t r ) {
    switch( l ) {
    case lo_unknown: return true;
    case lo_constant: return r == lo_constant;
    case lo_linear: return r == lo_linear || r == lo_linalgn;
    case lo_linalgn: return r == lo_linalgn;
    case lo_variable:
    default: UNREACHABLE_CASE_STATEMENT;
    }
}

inline constexpr layout_t _arith_cst( layout_t l ) {
    return l == lo_constant ? lo_constant : lo_unknown;
}

inline constexpr layout_t _arith_cst( layout_t l, layout_t r ) {
    return l == lo_constant && r == lo_constant ? lo_constant : lo_unknown;
}

inline constexpr layout_t _arith_add( layout_t l, layout_t r ) {
    if ( l == lo_constant && r == lo_constant )
	return lo_constant;
    else if ( l == lo_constant )
	return r == lo_linear || r == lo_linalgn ? lo_linear : lo_unknown;
    else if ( r == lo_constant )
	return l == lo_linear || l == lo_linalgn ? lo_linear : lo_unknown;
    else
	return lo_unknown;
}

namespace detail {

// template<class Traits>
// class vector_impl;

template<class Traits, layout_t Layout>
class vec;

template<class Traits>
class mask_impl;

template<typename T, unsigned short VL>
struct vdata_traits;

/***********************************************************************
 * mask configurations
 *
 * member_type is a legacy element. It is to be replaced by element_type
 * A vector is VL x member_type, however, a bitmask is VL bits.
 * We cannot easily represent a single bit in a C++ type (every variable
 * is at least 1 byte large). So for a bitmask, member_type is void,
 * and element_type is bool.
 * The code needs to be migrated to use of element_type only and should
 * avoid assumptions that sizeof(member_type) x VL == sizeof(type).
 ***********************************************************************/
template<unsigned short W_, unsigned short VL_>
struct mask_logical_traits {
    static constexpr unsigned short W = W_;
    static constexpr unsigned short B = 8*W;
    static constexpr unsigned short VL = VL_;
    using member_type = logical<W>;
    using element_type = member_type;
    using traits = vector_type_traits_vl<member_type, VL>;
    using type = typename traits::type;
    using pointer_type = member_type;
    using tag_type = target::mt_vmask;
    using index_type = index_type_of_size<sizeof(member_type)>;

    using prefmask_traits = detail::mask_logical_traits<W, VL>;

    template<unsigned short VL2>
    struct rebindVL;
    // { using type = mask_logical_traits<W,VL2>; };

    template<unsigned short VL2>
    using rebindVL_t = typename rebindVL<VL2>::type;

    template<unsigned short W2>
    struct rebindW {
	using type = mask_logical_traits<W2,VL>;
    };

    template<typename B> // scalar only
    static constexpr auto get_val( B b ) { return logical<W>::get_val( b ); }
};

template<unsigned short B_, unsigned short VL_>
struct mask_bit_logical_traits {
    static constexpr unsigned short W = 0;
    static constexpr unsigned short B = B_;
    static constexpr unsigned short VL = VL_;
    using member_type = bitfield<B>;
    using element_type = bitfield<B>;
    using traits = vector_type_traits_vl<element_type, VL>;
    using type = typename traits::type;
    using pointer_type = typename traits::pointer_type;
    using tag_type = std::conditional_t<B==1,target::mt_mask,target::mt_vmask>;
    using index_type = VID; // something sensible

    using prefmask_traits = detail::mask_bit_logical_traits<B, VL>;

    template<typename U>
    struct rebindTy {
	using type = vdata_traits<U,VL>;
    };

    template<unsigned short VL2>
    struct rebindVL;
    // {
    // using type = mask_bit_logical_traits<B,VL2>;
    // };

    template<unsigned short VL2>
    using rebindVL_t = typename rebindVL<VL2>::type;

    template<unsigned short B2>
    struct rebindW {
	using type = mask_bit_logical_traits<B2,VL>;
    };

    template<typename Bl> // scalar only
    static constexpr auto get_val( Bl b ) { return bitfield<B>::get_val( b ); }
};

// Bit mask
template<unsigned short VL_>
struct mask_bit_traits {
    static constexpr unsigned short W = 0;
    static constexpr unsigned short B = 1;
    static constexpr unsigned short VL = VL_;
    using member_type = bitfield<1>; // no member_type present
    using element_type = bitfield<1>;
    using traits = mask_type_traits<VL>;
    using type = typename traits::type;
    using pointer_type = unsigned char; // type; // Pointing to a vector!
    using tag_type = target::mt_mask;
    using index_type = VID; // Preferred index type, likely vector width

    using prefmask_traits = detail::mask_bit_traits<VL>;

    template<unsigned short VL2>
    struct rebindVL;

    template<unsigned short VL2>
    using rebindVL_t = typename rebindVL<VL2>::type;

    template<typename B> // scalar only
    static constexpr auto get_val( B b ) { return !!b; }
};

// template<unsigned short VL_>
// using mask_bit_traits = mask_bit_logical_traits<1,VL_>;

// Bool value, only at vector length 1 (scalar)
struct mask_bool_traits {
    static constexpr unsigned short W = 1;
    static constexpr unsigned short B = sizeof(bool)*8;
    static constexpr unsigned short VL = 1;
    using member_type = bool;
    using element_type = member_type;
    using traits = mask_type_traits<VL>;
    using type = typename traits::type;
    using pointer_type = member_type;
    using tag_type = target::mt_bool;
    using index_type = VID; // something sensible

    using prefmask_traits = detail::mask_bool_traits;

    template<unsigned short VL2>
    struct rebindVL; // defined below

    template<unsigned short VL2>
    using rebindVL_t = typename rebindVL<VL2>::type;

    template<typename B> // scalar only
    static constexpr auto get_val( B b ) { return !!b; }
};

/***********************************************************************
 * mask traits type checking
 ***********************************************************************/
template<typename T>
struct is_mask_logical_traits : public std::false_type { };

template<unsigned short W, unsigned short VL>
struct is_mask_logical_traits<mask_logical_traits<W,VL>>
    : public std::true_type { };

template<typename T>
constexpr bool is_mask_logical_traits_v = is_mask_logical_traits<T>::value;

template<typename T>
struct is_mask_bit_logical_traits : public std::false_type { };

template<unsigned short B, unsigned short VL>
struct is_mask_bit_logical_traits<mask_bit_logical_traits<B,VL>>
    : public std::true_type { };

template<typename T>
struct is_mask_bit_traits : public std::false_type { };

template<unsigned short VL>
struct is_mask_bit_traits<mask_bit_traits<VL>>
    : public std::true_type { };

template<typename T>
struct is_mask_bool_traits : public std::false_type { };

template<>
struct is_mask_bool_traits<mask_bool_traits> : public std::true_type { };

template<typename T>
struct is_mask_traits {
    static constexpr bool value =
	is_mask_logical_traits<T>::value
	|| is_mask_bit_logical_traits<T>::value
	|| is_mask_bool_traits<T>::value
	|| is_mask_bit_traits<T>::value;
};

template<typename T>
constexpr bool is_mask_traits_v = is_mask_traits<T>::value;

template<typename T>
struct is_vm_traits : public std::false_type { };

template<unsigned short W, unsigned short VL>
struct is_vm_traits<mask_logical_traits<W,VL>> : public std::true_type { };

template<unsigned short B, unsigned short VL>
struct is_vm_traits<mask_bit_logical_traits<B,VL>> : public std::true_type { };

template<unsigned short VL>
struct is_vm_traits<mask_bit_traits<VL>> : public std::true_type { };

template<>
struct is_vm_traits<mask_bool_traits> : public std::true_type { };

template<typename T, unsigned short VL>
struct is_vm_traits<vdata_traits<T,VL>> : public std::true_type { };

template<typename T>
constexpr bool is_vm_traits_v = is_vm_traits<T>::value;

/***********************************************************************
 * logical/bit mask traits
 ***********************************************************************/
template<unsigned short W, unsigned short VL, typename Enable = void>
struct mask_traits : public mask_logical_traits<W, VL> { };

template<unsigned short VL>
struct mask_traits<0,VL> : public mask_bit_traits<VL> { };

/***********************************************************************
 * selecting preferred mask type for a vector
 ***********************************************************************/
template<unsigned short W, unsigned short VL, typename Enable = void>
struct mask_sel {
    using type = mask_logical_traits<W, VL>;
};

// Don't create too wide masks. Cut off at vector lanes of 8 bytes.
// Anything larger relates to longint and it becomes way more efficient
// to use bitmasks. This sets a preference but does not preclude wide masks.
/*
*/
template<unsigned short W, unsigned short VL>
struct mask_sel<W,VL,std::enable_if_t<(W>8) && (VL>1)>> {
    using type = mask_bit_traits<VL>;
};

template<unsigned short W>
struct mask_sel<W,1,std::enable_if_t<W!=0>> {
    using type = mask_bool_traits;
};

template<unsigned short VL>
struct mask_sel<0,VL> {
    using type = mask_bit_traits<VL>;
};

#if __AVX512F__
template<unsigned short W, unsigned short VL>
struct mask_sel<W,VL,typename std::enable_if<(W*VL*8>=512) || (W == 1 && VL != 1)>::type> {
    using type = mask_bit_traits<VL>;
};
#endif // __AVX512F__

#if __AVX512VL__
template<unsigned short W, unsigned short VL>
struct mask_sel<W,VL,typename std::enable_if<(W*VL*8==256) && !((W*VL*8>=512) || (W == 1))>::type> {
    using type = mask_bit_traits<VL>;
};
#endif // __AVX512VL__

#if __AVX512VL__ && __AVX512BW__
template<unsigned short W, unsigned short VL>
struct mask_sel<W,VL,typename std::enable_if<(W*VL*8==128) && !((W*VL*8>=512) || (W == 1))>::type> {
    using type = mask_bit_traits<VL>;
};
#endif // __AVX512VL__ && __AVX512BW__

// bitfields
template<unsigned short Bits, unsigned short VL, typename Enable = void>
struct bitfield_mask_sel {
    using type = mask_bit_logical_traits<Bits, VL>;
};

template<unsigned short Bits>
struct bitfield_mask_sel<Bits,1> {
    using type = mask_bool_traits;
};


template<unsigned short Bits, unsigned short VL>
using bitfield_mask_preferred_traits =
    typename bitfield_mask_sel<Bits,VL>::type;

template<typename T, unsigned short VL>
struct mask_preferred_traits_helper {
    using type = typename mask_sel<sizeof(T),VL>::type;
};

template<unsigned short Bits, unsigned short VL>
struct mask_preferred_traits_helper<bitfield<Bits>,VL> {
    using type = typename bitfield_mask_sel<Bits,VL>::type;
};

template<typename T, unsigned short VL>
using mask_preferred_traits_type =
    typename mask_preferred_traits_helper<T,VL>::type;

template<unsigned short W, unsigned short VL>
using mask_preferred_traits_width = typename mask_sel<W,VL>::type;

// Stop using this
// template<unsigned short W, unsigned short VL>
// using mask_preferred_traits = mask_preferred_traits_width<W,VL>;

} // namespace detail

template<typename T, unsigned short VL>
using type_sel = detail::mask_preferred_traits_type<T, VL>;

namespace detail {

// When extending a bool at VL=1 to a wider mask, we don't have much information
// to go on to select between a vector mask and a bit mask. Use the preferred
// version.
// Problem: when working with bitfields, this resorts to 1-bit masks rather
// than B-bit bit_logical masks. For that reason, should avoid its use.
template<unsigned short VL2>
struct mask_bool_traits::rebindVL {
    using type = mask_preferred_traits_width<sizeof(VID),VL2>;
};

// When shortening a bitmask to VL=1, switch to bool.
template<unsigned short VL>
struct mask_bit_traits_rebindVL {
    using type = mask_bit_traits<VL>;
};

template<>
struct mask_bit_traits_rebindVL<1> {
    using type = mask_bool_traits;
};

template<unsigned short VL_>
template<unsigned short VL2>
struct mask_bit_traits<VL_>::rebindVL {
    using type = typename mask_bit_traits_rebindVL<VL2>::type;
};

template<unsigned short B, unsigned short VL>
struct mask_bit_logical_traits_rebindVL {
    using type = mask_bit_logical_traits<B,VL>;
};

template<unsigned short B>
struct mask_bit_logical_traits_rebindVL<B,1> {
    using type = mask_bool_traits;
};

template<unsigned short B, unsigned short VL_>
template<unsigned short VL2>
struct mask_bit_logical_traits<B,VL_>::rebindVL {
    using type = typename mask_bit_logical_traits_rebindVL<B,VL2>::type;
};

template<unsigned short B, unsigned short VL>
struct mask_logical_traits_rebindVL {
    using type = mask_logical_traits<B,VL>;
};

template<unsigned short B>
struct mask_logical_traits_rebindVL<B,1> {
    using type = mask_bool_traits;
};

template<unsigned short B, unsigned short VL_>
template<unsigned short VL2>
struct mask_logical_traits<B,VL_>::rebindVL {
    using type = typename mask_logical_traits_rebindVL<B,VL2>::type;
};

/***********************************************************************
 * vector configurations
 ***********************************************************************/
template<typename T, unsigned short VL_>
struct vdata_traits {
    static_assert( !std::is_void<T>::value,
		   "vdata_traits not applicable to void type" );

    static constexpr unsigned short VL = VL_;
    static constexpr unsigned short B = sizeof(T)*8;
    using member_type = T;
    using element_type = member_type;
    static constexpr unsigned short W = sizeof(member_type);
    using traits = vector_type_traits_vl<member_type, VL>;
    using type = typename traits::type;
    using pointer_type = member_type;
    using index_type = index_type_of_size<sizeof(member_type)>;

    // Copied vector_impl/vec
    using prefmask_traits = detail::mask_preferred_traits_type<T, VL>;
    using simd_mask_type = mask_impl<prefmask_traits>;
    using tag_type = typename prefmask_traits::tag_type;

    template<typename U>
    struct rebindTy {
	using type = vdata_traits<U,VL>;
    };

    template<unsigned short VL2>
    struct rebindVL {
	using type = vdata_traits<member_type,VL2>;
    };

    template<unsigned short VL2>
    using rebindVL_t = typename rebindVL<VL2>::type;
};

template<unsigned short Bits, unsigned short VL_>
struct vdata_traits<bitfield<Bits>,VL_> {
    static constexpr unsigned short VL = VL_;
    static constexpr unsigned short B = Bits;
    using member_type = bitfield<Bits>;
    using element_type = bitfield<Bits>;
    static constexpr unsigned short W = 0; // ???
    using traits = vector_type_traits_vl<element_type, VL>;
    using type = typename traits::type;
    using pointer_type = typename traits::pointer_type;
    using index_type = VID; // something sensible

    // Copied vector_impl/vec
    using prefmask_traits =
	detail::mask_preferred_traits_type<element_type, VL>;
    using simd_mask_type = mask_impl<prefmask_traits>;
    using tag_type = typename prefmask_traits::tag_type;

    static_assert( prefmask_traits::B == Bits || VL == 1, "check" );

    template<typename U>
    struct rebindTy {
	using type = vdata_traits<U,VL>;
    };

    template<unsigned short VL2>
    struct rebindVL {
	using type = vdata_traits<element_type,VL2>;
    };

    template<unsigned short VL2>
    using rebindVL_t = typename rebindVL<VL2>::type;
};

} // namespace detail

/***********************************************************************
 * quick access to the data type/traits
 ***********************************************************************/
namespace detail {
template<typename T, unsigned short VL>
struct sel_ty {
    using type = vdata_traits<T,VL>;
};

template<unsigned short VL>
struct sel_ty<void,VL> {
    using type = mask_bit_traits<VL>;
};

} // namespace detail

template<typename T, unsigned short VL>
using ty = typename detail::sel_ty<T,VL>::type;

/***********************************************************************
 * testing for vector type
 ***********************************************************************/

template<typename T>
struct is_vector : std::false_type { };

// template<typename Traits>
// struct is_vector<detail::vector_impl<Traits>> : std::true_type { };

template<typename Traits, layout_t Layout>
struct is_vector<detail::vec<Traits,Layout>> : std::true_type { };

/***********************************************************************
 * selecting between vector and mask
 ***********************************************************************/

namespace detail {

template<typename VTr>
struct container_sel {
    using type = vec<VTr,simd::lo_unknown>; // it's mask, so no relevance
};

template<unsigned short VL>
struct container_sel<mask_bit_traits<VL>> {
    using type = mask_impl<mask_bit_traits<VL>>;
};

} // namespace detail

template<typename VTr>
using container = typename detail::container_sel<VTr>::type;

/***********************************************************************
 * testing for matching data types/traits
 ***********************************************************************/

template<typename VTr, unsigned short VL, typename Enable = void>
struct matchVLtu;

template<typename VTr, unsigned short VL>
struct matchVLtu<VTr,VL,
		 typename std::enable_if<!std::is_void<VTr>::value>::type> {
    static constexpr bool value = VTr::VL == VL;
};

template<typename VTr, typename MTr, unsigned short VL, typename Enable = void>
struct matchVLttu;

template<typename VTr, typename MTr, unsigned short VL>
struct matchVLttu<VTr,MTr,VL,
		  typename std::enable_if<!std::is_void<VTr>::value
					  && !std::is_void<MTr>::value>::type> {
    static constexpr bool value = VTr::VL == VL && MTr::VL == VL;
};

template<typename VTr, typename MTr1, typename MTr2, unsigned short VL>
struct matchVLtttu {
    static constexpr bool value =
	VTr::VL == VL && MTr1::VL == VL && MTr2::VL == VL;
};

template<typename VTr, typename MTr1, typename MTr2, unsigned short VL,
	 typename Enable = void>
struct matchVLttotu;

template<typename VTr, typename MTr1, typename MTr2, unsigned short VL>
struct matchVLttotu<VTr,MTr1,MTr2,VL,
		    typename std::enable_if<!std::is_void<MTr1>::value
					    && !std::is_void<MTr2>::value>::type> {
    static constexpr bool value =
	VTr::VL == VL && MTr1::VL == VL && MTr2::VL == VL;
};

template<typename VTr, typename MTr1, unsigned short VL>
struct matchVLttotu<VTr,MTr1,void,VL,
		    typename std::enable_if<!std::is_void<MTr1>::value>::type> {
    static constexpr bool value = VTr::VL == VL && MTr1::VL == VL;
};

template<typename VTr, typename MTr2, unsigned short VL>
struct matchVLttotu<VTr,void,MTr2,VL,
		    typename std::enable_if<!std::is_void<MTr2>::value>::type> {
    static constexpr bool value = VTr::VL == VL && MTr2::VL == VL;
};

template<typename VTr, typename MTr1, typename MTr2, typename Enable = void>
struct matchVLttt;

template<typename VTr, typename MTr1, typename MTr2>
struct matchVLttt<VTr,MTr1,MTr2,
		  typename std::enable_if<!std::is_void<VTr>::value
					  && !std::is_void<MTr1>::value
					  && !std::is_void<MTr2>::value
					  >::type> {
    static constexpr bool value = MTr1::VL == VTr::VL && MTr2::VL == VTr::VL;
};

template<typename VTr, typename MTr1, typename MTr2, typename Enable = void>
struct matchVLttot;

template<typename VTr, typename MTr1, typename MTr2>
struct matchVLttot<VTr,MTr1,MTr2,
		  typename std::enable_if<!std::is_void<VTr>::value
					  && !std::is_void<MTr1>::value
					  && !std::is_void<MTr2>::value
					  >::type> {
    static constexpr bool value = MTr1::VL == VTr::VL && MTr2::VL == VTr::VL;
};

template<typename VTr, typename MTr1>
struct matchVLttot<VTr,MTr1,void,
		  typename std::enable_if<!std::is_void<VTr>::value
					  && !std::is_void<MTr1>::value
					  >::type> {
    static constexpr bool value = MTr1::VL == VTr::VL;
};

template<typename VTr, typename MTr2>
struct matchVLttot<VTr,void,MTr2,
		  typename std::enable_if<!std::is_void<VTr>::value
					  && !std::is_void<MTr2>::value
					  >::type> {
    static constexpr bool value = MTr2::VL == VTr::VL;
};

template<typename VTr>
struct matchVLttot<VTr,void,void,
		  typename std::enable_if<!std::is_void<VTr>::value
					  >::type> : public std::true_type { };

template<typename VTr, typename MTr>
struct matchVLtt {
    static constexpr bool value = VTr::VL == MTr::VL;
};

template<typename VTr>
struct matchVLtt<VTr,void> : public std::false_type { };

template<typename MTr>
struct matchVLtt<void,MTr> : public std::false_type { };

// To make things easier: if not void, must match VL
template<typename VTr, unsigned short VL>
struct matchVL_helper {
    static constexpr bool value = VTr::VL == VL;
};

template<unsigned short VL> // this differs from matchVLtu
struct matchVL_helper<void,VL> : public std::true_type { };

template<unsigned short VL, typename... Trs>
struct matchVL_;

template<unsigned short VL>
struct matchVL_<VL> : public std::true_type { };

template<unsigned short VL, typename Tr>
struct matchVL_<VL,Tr> : public matchVL_helper<Tr,VL> { };

template<unsigned short VL, typename Tr, typename... Trs>
struct matchVL_<VL,Tr,Trs...> {
    static constexpr bool value = matchVL_<VL,Tr>::value
	&& matchVL_<VL,Trs...>::value;
};

template<typename... Trs>
struct matchVL;

template<>
struct matchVL<> : public std::true_type { };

template<typename Tr, typename... Trs>
struct matchVL<Tr,Trs...> : public matchVL_<Tr::VL,Trs...> { };

template<typename... Trs>
struct matchVL<void,Trs...> : public matchVL<Trs...> { };


/***********************************************************************
 * Enable_if variations
 ***********************************************************************/
template<typename Tr>
using enable_if_scalar_t = std::enable_if_t<Tr::VL == 1, nullptr_t>;

template<typename Tr>
using enable_if_not_scalar_t = std::enable_if_t<Tr::VL != 1, nullptr_t>;


} // namespace simd

#endif // GRAPTOR_SIMD_DECL_H
