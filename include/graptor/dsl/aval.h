// -*- C++ -*-
#ifndef GRAPTOR_DSL_AVAL_H
#define GRAPTOR_DSL_AVAL_H

#include "graptor/simd/decl.h"

namespace expr {

// Pre-declarations for r-value and l-value classes
template<typename VTr, layout_t Layout, typename MTr, typename Select = void>
class rvalue;
template<typename VTr, typename I, typename MTr, typename Enc, bool NT,
	 layout_t Layout>
class lvalue;

// Templates to check is-a properties
template<typename T>
struct is_rvalue : std::false_type { };

template<typename VTr, layout_t Layout, typename MTr>
struct is_rvalue<rvalue<VTr,Layout,MTr>> : std::true_type { };

template<typename T>
struct is_lvalue : std::false_type { };

template<typename VTr, typename I, typename MTr, typename Enc, bool NT,
	 layout_t Layout>
struct is_lvalue<lvalue<VTr,I,MTr,Enc,NT,Layout>> : std::true_type { };

template<typename T>
struct is_logical_type : std::false_type { };

template<unsigned short W>
struct is_logical_type<logical<W>> : std::true_type { };

template<>
struct is_logical_type<bool> : std::true_type { };

// Class definitions
// Version with both vector and mask
template<typename VTr, layout_t Layout, typename MTr>
class rvalue<VTr,Layout,MTr,
	     typename std::enable_if<VTr::VL == MTr::VL>::type> {
public:
    static constexpr unsigned short VL = VTr::VL;
    static constexpr unsigned short W = MTr::W;
    //typedef simd::detail::vector_impl<VTr> value_type;
    using value_type = simd::vec<VTr,Layout>;
    using mask_type = simd::detail::mask_impl<MTr>;
    using value_traits = VTr;
    using mask_traits = MTr;

    static_assert( sizeof(typename VTr::member_type) == W // matching vectors
		   || ( W == 1 && VL == 1 ) // scalar bool
		   || ( sizeof(typename VTr::member_type) == 8 && W == 4 ) // exception which should be harmless ...
		   || ( sizeof(typename VTr::member_type) == 4 && W == 8 ) // exception which should be harmless ...
		   || W == 0 // bit mask
		   || VTr::W > 8, // longint
		   "simplification" );

    rvalue( const value_type & v, const mask_type & m )
	: m_val( v ), m_mask( m ) { }
    rvalue( value_type && v, mask_type && m )
	: m_val( std::forward<value_type>( v ) ),
	  m_mask( std::forward<mask_type>( m ) ) { }

    value_type value() const { return m_val; }
    mask_type mask() const { return m_mask; }

private:
    value_type m_val;
    mask_type m_mask;
};

// Version with mask but no data
template<layout_t Layout,typename MTr>
class rvalue<void,Layout,MTr,
	     typename std::enable_if<!std::is_void<MTr>::value>::type> {
public:
    static constexpr unsigned short VL = MTr::VL;
    static constexpr unsigned short W = MTr::W;
    typedef simd::detail::mask_impl<MTr> mask_type;
    using value_traits = void;
    using mask_traits = MTr;

    static_assert( Layout == simd::lo_variable || Layout == simd::lo_unknown,
		   "in future, only lo_unknown should be used on mask-only "
		   "rvalues" );

    rvalue( mask_type m ) : m_mask( m ) { }
    rvalue( mask_type && m ) : m_mask( std::forward<mask_type>( m ) ) { }

    mask_type mask() const { return m_mask; }

private:
    mask_type m_mask;
};

// Version with vector but no mask
template<typename VTr, layout_t Layout>
class rvalue<VTr,Layout,void,
	     typename std::enable_if<!std::is_void<VTr>::value>::type> {
public:
    static constexpr unsigned short VL = VTr::VL;
    // static constexpr unsigned short W = 0; -- 0 means multiple different things
    using value_type = simd::vec<VTr,Layout>;
    using mask_type = nomask<VL>;
    using value_traits = VTr;
    using mask_traits = void;

    rvalue( const value_type & v ) : m_val( v ) { }
    rvalue( value_type && v ) : m_val( std::forward<value_type>( v ) ) { }

    value_type value() const { return m_val; }
    mask_type mask() const { return nomask<VL>(); }
private:
    value_type m_val;
};

/***********************************************************************
 * Utilities for making rvalues. Simplify specification of types and
 * perform a few sanity checks.
 * These methods apply on the new simd::vec<> types.
 ***********************************************************************/

template<typename VTr, layout_t Layout, typename MTr>
__attribute__((always_inline))
inline auto
make_rvalue( const simd::vec<VTr,Layout> & v,
	     const simd::detail::mask_impl<MTr> & m,
	     typename std::enable_if<VTr::VL == MTr::VL>::type * = nullptr ) {
    return rvalue<VTr,Layout,MTr>( v, m );
}

template<typename VTr, layout_t Layout, typename MTr>
__attribute__((always_inline))
inline auto
make_rvalue( simd::vec<VTr,Layout> && v,
	     simd::detail::mask_impl<MTr> && m,
	     typename std::enable_if<VTr::VL == MTr::VL>::type * = nullptr ) {
    return rvalue<VTr,Layout,MTr>(
	std::forward<simd::vec<VTr,Layout>>( v ),
	std::forward<simd::detail::mask_impl<MTr>>( m ) );
}

template<typename VTr,layout_t Layout>
__attribute__((always_inline))
inline auto
make_rvalue( const simd::vec<VTr,Layout> & v ) {
    return rvalue<VTr,Layout,void>( v );
}

template<typename VTr,layout_t Layout>
__attribute__((always_inline))
inline auto
make_rvalue( simd::vec<VTr,Layout> && v ) {
    return rvalue<VTr,Layout,void>( std::forward<simd::vec<VTr,Layout>>( v ) );
}

template<typename MTr>
__attribute__((always_inline))
inline auto
make_rvalue( simd::detail::mask_impl<MTr> m ) {
    return rvalue<void,lo_unknown,MTr>( m );
}

template<typename VTr, layout_t Layout>
__attribute__((always_inline))
inline auto
make_rvalue( simd::vec<VTr,Layout> v, nomask<VTr::VL> ) {
    return rvalue<VTr,Layout, void>( v );
}

template<typename MTr>
__attribute__((always_inline))
inline auto
make_rvalue( simd::detail::mask_impl<MTr> m, nomask<MTr::VL> ) {
    return rvalue<void,lo_unknown,MTr>( m );
}

/***********************************************************************
 * Utilities for making rvalues. Simplify specification of types and
 * perform a few sanity checks.
 * These methods apply on the old simd::detail::vector_impl<> types and
 * are to be phased out.
 ***********************************************************************/

#if 0
template<typename VTr, typename MTr>
__attribute__((always_inline))
inline auto
make_rvalue( simd::detail::vector_impl<VTr> v,
	     const simd::detail::mask_impl<MTr> & m,
	     typename std::enable_if<VTr::VL == MTr::VL>::type * = nullptr ) {
    return rvalue<VTr,simd::lo_variable,MTr>( v, m );
}

template<typename VTr>
__attribute__((always_inline))
inline auto
make_rvalue( simd::detail::vector_impl<VTr> v ) {
    return rvalue<VTr,simd::lo_variable,void>( v );
}

template<typename VTr>
__attribute__((always_inline))
inline auto
make_rvalue( simd::detail::vector_impl<VTr> v, nomask<VTr::VL> ) {
    return rvalue<VTr,simd::lo_variable, void>( v );
}
#endif

/***********************************************************************
 * Utilities for merging value-only and mask-only rvalues.
 ***********************************************************************/

template<typename VTr,layout_t Layout,typename MTr>
__attribute__((always_inline))
inline auto
merge_rvalues( rvalue<VTr,Layout,void> v, const rvalue<void,lo_unknown,MTr> & m,
	       typename std::enable_if<VTr::VL==MTr::VL>::type * = nullptr ) {
    return make_rvalue( v.value(), m.mask() );
}

template<typename MTr>
typename std::enable_if<simd::detail::is_mask_logical_traits<MTr>::value && MTr::VL == 1, bool>::type
    is_true( rvalue<void,lo_unknown,MTr> r ) { return r.mask().get() != 0; }

inline bool is_true( rvalue<void,lo_unknown,simd::detail::mask_bool_traits> r ) {
    return r.mask().get();
}

template<typename VTr,layout_t Layout>
typename std::enable_if<!std::is_void<VTr>::value && VTr::VL == 1, bool>::type
is_true( rvalue<VTr,Layout,void> r ) {
    return r.value().at(0) != typename VTr::element_type(0);
}

template<typename MTr, layout_t Layout>
bool is_false( rvalue<void,Layout,MTr> r ) {
    return r.mask().is_all_false();
}

template<typename VTr, layout_t Layout>
typename std::enable_if<!std::is_void<VTr>::value && VTr::VL == 1, bool>::type
is_false( rvalue<VTr,Layout,void> r ) {
    return r.value().at(0) == typename VTr::element_type(0);
}

template<typename VTr, typename I, typename MTr, typename Enc, bool NT,
	 layout_t Layout>
class lvalue {
    static_assert( VTr::VL == MTr::VL, "need matching vector length" );
public:
    static constexpr unsigned short VL = VTr::VL;
    static constexpr layout_t layout = Layout;
    typedef simd::detail::vector_ref_impl<VTr,I,Enc,NT,layout> value_type;
    typedef simd::detail::mask_impl<MTr> mask_type;
    
    lvalue( value_type v, mask_type m ) : m_val( v ), m_mask( m ) { }

    value_type value() const { return m_val; }
    mask_type mask() const { return m_mask; }
private:
    value_type m_val;
    mask_type m_mask;
};

template<typename VTr, typename I, typename Enc, bool NT, layout_t Layout>
class lvalue<VTr,I,void,Enc,NT,Layout> {
public:
    static constexpr unsigned short VL = VTr::VL;
    static constexpr layout_t layout = Layout;
    typedef simd::detail::vector_ref_impl<VTr,I,Enc,NT,layout> value_type;
    typedef nomask<VL> mask_type;
    
    lvalue( value_type v ) : m_val( v ) { }

    value_type value() const { return m_val; }
    mask_type mask() const { return nomask<VL>(); }
private:
    value_type m_val;
};

template<typename VTr, typename MTr, typename I, typename Enc, bool NT,
	 layout_t Layout>
__attribute__((always_inline))
inline auto
make_lvalue( simd::detail::vector_ref_impl<VTr,I,Enc,NT,Layout> v,
	     simd::detail::mask_impl<MTr> m,
	     typename std::enable_if<VTr::VL == MTr::VL>::type * = nullptr ) {
    return lvalue<VTr,I,MTr,Enc,NT,Layout>( v, m );
}

template<typename VTr, typename I, typename Enc, bool NT,
	 layout_t Layout>
__attribute__((always_inline))
inline auto
make_lvalue( simd::detail::vector_ref_impl<VTr,I,Enc,NT,Layout> v ) {
    return lvalue<VTr,I,void,Enc,NT,Layout>( v );
}


template<typename VTr>
auto join_mask( simd::nomask<VTr::VL> l,
		simd::nomask<VTr::VL> r ) {
    return l;
}

template<typename VTr, typename MTr>
auto join_mask( simd::detail::mask_impl<MTr> l,
		simd::nomask<VTr::VL> r ) {
    return l;
}

template<typename VTr, typename MTr>
auto join_mask( simd::nomask<VTr::VL> l,
		simd::detail::mask_impl<MTr> r ) {
    return r;
}

template<typename VTr, typename MTr1, typename MTr2>
auto join_mask( simd::detail::mask_impl<MTr1> l,
		simd::detail::mask_impl<MTr2> r ) {
    static_assert( VTr::VL == MTr1::VL && VTr::VL == MTr2::VL,
		   "vector lengths of masks must match with target data type" );

    using MTr = typename VTr::prefmask_traits;
    
    if constexpr ( MTr1::W == MTr2::W )
	return l & r; // should we convert to width of VTr::W?
    else if constexpr ( MTr::W == MTr1::W )
	return l & r.template convert<MTr>();
    else if constexpr ( MTr::W == MTr2::W )
	return l.template convert<MTr>() & r;
    else
	return l.template convert<MTr>() & r.template convert<MTr>();
}

template<typename VTr, typename MTr1>
auto force_mask( simd::detail::mask_impl<MTr1> m ) {
    using MTr = typename VTr::prefmask_traits;

    if constexpr ( std::is_same_v<MTr,MTr1> )
	return m;
    else if constexpr ( simd::detail::is_mask_bit_traits<MTr>::value
			&& simd::detail::is_mask_logical_traits<MTr1>::value
			&& MTr1::B == VTr::B )
	return m;
    else
	return m.template convert<MTr>();
}

template<typename VTr, unsigned short VL>
auto force_mask( simd::nomask<VL> m ) {
    return m;
}

/**=====================================================================
 * New rvalue declarations using a state-based (sb) approach for
 * managing the mask
 *======================================================================*/
namespace sb {

/**=====================================================================
 * A mask pack maintains multiple equivalent masks in different formats
 *
 * The mask pack cannot guarantee that all masks represent the same value.
 * It ensures no two masks stored in the pack have the same type.
 *======================================================================*/
namespace detail {
template<int Idx, typename Tr, typename Tuple, typename Enable = void>
struct get_matching_index;

template<int Idx, typename Tr>
struct get_matching_index<Idx,Tr,std::tuple<>> {
    static constexpr int value = -1;
};

template<int Idx, typename Tr, typename... MTr>
struct get_matching_index<Idx,Tr,std::tuple<Tr,MTr...>> {
    static constexpr int value = Idx;
};

template<int Idx, typename Tr, typename MTr0, typename... MTr>
struct get_matching_index<Idx,Tr,std::tuple<MTr0,MTr...>,
			  std::enable_if_t<!std::is_same_v<Tr,MTr0>>> {
    static constexpr int value =
	get_matching_index<Idx+1,Tr,std::tuple<MTr...>>::value;
};

} // namespace detail

template<typename... MTr>
class mask_pack;

template<typename... MTr>
class mask_pack {
public:
    using self_type = mask_pack<MTr...>;
    using tuple_type = std::tuple<simd::detail::mask_impl<MTr>...>;
    static constexpr size_t num_masks = sizeof...( MTr );

public:
    mask_pack( self_type && p )
	: m_masks( std::forward<tuple_type>( p.m_masks ) ) { }
    mask_pack( const self_type & p ) : m_masks( p.m_masks ) { }
    mask_pack( tuple_type && t )
	: m_masks( std::forward<tuple_type>( t ) ) { }
    mask_pack( const tuple_type & t )
	: m_masks( t ) { }
    template<typename U = self_type>
    mask_pack( std::enable_if_t<U::num_masks == 0> * = nullptr ) { }

    static constexpr bool is_empty() {
	return sizeof...( MTr ) == 0;
    }
    
    template<typename Tr>
    static constexpr bool has_mask() {
	return sizeof...( MTr ) > 0
	    && detail::get_matching_index<0,Tr,std::tuple<MTr...>>::value >= 0;
    }
    
    template<typename Tr>
    auto get_mask() const {
	if constexpr ( sizeof...( MTr ) == 0 )
	    return simd::nomask<Tr::VL>();
	else {
	    static constexpr int idx =
		detail::get_matching_index<0,Tr,std::tuple<MTr...>>::value;
	    if constexpr ( idx >= 0 ) {
		return std::get<idx>( m_masks );
	    } else if constexpr ( simd::detail::is_mask_traits_v<Tr> ) {
		// Pick any of the masks and convert to desired type
		return std::get<0>( m_masks ).template convert_data_type<Tr>();
	    } else {
		// Pick any of the masks and convert to desired type
		using UTr = typename Tr::prefmask_traits;
		return std::get<0>( m_masks ).template convert_data_type<UTr>();
	    }
	}
    }

    template<typename Tr>
    auto get_mask_for( const simd::detail::mask_impl<Tr> & ) const {
	return get_mask<Tr>();
    }
    template<typename Tr, simd::layout_t Layout>
    auto get_mask_for( const simd::detail::vec<Tr,Layout> & ) const {
	using MTr1 = typename Tr::prefmask_traits;
	return get_mask<MTr1>();
    }

    template<typename Tr>
    auto clone_and_add() const {
	if constexpr ( num_masks == 0 ) {
	    return *this;
	} else {
	    using ATr = std::conditional_t<
		simd::detail::is_mask_traits_v<Tr>,
		Tr,
		typename Tr::prefmask_traits>;
	    static constexpr int idx =
		detail::get_matching_index<0,ATr,std::tuple<MTr...>>::value;
	    if constexpr ( idx >= 0 )
		return *this;
	    else {
		// There is an option to choose the most suitable mask to
		// convert from
		auto m = force_mask<ATr>( std::get<0>( m_masks ) );
		auto p = std::tuple_cat( m_masks, std::make_tuple( m ) );
		return mask_pack<MTr...,ATr>( p );
	    }
	}
    }

    template<typename Tr>
    auto get_any() const {
	if constexpr ( sizeof...( MTr ) == 0 )
	    return simd::nomask<Tr::VL>();
	else
	    // TODO: find the mask that is most easily converted to Tr
	    return std::get<0>( m_masks ); // TEMP
    }

private:
    tuple_type m_masks;
};

template<typename MTr>
auto create_mask_pack( simd::detail::mask_impl<MTr> m ) {
    return mask_pack<MTr>( m );
}

inline auto create_mask_pack() {
    return mask_pack<>();
}

template<typename T>
struct is_mask_pack : public std::false_type { };

template<typename... MTr>
struct is_mask_pack<mask_pack<MTr...>> : public std::true_type { };

template<typename T>
constexpr bool is_mask_pack_v = is_mask_pack<T>::value;

template<typename T>
struct is_empty_mask_pack : public std::false_type { };

template<>
struct is_empty_mask_pack<mask_pack<>> : public std::true_type { };

template<typename T>
constexpr bool is_empty_mask_pack_v = is_empty_mask_pack<T>::value;

// Pre-declarations for r-value and l-value classes
template<typename VTr, layout_t Layout, typename... MP>
class rvalue;

template<typename VTr, typename I, typename Enc, bool NT, layout_t Layout,
	 typename... MP>
class lvalue;

/**=====================================================================
 * rvalue holding a vector or mask and a mask pack
 *======================================================================*/
template<typename VTr, layout_t Layout, typename... MP>
class rvalue {
public:
    static constexpr unsigned short VL = VTr::VL;
    using value_type =
	std::conditional_t<simd::detail::is_mask_traits_v<VTr>,
			   simd::detail::mask_impl<VTr>,
			   simd::vec<VTr,Layout>>;
    using data_type = VTr;
    using mask_pack_type = mask_pack<MP...>;

    rvalue( const value_type & v, const mask_pack_type & m )
	: m_val( v ), m_mpack( m ) { }
    rvalue( value_type && v, mask_pack_type && m )
	: m_val( std::forward<value_type>( v ) ),
	  m_mpack( std::forward<mask_pack_type>( m ) ) { }

    rvalue<VTr,Layout> uvalue() const {
	return rvalue<VTr,Layout>( m_val, std::move( mask_pack<>() ) );
    }
    value_type value() const { return m_val; }
    mask_pack_type mpack() const { return m_mpack; }

    constexpr bool has_mask() const { return sizeof...( MP ) != 0; }

private:
    value_type m_val;
    mask_pack_type m_mpack;
};

/**=====================================================================
 * lvalue holding a vector_ref and a mask pack
 *======================================================================*/
template<typename VTr, typename I, typename Enc, bool NT, layout_t Layout,
	 typename... MP>
class lvalue {
public:
    using value_type = simd::detail::vector_ref_impl<VTr,I,Enc,NT,Layout>;
    using mask_pack_type = mask_pack<MP...>;
    
    lvalue( value_type v, mask_pack_type m ) : m_val( v ), m_mpack( m ) { }

    lvalue<VTr,I,Enc,NT,Layout> uvalue() const {
	return lvalue<VTr,I,Enc,NT,Layout>( m_val, mask_pack<>() );
    }
    value_type value() const { return m_val; }
    mask_pack_type mpack() const { return m_mpack; }

    constexpr bool has_mask() const { return sizeof...( MP ) != 0; }

private:
    value_type m_val;
    mask_pack_type m_mpack;
};


} // namespace sb

/**=====================================================================
 * Create an rvalue or lvalue
 *======================================================================*/
template<typename VTr, layout_t Layout, typename... MP>
static auto make_rvalue( const simd::vec<VTr,Layout> & v,
			 const sb::mask_pack<MP...> & m ) {
    return sb::rvalue<VTr,Layout,MP...>( v, m );
}

template<typename VTr, layout_t Layout, typename... MP>
static auto make_rvalue( simd::vec<VTr,Layout> && v,
			 sb::mask_pack<MP...> && m ) {
    return sb::rvalue<VTr,Layout,MP...>(
	std::forward<simd::vec<VTr,Layout>>( v ),
	std::forward<sb::mask_pack<MP...>>( m ) );
}

template<typename VTr, typename... MP>
static auto make_rvalue( const simd::detail::mask_impl<VTr> & v,
			 const sb::mask_pack<MP...> & m ) {
    return sb::rvalue<VTr,simd::lo_unknown,MP...>( v, m );
}

template<typename VTr, typename... MP>
static auto make_rvalue( simd::detail::mask_impl<VTr> && v,
			 sb::mask_pack<MP...> && m ) {
    return sb::rvalue<VTr,simd::lo_unknown,MP...>(
	std::forward<simd::detail::mask_impl<VTr>>( v ),
	std::forward<sb::mask_pack<MP...>>( m ) );
}

template<typename VTr, typename I, typename Enc, bool NT, layout_t Layout,
	 typename... MP>
static auto
make_lvalue( const simd::detail::vector_ref_impl<VTr,I,Enc,NT,Layout> & v,
	     const sb::mask_pack<MP...> & m ) {
    return sb::lvalue<VTr,I,Enc,NT,Layout,MP...>( v, m );
}

template<typename VTr, typename I, typename Enc, bool NT, layout_t Layout,
	 typename... MP>
static auto
make_lvalue( simd::detail::vector_ref_impl<VTr,I,Enc,NT,Layout> && v,
	     sb::mask_pack<MP...> && m ) {
    return sb::lvalue<VTr,I,Enc,NT,Layout,MP...>(
	std::forward<simd::detail::vector_ref_impl<VTr,I,Enc,NT,Layout>>( v ),
	std::forward<sb::mask_pack<MP...>>( m ) );
}

template<typename MTr, simd::layout_t Layout, typename... MP>
typename std::enable_if<simd::detail::is_mask_logical_traits<MTr>::value && MTr::VL == 1, bool>::type
is_true( rvalue<MTr,Layout,MP...> r ) {
    return r.value().get() != 0;
}

inline bool is_true( sb::rvalue<simd::detail::mask_bool_traits,lo_unknown> r ) {
    return r.value().get();
}

template<typename VTr,layout_t Layout, typename... MTr>
typename std::enable_if<!std::is_void<VTr>::value && VTr::VL == 1, bool>::type
is_true( sb::rvalue<VTr,Layout,MTr...> r ) {
    return r.value().at(0) != typename VTr::element_type(0);
}

template<typename VTr, layout_t Layout, typename... MTr>
bool is_false( sb::rvalue<VTr,Layout,MTr...> r ) {
    if constexpr ( !std::is_void<VTr>::value && VTr::VL == 1 )
	return r.value().at(0) == typename VTr::element_type(0);
    else
	return r.value().is_all_false();
}


} // namespace expr

#endif // GRAPTOR_DSL_AVAL_H
