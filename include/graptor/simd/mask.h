// -*- c++ -*-

#ifndef GRAPTOR_SIMD_MASK_H
#define GRAPTOR_SIMD_MASK_H

#include "graptor/target/vector.h"
#include "graptor/simd/decl.h"

namespace simd {

/***********************************************************************
 * conversion
 ***********************************************************************/
namespace detail {

template<typename To, typename From, typename = void>
struct mask_cvt;

template<typename To>
struct mask_cvt<To,To> {
    static typename To::type convert( typename To::type v ) { return v; }
};

template<>
struct mask_cvt<mask_bit_traits<1>,mask_bool_traits> {
    static auto convert( typename mask_bool_traits::type v ) {
	return typename mask_bit_traits<1>::type(
	    typename mask_bit_traits<1>::type(!!v) );
    }
};

template<unsigned short B>
struct mask_cvt<mask_bit_logical_traits<B,1>,mask_bool_traits> {
    static typename mask_bit_logical_traits<B,1>::type
    convert( typename mask_bool_traits::type v ) {
	return bitfield<B>( v ? ((((uint8_t)1)<<B)-1) : 0 );
    }
};

template<unsigned short B>
struct mask_cvt<mask_bool_traits,mask_bit_logical_traits<B,1>> {
    static typename mask_bool_traits::type
    convert( typename mask_bit_logical_traits<B,1>::type v ) {
	// Only top bit matters
	return v.get() >> (B-1);
    }
};

template<unsigned short W>
struct mask_cvt<mask_logical_traits<W,1>,mask_bool_traits> {
    static typename mask_logical_traits<W,1>::type
    convert( typename mask_bool_traits::type v ) {
	return mask_logical_traits<W,1>::get_val( v );
    }
};

template<>
struct mask_cvt<mask_bool_traits,mask_bit_traits<1>> {
    static typename mask_bool_traits::type
	convert( typename mask_bit_traits<1>::type v ) {
	return typename mask_bool_traits::type(
	    (v & mask_bit_traits<1>::type(1)) != 0 );
    }
};

template<unsigned short W>
struct mask_cvt<mask_bool_traits,mask_logical_traits<W,1>> {
    static typename mask_bool_traits::type
	convert( typename mask_logical_traits<W,1>::type v ) {
	return typename mask_bool_traits::type( v != 0 );
    }
};

template<unsigned short W, unsigned short VL>
struct mask_cvt<mask_logical_traits<W,VL>,mask_bit_traits<VL>> {
    static typename mask_logical_traits<W,VL>::type
	convert( typename mask_bit_traits<VL>::type v ) {
	return mask_bit_traits<VL>::traits::template asvector<logical<W>>( v );
    }
};

template<unsigned short W, unsigned short VL>
struct mask_cvt<mask_bit_traits<VL>,mask_logical_traits<W,VL>> {
    static typename mask_bit_traits<VL>::type
	convert( typename mask_logical_traits<W,VL>::type v ) {
	return mask_logical_traits<W,VL>::traits::asmask( v );
    }
};

template<unsigned short W1, unsigned short W2, unsigned short VL>
struct mask_cvt<mask_logical_traits<W1,VL>,mask_logical_traits<W2,VL>,
		std::enable_if_t<W1 != W2>> {
    static typename mask_logical_traits<W1,VL>::type
	convert( typename mask_logical_traits<W2,VL>::type v ) {
	return typename mask_logical_traits<W1,VL>::type(
	    conversion_traits<typename mask_logical_traits<W2,VL>::element_type,
	    typename mask_logical_traits<W1,VL>::element_type, VL>
	    ::convert( v ) );
    }
};

template<unsigned short W, unsigned short B, unsigned short VL>
struct mask_cvt<mask_logical_traits<W,VL>,mask_bit_logical_traits<B,VL>> {
    static typename mask_logical_traits<W,VL>::type
	convert( typename mask_bit_logical_traits<B,VL>::type v ) {
	return typename mask_logical_traits<W,VL>::type(
	    conversion_traits<typename mask_bit_logical_traits<B,VL>::element_type,
	    typename mask_logical_traits<W,VL>::element_type, VL>
	    ::convert( v ) );
    }
};

template<unsigned short W, unsigned short B, unsigned short VL>
struct mask_cvt<mask_bit_logical_traits<B,VL>,mask_logical_traits<W,VL>> {
    static typename mask_bit_logical_traits<B,VL>::type
	convert( typename mask_logical_traits<W,VL>::type v ) {
	return typename mask_bit_logical_traits<B,VL>::type(
	    conversion_traits<typename mask_logical_traits<W,VL>::element_type,
	    typename mask_bit_logical_traits<B,VL>::element_type, VL>
	    ::convert( v ) );
    }
};

template<unsigned short B, unsigned short VL>
struct mask_cvt<mask_bit_traits<VL>,mask_bit_logical_traits<B,VL>> {
    static typename mask_bit_traits<VL>::type
	convert( typename mask_bit_logical_traits<B,VL>::type v ) {
	return mask_bit_logical_traits<B,VL>::traits::asmask( v );
    }
};

template<unsigned short B, unsigned short VL>
struct mask_cvt<mask_bit_logical_traits<B,VL>,mask_bit_traits<VL>> {
    static typename mask_bit_logical_traits<B,VL>::type
	convert( typename mask_bit_traits<VL>::type v ) {
	return mask_bit_traits<VL>::traits::template asvector<bitfield<B>>( v );
    }
};

} // namespace detail

/***********************************************************************
 * nomask
 ***********************************************************************/
template<unsigned short VL>
struct nomask;

template<typename T>
struct is_nomask : std::false_type { };

template<unsigned short VL>
struct is_nomask<nomask<VL>> : std::true_type { };

template<unsigned short VL_>
struct nomask {
    static constexpr unsigned short VL = VL_;

    bool is_all_false() const { return false; }

    bool mask() const { return true; } // only for VL=1 and used as bool...
    bool data() const { return true; } // only for VL=1 and used as bool...
};

template<unsigned short VL>
auto operator & ( nomask<VL>, nomask<VL> ) { return nomask<VL>(); }

template<typename V, unsigned short VL>
auto operator & ( nomask<VL>, V & v ) { return v; }

template<typename V, unsigned short VL>
auto operator & ( nomask<VL>, const V & v ) { return v; }

template<typename V, unsigned short VL>
auto operator & ( V & v, nomask<VL> ) { return v; }

template<typename V, unsigned short VL>
auto operator & ( const V & v, nomask<VL> ) { return v; }

namespace detail {

/***********************************************************************
 * mask
 ***********************************************************************/

/**
 * mask_impl: Generic implementation of masks (concrete values)
 **/
template<class Traits>
class mask_impl {
public:
    static constexpr unsigned short VL = Traits::VL;
    static constexpr layout_t layout = lo_unknown;

    using mask_traits = Traits;
    using data_type = mask_traits;
    using traits = typename mask_traits::traits;
    using type = typename mask_traits::type;
    using element_type = typename mask_traits::element_type;
    using self_type = mask_impl<Traits>;

public:
    mask_impl( type m ) : m_mask( m ) { }
    template<layout_t Layout>
    mask_impl( const vec<mask_traits,Layout> & v ) : m_mask( v.data() ) { }
    mask_impl() { }

    mask_impl( const mask_impl & m ) : m_mask( m.m_mask ) { }
    mask_impl( mask_impl && m ) : m_mask( std::forward<type>( m.m_mask ) ) { }

    mask_impl & operator = ( const mask_impl & m ) {
	m_mask = m.m_mask;
	return *this;
    }
    mask_impl & operator = ( mask_impl && m ) {
	m_mask = std::forward<type>( m.m_mask );
	return *this;
    }

    self_type asmask() const { return *this; }

    void setl0( element_type data ) {
	m_mask = traits::setl0( data );
    }
    template<typename Tr1>
    typename std::enable_if<
	!is_mask_bool_traits<Tr1>::value && Tr1::VL == 1>::type *
    setl0( mask_impl<Tr1> v ) {
	static_assert( is_logical<typename Tr1::traits::member_type>::value, "oops" );
	m_mask = traits::setl0( element_type::get_val( v.at(0) ) );
    }
    void setl0( mask_impl<mask_bool_traits> v ) {
	m_mask = traits::setl0( (element_type)v.at(0) );
    }

    template<typename V>
    auto asvector() const;

    template<typename UTr>
    auto convert_data_type() const {
	if constexpr ( detail::is_mask_traits<UTr>::value )
	    return convert<UTr>();
	else
	    return asvector<typename UTr::element_type>();
    }

    type get() const { return m_mask; }
    type & get() { return m_mask; }
    void set( type m ) { m_mask = m; }
    type data() const { return m_mask; }
    type & data() { return m_mask; }

    element_type at( unsigned short l ) { return traits::lane( m_mask, l ); }

    [[deprecated("seems unused; unsigned covers only 32 vector length")]]
    void from_int( unsigned m ) { m_mask = traits::from_int( m ); }

    static constexpr self_type true_mask() {
	return self_type( traits::setone() );
    }
    static constexpr self_type false_mask() {
	return self_type( traits::setzero() );
    }
    static constexpr self_type zero_val() {
	return self_type( traits::setzero() );
    }

    bool is_all_false() {
	return traits::is_all_false( get() );
    }

#if 0
    // scalar bool version
    template<unsigned short W2>
    auto
    convert_width( typename std::enable_if<W2 == 1 && VL == 1>::type * = nullptr
	) {
	using traits2 = mask_bool_traits;
	return mask_impl<traits2>(
	    conversion_traits<typename traits::member_type,
	    typename traits2::member_type,
	    VL>::convert( m_mask ) );
	// If converting logical, ensure we end up with a bool
	// return traits2::cmpne( w, traits2::setzero() );
    }
    template<unsigned short W2>
    auto
    convert_width( typename std::enable_if<
		   (W2 > 1 || (W2 == 1 && VL != 1)) && !is_mask_bit_traits<Traits>::value
		   >::type * = nullptr
	) {
	// Convert width
	using traits2 = mask_logical_traits<W2,VL>;
	return mask_impl<traits2>(
	    conversion_traits<typename traits::member_type,
	    typename traits2::member_type,VL>::convert( m_mask ) );
    }
    template<unsigned short W2>
    auto
    convert_width( typename std::enable_if<
		   (W2 > 1 || (W2 == 1 && VL != 1)) && is_mask_bit_traits<Traits>::value
		   >::type * = nullptr
	) {
	// Convert width. This returns a mask_impl with mask_logical_traits,
	// not a vector_impl
	using traits2 = mask_logical_traits<W2,VL>;
	return mask_impl<traits2>(
	    traits2::traits::asvector<logical<W2>>( m_mask ) );
    }
    template<unsigned short W2>
    auto
    convert_width( typename std::enable_if<
		   W2 == 0 && !is_mask_bit_traits<Traits>::value
		   >::type * = nullptr
	) {
	using traits2 = mask_bit_traits<VL>;
	return mask_impl<traits2>( traits::asmask( m_mask ) );
    }
    template<unsigned short W2>
    auto
    convert_width( typename std::enable_if<
		   W2 == 0 && is_mask_bit_traits<Traits>::value
		   >::type * = nullptr
	) {
	return *this;
    }
#endif

    template<unsigned short W2>
    auto convert_width() {
	using VTr = mask_preferred_traits_width<W2,VL>;
	return convert<VTr>();
    }

    template<typename VTr>
    mask_impl<VTr> convert() const {
	return mask_impl<VTr>( mask_cvt<VTr,mask_traits>::convert( get() ) );
    }

    self_type lor_assign( self_type r ) {
	// Calculate changes
	auto mod = traits::logical_and( traits::logical_invert( m_mask ),
					r.get() );
	// Store data
	m_mask = traits::logical_or( m_mask, r.get() );
	// Return changes
	return self_type( mod );
    }

    self_type lor_assign( nomask<VL> ) {
	// Change mask
	auto mod = traits::logical_invert( m_mask );

	// Absence of mask implicitly implies true
	m_mask = traits::setone();

	// Return changes
	return self_type( mod );
    }

    self_type logical_invert() const {
	return self_type( traits::logical_invert( get() ) ); 
    }

private:
    type m_mask;
};

template<typename Tr>
mask_impl<Tr> operator & (
    mask_impl<Tr> l, mask_impl<Tr> r ) {
    return mask_impl<Tr>( 
	mask_impl<Tr>::traits::logical_and( l.get(), r.get() ) ); 
}

template<unsigned short W, unsigned short VL>
mask_impl<mask_bit_traits<VL>> operator & (
    mask_impl<mask_bit_traits<VL>> l, mask_impl<mask_logical_traits<W,VL>> r ) {
    using Tr = mask_bit_traits<VL>;
    auto rcvt = r.template convert_width<0>();
    return mask_impl<Tr>( 
	mask_impl<Tr>::traits::logical_and( l.get(), rcvt.get() ) ); 
}

template<unsigned short W, unsigned short VL>
mask_impl<mask_bit_traits<VL>> operator & (
    mask_impl<mask_logical_traits<W,VL>> l, mask_impl<mask_bit_traits<VL>> r ) {
    using Tr = mask_bit_traits<VL>;
    auto lcvt = l.template convert_width<0>();
    return mask_impl<Tr>( 
	mask_impl<Tr>::traits::logical_and( lcvt.get(), r.get() ) ); 
}

template<typename Tr>
mask_impl<Tr> operator | (
    mask_impl<Tr> l, mask_impl<Tr> r ) {
    return mask_impl<Tr>( 
	mask_impl<Tr>::traits::logical_or( l.get(), r.get() ) ); 
}

} // namespace detail

/**
 * mask: A SIMD mask that adapts to circumstances, choosing between logical<W>,
 *       bool and bit mask.
 **/
template<unsigned short W, unsigned short VL>
using mask = detail::mask_impl<detail::mask_preferred_traits_width<W,VL>>;

template<typename T, unsigned short VL>
struct mask_ty_selector {
    using type = detail::mask_impl<detail::mask_preferred_traits_type<T,VL>>;
};

template<unsigned short VL>
struct mask_ty_selector<void,VL> {
    using type = detail::mask_impl<detail::mask_bit_traits<VL>>;
};

template<typename T, unsigned short VL>
using mask_ty = typename mask_ty_selector<T,VL>::type;

/**
 * mask_logical: A SIMD mask composed of logical<W> values
 **/ 
template<unsigned short W, unsigned short VL>
using mask_logical = detail::mask_impl<detail::mask_logical_traits<W,VL>>;

} // namespace simd

#endif // GRAPTOR_SIMD_MASK_H
