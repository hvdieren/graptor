// -*- c++ -*-

#ifndef GRAPTOR_SIMD_VECTOR_H
#define GRAPTOR_SIMD_VECTOR_H

#include <type_traits>

#include "graptor/longint.h"
#include "graptor/target/vector.h"
#include "graptor/target/algo.h"
#include "graptor/simd/decl.h"

namespace simd {

namespace detail {

// template<class Traits>
// class vector_impl;

template<class Traits,layout_t Layout>
class vec;

/***********************************************************************
 * vector -- generic class with dynamically initialised layout member
 *           this class is to be phased out and replaced by class vec
 ***********************************************************************/
template<class Traits, layout_t Layout>
class vec;

template<class Traits>
class vec<Traits,lo_variable> {
public:
    static constexpr unsigned short VL = Traits::VL;

    using vector_traits = Traits;
    using data_type = vector_traits;
    using self_type = vec<vector_traits,lo_variable>;

    // Vector types and ops
    using member_type = typename vector_traits::member_type;
    using element_type = typename vector_traits::element_type;
    using type = typename vector_traits::type;
    using traits = typename vector_traits::traits;

    // Logical masks
    // using logmask_traits = detail::mask_logical_traits<sizeof(member_type), VL>;

	// Bitmasks
    // using bitmask_traits = detail::mask_bit_traits<VL>;
    // using bmask_traits = typename bitmask_traits::traits;
    // using bmask_type = typename bitmask_traits::type;

    // Preferred masks
    // using prefmask_traits = typename detail::mask_preferred_traits_type<element_type, VL>;
    using prefmask_traits = typename vector_traits::prefmask_traits;
    using mask_traits = typename prefmask_traits::traits;
    using mask_type = typename prefmask_traits::type;
    using simd_mask_type = mask_impl<prefmask_traits>;

    using tag_type = typename prefmask_traits::tag_type;
    static_assert( std::is_same_v<tag_type,typename Traits::tag_type>, "check" );

    static_assert( !is_vector<member_type>::value,
		   "type cannot be vector_impl" );
    static_assert( sizeof(member_type) <= 8
		   || is_logical<member_type>::value
		   || is_longint<member_type>::value,
		   "type cannot be native vector" );

public:
    GG_INLINE vec()
	: m_data( traits::setzero() ),
	  // m_layout( lo_constant )
	  m_layout( lo_unknown ) // prepare for const m_layout
	{ }
    explicit GG_INLINE vec( type data, layout_t layout = ( VL == 1 ? lo_constant : lo_unknown ) )
	: m_data( data ), m_layout( layout ) { }
    explicit GG_INLINE vec( type data, bool linear )
	: m_data( data ), m_layout( linear ? lo_linalgn : lo_unknown ) { }
    template<typename Traits2,layout_t Layout2>
    explicit GG_INLINE vec( vec<Traits2,Layout2> v,
			    typename std::enable_if<std::is_integral<typename Traits2::member_type>::value && Traits2::VL == VL && (VL>1)>::type * = nullptr )
	: vec( v.data(), v.get_layout() ) { }
    template<typename Traits2,layout_t Layout2>
    explicit GG_INLINE vec( vec<Traits2,Layout2> v,
			    typename std::enable_if<Traits2::VL == 1>::type * = nullptr )
	: m_data( traits::set1( (member_type)v.data() ) ),
	  m_layout( lo_constant ) { }
    // Initialisation with member_type
    template<typename U>
    explicit GG_INLINE vec(
	U data,
	layout_t layout = lo_constant, // lo_unknown, -- potentally problemsome, see comment below
	typename std::enable_if<std::is_same<U,member_type>::value
	&& ! std::is_same<type,member_type>::value>::type * = nullptr )
	: m_data( traits::set1( data ) ), m_layout( layout ) { } // lo_constant ) { } -- this should be correct as lo_constant but setting it such triggers erroneous calculation (frontier density)

    vec( const vec & v )
	: m_data( v.m_data ),
	  m_layout( v.m_layout ) { }
    vec( vec && v )
	: m_data( std::move( v.m_data ) ),
	  m_layout( v.m_layout ) { }

    template<layout_t Layout>
    vec & operator = ( const vec<vector_traits,Layout> & v ) {
	m_data = v.m_data;
	// m_layout = v.m_layout;
	assert( maintains_left( m_layout, v.get_layout() )
		&& "simd_vector assignment" );
	return *this;
    }
    template<layout_t Layout>
    vec & operator = ( vec<vector_traits,Layout> && v ) {
	m_data = std::move( v.data() );
	// m_layout = v.m_layout;
	assert( maintains_left( m_layout, v.get_layout() )
		&& "simd_vector assignment" );
	return *this;
    }

public:
    static constexpr auto true_mask()  { return simd_mask_type::true_mask();  }
    static constexpr auto false_mask() { return simd_mask_type::false_mask(); }
    static constexpr auto zero_val() { return vec<vector_traits,lo_constant>( traits::setzero() ); }

    static constexpr auto allones_val() { return vec<vector_traits,lo_constant>( traits::setone() ); }
    static constexpr auto allones_shr1_val() { return vec<vector_traits,lo_constant>( traits::setone_shr1() ); }
    static constexpr auto one_val() { return vec<vector_traits,lo_constant>( traits::setoneval() ); }
    
    member_type at(unsigned i) const { return traits::lane( data(), i ); }

#if 0
    [[deprecated("avoid this method in order to make m_layout constexpr")]]
    void set1inc( member_type a ) {
	m_data = traits::set1inc( a );
	m_layout = lo_linalgn;
    }
#endif
    template<bool aligned = false>
    static auto s_set1inc( element_type a ) {
	return vec<vector_traits,lo_unknown>::template s_set1inc<aligned>( a );
	// return vec<vector_traits,
	// aligned ? lo_linalgn : lo_linear>( traits::set1inc( a ) );
    }
#if 0
    [[deprecated("avoid this method in order to make m_layout constexpr")]]
    void set1inc0() {
	m_data = traits::set1inc0();
	m_layout = lo_linalgn;
    }
#endif
    static auto s_set1inc0() {
	return vec<vector_traits,lo_linalgn>( traits::set1inc0() );
    }

    template<typename U>
    auto convert_to() const {
	return convert_to( wrap_type<U>() );
    }

    template<typename UTr>
    auto convert_data_type() const {
	if constexpr ( detail::is_mask_traits<UTr>::value )
	    return asmask<UTr>();
	else
	    return convert_to<typename UTr::element_type>();
    }

    template<typename MTr = prefmask_traits>
    simd::detail::mask_impl<MTr>
    asmask( std::enable_if_t<MTr::VL != 1> * = nullptr ) const {
	using ty = mask_logical_traits<vector_traits::W,VL>;
	return simd::detail::mask_impl<MTr>( mask_cvt<MTr,ty>::convert( data() ) );
    }

    template<typename MTr = prefmask_traits>
    simd::detail::mask_impl<MTr>
    asmask(
	std::enable_if_t<is_mask_bool_traits<MTr>::value
	&& std::is_same<member_type,bool>::value > * = nullptr
	) const {
	return simd::detail::mask_impl<MTr>( data() ); // scalar copy
    }


    template<typename MTr = prefmask_traits>
    simd::detail::mask_impl<MTr>
    asmask(
	std::enable_if_t<is_mask_logical_traits<MTr>::value
	&& MTr::VL == 1> * = nullptr
	) const {
	return simd::detail::mask_impl<MTr>( logical<MTr::W>( data() != 0 ) );
    }

    template<typename MTr = prefmask_traits>
    simd::detail::mask_impl<MTr>
    asmask(
	std::enable_if_t<is_mask_bool_traits<MTr>::value
	&& !std::is_same<member_type,bool>::value > * = nullptr
	) const {
	return simd::detail::mask_impl<MTr>( data() != 0 ); // scalar cmp
    }

    [[deprecated("avoid this method in order to make m_layout constexpr")]]
    void store( type data ) { m_data = data; }
    [[deprecated("ntstore() should not exist in vector_imp")]]
    void ntstore( type data ) {
	store( data ); // doesn't distinguish as m_data is held in register
    }
    [[deprecated("avoid this method in order to make m_layout constexpr")]]
    void set( type data ) { m_data = data; }
    [[deprecated("avoid this method in order to make m_layout constexpr")]]
    void setl0( element_type data ) {
	m_data = traits::setl0( data );
	m_layout = lo_unknown;
    }

    // Unsafe method to update contents. Assumes caller guarantees that
    // semantics remain correct.
    template<layout_t Layout2>
    void set_unsafe( vec<vector_traits,Layout2> data ) { m_data = data.data(); }

    // [[deprecated("avoid this method in order to make m_layout constexpr")]]
    // void setl0( vec<typename vector_traits::template rebindVL<1>::type,lo_variable> v );
    
#if 0
    [[deprecated("avoid this method in order to make m_layout constexpr")]]
    GG_INLINE void load( const member_type * ptr ) {
	m_data = traits::load( ptr );
	m_layout = lo_unknown;
    }
    [[deprecated("avoid this method in order to make m_layout constexpr")]]
    GG_INLINE void ntload( const member_type * ptr ) {
	m_data = traits::ntload( ptr );
	m_layout = lo_unknown;
    }
    [[deprecated("avoid this method in order to make m_layout constexpr")]]
    GG_INLINE void loadu( const member_type * ptr ) {
	m_data = traits::loadu( ptr );
	m_layout = lo_unknown;
    }
#endif
    // replacement method
    static GG_INLINE
    vec<vector_traits,lo_unknown> load_from( const member_type * ptr ) {
	return vec<vector_traits,lo_unknown>( traits::load( ptr ) );
    }

#if 0 // to be revisited
    [[deprecated("avoid this method in order to make m_layout constexpr")]]
    GG_INLINE void sll( unsigned int s ) {
	m_data = traits::sll( data(), s );
	m_layout = lo_unknown;
    }
    [[deprecated("avoid this method in order to make m_layout constexpr")]]
    GG_INLINE void sra( unsigned int s ) {
	m_data = traits::sra( data(), s );
	m_layout = lo_unknown;
    }
    [[deprecated("avoid this method in order to make m_layout constexpr")]]
    GG_INLINE void sra1() {
	m_data = traits::sra1( data() );
	m_layout = lo_unknown;
    }
#endif

public:
    type data() const { return m_data; }
    type & data() { return m_data; } // used in simd_vector_ref ctor

    bool is_linear() const { return m_layout != lo_unknown; }
    layout_t get_layout() const { return m_layout; }

    self_type abs() const { return make_vector( traits::abs( data() ) ); }
	//self_type sqrt() const { return make_vector( traits::sqrt( data() ) ); }
    self_type sqrt() const { return make_vector( target::scalar_fp<type>::sqrt( data() ) ); }
    self_type bitwise_invert() const {
	return make_vector( traits::bitwise_invert( data() ) );
    }
    self_type logical_invert() const {
	return make_vector( traits::logical_invert( data() ) );
    }

    template<typename ReturnTy>
    vec<ty<ReturnTy,VL>,lo_unknown> tzcnt() const {
	return vec<ty<ReturnTy,VL>,lo_unknown>(
	    target::tzcnt<ReturnTy,element_type,VL>::compute( data() ) );
    }

    template<typename ReturnTy>
    vec<ty<ReturnTy,VL>,lo_unknown> lzcnt() const {
	return vec<ty<ReturnTy,VL>,lo_unknown>(
	    target::lzcnt<ReturnTy,element_type,VL>::compute( data() ) );
    }

    simd_mask_type bor_assign( self_type r, simd_mask_type m ) {
	self_type & l = *this;
	auto a = l.data(); // load data
	auto b = r.data(); // load data
	auto cmask = m.template asvector<member_type>();
#if ALT_BOR_ASSIGN
	auto upd = a | b;
	l.set( upd, m ); // store data
	// Ensures that return value is stronger than m
	return (upd != a).asmask() & m;
#else
	auto mod = ~a & b;
	l.set( a | b, m ); // store data
	auto v = make_vector( mod );
	auto z = zero_val();
	// Ensures that return value is stronger than m
	return (v != z).asmask() & m;
#endif
    }
    simd_mask_type lor_assign( self_type r ) {
	self_type & l = *this;
	auto a = l.data(); // load data
	auto b = r.data(); // load data
	auto mod = ~a & b;
	a |= b;
	l.set( a ); // store data
	auto v = make_vector( mod );
	return v.asmask();
    }

private:
    type m_data;
    const layout_t m_layout;

private:
    template<typename Ty> struct wrap_type { };
    
    template<typename U>
    auto convert_to( wrap_type<U> ) const {
	return vec<typename vector_traits::template rebindTy<U>::type,lo_variable>( conversion_traits<member_type,U,VL>::convert( data() ) );
    }

public:
    static auto make_vector( type data, layout_t layout = lo_unknown ) {
	return self_type( data, layout );
    }
    static auto make_mask( const mask_type & mask ) {
	return simd_mask_type( mask );
    }
};

template<class Traits>
using vector_impl  = vec<Traits,lo_variable>;

template<class Traits>
template<typename V>
auto mask_impl<Traits>::asvector() const {
    using Ty = vector_impl<vdata_traits<typename add_logical<V>::type,VL>>;
    // return Ty( Ty::traits::asvector( get() ) );
    using cvt = mask_cvt<mask_logical_traits<sizeof(V),Traits::VL>,Traits>;
    auto v = cvt::convert( get() );
    return Ty( v );
}

// Override vector_impl for bit masks by inheriting behavior of mask
/*
template<unsigned short VL, layout_t Layout>
class vec<mask_bit_traits<VL>,Layout>
    : public mask_impl<mask_bit_traits<VL>> {
    using mask_impl<mask_bit_traits<VL>>::mask_impl;
};
*/

/*
template<class Traits>
void vec<Traits,lo_variable>::setl0(
    vec<typename vec::template rebindVL<1>::type,lo_constant> v ) {
    m_data = traits::setl0( v.at(0) );
    // m_layout = lo_constant; // lo_unknown??
}
*/

/***********************************************************************
 * vector -- template class with constexpr layout member
 ***********************************************************************/
template<class Traits,layout_t Layout>
class vec {
public:
    static constexpr unsigned short VL = Traits::VL;
    static constexpr layout_t layout = Layout;

    static_assert( layout != lo_variable,
		   "this class does not have a variably-defined layout" );

    using vector_traits = Traits;
    using data_type = vector_traits;
    using self_type = vec<vector_traits,layout>;

    // Vector types and ops
    using member_type = typename vector_traits::member_type;
    using element_type = typename vector_traits::element_type;
    using type = typename vector_traits::type;
    using traits = typename vector_traits::traits;

    // Logical masks
    // using logmask_traits = detail::mask_logical_traits<sizeof(member_type), VL>;

	// Bitmasks
    // using bitmask_traits = detail::mask_bit_traits<VL>;
    // using bmask_traits = typename bitmask_traits::traits;
    // using bmask_type = typename bitmask_traits::type;

    // Preferred masks
    // using prefmask_traits = detail::mask_preferred_traits_type<element_type, VL>;
    using prefmask_traits = typename vector_traits::prefmask_traits;
    using mask_traits = typename prefmask_traits::traits;
    using mask_type = typename prefmask_traits::type;
    using simd_mask_type = mask_impl<prefmask_traits>;

    using tag_type = typename prefmask_traits::tag_type;
    static_assert( std::is_same_v<tag_type,typename Traits::tag_type>, "check" );

    static_assert( !is_vector<member_type>::value,
		   "type cannot be vec" );
    static_assert( sizeof(member_type) <= 8
		   || is_logical<member_type>::value
		   || is_longint<member_type>::value,
		   "type cannot be native vector" );

public:
    // Set to zero for safety only, will not necessarily meet expectations
    // expressed in layout.
    // [[deprecated("default constructor violates layout constraints")]]
    GG_INLINE vec() : m_data( traits::setzero() ) { }

// private:
    explicit GG_INLINE vec( type data ) : m_data( data ) {
/*
	static_assert( layout == lo_unknown
		       || ( layout == lo_constant && VL == 1),
		       "appreciation of layout constraint" );
*/
    }

public:
    // Initialisation with member_type
    template<typename U>
    explicit GG_INLINE vec(
	U data,
	std::enable_if_t<std::is_same_v<U,member_type>
	&& ( layout == lo_constant || layout == lo_unknown )> * = nullptr )
	: m_data( traits::set1( data ) ) { }

    // Copy constructors
    vec( const vec & v ) : m_data( v.m_data ) { }
    vec( vec && v ) : m_data( std::move( v.m_data ) ) { }

    // Assignment operators
    vec & operator = ( const vec & v ) {
	m_data = v.m_data;
	return *this;
    }
    vec & operator = ( vec && v ) {
	m_data = std::move( v.m_data );
	return *this;
    }

public:
    static constexpr auto true_mask()  { return simd_mask_type::true_mask();  }
    static constexpr auto false_mask() { return simd_mask_type::false_mask(); }
    static constexpr auto zero_val() { return vec<vector_traits,lo_constant>( traits::setzero() ); }

    static constexpr auto allones_val() { return vec<vector_traits,lo_constant>( traits::setone() ); }
    static constexpr auto allones_shr1_val() { return vec<vector_traits,lo_constant>( traits::setone_shr1() ); }
    static constexpr auto one_val() { return vec<vector_traits,lo_constant>( traits::setoneval() ); }
    
    member_type at(unsigned i) const { return traits::lane( data(), i ); }

    // These static member functions create a new vector. The vector created
    // has an appropriate layout encoded as indicated by the semantics of the
    // method.
    template<bool aligned = false>
    static auto s_set1inc( element_type a ) {
	return vec<vector_traits,aligned ? lo_linalgn : lo_linear>(
	    traits::set1inc( a ) );
    }
    static auto s_set1inc0() {
	return vec<vector_traits,lo_linalgn>( traits::set1inc0() );
    }

    static GG_INLINE auto load_from( const member_type * ptr ) {
	return vec<vector_traits,lo_unknown>( traits::load( ptr ) );
    }

    template<typename U>
    auto convert_to() const {
	// return convert_to( wrap_type<U>() );
	return vec<typename vector_traits::template rebindTy<U>::type,
		   layout>( conversion_traits<member_type,U,VL>::convert( data() ) );
    }

    template<typename UTr>
    auto convert_data_type() const {
	if constexpr ( detail::is_mask_traits<UTr>::value )
	    return asmask<UTr>();
	else
	    return convert_to<typename UTr::element_type>();
    }

    template<typename MTr = prefmask_traits>
    simd::detail::mask_impl<MTr>
    asmask() const {
	if constexpr ( MTr::VL != 1 ) {
	    if constexpr ( is_bitfield_v<element_type> ) {
		using ty = prefmask_traits;
		return simd::detail::mask_impl<MTr>(
		    mask_cvt<MTr,ty>::convert( data() ) );
	    } else if constexpr ( is_logical_v<element_type> ) {
		using ty = mask_logical_traits<vector_traits::W,VL>;
		return simd::detail::mask_impl<MTr>(
		    mask_cvt<MTr,ty>::convert( data() ) );
	    } else if constexpr ( std::is_same_v<bool,element_type> ) {
		using ty = mask_logical_traits<vector_traits::W,VL>;
		auto intm = convert_to<typename ty::element_type>();
		return simd::detail::mask_impl<MTr>(
		    mask_cvt<MTr,ty>::convert( intm.data() ) );
	    }
	} else {
	    if constexpr ( is_mask_bool_traits<MTr>::value ) {
		if constexpr ( std::is_same_v<member_type,bool> )
		    return simd::detail::mask_impl<MTr>( data() );
		else
		    return simd::detail::mask_impl<MTr>( data() != 0 );
	    } else if constexpr ( is_mask_logical_traits<MTr>::value ) {
		return simd::detail::mask_impl<MTr>( logical<MTr::W>( data() != 0 ) );
	    } else if constexpr ( is_mask_bit_logical_traits<MTr>::value ) {
		return simd::detail::mask_impl<MTr>(
		    bitfield<MTr::B>( data() == 0 ? 0 : ((((uint8_t)1)<<MTr::B)-1) ) );
	    }
	}
	assert( 0 && "NYI" );
    }

    [[deprecated("this method may violate the layout constraint")]]
    void store( type data ) { m_data = data; }

    [[deprecated("this method may violate the layout constraint")]]
    void set( type data ) { m_data = data; }

    [[deprecated("this method may violate the layout constraint")]]
    void setl0( vec<typename vector_traits::template rebindVL<1>::type,
		lo_constant> v );

    // Unsafe method to update contents. Assumes caller guarantees that
    // semantics remain correct.
    template<layout_t Layout2>
    void set_unsafe( vec<vector_traits,Layout2> data ) { m_data = data.data(); }

#if 0 // to be revisited
    [[deprecated("avoid this method in order to make m_layout constexpr")]]
    GG_INLINE void sll( unsigned int s ) {
	m_data = traits::sll( data(), s );
	m_layout = lo_unknown;
    }
    [[deprecated("avoid this method in order to make m_layout constexpr")]]
    GG_INLINE void sra( unsigned int s ) {
	m_data = traits::sra( data(), s );
	m_layout = lo_unknown;
    }
    [[deprecated("avoid this method in order to make m_layout constexpr")]]
    GG_INLINE void sra1() {
	m_data = traits::sra1( data() );
	m_layout = lo_unknown;
    }
#endif

public:
    type data() const { return m_data; }
    type & data() { return m_data; } // used in simd_vector_ref ctor

    constexpr bool is_linear() const { return layout != lo_unknown; }
    constexpr layout_t get_layout() const { return layout; }

    self_type abs() const { return self_type( traits::abs( data() ) ); }
	//self_type sqrt() const { return self_type( traits::sqrt( data() ) ); }
    self_type sqrt() const { return self_type( target::scalar_fp<type>::sqrt( data() ) ); }
    self_type bitwise_invert() const {
	return self_type( traits::bitwise_invert( data() ) );
    }
    self_type logical_invert() const {
	return make_vector( traits::logical_invert( data() ) );
    }

    template<typename ReturnTy>
    vec<ty<ReturnTy,VL>,lo_unknown> tzcnt() const {
	return vec<ty<ReturnTy,VL>,lo_unknown>(
	    target::tzcnt<ReturnTy,element_type,VL>::compute( data() ) );
    }

    template<typename ReturnTy>
    vec<ty<ReturnTy,VL>,lo_unknown> lzcnt() const {
	return vec<ty<ReturnTy,VL>,lo_unknown>(
	    target::lzcnt<ReturnTy,element_type,VL>::compute( data() ) );
    }

    simd_mask_type bor_assign( self_type r, simd_mask_type m ) {
	static_assert( layout == lo_unknown, "limitation" );
	self_type & l = *this;
	auto a = l.data(); // load data
	auto b = r.data(); // load data
	auto cmask = m.template asvector<member_type>();
#if ALT_BOR_ASSIGN
	auto upd = a | b;
	l.set( upd, m ); // store data
	// Ensures that return value is stronger than m
	return (upd != a).asmask() & m;
#else
	auto mod = ~a & b;
	l.set( a | b, m ); // store data
	auto v = make_vector( mod );
	auto z = zero_val();
	// Ensures that return value is stronger than m
	return (v != z).asmask() & m;
#endif
    }
    simd_mask_type lor_assign( self_type r ) {
	static_assert( layout == lo_unknown, "limitation" );
	self_type & l = *this;
	auto a = l.data(); // load data
	auto b = r.data(); // load data
	auto mod = ~a & b;
	a |= b;
	l.set( a ); // store data
	auto v = make_vector( mod );
	return v.asmask();
    }

private:
    type m_data;

private:
    template<typename Ty> struct wrap_type { };
    
public:
    static auto make_vector( type data ) {
	return self_type( data );
    }
    static auto make_mask( const mask_type & mask ) {
	return simd_mask_type( mask );
    }
};

template<class Traits,layout_t Layout>
void vec<Traits,Layout>::setl0(
    vec<typename vector_traits::template rebindVL<1>::type,lo_constant> v ) {
    static_assert( Layout == lo_unknown, "layout restrition" );
    m_data = traits::setl0( v.at(0) );
}

} // namespace detail

// type vector is to be replaced by vec
template<typename T, unsigned short VL>
using vector = detail::vector_impl<detail::vdata_traits<T,VL>>;

using detail::vec;

/**********************************************************************
 * Creation methods
 **********************************************************************/
template<typename Tr, layout_t layout = lo_unknown>
auto load_from( const typename Tr::member_type * addr ) {
    return vec<Tr,layout>( Tr::traits::load( addr ) );
}

template<typename Tr>
auto create_constant( typename Tr::member_type a ) {
    return vec<Tr,lo_constant>( a );
}

template<typename T, unsigned short VL>
auto create_constant( T a ) {
    return vec<simd::ty<T,VL>,lo_constant>( a );
}

template<typename Tr>
auto create_unknown( typename Tr::type a ) {
    return vec<Tr,Tr::VL == 1 ? lo_constant : lo_unknown>( a );
}

template<typename T>
auto create_scalar( T a,
		    std::enable_if_t<!detail::is_vm_traits_v<T>
		    && ( std::is_integral_v<T>
			 || is_logical_v<T>
			 || std::is_floating_point_v<T> )> *
		    = nullptr ) {
    return vec<simd::ty<T,1>,lo_constant>( a );
}

template<typename Tr>
auto create_scalar( typename Tr::type a,
		    std::enable_if_t<detail::is_vm_traits_v<Tr>
		    && Tr::VL == 1> * = nullptr ) {
    return vec<Tr,lo_constant>( a );
}

template<typename Tr, bool aligned = false>
auto create_set1inc( typename Tr::member_type a ) {
    return vec<Tr,aligned ? lo_linalgn : lo_linear>( Tr::traits::set1inc( a ) );
}

template<typename T, unsigned short VL, bool aligned = false>
auto create_set1inc( T a ) {
    return create_set1inc<simd::ty<T,VL>,aligned>( a );
}

template<typename Tr>
auto create_set1inc0() {
    return vec<Tr,lo_linalgn>( Tr::traits::set1inc0() );
}

template<typename T, unsigned short VL>
auto create_set1inc0() {
    return create_set1inc0<simd::ty<T,VL>>();
}

template<typename Tr>
constexpr auto create_zero() {
    return vec<Tr,lo_constant>( Tr::traits::setzero() );
}

template<typename Tr>
constexpr auto create_one() {
    return vec<Tr,lo_constant>( Tr::traits::setoneval() );
}

template<typename Tr>
constexpr auto create_allones() {
    return vec<Tr,lo_constant>( Tr::traits::setone() );
}

template<typename Tr>
constexpr auto create_allones_shr1() {
    return vec<Tr,lo_constant>( Tr::traits::setone_shr1() );
}

} // namespace simd

#endif // GRAPTOR_SIMD_VECTOR_H
