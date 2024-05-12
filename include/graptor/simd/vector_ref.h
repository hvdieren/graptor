// -*- c++ -*-

#ifndef GRAPTOR_SIMD_VECTOR_REF_H
#define GRAPTOR_SIMD_VECTOR_REF_H

#include "graptor/target/vector.h"
#include "graptor/simd/decl.h"
#include "graptor/simd/vector.h"
#include "graptor/encoding.h"

namespace simd {

namespace detail {
template<class Traits, typename I, typename Encoding, bool NT_, layout_t Layout>
class vector_ref_impl;
}

template<class Tr, typename I, typename Enc, bool NT, layout_t Layout>
auto create_vector_ref_scalar(
    typename detail::vector_ref_impl<Tr,I,Enc,NT,Layout>::storage_type* addr,
    I idx,
    std::enable_if_t<Layout == lo_linalgn || Layout == lo_linear || Layout == lo_constant> * = nullptr );

template<class Tr, typename I, typename Enc, bool NT, layout_t Layout>
auto create_vector_ref_vec(
    // typename detail::vector_ref_impl<Tr,I,Enc,NT,Layout>::storage_type* addr,
    typename Enc::storage_type* addr,
    const simd::detail::vec<simd::ty<I,Tr::VL>,Layout> & vidx );

template<class Tr, typename I, typename Enc, bool NT>
auto create_vector_ref_cacheop(
    typename detail::vector_ref_impl<Tr,I,Enc,NT,lo_linalgn>::storage_type* addr,
    typename detail::vector_ref_impl<Tr,I,Enc,NT,lo_linalgn>::index_type idx );

namespace detail {

/***********************************************************************
 * vector reference
 ***********************************************************************/
// Base cases covers all vector load/store ops -- scalar index
// as well as gather/scatter ops -- vector index
// Contrary to vector_impl, we do not overlay the vector and the scalar
// onto the same fiedls (pod-style union). The reason for this is to be found
// in the quality of the generated code.
template<class Traits, typename I, typename Encoding, bool NT_, layout_t Layout>
class vector_ref_impl {
public:
    static constexpr unsigned short VL = Traits::VL;
    static constexpr bool NT = NT_;
    static constexpr layout_t layout = Layout;

    static_assert( layout != lo_variable,
		   "variable layout not allowed in simd_vector_ref" );

    using vector_traits = Traits;
    using self_type = vector_ref_impl<vector_traits, I, Encoding, NT, Layout>;
    using vector_type = vec<vector_traits,lo_unknown>;
    
    using member_type = typename vector_traits::member_type;
    using element_type = typename vector_traits::element_type;
    using type = typename vector_traits::type;
    using traits = typename vector_traits::traits;

    using index_type = I;
    using index_traits = vdata_traits<index_type, VL>;
    using itraits = typename index_traits::traits;
    using vindex_type = typename index_traits::type;

    using scalar_traits = typename vector_traits::template rebindVL<1>::type;

    using encoding = Encoding;
    using storage_type = typename encoding::storage_type;

    // using vmask_type = typename traits::vmask_type;

    // Logical masks
    using logmask_traits = detail::mask_logical_traits<sizeof(element_type), VL>;
    using simd_vmask_type = mask_impl<logmask_traits>;

    // Preferred masks
    using prefmask_traits = detail::mask_preferred_traits_type<element_type, VL>;
    using mask_traits = typename prefmask_traits::traits;
    using mask_type = typename prefmask_traits::type;
    using simd_mask_type = mask_impl<prefmask_traits>;

    template<typename Tr, typename V, typename Enc__, bool NT__, layout_t Layout__>
    friend class vector_ref_impl;

    template<class Tr_, typename I_, typename Enc_, bool NT__, layout_t Layout__>
    friend auto simd::create_vector_ref_scalar(
	typename detail::vector_ref_impl<Tr_,I_,Enc_,NT__,Layout__>::storage_type* addr,
	I_ idx,
	std::enable_if_t<Layout__ == lo_linalgn || Layout__ == lo_linear || Layout__ == lo_constant> * );

    template<class Tr_, typename I_, typename Enc_, bool NT__, layout_t Layout__>
    friend auto simd::create_vector_ref_vec(
	// typename detail::vector_ref_impl<Tr_,I_,Enc_,NT__,Layout__>::storage_type* addr,
	typename Enc_::storage_type* addr,
	const simd::detail::vec<simd::ty<I_,Tr_::VL>,Layout__> & vidx ) ;

    template<class Tr_, typename I_, typename Enc_, bool NT__>
    friend auto simd::create_vector_ref_cacheop(
	typename detail::vector_ref_impl<Tr_,I_,Enc_,NT__,lo_linalgn>::storage_type* addr,
	typename detail::vector_ref_impl<Tr_,I_,Enc_,NT__,lo_linalgn>::index_type idx );

private:
    // Required due to unrestricted union
    GG_INLINE
    vector_ref_impl() { }

    template<typename U>
    GG_INLINE
    vector_ref_impl( storage_type* addr, U idx,
		     std::enable_if_t<
		     std::is_same_v<std::make_unsigned<U>,
		     std::make_unsigned<index_type>>
		     && ( layout == lo_linalgn || layout == lo_linear
		     || layout == lo_constant )> *
		     = nullptr )
	: m_addr( addr ), m_sidx( idx ) {
	static_assert( layout != lo_unknown, "Not allowed in unknown layout" );
    }

    vector_ref_impl( storage_type* addr,
		     const simd::detail::vec<simd::ty<I,VL>,layout> & vidx )
	: m_addr( addr ) {
	// vidx is used only if layout is unknown; otherwise we need the scalar
	if constexpr ( layout == lo_unknown )
	    m_vidx = vidx.data();
	else
	    m_sidx = vidx.at(0);
    }
    vector_ref_impl( storage_type* addr,
		     simd::detail::vec<simd::ty<I,VL>,layout> && vidx )
	: m_addr( addr ) {
	// vidx is used only if layout is unknown; otherwise we need the scalar
	if constexpr ( layout == lo_unknown )
	    m_vidx = std::move( vidx.data() );
	else
	    m_sidx = std::move( vidx.at(0) );
    }

public:

    GG_INLINE
    operator vector_type () const { return vector_type( data() ); }

    bool is_linear() const { return m_layout != lo_unknown; }
    layout_t get_layout() const { return m_layout; }

    static constexpr auto true_mask()  { return simd_mask_type::true_mask();  }
    static constexpr auto false_mask() { return simd_mask_type::false_mask(); }
    
    template<layout_t Layout_>
    inline simd_mask_type bor_assign( vec<vector_traits,Layout_> r, simd_mask_type m );
    template<layout_t Layout_>
    inline simd_mask_type bor_assign( vec<vector_traits,Layout_> r );
    inline simd_mask_type bor_assign( mask_impl<vector_traits> r );
    inline simd_mask_type bor_assign( mask_impl<vector_traits> r, mask_impl<vector_traits> m );

    template<layout_t Layout_>
    inline simd_mask_type band_assign( vec<vector_traits,Layout_> r, simd_mask_type m );
    template<layout_t Layout_>
    inline simd_mask_type band_assign( vec<vector_traits,Layout_> r );
    inline simd_mask_type band_assign( mask_impl<vector_traits> r );

    template<layout_t Layout_>
    inline simd_mask_type lor_assign( vec<vector_traits,Layout_> r, simd_mask_type m );
    template<layout_t Layout_>
    inline simd_mask_type lor_assign( vec<vector_traits,Layout_> r );
    template<layout_t Layout_>
    inline simd_mask_type lor_assign( vec<vector_traits,Layout_> r, simd::nomask<VL> );

    template<layout_t Layout_>
    GG_INLINE inline simd_mask_type land_assign( vec<vector_traits,Layout_> r, simd_mask_type m );
    template<layout_t Layout_>
    GG_INLINE inline simd_mask_type land_assign( vec<vector_traits,Layout_> r );
    template<layout_t Layout_>
    GG_INLINE inline simd_mask_type land_assign( vec<vector_traits,Layout_> r, simd::nomask<VL> );

    template<typename MTr, layout_t Layout_>
    GG_INLINE inline simd_mask_type bor_assign(
	vec<vector_traits,Layout_> r, mask_impl<MTr> m,
	std::enable_if_t<(vector_traits::W>8 && MTr::W<vector_traits::W)> *
	= nullptr ) {
	// First convert argument to appropriate width
	type mw = conversion_traits<
	    typename mask_logical_traits<MTr::W,VL>::element_type,
	    typename mask_logical_traits<vector_traits::W,VL>::element_type, VL>
	    ::convert( m.get() );
	using MTrW = typename MTr::template rebindW<vector_traits::W>::type;
	return bor_assign( r, simd::detail::mask_impl<MTrW>( mw ) );
    }

    template<layout_t Layout1, layout_t Layout2>
    bool cas( vec<vector_traits,Layout1> old, vec<vector_traits,Layout2> val ) {
	static_assert( VL == 1, "only supported for VL=1" );
	// static_assert( layout != lo_unknown, "cannot do CAS in gather/scatter" );
	member_type n = val.at(0); // assumes VL == 1
	member_type o = old.at(0);
	bool r = encoding::template cas<vector_traits>( &m_addr[m_sidx], o, n );
	return r;
    }
    
    template<layout_t Layout_>
    auto atomic_min( vec<vector_traits,Layout_> val ) {
	static_assert( VL == 1, "only supported for VL=1" );

	member_type n = val.at(0); // assumes VL == 1
	member_type oc;
	member_type o;
	bool r = false;
	do {
	    // o = encoding::template load<vector_traits>( m_addr, m_sidx );
	    oc = encoding::template ldcas<vector_traits>( m_addr, m_sidx );
	    o = encoding::template extract<vector_traits>( oc );
	} while( o > n
		 && !(r = encoding::template cas<vector_traits>(
			  m_addr, m_sidx, oc, n ))
	    );
	using L = typename add_logical<member_type>::type;
	return r
	    ? vector<L,1>::true_mask()
	    : vector<L,1>::false_mask();
    }
    template<layout_t Layout_>
    auto atomic_max( vec<vector_traits,Layout_> val ) {
	static_assert( VL == 1, "only supported for VL=1" );
	// static_assert( layout != lo_unknown, "cannot do CAS in gather/scatter" );

	member_type n = val.at(0); // assumes VL == 1
	member_type o;
	bool r = false;
	do {
	    o = encoding::template load<vector_traits>( m_addr, m_sidx );
	} while( o < n
		 && !(r = encoding::template cas<vector_traits>(
			  &m_addr[m_sidx], o, n ))
	    );
	using L = typename add_logical<member_type>::type;
	return r
	    ? vector<L,1>::true_mask()
	    : vector<L,1>::false_mask();
    }
    template<layout_t Layout_>
    auto atomic_add( vec<vector_traits,Layout_> val ) {
	return atomic_add<true>( val );
    }
    template<bool conditional, layout_t Layout_>
    auto atomic_add( vec<vector_traits,Layout_> val ) {
	static_assert( VL == 1, "only supported for VL=1" );
	// static_assert( layout != lo_unknown, "cannot do CAS in gather/scatter" );

	using stored_type = typename encoding::stored_type;

	member_type d = val.at(0); // assumes VL == 1
	stored_type o;
	stored_type n;
	bool r = false;
	if constexpr ( std::is_floating_point_v<member_type> ) {
	    volatile stored_type * base = reinterpret_cast<volatile stored_type *>( m_addr );
	    do {
		o = encoding::template load<vector_traits>( m_addr, m_sidx );
		n = static_cast<stored_type>(
		    static_cast<member_type>( o ) + d );
	    } while( !(r = encoding::template cas<vector_traits>(
			   m_addr, m_sidx, o, n )) );
	} else {
	    volatile member_type * ptr = &m_addr[m_sidx];
	    do {
		o = *ptr;
		n = static_cast<stored_type>(
		    static_cast<member_type>( o ) + d );
	    } while( !(r = encoding::template cas<vector_traits>(
			   m_addr, m_sidx, o, n )) );
	}
	using L = typename add_logical<member_type>::type;
	// TODO: just return true_mask() case?
	if constexpr ( conditional )
	    return r ? vector<L,1>::true_mask() : vector<L,1>::false_mask();
	else
	    return vec<vector_traits,lo_constant>( o );
    }
    template<bool conditional, layout_t Layout_>
    auto atomic_count_down( vec<vector_traits,Layout_> lim ) {
	static_assert( VL == 1, "only supported for VL=1" );
	static_assert( std::is_integral_v<member_type>,
		       "count-down only makes sense for integral types" );

	using stored_type = typename encoding::stored_type;
	using L = typename add_logical<member_type>::type;

	member_type l = lim.at(0); // assumes VL == 1
	member_type d = traits::setone(); // -1
	stored_type o;
	stored_type n;
	bool r = false;
	volatile member_type * ptr = m_addr;
	do {
	    o = ptr[m_sidx];
	    if( o <= l ) {
		if constexpr ( conditional )
		    return vector<L,1>::false_mask();
		else
		    return vec<vector_traits,lo_constant>( o );
	    }
	    n = static_cast<stored_type>(
		static_cast<member_type>( o ) + d );
	} while( !(r = encoding::template cas<vector_traits>(
		       ptr, m_sidx, o, n )) );
	if constexpr ( conditional ) {
	    // Return true if the value was lowered to equal the limit.
	    // If the value was already below or equal to the limit, then
	    // false is returned.
	    // Note: to arrive here with n == l implies that previously the
	    // check o <= l failed.
	    return n == l
		? vector<L,1>::true_mask()
		: vector<L,1>::false_mask();
	} else
	    return vec<vector_traits,lo_constant>( o );
    }
    template<layout_t Layout_>
    auto atomic_setif( vec<vector_traits,Layout_> val ) {
	static_assert( VL == 1, "only supported for VL=1" );
	// static_assert( layout != lo_unknown, "cannot do CAS in gather/scatter" );

	member_type n = val.at(0); // assumes VL == 1
	member_type o;
	bool r = false;
	do {
	    o = m_addr[m_sidx];
	} while( ~o == (VID)0
		 && !(r = encoding::template cas<vector_traits>(
			  m_addr, m_sidx, o, n )) );
	using L = typename add_logical<member_type>::type;
	return r
	    ? vector<L,1>::true_mask()
	    : vector<L,1>::false_mask();
    }
    template<layout_t Layout_>
    auto atomic_logicalor( vec<vector_traits,Layout_> val ) {
	static_assert( VL == 1, "only supported for VL=1" );
	// static_assert( layout != lo_unknown, "cannot do CAS in gather/scatter" );

	// TODO: as this is logicalor, just overwrite with true

	member_type nn = val.at(0); // assumes VL == 1
	member_type o, oc, n;
	bool r = false;
	do {
	    // o = encoding::template load<vector_traits>( m_addr, m_sidx );
	    oc = encoding::template ldcas<vector_traits>( m_addr, m_sidx );
	    o = encoding::template extract<vector_traits>( oc );
	    n = o | nn;
	} while( o != n
		 && !(r = encoding::template cas<vector_traits>(
			  m_addr, m_sidx, oc, n ))
		 // && !(r = encoding::template cas<vector_traits>(
		 // &m_addr[m_sidx], o, n ))
	    );
	using L = typename add_logical<member_type>::type;
	return r
	    ? vector<L,1>::true_mask()
	    : vector<L,1>::false_mask();
    }
    template<layout_t Layout_>
    auto atomic_bitwiseor( vec<vector_traits,Layout_> val ) {
	return atomic_logicalor( val );
    }
    template<layout_t Layout_>
    auto atomic_logicaland( vec<vector_traits,Layout_> val ) {
	static_assert( VL == 1, "only supported for VL=1" );
	// TODO: as this is logicaland, just overwrite with false

	member_type nn = val.at(0); // assumes VL == 1
	member_type o, n;
	bool r = false;
	do {
	    o = encoding::template load<vector_traits>( m_addr, m_sidx );
	    n = o && nn;
	} while( o != n
		 && !(r = encoding::template cas<vector_traits>(
			  &m_addr[m_sidx], o, n ))
	    );
	using L = typename add_logical<member_type>::type;
	return r
	    ? vector<L,1>::true_mask()
	    : vector<L,1>::false_mask();
    }
    template<layout_t Layout_>
    auto atomic_bitwiseand( vec<vector_traits,Layout_> val ) {
	static_assert( VL == 1, "only supported for VL=1" );

	member_type nn = val.at(0); // assumes VL == 1
	member_type o, oc, n;
	bool r = false;
	do {
	    // o = encoding::template load<vector_traits>( m_addr, m_sidx );
	    oc = encoding::template ldcas<vector_traits>( m_addr, m_sidx );
	    o = encoding::template extract<vector_traits>( oc );
	    n = o & nn;
	} while( o != n
		 && !(r = encoding::template cas<vector_traits>(
			  m_addr, m_sidx, oc, n ))
	    );
	using L = typename add_logical<member_type>::type;
	return r
	    ? vector<L,1>::true_mask()
	    : vector<L,1>::false_mask();
    }

    // Generic interface
    /*GG_INLINE*/ inline vector_type load( nomask<VL> = nomask<VL>() ) const {
	if constexpr ( NT ) {
	    return ntload();
	} else {
	    return tload();
	}
    }
    /*GG_INLINE*/ inline vector_type load( simd_vmask_type mask ) const {
	if constexpr ( NT ) {
	    return ntload( mask );
	} else {
	    return tload( mask );
	}
    }
    template<typename MTr>
    /*GG_INLINE*/ inline vector_type load( mask_impl<MTr> mask ) const {
	if constexpr ( NT ) {
	    return ntload( mask );
	} else {
	    return tload( mask );
	}
    }
    [[deprecated("seems there is no need for this method")]]
    GG_INLINE vector_type loadu() const {
	static_assert( NT, "no unaligned non-temporal loads supported" );
	return tloadu();
    }
    [[deprecated("seems there is no need for this method")]]
    GG_INLINE vector_type loadu( simd_vmask_type mask ) const {
	static_assert( NT, "no unaligned non-temporal loads supported" );
	return tloadu( mask );
    }

    // Temporal loads
    GG_INLINE vector_type tload( nomask<VL> = nomask<VL>() ) const {
	if constexpr ( m_layout == lo_unknown )
	    return vector_type(
		encoding::template gather<vector_traits>( m_addr, m_vidx ) );
	if constexpr ( m_layout == lo_constant )
	    return vector_type(
		traits::set1( member_type(
				  encoding::template loadu<scalar_traits>(
				      m_addr, m_sidx ) ) ) );
	if constexpr ( m_layout == lo_linear )
	    return vector_type( encoding::template loadu<vector_traits>( m_addr, m_sidx ) );
	if constexpr ( m_layout == lo_linalgn )
	    return vector_type( encoding::template load<vector_traits>( m_addr, m_sidx ) );
	UNREACHABLE_CASE_STATEMENT;
    }

    template<typename Tr>
    GG_INLINE vector_type tload( mask_impl<Tr> mask ) const {
	if constexpr ( m_layout == lo_unknown )
	    return vector_type( encoding::template gather<vector_traits>(
				    m_addr, m_vidx, mask.get() ) );
	else
	    return tload();
    }

    GG_INLINE vector_type tloadu() const {
	if constexpr ( m_layout == lo_unknown )
	    return vector_type(
		encoding::template gather<vector_traits>( m_addr, m_vidx ) );
	if constexpr ( m_layout == lo_constant )
	    return vector_type(
		traits::set1( member_type(
				  encoding::template load<scalar_traits>(
				      m_addr, m_sidx ) ) ) );
	if constexpr ( m_layout == lo_linear || m_layout == lo_linalgn )
	    return vector_type(
		encoding::template loadu<vector_traits>( m_addr, m_sidx ) );
	UNREACHABLE_CASE_STATEMENT;
    }

    GG_INLINE vector_type tloadu( simd_vmask_type mask ) const {
	if constexpr ( m_layout == lo_unknown )
	    return vector_type( encoding::template gather<vector_traits>(
				    m_addr, m_vidx, mask.data() ) );
	else
	    return tloadu();
    }
    
    // Non-temporal loads
    GG_INLINE vector_type ntload( nomask<VL> = nomask<VL>() ) const {
	if constexpr ( m_layout == lo_unknown )
	    return vector_type( encoding::template gather<vector_traits>(
				    m_addr, m_vidx ) );
	if constexpr ( m_layout == lo_linear )
	    // Linear will result in segmentation fault if not aligned,
	    // except for scalar operations, where linear, constant and
	    // linalgn all resolve to the same.
	    if constexpr ( VL != 1 )
		assert( 0 && "non-temporal loads should be aligned" );

	// fall-through to lo_constant
	if constexpr ( m_layout == lo_constant )
	    return vector_type(
		traits::set1( member_type(
				  encoding::template ntload<scalar_traits>(
				      m_addr, m_sidx ) ) ) );
	if constexpr ( m_layout == lo_linalgn )
	    return vector_type( encoding::template ntload<vector_traits>( m_addr, m_sidx ) );
	else
	    UNREACHABLE_CASE_STATEMENT;
    }

    template<typename Tr>
    GG_INLINE vector_type ntload( mask_impl<Tr> mask ) const {
	if( m_layout == lo_unknown )
	    return vector_type( encoding::template gather<vector_traits>(
				    m_addr, m_vidx, mask.get() ) );
	else
	    return ntload();
    }

#if 0
    template<_mm_hint HINT>
    GG_INLINE inline void
    prefetch( nomask<VL> = nomask<VL>() ) const {
	// Should call methods through encoding rather than directly in
	// traits
	static_assert(
	    std::is_same_v<encoding::stored_type,encoding::storage_type>,
	    "this method has not been updated for non-C array encoding" );

	switch( m_layout ) {
	case lo_unknown:
	    traits::template prefetch_gather<HINT>( m_addr, m_vidx );
	    break;
	case lo_constant:
	    using traits1 = vector_type_traits_vl<member_type, 1>;
	    _mm_prefetch( m_addr + m_sidx, HINT );
	    break;
	case lo_linear:
	case lo_linalgn:
	    _mm_prefetch( m_addr + m_sidx, HINT );
	    break;
	default:
	    UNREACHABLE_CASE_STATEMENT;
	}
    }

    template<_mm_hint HINT, typename Tr>
    GG_INLINE inline void
    prefetch( mask_impl<Tr> mask ) const {
	// Should call methods through encoding rather than directly in
	// traits
	static_assert(
	    std::is_same_v<encoding::stored_type,encoding::storage_type>,
	    "this method has not been updated for non-C array encoding" );

	switch( m_layout ) {
	case lo_unknown:
	    traits::template prefetch_gather<HINT>( m_addr, m_vidx, mask.get() );
	    break;
	case lo_constant:
	    using traits1 = vector_type_traits_vl<member_type, 1>;
	    _mm_prefetch( m_addr + m_sidx, HINT );
	    break;
	case lo_linear:
	case lo_linalgn:
	    _mm_prefetch( m_addr + m_sidx, HINT );
	    break;
	default:
	    UNREACHABLE_CASE_STATEMENT;
	}
    }
#endif

    // General interface storing temporal/non-temporal depending on NT flag
    // Note: Layout_ should be compatible with Layout if m_addr points
    //       to simd::vec?
    template<layout_t Layout_>
    /*GG_INLINE*/ inline
    void store( vec<vector_traits,Layout_> val, nomask<VL> m = nomask<VL>() ) {
	if constexpr ( NT ) {
	    ntstore( val, m );
	} else {
	    tstore( val, m );
	}
    }
    template<typename MTr, layout_t Layout_>
    /*GG_INLINE*/ inline void
    store( vec<vector_traits,Layout_> val, mask_impl<MTr> mask,
	   typename std::enable_if<simd::matchVLtu<MTr,VL>::value>::type *
	   = nullptr ) {
	if constexpr ( NT ) {
	    ntstore( val, mask );
	} else {
	    tstore( val, mask );
	}
    }
    /*GG_INLINE*/ inline
    void store( mask_impl<vector_traits> val, nomask<VL> m = nomask<VL>() ) {
	auto v = vec<vector_traits,lo_unknown>( val.data() );
	if constexpr ( NT ) {
	    ntstore( v, m );
	} else {
	    tstore( v, m );
	}
    }
    template<typename MTr>
    /*GG_INLINE*/ inline void
    store( mask_impl<vector_traits> val, mask_impl<MTr> mask,
	   typename std::enable_if<simd::matchVLtu<MTr,VL>::value>::type *
	   = nullptr ) {
	auto v = vec<vector_traits,lo_unknown>( val.data() );
	if constexpr ( NT ) {
	    ntstore( v, mask );
	} else {
	    tstore( v, mask );
	}
    }
    // Temporal store
    template<layout_t Layout_>
    GG_INLINE inline
    void tstore( vec<vector_traits,Layout_> val, nomask<VL> = nomask<VL>() ) {
	if constexpr ( m_layout == lo_unknown ) {
	    encoding::template scatter<vector_traits>( m_addr, m_vidx, val.data() );
	    return;
	}
	if constexpr ( m_layout == lo_constant ) {
	    if constexpr ( !is_array_encoding_zero_v<encoding> ) {
		assert( VL == 1 && "Don't know which value to store" );
		encoding::template store<vector_traits>( m_addr, m_sidx, val.data() );
	    }
	    return;
	}
	if constexpr ( m_layout == lo_linear ) {
	    encoding::template storeu<vector_traits>( m_addr, m_sidx, val.data() );
	    return;
	}
	if constexpr ( m_layout == lo_linalgn ) {
	    encoding::template store<vector_traits>( m_addr, m_sidx, val.data() );
	    return;
	}
	UNREACHABLE_CASE_STATEMENT;
    }

    template<typename MTr, layout_t Layout_>
    GG_INLINE inline void
    tstore( vec<vector_traits,Layout_> val, mask_impl<MTr> mask,
	    typename std::enable_if<simd::matchVLtu<MTr,VL>::value>::type *
	    = nullptr ) {
	if constexpr ( m_layout == lo_unknown ) {
	    encoding::template scatter<vector_traits>( m_addr, m_vidx, val.data(), mask.get() );
	    return;
	}
	if constexpr ( m_layout == lo_constant ) {
	    if constexpr ( !is_array_encoding_zero_v<encoding> ) {
		assert( VL == 1 );
		using mtraits = typename mask_impl<MTr>::traits;
		if( mtraits::cmpne( mask.data(), mtraits::setzero(), target::mt_bool() ) )
		    encoding::template store<vector_traits>( m_addr, m_sidx, val.data() );
	    }
	    return;
	}
	if constexpr ( m_layout == lo_linear ) {
	    // TODO: for an encoded array, it may be more efficient to do
	    //       the blend in the encoded domain (e.g., using bit
	    //       manipulation) rather than converting to and fro.
	    type old = encoding::template loadu<vector_traits>( m_addr, m_sidx );
	    type upd = traits::blend( mask.data(), old, val.data() );
	    encoding::template storeu<vector_traits>( m_addr, m_sidx, upd );
	    return;
	}
	if constexpr ( m_layout == lo_linalgn ) {
/*
	    type old = encoding::template load<vector_traits>( m_addr, m_sidx );
	    type upd = traits::blend( mask.data(), old, val.data() );
	    encoding::template store<vector_traits>( m_addr, m_sidx, upd );
*/
	    encoding::template store<vector_traits,MTr>( m_addr, m_sidx, val.data(), mask.data() );
	    return;
	}
	UNREACHABLE_CASE_STATEMENT;
    }

    // Non-temporal store
    template<layout_t Layout_>
    GG_INLINE inline
    void ntstore( vec<vector_traits,Layout_> val, nomask<VL> = nomask<VL>() ) {
	switch( m_layout ) {
	case lo_unknown:
	    encoding::template scatter<vector_traits>(
		m_addr, m_vidx, val.data() );
	    break;
	case lo_linear:
	    // Linear will result in segmentation fault if not aligned,
	    // except for scalar operations, where linear, constant and
	    // linalgn all resolve to the same.
	    if constexpr ( VL != 1 )
		assert( 0 && "non-temporal stores should be aligned" );
	    // fall-through to lo_constant
	case lo_constant:
	    if constexpr ( !is_array_encoding_zero_v<encoding> ) {
		assert( VL == 1 && "Don't know which value to store" );
		encoding::template ntstore<vector_traits>( m_addr, m_sidx, val.data() );
	    }
	    break;
	case lo_linalgn:
	    encoding::template ntstore<vector_traits>( m_addr, m_sidx, val.data() );
	    break;
	default:
	    UNREACHABLE_CASE_STATEMENT;
	}
    }

    template<typename MTr, layout_t Layout_>
    GG_INLINE inline void
    ntstore( vec<vector_traits,Layout_> val, mask_impl<MTr> mask,
	   typename std::enable_if<simd::matchVLtu<MTr,VL>::value>::type *
	   = nullptr ) {
	tstore( val, mask ); // cannot do selective non-temporal.
    }

    type data() const { return load(); }
    member_type lane0() { return traits::lane0( data() ); }

    static vec<vector_traits,lo_unknown> make_vector( type val ) {
	return vec<vector_traits,lo_unknown>( val );
    }

private:
    storage_type *m_addr;
    vindex_type m_vidx;
    index_type m_sidx;
    static constexpr layout_t m_layout = Layout;
};

// Overload for lo_variable layout - not supported - empty definition
template<class Traits, typename I, typename Encoding, bool NT_>
class vector_ref_impl<Traits,I,Encoding,NT_,lo_variable> { };

// TODO: bitmask ref does not yet support non-temporal hint
/*
template<unsigned short VL, typename I, typename Encoding, bool NT,
	 layout_t Layout>
class vector_ref_impl<mask_bit_traits<VL>,I,Encoding,NT,Layout>
    : public mask_ref_impl<mask_bit_traits<VL>,I> {
    // Copy over constructors
    using mask_ref_impl<mask_bit_traits<VL>,I>::mask_ref_impl;
};
*/

} // namespace detail

template<typename T, typename I, unsigned short VL,
	 typename Encoding = array_encoding<T>, bool NT = false,
	 layout_t Layout = lo_unknown>
using vector_ref = detail::vector_ref_impl<detail::vdata_traits<T,VL>, I,
					   Encoding, NT, Layout>;

/**********************************************************************
 * Creation methods
 **********************************************************************/
template<class Tr, typename I, typename Enc, bool NT, layout_t Layout>
auto create_vector_ref_scalar(
    typename detail::vector_ref_impl<Tr,I,Enc,NT,Layout>::storage_type* addr,
    I idx,
    std::enable_if_t<Layout == lo_linalgn || Layout == lo_linear
    || Layout == lo_constant> * ) {
    static_assert( Layout != lo_variable, "ruling out lo_variable" );
    return detail::vector_ref_impl<Tr,I,Enc,NT,Layout>( addr, idx );
}

template<class Tr, typename I, typename Enc, bool NT, layout_t Layout>
auto create_vector_ref_vec(
    typename Enc::storage_type* addr,
    const simd::detail::vec<simd::ty<I,Tr::VL>,Layout> & vidx ) {
    static_assert( Layout != lo_variable, "ruling out lo_variable" );
    // assert( vidx.get_layout() == Layout );
    if constexpr ( Tr::VL == 1 )
	return create_vector_ref_scalar<Tr,I,Enc,NT,lo_constant>(
	    addr, vidx.data() );
    else
	return detail::vector_ref_impl<Tr,I,Enc,NT,Layout>( addr, vidx );
}

template<class Tr, typename I, typename Enc, bool NT, layout_t Layout>
auto create_vector_ref_cacheop(
    vec<Tr,Layout> *v,
    typename detail::vector_ref_impl<Tr,I,Enc,NT,lo_linalgn>::index_type idx ) {
    static_assert( Layout != lo_variable, "ruling out lo_variable" );
    return create_vector_ref_cacheop<Tr,I,Enc,NT>(
	reinterpret_cast<typename detail::vector_ref_impl<Tr,I,Enc,NT,lo_linalgn>::storage_type *>( &v->data() ),
	idx );
}

template<class Tr, typename I, typename Enc, bool NT>
auto create_vector_ref_cacheop(
    typename detail::vector_ref_impl<Tr,I,Enc,NT,lo_linalgn>::storage_type* addr,
    typename detail::vector_ref_impl<Tr,I,Enc,NT,lo_linalgn>::index_type idx ) {
    return detail::vector_ref_impl<Tr,I,Enc,NT,lo_linalgn>(
	addr, idx );
}

} // namespace simd

#endif // GRAPTOR_SIMD_VECTOR_REF_H
