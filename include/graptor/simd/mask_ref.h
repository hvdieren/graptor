// -*- c++ -*-

#ifndef GRAPTOR_SIMD_MASK_REF_H
#define GRAPTOR_SIMD_MASK_REF_H

#include "graptor/simd/decl.h"
#include "graptor/simd/mask.h"
#include "graptor/target/vector.h"

namespace simd {

/***********************************************************************
 * mask_ref: reference to mask
 ***********************************************************************/
namespace detail {
template<class Traits, typename I>
class mask_ref_impl {
public:
    static constexpr unsigned short VL = Traits::VL;

    using mask_traits = Traits;
    using traits = typename mask_traits::traits;
    using type = typename mask_traits::type;
    using pointer_type = typename mask_traits::pointer_type;

    using itraits = vector_type_traits_vl<I, VL>;
    using index_type = I;
    using vindex_type = typename itraits::type;

    using data_type = mask_impl<Traits>;
    using data1_type = mask_impl<typename Traits::template rebindVL<1>::type>;
    
    using self_type = mask_ref_impl<Traits, I>;

    // using encoding = array_encoding<typename mask_traits::pointer_type>;
    using encoding = array_encoding_bit<1>;
    using storage_type = typename encoding::storage_type;

public:
    template<typename U>
    GG_INLINE
    mask_ref_impl( pointer_type * addr, U idx,
		   typename std::enable_if<std::is_same<std::make_unsigned<U>, std::make_unsigned<index_type>>::value>::type * = nullptr )
	: m_addr( (storage_type*)addr ), m_sidx( idx ), m_layout( lo_linalgn ) { }

    template<typename U>
    GG_INLINE
    mask_ref_impl( pointer_type * addr, U vidx, layout_t layout = lo_unknown,
		   typename std::enable_if<std::is_same<std::make_unsigned<U>, std::make_unsigned<vindex_type>>::value && VL != 1>::type * = nullptr )
	: m_addr( (storage_type*)addr ), m_vidx( vidx ), m_layout( layout ) { }

    template<layout_t Layout>
    GG_INLINE
    mask_ref_impl( void * addr,
		   vec<vdata_traits<index_type,VL>,Layout> vidx ) 
	: m_addr( reinterpret_cast<storage_type *>( addr ) ),
	  m_vidx( vidx.data() ), m_layout( vidx.get_layout() ) { }

    GG_INLINE
    mask_ref_impl( mask_impl<Traits> *v, index_type i )
	: m_addr( (storage_type*)&v->get() ), m_sidx( i ),
	  m_layout( lo_linalgn ) { }

    GG_INLINE
    mask_ref_impl( const mask_ref_impl<Traits,I> & r )
	: m_addr( r.m_addr ), m_vidx( r.m_vidx ), m_layout( r.m_layout ) { }

    // Querying and setting layout
    bool is_linear() const { return m_layout != lo_unknown; }
    layout_t get_layout() const { return m_layout; }
    
    // Accessing through the reference
    GG_INLINE data_type load() const {
	switch( m_layout ) {
	case lo_unknown:
	    return data_type( encoding::gather<mask_traits>( m_addr, m_vidx ) );
	case lo_constant:
	{
	    using traits1 = typename data1_type::traits;
	    return data_type(
		traits::set1( typename traits1::member_type(
				  traits::loads( m_addr, m_sidx ) ) ) );
	}
	case lo_linear:
	    return data_type( encoding::loadu<mask_traits>( m_addr, m_sidx ) );
	case lo_linalgn:
	    return data_type( encoding::load<mask_traits>( m_addr, m_sidx ) );
	default:
	    UNREACHABLE_CASE_STATEMENT;
	}
    }

    template<unsigned short VL_>
    GG_INLINE data_type load( simd::nomask<VL_> ) const {
	return load();
    }

    template<typename MTr>
    GG_INLINE
    typename std::enable_if<matchVLtt<mask_traits,MTr>::value, data_type>::type
    load( mask_impl<MTr> m ) const {
	// The gather case requires the mask; the other cases are ok as long
	// as at least one of the lanes is active
	switch( m_layout ) {
	case lo_unknown:
	    return data_type( encoding::gather<mask_traits>( m_addr, m_vidx, m.data() ) );
	case lo_constant:
	{
	    using traits1 = typename data1_type::traits;
	    return data_type(
		traits::set1( typename traits1::member_type(
				  traits::loads( m_addr, m_sidx ) ) ) );
	}
	case lo_linear:
	    return data_type( encoding::loadu<mask_traits>( m_addr, m_sidx ) );
	case lo_linalgn:
	    return data_type( encoding::load<mask_traits>( m_addr, m_sidx ) );
	default:
	    UNREACHABLE_CASE_STATEMENT;
	}
    }

    mask_impl<mask_traits> bor_assign( mask_impl<mask_traits> r ) {
	auto a = load(); // load data
	auto mod = ~a & r;
	store( a | r ); // store data
	auto v = mod;
	auto z = mask_impl<mask_traits>::false_mask();
	return (v != z).asmask();
    }

    mask_impl<mask_traits>
    bor_assign( mask_impl<mask_traits> r, mask_impl<mask_traits> m ) {
	auto a = load(); // load data
	auto mod = ~a & r;
	store( a | r, m ); // store data
	auto v = mod;
	return v & m;
	auto z = mask_impl<mask_traits>::false_mask();
	return (v != z).asmask() & m;
    }

    GG_INLINE
    void store( mask_impl<mask_traits> val, nomask<VL> = nomask<VL>() ) {
	switch( m_layout ) {
	case lo_unknown:
	    encoding::scatter<mask_traits>( m_addr, m_vidx, val.get() );
	    break;
	case lo_constant:
	    assert( VL == 1 );
	    encoding::store<mask_traits>( m_addr, m_sidx, val.get() );
	    break;
	case lo_linear:
	    encoding::storeu<mask_traits>( m_addr, m_sidx, val.get() );
	    break;
	case lo_linalgn:
	    encoding::store<mask_traits>( m_addr, m_sidx, val.get() );
	    break;
	default:
	    UNREACHABLE_CASE_STATEMENT;
	}
    }

    template<typename MTr>
    GG_INLINE inline void
    store( mask_impl<mask_traits> val, mask_impl<MTr> mask,
	   typename std::enable_if<simd::matchVLtu<MTr,VL>::value>::type *
	   = nullptr ) {
	switch( m_layout ) {
	case lo_unknown:
	    encoding::scatter<mask_traits>( m_addr, m_vidx, val.get(), mask.get() );
	    break;
	case lo_constant:
	    assert( VL == 1 );
	    encoding::store<mask_traits>( m_addr, m_sidx, val.get() );
	    break;
	case lo_linear:
	    assert( 0 && "TODO: Seems wrong" );
	    encoding::storeu<mask_traits>( m_addr, m_sidx, val.get() );
	    break;
	case lo_linalgn:
	    assert( 0 && "TODO: Seems wrong" );
	    encoding::store<mask_traits>( m_addr, m_sidx, val.get() );
	    break;
	default:
	    UNREACHABLE_CASE_STATEMENT;
	}
    }
 
private:
    storage_type *m_addr;
    union {
	vindex_type m_vidx;
	index_type m_sidx;
    };
    const layout_t m_layout;
};

} // namespace detail

template<unsigned short W, typename I, unsigned short VL>
using mask_ref = detail::mask_ref_impl<detail::mask_preferred_traits_width<W,VL>, I>;

} // namespace simd

#endif // GRAPTOR_SIMD_MASK_REF_H
