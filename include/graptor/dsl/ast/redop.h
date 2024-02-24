// -*- c++ -*-
#ifndef GRAPTOR_DSL_AST_REDOP_H
#define GRAPTOR_DSL_AST_REDOP_H

namespace expr {

/**
 * redop: A reduction operation.
 */
template<typename E1, typename E2, typename RedOp>
struct redop : public expr_base {
    static constexpr unsigned short VL = std::max( E1::VL, E2::VL );
    static constexpr op_codes opcode = op_redop;

    static_assert( std::is_class<RedOp>::value, "RedOp must be a class" );
    static_assert( is_refop<E1>::value || is_maskrefop<E1>::value
		   || is_cacheop<E1>::value
		   || is_masked_cacheop<E1>::value,
		   "RedOp::E1 must be one of refop, maskrefop, cacheop" );

    // Assumptions: L is a refop/maskrefop/cacheop;
    //              T is compatible with the type of *L
    static_assert( RedOp::template enable<E1, E2>::value,
		   "RedOp enable condition must be true" );

    using data_type = typename RedOp::template types<E1,E2>::result_type;
    using type = typename data_type::member_type;
    using ref_type = E1;
    using val_type = E2;
    using redop_type = RedOp;

    GG_INLINE redop( ref_type ref, val_type val, RedOp ) : m_ref( ref ), m_val( val ) { }

    // Unit is expressed using the data_type of the left argument
    static constexpr auto unit() { return redop_type::template unit<typename ref_type::data_type>(); }

    GG_INLINE const ref_type & ref() const { return m_ref; }
    GG_INLINE const val_type & val() const { return m_val; }

private:
    ref_type m_ref;
    val_type m_val;
};

template<typename RedOp,
	 typename VTr, layout_t Layout1,
	 typename MTr, layout_t Layout2>
static auto
set_unit_if_disabled( sb::rvalue<VTr,Layout1> v,
		      sb::rvalue<MTr,Layout2> m ) {
    auto u = RedOp::template vunit<VTr>();
    return make_rvalue( ::iif( m.value(), u, v.value() ), sb::create_mask_pack() );
}

template<typename E1, typename E2, typename RedOp>
static constexpr
auto make_redop( E1 e1, E2 e2, RedOp op ) {
    if constexpr ( e2.opcode == op_constant ) {
	using Tr = typename RedOp::template types<E1,E2>::infer_type;
	return make_redop( e1, e2.template expand<Tr>(), op );
    } else
	return redop<E1,E2,RedOp>( e1, e2, op );
}

/* redop: logical or
 */

struct redop_logicalor {
    template<typename E1, typename E2>
    struct types {
	using result_type = typename E1::data_type::prefmask_traits;
	// TODO: prefmask_traits?
	// TODO: replaced to E1 to expand A[v] |= _true
	using cache_type = typename add_logical<typename E1::type>::type;
	using infer_type = typename E1::data_type;
    };

    static constexpr bool is_idempotent = true;
    static constexpr bool is_benign_race = false; // load-modify-store
    static constexpr bool is_single_trigger = false;

    // TODO: should have an ID here instead of string
    static constexpr char const * name = "redop_logicalor";

    template<typename E1, typename E2>
    using enable = std::true_type; // enable_if_compatible<U1,U2>;

    // template<typename T>
    // static constexpr auto zero() { return ~T(0); }

    template<typename T>
    static constexpr auto unit() { return typename T::element_type(0); }

    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename I, typename Enc, bool NT, typename MPack>
    static auto
    evaluate( sb::lvalue<VTr,I,Enc,NT,Layout1> l,
	      sb::rvalue<VTr,Layout2> r,
	      const MPack & mpack ) {
	// Apply the mask in the addition operation (ALU), perform a full
	// store. Alternative: only store back relevant values.
	// Best situation may depend on whether the lvalue is linear or not.
	if constexpr ( MPack::is_empty() ) {
	    return make_rvalue( l.value().lor_assign( r.value() ), mpack );
	} else {
	    using MTr = typename VTr::prefmask_traits;
	    auto mask = mpack.template get_mask<MTr>();
	    return make_rvalue( l.value().lor_assign( r.value(), mask ),
				mpack );
	} 
    }


    template<typename VTr, typename MTr1, typename MTr2, typename I,
	     typename Enc, bool NT, layout_t LayoutR, layout_t Layout>
    static auto
    evaluate( lvalue<VTr,I,MTr1,Enc,NT,LayoutR> l, rvalue<VTr,Layout,MTr2> r,
	      std::enable_if_t<simd::matchVL_<VTr::VL,MTr1,MTr2>::value> *
	      = nullptr ) {
	return make_rvalue( l.value().lor_assign( r.value(), l.mask() & r.mask() ) );
    }

    template<typename VTr, typename MTr1, typename MTr2, typename I,
	     typename Enc,  bool NT, layout_t LayoutR, layout_t Layout>
    static auto
    evaluate( lvalue<VTr,I,MTr1,Enc,NT,LayoutR> l, rvalue<void,Layout,MTr2> r,
	      typename std::enable_if<simd::matchVLttt<VTr,MTr1,MTr2>::value>::type * = nullptr ) {
	auto rm = r.mask().template asvector<typename VTr::member_type>();
	// Or with rvalue rm, under the mask specified by l
	return make_rvalue( l.value().lor_assign( rm, l.mask() ) );
    }

    template<typename VTr, typename MTr, typename I,
	     typename Enc,  bool NT, layout_t LayoutR, layout_t Layout>
    static auto
    evaluate( lvalue<VTr,I,void,Enc,NT,LayoutR> l, rvalue<void,Layout,MTr> r,
	      typename std::enable_if<simd::matchVLtt<VTr,MTr>::value>::type * = nullptr ) {
	auto m = r.mask().template asvector<typename VTr::member_type>();
	return make_rvalue( l.value().lor_assign( m ) );
    }

    // Specialising the code to void mask or not makes the code more efficient
    template<typename VTr, layout_t Layout>
    static GG_INLINE auto
    evaluate1( rvalue<VTr,Layout,void> r,
	       typename std::enable_if<!std::is_void<VTr>::value>::type * = nullptr ) {
	return make_rvalue( reduce_logicalor( r.value() ) );
    }
    template<typename MTr, layout_t Layout>
    static GG_INLINE auto
    evaluate1( rvalue<void,Layout,MTr> r,
	       typename std::enable_if<!std::is_void<MTr>::value>::type * = nullptr ) {
	return make_rvalue( reduce_logicalor( r.mask() ) );
    }
    template<typename VTr, typename MTr, layout_t Layout>
    static GG_INLINE auto
    evaluate1( rvalue<VTr,Layout,MTr> r,
	       typename std::enable_if<simd::matchVLtt<VTr,MTr>::value>::type * = nullptr ) {
	return make_rvalue( reduce_logicalor( r.value(), r.mask() ) );
    }
    template<typename VTr, layout_t Layout, typename MPack>
    static auto
    evaluate1( sb::rvalue<VTr,Layout> r, const MPack & mpack ) {
	auto v = r.value();
	auto m = mpack.get_mask_for( v );
	return make_rvalue( reduce_logicalor( v, m ), sb::mask_pack<>() );
    }

    template<typename VTr, typename MTr1, typename MTr2, typename I,
	     typename Enc,  bool NT, layout_t LayoutR, layout_t Layout>
    static GG_INLINE auto
    evaluate_atomic( lvalue<VTr,I,MTr1,Enc,NT,LayoutR> l, rvalue<VTr,Layout,MTr2> r,
		     std::enable_if_t<
		     simd::matchVLttotu<VTr,MTr1,MTr2,1>::value> * = nullptr ) {
	auto mask = l.mask() & r.mask();
	if( mask.at(0) ) {
	    auto val = l.value().atomic_logicalor( r.value() );
	    return make_rvalue( val && mask );
	} else {
	    auto zval = decltype(l.value())::false_mask(); // zero_val();
	    return make_rvalue( zval && mask );
	}
    }
    template<typename VTr, typename I,
	     typename Enc,  bool NT, layout_t LayoutR, layout_t Layout>
    static GG_INLINE auto
    evaluate_atomic( lvalue<VTr,I,void,Enc,NT,LayoutR> l, rvalue<VTr,Layout,void> r,
		     typename std::enable_if<
		     simd::matchVLtu<VTr,1>::value>::type * = nullptr ) {
	auto val = l.value().atomic_logicalor( r.value() );
	return make_rvalue( val );
    }
    template<typename VTr, typename I,
	     typename Enc,  bool NT, layout_t LayoutR, layout_t Layout>
    static GG_INLINE auto
    evaluate_atomic( lvalue<VTr,I,void,Enc,NT,LayoutR> l, rvalue<void,Layout,simd::detail::mask_bool_traits> r,
		     typename std::enable_if<
		     simd::matchVLtu<VTr,1>::value>::type * = nullptr ) {
	auto v = r.mask().template asvector<typename VTr::member_type>();
	auto val = l.value().atomic_logicalor( v );
	return make_rvalue( val );
    }
    template<typename VTr, typename I, typename Enc,  bool NT, layout_t LayoutR,
	     layout_t Layout, typename MPack>
    static auto
    evaluate_atomic( sb::lvalue<VTr,I,Enc,NT,LayoutR> l,
		     sb::rvalue<VTr,Layout> r,
		     const MPack & mpack ) {
	auto rval = r.value();
	auto mask = mpack.template get_any<simd::detail::mask_bool_traits>();
	if( mask.data() ) {
	    auto val = l.value().atomic_logicalor( rval );
	    return make_rvalue( val, mpack );
	} else {
	    // Mask disabled -> no change in the value
	    auto mfalse = simd::detail::mask_impl<simd::detail::mask_bool_traits>
		::false_mask();
	    return make_rvalue( mfalse, mpack );
	}
    }
};


template<typename E1, typename E2>
auto
make_redop_lor( E1 l, E2 r,
		std::enable_if_t<is_logical_type<typename E1::type>::value>
		* = nullptr ) {
    return make_redop( l, r, redop_logicalor() );
}

template<typename E1, typename E2>
auto operator |= ( E1 l, E2 r ) -> decltype(make_redop_lor(l,r)) {
    return make_redop_lor( l, r );
}

/* redop: bitwise or
 */

struct redop_bitwiseor {
    template<typename E1, typename E2>
    struct types {
	// using result_type = typename add_logical<typename E1::type>::type; // bool-like;
	// using result_type = typename E1::type; // bool-like;
	using result_type = typename E1::data_type::prefmask_traits;
	using cache_type = typename E2::type;
	using infer_type = typename E1::data_type;
    };

    static constexpr bool is_idempotent = true;
    static constexpr bool is_benign_race = false; // load-modify-store
    static constexpr bool is_single_trigger = false;

    // TODO: should have an ID here instead of string
    static constexpr char const * name = "redop_bitwiseor";

    template<typename E1, typename E2>
    using enable = std::true_type; // enable_if_compatible<U1,U2>;

    template<typename T>
    static constexpr auto unit() { return typename T::element_type(0); }

    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename I, typename Enc, bool NT, typename MPack>
    static auto
    evaluate( sb::lvalue<VTr,I,Enc,NT,Layout1> l,
	      sb::rvalue<VTr,Layout2> r,
	      const MPack & mpack ) {
	// Apply the mask in the addition operation (ALU), perform a full
	// store. Alternative: only store back relevant values.
	// Best situation may depend on whether the lvalue is linear or not.
	if constexpr ( MPack::is_empty() ) {
	    return make_rvalue( l.value().bor_assign( r.value() ), mpack );
	} else {
	    using MTr = typename VTr::prefmask_traits;
	    auto mask = mpack.template get_mask<MTr>();
	    return make_rvalue( l.value().bor_assign( r.value(), mask ),
				mpack );
	} 
    }

#if 0
    template<typename VTr, typename MTr1, typename MTr2, typename I,
	     typename Enc,  bool NT, layout_t LayoutR, layout_t Layout>
    static auto
    evaluate( lvalue<VTr,I,MTr1,Enc,NT,LayoutR> l, rvalue<VTr,Layout,MTr2> r,
	      typename std::enable_if<
	      simd::matchVLttot<VTr,MTr1,MTr2>::value>::type * = nullptr ) {
	auto m = join_mask<VTr>( l.mask(), r.mask() );
	using MTr = typename decltype(m)::mask_traits;
	if constexpr ( simd::detail::is_mask_logical_traits<MTr>::value
		       && MTr::W != VTr::W ) {
	    auto mm = m.template convert_width<VTr::W>();
	    return make_rvalue( l.value().bor_assign( r.value(), mm ) );
	} else
	    return make_rvalue( l.value().bor_assign( r.value(), m ) );
    }

    template<typename VTr, typename I,
	     typename Enc,  bool NT, layout_t LayoutR, layout_t Layout>
    static auto
    evaluate( lvalue<VTr,I,void,Enc,NT,LayoutR> l, rvalue<VTr,Layout,void> r,
	      typename std::enable_if<
	      !std::is_void<VTr>::value>::type * = nullptr ) {
	auto lval = l.value().load();
	auto res = make_rvalue( l.value().bor_assign( r.value() ) );
	return res;
    }

    // Case of ref to bitmask, updated with bitmask
    template<typename MTr, typename I,
	     typename Enc,  bool NT, layout_t LayoutR, layout_t Layout>
    static auto
    evaluate( lvalue<MTr,I,void,Enc,NT,LayoutR> l, rvalue<void,Layout,MTr> r,
	      typename std::enable_if<
	      !std::is_void<MTr>::value>::type * = nullptr ) {
	auto lval = l.value().load();
	auto res = make_rvalue( l.value().bor_assign( r.mask() ) );
	return res;
    }    
    template<typename MTr, typename I, 
	     typename Enc, bool NT, layout_t LayoutR, layout_t Layout>
    static auto
    evaluate( lvalue<MTr,I,MTr,Enc,NT,LayoutR> l, rvalue<void,Layout,MTr> r,
	      typename std::enable_if<
	      !std::is_void<MTr>::value>::type * = nullptr ) {
	auto lval = l.value().load();
	auto res = make_rvalue( l.value().bor_assign( r.mask(), l.mask() ) );
	return res;
    }    

/*
    template<typename V, typename U, unsigned short W1, unsigned short W2,
	     unsigned short VL>
    static auto
    evaluate( lvalue<V,U,W1,VL> l, rvalue<void,W2,VL> r ) {
	return make_rvalue( l.value(), l.mask() & r.mask() );
    }
*/

    // Specialising the code to void mask or not makes the code more efficient
    template<typename VTr, layout_t Layout>
    static GG_INLINE auto evaluate1( rvalue<VTr,Layout,void> r ) {
	return make_rvalue( reduce_bitwiseor( r.value() ) );
    }
    template<typename MTr, layout_t Layout>
    static GG_INLINE auto evaluate1( rvalue<void,Layout,MTr> r ) {
	return make_rvalue( reduce_bitwiseor( r.mask() ) );
    }
    template<typename VTr,typename MTr, layout_t Layout>
    static GG_INLINE auto
    evaluate1( rvalue<VTr,Layout,MTr> r,
	       std::enable_if_t<simd::matchVLtt<VTr,MTr>::value> * = nullptr ) {
	return make_rvalue( reduce_bitwiseor( r.value(), r.mask() ) );
    }
#endif

    template<typename VTr, layout_t Layout, typename MPack>
    static auto
    evaluate1( sb::rvalue<VTr,Layout> r, const MPack & mpack ) {
	if constexpr ( MPack::is_empty() ) {
	    auto v = r.value();
	    return make_rvalue( reduce_bitwiseor( v ), sb::mask_pack<>() );
	} else {
	    auto v = r.value();
	    auto m = mpack.get_mask_for( v );
	    return make_rvalue( reduce_bitwiseor( v, m ), sb::mask_pack<>() );
	}
    }

#if 0
/*
    template<unsigned short W, unsigned short VL>
    static GG_INLINE auto
    evaluate1( rvalue<void,W,VL> r ) {
	return make_rvalue( reduce_bitwiseor( r.mask() ) );
    }
    template<typename T, unsigned short W, unsigned short VL>
    static GG_INLINE auto
    evaluate1( rvalue<T,W,VL> r,
	       typename std::enable_if<!std::is_void<T>::value>::type * = nullptr ) {
	return make_rvalue( reduce_bitwiseor( r.value(), r.mask() ) );
    }
*/

    template<typename VTr, typename MTr1, typename MTr2, typename I,
	     typename Enc,  bool NT, layout_t LayoutR, layout_t Layout>
    static auto
    evaluate_atomic( lvalue<VTr,I,MTr1,Enc,NT,LayoutR> l, rvalue<VTr,Layout,MTr2> r,
		     typename std::enable_if<
		     simd::matchVLttotu<VTr,MTr1,MTr2,1>::value>::type *
		     = nullptr ) {
	auto mask = l.mask() & r.mask();
	if( mask.at(0) ) {
	    auto val = l.value().atomic_bitwiseor( r.value() );
	    return make_rvalue( val );
	} else {
	    // auto zval = simd::detail::vec<VTr,lo_linalgn>::zero_val();
	    return make_rvalue( mask );
	}
    }
    template<typename VTr, typename I,
	     typename Enc,  bool NT, layout_t LayoutR, layout_t Layout>
    static auto
    evaluate_atomic( lvalue<VTr,I,void,Enc,NT,LayoutR> l, rvalue<VTr,Layout,void> r,
		     typename std::enable_if<
		     simd::matchVLtu<VTr,1>::value>::type *
		     = nullptr ) {
	auto val = l.value().atomic_bitwiseor( r.value() );
	return make_rvalue( val );
    }
#endif

    template<typename VTr, typename I, typename Enc,  bool NT, layout_t LayoutR,
	     layout_t Layout, typename MPack>
    static auto
    evaluate_atomic( sb::lvalue<VTr,I,Enc,NT,LayoutR> l,
		     sb::rvalue<VTr,Layout> r,
		     const MPack & mpack ) {
	auto rval = r.value();
	auto mask = mpack.template get_any<simd::detail::mask_bool_traits>();
	if( mask.data() ) {
	    auto val = l.value().atomic_bitwiseor( rval );
	    return make_rvalue( val, mpack );
	} else {
	    auto mtrue = simd::detail::mask_impl<simd::detail::mask_bool_traits>
		::true_mask();
	    return make_rvalue( mtrue, mpack );
	}
    }
};

template<typename E1, typename E2>
auto
make_redop_bor( E1 l, E2 r,
		std::enable_if_t<!is_logical_type<typename E1::type>::value>
		* = nullptr ) {
    return make_redop( l, r, redop_bitwiseor() );
}

template<typename E1, typename E2>
auto operator |= ( E1 l, E2 r ) -> decltype(make_redop_bor(l,r)) {
    return make_redop_bor( l, r );
}

/* redop: logical and
 */

struct redop_logicaland {
    template<typename E1, typename E2>
    struct types {
	// using result_type = typename add_logical<typename E1::type>::type; // bool-like;
	using result_type = typename E1::data_type::prefmask_traits;
	using cache_type = typename add_logical<typename E1::type>::type;
	using infer_type = typename E1::data_type;
    };

    static constexpr bool is_idempotent = true;
    static constexpr bool is_benign_race = false; // load-modify-store
    static constexpr bool is_single_trigger = false;

    // TODO: should have an ID here instead of string
    static constexpr char const * name = "redop_logicaland";

    template<typename E1, typename E2>
    using enable = std::true_type; // enable_if_compatible<U1,U2>;

    // template<typename T>
    // static constexpr auto zero() { return ~T(0); }

    // Caution -- "fixed" without debugging
    template<typename T>
    static constexpr auto unit() { return ~typename T::element_type(0); }

    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename I, typename Enc, bool NT, typename MPack>
    static auto
    evaluate( sb::lvalue<VTr,I,Enc,NT,Layout1> l,
	      sb::rvalue<VTr,Layout2> r,
	      const MPack & mpack ) {
	// Apply the mask in the addition operation (ALU), perform a full
	// store. Alternative: only store back relevant values.
	// Best situation may depend on whether the lvalue is linear or not.
	if constexpr ( MPack::is_empty() ) {
	    return make_rvalue( l.value().land_assign( r.value() ), mpack );
	} else {
	    using MTr = typename VTr::prefmask_traits;
	    auto mask = mpack.template get_mask<MTr>();
	    return make_rvalue( l.value().land_assign( r.value(), mask ),
				mpack );
	} 
    }

    template<typename VTr, typename I, typename Enc,  bool NT, layout_t LayoutR,
	     layout_t Layout, typename MPack>
    static auto
    evaluate_atomic( sb::lvalue<VTr,I,Enc,NT,LayoutR> l,
		     sb::rvalue<VTr,Layout> r,
		     const MPack & mpack ) {
	auto rval = r.value();
	auto mask = mpack.template get_any<simd::detail::mask_bool_traits>();
	if( mask.data() ) {
	    auto val = l.value().atomic_logicaland( rval );
	    return make_rvalue( val, mpack );
	} else {
	    // Mask disabled -> no change in the value
	    auto mfalse = simd::detail::mask_impl<simd::detail::mask_bool_traits>
		::false_mask();
	    return make_rvalue( mfalse, mpack );
	}
    }

    template<typename VTr, layout_t Layout, typename MPack>
    static auto
    evaluate1( sb::rvalue<VTr,Layout> r, const MPack & mpack ) {
	auto v = r.value();
	if constexpr ( !mpack.is_empty() ) {
	    auto m = mpack.get_mask_for( v );
	    v &= m.data();
	}
	return make_rvalue( reduce_logicaland( v ), sb::mask_pack<>() );
    }
};

template<typename E1, typename E2>
auto
make_redop_land( E1 l, E2 r,
		 std::enable_if_t<is_logical_type<typename E1::type>::value>
		 * = nullptr ) {
    return make_redop( l, r, redop_logicaland() );
}

template<typename E1, typename E2>
auto operator &= ( E1 l, E2 r ) -> decltype(make_redop_land(l,r)) {
    return make_redop_land( l, r );
}

/* redop: bitwise and
 */

struct redop_bitwiseand {
    template<typename E1, typename E2>
    struct types {
	// using result_type = typename E1::type; // bool-like;
	using result_type = typename E1::data_type::prefmask_traits;
	using cache_type = typename E1::type;
	using infer_type = typename E1::data_type;
    };

    static constexpr bool is_idempotent = true;
    static constexpr bool is_benign_race = false; // load-modify-store
    static constexpr bool is_single_trigger = false;

    // TODO: should have an ID here instead of string
    static constexpr char const * name = "redop_bitwiseand";

    template<typename E1, typename E2>
    using enable = std::true_type; // enable_if_compatible<U1,U2>;

    // Caution -- "fixed" without debugging
    template<typename T>
    static constexpr auto unit() { return ~typename T::element_type(0); }

    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename I, typename Enc, bool NT, typename MPack>
    static auto
    evaluate( sb::lvalue<VTr,I,Enc,NT,Layout1> l,
	      sb::rvalue<VTr,Layout2> r,
	      const MPack & mpack ) {
	// Apply the mask in the addition operation (ALU), perform a full
	// store. Alternative: only store back relevant values.
	// Best situation may depend on whether the lvalue is linear or not.
	if constexpr ( MPack::is_empty() ) {
	    return make_rvalue( l.value().band_assign( r.value() ), mpack );
	} else {
	    using MTr = typename VTr::prefmask_traits;
	    auto mask = mpack.template get_mask<MTr>();
	    return make_rvalue( l.value().band_assign( r.value(), mask ),
				mpack );
	} 
    }

    template<typename VTr, typename MTr1, typename MTr2, typename I,
	     typename Enc, bool NT, layout_t LayoutR, layout_t Layout>
    static auto
    evaluate( lvalue<VTr,I,MTr1,Enc,NT,LayoutR> l, rvalue<VTr,Layout,MTr2> r,
	      std::enable_if_t<simd::matchVL_<VTr::VL,MTr1,MTr2>::value> *
	      = nullptr ) {
	return make_rvalue( l.value().band_assign( r.value(), l.mask() & r.mask() ) );
    }

    template<typename VTr, typename I, typename Enc,  bool NT, layout_t LayoutR,
	     layout_t Layout, typename MPack>
    static auto
    evaluate_atomic( sb::lvalue<VTr,I,Enc,NT,LayoutR> l,
		     sb::rvalue<VTr,Layout> r,
		     const MPack & mpack ) {
	auto rval = r.value();
	auto mask = mpack.template get_any<simd::detail::mask_bool_traits>();
	if( mask.data() ) {
	    auto val = l.value().atomic_bitwiseand( rval );
	    return make_rvalue( val, mpack );
	} else {
	    // Mask disabled -> no change in the value
	    auto mfalse = simd::detail::mask_impl<simd::detail::mask_bool_traits>
		::false_mask();
	    return make_rvalue( mfalse, mpack );
	}
    }
};

template<typename E1, typename E2>
auto
make_redop_band( E1 l, E2 r,
		 std::enable_if_t<!is_logical_type<typename E1::type>::value>
		 * = nullptr ) {
    return make_redop( l, r, redop_bitwiseand() );
}

template<typename E1, typename E2>
auto operator &= ( E1 l, E2 r ) -> decltype(make_redop_band(l,r)) {
    return make_redop_band( l, r );
}


/* redop: min
 */
struct redop_min {
    template<typename E1, typename E2>
    struct types {
	using result_type = typename E1::data_type::prefmask_traits;
	using cache_type = typename E1::data_type;
	using infer_type = typename E1::data_type;
    };

    static constexpr bool is_idempotent = true;
    static constexpr bool is_benign_race = false; // load-modify-store
    static constexpr bool is_single_trigger = false;

    // TODO: should have an ID here instead of string
    static constexpr char const * name = "redop_min";

    template<typename E1, typename E2>
    using enable = std::true_type; // enable_if_compatible<U1,U2>;

    // These values assume positive integers (unsigned int)
    template<typename T>
    static constexpr auto zero() { return T(0); }

    // TODO: divide by 2: temporary hack because vector ops do not distinguish
    // signed/unsigned
    template<typename T>
    static constexpr auto unit() { return std::numeric_limits<typename T::element_type>::max() / 2; }

    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename I, typename Enc, bool NT, typename MPack>
    static auto
    evaluate( sb::lvalue<VTr,I,Enc,NT,Layout1> l,
	      sb::rvalue<VTr,Layout2> r,
	      const MPack & mpack ) {
	// Apply the mask in the addition operation (ALU), perform a full
	// store. Alternative: only store back relevant values.
	// Best situation may depend on whether the lvalue is linear or not.
	if constexpr ( MPack::is_empty() ) {
	    auto lval = l.value().load();
	    auto sel = ( r.value() < lval );
	    // Note: mask not needed for store - already accounted for in iif()
	    // Mask may be more efficient in a scatter
	    l.value().store( ::iif( sel, lval, r.value() ) );
	    return make_rvalue( sel, mpack );
	} else {
	    using MTr = typename VTr::prefmask_traits;
	    auto mask = mpack.template get_mask<MTr>();
	    auto lval = l.value().load( mask );
	    auto less = ( r.value() < lval );
	    
	    // Just in case that < does not return VTr::prefmask_traits
	    auto cmask = mask.template convert_data_type<
		typename decltype(less)::mask_traits>();
	    auto sel = less && cmask;
	    // Note: mask not needed for store - already accounted for in iif()
	    if constexpr ( Layout1 == simd::lo_linear
			   || Layout1 == simd::lo_linalgn ) {
		l.value().store( ::iif( sel, lval, r.value() ) );
	    } else {
		// Mask is necessary in a scatter in case of invalid addresses
		l.value().store( r.value(), sel );
	    }
	    return make_rvalue( sel, mpack );
	} 
    }

    template<typename VTr, layout_t Layout, typename MPack>
    static auto
    evaluate1( sb::rvalue<VTr,Layout> r, const MPack & mpack ) {
	auto v = r.value();
	auto m = mpack.get_mask_for( v );
	return make_rvalue( reduce_min( v, m ), sb::mask_pack<>() );
    }

    
    template<typename VTr, typename MTr1, typename MTr2, typename I,
	     typename Enc,  bool NT, layout_t LayoutR, layout_t Layout>
    static GG_INLINE auto
    evaluate( lvalue<VTr,I,MTr1,Enc,NT,LayoutR> l, rvalue<VTr,Layout,MTr2> r,
	      typename std::enable_if<
	      simd::matchVLttot<VTr,MTr1,MTr2>::value>::type * = nullptr ) {
	auto mask = l.mask() & r.mask();
	auto lval = l.value().load(mask);
	// And in mask in order to keep cache accurate.
	// This is an optimisation opportunity - 1 assembly op/CSC destination
	// auto cmask = mask; // .asmask();
	auto less = ( r.value() < lval );
	auto cmask = mask.template convert_width<decltype(less)::mask_traits::W>(); // .asmask();
	auto sel = less & cmask;
	// Potential optimisation: gather without mask if mask all true (void)
	// Note: not applicable for CSC, COO; may be useful for CSR
	l.value().store( ::iif( sel, lval, r.value() ), cmask ); // TODO: redundancy: mask not needed for store - already accounted for in iif()
	return make_rvalue( sel );
    }

    template<typename VTr, typename I,
	     typename Enc,  bool NT, layout_t LayoutR, layout_t Layout>
    static GG_INLINE auto
    evaluate( lvalue<VTr,I,void,Enc,NT,LayoutR> l, rvalue<VTr,Layout,void> r,
	      typename std::enable_if<
	      !std::is_void<VTr>::value>::type * = nullptr ) {
	auto lval = l.value().load();
	auto rval = r.value();
	auto sel = ( rval < lval );
	// Potential optimisation: gather without mask if mask all true (void)
	// Note: not applicable for CSC, COO; may be useful for CSR
	l.value().store( ::iif( sel, lval, rval ) );
	return make_rvalue( sel );
    }

    template<typename VTr, layout_t Layout>
    static GG_INLINE auto evaluate1( rvalue<VTr,Layout,void> r ) {
	return make_rvalue( reduce_min( r.value() ) );
    }
    template<typename VTr, typename MTr, layout_t Layout>
    static GG_INLINE auto
    evaluate1( rvalue<VTr,Layout,MTr> r,
	       typename std::enable_if<simd::matchVLtt<VTr,MTr>::value>::type *
	       = nullptr ) {
	return make_rvalue( reduce_min( r.value(), r.mask() ) );
    }

    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename I, typename Enc, bool NT, typename MPack>
    static auto
    evaluate_atomic( sb::lvalue<VTr,I,Enc,NT,Layout1> l,
		     sb::rvalue<VTr,Layout2> r,
		     const MPack & mpack ) {
	static_assert( VTr::VL == 1, "atomics always scalar" );
	using MTr = typename VTr::prefmask_traits;
	auto mask = mpack.template get_mask<MTr>();
	if( mask.data() ) {
	    auto oval = l.value().atomic_min( r.value() );
	    return make_rvalue( oval, mpack );
	} else
	    return make_rvalue( simd::detail::mask_impl<MTr>::false_mask(),
				mpack );
    }

    template<typename VTr, typename MTr1, typename MTr2, typename I,
	     typename Enc,  bool NT, layout_t LayoutR, layout_t Layout>
    static auto
    evaluate_atomic( lvalue<VTr,I,MTr1,Enc,NT,LayoutR> l, rvalue<VTr,Layout,MTr2> r,
		     typename std::enable_if<
		     simd::matchVLttotu<VTr,MTr1,MTr2,1>::value>::type *
		     = nullptr ) {
	auto mask = l.mask() & r.mask();
	if( mask.data() ) {
	    auto oval = l.value().atomic_min( r.value() );
	    // return make_rvalue( oval, mask );
	    return make_rvalue( oval );
	} else
	    return make_rvalue( mask );
    }

    template<typename VTr, typename I,
	     typename Enc,  bool NT, layout_t LayoutR, layout_t Layout>
    static auto
    evaluate_atomic( lvalue<VTr,I,void,Enc,NT,LayoutR> l, rvalue<VTr,Layout,void> r,
		     typename std::enable_if<
		     simd::matchVLtu<VTr,1>::value>::type *
		     = nullptr ) {
	auto oval = l.value().atomic_min( r.value() );
	return make_rvalue( oval );
    }
};

template<typename A, typename T, unsigned short VL>
template<typename E>
auto refop<A,T,VL>::min( E rhs ) {
    return redop<self_type,E,redop_min>( *this, rhs, redop_min() );
}

template<typename A, typename T, unsigned short VL>
template<typename E, typename C>
auto refop<A,T,VL>::min( E rhs, C cond ) {
    return make_redop(
	*this, add_predicate( rhs, cond ), redop_min() );
}

template<typename T, unsigned short AID>
template<typename E>
auto scalar<T,AID>::min( E rhs ) {
    return make_redop( make_refop<E::VL>(), rhs, redop_min() );
}
    
/* redop: max
 */
struct redop_max {
    template<typename E1, typename E2>
    struct types {
	using result_type = typename E1::data_type::prefmask_traits;
	using cache_type = typename E1::data_type;
	using infer_type = typename E1::data_type;
    };

    static constexpr bool is_idempotent = true;
    static constexpr bool is_benign_race = false; // load-modify-store
    static constexpr bool is_single_trigger = false;
    
    // TODO: should have an ID here instead of string
    static constexpr char const * name = "redop_max";

    // Unit is expressed using the data_type of the left argument
    template<typename T>
    static constexpr auto unit() {
	return numeric_limits<typename T::element_type>::lowest();
    }

    template<typename E1, typename E2>
    using enable = std::true_type; // enable_if_compatible<U1,U2>;

    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename I, typename Enc, bool NT, typename MPack>
    static auto
    evaluate( sb::lvalue<VTr,I,Enc,NT,Layout1> l,
	      sb::rvalue<VTr,Layout2> r,
	      const MPack & mpack ) {
	// Apply the mask in the addition operation (ALU), perform a full
	// store. Alternative: only store back relevant values.
	// Best situation may depend on whether the lvalue is linear or not.
	if constexpr ( MPack::is_empty() ) {
	    auto lval = l.value().load();
	    auto sel = ( r.value() > lval );
	    // Note: mask not needed for store - already accounted for in iif()
	    // Mask may be more efficient in a scatter
	    l.value().store( ::iif( sel, lval, r.value() ) );
	    return make_rvalue( sel, mpack );
	} else {
	    using MTr = typename VTr::prefmask_traits;
	    auto mask = mpack.template get_mask<MTr>();
	    auto lval = l.value().load( mask );
	    auto less = ( r.value() > lval );
	    
	    // Just in case that < does not return VTr::prefmask_traits
	    auto cmask = mask.template convert_data_type<
		typename decltype(less)::mask_traits>();
	    auto sel = less && cmask;
	    // Note: mask not needed for store - already accounted for in iif()
	    // Mask may be more efficient in a scatter
	    l.value().store( ::iif( sel, lval, r.value() ) );
	    return make_rvalue( sel, mpack );
	} 
    }

    template<typename VTr, typename MTr1, typename MTr2, typename I,
	     typename Enc,  bool NT, layout_t LayoutR, layout_t Layout>
    static GG_INLINE auto
    evaluate( lvalue<VTr,I,MTr1,Enc,NT,LayoutR> l, rvalue<VTr,Layout,MTr2> r,
	      typename std::enable_if<
	      simd::matchVLttot<VTr,MTr1,MTr2>::value>::type * = nullptr ) {
	auto mask = l.mask() & r.mask();
	auto lval = l.value().load(mask);
	// And in mask in order to keep cache accurate.
	// This is an optimisation opportunity - 1 assembly op/CSC destination
	auto cmask = mask; // .asmask();
	auto sel = ( r.value() > lval ) & cmask;
	// Potential optimisation: gather without mask if mask all true (void)
	// Note: not applicable for CSC, COO; may be useful for CSR
	l.value().store( ::iif( sel, lval, r.value() ), cmask ); // TODO: redundancy: mask not needed for store - already accounted for in iif()
	return make_rvalue( sel );
    }

    template<typename VTr, typename I,
	     typename Enc,  bool NT, layout_t LayoutR, layout_t Layout>
    static GG_INLINE auto
    evaluate( lvalue<VTr,I,void,Enc,NT,LayoutR> l, rvalue<VTr,Layout,void> r,
	      typename std::enable_if<
	      !std::is_void<VTr>::value>::type * = nullptr ) {
	auto lval = l.value().load();
	auto sel = ( r.value() > lval );
	// Potential optimisation: gather without mask if mask all true (void)
	// Note: not applicable for CSC, COO; may be useful for CSR
	l.value().store( ::iif( sel, lval, r.value() ) );
	return make_rvalue( sel );
    }

    template<typename VTr, layout_t Layout>
    static GG_INLINE auto
    evaluate1( rvalue<VTr,Layout,void> r,
	      typename std::enable_if<
	      !std::is_void<VTr>::value>::type * = nullptr ) {
	return make_rvalue( reduce_max( r.value() ) );
    }
    template<typename VTr, layout_t Layout, typename MTr>
    static GG_INLINE auto
    evaluate1( rvalue<VTr,Layout,MTr> r,
	      typename std::enable_if<
	       simd::matchVLtt<VTr,MTr>::value>::type * = nullptr ) {
	return make_rvalue( reduce_max( r.value(), r.mask() ) );
    }

    template<typename VTr, layout_t Layout, typename MPack>
    static GG_INLINE auto
    evaluate1( sb::rvalue<VTr,Layout> r,
	       const MPack & mpack ) {
	auto mpack2 = mpack.template clone_and_add<VTr>();
	auto m = mpack2.get_mask_for( r.value() );
	return make_rvalue( reduce_max( r.value(), m ), sb::mask_pack<>() );
    }


    template<typename VTr, typename MTr1, typename MTr2, typename I,
	     typename Enc,  bool NT, layout_t LayoutR, layout_t Layout>
    static GG_INLINE auto
    evaluate_atomic( lvalue<VTr,I,MTr1,Enc,NT,LayoutR> l, rvalue<VTr,Layout,MTr2> r,
		     typename std::enable_if<
		     simd::matchVLttotu<VTr,MTr1,MTr2,1>::value>::type *
		     = nullptr ) {
	auto mask = l.mask() & r.mask();
	if( mask.mask() ) {
	    auto oval = l.value().atomic_max( r.value() );
	    // return make_rvalue( oval, mask );
	    return make_rvalue( oval );
	} else
	    return make_rvalue( mask );
    }

    template<typename VTr, typename I,
	     typename Enc,  bool NT, layout_t LayoutR, layout_t Layout>
    static GG_INLINE auto
    evaluate_atomic( lvalue<VTr,I,void,Enc,NT,LayoutR> l, rvalue<VTr,Layout,void> r,
		     typename std::enable_if<
		     simd::matchVLtu<VTr,1>::value>::type *
		     = nullptr ) {
	auto oval = l.value().atomic_max( r.value() );
	return make_rvalue( oval );
    }
    template<typename VTr, typename I, typename Enc,  bool NT, layout_t LayoutR,
	     layout_t Layout, typename MPack>
    static auto
    evaluate_atomic( sb::lvalue<VTr,I,Enc,NT,LayoutR> l,
		     sb::rvalue<VTr,Layout> r,
		     const MPack & mpack ) {
	auto rval = r.value();
	auto mask = mpack.template get_any<simd::detail::mask_bool_traits>();
	if( mask.data() ) {
	    auto val = l.value().atomic_max( rval );
	    return make_rvalue( val, mpack );
	} else {
	    // Mask disabled -> no change in the value
	    auto mfalse = simd::detail::mask_impl<simd::detail::mask_bool_traits>
		::false_mask();
	    return make_rvalue( mfalse, mpack );
	}
    }
};

template<typename A, typename T, unsigned short VL>
template<typename E>
auto refop<A,T,VL>::max( E rhs ) {
    return redop<self_type,E,redop_max>( *this, rhs, redop_max() );
}

template<typename A, typename T, unsigned short VL>
template<typename E, typename C>
auto refop<A,T,VL>::max( E rhs, C cond ) {
    return make_redop(
	*this, add_predicate( rhs, cond ), redop_max() );
}

template<typename T, unsigned short AID>
template<typename E>
auto scalar<T,AID>::max( E rhs ) {
    return make_redop( make_refop<E::VL>(), rhs, redop_max() );
}
    
/* redop: add
 */
template<bool conditional_>
struct redop_add {
    template<typename E1, typename E2>
    struct types {
	using result_type =
	    std::conditional_t<conditional_,
			       typename E1::data_type::prefmask_traits,
			       typename E1::data_type>;
	using cache_type = typename E1::data_type;
	using infer_type = typename E1::data_type;
    };

    static constexpr bool is_commutative = true;
    static constexpr bool is_idempotent = false;
    static constexpr bool is_benign_race = false; // load-modify-store
    static constexpr bool is_single_trigger = false;
    static constexpr bool conditional = conditional_;
    
    // TODO: should have an ID here instead of string
    static constexpr char const * name = "redop_add";

    template<typename DT>
    static constexpr auto unit() { return typename DT::element_type(0); }
    template<typename DT>
    static constexpr auto vunit() { return simd::create_zero<DT>(); }

    template<typename E1, typename E2>
    using enable = std::true_type; // enable_if_compatible<U1,U2>;

    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename I, typename Enc, bool NT, typename MPack>
    __attribute__((always_inline))
    static inline auto evaluate( sb::lvalue<VTr,I,Enc,NT,Layout1> l,
				 sb::rvalue<VTr,Layout2> r,
				 const MPack & mpack ) {
	// Apply the mask in the addition operation (ALU), perform a full
	// store. Alternative: only store back relevant values.
	// Best situation may depend on whether the lvalue is linear or not.
	if constexpr ( MPack::is_empty() ) {
	    auto lval = l.value().load();
	    l.value().store( lval + r.value() );
	    if constexpr ( conditional )
		return make_rvalue( lval.true_mask(), mpack ); // updated values
	    else
		return make_rvalue( lval, mpack ); // updated values
	} else {
	    using MTr = typename VTr::prefmask_traits;
	    auto mask = mpack.template get_mask<MTr>();
	    if( l.value().is_linear() ) {
		auto lval = l.value().load();
		l.value().store( add( lval, mask, lval, r.value() ) );
		if constexpr ( conditional )
		    return make_rvalue( mask, mpack ); // updated values
		else
		    return make_rvalue( lval, mpack ); // updated values
	    } else {
		auto lval = l.value().load( mask );
		l.value().store( lval + r.value(), mask );
		if constexpr ( conditional )
		    return make_rvalue( mask, mpack ); // updated values
		else
		    return make_rvalue( lval, mpack ); // updated values
	    }
	}
    }

    template<typename VTr, layout_t Layout, typename MPack>
    static GG_INLINE auto
    evaluate1( sb::rvalue<VTr,Layout> r, const MPack & mpack ) {
	if constexpr ( MPack::is_empty() )
	    return make_rvalue( reduce_add( r.value() ), mpack );
	else {
	    // Send an empty mask_pack to indicate an update happened.
	    // This assumes that the mpack mask is not all-false.
	    // If the mpack is all-false, then reduce_add should return 0,
	    // the unit of the additive operation.
	    auto m = mpack.template get_any<VTr>();
	    return make_rvalue( reduce_add( r.value(), m ), sb::mask_pack<>() );
	}
    }

    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename I, typename Enc, bool NT, typename MPack>
    static auto
    evaluate_atomic( sb::lvalue<VTr,I,Enc,NT,Layout1> l,
		     sb::rvalue<VTr,Layout2> r,
		     const MPack & mpack ) {
	static_assert( VTr::VL == 1, "atomics always scalar" );
	using MTr = typename VTr::prefmask_traits;
	auto mask = mpack.template get_mask<MTr>();
	if( mask.data() ) {
	    auto oval = l.value().template atomic_add<conditional>( r.value() );
	    return make_rvalue( oval, mpack );
	} else {
	    if constexpr ( conditional )
		return make_rvalue( simd::detail::mask_impl<MTr>::false_mask(),
				    mpack );
	    else
		return make_rvalue( r.value().zero_val(), mpack );
	}
    }
};

template<typename E1, typename E2>
auto add( E1 l, E2 r ) { // E1 must be l-value (levaluate succeeds)
    return make_redop( l, r, redop_add<true>() );
}

template<typename A, typename T, unsigned short VL>
template<typename E>
auto refop<A,T,VL>::operator += ( E rhs ) {
    return make_redop( *this, rhs, redop_add<true>() );
}
    
template<typename A, typename T, unsigned short VL>
template<typename E>
auto refop<A,T,VL>::add( E rhs ) {
    return make_redop( *this, rhs, redop_add<false>() );
}
    
template<typename T, unsigned short AID>
template<typename E>
auto scalar<T,AID>::operator += ( E rhs ) {
    return make_redop( make_refop<E::VL>(), rhs, redop_add<true>() );
}
    
template<typename T, unsigned short AID>
template<typename E>
auto scalar<T,AID>::add( E rhs ) {
    return make_redop( make_refop<E::VL>(), rhs, redop_add<false>() );
}
    
/* redop: count_down
 */
template<bool conditional_>
struct redop_count_down {
    template<typename E1, typename E2>
    struct types {
	using result_type =
	    std::conditional_t<conditional_,
			       typename E1::data_type::prefmask_traits,
			       typename E1::data_type>;
	using cache_type = typename E1::data_type;
	using infer_type = typename E1::data_type;
    };

    static constexpr bool is_idempotent = false;
    static constexpr bool is_benign_race = false; // load-modify-store
    static constexpr bool is_single_trigger = true; // conditional_;
    static constexpr bool conditional = conditional_;

    // TODO: should have an ID here instead of string
    static constexpr char const * name = "redop_count_down";

    template<typename T>
    static constexpr auto unit() { return typename T::element_type(0); }

    template<typename E1, typename E2>
    using enable = std::true_type; // enable_if_compatible<U1,U2>;

    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename I, typename Enc, bool NT, typename MPack>
    __attribute__((always_inline))
    static inline auto evaluate( sb::lvalue<VTr,I,Enc,NT,Layout1> l,
				 sb::rvalue<VTr,Layout2> r,
				 const MPack & mpack ) {
	// Apply the mask in the addition operation (ALU), perform a full
	// store. Alternative: only store back relevant values.
	// Best situation may depend on whether the lvalue is linear or not.
	auto neg_one = simd::template create_allones<VTr>();
	if constexpr ( MPack::is_empty() ) {
	    auto lval = l.value().load();
	    auto nval = lval + neg_one;
	    auto ok = lval > r.value();
	    auto sval = ::iif( ok, lval, nval );
	    l.value().store( sval );
	    if constexpr ( conditional )
		return make_rvalue( ok && sval == r.value(), mpack ); // threshold reached
	    else
		return make_rvalue( lval, mpack );
	} else {
	    using MTr = typename VTr::prefmask_traits;
	    auto mask = mpack.template get_mask<MTr>();
	    if( l.value().is_linear() ) {
		auto lval = l.value().load();
		auto nval = lval + neg_one;
		auto ok = mask && ( lval > r.value() );
		auto sval = ::iif( ok, lval, nval );
		l.value().store( sval );
		if constexpr ( conditional )
		    return make_rvalue( ok && sval == r.value(), mpack ); // threshold reached
		else
		    return make_rvalue( lval, mpack );
	    } else {
		auto lval = l.value().load( mask );
		auto nval = lval + neg_one;
		auto ok = mask && ( lval > r.value() );
		auto sval = ::iif( ok, lval, nval );
		l.value().store( sval, mask );
		if constexpr ( conditional )
		    return make_rvalue( ok && sval == r.value(), mpack ); // threshold reached
		else
		    return make_rvalue( lval, mpack );
	    }
	}
    }

    template<typename VTr, layout_t Layout, typename MPack>
    static GG_INLINE auto
    evaluate1( sb::rvalue<VTr,Layout> r, const MPack & mpack ) {
	// Oops. Maybe the behaviour of redop_add is meaningful?
	assert( 0 && "Impossible" );
	if constexpr ( MPack::is_empty() )
	    return make_rvalue( reduce_add( r.value() ), mpack );
	else {
	    // Send an empty mask_pack to indicate an update happened.
	    // This assumes that the mpack mask is not all-false.
	    // If the mpack is all-false, then reduce_add should return 0,
	    // the unit of the additive operation.
	    auto m = mpack.template get_any<VTr>();
	    return make_rvalue( reduce_add( r.value(), m ), sb::mask_pack<>() );
	}
    }

    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename I, typename Enc, bool NT, typename MPack>
    static auto
    evaluate_atomic( sb::lvalue<VTr,I,Enc,NT,Layout1> l,
		     sb::rvalue<VTr,Layout2> r,
		     const MPack & mpack ) {
	static_assert( VTr::VL == 1, "atomics always scalar" );
	if constexpr ( MPack::is_empty() ) {
	    auto oval =
		l.value().template atomic_count_down<conditional>( r.value() );
	    return make_rvalue( oval, mpack );
	} else {
	    using MTr = typename VTr::prefmask_traits;
	    auto mask = mpack.template get_mask<MTr>();
	    if( mask.data() ) {
		auto oval =
		    l.value().template atomic_count_down<conditional>( r.value() );
		return make_rvalue( oval, mpack );
	    } else
		if constexpr ( conditional )
		    return make_rvalue( simd::detail::mask_impl<MTr>::false_mask(),
					mpack );
		else
		    return make_rvalue( r.value().zero_val(), mpack );
	}
    }
};

template<typename A, typename T, unsigned short VL>
template<typename E>
auto refop<A,T,VL>::count_down( E rhs ) {
    return redop<self_type,E,redop_count_down<true>>( *this, rhs, redop_count_down<true>() );
}

template<typename A, typename T, unsigned short VL>
template<typename E>
auto refop<A,T,VL>::count_down_value( E rhs ) {
    return redop<self_type,E,redop_count_down<false>>( *this, rhs, redop_count_down<false>() );
}


/* redop: mul
 */
struct redop_mul {
    template<typename E1, typename E2>
    struct types {
	// TODO: simd_mask
	// using result_type = logical<sizeof(typename E1::type)>; // bool-like;
	using result_type = typename E1::data_type::prefmask_traits;
	using cache_type = logical<sizeof(typename E2::type)>;
	using infer_type = typename E1::data_type;
    };

    static constexpr bool is_idempotent = false;
    static constexpr bool is_benign_race = false; // load-modify-store
    static constexpr bool is_single_trigger = false;

    // TODO: should have an ID here instead of string
    static constexpr char const * name = "redop_mul";

    template<typename T>
    static constexpr auto zero() { return T(0); }

    template<typename T>
    static constexpr auto unit() { return typename T::element_type(1); }

    template<typename E1, typename E2>
    using enable = std::true_type; // enable_if_compatible<U1,U2>;

    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename I, typename Enc, bool NT, typename MPack>
    __attribute__((always_inline))
    static inline auto evaluate( sb::lvalue<VTr,I,Enc,NT,Layout1> l,
				 sb::rvalue<VTr,Layout2> r,
				 const MPack & mpack ) {
	// Apply the mask in the addition operation (ALU), perform a full
	// store. Alternative: only store back relevant values.
	// Best situation may depend on whether the lvalue is linear or not.
	if constexpr ( MPack::is_empty() ) {
	    auto lval = l.value().load();
	    l.value().store( lval * r.value() );
	    return make_rvalue( lval.true_mask(), mpack ); // updated values
	} else {
	    using MTr = typename VTr::prefmask_traits;
	    auto mask = mpack.template get_any<MTr>();
	    if( l.value().is_linear() ) {
		auto lval = l.value().load();
		l.value().store( mul( lval, mask, lval, r.value() ) );
	    } else {
		auto lval = l.value().load( mask );
		l.value().store( lval * r.value(), mask );
	    }
	    return make_rvalue( mask, mpack ); // updated values
	}
    }

    template<typename VTr, typename MTr1, typename MTr2, typename I,
	     typename Enc,  bool NT, layout_t LayoutR, layout_t Layout>
    static __attribute__((always_inline))
    auto evaluate( lvalue<VTr,I,MTr1,Enc,NT,LayoutR> l, rvalue<VTr,Layout,MTr2> r ) {
	// Apply the mask in the addition operation (ALU), perform a full
	// store. Alternative: only store back relevant values.
	// Best situation may depend on whether the lvalue is linear or not.
	auto mask = l.mask() & r.mask();
	if( l.value().is_linear() ) {
	    auto lval = l.value().load();
	    l.value().store( mul( lval, mask, lval, r.value() ) );
	} else {
	    auto lval = l.value().load(mask);
	    l.value().store( lval * r.value(), mask );
	}
	return make_rvalue( mask ); // updated values
    }

    template<typename VTr, typename I,
	     typename Enc,  bool NT, layout_t LayoutR, layout_t Layout>
    static __attribute__((always_inline))
    auto evaluate( lvalue<VTr,I,void,Enc,NT,LayoutR> l, rvalue<VTr,Layout,void> r ) {
	auto lval = l.value().load();
	l.value().store( lval * r.value() );
	return make_rvalue( decltype(lval)::true_mask() );
    }


    template<typename VTr, layout_t LayoutR, layout_t Layout>
    static GG_INLINE auto
    evaluate1( rvalue<VTr,Layout,void> r,
	      typename std::enable_if<
	      !std::is_void<VTr>::value>::type * = nullptr ) {
	return make_rvalue( reduce_mul( r.value() ) );
    }
    template<typename VTr, typename MTr, layout_t LayoutR, layout_t Layout>
    static GG_INLINE auto
    evaluate1( rvalue<VTr,Layout,MTr> r,
	      typename std::enable_if<
	       simd::matchVLtt<VTr,MTr>::value>::type * = nullptr ) {
	return make_rvalue( reduce_mul( r.value(), r.mask() ) );
    }
};


template<typename A, typename T, unsigned short VL>
template<typename E>
auto refop<A,T,VL>::operator *= ( E rhs ) {
    return make_redop( *this, rhs, redop_mul() );
}

template<typename T, unsigned short AID>
template<typename E>
auto scalar<T,AID>::operator *= ( E rhs ) {
    return make_redop( make_refop<E::VL>(), rhs, redop_mul() );
}

template<typename A, typename T, unsigned short VL>
template<typename E>
auto refop<A,T,VL>::operator = ( E rhs ) {
    return make_storeop( *this, rhs );
}

/*
template<typename A, typename T, unsigned short VL>
template<typename E, typename C>
auto refop<A,T,VL>::operator = ( binop<E,C,binop_mask> rhs ) {
    return make_storeop( *this, rhs.data1(), rhs2.data2() );
}
*/

/*
template<typename A, typename T, unsigned short VL>
template<typename E, typename C>
auto refop<A,T,VL>::assign_if( E rhs, C cond ) {
    return make_storeop( *this, rhs, cond );
}
*/


/* redop: setif
 */
struct redop_setif {
    template<typename E1, typename E2>
    struct types {
	// TODO: simd_mask
	// using result_type = logical<sizeof(typename E1::type)>; // bool-like;
	using result_type = typename E1::data_type::prefmask_traits;
	using cache_type = logical<sizeof(typename E2::type)>;
	using infer_type = typename E1::data_type;
    };

    // It seems there are two options for sparse push:
    // A. benign-race but not single-trigger, such that duplicates are removed
    // B. not benign-race and single-trigger, such that no duplicates occur
    // Case A will imply that the zero flags are used with CAS to ensure
    // a vertex is woken up only once.
    static constexpr bool is_commutative = false;
    static constexpr bool is_idempotent = true;
    static constexpr bool is_benign_race = false; // store-only
    static constexpr bool is_single_trigger = true;
    
    // TODO: should have an ID here instead of string
    static constexpr char const * name = "redop_setif";

    template<typename E1, typename E2>
    using enable = std::true_type; // enable_if_compatible<U1,U2>;

    template<typename DT>
    static constexpr auto unit() { return ~typename DT::element_type(0); }
    template<typename DT>
    static constexpr auto vunit() { return simd::create_allones<DT>(); }

    template<typename VTr, typename I, typename Enc,  bool NT, layout_t LayoutR,
	     layout_t Layout, typename MPack>
    static __attribute__((always_inline))
    auto evaluate( sb::lvalue<VTr,I,Enc,NT,LayoutR> l, sb::rvalue<VTr,Layout> r,
		   const MPack & mpack ) {
	// Apply the mask in the addition operation (ALU), perform a full
	// store. Alternative: only store back relevant values.
	// Best situation may depend on whether the lvalue is linear or not.
	auto rval = r.value();
	if constexpr ( MPack::is_empty() ) {
	    auto lval = l.value().load();
	    auto sel = ( lval == decltype(lval)::allones_val() );
	    // auto sel = msbset( lval );
	    auto nval = ::iif( sel, lval, rval );
	    l.value().store( nval );
	    return make_rvalue( sel, mpack );
	} else {
	    // TODO: remove && mask in sel: it does not matter what value we
	    //       select in disabled lanes.
	    auto mask = mpack.get_mask_for( rval );
	    auto lval = l.value().load( mask );
	    auto sel = ( lval == decltype(lval)::allones_val() ) && mask;
	    // auto sel = msbset( lval ) && mask;
	    // Reformulate because conditional store performs a blend itself
	    // l.value().store( ::iif( sel, lval, rval ), mask );
	    l.value().store( rval, sel );
	    return make_rvalue( sel, mpack ); // updated values
	}
    }

    template<typename VTr, typename MTr1, typename MTr2, typename I,
	     typename Enc,  bool NT, layout_t LayoutR, layout_t Layout>
    static __attribute__((always_inline))
    auto evaluate( lvalue<VTr,I,MTr1,Enc,NT,LayoutR> l, rvalue<VTr,Layout,MTr2> r ) {
	// Apply the mask in the addition operation (ALU), perform a full
	// store. Alternative: only store back relevant values.
	// Best situation may depend on whether the lvalue is linear or not.
	auto mask = l.mask() & r.mask();
	auto rval = r.value();
	decltype(mask) sel;
	if( l.value().is_linear() ) {
	    auto lval = l.value().load();
	    sel = ( lval == decltype(lval)::allones_val() ) & mask;
	    l.value().store( ::iif( sel, lval, rval ) );
	} else {
	    auto lval = l.value().load(mask);
	    sel = ( lval == decltype(lval)::allones_val() ) & mask;
	    l.value().store( ::iif( sel, lval, rval ), mask );
	}
	return make_rvalue( sel ); // updated values
    }

    template<typename VTr, typename I,
	     typename Enc,  bool NT, layout_t LayoutR, layout_t Layout>
    static __attribute__((always_inline))
    auto evaluate( lvalue<VTr,I,void,Enc,NT,LayoutR> l, rvalue<VTr,Layout,void> r ) {
	auto lval = l.value().load();
	auto rval = r.value();
	auto sel = ( lval == decltype(lval)::allones_val() );
	    // & ( rval != decltype(rval)::allones_val() );
	auto nval = ::iif( sel, lval, rval );
	l.value().store( nval );
	return make_rvalue( sel );
    }


    template<typename VTr, layout_t LayoutR, layout_t Layout>
    static GG_INLINE auto
    evaluate1( rvalue<VTr,Layout,void> r,
	      typename std::enable_if<
	      !std::is_void<VTr>::value>::type * = nullptr ) {
	return make_rvalue( reduce_setif( r.value() ) );
    }
    template<typename VTr, typename MTr, layout_t LayoutR, layout_t Layout>
    static GG_INLINE auto
    evaluate1( rvalue<VTr,Layout,MTr> r,
	      typename std::enable_if<
	       simd::matchVLtt<VTr,MTr>::value>::type * = nullptr ) {
	return make_rvalue( reduce_setif( r.value(), r.mask() ) );
    }

    template<typename VTr, typename MTr1, typename MTr2, typename I,
	     typename Enc,  bool NT, layout_t LayoutR, layout_t Layout>
    static GG_INLINE auto
    evaluate_atomic( lvalue<VTr,I,MTr1,Enc,NT,LayoutR> l, rvalue<VTr,Layout,MTr2> r,
		     typename std::enable_if<
		     simd::matchVLttotu<VTr,MTr1,MTr2,1>::value>::type *
		     = nullptr ) {
	auto mask = l.mask() & r.mask();
	if( mask.get() ) {
	    auto oval = l.value().atomic_setif( r.value() );
	    return make_rvalue( oval );
	} else
	    return make_rvalue( mask );
    }

    template<typename VTr, typename I,
	     typename Enc,  bool NT, layout_t LayoutR, layout_t Layout>
    static GG_INLINE auto
    evaluate_atomic( lvalue<VTr,I,void,Enc,NT,LayoutR> l, rvalue<VTr,Layout,void> r,
		     typename std::enable_if<
		     simd::matchVLtu<VTr,1>::value>::type *
		     = nullptr ) {
	auto oval = l.value().atomic_setif( r.value() );
	return make_rvalue( oval );
    }

    template<typename VTr, typename I, typename Enc,  bool NT, layout_t LayoutR,
	     layout_t Layout, typename MPack>
    static GG_INLINE auto
    evaluate_atomic( sb::lvalue<VTr,I,Enc,NT,LayoutR> l,
		     sb::rvalue<VTr,Layout> r,
		     const MPack & mpack ) {
	static_assert( VTr::VL == 1,
		       "atomic operations apply only to scalars" );

	auto rval = r.value();
	if constexpr ( mpack.is_empty() ) {
	    auto oval = l.value().atomic_setif( r.value() );
	    return make_rvalue( oval, mpack );
	} else {
	    auto mask = mpack.get_mask_for( rval );
	    if( mask.get() ) {
		auto oval = l.value().atomic_setif( r.value() );
		return make_rvalue( oval, mpack );
	    } else
		return make_rvalue( mask, mpack );
	}
    }
};

template<typename E1, typename E2>
redop<E1,E2,redop_setif> setif( E1 l, E2 r ) { // E1 must be l-value (levaluate succeeds)
    return redop<E1,E2,redop_setif>( l, r, redop_setif() );
}

template<typename A, typename T, unsigned short VL>
template<typename E>
auto refop<A,T,VL>::setif( E rhs ) {
    return redop<self_type,E,redop_setif>( *this, rhs, redop_setif() );
}

/* redop: find_first
 * Returns lowest-numbered lane with a zero value
 */
struct redop_find_first {
    template<typename E1, typename E2>
    struct types {
	using result_type = typename E1::data_type::prefmask_traits;
	using cache_type = logical<sizeof(typename E2::type)>;
	using infer_type = typename E1::data_type;
    };

    static constexpr bool is_idempotent = true;
    static constexpr bool is_benign_race = false;
    static constexpr bool is_single_trigger = false;
    
    // TODO: should have an ID here instead of string
    static constexpr char const * name = "redop_find_first";

    template<typename E1, typename E2>
    using enable = std::true_type;

    template<typename T>
    static constexpr auto unit() { return ~typename T::element_type(0); }

    template<typename VTr, typename I, typename Enc,  bool NT, layout_t LayoutR,
	     layout_t Layout, typename MPack>
    static __attribute__((always_inline))
    auto evaluate( sb::lvalue<VTr,I,Enc,NT,LayoutR> l, sb::rvalue<VTr,Layout> r,
		   const MPack & mpack ) {
	assert( 0 && "NYI" );
	return r;
    }

    template<typename VTr, layout_t Layout, typename MPack>
    static auto
    evaluate1( sb::rvalue<VTr,Layout> r, const MPack & mpack ) {
	auto v = r.value();
	if constexpr ( mpack.is_empty() )
	    return make_rvalue( simd::find_first( v ), sb::mask_pack<>() );
	else {
	    auto m = mpack.get_mask_for( v ); // or any mask
	    return make_rvalue( simd::find_first( v, m ), sb::mask_pack<>() );
	}
    }
};

} // namespace expr

#endif // GRAPTOR_DSL_AST_REDOP_H
