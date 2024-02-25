// -*- c++ -*-
#ifndef GRAPTOR_DSL_AST_UNOP_H
#define GRAPTOR_DSL_AST_UNOP_H

/* TODO
 * - check use of unop_cvt_to_vector vs unop_switch_to_vector
 */

namespace expr {

/* unop
 * A unary operation applied to an expression.
 */
template<typename Expr, typename UnOp>
struct unop : public expr_base {
    using data_type = typename UnOp::template types<Expr>::result_type;
    using type [[deprecated("type to be removed in favour of data_type")]] = typename data_type::member_type;
    static constexpr unsigned short VL = UnOp::VL;
    using arg_type = Expr;
    using unop_type = UnOp;

    static constexpr op_codes opcode = op_unop;

    GG_INLINE unop( arg_type arg, UnOp ) : m_arg( arg ) { }

    GG_INLINE arg_type data() const { return m_arg; }

private:
    arg_type m_arg;
};

template<typename Expr, typename UnOp>
auto make_unop( Expr e, UnOp op ) {
    return unop<Expr,UnOp>( e, op );
}

// splat
template<unsigned short VL_>
struct unop_broadcast {
    template<typename E>
    struct types {
	using result_type = typename E::data_type;
    };

    template<unsigned short newVL>
    using rebindVL = unop_broadcast<newVL>;

    static constexpr unsigned short VL = VL_;
    
    static constexpr char const * name = "unop_broadcast";

    template<typename VTr, layout_t Layout, typename MPack>
    static GG_INLINE auto
    evaluate( sb::rvalue<VTr,Layout> r, const MPack & mpack ) {
	using OTr = typename VTr::template rebindVL<VL>::type;
	auto av = simd::template create_constant<OTr>( r.value().at( 0 ) );
	return make_rvalue( av, mpack );
    }
};

template<unsigned short VL, typename Expr>
auto
make_unop_broadcast( Expr e,
		     typename std::enable_if<Expr::VL != VL>::type * = nullptr ) {
    static_assert( Expr::VL == 1, "broadcast operates on scalars" );
    return unop<Expr,unop_broadcast<VL>>( e, unop_broadcast<VL>() );
}

template<unsigned short VL, typename Expr>
auto
make_unop_broadcast( Expr e,
		     typename std::enable_if<Expr::VL == VL>::type * = nullptr ) {
    static_assert( Expr::VL == 1, "broadcast operates on scalars" );
    return e; // already of correct length
}

template<typename RedOp>
struct unop_reduce {
    template<typename E>
    struct types {
	using result_type = typename E::data_type;
    };

    static constexpr unsigned short VL = 1;
    static constexpr char const * name = "unop_reduce";

    template<typename VTr, layout_t Layout, typename MTr>
    __attribute__((always_inline))
    static inline auto
    evaluate( rvalue<VTr,Layout,MTr> r ) {
	return RedOp::evaluate1( r );
    }

    template<typename MTr, layout_t Layout, typename MPack>
    static GG_INLINE auto
    evaluate( sb::rvalue<MTr,Layout> r, const MPack & mpack ) {
	return RedOp::evaluate1( r, mpack );
    }
};

template<typename RedOp, typename Expr>
auto
make_unop_reduce( Expr e, RedOp op,
		  typename std::enable_if<Expr::VL != 1>::type * = nullptr ) {
    return unop<Expr,unop_reduce<RedOp>>( e, unop_reduce<RedOp>() );
}

template<typename RedOp, typename Expr>
auto
make_unop_reduce( Expr e, RedOp op,
	   typename std::enable_if<Expr::VL == 1>::type * = nullptr ) {
    return e; // already scalar
}

template<unsigned short VL_>
struct unop_abs {
    template<typename E>
    struct types {
	using result_type = typename E::data_type;
    };

    template<unsigned short newVL>
    using rebindVL = unop_abs<newVL>;

    static constexpr unsigned short VL = VL_;
    static constexpr char const * name = "unop_abs";

    template<typename VTr, layout_t Layout, typename MTr>
    static GG_INLINE auto
    evaluate( rvalue<VTr,Layout,MTr> r,
	      typename std::enable_if<VTr::VL == MTr::VL>::type * = nullptr ) {
	return make_rvalue( r.value().abs(), r.mask() );
    }
    template<typename VTr, layout_t Layout>
    static GG_INLINE auto
    evaluate( rvalue<VTr,Layout,void> r,
	      typename std::enable_if<VTr::VL == VL>::type * = nullptr ) {
	return make_rvalue( r.value().abs() );
    }

    template<typename VTr, layout_t Layout, typename MPack>
    static GG_INLINE auto
    evaluate( sb::rvalue<VTr,Layout> r, const MPack & mpack ) {
	return make_rvalue( r.value().abs(), mpack );
    }
};

template<typename Expr>
auto make_unop_abs( Expr e ) {
    return unop<Expr,unop_abs<Expr::VL>>( e, unop_abs<Expr::VL>() );
}

template<typename Expr>
auto abs( Expr e ) {
    return make_unop_abs( e );
}

template<unsigned short VL_>
struct unop_sqrt {
    template<typename E>
    struct types {
	using result_type = typename E::data_type;
    };

    template<unsigned short newVL>
    using rebindVL = unop_sqrt<newVL>;

    static constexpr unsigned short VL = VL_;
    static constexpr char const * name = "unop_sqrt";

    template<typename VTr, layout_t Layout, typename MTr>
    static GG_INLINE auto
    evaluate( rvalue<VTr,Layout,MTr> r,
	      typename std::enable_if<VTr::VL == MTr::VL>::type * = nullptr ) {
	return make_rvalue( r.value().sqrt(), r.mask() );
    }
    template<typename VTr, layout_t Layout>
    static GG_INLINE auto
    evaluate( rvalue<VTr,Layout,void> r,
	      typename std::enable_if<VTr::VL == VL>::type * = nullptr ) {
	return make_rvalue( r.value().sqrt() );
    }
};

template<typename Expr>
auto make_unop_sqrt( Expr e ) {
    return unop<Expr,unop_sqrt<Expr::VL>>( e, unop_sqrt<Expr::VL>() );
}

template<unsigned short VL_>
struct unop_select_mask {
    template<typename E>
    struct types {
	using result_type = typename E::data_type;
    };

    template<unsigned short newVL>
    using rebindVL = unop_select_mask<newVL>;

    static constexpr unsigned short VL = VL_;

    static constexpr char const * name = "unop_select_mask";

    template<typename VTr, layout_t Layout, typename MTr>
    static GG_INLINE auto
    evaluate( rvalue<VTr,Layout,MTr> r ) {
	if constexpr ( std::is_void_v<MTr> )
	    return make_rvalue( simd::detail::vector_impl<VTr>::true_mask() );
	else
	    return make_rvalue( r.mask() );
    }
};

template<typename Expr>
auto make_unop_select_mask( Expr e ) {
    return make_unop( e, unop_select_mask<Expr::VL>() );
}

template<unsigned short VL_>
struct unop_remove_mask {
    template<typename E>
    struct types {
	using result_type = typename E::data_type;
    };

    template<unsigned short newVL>
    using rebindVL = unop_remove_mask<newVL>;

    static constexpr unsigned short VL = VL_;

    static constexpr char const * name = "unop_remove_mask";

    template<typename VTr, layout_t Layout, typename MPack>
    static GG_INLINE auto
    evaluate( sb::rvalue<VTr,Layout> r, const MPack & mpack ) {
	// Need to consider semantics - usage of mask has now changed
	return make_rvalue( r.value(), mpack );
    }
};

template<typename Expr>
auto make_unop_remove_mask( Expr e ) {
    return make_unop( e, unop_remove_mask<Expr::VL>() );
}

template<typename T, unsigned short VL_>
struct unop_cvt {
    template<typename E>
    struct types {
	using result_type = simd::ty<T,VL_>;
    };

    static constexpr unsigned short VL = VL_;

    static constexpr char const * name = "unop_cvt";
};

// TODO: probably better to specify target VTr/MTr rather than T
template<typename T, typename Expr>
auto make_unop_cvt( Expr e ) {
    return unop<Expr,unop_cvt<T,Expr::VL>>( e, unop_cvt<T,Expr::VL>() );
}

template<typename DT>
struct unop_cvt_data_type {
    static constexpr unsigned short VL = DT::VL;
    static constexpr char const * name = "unop_cvt_data_type";

    using data_type = DT;
    using T = typename data_type::member_type; // void for bitmask!
    using type = T;

    template<typename E>
    struct types {
	using result_type = data_type;
    };

    template<unsigned short newVL>
    using rebindVL = unop_cvt_data_type<
	typename data_type::template rebindVL<newVL>::type>;

    template<typename VTr, layout_t Layout, typename MPack>
    static GG_INLINE auto
    evaluate( sb::rvalue<VTr,Layout> a, const MPack & mpack ) {
	static_assert( VTr::VL == data_type::VL, "vector length must match" );
	auto r = a.value().template convert_data_type<data_type>();
	static_assert( std::is_same_v<typename decltype(r)::data_type,data_type> );
	return make_rvalue( r, mpack );
    }
    template<typename VTr, layout_t Layout>
    static GG_INLINE auto
    evaluate_confuse_lanes( sb::rvalue<VTr,Layout> a ) {
	static_assert( VTr::VL == data_type::VL, "vector length must match" );
	auto val = a.value();
	auto res = target::confused_convert<
	    typename VTr::member_type,T,VL>::compute( val.data() );
	    
	return make_rvalue(
	    simd::detail::vec<data_type,Layout>( res ),
	    sb::create_mask_pack() );
    }
};

template<typename Tr, typename Expr>
auto
make_unop_cvt_data_type( Expr e ) {
    static_assert( Tr::VL == Expr::VL, "VL match in unop_cvt_data_type" );

    if constexpr ( std::is_same_v<typename Expr::data_type,Tr> ) {
	// Already of the right type
	return e;
/*
    } else if constexpr ( is_unop_cvt_data_type<Expr>::value ) {
	// This is a convert. Cancel the previous conversion and put this one
	// instead. Re-run this method in case the argument is already of the
	// desired type.
	return make_unop_cvt_data_type<Tr>( e.data() );
*/
    } else
	return make_unop( e, unop_cvt_data_type<Tr>() );
}

template<typename T, typename Expr>
auto make_unop_cvt_type( Expr e ) {
    return make_unop_cvt_data_type<typename simd::ty<T,Expr::VL>>( e );
}

// Short-hand
template<typename T, typename Expr>
auto cast( Expr e ) {
    return make_unop_cvt_type<T>( e );
}

/**
 * unop_cvt_to_mask: In an rvalue, move the value to the mask position.
 *                   If a mask is present, perform logical and of value and mask
 */
template<typename Tr>
struct unop_cvt_to_mask {
    static constexpr unsigned short VL = Tr::VL;
    using OperateTy = typename Tr::member_type;
    using data_type = Tr;

    static_assert( simd::detail::is_mask_traits<Tr>::value, "Require valid mask data type" );
    
    // Doesn't work with logical<OperateTy> as  simd_vector operations revert
    // to bool masks and "undo" the conversion. Need to convert to vectors
    // before converting to mask (which may not be most efficient...).
    // static_assert( !is_logical<OperateTy> || VL_ > 1,
    // "cvt_to_mask->logical has no effect if VL==1" );
    template<typename E>
    struct types {
	using result_type = data_type;
    };

    // template<unsigned short newVL>
    // using rebindVL = unop_cvt_to_mask</*StoreTy,*/ OperateTy, newVL>;
    template<unsigned short newVL>
    using rebindVL = unop_cvt_to_mask<
	typename Tr::template rebindVL<newVL>::type>;

    static constexpr char const * name = "unop_cvt_to_mask";

    template<typename VTr, layout_t Layout, typename MTr>
    static GG_INLINE auto
    evaluate( rvalue<VTr,Layout,MTr> a ) {
	if constexpr ( std::is_void_v<VTr> ) {
	    auto m = convert_mask( a.mask() );
	    return make_rvalue( m );
	} else if constexpr ( simd::detail::is_mask_traits<VTr>::value ) {
	    // Not currently supporting case where a vector (value part of
	    // rvalue) holds a mask data_type.
	    // If we do, should convert using simd::mask_cvt<>() as used
	    // by masks.
	    assert( 0 && "NYI" );
	    // Return something
	    auto m = convert_mask( a.mask() );
	    return make_rvalue( m );
	} else {
	    // VTr is vdata_traits
	    auto v = a.value().template asmask<Tr>();

	    if constexpr ( std::is_same_v<MTr,void> ) {
		return make_rvalue( v );
	    } else {
		auto m = convert_mask( a.mask() );
		return make_rvalue( join_mask<data_type>( m, v ) );
	    }
	}
    }

    template<typename VTr, layout_t Layout, typename MPack>
    static GG_INLINE auto
    evaluate( sb::rvalue<VTr,Layout> a, const MPack & mpack ) {
	if constexpr ( std::is_void_v<VTr> ) {
	    auto m = convert_mask( a.value() );
	    return make_rvalue( m, mpack );
	} else if constexpr ( simd::detail::is_mask_traits<VTr>::value ) {
	    // Not currently supporting case where a vector (value part of
	    // rvalue) holds a mask data_type.
	    // If we do, should convert using simd::mask_cvt<>() as used
	    // by masks.
	    auto m = a.value().template convert_data_type<Tr>();
	    return make_rvalue( m, mpack );
	} else {
	    // VTr is vdata_traits
	    auto v = a.value().template asmask<Tr>();

	    return make_rvalue( v, mpack );
	}
    }
private:
    template<typename MTr>
    static auto convert_mask( simd::detail::mask_impl<MTr> m ) {
	return m.template convert<Tr>();
    }
    
public:
    
#if 0
    template<typename VTr, layout_t Layout, typename MTr>
    static GG_INLINE auto
    evaluate( rvalue<VTr,Layout,MTr> a,
	      typename std::enable_if<simd::matchVLttu<VTr,MTr,VL>::value
	      // && std::is_same<typename VTr::member_type, StoreTy>::value
	      && VTr::B != MTr::B && MTr::W != 0>::type * = nullptr ) {
	using OTr = typename VTr::template rebindTy<OperateTy>::type;
	auto zero = simd::vec<OTr,lo_unknown>::zero_val();
	simd::vec<OTr,Layout> acvt // expand StoreTy -> OperateTy
	    = a.value().template convert_to<OperateTy>();
	auto m = join_mask<data_type>( a.mask(), ( zero != acvt ) );
	auto r = make_rvalue( m );
	return r;
    }
    template<typename VTr, layout_t Layout>
    static GG_INLINE auto
    evaluate( rvalue<VTr,Layout,simd::detail::mask_bit_traits<VTr::VL>> a,
	      typename std::enable_if<simd::matchVLtu<VTr,VL>::value>::type *
	      = nullptr ) {
	static_assert( VTr::VL == VL, "need to match VL" );
	simd::detail::mask_impl<data_type> acvt // expand StoreTy -> OperateTy
	    = a.value().template asmask<data_type>();
	simd::detail::mask_impl<data_type> mcvt
	    = a.mask().template convert<data_type>();
	auto r = make_rvalue( acvt & mcvt );
	static_assert( std::is_void<typename decltype(r)::value_traits>::value, "must be mask" );
	static_assert( std::is_same<typename decltype(r)::mask_traits, data_type>::value, "check conversion to data_type" );
	return r;
    }
    template<typename VTr, layout_t Layout, typename MTr>
    static GG_INLINE auto
    evaluate( rvalue<VTr,Layout,MTr> a,
	      // Possibly need to relax to allow MTr == void
	      typename std::enable_if<simd::matchVLttu<VTr,MTr,VL>::value
	      // && std::is_same<typename VTr::member_type, StoreTy>::value
	      && VTr::B == MTr::B>::type * = nullptr ) {
	      // && sizeof(typename VTr::member_type)
	      // == sizeof(typename MTr::member_type)>::type * = nullptr ) {
	auto r = make_rvalue(
	    join_mask<data_type>( a.value().template asmask<data_type>(),
				  a.mask() ) );

	static_assert( std::is_void<typename decltype(r)::value_traits>::value, "must be mask" );
	static_assert( std::is_same<typename decltype(r)::mask_traits, data_type>::value, "check conversion to data_type" );
	return r;
    }
    template<typename VTr, layout_t Layout>
    static GG_INLINE auto
    evaluate( rvalue<VTr,Layout,void> a,
	      typename std::enable_if<simd::matchVLtu<VTr,VL>::value
	      // && std::is_same<typename VTr::member_type, StoreTy>::value
	      >::type * = nullptr ) {
	auto r = make_rvalue( a.value().template asmask<data_type>() );
	static_assert( std::is_void<typename decltype(r)::value_traits>::value, "must be mask" );
	static_assert( std::is_same<typename decltype(r)::mask_traits, data_type>::value, "check conversion to data_type" );
	return r;
    }
    template<layout_t Layout,typename MTr>
    static GG_INLINE auto
    evaluate( rvalue<void,Layout,MTr> a,
	      typename std::enable_if<simd::matchVLtu<MTr,VL>::value>::type *
	      = nullptr ) {
	simd::detail::mask_impl<data_type> mcvt
	    = a.mask().template convert<data_type>();
	return make_rvalue( mcvt );
    }
#endif
};

template<typename Tr, typename Expr>
auto make_unop_cvt_to_mask( Expr e ) {
    return make_unop( e, unop_cvt_to_mask<Tr>() );
}

/**
 * unop_cvt_to_pmask: In an rvalue, move the value to the mask position.
 *                    Convert the mask to the preferred mask type for VL.
 */
template<typename StoreTy, unsigned short VL_>
struct unop_cvt_to_pmask {
    static constexpr unsigned short VL = VL_;

    // using mask_traits = simd::detail::mask_preferred_traits_width<sizeof(VID),VL>;
    using mask_traits = simd::detail::mask_preferred_traits_width<sizeof(StoreTy),VL>;
    using data_type = mask_traits;
    // static_assert( !simd::detail::is_mask_bit_traits<mask_traits>::value,
    // "bitmask not yet supported here" );

    using OperateTy = typename mask_traits::type;

    // Doesn't work with logical<OperateTy> as  simd_vector operations revert
    // to bool masks and "undo" the conversion. Need to convert to vectors
    // before converting to mask (which may not be most efficient...).
    // static_assert( !is_logical<OperateTy> || VL_ > 1,
    // "cvt_to_mask->logical has no effect if VL==1" );
    template<typename E>
    struct types {
	using result_type = data_type;
    };

    template<unsigned short newVL>
    using rebindVL = unop_cvt_to_pmask<StoreTy, newVL>;

    static constexpr char const * name = "unop_cvt_to_pmask";

    template<typename VTr, layout_t Layout, typename MPack>
    static GG_INLINE auto
    evaluate( sb::rvalue<VTr,Layout> a, const MPack & mpack ) {
	if constexpr ( std::is_void_v<VTr> ) {
	    auto m = convert_mask( a.value() );
	    return make_rvalue( m, mpack );
	} else if constexpr ( simd::detail::is_mask_traits<VTr>::value ) {
	    // Not currently supporting case where a vector (value part of
	    // rvalue) holds a mask data_type.
	    // If we do, should convert using simd::mask_cvt<>() as used
	    // by masks.
	    auto m = a.value().template convert_data_type<data_type>();
	    return make_rvalue( m, mpack );
	} else {
	    // VTr is vdata_traits
	    auto v = a.value().template asmask<data_type>();

	    if constexpr ( MPack::is_empty() ) {
		return make_rvalue( v );
	    } else {
		auto m = mpack.template get_mask<data_type>();
		return make_rvalue( join_mask<data_type>( m, v ) );
	    }
	}
    }

#if 0
    template<typename VTr, layout_t Layout, typename MTr>
    static GG_INLINE auto
    evaluate( rvalue<VTr,Layout,MTr> a,
	      typename std::enable_if<simd::matchVLttu<VTr,MTr,VL>::value
	      && std::is_same<typename VTr::member_type, StoreTy>::value
	      && sizeof(typename VTr::member_type)
	      != sizeof(typename MTr::member_type)>::type * = nullptr ) {
	using OTr = typename VTr::template rebindTy<OperateTy>::type;
	auto zero = simd::detail::vector_impl<OTr>::zero_val();
	auto acvt // expand StoreTy -> OperateTy
	    = a.value().template convert_to<OperateTy>();
	auto r = make_rvalue( a.mask() & ( zero != acvt ) );
	static_assert( std::is_void<typename decltype(r)::value_traits>::value, "must be mask" );
	return r;
    }
    template<typename VTr, layout_t Layout, typename MTr>
    static GG_INLINE auto
    evaluate( rvalue<VTr,Layout,MTr> a,
	      // Possibly need to relax to allow MTr == void
	      typename std::enable_if<simd::matchVLttu<VTr,MTr,VL>::value
	      && std::is_same<typename VTr::member_type, StoreTy>::value
	      && sizeof(typename VTr::member_type)
	      == sizeof(typename MTr::member_type)>::type * = nullptr ) {
	auto r = make_rvalue( a.value().asmask() & a.mask() );
	static_assert( std::is_void<typename decltype(r)::value_traits>::value, "must be mask" );
	return r;
    }
    template<typename VTr, layout_t Layout>
    static GG_INLINE auto
    evaluate( rvalue<VTr,Layout,void> a,
	      typename std::enable_if<simd::matchVLtu<VTr,VL>::value
	      && std::is_same<typename VTr::member_type, StoreTy>::value
	      >::type * = nullptr ) {
	auto r = make_rvalue( a.value().asmask() );
	static_assert( std::is_void<typename decltype(r)::value_traits>::value, "must be mask" );
	return r;
    }
    template<layout_t Layout,typename MTr>
    static GG_INLINE auto
    evaluate( rvalue<void,Layout,MTr> a,
	      typename std::enable_if<simd::matchVLtu<MTr,VL>::value>::type *
	      = nullptr ) {
	return a;
    }
#endif
    
private:
    template<typename MTr>
    static auto convert_mask( simd::detail::mask_impl<MTr> m ) {
	return m.template convert<data_type>();
    }
};

template<typename StoreTy, typename Expr>
auto make_unop_cvt_to_pmask( Expr e ) {
    return make_unop( e, unop_cvt_to_pmask<StoreTy,Expr::VL>() );
}

#if 0
/**
 * unop_cvt_to_bool: 
 * TODO: is this not covered by make_unop_cvt_to_mask?
 */
template<typename OperateTy, typename StoreTy, unsigned short VL_>
struct unop_cvt_to_bool { // StoreTy is always bool or logical<>
    template<typename E>
    struct types {
	using result_type = StoreTy; // replace by data_type
    };

    static constexpr unsigned short VL = VL_;

    static constexpr char const * name = "unop_cvt_to_bool";

    template<typename VTr, layout_t Layout, typename MTr>
    static auto
    evaluate( rvalue<VTr,Layout,MTr> a,
	      typename std::enable_if<simd::matchVLttu<VTr,MTr,VL>::value
	      && std::is_same<typename VTr::member_type, OperateTy>::value
	      && sizeof(typename VTr::member_type)
	      == sizeof(typename MTr::member_type)>::type * = nullptr ) {
	using OTr = typename VTr::template rebindTy<StoreTy>::type;
	auto acvt // convert OperateTy -> StoreTy
	    = a.value().template convert_to<StoreTy>();
	return make_rvalue( acvt, a.mask() );
    }
    template<layout_t Layout,typename MTr>
    static auto
    evaluate( rvalue<void,Layout,MTr> a,
	      typename std::enable_if<simd::matchVLtu<MTr,VL>::value>::type *
	      = nullptr ) {
	using OTr = simd::detail::vdata_traits<StoreTy,VL>;
	auto acvt // convert OperateTy -> StoreTy
	    = a.mask().template convert_to<StoreTy>();
	return make_rvalue( acvt );
    }
    template<typename VTr, layout_t Layout>
    static auto
    evaluate( rvalue<VTr,Layout,void> a,
	      typename std::enable_if<simd::matchVLtu<VTr,VL>::value>::type *
	      = nullptr ) {
	return a;
    }
};

template<typename OperateTy, typename StoreTy, typename Expr>
auto make_unop_cvt_to_bool( Expr e,
			    typename std::enable_if<sizeof(OperateTy)!=sizeof(StoreTy)>::type * = nullptr ) {
    return unop<Expr,unop_cvt_to_bool<OperateTy,StoreTy,Expr::VL>>( e, unop_cvt_to_bool<OperateTy,StoreTy,Expr::VL>() );
}

template<typename OperateTy, typename StoreTy, typename Expr>
auto make_unop_cvt_to_bool(
    Expr e,
    typename std::enable_if<sizeof(OperateTy)==sizeof(StoreTy)>::type * = nullptr ) {
    return e;
}
#endif

#if 0
template<typename StoreTy, unsigned short VL_>
struct unop_cvt_to_vector {
    using data_type = simd::ty<StoreTy,VL_>;

    template<typename E>
    struct types {
	using result_type = data_type;
    };

    template<unsigned short newVL>
    using rebindVL = unop_cvt_to_vector<StoreTy, newVL>;

    static constexpr unsigned short VL = VL_;

    static constexpr char const * name = "unop_cvt_to_vector";

    template<layout_t Layout,typename MTr>
    static GG_INLINE auto evaluate( rvalue<void,Layout,MTr> a ) {
	return make_rvalue( a.mask().template asvector<StoreTy>() );
    }
    template<typename VTr, layout_t Layout>
    static GG_INLINE auto
    evaluate( rvalue<VTr,Layout,void> a,
	      typename std::enable_if<
	      is_logical<typename VTr::member_type>::value>::type *
	      = nullptr ) {
	return a;
    }
};
#endif

template<unsigned short VL_>
struct unop_switch_to_vector {
    static constexpr unsigned short VL = VL_;

    template<typename E>
    struct types {
	using result_type = typename E::data_type;
    };

    template<unsigned short newVL>
    using rebindVL = unop_switch_to_vector<newVL>;

    static constexpr char const * name = "unop_switch_to_vector";

    template<typename MTr, layout_t Layout, typename MPack>
    static GG_INLINE auto
    evaluate( sb::rvalue<MTr,Layout> a, const MPack & mpack ) {
	if constexpr ( MTr::VL == 1 ) {
	    using VTr = simd::ty<bool,1>;
	    return make_rvalue( simd::vec<VTr,lo_unknown>( a.value().data() ),
				mpack );
	} else if constexpr ( simd::detail::is_mask_bit_logical_traits<MTr>::value ) {
	    using VTr = simd::detail::vdata_traits<bitfield<MTr::B>,MTr::VL>;
	    return make_rvalue( simd::vec<VTr,lo_unknown>( a.value().get() ),
				mpack );
	} else if constexpr ( MTr::W == 0 ) {
	    using StoreTy = logical<sizeof(VID)>;
	    return make_rvalue( a.value().template asvector<StoreTy>(), mpack );
	} else if constexpr ( !simd::detail::is_mask_traits_v<MTr> ) {
	    return make_rvalue( a.value(), mpack );
	} else {
	    using StoreTy = logical<MTr::W>;
	    return make_rvalue( a.value().template asvector<StoreTy>(), mpack );
	}
    }

    template<layout_t Layout,typename MTr>
    static GG_INLINE auto evaluate( rvalue<void,Layout,MTr> a ) {
	if constexpr ( MTr::VL == 1 ) {
	    using VTr = simd::ty<bool,1>;
	    return make_rvalue( simd::vec<VTr,lo_unknown>( a.mask().get() ) );
	} else if constexpr ( simd::detail::is_mask_bit_logical_traits<MTr>::value ) {
	    using VTr = simd::detail::vdata_traits<bitfield<MTr::B>,MTr::VL>;
	    return make_rvalue( simd::vec<VTr,lo_unknown>( a.mask().get() ) );
	} else if constexpr ( MTr::W == 0 ) {
	    using StoreTy = logical<sizeof(VID)>;
	    return make_rvalue( a.mask().template asvector<StoreTy>() );
	} else {
	    using StoreTy = logical<MTr::W>;
	    return make_rvalue( a.mask().template asvector<StoreTy>() );
	}
    }
    template<layout_t Layout>
    static GG_INLINE auto evaluate( rvalue<void,Layout,simd::detail::mask_bool_traits> a ) {
	using VTr = simd::ty<bool,1>;
	return make_rvalue( simd::vec<VTr,lo_unknown>( a.mask().get() ) );
    }
    template<typename VTr, layout_t Layout>
    static GG_INLINE auto evaluate( rvalue<VTr,Layout,void> a ) {
	return a;
    }
};

template<typename Expr>
auto make_unop_switch_to_vector( Expr e ) {
    return make_unop( e, unop_switch_to_vector<Expr::VL>() );
}

template<typename StoreTy, typename Expr>
auto make_unop_cvt_to_vector( Expr e ) {
    return make_unop_switch_to_vector( e ); // drop StoreTy...
}


/**
 * unop_incseq: Given a scalar value a, create a vector {a+0, a+1, a+2, ...}
 * It is assumed here that the vector will be aligned, i.e., a % VL == 0
 * Aligned defaults to true.
 */
template<unsigned short VL_, bool aligned>
struct unop_incseq {
    template<typename E>
    struct types {
	using result_type = simd::ty<typename E::type,VL_>;
    };

    static constexpr unsigned short VL = VL_;

    static constexpr char const * name = "unop_cvt";

    template<typename VTr, layout_t Layout, typename MPack>
    static auto
    evaluate( sb::rvalue<VTr,Layout> a, const MPack & mpack ) {
	using OTr = typename VTr::template rebindVL<VL>::type;
	auto av = simd::detail::vector_impl<OTr>::
	    template s_set1inc<aligned>( a.value().data() );
	return make_rvalue( av, mpack );
    }
    
    template<typename VTr, layout_t Layout>
    static GG_INLINE auto
    evaluate( rvalue<VTr,Layout,void> a,
	      typename std::enable_if<simd::matchVLtu<VTr,1>::value>::type *
	      = nullptr ) {
	using OTr = typename VTr::template rebindVL<VL>::type;
	auto av = simd::detail::vector_impl<OTr>::
	    template s_set1inc<aligned>( a.value().data() );
	return make_rvalue( av );
    }
};

template<unsigned short VL, typename Expr>
auto make_unop_incseq( Expr e ) {
    static_assert( Expr::VL == 1, "unop_incseq applies to scalars only" );
    return unop<Expr,unop_incseq<VL,true>>( e, unop_incseq<VL,true>() );
}

template<unsigned short VL, bool aligned, typename Expr>
auto make_unop_incseq( Expr e ) {
    static_assert( Expr::VL == 1, "unop_incseq applies to scalars only" );
    return unop<Expr,unop_incseq<VL,aligned>>( e, unop_incseq<VL,aligned>() );
}

/**
 * unop_setl0: Given a scalar value, create a vector length VL containing
 *             this value in lane 0.
 */
template<typename Tr>
struct unop_setl0 {
    static_assert( simd::detail::is_vm_traits<Tr>::value,
		   "Require valid data type" );
    
    using data_type = Tr;
    static constexpr unsigned short VL = data_type::VL;

    template<typename E>
    struct types {
	using result_type = typename E::data_type;
    };

    static constexpr char const * name = "unop_setl0";

    template<layout_t Layout>
    static GG_INLINE auto
    evaluate( rvalue<typename Tr::template rebindVL<1>::type,Layout,void> a ) {
	simd::detail::vector_impl<data_type> av;
	return make_rvalue( 
	    simd::vec<data_type,lo_unknown>::setl0( a.value() ) );
    }
    template<layout_t Layout,typename MTr>
    static GG_INLINE auto
    evaluate( rvalue<typename Tr::template rebindVL<1>::type,Layout,MTr> a,
	      typename std::enable_if<MTr::VL == 1>::type * = nullptr ) {
	// Some detour with masks: if MTr is bool, then NTr may be
	// logical<VID,VL> so we need to properly convert bool to
	// logical<VID,1> first.
	using NTr = typename MTr::template rebindVL<VL>::type;
	using CTr = typename NTr::template rebindVL<1>::type;
	simd::detail::mask_impl<CTr> ac = a.mask().template convert<CTr>();
	return make_rvalue(
	    simd::vec<data_type,lo_unknown>::setl0( a.value() ),
	    simd::detail::mask_impl<NTr>::setl0( ac ) );
    }
    template<layout_t Layout,typename MTr>
    static GG_INLINE auto
    evaluate( rvalue<void,Layout,MTr> a,
	      typename std::enable_if<MTr::VL == 1>::type * = nullptr ) {
	using NTr = typename MTr::template rebindVL<VL>::type;
	return make_rvalue( simd::detail::mask_impl<NTr>::s_setl0( a.mask() ) );
    }
};

template<typename Tr, typename Expr>
auto make_unop_setl0( Expr e ) {
    return make_unop( e, unop_setl0<Tr>() );
}

/**
 * unop_tzcnt: Given a bit mask in each lane, count trailing zeroes.
 */
template<typename Tr, typename ReturnTy>
struct unop_tzcnt {
    using arg_type = Tr;
    static constexpr unsigned short VL = arg_type::VL;
    using return_type = ReturnTy;
    using data_type = typename simd::ty<ReturnTy,VL>;

    static_assert( sizeof(typename arg_type::member_type) == 4
		   ||  sizeof(typename arg_type::member_type) == 8,
		   "Operation only supported on 32-bit and 64-bit types" );

    template<typename E>
    struct types {
	using result_type = data_type;
    };

    static constexpr char const * name = "unop_tzcnt";

    template<typename VTr, layout_t Layout, typename MPack>
    static GG_INLINE auto
    evaluate( sb::rvalue<VTr,Layout> r, const MPack & mpack ) {
	auto c = r.value().template tzcnt<return_type>();
	return make_rvalue( c, mpack );
    }

};

template<typename ReturnTy, typename Expr>
auto make_unop_tzcnt( Expr e ) {
    using unop_type = unop_tzcnt<typename Expr::data_type, // simd::ty<typename Expr::type,Expr::VL>,
				 ReturnTy>;
    return unop<Expr,unop_type>( e, unop_type() );
}

template<typename ReturnTy, typename Expr>
auto tzcnt( Expr e ) {
    return make_unop_tzcnt<ReturnTy>( e );
}


/**
 * unop_lzcnt: Given a bit mask in each lane, count leading zeroes.
 */
template<typename Tr, typename ReturnTy>
struct unop_lzcnt {
    using arg_type = Tr;
    static constexpr unsigned short VL = arg_type::VL;
    using return_type = ReturnTy;
    using data_type = typename simd::ty<ReturnTy,VL>;

    static_assert( sizeof(typename arg_type::member_type) == 4
		   ||  sizeof(typename arg_type::member_type) == 8,
		   "Operation only supported on 32-bit and 64-bit types" );

    template<typename E>
    struct types {
	using result_type = data_type;
    };

    static constexpr char const * name = "unop_lzcnt";

    template<typename VTr, layout_t Layout, typename MPack>
    static GG_INLINE auto
    evaluate( sb::rvalue<VTr,Layout> r, const MPack & mpack ) {
	auto c = r.value().template lzcnt<return_type>();
	return make_rvalue( c, mpack );
    }

};

template<typename ReturnTy, typename Expr>
auto make_unop_lzcnt( Expr e ) {
    using unop_type = unop_lzcnt<simd::ty<typename Expr::type,Expr::VL>,
				 ReturnTy>;
    return unop<Expr,unop_type>( e, unop_type() );
}

template<typename ReturnTy, typename Expr>
auto lzcnt( Expr e ) {
    return make_unop_lzcnt<ReturnTy>( e );
}

/**
 * unop_invert: Bitwise flip of bitmask
 */
template<unsigned short VL_>
struct unop_invert {
    static constexpr unsigned short VL = VL_;

    template<typename E>
    struct types {
	using result_type = typename E::data_type;
    };

    static constexpr char const * name = "unop_invert";

    template<typename MTr, layout_t Layout, typename MPack>
    static GG_INLINE auto
    evaluate( sb::rvalue<MTr,Layout> r, const MPack & mpack ) {
	return make_rvalue( r.value().bitwise_invert(), mpack );
    }

    template<typename VTr, layout_t Layout, typename MTr>
    static GG_INLINE auto
    evaluate( rvalue<VTr,Layout,MTr> r,
	      typename std::enable_if<VTr::VL == MTr::VL>::type * = nullptr ) {
	return make_rvalue( r.value().bitwise_invert(), r.mask() );
    }
    template<typename VTr, layout_t Layout>
    static GG_INLINE auto
    evaluate( rvalue<VTr,Layout,void> r,
	      typename std::enable_if<VTr::VL == VL>::type * = nullptr ) {
	return make_rvalue( r.value().bitwise_invert() );
    }
};

template<typename Expr>
auto make_unop_invert( Expr e,
		       std::enable_if_t<std::is_base_of_v<expr_base,Expr>> *
		       = nullptr ) {
    if constexpr ( is_constant<Expr>::value ) {
	if constexpr ( Expr::vkind == vk_zero )
	    return constant<vk_truemask>();
	else if constexpr ( Expr::vkind == vk_truemask )
	    return constant<vk_zero>();
	else if constexpr ( Expr::vkind == vk_any )
	    return constant<vk_any,typename Expr::type>( ~e.get_value() );
	else
	    assert( 0 && "cannot bitwise-invert constant" );
    } else
	return unop<Expr,unop_invert<Expr::VL>>( e, unop_invert<Expr::VL>() );
}

template<typename Expr>
auto operator ~ ( Expr e ) -> decltype(make_unop_invert( e )) {
    return make_unop_invert( e );
}

/**
 * unop_linvert: Logical inversion
 */
template<unsigned short VL_>
struct unop_linvert {
    static constexpr unsigned short VL = VL_;

    template<typename E>
    struct types {
	using result_type = typename E::data_type;
    };

    static constexpr char const * name = "unop_linvert";

    template<typename MTr, layout_t Layout, typename MPack>
    static GG_INLINE auto
    evaluate( sb::rvalue<MTr,Layout> r, const MPack & mpack ) {
	return make_rvalue( r.value().logical_invert(), mpack );
    }

    template<typename VTr, layout_t Layout, typename MTr>
    static GG_INLINE auto
    evaluate( rvalue<VTr,Layout,MTr> r,
	      typename std::enable_if<VTr::VL == MTr::VL>::type * = nullptr ) {
	return make_rvalue( r.value().logical_invert(), r.mask() );
    }
    template<typename MTr, layout_t Layout>
    static GG_INLINE auto
    evaluate( rvalue<void,Layout,MTr> r,
	      typename std::enable_if<MTr::VL == VL>::type * = nullptr ) {
	return make_rvalue( r.mask().logical_invert() );
    }
    template<typename VTr, layout_t Layout>
    static GG_INLINE auto
    evaluate( rvalue<VTr,Layout,void> r,
	      typename std::enable_if<VTr::VL == VL>::type * = nullptr ) {
	return make_rvalue( r.value().logical_invert() );
    }
};

template<typename Expr>
auto make_unop_linvert( Expr e,
			std::enable_if_t<std::is_base_of_v<expr_base,Expr>> *
			= nullptr ) {
    return unop<Expr,unop_linvert<Expr::VL>>( e, unop_linvert<Expr::VL>() );
}

template<typename Expr>
auto operator ! ( Expr e ) -> decltype(make_unop_linvert( e )) {
    return make_unop_linvert( e );
}

/**
 * unop_shift: Logical/arithmetic shift by constant
 */
template<unsigned short Shift_, bool Left_, bool Arith_, unsigned short VL_>
struct unop_shift {
    static constexpr unsigned short Shift = Shift_;
    static constexpr bool Left = Left_;
    static constexpr bool Arith = Arith_;
    static constexpr unsigned short VL = VL_;

    template<typename E>
    struct types {
	using result_type = typename E::data_type;
    };

    static constexpr char const * name = "unop_shift";

    template<typename MTr, layout_t Layout, typename MPack>
    static GG_INLINE auto
    evaluate( sb::rvalue<MTr,Layout> r, const MPack & mpack ) {
	if constexpr ( Arith ) {
	    if constexpr( Left )
		return make_rvalue( simd::detail::slli<Shift>( r.value() ),
				    mpack );
	    else
		return make_rvalue( simd::detail::srai<Shift>( r.value() ),
				     mpack );
	} else {
	    if constexpr( Left )
		return make_rvalue( simd::detail::slli<Shift>( r.value() ),
				     mpack );
	    else
		return make_rvalue( simd::detail::srli<Shift>( r.value() ),
				     mpack );
	}
    }
};

template<unsigned short Shift, typename Expr>
auto slli( Expr e,
	   std::enable_if_t<std::is_base_of_v<expr_base,Expr>> *
	   = nullptr ) {
    using Op = unop_shift<Shift,true,false,Expr::VL>;
    return unop<Expr,Op>( e, Op() );
}

template<unsigned short Shift, typename Expr>
auto srli( Expr e,
	   std::enable_if_t<std::is_base_of_v<expr_base,Expr>> *
	   = nullptr ) {
    using Op = unop_shift<Shift,false,false,Expr::VL>;
    return unop<Expr,Op>( e, Op() );
}

template<unsigned short Shift, typename Expr>
auto srai( Expr e,
	   std::enable_if_t<std::is_base_of_v<expr_base,Expr>> *
	   = nullptr ) {
    using Op = unop_shift<Shift,false,true,Expr::VL>;
    return unop<Expr,Op>( e, Op() );
}

/**
 * unop_cmpneg: comparison for negative numbers
 */
template<unsigned short VL_>
struct unop_cmpneg {
    static constexpr unsigned short VL = VL_;

    template<typename E>
    struct types {
	using result_type = typename E::data_type::prefmask_traits;
    };

    static constexpr char const * name = "unop_cmpneg";

    template<typename MTr, layout_t Layout, typename MPack>
    static GG_INLINE auto
    evaluate( sb::rvalue<MTr,Layout> r, const MPack & mpack ) {
	return make_rvalue( simd::detail::cmpneg( r.value() ), mpack );
    }
};

template<typename Expr>
auto cmpneg( Expr e,
	     std::enable_if_t<std::is_base_of_v<expr_base,Expr>> * = nullptr ) {
    using Op = unop_cmpneg<Expr::VL>;
    return unop<Expr,Op>( e, Op() );
}

/**
 * unop_msbset: checks if top bit set
 */
template<unsigned short VL_>
struct unop_msbset {
    static constexpr unsigned short VL = VL_;

    template<typename E>
    struct types {
	using result_type = typename E::data_type::prefmask_traits;
    };

    static constexpr char const * name = "unop_msbset";

    template<typename MTr, layout_t Layout, typename MPack>
    static GG_INLINE auto
    evaluate( sb::rvalue<MTr,Layout> r, const MPack & mpack ) {
	return make_rvalue( simd::detail::msbset( r.value() ), mpack );
    }
};

template<typename Expr>
auto msbset( Expr e,
	     std::enable_if_t<std::is_base_of_v<expr_base,Expr>> * = nullptr ) {
    using Op = unop_msbset<Expr::VL>;
    return unop<Expr,Op>( e, Op() );
}

/**
 * unop_is_zero: checks if value is zero or false
 */
template<unsigned short VL_>
struct unop_is_zero {
    static constexpr unsigned short VL = VL_;

    template<typename E>
    struct types {
	using result_type = simd::detail::mask_bool_traits;
    };

    static constexpr char const * name = "unop_is_zero";

    template<typename MTr, layout_t Layout, typename MPack>
    static GG_INLINE auto
    evaluate( sb::rvalue<MTr,Layout> r, const MPack & mpack ) {
	return make_rvalue( simd::detail::is_zero( r.value() ), mpack );
    }
};

template<typename Expr>
auto
make_unop_is_zero(
    Expr e,
    std::enable_if_t<std::is_base_of_v<expr_base,Expr>> * = nullptr ) {
    using Op = unop_is_zero<Expr::VL>;
    return unop<Expr,Op>( e, Op() );
}

/**
 * unop_is_ones: checks if value is all ones.
 * For logical masks, it is required that the whole mask is set to 1s, not
 * just the top bit, to count is 'true'.
 */
template<unsigned short VL_>
struct unop_is_ones {
    static constexpr unsigned short VL = VL_;

    template<typename E>
    struct types {
	using result_type = simd::detail::mask_bool_traits;
    };

    static constexpr char const * name = "unop_is_ones";

    template<typename MTr, layout_t Layout, typename MPack>
    static GG_INLINE auto
    evaluate( sb::rvalue<MTr,Layout> r, const MPack & mpack ) {
	return make_rvalue( simd::detail::is_ones( r.value() ), mpack );
    }
};

template<typename Expr>
auto
make_unop_is_ones(
    Expr e,
    std::enable_if_t<std::is_base_of_v<expr_base,Expr>> * = nullptr ) {
    using Op = unop_is_ones<Expr::VL>;
    return unop<Expr,Op>( e, Op() );
}



} // namespace expr

#endif // GRAPTOR_DSL_AST_UNOP_H
