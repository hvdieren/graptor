// -*- c++ -*-
#ifndef GRAPTOR_DSL_AST_BINOP_H
#define GRAPTOR_DSL_AST_BINOP_H

#include "graptor/dsl/ast/memref.h"
#include "graptor/dsl/comp/extract_mask.h"

namespace expr {

/* binop
 * A binary operation applied to a pair of expressions.
 */
template<typename E1, typename E2, typename BinOp>
struct binop : public expr_base {
    using data_type = typename BinOp::template types<E1,E2>::result_type;
    using type = typename data_type::member_type;
    // using ltype = typename E1::type;
    // using rtype = typename E2::type;
    static constexpr unsigned short VL = std::max( E1::VL, E2::VL );
    using left_type = E1;
    using right_type = E2;
    using op_type = BinOp;

    static constexpr op_codes opcode = op_binop;

    static_assert( E1::VL == E2::VL, "vector lengths must match" );
    
    binop( left_type arg1, right_type arg2, BinOp )
	: m_arg1( arg1 ), m_arg2( arg2 ) { }

    const left_type & __attribute__((always_inline)) data1() const {
	return m_arg1;
    }
    const right_type & __attribute__((always_inline)) data2() const {
	return m_arg2;
    }

private:
    left_type m_arg1;
    right_type m_arg2;
};

template<typename E1, typename E2, typename BinOp>
auto make_binop( E1 e1, E2 e2, BinOp op ) {
    if constexpr ( e1.opcode == op_constant )
	return make_binop( expand_cst( e1, e2 ), e2, op );
    else if constexpr ( e2.opcode == op_constant )
	return make_binop( e1, expand_cst( e2, e1 ), op );
    else
	return binop<E1,E2,BinOp>( e1, e2, op );
}

/* binop: masking
 */
struct binop_mask {
    template<typename E1, typename E2>
    struct types {
	using result_type = typename E1::data_type;
    };

    static constexpr char const * name = "binop_mask";

    template<typename MTr, layout_t Layout,
	     typename MPack>
    __attribute__((always_inline))
    static inline auto set_mask( sb::rvalue<MTr,Layout> r,
				 const MPack & mpack ) {
	static_assert( simd::detail::is_mask_traits_v<MTr>,
		       "require mask traits" );
	
	if constexpr ( mpack.is_empty() ) {
	    return sb::create_mask_pack( r.value() );
	} else if constexpr ( mpack.template has_mask<MTr>() ) {
	    auto m = mpack.template get_mask<MTr>();
	    auto ml = r.value() && m;
	    return sb::create_mask_pack( ml );
	} else {
	    auto m = mpack.template get_any<MTr>();
	    auto mc = m.template convert_data_type<MTr>();
	    auto ml = r.value() && mc;
	    return sb::create_mask_pack( ml );
	}
    }
    
#if 0
    template<typename VTr, layout_t Layout1,
	     typename MTr, layout_t Layout2,
	     typename MPack>
    __attribute__((always_inline))
    static inline auto evaluate( sb::rvalue<VTr,Layout1> l,
				 sb::rvalue<MTr,Layout2> r,
				 const MPack & mpack ) {
	return make_rvalue( l.value(), set_mask( r, mpack ) );
    }
    
    template<typename VTr, layout_t Layout1, typename MTr, layout_t Layout2>
    static auto evaluate( rvalue<VTr,Layout1,MTr> l, rvalue<void,Layout2,MTr> r ) {
	return make_rvalue( l.value(), l.mask() & r.mask() );
    }
    
    template<typename VTr, layout_t Layout1, typename MTr1, typename MTr2, layout_t Layout2>
    static auto evaluate(
	rvalue<VTr,Layout1,MTr1> l, rvalue<void,Layout2,MTr2> r,
	std::enable_if_t<simd::matchVLttot<VTr,MTr1,MTr2>::value> *
	= nullptr ) {
	auto m = join_mask<VTr>( l.mask(), r.mask() );
	return make_rvalue( l.value(), m );
    }
    
    // Strange one.
    // Appears due to compilation process in APRv, MISv.
    template<typename MTr, layout_t Layout1, layout_t Layout2>
    static auto evaluate( rvalue<void,Layout1,MTr> l, rvalue<void,Layout2,MTr> r ) {
	auto res = l.mask() & r.mask();
	return make_rvalue( res );
    }
    
    template<typename MTr, layout_t Layout1, layout_t Layout2>
    static auto
    evaluate( rvalue<void,Layout1,simd::detail::mask_bit_traits<MTr::VL>> l,
	      rvalue<void,Layout2,MTr> r,
	      std::enable_if_t<!simd::detail::is_mask_bit_traits<MTr>::value> *
	      = nullptr ) {
	auto m = r.mask().template
	    convert<simd::detail::mask_bit_traits<MTr::VL>>();
	return make_rvalue( l.mask() & m );
    }

    template<unsigned short W, unsigned short VL, layout_t Layout1, layout_t Layout2>
    static auto
    evaluate( rvalue<void,Layout1,simd::detail::mask_logical_traits<W,VL>> l,
	      rvalue<void,Layout2,simd::detail::mask_bit_traits<VL>> r ) {
	auto m = l.mask().template convert<simd::detail::mask_bit_traits<VL>>();
	return make_rvalue( r.mask() & m );
    }

    template<typename MTr, layout_t Layout1, layout_t Layout2>
    static auto
    evaluate( rvalue<void,Layout1,simd::detail::mask_bool_traits> l,
	      rvalue<void,Layout2,MTr> r,
	      std::enable_if_t<!simd::detail::is_mask_bool_traits<MTr>::value
	      && MTr::VL == 1> * = nullptr ) {
	auto m = r.mask().template convert<simd::detail::mask_bool_traits>();
	return make_rvalue( l.mask() && m );
    }

    template<typename VTr, layout_t Layout1, layout_t Layout2>
    static auto
    evaluate( rvalue<VTr,Layout1,simd::detail::mask_bool_traits> l,
	      rvalue<void,Layout2,simd::detail::mask_bit_traits<1>> r,
	      std::enable_if_t<VTr::VL == 1> * = nullptr ) {
	simd::detail::mask_impl<simd::detail::mask_bool_traits>
	    m( l.mask().data() & (bool)r.mask().data() );
	return make_rvalue( l.value(), m );
    }
    
    template<typename VTr, layout_t Layout1, typename MTr, layout_t Layout2>
    static auto
    evaluate( rvalue<VTr,Layout1,void> l, rvalue<void,Layout2,MTr> r,
	      typename std::enable_if<simd::matchVLtt<VTr,MTr>::value>::type *
	      = nullptr ) {
	return make_rvalue( l.value(), r.mask() );
    }
#endif

    template<typename VTr, layout_t Layout, typename... MTr>
    static auto
    update_mask( sb::rvalue<VTr,Layout,MTr...> c ) {
	if constexpr ( sizeof...(MTr) == 0 ) {
	    return c;
	} else {
	    auto mpack2 = c.mpack().template clone_and_add<VTr>();
	    auto mask = mpack2.get_mask_for( c.value() );
	    auto cmask = c.value() && mask;
	    return make_rvalue( cmask, mpack2 );
	} 
    }

};

// Special case: if adding an empty mask (during re-writing of the AST),
// then immediately optimise it to drop the mask.
template<typename E1>
auto add_mask( E1 val, noop ) {
    return val;
}

template<typename E1, typename E2>
auto add_mask( E1 val, E2 mask ) {
    static_assert( E1::VL == E2::VL && "vector length match" );
    
    // Adding a true mask is a noop
    if constexpr ( is_value_vk<E2,vk_true>::value )
	return val;

    // Combining masks
    else if constexpr ( is_binop_mask<E1>::value )
	return add_mask( val.data1(), val.data2() && mask );

    // Align width of masks
    else if constexpr ( E1::data_type::B != E2::data_type::B ) {
	using Tr =
	    simd::detail::mask_preferred_traits_type<typename E1::type, E1::VL>;
	auto smask = make_unop_cvt_data_type<Tr>( mask );
	return make_binop( val, smask, binop_mask() );
    }

    // Default
    else
	return make_binop( val, mask, binop_mask() );
}

/**
 * binop_setmask: set LHS as a mask for the evaluation of RHS
 */
struct binop_setmask {
    template<typename E1, typename E2>
    struct types {
	// The result type is the RHS
	using result_type = typename E2::data_type;
    };

    static constexpr char const * name = "binop_setmask";
};

template<typename E1, typename E2>
auto set_mask( E1 mask, E2 expr ) {
    if constexpr ( is_noop<E2>::value )
	return expr;
    else if constexpr ( is_value_vk<E1,vk_true>::value )
	return expr;
    else if constexpr ( is_binop_setmask<E2>::value )
	return set_mask( mask && expr.data1(), expr.data2() );
    else
	return make_binop( mask, expr, binop_setmask() );
}

/* binop: predicate
 */
struct binop_predicate {
    template<typename E1, typename E2>
    struct types {
	using result_type = typename E1::data_type;
    };

    static constexpr char const * name = "binop_predicate";

    // This class has no evaluate methods. The expectation is that it always
    // occurs in conjunction with a redop
};

// Special case: if adding an empty mask (during re-writing of the AST),
// then immediately optimise it to drop the mask.
template<typename E1, typename E2>
auto add_predicate( E1 val, E2 mask ) {
    // Add a noop predicate is a noop
    if constexpr ( is_noop<E2>::value )
	return val;

    // Adding a true predicate is a noop
    else if constexpr ( is_value_vk<E2,vk_true>::value )
	return val;

    // Constant mask
    else if constexpr ( is_constant<E2>::value ) {
	// Remove true mask
	if constexpr ( E2::vkind == vk_true )
	    return val;
	// Squash computation of LHS if known to be inactive
	else if constexpr ( E2::vkind == vk_false )
	    return make_binop( expr::zero_val(val), mask, binop_predicate() );
    }

    // Combining predicates
    else if constexpr ( is_binop_mask<E1>::value )
	return add_predicate( val.data1(), val.data2() && mask );

    // Constant value
    else if constexpr ( is_constant<E1>::value )
	return add_predicate( expand_cst( val, mask ), mask );
    
    // Align width of predicates
    else if constexpr ( E1::data_type::B != E2::data_type::B ) {
	using Tr =
	    simd::detail::mask_preferred_traits_type<typename E1::type, E1::VL>;
	auto smask = make_unop_cvt_data_type<Tr>( mask );
	return make_binop( val, smask, binop_predicate() );
    }
    // Default
    else
	return make_binop( val, mask, binop_predicate() );
}

/**
 * binop: sequence
 *
 * The value returned by the sequence is the value of the right-hand-side.
 * The impact of the left-hand-side relates to side-effects on memory only
 * (storeop or redop to array_ro, array_intl or cacheop).
 */
struct binop_seq {
    template<typename E1, typename E2>
    struct types {
	using result_type = typename E2::data_type;
    };

    static constexpr char const * name = "binop_seq";
    
    template<typename VTr1, layout_t Layout1,
	     typename VTr2, layout_t Layout2,
	     typename MPack>
    __attribute__((always_inline))
    static inline auto evaluate( sb::rvalue<VTr1,Layout1> l,
				 sb::rvalue<VTr2,Layout2> r,
				 const MPack & mpack ) {
	return make_rvalue( r.value(), mpack );
    }
    
    template<typename VTr1, layout_t Layout1, typename MTr1, typename VTr2, typename MTr2, layout_t Layout2>
    static auto evaluate( rvalue<VTr1,Layout1,MTr1> l, rvalue<VTr2,Layout2,MTr2> r ) {
	// Both arguments are evaluated. The return value is that of
	// the right-most argument.
	return r;
    }
};

template<typename E1, typename E2>
typename std::enable_if<!is_noop<E1>::value && !is_noop<E2>::value,
    binop<E1,E2,binop_seq>>::type
make_seq( E1 left, E2 right ) {
    return make_binop( left, right, binop_seq() );
}

inline noop make_seq() { return noop(); }

inline noop make_seq( noop, noop ) { return noop(); }

template<typename E2>
E2 make_seq( noop, E2 right ) { return right; }

template<typename E1>
E1 make_seq( E1 left, noop ) { return left; }

template<typename E0, typename... EN>
auto make_seq( E0 e0, EN... en ) {
    if constexpr ( std::is_same_v<E0,noop> )
	return make_seq( en... );
    else
	return make_seq( e0, make_seq( en... ) );
}

/* binop: logical and
 */
struct binop_land {
    template<typename E1, typename E2>
    struct types {
	using result_type = typename E1::data_type;
    };

    static constexpr char const * name = "binop_land";
    
    template<typename VTr1, layout_t Layout1,
	     typename VTr2, layout_t Layout2,
	     typename MPack>
    __attribute__((always_inline))
    static inline auto evaluate( sb::rvalue<VTr1,Layout1> l,
				 sb::rvalue<VTr2,Layout2> r,
				 const MPack & mpack ) {
	if constexpr ( std::is_same_v<VTr1,VTr2> )
	    return make_rvalue( l.value() && r.value(), mpack );
	else {
	    auto rr = r.value().template convert_data_type<VTr1>();
	    return make_rvalue( l.value() && rr, mpack );
	}
    }
    
    template<typename VTr, layout_t Layout1, typename MTr, layout_t Layout2>
    static auto
    evaluate( rvalue<VTr,Layout1,MTr> l, rvalue<VTr,Layout2,MTr> r,
	      typename std::enable_if<!std::is_void<VTr>::value>::type *
	      = nullptr ) {
	return make_rvalue( l.value() & r.value(), l.mask() & r.mask() );
    }
    
    template<typename VTr, layout_t Layout1, typename MTr, layout_t Layout2>
    static auto
    evaluate( rvalue<VTr,Layout1,MTr> l, rvalue<void,Layout2,MTr> r,
	      typename std::enable_if<!std::is_void<VTr>::value>::type *
	      = nullptr ) {
	return make_rvalue( l.value(), l.mask() & r.mask() );
    }
    
    template<layout_t Layout1, typename MTr1, layout_t Layout2, typename MTr2>
    static auto evaluate( rvalue<void,Layout1,MTr1> l,
			  rvalue<void,Layout2,MTr2> r ) {
	auto m = join_mask<MTr1>( l.mask(), r.mask() );
	return make_rvalue( m );
    }

    template<layout_t Layout1, typename MTr, layout_t Layout2>
    static auto
    evaluate( rvalue<void,Layout1,simd::detail::mask_bit_traits<MTr::VL>> l,
	      rvalue<void,Layout2,MTr> r,
	      std::enable_if_t<!simd::detail::is_mask_bit_traits<MTr>::value> *
	      = nullptr ) {
	auto m = r.mask().template
	    convert<simd::detail::mask_bit_traits<MTr::VL>>();
	return make_rvalue( l.mask() & m );
    }
};

// Optimise for the cases where we try to merge no-op or true masks
template<typename E1, typename E2>
auto make_land( E1 l, E2 r,
		std::enable_if_t<std::is_base_of_v<expr_base,E1>
		&& std::is_base_of_v<expr_base,E2>> * = nullptr ) {
    if constexpr ( std::is_same_v<E1,noop> )
	return r;
    else if constexpr ( is_value_vk<E1,vk_true>::value )
	return r;
    else if constexpr ( is_value_vk<E1,vk_false>::value )
	return l;
    else if constexpr ( std::is_same_v<E2,noop> )
	return l;
    else if constexpr ( is_value_vk<E2,vk_true>::value )
	return l;
    else if constexpr ( is_value_vk<E2,vk_false>::value )
	return r;
    else if constexpr ( is_identical<E1,E2>::value )
	return l; // either l or r
    else if constexpr ( is_binop_land<E1>::value ) {
	if constexpr ( is_identical<typename E1::left_type,E2>::value )
	    return l;
	else if constexpr ( is_identical<typename E1::right_type,E2>::value )
	    return l;
	else
	    return make_binop( l, r, binop_land() );
    } else
	return make_binop( l, r, binop_land() );
}

// This overload is enabled/disabled based on whether the make_land function
// is applicable
template<typename E1, typename E2>
auto operator && ( E1 l, E2 r ) -> decltype(make_land(l,r)) {
    return make_land( l, r );
}

/* binop: bitwise and
 */
struct binop_band {
    template<typename E1, typename E2>
    struct types {
	using result_type = typename E1::data_type;
    };

    static constexpr char const * name = "binop_band";

    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename MPack>
    __attribute__((always_inline))
    static inline auto evaluate( sb::rvalue<VTr,Layout1> l,
				 sb::rvalue<VTr,Layout2> r,
				 const MPack & mpack ) {
	return make_rvalue( l.value() & r.value(), mpack );
    }
    
    template<typename VTr, layout_t Layout1, typename MTr, layout_t Layout2>
    static auto
    evaluate( rvalue<VTr,Layout1,MTr> l, rvalue<VTr,Layout2,MTr> r,
	      typename std::enable_if<!std::is_void<VTr>::value>::type *
	      = nullptr ) {
	return make_rvalue( l.value() & r.value(), l.mask() & r.mask() );
    }
    
    template<typename VTr, layout_t Layout1, typename MTr, layout_t Layout2>
    static auto
    evaluate( rvalue<VTr,Layout1,MTr> l, rvalue<void,Layout2,MTr> r,
	      typename std::enable_if<!std::is_void<VTr>::value>::type *
	      = nullptr ) {
	return make_rvalue( l.value(), l.mask() & r.mask() );
    }
    
    template<typename VTr, layout_t Layout1, typename MTr, layout_t Layout2>
    static auto
    evaluate( rvalue<VTr,Layout1,MTr> l, rvalue<VTr,Layout2,void> r,
	      std::enable_if_t<!std::is_void_v<VTr> && !std::is_void_v<MTr>> *
	      = nullptr ) {
	return make_rvalue( l.value() & r.value(), l.mask() );
    }
    
    template<layout_t Layout1, typename MTr, layout_t Layout2>
    static auto evaluate( rvalue<void,Layout1,MTr> l, rvalue<void,Layout2,MTr> r ) {
	return make_rvalue( l.mask() & r.mask() );
    }

    template<layout_t Layout1, typename MTr, layout_t Layout2>
    static auto
    evaluate( rvalue<void,Layout1,simd::detail::mask_bit_traits<MTr::VL>> l,
	      rvalue<void,Layout2,MTr> r,
	      std::enable_if_t<!simd::detail::is_mask_bit_traits<MTr>::value> *
	      = nullptr ) {
	auto m = r.mask().template
	    convert<simd::detail::mask_bit_traits<MTr::VL>>();
	return make_rvalue( l.mask() & m );
    }
};

// Optimise for the cases where we try to merge no-op masks
template<typename E1, typename E2>
auto make_band( E1 l, E2 r,
		std::enable_if_t<std::is_base_of_v<expr_base,E1>
		&& std::is_base_of_v<expr_base,E2>> * = nullptr ) {
    if constexpr ( std::is_same_v<E1,noop> )
	return r;
    else if constexpr ( std::is_same_v<E2,noop> )
	return l;
    // TODO: This is not exact - need to check if E1/E2 contain a vk_any,
    //       as then they will have the same type but may have different values.
    // if constexpr ( std::is_same_v<E1,E2> )
	// return l; // either l or r
    else
	return make_binop( l, r, binop_band() );
}

// This overload is enabled/disabled based on whether the make_band function
// is applicable
template<typename E1, typename E2>
auto operator & ( E1 l, E2 r ) -> decltype(make_band(l,r)) {
    return make_band( l, r );
}

/* binop: logical or
 */
struct binop_lor {
    template<typename E1, typename E2>
    struct types {
	using result_type = typename E1::data_type;
    };

    static constexpr char const * name = "binop_lor";
    
    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename MPack>
    __attribute__((always_inline))
    static inline auto evaluate( sb::rvalue<VTr,Layout1> l,
				 sb::rvalue<VTr,Layout2> r,
				 const MPack & mpack ) {
	return make_rvalue( l.value() || r.value(), mpack );
    }
    
    template<typename VTr, layout_t Layout1, typename MTr, layout_t Layout2>
    static auto
    evaluate( rvalue<VTr,Layout1,MTr> l, rvalue<VTr,Layout2,MTr> r,
	      typename std::enable_if<!std::is_void<VTr>::value>::type *
	      = nullptr ) {
	return make_rvalue( l.value() || r.value(), l.mask() & r.mask() );
    }
    
    template<typename VTr, layout_t Layout1, typename MTr, layout_t Layout2>
    static auto
    evaluate( rvalue<VTr,Layout1,MTr> l, rvalue<void,Layout2,MTr> r,
	      typename std::enable_if<!std::is_void<VTr>::value>::type *
	      = nullptr ) {
	return make_rvalue( l.value(), l.mask() & r.mask() );
    }
    
    template<layout_t Layout1, typename MTr, layout_t Layout2>
    static auto evaluate( rvalue<void,Layout1,MTr> l, rvalue<void,Layout2,MTr> r ) {
	// TODO: suspect to use | here and & above for masks...
	// TODO: this code suggests that an rvalue would be either
	//       a vector+mask or a mask+mask (where the first mask represents
	//       boolean values)
	return make_rvalue( l.mask() | r.mask() );
    }
};

// Optimise for the cases where we try to merge no-op masks
template<typename E>
auto make_lor( E l, noop r ) {
    return l;
}

template<typename E>
auto make_lor( noop l, E r,
	       typename std::enable_if<!std::is_same<E,noop>::value>::type * = nullptr ) {
    return r;
}

template<typename E1, typename E2>
auto make_lor( E1 l, E2 r,
	       typename std::enable_if_t<is_base_of<expr_base,E1>::value
	       && is_base_of<expr_base,E2>::value> * = nullptr ) {
    return make_binop( l, r, binop_lor() );
}

template<typename E1, typename E2>
auto operator || ( E1 l, E2 r ) -> decltype( make_lor( l, r ) ) {
    return make_lor( l, r );
}

/* binop: bitwise or
 */
struct binop_bor {
    template<typename E1, typename E2>
    struct types {
	using result_type = typename E1::data_type;
    };

    static constexpr char const * name = "binop_bor";
    
    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename MPack>
    __attribute__((always_inline))
    static inline auto evaluate( sb::rvalue<VTr,Layout1> l,
				 sb::rvalue<VTr,Layout2> r,
				 const MPack & mpack ) {
	return make_rvalue( l.value() | r.value(), mpack );
    }
    
    template<typename VTr, layout_t Layout1, typename MTr1, typename MTr2, layout_t Layout2>
    static auto
    evaluate( rvalue<VTr,Layout1,MTr1> l, rvalue<VTr,Layout2,MTr2> r,
	      std::enable_if_t<simd::matchVLttot<VTr,MTr1,MTr2>::value> *
	      = nullptr ) {
	auto m = join_mask<VTr>( l.mask(), r.mask() );
	return make_rvalue( l.value() | r.value(), m );
    }
    
    template<layout_t Layout1, typename MTr, layout_t Layout2>
    static auto evaluate( rvalue<void,Layout1,MTr> l, rvalue<void,Layout2,MTr> r ) {
	// TODO: suspect to use | here and & above for masks...
	return make_rvalue( l.mask() | r.mask() );
    }
};

template<typename E1, typename E2>
auto make_bor( E1 l, E2 r,
	       typename std::enable_if_t<is_base_of<expr_base,E1>::value
	       && is_base_of<expr_base,E2>::value> * = nullptr ) {
    return make_binop( l, r, binop_lor() );
}

template<typename E1, typename E2>
auto operator | ( E1 l, E2 r ) -> decltype( make_bor( l, r ) ) {
    return make_bor( l, r );
}

/* binop: bitwise xor
 */
struct binop_bxor {
    template<typename E1, typename E2>
    struct types {
	using result_type = typename E1::data_type;
    };

    static constexpr char const * name = "binop_bxor";
    
    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename MPack>
    __attribute__((always_inline))
    static inline auto evaluate( sb::rvalue<VTr,Layout1> l,
				 sb::rvalue<VTr,Layout2> r,
				 const MPack & mpack ) {
	return make_rvalue( l.value() ^ r.value(), mpack );
    }
    
    template<typename VTr, layout_t Layout1, typename MTr1, typename MTr2, layout_t Layout2>
    static auto
    evaluate( rvalue<VTr,Layout1,MTr1> l, rvalue<VTr,Layout2,MTr2> r,
	      std::enable_if_t<simd::matchVLttot<VTr,MTr1,MTr2>::value> *
	      = nullptr ) {
	return make_rvalue( l.value() ^ r.value(), l.mask() & r.mask() );
    }
    
    template<layout_t Layout1, typename MTr, layout_t Layout2>
    static auto evaluate( rvalue<void,Layout1,MTr> l, rvalue<void,Layout2,MTr> r ) {
	// TODO: suspect to use ^ here and & above for masks...
	return make_rvalue( l.mask() ^ r.mask() );
    }
};

template<typename E1, typename E2>
auto make_bxor( E1 l, E2 r,
	       typename std::enable_if_t<is_base_of<expr_base,E1>::value
	       && is_base_of<expr_base,E2>::value> * = nullptr ) {
    return make_binop( l, r, binop_bxor() );
}

template<typename E1, typename E2>
auto operator ^ ( E1 l, E2 r ) -> decltype( make_bxor( l, r ) ) {
    return make_bxor( l, r );
}

/* binop: cmpne
 */
struct binop_cmpne {
    template<typename E1, typename E2>
    struct types {
	using result_type = typename E1::data_type::prefmask_traits;
    };
    
    static constexpr char const * name = "binop_cmpne";

    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename MPack>
    __attribute__((always_inline))
    static inline auto evaluate( sb::rvalue<VTr,Layout1> l,
				 sb::rvalue<VTr,Layout2> r,
				 const MPack & mpack ) {
	return make_rvalue( ( l.value() != r.value() ), mpack );
    }
    
    template<typename VTr, layout_t Layout1, typename MTr1, typename MTr2, layout_t Layout2>
    __attribute__((always_inline))
    static inline auto evaluate( rvalue<VTr,Layout1,MTr1> l, rvalue<VTr,Layout2,MTr2> r,
				 std::enable_if_t<simd::matchVLttot<VTr,MTr1,MTr2>::value> * = nullptr ) {
	return make_rvalue( ( l.value() != r.value() ) & l.mask() & r.mask() );
    }

    template<layout_t Layout1, typename MTr, layout_t Layout2>
    static inline auto evaluate( rvalue<void,Layout1,MTr> l, rvalue<void,Layout2,MTr> r ) {
	return make_rvalue( l.mask() != r.mask() );
    }
};

template<typename E1, typename E2>
auto make_cmpne( E1 l, E2 r,
		 typename std::enable_if_t<is_base_of<expr_base,E1>::value
		 && is_base_of<expr_base,E2>::value> * = nullptr ) {
    return make_binop( l, r, binop_cmpne() );
}

template<typename E1, typename E2>
auto operator != ( E1 l, E2 r ) -> decltype(make_cmpne(l,r)) {
    return make_cmpne( l, r );
}



/* binop: cmpeq
 */
struct binop_cmpeq {
    template<typename E1, typename E2>
    struct types {
	using result_type = typename E1::data_type::prefmask_traits;
    };
    
    static constexpr char const * name = "binop_cmpeq";
    
    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename MPack>
    __attribute__((always_inline))
    static inline auto evaluate( sb::rvalue<VTr,Layout1> l,
				 sb::rvalue<VTr,Layout2> r,
				 const MPack & mpack ) {
	return make_rvalue( ( l.value() == r.value() ), mpack );
    }
    
    template<typename VTr, layout_t Layout1, layout_t Layout2>
    __attribute__((always_inline))
    static inline auto evaluate( const rvalue<VTr,Layout1,void> & l,
				 const rvalue<VTr,Layout2,void> & r ) {
	return make_rvalue( l.value() == r.value() );
    }

    template<typename VTr, layout_t Layout1, layout_t Layout2>
    __attribute__((always_inline))
    static inline auto evaluate( rvalue<VTr,Layout1,void> && l,
				 rvalue<VTr,Layout2,void> && r ) {
	return make_rvalue( std::move( l.value() ) == std::move( r.value() ) );
    }
    
    template<typename VTr, layout_t Layout1, typename MTr1, typename MTr2, layout_t Layout2>
    __attribute__((always_inline))
    static inline auto evaluate( const rvalue<VTr,Layout1,MTr1> & l,
				 const rvalue<VTr,Layout2,MTr2> & r,
				 std::enable_if_t<!std::is_void<VTr>::value> * = nullptr ) {
	auto c = ( l.value() == r.value() );
	using MTr = typename decltype(c)::mask_traits;
	auto m = l.mask() & r.mask(); // assumes MTr1 and MTr2 are compatible
	simd::detail::mask_impl<MTr> mcvt = m.template convert<MTr>();
	return make_rvalue( c & mcvt );
    }

    template<layout_t Layout1, typename MTr, layout_t Layout2>
    static inline auto evaluate( const rvalue<void,Layout1,MTr> & l,
				 const rvalue<void,Layout2,MTr> & r ) {
	return make_rvalue( l.mask() == r.mask() );
    }
};

template<typename E1, typename E2>
auto make_cmpeq( E1 l, E2 r,
		 std::enable_if_t<is_base_of<expr_base,E1>::value
		 && is_base_of<expr_base,E2>::value> * = nullptr ) {
    return make_binop( l, r, binop_cmpeq() );
}

template<typename E1, typename E2>
auto operator == ( E1 l, E2 r ) -> decltype(make_cmpeq( l, r )) {
    return make_cmpeq( l, r );
}

/* binop: cmpge
 */
struct binop_cmpge {
    template<typename E1, typename E2>
    struct types {
	using result_type = typename E1::data_type::prefmask_traits;
    };
    
    static constexpr char const * name = "binop_cmpge";
    
    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename MPack>
    __attribute__((always_inline))
    static inline auto evaluate( sb::rvalue<VTr,Layout1> l,
				 sb::rvalue<VTr,Layout2> r,
				 const MPack & mpack ) {
	return make_rvalue( ( l.value() >= r.value() ), mpack );
    }
    
    template<typename VTr, layout_t Layout1, typename MTr1, typename MTr2, layout_t Layout2>
    __attribute__((always_inline))
    static inline auto evaluate( rvalue<VTr,Layout1,MTr1> l, rvalue<VTr,Layout2,MTr2> r ) {
	return make_rvalue( ( l.value() >= r.value() )
			    & l.mask() & r.mask() );
    }
};

template<typename E1, typename E2>
binop<E1,E2,binop_cmpge> make_cmpge( E1 l, E2 r ) {
    return make_binop( l, r, binop_cmpge() );
}

template<typename E1, typename E2>
typename std::enable_if<is_base_of<expr_base,E1>::value && is_base_of<expr_base,E2>::value,
			binop<E1,E2,binop_cmpge>>::type
operator >= ( E1 l, E2 r ) {
    return make_binop( l, r, binop_cmpge() );
}

/* binop: cmpgt
 */
struct binop_cmpgt {
    template<typename E1, typename E2>
    struct types {
	using result_type = typename E1::data_type::prefmask_traits;
    };
    
    static constexpr char const * name = "binop_cmpgt";
    
    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename MPack>
    __attribute__((always_inline))
    static inline auto evaluate( sb::rvalue<VTr,Layout1> l,
				 sb::rvalue<VTr,Layout2> r,
				 const MPack & mpack ) {
	return make_rvalue( ( l.value() > r.value() ), mpack );
    }
    
    template<typename VTr, layout_t Layout1, typename MTr1, typename MTr2, layout_t Layout2>
    __attribute__((always_inline))
    static inline auto evaluate( rvalue<VTr,Layout1,MTr1> l, rvalue<VTr,Layout2,MTr2> r ) {
	auto m = join_mask<VTr>( l.mask(), r.mask() );
	return make_rvalue( ( l.value() > r.value() ) && m );
    }
};

template<typename E1, typename E2>
auto make_cmpgt( E1 l, E2 r,
		 std::enable_if_t<is_base_of<expr_base,E1>::value
		 && is_base_of<expr_base,E2>::value> * = nullptr ) {
    return make_binop( l, r, binop_cmpgt() );
}

template<typename E1, typename E2>
auto operator > ( E1 l, E2 r ) -> decltype(make_cmpgt( l, r )) {
    return make_cmpgt( l, r );
}

/* binop: cmple
 */
struct binop_cmple {
    template<typename E1, typename E2>
    struct types {
	using result_type = typename E1::data_type::prefmask_traits;
    };
    
    static constexpr char const * name = "binop_cmple";
    
    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename MPack>
    __attribute__((always_inline))
    static inline auto evaluate( sb::rvalue<VTr,Layout1> l,
				 sb::rvalue<VTr,Layout2> r,
				 const MPack & mpack ) {
	return make_rvalue( ( r.value() >= l.value() ), mpack );
    }
    
    template<typename VTr, layout_t Layout1, typename MTr1, typename MTr2, layout_t Layout2>
    __attribute__((always_inline))
    static inline auto evaluate( rvalue<VTr,Layout1,MTr1> l, rvalue<VTr,Layout2,MTr2> r ) {
	return make_rvalue( ( r.value() >= l.value() ) // swap L<->R
			    & l.mask() & r.mask() );
    }
};

template<typename E1, typename E2>
binop<E1,E2,binop_cmple> make_cmple( E1 l, E2 r ) {
    return make_binop( l, r, binop_cmple() );
}

template<typename E1, typename E2>
typename std::enable_if<is_base_of<expr_base,E1>::value && is_base_of<expr_base,E2>::value,
			binop<E1,E2,binop_cmple>>::type
operator <= ( E1 l, E2 r ) {
    return make_binop( l, r, binop_cmple() );
}


/* binop: cmplt
 */
struct binop_cmplt {
    template<typename E1, typename E2>
    struct types {
	using result_type = typename E1::data_type::prefmask_traits;
    };
    
    static constexpr char const * name = "binop_cmplt";
    
    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename MPack>
    __attribute__((always_inline))
    static inline auto evaluate( sb::rvalue<VTr,Layout1> l,
				 sb::rvalue<VTr,Layout2> r,
				 const MPack & mpack ) {
	return make_rvalue( ( l.value() < r.value() ), mpack );
    }
    
    template<typename VTr, layout_t Layout1,
	     typename MTr1, typename MTr2, layout_t Layout2>
    __attribute__((always_inline))
    static inline auto evaluate( rvalue<VTr,Layout1,MTr1> l, rvalue<VTr,Layout2,MTr2> r ) {
	return make_rvalue( ( l.value() < r.value() )
			    && l.mask() && r.mask() );
    }
};

template<typename E1, typename E2>
binop<E1,E2,binop_cmplt> make_cmplt( E1 l, E2 r ) {
    return make_binop( l, r, binop_cmplt() );
}

template<typename E1, typename E2>
typename std::enable_if<is_base_of<expr_base,E1>::value && is_base_of<expr_base,E2>::value,
			binop<E1,E2,binop_cmplt>>::type
operator < ( E1 l, E2 r ) {
    return make_binop( l, r, binop_cmplt() );
}


/* binop: mul
 */
struct binop_mul {
    template<typename E1, typename E2>
    struct types {
	using result_type = typename E1::data_type;
    };
    
    static constexpr char const * name = "binop_mul";
    
    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename MPack>
    __attribute__((always_inline))
    static inline auto evaluate( sb::rvalue<VTr,Layout1> l,
				 sb::rvalue<VTr,Layout2> r,
				 const MPack & mpack ) {
	return make_rvalue( l.value() * r.value(), mpack );
    }
    
    template<typename VTr, layout_t Layout1, typename MTr1, typename MTr2, layout_t Layout2>
    __attribute__((always_inline))
    static inline auto evaluate( rvalue<VTr,Layout1,MTr1> l, rvalue<VTr,Layout2,MTr2> r ) {
	return make_rvalue( l.value() * r.value(), l.mask() & r.mask() );
    }
};

template<typename E1, typename E2>
binop<E1,E2,binop_mul> make_mul( E1 l, E2 r ) {
    return make_binop( l, r, binop_mul() );
}

template<typename E1, typename E2>
typename std::enable_if<is_base_of<expr_base,E1>::value
&& is_base_of<expr_base,E2>::value,
			binop<E1,E2,binop_mul>>::type
operator * ( E1 l, E2 r ) {
    return make_binop( l, r, binop_mul() );
}

/* binop: add
 */
struct binop_add {
    template<typename E1, typename E2>
    struct types {
	using result_type = typename E1::data_type;
    };
    
    static constexpr char const * name = "binop_add";
    
    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename MPack>
    __attribute__((always_inline))
    static inline auto evaluate( sb::rvalue<VTr,Layout1> l,
				 sb::rvalue<VTr,Layout2> r,
				 const MPack & mpack ) {
	return make_rvalue( l.value() + r.value(), mpack );
    }
    
    template<typename VTr, layout_t Layout1, typename MTr1, typename MTr2, layout_t Layout2>
    __attribute__((always_inline))
    static inline auto evaluate( rvalue<VTr,Layout1,MTr1> l, rvalue<VTr,Layout2,MTr2> r ) {
	return make_rvalue( l.value() + r.value(), l.mask() & r.mask() );
    }
};

template<typename E1, typename E2>
auto make_add( E1 l, E2 r,
	       std::enable_if_t<is_base_of<expr_base,E1>::value
	       && is_base_of<expr_base,E2>::value> * = nullptr ) {
    return make_binop( l, r, binop_add() );
}

template<typename E1, typename E2>
auto operator + ( E1 l, E2 r ) -> decltype(make_add( l, r )) {
    return make_add( l, r );
}

/* binop: sub
 */
struct binop_sub {
    template<typename E1, typename E2>
    struct types {
	using result_type = typename E1::data_type;
    };
    
    static constexpr char const * name = "binop_sub";
    
    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename MPack>
    __attribute__((always_inline))
    static inline auto evaluate( sb::rvalue<VTr,Layout1> l,
				 sb::rvalue<VTr,Layout2> r,
				 const MPack & mpack ) {
	return make_rvalue( l.value() - r.value(), mpack );
    }
    
    template<typename VTr, layout_t Layout1, typename MTr1, typename MTr2, layout_t Layout2>
    __attribute__((always_inline))
    static inline auto evaluate( rvalue<VTr,Layout1,MTr1> l, rvalue<VTr,Layout2,MTr2> r ) {
	return make_rvalue( l.value() - r.value(), l.mask() & r.mask() );
    }
};

template<typename E1, typename E2>
auto make_sub( E1 l, E2 r,
		 typename std::enable_if_t<is_base_of<expr_base,E1>::value
		 && is_base_of<expr_base,E2>::value> * = nullptr ) {
    return make_binop( l, r, binop_sub() );
}

template<typename E1, typename E2>
auto operator - ( E1 l, E2 r ) -> decltype(make_sub(l,r)) {
    return make_sub( l, r );
}


/* binop: div
 */
struct binop_div {
    template<typename E1, typename E2>
    struct types {
	using result_type = typename E1::data_type;
    };
    
    static constexpr char const * name = "binop_div";
    
    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename MPack>
    __attribute__((always_inline))
    static inline auto evaluate( sb::rvalue<VTr,Layout1> l,
				 sb::rvalue<VTr,Layout2> r,
				 const MPack & mpack ) {
	return make_rvalue( l.value() / r.value(), mpack );
    }
    
    template<typename VTr, layout_t Layout1, typename MTr1, typename MTr2, layout_t Layout2>
    __attribute__((always_inline))
    static inline auto evaluate( rvalue<VTr,Layout1,MTr1> l, rvalue<VTr,Layout2,MTr2> r ) {
	return make_rvalue( l.value() / r.value(), l.mask() && r.mask() );
    }
};

template<typename E1, typename E2>
binop<E1,E2,binop_div> make_div( E1 l, E2 r ) {
    return make_binop( l, r, binop_div() );
}

template<typename E1, typename E2>
typename std::enable_if<is_base_of<expr_base,E1>::value
&& is_base_of<expr_base,E2>::value,
			binop<E1,E2,binop_div>>::type
operator / ( E1 l, E2 r ) {
    return make_binop( l, r, binop_div() );
}

/* binop: mod
 */
struct binop_mod {
    template<typename E1, typename E2>
    struct types {
	using result_type = typename E1::data_type;
    };
    
    static constexpr char const * name = "binop_mod";
    
    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename MPack>
    __attribute__((always_inline))
    static inline auto evaluate( sb::rvalue<VTr,Layout1> l,
				 sb::rvalue<VTr,Layout2> r,
				 const MPack & mpack ) {
	return make_rvalue( l.value() % r.value(), mpack );
    }
    
    template<typename VTr, layout_t Layout1, typename MTr1, typename MTr2, layout_t Layout2>
    __attribute__((always_inline))
    static inline auto evaluate( rvalue<VTr,Layout1,MTr1> l, rvalue<VTr,Layout2,MTr2> r ) {
	return make_rvalue( l.value() % r.value(), l.mask() && r.mask() );
    }
};

template<typename E1, typename E2>
binop<E1,E2,binop_div> make_mod( E1 l, E2 r ) {
    return make_binop( l, r, binop_mod() );
}

template<typename E1, typename E2>
typename std::enable_if<is_base_of<expr_base,E1>::value
&& is_base_of<expr_base,E2>::value,
			binop<E1,E2,binop_mod>>::type
operator % ( E1 l, E2 r ) {
    return make_binop( l, r, binop_mod() );
}


/* binop: min
 */
struct binop_min {
    template<typename E1, typename E2>
    struct types {
	using result_type = typename E1::data_type;
    };
    
    static constexpr char const * name = "binop_min";
    
    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename MPack>
    __attribute__((always_inline))
    static inline auto evaluate( sb::rvalue<VTr,Layout1> l,
				 sb::rvalue<VTr,Layout2> r,
				 const MPack & mpack ) {
	return make_rvalue( simd::min( l.value(), r.value() ), mpack );
    }
    
    template<typename VTr, layout_t Layout1, typename MTr1, typename MTr2, layout_t Layout2>
    __attribute__((always_inline))
    static inline auto evaluate( rvalue<VTr,Layout1,MTr1> l, rvalue<VTr,Layout2,MTr2> r ) {
	return make_rvalue( simd::min( l.value(), r.value() ), l.mask() && r.mask() );
    }
};

template<typename E1, typename E2>
auto min( E1 l, E2 r,
	  typename std::enable_if_t<is_base_of<expr_base,E1>::value
	  && is_base_of<expr_base,E2>::value> * = nullptr ) {
    return make_binop( l, r, binop_min() );
}

/* binop: max
 */
struct binop_max {
    template<typename E1, typename E2>
    struct types {
	using result_type = typename E1::data_type;
    };
    
    static constexpr char const * name = "binop_max";
    
    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename MPack>
    __attribute__((always_inline))
    static inline auto evaluate( sb::rvalue<VTr,Layout1> l,
				 sb::rvalue<VTr,Layout2> r,
				 const MPack & mpack ) {
	return make_rvalue( simd::max( l.value(), r.value() ), mpack );
    }
    
    template<typename VTr, layout_t Layout1, typename MTr1, typename MTr2, layout_t Layout2>
    __attribute__((always_inline))
    static inline auto evaluate( rvalue<VTr,Layout1,MTr1> l, rvalue<VTr,Layout2,MTr2> r ) {
	return make_rvalue( simd::max( l.value(), r.value() ), l.mask() && r.mask() );
    }
};

template<typename E1, typename E2>
auto max( E1 l, E2 r,
	  typename std::enable_if_t<is_base_of<expr_base,E1>::value
	  && is_base_of<expr_base,E2>::value> * = nullptr ) {
    return make_binop( l, r, binop_max() );
}


/* binop: srl
 */
struct binop_srl {
    template<typename E1, typename E2>
    struct types {
	using result_type = typename E1::data_type;
    };
    
    static constexpr char const * name = "binop_srl";
    
    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename MPack>
    __attribute__((always_inline))
    static inline auto evaluate( sb::rvalue<VTr,Layout1> l,
				 sb::rvalue<VTr,Layout2> r,
				 const MPack & mpack ) {
	return make_rvalue( l.value() >> r.value(), mpack );
    }
    
    template<typename VTr1, layout_t Layout1, typename MTr1, typename CTr, typename MTr2, layout_t Layout2>
    __attribute__((always_inline))
    static inline auto evaluate( rvalue<VTr1,Layout1,MTr1> l, rvalue<CTr,Layout2,MTr2> r ) {
	return make_rvalue( l.value() >> r.value(), l.mask() && r.mask() );
    }
};

template<typename E1, typename E2>
auto srl( E1 l, E2 r ) {
    return make_binop( l, r, binop_srl() );
}

template<typename E1, typename E2>
auto operator >> ( E1 l, E2 r ) {
    return srl( l, r );
}

/* binop: sll
 */
struct binop_sll {
    template<typename E1, typename E2>
    struct types {
	using result_type = typename E1::data_type;
    };
    
    static constexpr char const * name = "binop_sll";
    
    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename MPack>
    __attribute__((always_inline))
    static inline auto evaluate( sb::rvalue<VTr,Layout1> l,
				 sb::rvalue<VTr,Layout2> r,
				 const MPack & mpack ) {
	return make_rvalue( l.value() << r.value(), mpack );
    }
    
    template<typename VTr, layout_t Layout1, typename MTr1, typename CTr, typename MTr2, layout_t Layout2>
    __attribute__((always_inline))
    static inline auto evaluate( rvalue<VTr,Layout1,MTr1> l, rvalue<CTr,Layout2,MTr2> r ) {
	return make_rvalue( l.value() << r.value(), l.mask() && r.mask() );
    }
};

template<typename E1, typename E2>
auto sll( E1 l, E2 r ) {
    return make_binop( l, r, binop_sll() );
}

template<typename E1, typename E2>
auto operator << ( E1 l, E2 r ) {
    return sll( l, r );
}

/**====================================================================-
 * let expression.
 * Creates a temporary array, not-backed up by any storage. Code rewrite
 * Must replace these by cacheop. AID must be a uniquely used numeric
 * identifier. cont is a lambda that is called as a continuation. An AST
 * is passed to the continuation that represents a reference to the
 * cached storage.
 * Note: this drops the mask on init, if any, implying that the continuation
 *       may be applied to lanes that were not initialised.
 *=====================================================================*/
template<short AID, typename Init, typename L>
auto let( Init init, L cont ) {
    // Infer variable types
    using data_type = typename Init::data_type;
    using value_type = typename data_type::element_type;
    using index_type = VID; // seems natural, actually irrelevant
    using ITr = simd::ty<index_type,data_type::VL>;
    
    // Create variable
    auto var = array_intl<value_type,index_type,AID,
			  array_encoding<void>,false>();

    // Extract mask from initial value. Do a deep analysis, including
    // refop arguments.
    auto mask = extract_mask( init );

    // Create reference to variable
    auto ref = make_refop( var, value<ITr, vk_zero>() );

    // Create AST with continuation
    // The remove_mask call on the initial value is not critical to
    // correctness: if a mask is computed, it may result in unnecessary
    // blend operations. Extract_mask may find masks that remove_mask
    // cannot remove.
    // Could introduce an unop_remove_mask that explicitly discards
    // any mask when storing to ref.
    return make_seq( ref = remove_mask( init ),
		     cont( add_mask( ref, mask ) ) );
}

/***********************************************************************
 * Auxiliaries
 ***********************************************************************/
template<typename E, typename M>
auto get_mask( binop<E,M,binop_mask> e ) {
    return e.data2();
}

template<typename Expr>
auto get_mask( Expr e ) {
    return make_noop();
}

template<typename E, typename M>
auto get_mask_cond( binop<E,M,binop_mask> e ) {
    return e.data2();
}

template<typename E>
auto get_mask_cond( E e ) {
    return expr::true_val( e );
}

template<typename Expr>
auto remove_mask( Expr e ) {
    return e;
}

template<typename Expr, typename Mask>
auto remove_mask( binop<Expr,Mask,binop_mask> e ) {
    return e.data1();
}

} // namespace expr

#endif // GRAPTOR_DSL_AST_BINOP_H
