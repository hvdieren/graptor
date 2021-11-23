// -*- c++ -*-
#ifndef GRAPTOR_DSL_AST_TERNOP_H
#define GRAPTOR_DSL_AST_TERNOP_H

namespace expr {

/* ternop
 * A ternary operation.
 */
template<typename E1, typename E2, typename E3, typename TernOp>
struct ternop : public expr_base {
    static_assert( E1::VL == E2::VL && E2::VL == E3::VL,
		   "vector lengths must match" );

    using data_type = typename E3::data_type; // E1 is conditional for iif
    static constexpr unsigned short VL = E1::VL;
    using arg1_type = E1;
    using arg2_type = E2;
    using arg3_type = E3;
    using op_type = TernOp;

    static constexpr op_codes opcode = op_ternop;
    
    ternop( arg1_type arg1, arg2_type arg2, arg3_type arg3, TernOp )
	: m_arg1( arg1 ), m_arg2( arg2 ), m_arg3( arg3 ) { }

    const arg1_type & __attribute__((always_inline)) data1() const {
	return m_arg1;
    }
    const arg2_type & __attribute__((always_inline)) data2() const {
	return m_arg2;
    }
    const arg3_type & __attribute__((always_inline)) data3() const {
	return m_arg3;
    }

private:
    arg1_type m_arg1;
    arg2_type m_arg2;
    arg3_type m_arg3;
};

template<typename E1, typename E2, typename E3, typename TernOp>
static constexpr
auto make_ternop( E1 e1, E2 e2, E3 e3, TernOp op ) {
    if constexpr ( e2.opcode == op_constant ) {
	if constexpr ( e3.opcode == op_constant ) {
	    using Tr = simd::ty<VID,e1.VL>;
	    return make_ternop( e1,
				e2.template expand<Tr>(),
				e3.template expand<Tr>(),
				op );
	} else
	    return make_ternop( e1, expand_cst( e2, e3 ), e3, op );
    } else if constexpr ( e3.opcode == op_constant )
	return make_ternop( e1, e2, expand_cst( e3, e2 ), op );
    else
	return ternop<E1,E2,E3,TernOp>( e1, e2, e3, op );
}

/* ternop: conditional select (iif/blend)
 */
struct ternop_iif {
    template<typename C, typename E1, typename E2>
    struct types {
	using result_type = typename E2::type;
    };

    static constexpr char const * name = "ternop_iif";

    template<typename VTr, layout_t Layout1, typename MTr, layout_t Layout2,
	     layout_t Layout3, typename MPack>
    static auto evaluate( const sb::rvalue<MTr,Layout1> & c,
			  const sb::rvalue<VTr,Layout2> & l,
			  const sb::rvalue<VTr,Layout3> & r,
			  const MPack & mpack ) {
	if constexpr ( MTr::B != VTr::B && MTr::B != 1 ) {
	    using PTr = typename VTr::prefmask_traits;
	    auto cc = c.value().template convert_data_type<PTr>();
	    return make_rvalue( iif( cc, l.value(), r.value() ), mpack );
	} else {
	    return make_rvalue( iif( c.value(), l.value(), r.value() ),
				mpack );
	}
    }
    
    template<typename VTr, layout_t Layout1, typename MTr, layout_t Layout2,
	     layout_t Layout3>
    static auto evaluate( const rvalue<void,Layout1,MTr> & c,
			  const rvalue<VTr,Layout2,void> & l,
			  const rvalue<VTr,Layout3,void> & r ) {
	if constexpr ( MTr::W != VTr::W && MTr::W != 0 ) {
	    using PTr = typename VTr::prefmask_traits;
	    auto cc = c.mask().template convert<PTr>();
	    return make_rvalue( iif( cc, l.value(), r.value() ) );
	} else {
	    return make_rvalue( iif( c.mask(), l.value(), r.value() ) );
	}
    }
    
    template<typename VTr, layout_t Layout1, typename MTr1,
	     typename MTr2, layout_t Layout2, typename MTr3, layout_t Layout3>
    static auto evaluate( const rvalue<void,Layout1,MTr1> & c,
			  const rvalue<VTr,Layout2,MTr2> & l,
			  const rvalue<VTr,Layout3,MTr3> & r ) {
	static_assert( !std::is_same<MTr1,void>::value, "MTr1 must be valid" );
	auto m = join_mask<VTr>( l.mask(), r.mask() );
	if constexpr ( MTr1::W != VTr::W && MTr1::W != 0 ) {
	    using PTr = typename VTr::prefmask_traits;
	    auto cc = c.mask().template convert<PTr>();
	    return make_rvalue( iif( cc, l.value(), r.value() ), m );
	} else {
	    return make_rvalue( iif( c.mask(), l.value(), r.value() ), m );
	}
    }
};

template<typename C, typename E1, typename E2>
static constexpr auto
iif( C c, E1 e1, E2 e2 ) {
    static_assert( std::is_base_of<expr_base,C>::value, "requirement" );
    static_assert( std::is_base_of<expr_base,E1>::value, "requirement" );
    static_assert( std::is_base_of<expr_base,E2>::value, "requirement" );
    return make_ternop( c, e1, e2, ternop_iif() );
}

// An algorithm
template<short AID_Kahan_Y, short AID_Kahan_T,
typename Compensation, typename State, typename Expr>
static auto
Kahan( Compensation c, State s, Expr e,
       std::enable_if_t<std::is_base_of_v<expr_base,Compensation>
		        && std::is_base_of_v<expr_base,State>
		        && std::is_base_of_v<expr_base,Expr>> * = nullptr ) {
    return let<AID_Kahan_Y>(
	e + c,
	[&]( auto y ) {
	    return let<AID_Kahan_T>( 
		s,
		[&]( auto t ) {
		    return make_seq(
			s = t + y,
			c = (t - s) + y,
			true_val( s ) );
		} );
	} );
}

} // namespace expr

#endif // GRAPTOR_DSL_AST_TERNOP_H
