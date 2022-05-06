// -*- c++ -*-
#ifndef GRAPTOR_DSL_AST_DFSAOP_H
#define GRAPTOR_DSL_AST_DFSAOP_H

#include "graptor/simd/ops.h"
#include "graptor/dsl/aval.h"

namespace expr {

/* dfsaop
 * A ternary operation.
 */
template<typename S, typename U, typename C, typename DFSAOp>
struct dfsaop : public expr_base {
    static_assert( S::VL == U::VL && U::VL == C::VL,
		   "vector lengths must match" );

    using data_type = typename DFSAOp::template types<S,U,C>::result_type;
    using type = typename data_type::member_type;
    static constexpr unsigned short VL = S::VL;
    using state_type = S;
    using update_type = U;
    using condition_type = C;
    using op_type = DFSAOp;

    static constexpr op_codes opcode = op_dfsaop;
    
    dfsaop( state_type state, update_type update, condition_type condition,
	    DFSAOp )
	: m_state( state ), m_update( update ), m_condition( condition ) { }

    state_type state() const { return m_state; }
    update_type update() const { return m_update; }
    condition_type condition() const { return m_condition; }

private:
    state_type m_state;
    update_type m_update;
    condition_type m_condition;
};

template<typename S, typename U, typename C, typename DFSAOp>
dfsaop<S,U,C,DFSAOp> make_dfsaop( S state, U update, C condition, DFSAOp op ) {
    return dfsaop<S,U,C,DFSAOp>( state, update, condition, op );
}

/* dfsaop: Maximum Independent Set (application-specific)
 */
struct dfsaop_MIS {
    // Relation determining new value of d:
    // d\s ud ci in ou
    //  ud *d *d ou *d
    //  ci ud ud ou *d -- set ud only if d < s ...
    //  in *d *d ou *d
    //  ou *d *d *d *d
    // *d: keep d as it was
    enum mis_state_t {
	mis_undecided = 0,
	mis_conditionally_in = 1,
	mis_out = 2,
	mis_in = 3
    };

    template<typename S, typename U, typename C>
    struct types {
	using result_type = typename S::data_type::prefmask_traits;
    };

    static constexpr char const * name = "dfsaop_MIS";

    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename MTr, layout_t Layout3,
	     typename MPack>
    static auto evaluate(
	sb::rvalue<VTr,Layout1> s,
	sb::rvalue<VTr,Layout2> u,
	sb::rvalue<MTr,Layout3> c,
	const MPack & mpack ) {
	static_assert( VTr::VL == MTr::VL, "vector length match" );
	static_assert( simd::detail::is_mask_traits_v<MTr>,
		       "c argument must be a mask" );
	simd::detail::vec<VTr,simd::lo_unknown> r
	    = eval( s.value(), u.value(), c.value() );
	return make_rvalue( r, mpack );
    }

    template<typename VTr, layout_t Layout1, layout_t Layout2,
	     typename MTr, layout_t Layout3,
	     typename I, typename Enc, bool NT, typename MPack>
    static auto
    evaluate( sb::lvalue<VTr,I,Enc,NT,Layout1> s,
	      sb::rvalue<VTr,Layout2> u,
	      sb::rvalue<MTr,Layout3> c,
	      const MPack & mpack ) {
	static_assert( VTr::VL == MTr::VL, "vector length match" );
	static_assert( simd::detail::is_mask_traits_v<MTr>,
		       "c argument must be a mask" );

	auto mpack2 = mpack.template clone_and_add<MTr>();
	auto upd = mpack2.template get_mask<MTr>();

	simd::detail::vector_ref_impl<VTr,I,Enc,NT,Layout1> sref = s.value();
	simd::vec<VTr,simd::lo_unknown> sval = sref.load( upd );
	simd::vec<VTr,simd::lo_unknown> r = eval( sval, u.value(), c.value() );
	sref.store( r, upd );
	return make_rvalue( sval != r, mpack2 );
    }

    // used as a ternop (returns value like storeop and also update mask)
    template<typename VTr, layout_t Layout1, typename MTr1,
	     typename MTr2, layout_t Layout2, typename MTr3, layout_t Layout3>
    static auto evaluate(
	rvalue<VTr,Layout1,MTr1> s, rvalue<VTr,Layout2,MTr2> u, rvalue<void,Layout3,MTr3> c,
	std::enable_if_t<simd::matchVL<VTr,MTr1,MTr2,MTr3>::value> *
	= nullptr ) {
	simd::detail::vector_impl<VTr> r = eval( s.value(), u.value(), c.mask()  );
	return make_rvalue( r, s.mask() & u.mask() );
    }
    
    // used as dfsaop (similar to redop: return update mask, no value)
    template<typename VTr, typename I, typename MTr1,
	     layout_t LayoutR, layout_t Layout2, typename MTr2,
	     layout_t Layout3, typename MTr3, typename Enc, bool NT>
    static auto evaluate(
	lvalue<VTr,I,MTr1,Enc,NT,LayoutR> s, rvalue<VTr,Layout2,MTr2> u, rvalue<void,Layout3,MTr3> c,
	std::enable_if_t<simd::matchVL<VTr,MTr1,MTr2,MTr3>::value> *
	= nullptr ) {
	auto upd = s.mask() & u.mask();
	simd::detail::vector_ref_impl<VTr,I,Enc,NT,LayoutR> sref = s.value();
	simd::vec<VTr,simd::lo_unknown> sval = sref.load( upd );
	simd::vec<VTr,simd::lo_unknown> r = eval( sval, u.value(), c.mask() );
	sref.store( r, upd );
	if constexpr ( std::is_void_v<MTr1> && std::is_void_v<MTr2> ) {
	    using Tr = typename VTr::prefmask_traits;
	    return make_rvalue( simd::detail::vector_impl<Tr>::true_mask() );
	} else {
	    return make_rvalue( upd );
	}
    }

    template<typename VTr, typename MTr1,
	     layout_t LayoutR, layout_t Layout2, typename MTr2,
	     layout_t Layout3, typename MTr3,
	     typename I, typename Enc, bool NT>
    static GG_INLINE auto
    evaluate_atomic( lvalue<VTr,I,MTr1,Enc,NT,LayoutR> s, rvalue<VTr,Layout2,MTr2> u,
		     rvalue<void,Layout3,MTr3> c,
		     std::enable_if_t<simd::matchVL<VTr,MTr1,MTr2>::value> *
		     = nullptr ) {
	simd::detail::vector_impl<VTr> r, sval;
	simd::detail::vector_ref_impl<VTr,I,Enc,NT,LayoutR> sref = s.value();
	auto upd = s.mask() & u.mask();
	if( upd.data() ) {
	    do {
		sval = sref.load();
		r = eval( sval, u.value(), c.mask()  );
	    } while( ( r != sval ).data() && !sref.cas( sval, r ) );
	    return make_rvalue( r != sval );
	} else
	    return make_rvalue( upd );
    }

    template<typename VTr, typename I, typename MTr, typename Enc, bool NT,
	     layout_t LayoutR,
	     layout_t Layout2,
	     layout_t Layout3,
	     typename MPack>
    static auto
    evaluate_atomic( sb::lvalue<VTr,I,Enc,NT,LayoutR> s,
		     sb::rvalue<VTr,Layout2> u,
		     sb::rvalue<MTr,Layout3> c,
		     const MPack & mpack ) {
	simd::detail::vector_impl<VTr> r, sval;
	simd::detail::vector_ref_impl<VTr,I,Enc,NT,LayoutR> sref = s.value();
	auto upd = mpack.template get_any<simd::detail::mask_bool_traits>();
	if( upd.data() ) {
	    do {
		sval = sref.load();
		r = eval( sval, u.value(), c.value()  );
	    } while( ( r != sval ).data() && !sref.cas( sval, r ) );
	    return make_rvalue( r != sval );
	} else
	    return make_rvalue( upd );
    }

    template<typename VTr, typename I, typename MTr, typename Enc, bool NT,
	     layout_t LayoutR,
	     layout_t Layout2,
	     layout_t Layout3,
	     typename MPack>
    static auto
    evaluate_atomic( sb::lvalue<VTr,I,Enc,NT,LayoutR> s,
		     sb::rvalue<VTr,Layout2> u,
		     sb::rvalue<MTr,Layout3> c,
		     const MPack & mpack ) {
	simd::detail::vector_impl<VTr> r, sval;
	simd::detail::vector_ref_impl<VTr,I,Enc,NT,LayoutR> sref = s.value();
	auto upd = mpack.template get_any<simd::detail::mask_bool_traits>();
	if( upd.data() ) {
	    do {
		sval = sref.load();
		r = eval( sval, u.value(), c.value()  );
	    } while( ( r != sval ).data() && !sref.cas( sval, r ) );
	    return make_rvalue( r != sval );
	} else
	    return make_rvalue( upd );
    }

private:
    template<typename VTr, typename MTr, layout_t Layout1, layout_t Layout2>
    static simd::vec<VTr,simd::lo_unknown> eval(
	simd::vec<VTr,Layout1> sval,
	simd::vec<VTr,Layout2> uval,
	simd::detail::mask_impl<MTr> c,
	std::enable_if_t<simd::matchVL<VTr,MTr>::value> * = nullptr ) {
	using Vec = simd::vec<VTr,simd::lo_unknown>;
	using Scalar = typename VTr::member_type;

	Vec u_in( (Scalar)mis_in );
	Vec s_conditionally_in( (Scalar)mis_conditionally_in );
	Vec s_undecided( (Scalar)mis_undecided );
	Vec s_out( (Scalar)mis_out );
	
	// Optimisation: uval < s_out is inefficient in 2-bit vectors that
	// return 2-bit logical masks. Replace by tweaked instruction sequence
	// Instruction sequence only works for vectorised bitfield<2>, not in
	// scalar case (e.g., sparse)
	if constexpr ( VTr::B == 2 && VTr::VL > 1 ) {
	    // Version seems to be incorrect.
	    Vec nval =
		::iif( uval == u_in,
		       // false case: flags[s] != IN
		       ::iif( join_mask<VTr>(
				  join_mask<VTr>( c,
						  sval == s_conditionally_in ),
				  (~uval).asmask() ), // flips top bit, becomes mask
			      // false case: no-op
			      sval,
			      // true case:
			      s_undecided ),
		       // true case: flags[s] == IN
		       s_out );
	    return nval;
	} else {
	    Vec nval =
		::iif( uval == u_in,
		       // false case: flags[s] != IN
		       ::iif( join_mask<VTr>(
				  join_mask<VTr>( c,
						  sval == s_conditionally_in ),
				  uval < s_out ),
			      // false case: no-op
			      sval,
			      // true case:
			      s_undecided ),
		       // true case: flags[s] == IN
		       s_out );
	    return nval;
	}
    }

public:
    // used as a redop
    template<typename VTr, layout_t Layout>
    static auto evaluate1( rvalue<VTr,Layout,void> v ) {
	// *** THIS DEPENDS ON THE NUMERIC VALUES OF THE ENUM ***
	// If any lane is out, we're out
	// Else, if any lane is in, we're in
	// Else, if any lane is undecided, we're undecided
	// Else, if any lane is conditionally_in, we're in
	// So we're looking for a maximum with in/out inverted
	// Do a trick: flip lowest bit, then do maximum.
	// This avoids iterating over (potentially) many lanes.
	simd::detail::vector_impl<VTr> flip =
	    simd::detail::vector_impl<VTr>::one_val();
	using VTr1 = typename VTr::template rebindVL<1>::type;
	simd::detail::vector_impl<VTr1> v1 = reduce_max( v.value() ^ flip );
	return make_rvalue(
	    simd::detail::vector_impl<VTr1>( v1.data() ^ 1 ) ); // scalar ^
    }
    template<typename VTr, layout_t Layout, typename MTr>
    static auto
    evaluate1( rvalue<VTr,Layout,MTr> v,
	       std::enable_if_t<!std::is_void<MTr>::value> * = nullptr ) {
	// As above, now with mask
	simd::detail::vector_impl<VTr> flip =
	    simd::detail::vector_impl<VTr>::one_val();
	using VTr1 = typename VTr::template rebindVL<1>::type;
	simd::detail::vector_impl<VTr1> v1
	    = reduce_max( v.value() ^ flip, v.mask() );
	return make_rvalue(
	    simd::detail::vector_impl<VTr1>( v1.data() ^ 1 ), // scalar ^
	    reduce_logicalor( v.mask() ) );
    }
};

template<typename S, typename U, typename C, typename DFSAOp>
static constexpr 
std::enable_if_t<std::is_base_of<expr_base,S>::value
		 && std::is_base_of<expr_base,U>::value
		 && std::is_base_of<expr_base,C>::value,
		 dfsaop<S,U,C,DFSAOp>>
make_dfsaop( S state, U update, C condition ) {
    return dfsaop<S,U,C,DFSAOp>( state, update, condition, DFSAOp() );
}

template<typename S, typename U, typename C>
static constexpr dfsaop<S,U,C,dfsaop_MIS>
make_dfsaop_MIS( S state, U update, C condition ) {
    return make_dfsaop( state, update, condition, dfsaop_MIS() );
}

} // namespace expr

#endif // GRAPTOR_DSL_AST_DFSAOP_H
