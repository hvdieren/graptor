// -*- c++ -*-
#ifndef GRAPTOR_DSL_EMAP_UTILS_H
#define GRAPTOR_DSL_EMAP_UTILS_H

#include <cstdio>
#include <cstddef>
#include <cassert>
#include <type_traits>
#include <algorithm>
#include <iostream>

template<typename Operator>
struct callable_relax_t {
    auto operator() () {
	return op->relax( expr::value<simd::ty<VID,1>,expr::vk_src>(),
			  expr::value<simd::ty<VID,1>,expr::vk_dst>(),
			  expr::value<simd::ty<EID,1>,expr::vk_edge>() );
    }
    Operator * op;
};

template<typename Operator>
struct callable_active_t {
    auto operator() () {
	return op->active( expr::value<simd::ty<VID,1>,expr::vk_dst>() );
    }
    Operator * op;
};

template<typename Operator, typename Enable = void>
struct has_active_t {
    static constexpr bool value
    = !expr::is_constant_true<std::invoke_result_t<callable_active_t<Operator>>>::value;
};

template<typename Operator>
static constexpr bool hasActive( Operator op ) {
    // auto is_active = op.active( expr::value<simd::ty<VID,1>,expr::vk_dst>() );
    // return !expr::is_constant_true<decltype(is_active)>::value;
    return has_active_t<Operator>::value;
}

template<typename Operator>
struct has_redop_operator_t
    : public expr::is_redop<std::invoke_result_t<callable_relax_t<Operator>>> { };


template<typename GraphType, typename Tr>
void validate( const GraphType & Gcsr,
	       simd::detail::vector_impl<Tr> vsrc,
	       simd::detail::vector_impl<Tr> vdst ) {
    for( unsigned short l=0; l < Tr::VL; ++l ) {
	assert( vsrc.at(l) != ~VID(0) );
	assert( vdst.at(l) != ~VID(0) );
	if( vsrc.at(l) != (~VID(0))>>1
	    && vdst.at(l) != (~VID(0))>>1
	    && !Gcsr.hasEdge( vsrc.at(l), vdst.at(l) ) ) {
	    std::cerr << "ERROR: l=" << l << " src=" << vsrc.at(l)
		      << " dst=" << vdst.at(l) << "\n";
	    assert( false && "non-existing edge visited" );
	}
    }
}

template<typename GraphType, typename Tr, typename Tr1>
typename std::enable_if<std::is_same<typename Tr::element_type,
				     typename Tr1::element_type>::value
&& Tr1::VL == 1 && Tr::VL != 1>::type
validate( const GraphType & Gcsr,
	       simd::detail::vector_impl<Tr> vsrc,
	       simd::detail::vector_impl<Tr1> sdst ) {
    assert( sdst.at(0) != ~VID(0) );
    if( sdst.at(0) == (~VID(0))>>1 ) // all edges inactive
	return;
    for( unsigned short l=0; l < Tr::VL; ++l ) {
	assert( vsrc.at(l) != ~VID(0) );
	if( vsrc.at(l) != (~VID(0))>>1
	    && !Gcsr.hasEdge( vsrc.at(l), sdst.at(0) ) ) {
	    std::cerr << "ERROR: l=" << l << " src=" << vsrc.at(l)
		      << " dst=" << sdst.at(0) << "\n";
	    assert( false && "non-existing edge visited" );
	}
    }
}

// Ties in with how LICM works: if pe() is a value of vk_smk, then there is
// actual post-expression (it has no side-effects).
template<typename Expr0, typename Expr1>
auto append_pe( Expr0 e0, Expr1 e1 ) {
    return expr::make_seq( e0, e1 );
}

template<typename Tr, typename Expr1>
auto append_pe( expr::value<Tr,expr::vk_smk> e0, Expr1 e1 ) {
    return e1;
}

template<unsigned short VL, typename Expr, typename Cache>
GG_INLINE inline void
GraptorCSRVertexOp( unsigned int p, const partitioner & part,
		    Expr e, Cache & c ) {
    VID lo = part.start_of(p);
    VID hi = part.end_of(p);
    auto vdst = simd::create_set1inc<VID, VL, false>( lo );
    auto vpid = simd::template create_constant<simd::ty<VID,1>>( VL*(VID)p );
    auto vvstep = simd::create_constant<VID,VL>( (VID)VL );
    for( VID v=lo; v < hi; v += VL ) {
	// Evaluate hoisted part of expression.
	// This will include the count of active
	// vertices and edges in the new frontier.
	auto m1 = expr::create_value_map_new2<VL,expr::vk_pid,expr::vk_dst>(
	    vpid, vdst );
	expr::evaluate( c, m1, e );
	vdst += vvstep;
    }
}

template<unsigned short VL, typename Environment, typename Expr, typename Cache>
GG_INLINE inline void
GraptorCSRVertexOp( const Environment & env,
		    unsigned int p, const partitioner & part,
		    Expr e, Cache & c ) {
    VID lo = part.start_of(p);
    VID hi = part.end_of(p);
    auto vdst = simd::create_set1inc<VID, VL, false>( lo );
    auto vpid = simd::template create_constant<simd::ty<VID,1>>( VL*(VID)p );
    auto vvstep = simd::create_constant<VID,VL>( (VID)VL );
    for( VID v=lo; v < hi; v += VL ) {
	// Evaluate hoisted part of expression.
	// This will include the count of active
	// vertices and edges in the new frontier.
	auto m1 = expr::create_value_map_new2<VL,expr::vk_pid,expr::vk_dst>(
	    vpid, vdst );
	env.evaluate( c, m1, e );
	vdst += vvstep;
    }
}

#endif // GRAPTOR_DSL_EMAP_UTILS_H

