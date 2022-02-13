// -*- C++ -*-
#ifndef GRAPTOR_API_FUSION_H
#define GRAPTOR_API_FUSION_H

#include "graptor/api/utils.h"
#include "graptor/dsl/ast/decl.h"

class frontier;

namespace api {

/************************************************************************
 * Definition of fusion method
 ************************************************************************/
template<typename Fn>
using is_fusion_method =
    std::is_invocable<Fn,expr::value<simd::ty<VID,1>,expr::vk_dst>>;

template<typename RFn>
struct arg_fusion {
    arg_fusion( RFn method ) : m_method( method ) { }

    template<typename VIDDst>
    auto fusionop( VIDDst d ) const {
	return m_method( d );
    }

    const RFn m_method;
};

struct missing_fusion_argument {
    template<typename VIDDst>
    auto fusionop( VIDDst d ) const {
	// A value of false indicates that a vertex cannot be processed
	// immediately in the fused operation, hence it disables fusion.
	return expr::value<simd::ty<bool,VIDDst::VL>,expr::vk_false>();
    }
};

template<typename... Args>
auto fusion( Args && ... args ) {
    auto & fn = get_argument_value<is_fusion_method,missing_fusion_argument>(
	args... );

    static_assert( !std::is_same_v<std::decay_t<decltype(fn)>,
		   missing_fusion_argument>,
		   "must specify fusion method to relax operation" );

    return arg_fusion<std::decay_t<decltype(fn)>>( fn );
}

template<typename T>
struct is_fusion : public std::false_type { };

template<typename Fn>
struct is_fusion<arg_fusion<Fn>> : public std::true_type { };

template<typename Op>
struct has_fusion_op {
    static constexpr bool value = 
	!expr::is_constant_false<
	decltype(((Op*)nullptr)->fusionop(
		     expr::value<simd::ty<VID,1>,expr::vk_dst>() ))>::value;
};

template<typename Op>
constexpr bool has_fusion_op_v = has_fusion_op<Op>::value;

} // namespace api

#endif // GRAPTOR_API_FUSION_H
