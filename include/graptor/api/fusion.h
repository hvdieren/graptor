// -*- C++ -*-
#ifndef GRAPTOR_API_FUSION_H
#define GRAPTOR_API_FUSION_H

#include "graptor/api/utils.h"
#include "graptor/dsl/ast/decl.h"

class frontier;

namespace api {

/************************************************************************
 * Definition of fusion flags.
 * The utility functions assume that the fusion_flags values fit in the
 * int type, i.e., sizeof(fusion_flags) <= sizeof(int).
 ************************************************************************/
enum class fusion_flags {
    no_duplicate_processing = 1,
    no_duplicate_reporting = 2,
    no_reporting_processed = 4,
    default_flags = no_duplicate_reporting,
    no_activation_checking = 0
};

inline constexpr bool is_set( fusion_flags flags, fusion_flags f ) {
    static_assert( sizeof(fusion_flags) <= sizeof(int) );
    return ( int(flags) & int(f) ) != 0;
}

inline constexpr fusion_flags
operator | ( fusion_flags lhs, fusion_flags rhs ) {
    static_assert( sizeof(fusion_flags) <= sizeof(int) );
    return fusion_flags( int(lhs) | int(rhs) );
}

template<fusion_flags fl>
struct fusion_flags_argument {
    static constexpr fusion_flags flags = fl;
};

static constexpr auto no_duplicate_processing =
    fusion_flags_argument<fusion_flags::no_duplicate_processing>();
static constexpr auto no_duplicate_reporting =
    fusion_flags_argument<fusion_flags::no_duplicate_reporting>();
static constexpr auto no_reporting_processed =
    fusion_flags_argument<fusion_flags::no_reporting_processed>();
static constexpr auto no_activation_checking =
    fusion_flags_argument<fusion_flags::no_activation_checking>();
static constexpr auto default_fusion_flags =
    fusion_flags_argument<fusion_flags::default_flags>();

template<fusion_flags lhs, fusion_flags rhs>
constexpr fusion_flags_argument<lhs | rhs>
operator | ( fusion_flags_argument<lhs>, fusion_flags_argument<rhs> ) {
    return fusion_flags_argument<lhs | rhs>();
}

template<typename Fn>
struct is_fusion_flags : public std::false_type { };

template<fusion_flags flags>
struct is_fusion_flags<fusion_flags_argument<flags>>
    : public std::true_type { };

/************************************************************************
 * Definition of fusion method
 ************************************************************************/
template<typename Fn>
using is_fusion_method =
    std::is_invocable<Fn,expr::value<simd::ty<VID,1>,expr::vk_src>,
		      expr::value<simd::ty<VID,1>,expr::vk_dst>,
		      expr::value<simd::ty<EID,1>,expr::vk_edge> >;

template<fusion_flags flags, typename RFn>
struct arg_fusion {
    arg_fusion( RFn method ) : m_method( method ) { }

    template<typename VIDSrc, typename VIDDst, typename EIDEdge>
    auto fusionop( VIDSrc s, VIDDst d, EIDEdge e ) const {
	return m_method( s, d, e );
    }

    constexpr fusion_flags get_flags() const { return flags; }

private:
    const RFn m_method;
};

struct missing_fusion_argument {
    template<typename VIDSrc, typename VIDDst, typename EIDEdge>
    auto fusionop( VIDSrc s, VIDDst d, EIDEdge e ) const {
	// A value of 0 indicates that a vertex cannot be processed
	// immediately in the fused operation, hence it disables fusion.
	// The value of 0 causes the vertex to be returned in the frontier
	// (if any), which is a useful default.
	return expr::value<simd::ty<bool,VIDDst::VL>,expr::vk_zero>();
    }

    constexpr fusion_flags get_flags() const {
	return fusion_flags::default_flags;
    }
};

template<typename... Args>
auto fusion( Args && ... args ) {
    auto & fn = get_argument_value<is_fusion_method,missing_fusion_argument>(
	args... );
    constexpr fusion_flags fl = get_argument_type_t<
	is_fusion_flags,
	fusion_flags_argument<fusion_flags::default_flags>,
	Args...>::flags;

    static_assert( !std::is_same_v<std::decay_t<decltype(fn)>,
		   missing_fusion_argument>, "must specify fusion method" );

    return arg_fusion<fl,std::decay_t<decltype(fn)>>( fn );
}

template<typename T>
struct is_fusion : public std::false_type { };

template<fusion_flags flags, typename Fn>
struct is_fusion<arg_fusion<flags,Fn>> : public is_fusion_method<Fn> { };

template<typename Op>
struct has_fusion_op {
private:
    using ty = decltype(((Op*)nullptr)->fusionop(
			    expr::value<simd::ty<VID,1>,expr::vk_src>(),
			    expr::value<simd::ty<VID,1>,expr::vk_dst>(),
			    expr::value<simd::ty<EID,1>,expr::vk_edge>()
			    ));
public:
    static constexpr bool value = 
	!expr::is_constant_false<ty>::value
	&& !expr::is_constant_zero<ty>::value;
};

template<typename Op>
constexpr bool has_fusion_op_v = has_fusion_op<Op>::value;

} // namespace api

#endif // GRAPTOR_API_FUSION_H
