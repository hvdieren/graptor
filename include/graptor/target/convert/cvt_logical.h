// -*- c++ -*-
#ifndef GRAPTOR_TARGET_CONVERT_LOGICAL_H
#define GRAPTOR_TARGET_CONVERT_LOGICAL_H

#include <x86intrin.h>
#include <immintrin.h>
#include <type_traits>

namespace conversion {

// Use signed integers in auxiliary width conversions when allowed, as
// unsigned conversions in AVX2 and earlier can be inefficient.

template<unsigned short B, unsigned short VL>
struct int_conversion_traits<logical<B>, bool, VL> {
    using src_traits = vector_type_traits_vl<logical<B>, VL>;
    using dst_traits = vector_type_traits_vl<bool, VL>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	if constexpr ( B == 1 ) {
	    return src_traits::srli( a, 8*src_traits::W-1 );
	} else {
	    auto b = src_traits::srli( a, 8*src_traits::W-1 );
	    return int_conversion_traits<typename src_traits::int_type, int8_t,
					 VL>::convert( b );
	}
    }
};

template<unsigned short B, unsigned short VL>
struct int_conversion_traits<bool, logical<B>, VL> {
    using src_traits = vector_type_traits_vl<bool, VL>;
    using dst_traits = vector_type_traits_vl<logical<B>, VL>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	if constexpr ( B != 1 ) {
	    auto b = int_conversion_traits<uint8_t,
					   typename dst_traits::int_type,
					   VL>::convert( a );
	    return dst_traits::cmpne( b, dst_traits::setzero(),
				      target::mt_vmask() );
	} else
	    return dst_traits::cmpne( a, dst_traits::setzero(),
				      target::mt_vmask() );
    }
};

template<unsigned short B, unsigned short C, unsigned short VL>
struct int_conversion_traits<logical<B>, logical<C>, VL> {
    using src_traits = vector_type_traits_vl<logical<B>, VL>;
    using dst_traits = vector_type_traits_vl<logical<C>, VL>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	if constexpr ( B > C ) {
	    // First shift (without replication) the msb into position, then do
	    // a truncating conversion. srli is used because support is better,
	    // e.g., AVX2 has srli_epi64 but srai_epi64 requires AVX512VL
	    auto b = src_traits::srli( a, 8*(B-C) );
	    return int_conversion_traits<typename src_traits::int_type,
					 typename dst_traits::int_type,
					 VL>::convert( b );
	} else {
	    // Do a widening signed integer conversion to replicate the
	    // most significant bit of the logical
	    return int_conversion_traits<
		std::make_signed_t<typename src_traits::int_type>,
		std::make_signed_t<typename dst_traits::int_type>,
		VL>::convert( a );
	}
    }
};


} // namespace conversion

#endif // GRAPTOR_TARGET_CONVERT_LOGICAL_H
