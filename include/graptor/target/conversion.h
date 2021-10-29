// -*- c++ -*-
#ifndef GRAPTOR_TARGET_CONVERSION_H
#define GRAPTOR_TARGET_CONVERSION_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include "graptor/target/decl.h"

alignas(64) extern const uint8_t conversion_4fx8_cfp16x8_shuffle[32];
alignas(64) extern const uint8_t conversion_4x4_2x4_shuffle[16];
alignas(64) extern const uint8_t conversion_8x4_2x4_shuffle[32];
alignas(64) extern const uint8_t conversion_4x8_1x8_shuffle[32];
alignas(64) extern const uint8_t conversion_4x8_1x8_shuffle_hi[32];

// namespace target {

/***********************************************************************
 * Frequently used conversion utility
 ***********************************************************************/
#if __AVX2__
#include "avx2_bitwise.h"
namespace target {
inline __m128i convert_4b_1b( __m256i a ) {
#if __AVX512F__ && __AVX512VL__
    __m128i c = _mm256_cvtepi32_epi8( a );
#else
    const __m256i ctrl = _mm256_load_si256(
	reinterpret_cast<const __m256i*>( conversion_4x8_1x8_shuffle ) );
    __m256i b = _mm256_shuffle_epi8( a, ctrl );
    using it = avx2_bitwise;
    __m128i lo = it::lower_half( b );
    __m128i hi = it::upper_half( b );
    __m128i c = _mm_or_si128( lo, hi );
#endif
    return c;
}

inline __m128i convert_4b_1b_hi( __m256i a ) {
#if __AVX512F__ && __AVX512VL__
    __m128i c = _mm_bslli_si128( _mm256_cvtepi32_epi8( a ), 8 );
#else
    const __m256i ctrl = _mm256_load_si256(
	reinterpret_cast<const __m256i*>( conversion_4x8_1x8_shuffle_hi ) );
    __m256i b = _mm256_shuffle_epi8( a, ctrl );
    using it = avx2_bitwise;
    __m128i lo = it::lower_half( b );
    __m128i hi = it::upper_half( b );
    __m128i c = _mm_or_si128( lo, hi );
#endif
    return c;
}
} // namespace target
#endif // __AVX2__

/***********************************************************************
 * Lane width and type conversions
 ***********************************************************************/
template<typename T, typename V, unsigned short VL, typename Enable = void>
struct conversion_traits;

/***********************************************************************
 * Auxiliary for floating point types
 ***********************************************************************/
template<typename FTy> struct fp_traits;

template<>
struct fp_traits<float> {
    static constexpr bool sign_bit = true;
    static constexpr unsigned short exponent_bits = 8;
    static constexpr unsigned short mantissa_bits = 23;
    static constexpr bool exponent_truncated = false;
    static constexpr bool maybe_zero = true;
    static constexpr int exponent_bias = 0;
};

template<>
struct fp_traits<double> {
    static constexpr bool sign_bit = true;
    static constexpr unsigned short exponent_bits = 11;
    static constexpr unsigned short mantissa_bits = 52;
    static constexpr bool exponent_truncated = false;
    static constexpr bool maybe_zero = true;
    static constexpr int exponent_bias = 0;
};


// Convert biased exponent to biased exponent, where the bias is
// 2**e-1 for e exponent bits
// Or convert against a truncated exponent that is always negative
// and assumed to fit the range
template<typename SrcTy, typename DstTy, unsigned short VL>
std::enable_if_t<(sizeof(DstTy)>=sizeof(SrcTy)),
		     typename vector_type_traits_vl<DstTy,VL>::itype>
cvt_float_widen_static( typename vector_type_traits_vl<DstTy,VL>::itype a ) {
    using src_traits = vector_type_traits_vl<SrcTy,VL>;
    using dst_traits = vector_type_traits_vl<DstTy,VL>;

    constexpr unsigned short ss = fp_traits<SrcTy>::sign_bit ? 1 : 0;
    constexpr unsigned short se = fp_traits<SrcTy>::exponent_bits;
    constexpr unsigned short sm = fp_traits<SrcTy>::mantissa_bits;
    constexpr unsigned short st = fp_traits<SrcTy>::exponent_truncated ? 1 : 0;
    constexpr bool sz = fp_traits<SrcTy>::maybe_zero;
    constexpr int sb = fp_traits<SrcTy>::exponent_bias;
    constexpr unsigned short sW = src_traits::W;
    constexpr unsigned short ds = fp_traits<DstTy>::sign_bit ? 1 : 0;
    constexpr unsigned short de = fp_traits<DstTy>::exponent_bits;
    constexpr unsigned short dm = fp_traits<DstTy>::mantissa_bits;
    constexpr unsigned short dW = dst_traits::W;
    constexpr unsigned short dt = fp_traits<DstTy>::exponent_truncated ? 1 : 0;
    constexpr bool dz = fp_traits<DstTy>::maybe_zero;
    constexpr int db = fp_traits<DstTy>::exponent_bias;

    using sitype = typename src_traits::int_type;
    using ditype = typename dst_traits::int_type;

    static_assert( sW <= dW, "due to SFINAE restriction" );
    static_assert( !dt, "have not thought through the case where dt==true" );
    static_assert( db == 0, "bias currently not taken into account" );

    if constexpr ( sW < dW ) {
	using it = typename dst_traits::int_traits;

	static_assert( ds && se <= de && sm <= dm,
		       "implemented config" );

	// If the source is truncated, we need to fill in de-se bits with the
	// pattern 01...1. If the source is not truncated (and the destination
	// is not), we add a pattern of de-se+1 bits that is 01...1 (1 extra 1)
	// to the exponent, where the lowest bit of the pattern is aligned to
	// the highest of the source's exponent.
	// Note that when de == se, the mask will be zero still when using
	// vector instructions (depends on implementation of scalar_int for
	// scalar case).
	const auto eoff = it::srli(
	    it::slli( it::setone(), dW*8-(de-se)+st ), 1+ds );

	static_assert( ss+se+sm == sW*8, "assumption" );

	if constexpr ( ss != 0 ) {
	    static_assert( ds, "assumption - impacts mask" );

	    // smask identifies the sign bit, and other bits to retain from a
	    const auto mask = it::srli( it::setone(), 1+(de-se) );
	    const auto smask
		= it::bitwise_or( it::slli( it::setone(), dW*8-1 ), mask );

	    auto b = it::slli( a, (dW-sW) * 8 );

	    typename it::type e;
	    
	    if constexpr ( de != se ) {
		auto c = it::srai( b, de-se );

		// TODO: bitblend through vpternlog is most efficient when
		//       final argument is a dead value, as this value
		//       will be overwritten by the assembly instruction
		if constexpr ( st )
		    e = it::bitblend( smask, eoff, c );
		else {
		    auto d = it::bitwise_andnot( smask, c );
		    e = it::add( d, eoff ); // need addition if exp > 0
		}
	    } else
		e = b;

	    if constexpr ( sb != 0 ) { // adjust bias
		const auto bias
		    = it::set1( typename it::member_type( sb ) << dm );
		e = it::add( e, bias );
	    }

	    if constexpr ( dz && sz ) { // support zero value
		auto z = it::setzero();
		auto s = it::cmpeq( z, a, typename it::mt_preferred() );
		auto f = it::blend( s, e, z );
		return f;
	    } else
		return e;
	} else {
	    typename it::type e;
	    
	    if constexpr ( true || de != se ) {
		const auto delta = (de-se)+(ds-ss);
		const auto mask = it::srli( it::setone(), delta );
		// Also assuming exponent is positive
		auto b = it::slli( a, (dW-sW) * 8 - delta );

		if constexpr ( st )
		    e = it::bitblend( mask, eoff, b );
		else {
		    // Not guaranteed that exp < 0, or no optimisation applied,
		    // or ternary not available.
		    // The bitwise_andnot is required as the higher-order part
		    // of a lane may contain undefined bits (not zero),
		    // including those bits where we are about to extend the
		    // exponent.
		    auto d = it::bitwise_andnot( mask, b );
		    e = it::add( d, eoff );
		}
	    } else {
		// Problem here is that a sign bit in the destination may
		// be undefined, so we need to make sure to override it.
		// However, for some reason, sign bit position is already zero
		// in appear in practice, so no need to mask it out (hence
		// this path is de-activated)...
		// A solution with two shifts. One shift is redundant to
		// shift in encoding_wide, if called from there.
		// A solution with shift and bitwise and-not is also possible
		// but requires an additional vector register to hold a constant
		// Shift all the way to the left
		auto b = it::slli( a, (dW-sW) * 8 );
		if constexpr ( ds )
		    // Shift in zero sign bit
		    e = it::srli( b, 1 );
		else
		    e = b;
	    }

	    if constexpr ( sb != 0 ) { // adjust bias
		const auto bias
		    = it::set1( typename it::member_type( sb ) << dm );
		e = it::add( e, bias );
	    }

	    if constexpr ( dz && sz ) { // support zero value
		auto z = it::setzero();
		auto s = it::cmpeq( z, a, typename it::mt_preferred() );
		auto f = it::blend( s, e, z );
		return f;
	    } else
		return e;
	}
    } else { // dW == sW -- assume identical formats
	static_assert( ss == ds && se == de && sm == dm && st == dt,
		       "NYI" );
	static_assert( sb == 0, "bias currently not taken into account" );
	return a;
    }

    assert( 0 && "NYI" );
}

template<typename SrcTy, typename DstTy, unsigned short VL>
std::enable_if_t<(sizeof(DstTy)>=sizeof(SrcTy)),
		     typename vector_type_traits_vl<DstTy,VL>::itype>
cvt_float_widen( typename vector_type_traits_vl<DstTy,VL>::itype a ) {
    return cvt_float_widen_static<SrcTy,DstTy,VL>( a );
}

template<typename SrcTy, typename DstTy, unsigned short VL>
std::enable_if_t<(sizeof(DstTy)<sizeof(SrcTy)),
		 typename vector_type_traits_vl<SrcTy,VL>::itype>
cvt_float_narrowing_static(
    typename vector_type_traits_vl<SrcTy,VL>::itype a ) {
    using src_traits = vector_type_traits_vl<SrcTy,VL>;
    using dst_traits = vector_type_traits_vl<DstTy,VL>;

    constexpr unsigned short ss = fp_traits<SrcTy>::sign_bit ? 1 : 0;
    constexpr unsigned short se = fp_traits<SrcTy>::exponent_bits;
    constexpr unsigned short sm = fp_traits<SrcTy>::mantissa_bits;
    constexpr int sb = fp_traits<SrcTy>::exponent_bias;
    constexpr unsigned short sW = src_traits::W;
    constexpr unsigned short st = fp_traits<SrcTy>::exponent_truncated ? 1 : 0;
    constexpr unsigned short ds = fp_traits<DstTy>::sign_bit ? 1 : 0;
    constexpr unsigned short de = fp_traits<DstTy>::exponent_bits;
    constexpr unsigned short dm = fp_traits<DstTy>::mantissa_bits;
    constexpr int db = fp_traits<DstTy>::exponent_bias;
    constexpr bool dz = fp_traits<DstTy>::maybe_zero;
    constexpr unsigned short dW = dst_traits::W;
    constexpr unsigned short dt = fp_traits<DstTy>::exponent_truncated ? 1 : 0;

    using sitype = typename src_traits::int_type;
    using ditype = typename dst_traits::int_type;

    static_assert( sW > dW, "due to SFINAE restriction" );
    static_assert( !st, "haven't thought about st == true" );
    static_assert( sb == 0, "bias currently not taken into account" );

    using it = typename src_traits::int_traits;

    if constexpr ( dt ) {
	// We only need a slice of the exponent and the mantissa
	// Only the lower part of the source word corresponding
	// to the width dW is defined; the higher part of the source is
	// undefined
	if constexpr ( ds == 0 ) {
	    if constexpr ( (de+dm) == dW*8 ) {
		auto b = it::srli( a, sm-dm );
		if constexpr ( db != 0 ) {
		    // TODO: do we need to zero the sign bit? It should
		    //       be zero already by assumption of ds == 0
		    const auto bias
			= it::set1( typename it::member_type( db ) << dm );
		    b = it::sub( b, bias );

		    // recognise zero, distorted by bias manipulation
		    if constexpr ( dz ) {
			// zero is correctly treated if db == 0 or no bias
			// manipulation is applied at all
			auto zero = it::setzero();
			auto is_zero = it::cmpeq( a, zero,
						  typename it::mt_preferred() );
			b = it::blend( is_zero, b, zero );
		    }
		}
		return b;
	    } else {
		static_assert( (de+dm) < dW*8, "by elimination" );
		static_assert( db == 0, "bias currently not taken into account" );
		const auto mask = it::srli( it::setone(), sW*8-(de+dm) );
		auto b = it::srli( a, sm-dm );
		auto c = it::bitwise_and( b, mask );
		return c;
	    }
	} else {
	    // Additionally needs a sign bit
	    auto b = it::srli( a, sm-dm ); // position mantissa
	    auto s = it::srli( a, (sW*8-1)-(de+dm) ); // position sign bit

	    if constexpr ( db != 0 ) {
		// TODO: do we need to zero the sign bit? It should
		//       be zero already by assumption of ds == 0
		const auto bias
		    = it::set1( typename it::member_type( db ) << dm );
		b = it::sub( b, bias );

		// recognise zero, distorted by bias manipulation
		if constexpr ( dz ) {
		    // zero is correctly treated if db == 0 or no bias
		    // manipulation is applied at all
		    auto zero = it::setzero();
		    auto is_zero = it::cmpeq( a, zero,
					      typename it::mt_preferred() );
		    b = it::blend( is_zero, b, zero );
		}
	    }

	    // Mask has 1-bits in sign bit position and all higher
	    // positions (cleared in s by srli)
	    const auto mask = it::slli( it::setone(), de+dm );
	    auto d = it::bitblend( mask, b, s );
	    return d;
	}
    } else {
	static_assert( db == 0, "bias currently not taken into account" );

	// Assumes exp < 0, or subtraction does not carry into sign bit
	const auto eoff =
	    it::srli( it::slli( it::setone(), sW*8-(se-de) ), 1+ss );
	const auto smsk = it::srli( it::setone(), 1 );

	auto b = it::sub( a, eoff );
	auto s = it::bitwise_andnot( smsk, a );
	auto c = it::slli( b, (se-de) + (ss-ds) );
	auto d = it::bitwise_or( c, s );
	auto e = it::srli( d, (sW-dW)*8 );

/*
	for( unsigned short l=0; l < VL; ++l ) {
	    auto v = it::lane( e, l );
	    auto m = ditype(1) << (dW*8-2);
	    assert( (v & m) == 0 && "exponent out of range" );
	}
*/
    
	return e;
    }
}

template<typename SrcTy, typename DstTy, unsigned short VL>
std::enable_if_t<(sizeof(DstTy)<sizeof(SrcTy)),
		 typename vector_type_traits_vl<SrcTy,VL>::itype>
cvt_float_narrowing( typename vector_type_traits_vl<SrcTy,VL>::itype a ) {
    return cvt_float_narrowing_static<SrcTy,DstTy,VL>( a );
}

template<typename SrcTy, typename DstTy, unsigned short VL>
std::enable_if_t<(sizeof(DstTy)>=sizeof(SrcTy)),
		     typename vector_type_traits_vl<DstTy,VL>::type>
cvt_float_width( typename vector_type_traits_vl<SrcTy,VL>::type a ) {
    using src_traits = vector_type_traits_vl<SrcTy,VL>;
    using dst_traits = vector_type_traits_vl<DstTy,VL>;

    using sitype = typename src_traits::int_type;
    using ditype = typename dst_traits::int_type;

    auto aa = src_traits::castint( a );
    
    // Widening conversion
    auto b = conversion_traits<sitype,ditype,VL>::convert( aa );

    // Modify bit pattern
    auto c = cvt_float_widen<SrcTy,DstTy,VL>( b );

    // auto x = cvt_float_narrowing<DstTy,SrcTy,VL>( c );
    // assert( dst_traits::int_traits::cmpeq( x, b, target::mt_bool() ) );

    auto f = dst_traits::int_traits::castfp( c );

    return f;
}

template<typename SrcTy, typename DstTy, unsigned short VL>
std::enable_if_t<(sizeof(DstTy)<sizeof(SrcTy)),
		     typename vector_type_traits_vl<DstTy,VL>::type>
cvt_float_width( typename vector_type_traits_vl<SrcTy,VL>::type a ) {
    using src_traits = vector_type_traits_vl<SrcTy,VL>;
    using dst_traits = vector_type_traits_vl<DstTy,VL>;

    using sitype = typename src_traits::int_type;
    using ditype = typename dst_traits::int_type;

    auto aa = src_traits::castint( a );

    // Float conversion process
    auto b = cvt_float_narrowing<SrcTy,DstTy,VL>( aa );

    // Narrowing conversion
    // auto
    typename dst_traits::itype c = conversion_traits<sitype,ditype,VL>::convert( b );

    auto f = dst_traits::int_traits::castfp( c );

    return f;
}

template<typename SrcTy, typename DstTy, unsigned short VL>
auto
cvt_float_from_spaced_int( typename vector_type_traits_vl<DstTy,VL>::itype a ) {
    using src_traits = vector_type_traits_vl<SrcTy,VL>;
    using dst_traits = vector_type_traits_vl<DstTy,VL>;

    // Modify bit pattern
    auto b = cvt_float_widen<SrcTy,DstTy,VL>( a );

    return dst_traits::int_traits::castfp( b );
}

template<typename SrcTy, typename DstTy, unsigned short VL>
auto
cvt_uint_from_spaced_int( typename vector_type_traits_vl<DstTy,VL>::itype a ) {
    using src_traits = vector_type_traits_vl<SrcTy,VL>;
    using dst_traits = vector_type_traits_vl<DstTy,VL>;

    // No sign extension - zero top part of each lane
    auto mask = dst_traits::slli( dst_traits::setone(), 8*src_traits::W );
    auto b = dst_traits::bitwise_andnot( mask, a );

    return b;
}

/***********************************************************************
 * Lane width and type conversions: implementations
 ***********************************************************************/
namespace conversion {

template<typename T, typename V, unsigned short VL, typename Enable = void>
struct int_conversion_traits {
    // If narrowest type is recursively defined, then so is widest.
    using src_traits = vector_type_traits_vl<T, VL>;
    using dst_traits = vector_type_traits_vl<V, VL>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	// 1x8 is the smallest vector unit we consider; breaking it up
	// recursively is not possible as 1x4 is not supported. Need bespoke
	// code for this.
#if __AVX512F__
	if constexpr ( src_traits::W == 1 && VL == 16 ) {
	    assert( 0 && "NYI" );
	} else if constexpr( dst_traits::W == 1 && VL == 16 ) {
	    assert( 0 && "NYI" );
	} else
#endif
#if __AVX2__
	if constexpr ( src_traits::W == 1 && VL == 8 ) {
	    assert( 0 && "NYI" );
	} else if constexpr( dst_traits::W == 1 && VL == 8 ) {
	    assert( 0 && "NYI" );
	} else
#endif
	{
	    constexpr unsigned short VL1
		= std::conditional_t<(sizeof(T) > sizeof(V)),src_traits,dst_traits>
		::lo_half_traits::vlen;
	    constexpr unsigned short VL2
		= std::conditional_t<(sizeof(T) > sizeof(V)),src_traits,dst_traits>
		::hi_half_traits::vlen;
	    static_assert( VL1 >= VL2, "most data goes in lower part" );
	    using lo_conv_traits = int_conversion_traits<T,V,VL1>;
	    using hi_conv_traits = int_conversion_traits<T,V,VL2>;
	    return dst_traits::set_pair(
		hi_conv_traits::convert( src_traits::upper_half( a ) ),
		lo_conv_traits::convert( src_traits::lower_half( a ) ) );
	}
    }
};

template<typename T, typename V, unsigned short VL, typename Enable = void>
struct fp_conversion_traits {
    // If narrowest type is recursively defined, then so is widest.
    using src_traits = vector_type_traits_vl<T, VL>;
    using dst_traits = vector_type_traits_vl<V, VL>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	// 1x8 is the smallest vector unit we consider; breaking it up
	// recursively is not possible as 1x4 is not supported. Need bespoke
	// code for this.
#if __AVX512F__
	if constexpr ( src_traits::W == 1 && VL == 16 ) {
	    assert( 0 && "NYI" );
	} else if constexpr( dst_traits::W == 1 && VL == 16 ) {
	    assert( 0 && "NYI" );
	} else
#endif
#if __AVX2__
	if constexpr ( src_traits::W == 1 && VL == 8 ) {
	    assert( 0 && "NYI" );
	} else if constexpr( dst_traits::W == 1 && VL == 8 ) {
	    assert( 0 && "NYI" );
	} else
#endif
	if constexpr ( false ) {
	} else {
	    constexpr unsigned short VL1
		= std::conditional_t<(sizeof(T) > sizeof(V)),src_traits,dst_traits>
		::lo_half_traits::vlen;
	    constexpr unsigned short VL2
		= std::conditional_t<(sizeof(T) > sizeof(V)),src_traits,dst_traits>
		::hi_half_traits::vlen;
	    static_assert( VL1 >= VL2, "most data goes in lower part" );
	    using lo_conv_traits = fp_conversion_traits<T,V,VL1>;
	    using hi_conv_traits = fp_conversion_traits<T,V,VL2>;
	    return dst_traits::set_pair(
		hi_conv_traits::convert( src_traits::upper_half( a ) ),
		lo_conv_traits::convert( src_traits::lower_half( a ) ) );
	}
    }
};

template<typename T, typename V, unsigned short VL, typename Enable = void>
struct bf_conversion_traits {
    // If narrowest type is recursively defined, then so is widest.
    using src_traits = vector_type_traits_vl<T, VL>;
    using dst_traits = vector_type_traits_vl<V, VL>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	// 1x8 is the smallest integer vector unit we consider; breaking it up
	// recursively is not possible as 1x4 is not supported. Need bespoke
	// code for this.
	constexpr unsigned short iB
	    = std::conditional_t<is_bitfield_v<T>,dst_traits,src_traits>::W;
#if __AVX512F__
	if constexpr ( iB == 8 && VL == 16 ) {
	    assert( 0 && "NYI" );
	} else
#endif
#if __AVX2__
	if constexpr ( iB == 8 && VL == 8 ) {
	    assert( 0 && "NYI" );
	} else
#endif
	{
	    constexpr unsigned short VL1
		= std::conditional_t<(sizeof(T) > sizeof(V)),src_traits,dst_traits>
		::lo_half_traits::vlen;
	    constexpr unsigned short VL2
		= std::conditional_t<(sizeof(T) > sizeof(V)),src_traits,dst_traits>
		::hi_half_traits::vlen;
	    static_assert( VL1 >= VL2, "most data goes in lower part" );
	    using lo_conv_traits = bf_conversion_traits<T,V,VL1>;
	    using hi_conv_traits = bf_conversion_traits<T,V,VL2>;
	    return dst_traits::set_pair(
		hi_conv_traits::convert( src_traits::upper_half( a ) ),
		lo_conv_traits::convert( src_traits::lower_half( a ) ) );
	}
    }
};

} // namespace conversion

#include "graptor/target/convert/cvt_logical.h"
#include "graptor/target/convert/cvt_sign.h"
#include "graptor/target/convert/cvt_bitfield.h"

#include "graptor/target/convert/cvt_1i_2i.h"
#include "graptor/target/convert/cvt_1i_4i.h"
#include "graptor/target/convert/cvt_1i_8i.h"
#include "graptor/target/convert/cvt_2i_1i.h"
#include "graptor/target/convert/cvt_2i_4i.h"
#include "graptor/target/convert/cvt_2i_8i.h"
#include "graptor/target/convert/cvt_4i_1i.h"
#include "graptor/target/convert/cvt_4i_2i.h"
#include "graptor/target/convert/cvt_4i_8i.h"
#include "graptor/target/convert/cvt_8i_1i.h"
#include "graptor/target/convert/cvt_8i_2i.h"
#include "graptor/target/convert/cvt_8i_4i.h"

#include "graptor/target/convert/cvt_4i_8f.h"
#include "graptor/target/convert/cvt_4i_4f.h"
#include "graptor/target/convert/cvt_8i_8f.h"
#include "graptor/target/convert/cvt_8f_8i.h"
#include "graptor/target/convert/cvt_8f_4i.h"
#include "graptor/target/convert/cvt_8f_4f.h"

namespace conversion {


/***********************************************************************
 * Lane width and type conversions: bitfields
 ***********************************************************************/
template<typename T, typename U, unsigned short VL>
struct int_conversion_traits<T, U, VL,
			 typename std::enable_if_t<
			     !std::is_same<T,U>::value
			     && is_bitfield_v<T>
			     && !is_bitfield_v<U>
			     && VL != 1>> {
    using src_traits = vector_type_traits_vl<T, VL>;
    using dst_traits = vector_type_traits_vl<U, VL>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	if constexpr ( T::bits*VL <= 8*dst_traits::W ) {
	    // Broadcast the bit pattern to all lanes, shift and mask out
	    // relevant bits
/* delivers only the lsb
	    auto b = dst_traits::set1( U( a ) );
	    const auto inc = dst_traits::set1inc0();
	    const auto sh = dst_traits::slli( inc, ilog2(T::bits) );
	    auto c = dst_traits::srlv( b, sh );
	    const auto m = dst_traits::srli(
		dst_traits::setone(), 8*dst_traits::W-T::bits );
	    auto d = dst_traits::bitwise_and( c, m );
	    return d;
*/
#if __AVX2__ && !__AVX512VL__
	    if constexpr ( is_logical_v<U> && sizeof(U) == 8 ) {
		// Work-around for missing AVX2 64-bit srai
		// As values are either all 0 or all 1, replicate
		// successive half-width lanes with the same value
		using tr = vector_type_traits_vl<uint32_t,VL*2>;
		auto b = tr::set1( uint32_t( a ) );
		const auto all = tr::setone();
		const auto one = tr::srli( tr::setone(), 8*tr::W-1 );
		const auto inc = tr::set1inc0();
		const auto sh = tr::bitwise_andnot( one, inc );
		const auto w = tr::set1( uint32_t( 8*tr::W-T::bits ) );
		const auto osh = tr::sub( w, sh );
		auto c = tr::sllv( b, osh );
		auto d = tr::srai( c, 8*tr::W-1 );
		return d;
	    }
#endif
	    auto b = dst_traits::set1( U( a ) );
	    const auto inc = dst_traits::set1inc0();
	    const auto sh = dst_traits::slli( inc, ilog2(T::bits) );
	    const auto w = dst_traits::set1( U( 8*dst_traits::W-T::bits ) );
	    const auto osh = dst_traits::sub( w, sh );
	    auto c = dst_traits::sllv( b, osh );
	    typename dst_traits::type d;
	    if constexpr ( is_logical_v<U> )
		d = dst_traits::srai( c, 8*dst_traits::W-T::bits );
	    else
		d = dst_traits::srli( c, 8*dst_traits::W-T::bits );
	    return d;
	}

	if constexpr ( dst_traits::W > 1 && VL > 1 ) {
	    if constexpr ( is_logical_v<U> ) {
		auto b = int_conversion_traits<T,logical<1>,VL>::convert( a );
		return int_conversion_traits<logical<1>,U,VL>::convert( b );
	    } else {
		auto b = int_conversion_traits<T,uint8_t,VL>::convert( a );
		return int_conversion_traits<uint8_t,U,VL>::convert( b );
	    }
	}

	if constexpr ( T::bits == 2 && VL == 8 && dst_traits::W == 1 ) {
	    // Take 16 bits (compacted) and scatter into 8 lanes of 1 byte
	    long long unsigned int mask = is_logical_v<U>
		? 0xc0c0c0c0c0c0c0c0ULL : 0x0303030303030303ULL;
	    auto b = _pdep_u64( (uint64_t)a, mask );
	    return _mm_cvtsi64_m64( b );
	}

	if constexpr ( T::bits == 2 && VL == 16 && dst_traits::W == 1 ) {
	    // Take 16 bits (compacted) and scatter into 8 lanes of 1 byte
	    long long unsigned int mask = is_logical_v<U>
		? 0xc0c0c0c0c0c0c0c0ULL : 0x0303030303030303ULL;
	    auto w0 = _pdep_u64( a, mask );
	    auto w1 = _pdep_u64( a >> 16, mask );
	    return _mm_set_epi64x( w1, w0 );
	}

	if constexpr ( T::bits == 2 && VL == 32 && dst_traits::W == 1 ) {
	    // Take 64 bits (compacted) and scatter into 32 lanes of 1 byte
	    long long unsigned int mask = is_logical_v<U>
		? 0xc0c0c0c0c0c0c0c0ULL : 0x0303030303030303ULL;
	    auto w0 = _pdep_u64( a, mask );
	    auto w1 = _pdep_u64( a >> 16, mask );
	    auto w2 = _pdep_u64( a >> 32, mask );
	    auto w3 = _pdep_u64( a >> 48, mask );
	    return _mm256_set_epi64x( w3, w2, w1, w0 );
	}

	if constexpr ( T::bits == 2 && dst_traits::W == 1
#if __AVX512F__
		       && VL >= 64
#elif __AVX2__
		       && VL > 32
#elif __SSE4_2__
		       && VL > 16
#endif
	    ) {
	    auto lo = int_conversion_traits<T,U,VL/2>::convert(
		src_traits::lower_half( a ) );
	    auto hi = int_conversion_traits<T,U,VL/2>::convert(
		src_traits::upper_half( a ) );
	    return dst_traits::set_pair( hi, lo );
	}
	
	assert( 0 && "NYI" );
    }
};

template<typename T, typename U, unsigned short VL>
struct int_conversion_traits<T, U, VL,
			 typename std::enable_if_t<
			     !std::is_same<T,U>::value
			     && !is_bitfield_v<T>
			     && is_bitfield_v<U>
			     && VL != 1>> {
    using src_traits = vector_type_traits_vl<T, VL>;
    using dst_traits = vector_type_traits_vl<U, VL>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	if constexpr ( src_traits::W > 1 ) {
	    if constexpr ( is_logical_v<T> || is_logical_v<U> ) {
		auto b = int_conversion_traits<T,logical<1>,VL>::convert( a );
		return int_conversion_traits<logical<1>,U,VL>::convert( b );
	    } else {
		auto b = int_conversion_traits<T,uint8_t,VL>::convert( a );
		return int_conversion_traits<uint8_t,U,VL>::convert( b );
	    }
	}
	
	if constexpr ( U::bits == 2 ) {
	    // Pick top bits
	    long long unsigned int mask = is_logical_v<T>
		? 0xc0c0c0c0c0c0c0c0ULL : 0x0303030303030303ULL;

	    if constexpr ( VL == 8 && src_traits::W == 1 ) {
		uint64_t m = mask;
		uint64_t b = _mm_cvtm64_si64( a );
		uint64_t c = _pext_u64( b, m );
		return static_cast<typename dst_traits::type>( c );
	    }
	    if constexpr ( VL == 16 && src_traits::W == 1 ) {
		uint64_t w0 = _pext_u64( _mm_extract_epi64( a, 0 ), mask );
		uint64_t w1 = _pext_u64( _mm_extract_epi64( a, 1 ), mask );
		auto c = ( w1 << 16 ) | w0;
		return static_cast<typename dst_traits::type>( c );
	    }
	    if constexpr ( VL == 32 && src_traits::W == 1 ) {
		uint64_t w0 = _pext_u64( _mm256_extract_epi64( a, 0 ), mask );
		uint64_t w1 = _pext_u64( _mm256_extract_epi64( a, 1 ), mask );
		uint64_t w2 = _pext_u64( _mm256_extract_epi64( a, 2 ), mask );
		uint64_t w3 = _pext_u64( _mm256_extract_epi64( a, 3 ), mask );
		auto c = ( w3 << 48 ) | ( w2 << 32 ) | ( w1 << 16 ) | w0;
		return static_cast<typename dst_traits::type>( c );
	    }
	    if constexpr ( VL == 64 && src_traits::W == 1 ) {
// #if __AVX512F__
		// assert( 0 && "NYI" );
// #else
		auto lo = int_conversion_traits<T,U,VL/2>::convert(
		    src_traits::lower_half( a ) );
		auto hi = int_conversion_traits<T,U,VL/2>::convert(
		    src_traits::upper_half( a ) );
		return dst_traits::set_pair( hi, lo );
// #endif
	    }
	    if constexpr ( VL >= 128 && src_traits::W == 1 ) {
		auto lo = int_conversion_traits<T,U,VL/2>::convert(
		    src_traits::lower_half( a ) );
		auto hi = int_conversion_traits<T,U,VL/2>::convert(
		    src_traits::upper_half( a ) );
		return dst_traits::set_pair( hi, lo );
	    }
	}
	
	assert( 0 && "NYI" );
    }
};

/***********************************************************************
 * Lane width and type conversions: vectors
 ***********************************************************************/

#if __AVX2__
template<>
struct fp_conversion_traits<float, double, 4> {
    using src_traits = vector_type_traits_vl<float, 4>;
    using dst_traits = vector_type_traits_vl<double, 4>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	// Converts 4x float (SSE4) to 4x double (AVX). Assumes AVX.
	return _mm256_cvtps_pd( a );
    }
};
#elif __SSE4_2__
template<>
struct fp_conversion_traits<float, double, 4> {
    using src_traits = vector_type_traits_vl<float, 4>;
    using dst_traits = vector_type_traits_vl<double, 4>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	// Converts 4x float (SSE4) to 4x double. Assumes no AVX.
	__m128d lo = _mm_cvtps_pd( a );
	__m128d hi = _mm_cvtps_pd( _mm_shuffle_ps( a, a, 0b11101110 ) );
	return dst_traits::set( hi, lo );
    }
};
#endif // __AVX2__

#if __AVX2__
template<>
struct fp_conversion_traits<long unsigned int, float, 4> {
    using src_traits = vector_type_traits_vl<long unsigned int, 4>;
    using dst_traits = vector_type_traits_vl<float, 4>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	__m256i c = _mm256_shuffle_epi32( a, 0b10001000 );
	__m256i cc = _mm256_permute4x64_epi64( c, 0b1000 );
	__m128i s = _mm256_castsi256_si128( cc );
	return _mm_cvtepi32_ps( s );
    }
};
#elif __SSE4_2__
template<>
struct fp_conversion_traits<long unsigned int, float, 4> {
    using src_traits = vector_type_traits_vl<long unsigned int, 4>;
    using dst_traits = vector_type_traits_vl<float, 4>;
    using cnv_traits = fp_conversion_traits<long unsigned int, float, 2>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	__m128i lo = src_traits::lower_half( a );
	__m128i hi = src_traits::upper_half( a );
	__m128 flo = _mm_cvtpd_ps( uint64_to_double( lo ) );
	__m128 fhi = _mm_cvtpd_ps( uint64_to_double( hi ) );
	return _mm_shuffle_ps( flo, fhi, 0b01000100 );
    }

    // https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx
    static __m128d uint64_to_double(__m128i x) {
	x = _mm_or_si128(x, _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)));
	return _mm_sub_pd(_mm_castsi128_pd(x), _mm_set1_pd(0x0010000000000000));
    }
};
#else
/* -- handle using recursion
template<>
struct fp_conversion_traits<long unsigned int, float, 4> {
    using src_traits = vector_type_traits_vl<long unsigned int, 4>;
    using dst_traits = vector_type_traits_vl<float, 4>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	using luint = long unsigned int;
	using half_traits = vector_type_traits_vl<long unsigned int, 2>;
	luint m0 = half_traits::lane( src_traits::lower_half( a ), 0 );
	luint m1 = half_traits::lane( src_traits::lower_half( a ), 1 );
	luint m2 = half_traits::lane( src_traits::upper_half( a ), 0 );
	luint m3 = half_traits::lane( src_traits::upper_half( a ), 1 );
	return dst_traits::set( (float)m3, (float)m2, (float)m1, (float)m0 );
    }
};
*/
#endif

#ifdef __AVX512F__
template<>
struct fp_conversion_traits<long unsigned int, float, 8> {
    using src_traits = vector_type_traits_vl<long unsigned int, 8>;
    using dst_traits = vector_type_traits_vl<float, 8>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	__m512 f = _mm512_cvt_roundepu32_ps( a, _MM_FROUND_CUR_DIRECTION );
	__m512 c = _mm512_shuffle_ps( f, f, _MM_SHUFFLE( 2, 0, 2, 0 ) );
	return _mm512_castps512_ps256( c );
    }
};

template<>
struct fp_conversion_traits<long unsigned int, float, 16> {
    using src_traits = vector_type_traits_vl<long unsigned int, 16>;
    using dst_traits = vector_type_traits_vl<float, 16>;
    using idx_traits = vector_type_traits_vl<unsigned int, 16>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	// Note: this first shrinks the width because AVX512DQ instructions
	// are required to convert epu64 to ps
	__m512i ca = _mm512_castsi256_si512( _mm512_cvtepi64_epi32( a.a ) );
	__m512i cb = _mm512_castsi256_si512( _mm512_cvtepi64_epi32( a.b ) );
	__m512i c = _mm512_shuffle_i32x4( ca, cb, _MM_SHUFFLE( 1, 0, 1, 0 ) );
	return _mm512_cvt_roundepu32_ps( c, _MM_FROUND_CUR_DIRECTION );
    }
};

template<>
struct fp_conversion_traits<float, double, 8> {
    using src_traits = vector_type_traits_vl<float, 8>;
    using dst_traits = vector_type_traits_vl<double, 8>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	return _mm512_cvtps_pd( a );
    }
};

template<>
struct fp_conversion_traits<float, double, 16> {
    using src_traits = vector_type_traits_vl<float, 16>;
    using dst_traits = vector_type_traits_vl<double, 16>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	auto lo = src_traits::lower_half( a );
	auto hi = src_traits::upper_half( a );
	return dst_traits::type{ _mm512_cvtps_pd( lo ), _mm512_cvtps_pd( hi ) };
    }
};

template<>
struct fp_conversion_traits<unsigned long, unsigned long long, 8> {
    using src_traits = vector_type_traits_vl<unsigned, 8>;
    using dst_traits = vector_type_traits_vl<unsigned long, 8>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	return _mm512_cvtepi32_epi64( a );
    }
};

template<>
struct fp_conversion_traits<bool, logical<8>, 16> {
    using src_traits = vector_type_traits_vl<bool, 16>;
    using dst_traits = vector_type_traits_vl<logical<8>, 16>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	return typename dst_traits::type{
	    _mm512_cvtepi8_epi64( a ),
	    _mm512_cvtepi8_epi64( _mm_srli_si128( a, 64 ) ) };
    }
};

#elif __AVX2__ // __AVX512F__ above

template<>
struct fp_conversion_traits<float, double, 8> {
    using src_traits = vector_type_traits_vl<float, 8>;
    using dst_traits = vector_type_traits_vl<double, 8>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	using half_traits = fp_conversion_traits<float, double, 4>;
	__m256d lo = half_traits::convert( src_traits::lower_half( a ) );
	__m256d hi = half_traits::convert( src_traits::upper_half( a ) );
	return dst_traits::type{ lo, hi };
    }
};

template<>
struct fp_conversion_traits<long unsigned int, float, 8> {
    // No AVX512 available
    using src_traits = vector_type_traits_vl<long unsigned int, 8>;
    using dst_traits = target::avx2_4fx8<float>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	// No suitable vector operations available pre-AVX512.
	// Take apart vector and rebuild
	using half_traits = typename src_traits::lo_half_traits;
	typename half_traits::type lo = src_traits::lower_half( a );
	typename half_traits::type hi = src_traits::upper_half( a );
	return dst_traits::set(
	    (float)half_traits::lane3( hi ),
	    (float)half_traits::lane2( hi ),
	    (float)half_traits::lane1( hi ),
	    (float)half_traits::lane0( hi ),
	    (float)half_traits::lane3( lo ),
	    (float)half_traits::lane2( lo ),
	    (float)half_traits::lane1( lo ),
	    (float)half_traits::lane0( lo ) );
    }
};
#endif // __AVX512F__


/*
#if __SSE4_2__
template<>
struct conversion_traits<float, double, 2> {
    using src_traits = vector_type_traits_vl<float, 2>;
    using dst_traits = vector_type_traits_vl<double, 2>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	// Converts 2x float (MMX) to 2x double (SSE4). Assumes SS4.
	return _mm256_cvtps_pd( a );
    }
};
#endif
*/

} // namespace conversion

template<typename T, typename U>
struct conversion_traits<T, U, 1> {
    using src_traits = vector_type_traits_vl<T, 1>;
    using dst_traits = vector_type_traits_vl<U, 1>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	// Vector conversions do sign extensions, so should scalar.
	// return static_cast<typename dst_traits::type>( a );
	if constexpr ( std::is_same_v<T,U> ) {
	    return a;
	} else if constexpr ( std::is_same_v<T,bool> && is_logical_v<U> ) {
	    return U::template get_val<bool>( a );
	} else if constexpr ( std::is_same_v<U,bool> && is_logical_v<T> ) {
	    return !!a;
	} else {
	    return static_cast<typename dst_traits::type>( a );
	}
    }
};

// Minimum vector length and vector size requirements
template<typename T, typename U, unsigned short VL>
struct conversion_traits<T,U,VL,
			 std::enable_if_t<(VL>=4)
			 && sizeof(T)*VL >= 4
			 && sizeof(U)*VL >= 4
			 && !is_bitfield_v<T>
			 && !is_bitfield_v<U>>> {
    using src_traits = vector_type_traits_vl<T, VL>;
    using dst_traits = vector_type_traits_vl<U, VL>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	if constexpr ( std::is_same_v<T,U> )
	    return a;
	else if constexpr ( std::is_floating_point_v<T> || std::is_floating_point_v<U> )
	    return conversion::fp_conversion_traits<T,U,VL>::convert( a );
	else
	    return conversion::int_conversion_traits<T,U,VL>::convert( a );
    }
};

template<typename T, typename U, unsigned short VL>
struct conversion_traits<T,U,VL,
			 std::enable_if_t<
			     VL != 1
			     && ( is_bitfield_v<T> || is_bitfield_v<U> )>> {
    using src_traits = vector_type_traits_vl<T, VL>;
    using dst_traits = vector_type_traits_vl<U, VL>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	if constexpr ( is_bitfield_v<T> != is_bitfield_v<U> )
	    return conversion::bf_conversion_traits<T,U,VL>
		::convert( a );
	else if constexpr ( std::is_same_v<T,U> )
	    return a;

	// bitfield-bitfield conversion not currently supported unless no-op
	assert( 0 && "NYI" );
    }
};

#endif // GRAPTOR_TARGET_CONVERSION_H
