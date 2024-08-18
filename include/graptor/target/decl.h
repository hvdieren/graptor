// -*- c++ -*-
#ifndef GRAPTOR_TARGET_DECL_H
#define GRAPTOR_TARGET_DECL_H

#include <x86intrin.h>
#include <immintrin.h>

/***********************************************************************
 * Pairs of vectors to build longer ones.
 ***********************************************************************/
template<typename U, typename V>
struct vpair {
    U a;
    V b;

    vpair( const U & a_, const V & b_ ) : a( a_ ), b( b_ ) { }
    vpair( U && a_, V && b_ ) :
	a( std::forward<U>( a_ ) ), b( std::forward<V>( b_ ) ) { }
    vpair( const vpair & p ) : a( p.a ), b( p.b ) { }
    vpair( vpair && p ) : a( std::move( p.a ) ), b( std::move( p.b ) ) { }
    vpair() { }
    
    const vpair & operator = ( const vpair & p ) {
	a = p.a;
	b = p.b;
	return *this;
    }
    const vpair & operator = ( vpair && p ) {
	a = std::move( p.a );
	b = std::move( p.b );
	return *this;
    }
/*
    template<typename S, typename T>
    operator vpair<S,T> () const {
	vpair<S,T> v;
	v.a = (S)a;
	v.b = (T)b;
	return v;
    }
*/
};

template<typename T>
struct is_vpair : public std::false_type { }; 

template<typename U, typename V>
struct is_vpair<vpair<U,V>> : public std::true_type { };

template<typename T>
static constexpr bool is_vpair_v = is_vpair<T>::value;

/***********************************************************************
 * Hardware registers
 * This class was introduced to avoid dependencies between target
 * implementation traits and among themselves and vector_type_traits.
 * Using this class, methods can accept a hw_reg of a specific length.
 *
 * A pending issue is disambiguation between hw_reg of the same total
 * length but different makeup. Normally, this should not be necessary,
 * except in the case of storing 8-byte vectors in a 16-byte SSE vector,
 * where an index may be supplied at different width, e.g., gather of
 * W=2, VL=4 requiring indices with W=4, where we can't know if
 * the index is W=2 or W=4.
 ***********************************************************************/
template<unsigned short W, unsigned short VL>
struct hw_reg { // recursive case; base cases follow through overrides
    using half = hw_reg<W,VL/2>;
    using itype = vpair<typename half::itype,typename half::itype>;
    using dtype = vpair<typename half::dtype,typename half::dtype>;
    using ftype = vpair<typename half::ftype,typename half::ftype>;
};

#if __SSE4_2__
struct sse42_hw_reg {
    using itype = __m128i;
    using dtype = __m128d;
    using ftype = __m128;
};

template<>
struct hw_reg<8,2> : public sse42_hw_reg { };
template<>
struct hw_reg<4,4> : public sse42_hw_reg { };
template<>
struct hw_reg<2,8> : public sse42_hw_reg { };
template<>
struct hw_reg<1,16> : public sse42_hw_reg { };
#endif

#if __AVX2__
struct avx2_hw_reg {
    using itype = __m256i;
    using dtype = __m256d;
    using ftype = __m256;
};

template<>
struct hw_reg<8,4> : public avx2_hw_reg { };
template<>
struct hw_reg<4,8> : public avx2_hw_reg { };
template<>
struct hw_reg<2,16> : public avx2_hw_reg { };
template<>
struct hw_reg<1,32> : public avx2_hw_reg { };
#endif

#if __AVX512F__
struct avx512_hw_reg {
    using itype = __m512i;
    using dtype = __m512d;
    using ftype = __m512;
};

template<>
struct hw_reg<8,8> : public avx512_hw_reg { };
template<>
struct hw_reg<4,16> : public avx512_hw_reg { };
template<>
struct hw_reg<2,32> : public avx512_hw_reg { };
template<>
struct hw_reg<1,64> : public avx512_hw_reg { };
#endif


namespace target {

/***********************************************************************
 * Tag types to overload functions that return a mask of some sort.
 ***********************************************************************/
struct mt_bool { };
struct mt_mask { };
struct mt_vmask { };

/***********************************************************************
 * Determining the appropriate mask type as a function of vector length
 ***********************************************************************/
// TODO: Used only in combine_mask utility below; try to replace with
// typename bitmask_traits<VL>::type. Current version does not support
// recursively composed bitmasks (non-power-of-2 vector lengths).
template<unsigned short VL>
struct select_mask_type {
    using type = longint<VL/8>;
};

template<>
struct select_mask_type<1> {
    using type = unsigned char;
};

template<>
struct select_mask_type<2> {
    using type = unsigned char;
};

template<>
struct select_mask_type<4> {
    using type = unsigned char;
};

template<>
struct select_mask_type<8> {
    using type = __mmask8;
};

template<>
struct select_mask_type<16> {
    using type = __mmask16;
};

template<>
struct select_mask_type<32> {
    using type = __mmask32;
};

template<>
struct select_mask_type<64> {
    using type = __mmask64;
};

template<>
struct select_mask_type<128> {
    using type = __m128i;
};

template<>
struct select_mask_type<256> {
    using type = __m256i;
};

template<unsigned short VL>
using mask_type_t = typename select_mask_type<VL>::type;
}

/***********************************************************************
 * Vector traits
 ***********************************************************************/
template<typename T, unsigned short nbytes, typename = void>
struct vector_type_traits;

template<typename T, typename V>
struct vector_type_traits_of;

template<typename VT, unsigned short VL>
struct vector_type_traits_with;

template<typename T, unsigned short VL, typename Enable = void>
struct vector_type_traits_vl;

#include "graptor/longint.h"
#include "graptor/target/bitmask.h"

namespace target {

/***********************************************************************
 * Mask utility
 ***********************************************************************/
template<unsigned short VL1, unsigned short VL2>
inline mask_type_t<VL1+VL2> combine_mask(
    mask_type_t<VL1> lo, mask_type_t<VL2> hi ) {
    if constexpr ( is_longint_v<mask_type_t<VL1+VL2>> ) {
	using traits =
	    vector_type_traits<uint64_t,mask_type_t<VL1+VL2>::W>;
	if constexpr ( VL1 == VL2 && VL1 <= 64 ) {
	    mask_type_t<VL1+VL2> wc( traits::set_pair( hi, lo ) );
	    return wc;
	} else if constexpr ( VL1 == VL2 ) {
	    if constexpr ( !is_longint_v<mask_type_t<VL1>> ) {
		return mask_type_t<VL1+VL2>( traits::set_pair( hi, lo ) );
	    } else {
		return mask_type_t<VL1+VL2>(
		    traits::set_pair( hi.get(), lo.get() ) );
	    }
	} else {
	    mask_type_t<VL1+VL2> wlo( lo );
	    mask_type_t<VL1+VL2> whi( hi );
	    mask_type_t<VL1+VL2> wc(
		traits::bitwise_or(
		    traits::template bslli<VL1/8>( whi.get() ), wlo.get() ) );
	    return wc;
	}
    } else {
	// mask_type_t<VL1+VL2> wlo( lo );
	// mask_type_t<VL1+VL2> whi( hi );
	// return (whi << VL1) | wlo;
	using traits = mask_type_traits<VL1+VL2>;
	return traits::set_pair( hi, lo );
    }
}


/***********************************************************************
 * Entry point to type traits for mask operations
 ***********************************************************************/

/***********************************************************************
 * Entry point to type traits for vector operations
 ***********************************************************************/
#if __AVX2__
template<typename T = uint32_t>
struct avx2_4x8;

template<typename T = uint64_t>
struct avx2_8x4;
#endif // __AVX2__


} // namespace target

#endif // GRAPTOR_TARGET_DECL_H
