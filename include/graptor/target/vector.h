// -*- c++ -*-

#ifndef GRAPTOR_TARGET_VECTOR_H
#define GRAPTOR_TARGET_VECTOR_H

#include <type_traits>
#include <x86intrin.h>
#include <immintrin.h>

#include "graptor/utils.h"
#include "graptor/itraits.h"
#include "graptor/customfp.h"
#include "graptor/vcustomfp.h"
#include "graptor/bitfield.h"
#include "graptor/target/decl.h"

/***********************************************************************
 * Missing intrinsics on gcc 4.9.2. Copied from clang source base
 * These intrinsics are available on gcc 10. May need to interpolate
 * between versions.
 ***********************************************************************/
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 10 
#if __AVX512F__
static __inline __m512i
_mm512_zextsi128_si512(__m128i __a)
{
    __m256i __b = _mm256_castsi128_si256( __a );
    return _mm512_inserti64x4( _mm512_setzero_si512(), __b, 0x0 );
}

static __inline __m512i
_mm512_zextsi256_si512(__m256i __a)
{
    return _mm512_inserti64x4( _mm512_setzero_si512(), __a, 0x0 );
}
#endif

#if __AVX__
static __inline __m256i
_mm256_zextsi128_si256(__m128i __a)
{
    __m256i __b = _mm256_castsi128_si256( __a );
    return _mm256_permute2x128_si256( __b, __b, 0x80 );
}
#endif
#endif // __GNUC__ < 10

/***********************************************************************
 * Useful constants
 ***********************************************************************/
alignas(64) extern const uint8_t increasing_sequence_epi8[64];
alignas(64) extern const uint16_t increasing_sequence_epi16[16];
alignas(64) extern const uint32_t increasing_sequence_epi32[16];
alignas(64) extern const uint64_t increasing_sequence_epi64[16];
alignas(64) extern const uint32_t movemask_lut_epi32[16*4];

/***********************************************************************
 * Vector traits selection (int)
 ***********************************************************************/
#ifndef GRAPTOR_USE_MMX
#define GRAPTOR_USE_MMX 0
#endif

//#include "graptor/target/native_1x4.h"
#include "graptor/target/scalar_int.h"
#include "graptor/target/scalar_bool.h"

#include "graptor/target/mmx_1x8.h"
#include "graptor/target/mmx_2x4.h"
#include "graptor/target/mmx_4x2.h"

#include "graptor/target/sse42_1x16.h"
#include "graptor/target/sse42_2x8.h"
#include "graptor/target/sse42_4x4.h"
#include "graptor/target/sse42_8x2.h"
#include "graptor/target/sse42_1x8.h"
#include "graptor/target/sse42_1x4.h"

#include "graptor/target/avx2_1x32.h"
#include "graptor/target/avx2_2x16.h"
#include "graptor/target/avx2_4x8.h"
#include "graptor/target/avx2_8x4.h"

#include "graptor/target/avx2_16x2.h"

#include "graptor/target/avx512_1x64.h"
#include "graptor/target/avx512_2x32.h"
#include "graptor/target/avx512_4x16.h"
#include "graptor/target/avx512_8x8.h"

#include "graptor/target/bitfield.h"

#include "graptor/target/vt_recursive.h"

// #include "graptor/target/pseudo_1x.h"

namespace target {

/*! Selection of vector traits implementation for integral types and logical
 */
template<typename T, unsigned short type_size, unsigned short nbytes>
struct vint_traits_select {
    static_assert( nbytes > type_size, "recursive case" );
    static_assert( sizeof(T) == type_size, "assumption" );
    static constexpr unsigned short lo_size = next_ipow2( nbytes / 2 );
    using type = vt_recursive<
	T,type_size,nbytes,
	typename vint_traits_select<T,type_size,lo_size>::type,
	typename vint_traits_select<T,type_size,nbytes-lo_size>::type>;
};

template<typename T, unsigned short type_size>
struct vint_traits_select<T,type_size,type_size> {
    static_assert( sizeof(T) == type_size, "assumption" );
    using type = target::scalar_int<T>;
};

template<>
struct vint_traits_select<bool,sizeof(bool),sizeof(bool)> {
    using type = target::scalar_bool;
};

#if __MMX__ && __SSE4_2__

#if GRAPTOR_USE_MMX
template<typename T>
struct vint_traits_select<T,1,8> {
    static_assert( sizeof(T) == 1, "assumption" );
    using type = target::mmx_1x8<T>;
};

template<typename T>
struct vint_traits_select<T,2,8> {
    static_assert( sizeof(T) == 2, "assumption" );
    using type = target::mmx_2x4<T>;
};
#else
template<typename T>
struct vint_traits_select<T,1,8> {
    static_assert( sizeof(T) == 1, "assumption" );
    using type = target::sse42_1x8<T>;
};

template<typename T>
struct vint_traits_select<T,2,8> {
    static_assert( sizeof(T) == 2, "assumption" );
    using type = target::sse42_2x4<T>;
};
#endif // GRAPTOR_USE_MMX

#endif // __MMX__ && __SSE4_2__

#if __SSE4_2__
template<typename T>
struct vint_traits_select<T,1,4> {
    static_assert( sizeof(T) == 1, "assumption" );
    using type = target::sse42_1x4<T>;
};
#endif // __SSE4_2__

#if __MMX__
template<typename T>
struct vint_traits_select<T,4,8> {
    static_assert( sizeof(T) == 4, "assumption" );
    using type = target::mmx_4x2<T>;
};
#endif // __MMX__

#if __SSE4_2__
template<typename T>
struct vint_traits_select<T,1,16> {
    static_assert( sizeof(T) == 1, "assumption" );
    using type = target::sse42_1x16<T>;
};

template<typename T>
struct vint_traits_select<T,2,16> {
    static_assert( sizeof(T) == 2, "assumption" );
    using type = target::sse42_2x8<T>;
};

template<typename T>
struct vint_traits_select<T,4,16> {
    static_assert( sizeof(T) == 4, "assumption" );
    using type = target::sse42_4x4<T>;
};

template<typename T>
struct vint_traits_select<T,8,16> {
    static_assert( sizeof(T) == 8, "assumption" );
    using type = target::sse42_8x2<T>;
};
#endif // __SSE4_2__

#if __AVX2__
template<typename T>
struct vint_traits_select<T,1,32> {
    static_assert( sizeof(T) == 1, "assumption" );
    using type = target::avx2_1x32<T>;
};

template<typename T>
struct vint_traits_select<T,2,32> {
    static_assert( sizeof(T) == 2, "assumption" );
    using type = target::avx2_2x16<T>;
};

template<typename T>
struct vint_traits_select<T,4,32> {
    static_assert( sizeof(T) == 4, "assumption" );
    using type = target::avx2_4x8<T>;
};

template<typename T>
struct vint_traits_select<T,8,32> {
    static_assert( sizeof(T) == 8, "assumption" );
    using type = target::avx2_8x4<T>;
};
#endif // __AVX2__

#if __AVX512F__
template<typename T>
struct vint_traits_select<T,1,64> {
    static_assert( sizeof(T) == 1, "assumption" );
    using type = target::avx512_1x64<T>;
};

template<typename T>
struct vint_traits_select<T,2,64> {
    static_assert( sizeof(T) == 2, "assumption" );
    using type = target::avx512_2x32<T>;
};

template<typename T>
struct vint_traits_select<T,4,64> {
    static_assert( sizeof(T) == 4, "assumption" );
    using type = target::avx512_4x16<T>;
};

template<typename T>
struct vint_traits_select<T,8,64> {
    static_assert( sizeof(T) == 8, "assumption" );
    using type = target::avx512_8x8<T>;
};
#endif // __AVX512F__

} // namespace target

/***********************************************************************
 * Vector traits selection (fp)
 ***********************************************************************/
#include "graptor/target/scalar_fp.h"
//#include "graptor/target/mmx_2fx4.h"
#include "graptor/target/sse42_4fx4.h"
#include "graptor/target/sse42_8fx2.h"
#include "graptor/target/avx2_4fx8.h"
#include "graptor/target/avx2_8fx4.h"
#include "graptor/target/avx512_4fx16.h"
#include "graptor/target/avx512_8fx8.h"
#ifndef __clang__
#include "graptor/target/avx512fp16_2fx8.h"
#endif

namespace target {

template<typename T, unsigned short nbytes, typename = void>
struct vfp_traits_select {
    static_assert( nbytes > sizeof(T), "recursive case" );
    using type = vt_recursive<
	T,sizeof(T),nbytes,
	typename vfp_traits_select<T,next_ipow2(nbytes/2)>::type,
	typename vfp_traits_select<T,nbytes-next_ipow2(nbytes/2)>::type>;
};

template<typename T, unsigned short nbytes>
struct vfp_traits_select<
    T,nbytes,std::enable_if_t<sizeof(T) == nbytes>> {
    using type = target::scalar_fp<T>;
};

#if __SSE4_2__
template<typename T, unsigned short nbytes>
struct vfp_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 4 && nbytes == 16)
			      && !is_customfp_v<T> && !is_vcustomfp_v<T>>> {
    using type = sse42_4fx4<T>;
};

template<typename T, unsigned short nbytes>
struct vfp_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 8 && nbytes == 16)
			      && !is_customfp_v<T> && !is_vcustomfp_v<T>>> {
    using type = sse42_8fx2<T>;
};
#endif

#if __AVX2__
template<typename T, unsigned short nbytes>
struct vfp_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 4 && nbytes == 32)
			      && !is_customfp_v<T> && !is_vcustomfp_v<T>>> {
    using type = avx2_4fx8<T>;
};

template<typename T, unsigned short nbytes>
struct vfp_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 8 && nbytes == 32)
			      && !is_customfp_v<T> && !is_vcustomfp_v<T>>> {
    using type = avx2_8fx4<T>;
};
#endif // __AVX2__

#if __AVX512F__
#if defined( __AVX512FP16__ ) && !defined( __clang__ )
template<typename T, unsigned short nbytes>
struct vfp_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 2 && nbytes == 16)
			      && !is_customfp_v<T> && !is_vcustomfp_v<T>>> {
    using type = avx512fp16_2fx8<T>;
};
#endif
    
template<typename T, unsigned short nbytes>
struct vfp_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 4 && nbytes == 64)
			      && !is_customfp_v<T> && !is_vcustomfp_v<T>>> {
    using type = avx512_4fx16<T>;
};

template<typename T, unsigned short nbytes>
struct vfp_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 8 && nbytes == 64)
			      && !is_customfp_v<T> && !is_vcustomfp_v<T>>> {
    using type = avx512_8fx8<T>;
};
#endif // __AVX512F__

} // namespace target

/***********************************************************************
 * Vector traits
 ***********************************************************************/

template<typename T, unsigned short nbytes>
using vector_type_int_traits =
    target::vint_traits_select<T,sizeof(T),nbytes>::type;

template<typename T, unsigned short nbytes>
struct vector_type_traits<
    T,nbytes,std::enable_if_t<std::is_floating_point_v<T>>>
    : public target::vfp_traits_select<T,nbytes>::type { };

template<bool S, unsigned short E, unsigned short M, bool Z, int B,
	 unsigned short nbytes>
struct vector_type_traits<detail::customfp_em<S,E,M,Z,B>,nbytes>
    : public target::vint_traits_select<detail::customfp_em<S,E,M,Z,B>,(7+detail::customfp_em<S,E,M,Z,B>::bit_size)/8,nbytes>::type { };

template<typename Cfg, unsigned short nbytes>
struct vector_type_traits<vcustomfp<Cfg>,nbytes>
    : public target::vint_traits_select<vcustomfp<Cfg>,(7+Cfg::bit_size)/8,nbytes>::type { };

template<typename T, unsigned short nbytes>
struct vector_type_traits<T,
			  nbytes,
			  std::enable_if_t<std::is_integral_v<T>>>
    : public vector_type_int_traits<T,nbytes> { };

template<unsigned short nbits, unsigned short nbytes>
struct vector_type_traits<bitfield<nbits>,nbytes>
    : public target::vector_type_bitfield_traits<nbits,nbytes> { };

// Wider logical types are defined in vt_longint.h
template<unsigned short W, unsigned short nbytes>
struct vector_type_traits<logical<W>, nbytes, std::enable_if_t<(W<=8)>>
    : public vector_type_int_traits<logical<W>,nbytes> { };

/***********************************************************************
 * Mask traits (AVX-512)
 ***********************************************************************/
#include "graptor/target/bitmask.h"

/***********************************************************************
 * Alternative ways to access vector_type_traits
 ***********************************************************************/
template<typename T, typename V>
struct vector_type_traits_of :
    public vector_type_traits<T, sizeof(V)> {
};

template<typename VT, unsigned short VL>
struct vector_type_traits_with :
    public vector_type_traits<typename int_type_of_size<sizeof(VT)/VL>::type, sizeof(VT)> {
};

template<typename T, unsigned short VL, typename Enable>
struct vector_type_traits_vl :
    public vector_type_traits<T, sizeof(T)*VL> {
};

template<unsigned short VL>
struct vector_type_traits_vl<bitfield<1>,VL> :
    public vector_type_traits<bitfield<1>,VL/8> {
};

template<unsigned short VL>
struct vector_type_traits_vl<bitfield<2>,VL> :
    public vector_type_traits<bitfield<2>,VL/4> {
};

template<unsigned short VL>
struct vector_type_traits_vl<bitfield<4>,VL> :
    public vector_type_traits<bitfield<4>,VL/2> {
};

template<bool S, unsigned short E, unsigned short M, bool Z, int B>
struct vector_type_traits_vl<detail::customfp_em<S,E,M,Z,B>,3,
			     std::enable_if_t<detail::customfp_em<S,E,M,Z,B>::bit_size==21>> :
    public vector_type_traits<detail::customfp_em<S,E,M,Z,B>, 8> {
};

template<bool S, unsigned short E, unsigned short M, bool Z, int B>
struct vector_type_traits_vl<detail::customfp_em<S,E,M,Z,B>,12,
			     std::enable_if_t<detail::customfp_em<S,E,M,Z,B>::bit_size==21>> :
    public vector_type_traits<detail::customfp_em<S,E,M,Z,B>, 32> {
};

template<bool S, unsigned short E, unsigned short M, bool Z, int B>
struct vector_type_traits_vl<detail::customfp_em<S,E,M,Z,B>,24,
			     std::enable_if_t<detail::customfp_em<S,E,M,Z,B>::bit_size==21>> :
    public vector_type_traits<detail::customfp_em<S,E,M,Z,B>, 64> {
};

#include "graptor/target/conversion.h"

#include "graptor/target/impl/avx2_4x8.h"
#include "graptor/target/impl/sse42_4x4.h"
#include "graptor/target/impl/sse42_1x16.h"

#endif // GRAPTOR_TARGET_VECTOR_H
