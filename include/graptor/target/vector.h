// -*- c++ -*-

#ifndef GRAPTOR_TARGET_VECTOR_H
#define GRAPTOR_TARGET_VECTOR_H

#include <type_traits>
#include <x86intrin.h>
#include <immintrin.h>

#include "graptor/utils.h"
#include "graptor/itraits.h"
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
 * stuff
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

//#include "graptor/target/vt_doubled.h"
#include "graptor/target/vt_recursive.h"

// #include "graptor/target/pseudo_1x.h"

namespace target {

template<typename T, unsigned short nbytes, typename = void>
struct vint_traits_select {
    static_assert( nbytes > sizeof(T), "recursive case" );
    using type = vt_recursive<
	T,sizeof(T),nbytes,
	typename vint_traits_select<T,next_ipow2(nbytes/2)>::type,
	typename vint_traits_select<T,nbytes-next_ipow2(nbytes/2)>::type>;
};

#if 0
// Requires some fixes around the use of vector_type_int_traits in
// native_1x4.
template<typename T>
struct vint_traits_select<T,4,std::enable_if_t<sizeof(T)==1>> {
    using type = target::native_1x4<T>;
};
#endif

template<unsigned short nbits>
struct vint_traits_select<
    bitfield<nbits>,0,std::enable_if_t<nbits==1||nbits==2||nbits==4>> {
    using type = target::bitfield_scalar<nbits>;
};

template<unsigned short nbits,unsigned short nbytes>
struct vint_traits_select<
    bitfield<nbits>,nbytes,std::enable_if_t<nbits==1||nbits==2||nbits==4>> {
    using type = target::bitfield_24_byte<nbits,nbytes>;
};

template<typename T, unsigned short nbytes>
struct vint_traits_select<
    T,nbytes,std::enable_if_t<!std::is_same<T,bool>::value
			      && sizeof(T) == nbytes
			      && !is_bitfield_v<T>>> {
    using type = target::scalar_int<T>;
};

template<>
struct vint_traits_select<bool,1> {
    using type = target::scalar_bool;
};

/*
template<typename T, unsigned short nbytes>
struct vint_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 1 && nbytes == 2)>> {
    using type = pseudo_1x<T,2>;
};

template<typename T, unsigned short nbytes>
struct vint_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 1 && nbytes == 4)>> {
    using type = pseudo_1x<T,4>;
};
*/

#if __MMX__ && __SSE4_2__
template<typename T, unsigned short nbytes>
struct vint_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 1 && nbytes == 8
			       && !is_bitfield_v<T>)>> {
#if GRAPTOR_USE_MMX
    using type = mmx_1x8<T>;
#else
    using type = sse42_1x8<T>;
#endif
};

template<typename T, unsigned short nbytes>
struct vint_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 1 && nbytes == 4
			       && !is_bitfield_v<T>)>> {
#if GRAPTOR_USE_MMX && 0 
    // using type = mmx_1x4<T>; NYI
#else
    using type = sse42_1x4<T>;
#endif
};

template<typename T, unsigned short nbytes>
struct vint_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 2 && nbytes == 8)>> {
#if GRAPTOR_USE_MMX
    using type = mmx_2x4<T>;
#else
    using type = sse42_2x4<T>;
#endif
};
#endif // __MMX__ && __SSE4_2__

#if __MMX__
template<typename T, unsigned short nbytes>
struct vint_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 4 && nbytes == 8)>> {
    using type = mmx_4x2<T>;
};
#else
/*
template<typename T, unsigned short nbytes>
struct vint_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 1 && nbytes == 8)>> {
    using type = pseudo_1x<T,8>;
};
*/
#endif // __MMX__

#if __SSE4_2__
// Requires some fixes around the use of vector_type_int_traits
template<typename T, unsigned short nbytes>
struct vint_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 1 && !is_bitfield_v<T>
			       && nbytes == 16)>> {
    using type = sse42_1x16<T>;
};

template<typename T, unsigned short nbytes>
struct vint_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 2 && !is_bitfield_v<T>
			        && nbytes == 16)>> {
    using type = sse42_2x8<T>;
};

template<typename T, unsigned short nbytes>
struct vint_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 4 && !is_bitfield_v<T>
			        && nbytes == 16)>> {
    using type = sse42_4x4<T>;
};

template<typename T, unsigned short nbytes>
struct vint_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 8 && !is_bitfield_v<T>
			        && nbytes == 16)>> {
    using type = sse42_8x2<T>;
};
#endif // __SSE4_2__

#if __AVX2__
template<typename T, unsigned short nbytes>
struct vint_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 1 && !is_bitfield_v<T>
			       && nbytes == 32)>> {
    using type = avx2_1x32<T>;
};

template<typename T, unsigned short nbytes>
struct vint_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 2 && nbytes == 32)>> {
    using type = avx2_2x16<T>;
};

template<typename T, unsigned short nbytes>
struct vint_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 4 && nbytes == 32)>> {
    using type = avx2_4x8<T>;
};

template<typename T, unsigned short nbytes>
struct vint_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 8 && nbytes == 32)>> {
    using type = avx2_8x4<T>;
};

template<typename T, unsigned short nbytes>
struct vint_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 16 && nbytes == 32)>> {
    using type = avx2_16x2<T>;
};
#endif // __AVX2__

#if __AVX512F__
template<typename T, unsigned short nbytes>
struct vint_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 1 && nbytes == 64)
			      && !is_bitfield_v<T>>> {
    using type = avx512_1x64<T>;
};

template<typename T, unsigned short nbytes>
struct vint_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 2 && nbytes == 64)
			      && !is_bitfield_v<T>>> {
    using type = avx512_2x32<T>;
};

template<typename T, unsigned short nbytes>
struct vint_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 4 && nbytes == 64)>> {
    using type = avx512_4x16<T>;
};

template<typename T, unsigned short nbytes>
struct vint_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 8 && nbytes == 64)>> {
    using type = avx512_8x8<T>;
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
    T,nbytes,std::enable_if_t<(sizeof(T) == 4 && nbytes == 16)>> {
    using type = sse42_4fx4<T>;
};

template<typename T, unsigned short nbytes>
struct vfp_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 8 && nbytes == 16)>> {
    using type = sse42_8fx2<T>;
};
#endif

#if __AVX2__
template<typename T, unsigned short nbytes>
struct vfp_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 4 && nbytes == 32)>> {
    using type = avx2_4fx8<T>;
};

template<typename T, unsigned short nbytes>
struct vfp_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 8 && nbytes == 32)>> {
    using type = avx2_8fx4<T>;
};
#endif // __AVX2__

#if __AVX512F__
template<typename T, unsigned short nbytes>
struct vfp_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 4 && nbytes == 64)>> {
    using type = avx512_4fx16<T>;
};

template<typename T, unsigned short nbytes>
struct vfp_traits_select<
    T,nbytes,std::enable_if_t<(sizeof(T) == 8 && nbytes == 64)>> {
    using type = avx512_8fx8<T>;
};
#endif // __AVX512F__

} // namespace target

// template<typename T, unsigned short nbytes, typename = void>
// struct vector_type_int_traits;

template<typename T, unsigned short nbytes>
using vector_type_int_traits =
    typename target::vint_traits_select<T,nbytes>::type;

/***********************************************************************
 * Mask traits (AVX-512)
 ***********************************************************************/
#include "graptor/target/bitmask.h"

/***********************************************************************
 * Vector traits
 ***********************************************************************/

template<typename T, unsigned short nbytes>
struct vector_type_traits<T,
			  nbytes,
			  std::enable_if_t<std::is_integral_v<T>
					   || is_bitfield_v<T>>>
    : public vector_type_int_traits<T,nbytes> { };


// Wider logical types are defined in vt_longint.h
#if __AVX512F__ || __AVX2__
template<unsigned short W, unsigned short nbytes>
struct vector_type_traits<logical<W>, nbytes, std::enable_if_t<(W<=8)>>
    : public vector_type_int_traits<logical<W>,nbytes> { };
#else
template<unsigned short W, unsigned short nbytes>
struct vector_type_traits<logical<W>, nbytes, std::enable_if_t<(W<=8)>>
    : public vector_type_int_traits<logical<W>,nbytes> { };
#endif

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

#include "graptor/target/conversion.h"

#include "graptor/target/impl/avx2_4x8.h"
#include "graptor/target/impl/sse42_4x4.h"
#include "graptor/target/impl/sse42_1x16.h"

#endif // GRAPTOR_TARGET_VECTOR_H
