// -*- c++ -*-
#ifndef GRAPTOR_TARGET_VTLONGINT_H
#define GRAPTOR_TARGET_VTLONGINT_H

#include "graptor/target/vector.h"

#include "graptor/target/sse42_16x1.h"
#include "graptor/target/avx2_16x2.h"
#include "graptor/target/vt_recursive.h"

/***********************************************************************
 * Select which version to use depending on the vectorization
 * capabilities of the hardware
 ***********************************************************************/
template<typename T, unsigned short W, unsigned short nbytes, typename = void>
struct vt_longint_select;

#define VT_LONGINT_SELECT_BASE 0

#if __AVX2__
// AVX2 supports everything up to 32 bytes
#if VT_LONGINT_SELECT_BASE < 32
#undef VT_LONGINT_SELECT_BASE
#define VT_LONGINT_SELECT_BASE 32
#endif

template<typename T, unsigned short W, unsigned short nbytes>
struct vt_longint_select<T, W, nbytes, std::enable_if_t<(W==16 && nbytes==32)>> {
    using type = target::avx2_16x2<T>;
};
#endif

#if __SSE4_2__
// SSE4.2 supports everything up to 32 bytes
#if VT_LONGINT_SELECT_BASE < 16
#undef VT_LONGINT_SELECT_BASE
#define VT_LONGINT_SELECT_BASE 16
#endif
template<typename T, unsigned short W, unsigned short nbytes>
struct vt_longint_select<T, W, nbytes, std::enable_if_t<(W==16 && nbytes==16)>> {
    using type = target::sse42_16x1<T>;
};
#endif

// Default case. Enable only for cases with more bytes than the bases cases
template<typename T, unsigned short W, unsigned short nbytes>
struct vt_longint_select<T, W, nbytes,
			 std::enable_if_t<(nbytes>VT_LONGINT_SELECT_BASE)>> {
    using type = target::vt_recursive<
	T, W, nbytes, typename vt_longint_select<T, W, nbytes/2>::type >;
};

/***********************************************************************
 * Make definitions accessible through common framework
 ***********************************************************************/
template<unsigned short W, unsigned short nbytes>
struct vector_type_traits<longint<W>, nbytes>
    : public vt_longint_select<longint<W>, W, nbytes>::type { };

template<unsigned short W, unsigned short nbytes>
struct vector_type_traits<logical<W>, nbytes, std::enable_if_t<(W>8)>>
    : public vt_longint_select<logical<W>, W, nbytes>::type { };

/***********************************************************************
 * Element width conversion
 ***********************************************************************/
template<unsigned short WFrom, unsigned short WTo>
struct conversion_traits<logical<WFrom>, logical<WTo>, 8,
			 std::enable_if_t<(WFrom>8) && (WFrom%WTo==0)>> {
    // View source vector through the lens of a very long vector and pick
    // just a few of the lanes. Assuming WFrom >> WTo, extract + insert
    // is the best approach.
    // If WFrom/To shows multiple words are picked from the same 128-bit lane,
    // a shuffle may halve the work.
    static constexpr unsigned short R = WFrom/WTo;
    using src_traits = vector_type_traits_vl<logical<WTo>, 8*R>;
    using dst_traits = vector_type_traits_vl<logical<WTo>, 8>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	return dst_traits::set(
	    src_traits::lane( a, 8*R-1 ),
	    src_traits::lane( a, 7*R-1 ),
	    src_traits::lane( a, 6*R-1 ),
	    src_traits::lane( a, 5*R-1 ),
	    src_traits::lane( a, 4*R-1 ),
	    src_traits::lane( a, 3*R-1 ),
	    src_traits::lane( a, 2*R-1 ),
	    src_traits::lane( a, 1*R-1 ) );
		
    }
};

#if __AVX2__
template<>
struct conversion_traits<logical<4>, logical<16>, 8> {
    // Blow up source vector R times.
    static constexpr unsigned short R = 4;
    using src_traits = vector_type_traits_vl<logical<4>, 8>;
    using mid_traits = vector_type_traits_vl<logical<16>, 4>;
    using dst_traits = vector_type_traits_vl<logical<16>, 8>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	// Making strong assumptions: we have AVX2 support, so we know we
	// replicate each lane 4 times.
	static_assert( std::is_same<__m256i,typename src_traits::type>::value,
		       "assumption" );
	// a = ( a7, a6, a5, a4, a3, a2, a1, a0 )
	// v0a = ( a5, a5, a4, a4, a1, a1, a0, a0 )
	__m256i v0a = _mm256_unpacklo_epi32( a, a );
	// v0 = ( a1, a1, a1, a1, a0, a0, a0, a0 )
	__m256i v0 = _mm256_permute4x64_epi64( v0a, 0b01010000 );
	// v2 = ( a5, a5, a5, a5, a4, a4, a4, a4 )
	__m256i v2 = _mm256_permute4x64_epi64( v0a, 0b11111010 );

	// v0b = ( a7, a7, a6, a6, a3, a3, a2, a2 )
	__m256i v1a = _mm256_unpackhi_epi32( a, a );
	// v1 = ( a3, a3, a3, a3, a2, a2, a2, a2 )
	__m256i v1 = _mm256_permute4x64_epi64( v1a, 0b01010000 );
	// v1 = ( a7, a7, a7, a7, a6, a6, a6, a6 )
	__m256i v3 = _mm256_permute4x64_epi64( v1a, 0b11111010 );

	return dst_traits::set_pair(
	    mid_traits::set_pair( v3, v2 ),
	    mid_traits::set_pair( v1, v0 ) );
    }
};
#endif // __AVX2__

#include "graptor/longint_impl.h"

#endif //  GRAPTOR_TARGET_VTLONGINT_H
