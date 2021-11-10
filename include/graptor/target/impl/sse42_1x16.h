// -*- c++ -*-
#ifndef GRAPTOR_TARGET_IMPL_SSE42_1x16_H
#define GRAPTOR_TARGET_IMPL_SSE42_1x16_H

#include "graptor/target/sse42_1x16.h"
#include "graptor/target/avx2_4x8.h"

namespace target {

#if __AVX512F__
#else
// Assumes __AVX2__
#if __AVX2__
#else
#error expecting that __AVX2__ is defined
#endif
template<unsigned short VL, typename T>
typename sse42_1xL<VL,T>::type
sse42_1xL<VL,T>::gather( const typename sse42_1xL<VL,T>::member_type *a,
			 typename avx2_4x8<uint32_t>::type b,
			 typename avx2_4x8<uint32_t>::type m ) {
    if constexpr ( VL == 8 ) {
	using it = avx2_4x8<uint32_t>;
	auto g = it::template gather_w<W>(
	    reinterpret_cast<const uint32_t *>( a ), b, m );
	__m128i i = convert_4b_1b( g );
	return i;
    } else
	assert( 0 && "NYI" );
}

template<unsigned short VL, typename T>
typename sse42_1xL<VL,T>::type
sse42_1xL<VL,T>::gather(
    const typename sse42_1xL<VL,T>::member_type *a,
    typename vt_recursive<uint32_t,4,64,avx2_4x8<uint32_t>>::type b,
    typename vt_recursive<uint32_t,4,64,avx2_4x8<uint32_t>>::type m )
{
    if constexpr ( VL == 16 ) {
	using xt = vt_recursive<uint32_t,4,64,avx2_4x8<uint32_t>>;
	using it = avx2_4x8<uint32_t>;
	auto g = it::template gather_w<W>(
	    reinterpret_cast<const uint32_t *>( a ),
	    xt::lower_half( b ), xt::lower_half( m ) );
	auto h = it::template gather_w<W>(
	    reinterpret_cast<const uint32_t *>( a ),
	    xt::upper_half( b ), xt::upper_half( m ) );
	__m128i i = convert_4b_1b( g );
	__m128i j = convert_4b_1b_hi( h );
	return bitwise_or( i, j );
    } else if constexpr ( VL == 8 ) {
	using xt = vt_recursive<uint64_t,8,64,avx2_8x4<uint64_t>>;
	using mt = vector_type_traits_vl<logical<4>,8>;
	auto mm = conversion_traits<logical<8>,logical<4>,8>::convert( m );
	// Loads 4x4
	__m128i g = _mm256_mask_i64gather_epi32(
	    setzero(), reinterpret_cast<const int32_t *>( a ),
	    xt::lower_half( b ), mt::lower_half( mm ), 1 );
	// Loads 4x4
	__m128i h = _mm256_mask_i64gather_epi32(
	    setzero(), reinterpret_cast<const int32_t *>( a ),
	    xt::upper_half( b ), mt::upper_half( mm ), 1 );
	__m256i k = avx2_bitwise::set_pair( h, g );
	// Convert using truncation
	return conversion_traits<uint32_t,uint8_t,8>::convert( k );
    } else
	assert( 0 && "NYI" );
}

// Assumes IdxT must be vpair
template<unsigned short VL, typename T>
template<typename IdxT>
typename sse42_1xL<VL,T>::type
sse42_1xL<VL,T>::gather( const typename sse42_1xL<VL,T>::member_type *a, IdxT b )
{
    if constexpr ( VL == 16 ) {
	using it = avx2_4x8<uint32_t>;
	__m256i g = it::template gather_w<W>(
	    reinterpret_cast<const uint32_t *>( a ), b.a );
	__m256i h = it::template gather_w<W>(
	    reinterpret_cast<const uint32_t *>( a ), b.b );
	__m128i i = convert_4b_1b( g );
	__m128i j = convert_4b_1b_hi( h );
	return bitwise_or( i, j );
    } else
	assert( 0 && "NYI" );
}

template<unsigned short VL, typename T>
void
sse42_1xL<VL,T>::scatter( member_type *a, vpair<__m256i,__m256i> b,
			  type c, vmask_type mask ) {
    if constexpr( VL == 8 ) {
	using wint_traits = vector_type_traits_vl<uint64_t,8>;
	for( unsigned int l=0; l < 8; ++l ) {
	    if( int_traits::lane( mask, l ) )
		a[wint_traits::lane( b, l )] = lane( c, l );
	}
    } else
	assert( 0 && "NYI" );
}

#endif // __AVX2__

} // namespace target

#endif // GRAPTOR_TARGET_IMPL_SSE42_1x16_H

