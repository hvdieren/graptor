// -*- c++ -*-
#ifndef GRAPTOR_TARGET_AVX2_4fx8_H
#define GRAPTOR_TARGET_AVX2_4fx8_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include "graptor/target/decl.h"

#if __AVX2__ // AVX2 implies SSE4.2
#include "graptor/target/sse42_4fx4.h"
#include "graptor/target/avx2_4x8.h"
#endif // __AVX2__

alignas(64) extern const uint32_t increasing_sequence_epi32[16];
alignas(64) extern const uint32_t avx2_4x8_evenodd_intlv_epi32_vl4[8];
alignas(64) extern const uint32_t avx2_4x8_evenodd_intlv_inv_epi32_vl4[8];

namespace target {

/***********************************************************************
 * AVX2 8 float
 ***********************************************************************/
#if __AVX2__
template<typename T = float>
struct avx2_4fx8 {
    static_assert( sizeof(T) == 4, 
		   "version of template class for 4-byte floats" );
public:
    static constexpr size_t W = 4;
    static constexpr size_t B = 8*W;
    static constexpr size_t vlen = 8;
    static constexpr size_t size = W * vlen;

    using member_type = T;
    using type = __m256;
    using vmask_type = __m256i;
    using itype = __m256i;
    using int_type = uint32_t;
    using mask_traits = mask_type_traits<vlen>;
    using mask_type = typename mask_traits::type;

    using half_traits = sse42_4fx4<T>;
    using int_traits = avx2_4x8<int_type>;
    using mt_preferred = target::mt_vmask;
    
    static mask_type asmask( vmask_type vmask ) {
	return int_traits::asmask( vmask );
    }
    
    static member_type lane( type a, size_t l ) {
	member_type r;
	__m128 lo = _mm256_extractf128_ps( a, 0 );
	__m128 hi = _mm256_extractf128_ps( a, 1 );
	switch( l ) {
	case 0:
	    _MM_EXTRACT_FLOAT( r, lo, 0 );
	    break;
	case 1:
	    _MM_EXTRACT_FLOAT( r, lo, 1 );
	    break;
	case 2:
	    _MM_EXTRACT_FLOAT( r, lo, 2 );
	    break;
	case 3:
	    _MM_EXTRACT_FLOAT( r, lo, 3 );
	    break;
	case 4:
	    _MM_EXTRACT_FLOAT( r, hi, 0 );
	    break;
	case 5:
	    _MM_EXTRACT_FLOAT( r, hi, 1 );
	    break;
	case 6:
	    _MM_EXTRACT_FLOAT( r, hi, 2 );
	    break;
	case 7:
	    _MM_EXTRACT_FLOAT( r, hi, 3 );
	    break;
	}
	return r;
    }
    static member_type lane0( type a ) { return lane( a, 0 ); }
    static member_type lane1( type a ) { return lane( a, 1 ); }
    static member_type lane2( type a ) { return lane( a, 2 ); }
    static member_type lane3( type a ) { return lane( a, 3 ); }
    static member_type lane4( type a ) { return lane( a, 4 ); }
    static member_type lane5( type a ) { return lane( a, 5 ); }
    static member_type lane6( type a ) { return lane( a, 6 ); }
    static member_type lane7( type a ) { return lane( a, 7 ); }

    static __m128 lower_half( type a  ) {
	return _mm256_extractf128_ps( a, 0 );
    }
    static __m128 upper_half( type a  ) {
	return _mm256_extractf128_ps( a, 1 );
    }

    static type blendm( vmask_type m, type a, type b ) {
	return blend( m, a, b );
    }
    static type blend( vmask_type m, type a, type b ) {
	return _mm256_blendv_ps( a, b, _mm256_castsi256_ps( m ) );
    }
    static type blendm( mask_type m, type a, type b ) {
	return blend( m, a, b );
    }
    static type blend( mask_type m, type a, type b ) {
#if __AVX512F__ && __AVX512VL__
	return _mm256_mask_blend_ps( m, a, b );
#else // __AVX512F__ && __AVX512VL__
	return blend( asvector( m ), a, b );
#endif // __AVX512F__ && __AVX512VL__
    }

    static type setzero() { return _mm256_setzero_ps(); }
    static type set1( member_type a ) { return _mm256_set1_ps( a ); }
    static type set_pair( __m128 hi, __m128 lo ) {
	return _mm256_insertf128_ps( _mm256_castps128_ps256( lo ), hi, 1 );
    }

    static type set( member_type a7, member_type a6,
		     member_type a5, member_type a4,
		     member_type a3, member_type a2,
		     member_type a1, member_type a0 ) {
	return _mm256_set_ps( a7, a6, a5, a4, a3, a2, a1, a0 );
    }

    // Set all lanes to the maximum lane
    static type set_max( type a ) {
	// Swap neighbouring values: ABCDEFGH -> BADCFEHG
	type a0 = _mm256_permute_ps( a, 0b10110001 );
	// Get the largest; all pairs of 2 lanes are now equal, e.g.: AADDFFHH
	type b0 = _mm256_max_ps( a0, a ); // AVX
	// Swap values across 4s: DDAAHHFF
	type a1 = _mm256_permute_ps( b0, 0b00011110 );
	// Get the largest; every 4 lanes are now equal, e.g.: AAAAFFFF
	type b1 = _mm256_max_ps( a1, b0 ); // AVX
	// Swap across SIMD halves: FFFFAAAA
	type a2 = _mm256_permute2f128_ps( b1, b1, 0b00000001 );
	// One more time max:
	return _mm256_max_ps( a2, b1 ); // AVX
    }

    static itype castint( type a ) { return _mm256_castps_si256( a ); }
    static type castfp( type a ) { return a; }

    static vmask_type cmpne( type a, type b, mt_vmask ) {
	auto r = _mm256_cmp_ps( a, b, _CMP_NEQ_OQ );
	return _mm256_castps_si256( r );
    }
    static vmask_type cmpeq( type a, type b, mt_vmask ) {
	auto r = _mm256_cmp_ps( a, b, _CMP_EQ_OQ );
	return _mm256_castps_si256( r );
    }
    static vmask_type cmpgt( type a, type b, mt_vmask ) {
	auto r = _mm256_cmp_ps( a, b, _CMP_GT_OQ );
	return _mm256_castps_si256( r );
    }
    static vmask_type cmpge( type a, type b, mt_vmask ) {
	auto r = _mm256_cmp_ps( a, b, _CMP_GE_OQ );
	return _mm256_castps_si256( r );
    }
    static vmask_type cmplt( type a, type b, mt_vmask ) {
	auto r = _mm256_cmp_ps( a, b, _CMP_LT_OQ );
	return _mm256_castps_si256( r );
    }
    static vmask_type cmple( type a, type b, mt_vmask ) {
	auto r = _mm256_cmp_ps( a, b, _CMP_LE_OQ );
	return _mm256_castps_si256( r );
    }

#if __AVX512VL__
    template<int flag>
    static mask_type cmp_mask( type a, type b ) {
	return _mm256_cmp_ps_mask( a, b, flag );
    }
#else
    template<int flag>
    static mask_type cmp_mask( type a, type b ) {
	auto r = _mm256_cmp_ps( a, b, flag );
	return _mm256_movemask_ps( r );
    }
#endif
    static mask_type cmpne( type a, type b, mt_mask ) {
	return cmp_mask<_CMP_NEQ_OQ>( a, b );
    }
    static mask_type cmpeq( type a, type b, mt_mask ) {
	return cmp_mask<_CMP_EQ_OQ>( a, b );
    }
    static mask_type cmpgt( type a, type b, mt_mask ) {
	return cmp_mask<_CMP_GT_OQ>( a, b );
    }
    static mask_type cmpge( type a, type b, mt_mask ) {
	return cmp_mask<_CMP_GE_OQ>( a, b );
    }
    static mask_type cmplt( type a, type b, mt_mask ) {
	return cmp_mask<_CMP_LT_OQ>( a, b );
    }
    static mask_type cmple( type a, type b, mt_mask ) {
	return cmp_mask<_CMP_LE_OQ>( a, b );
    }


    static vmask_type asvector( mask_type mask ) {
	return int_traits::asvector( mask );
    }

    static type add( type a, type b ) { return _mm256_add_ps( a, b ); }
    static type sub( type a, type b ) { return _mm256_sub_ps( a, b ); }
    static type mul( type a, type b ) { return _mm256_mul_ps( a, b ); }
    static type div( type a, type b ) { return _mm256_div_ps( a, b ); }
    static type add( type src, vmask_type m, type a, type b ) {
	return _mm256_blendv_ps( src, add( a, b ), _mm256_castsi256_ps( m ) );
    }
    static type add( type src, mask_type m, type a, type b ) {
	// return _mm256_mask_add_ps( src, m, a, b );
	// return _mm256_blend_ps( src, add( a, b ), m );
#if __AVX512F__ && __AVX512VL__
	return _mm256_mask_add_ps( src, m, a, b );
#else
	return add( src, asvector( m ), a, b );
#endif
    }
    static type mul( type src, vmask_type m, type a, type b ) {
	return _mm256_blendv_ps( src, mul( a, b ), _mm256_castsi256_ps( m ) );
    }
    static type mul( type src, mask_type m, type a, type b ) {
#if __AVX512F__ && __AVX512VL__
	return _mm256_mask_mul_ps( src, m, a, b );
#else
	return mul( src, asvector( m ), a, b );
#endif
    }

    static type rsqrt( type a ) {
	return _mm256_rsqrt_ps( a );
    }

    static type abs( type a ) {
	// type nega = _mm256_sub_ps( setzero(), a );
	// return _mm256_max_ps( nega, a );
	static const type sign_mask = _mm256_set1_ps(-0.f); // -0.f = 1 << 31
	return _mm256_andnot_ps( sign_mask, a );
    }

    template<unsigned short PermuteVL>
    static type permute_evenodd( type a ) {
	// Even/odd interleaving of the elements of a
	if constexpr ( PermuteVL == 4 ) {
	    const uint32_t * shuf = avx2_4x8_evenodd_intlv_epi32_vl4;
	    const itype mask =
		_mm256_load_si256( reinterpret_cast<const itype *>( shuf ) );
	    const type p = _mm256_permutevar8x32_ps( a, mask );
	    return p;
	} else
	    assert( 0 && "NYI" );
	return setzero();
    }

    template<unsigned short PermuteVL>
    static type permute_inv_evenodd( type a ) {
	// Even/odd interleaving of the elements of a
	if constexpr ( PermuteVL == 4 ) {
	    const uint32_t * shuf = avx2_4x8_evenodd_intlv_inv_epi32_vl4;
	    const itype mask =
		_mm256_load_si256( reinterpret_cast<const itype *>( shuf ) );
	    const type p = _mm256_permutevar8x32_ps( a, mask );
	    return p;
	} else
	    assert( 0 && "NYI" );
	return setzero();
    }

    static type load( const member_type *a ) {
	return _mm256_load_ps( a );
    }
    static type loadu( const member_type *a ) {
	return _mm256_loadu_ps( a );
    }
    static void store( member_type *addr, type val ) {
	_mm256_store_ps( addr, val );
    }
    static void storeu( member_type *addr, type val ) {
	_mm256_storeu_ps( addr, val );
    }
    static member_type reduce_add( type x ) {
	// https://stackoverflow.com/questions/13219146/how-to-sum-m256-horizontally
	// hiQuad = ( x7, x6, x5, x4 )
	const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
	// loQuad = ( x3, x2, x1, x0 )
	const __m128 loQuad = _mm256_castps256_ps128(x);
	// sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
	const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
	// loDual = ( -, -, x1 + x5, x0 + x4 )
	const __m128 loDual = sumQuad;
	// hiDual = ( -, -, x3 + x7, x2 + x6 )
	const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
	// sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
	const __m128 sumDual = _mm_add_ps(loDual, hiDual);
	// lo = ( -, -, -, x0 + x2 + x4 + x6 )
	const __m128 lo = sumDual;
	// hi = ( -, -, -, x1 + x3 + x5 + x7 )
	const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
	// sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
	const __m128 sum = _mm_add_ss(lo, hi);
	return _mm_cvtss_f32(sum);
    }
    static member_type reduce_add( type x, vmask_type m ) {
	// Replace inactive lanes by zero of addition operation,
	// then add using unmasked algorithm
	const type z = setzero();
	const type b = _mm256_blendv_ps( z, x, _mm256_castsi256_ps( m ) );
	return reduce_add( b );
    }
    static member_type reduce_min( type x ) {
	// hiQuad = ( x7, x6, x5, x4 )
	const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
	// loQuad = ( x3, x2, x1, x0 )
	const __m128 loQuad = _mm256_castps256_ps128(x);
	// sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
	const __m128 sumQuad = _mm_min_ps(loQuad, hiQuad);
	// loDual = ( -, -, x1 + x5, x0 + x4 )
	const __m128 loDual = sumQuad;
	// hiDual = ( -, -, x3 + x7, x2 + x6 )
	const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
	// sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
	const __m128 sumDual = _mm_min_ps(loDual, hiDual);
	// lo = ( -, -, -, x0 + x2 + x4 + x6 )
	const __m128 lo = sumDual;
	// hi = ( -, -, -, x1 + x3 + x5 + x7 )
	const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
	// sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
	const __m128 fin = _mm_min_ps(lo, hi);
	return _mm_cvtss_f32(fin);
    }
    static member_type reduce_min( type x, vmask_type m ) {
	// Replace inactive lanes by infinity,
	// then min using unmasked algorithm
	const __m256 inf = set1( std::numeric_limits<float>::infinity() );
	const type b = _mm256_blendv_ps( inf, x, _mm256_castsi256_ps( m ) );
	return reduce_min( b );
    }
    static member_type reduce_max( type x ) {
	// hiQuad = ( x7, x6, x5, x4 )
	const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
	// loQuad = ( x3, x2, x1, x0 )
	const __m128 loQuad = _mm256_castps256_ps128(x);
	// sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
	const __m128 sumQuad = _mm_max_ps(loQuad, hiQuad);
	// loDual = ( -, -, x1 + x5, x0 + x4 )
	const __m128 loDual = sumQuad;
	// hiDual = ( -, -, x3 + x7, x2 + x6 )
	const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
	// sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
	const __m128 sumDual = _mm_max_ps(loDual, hiDual);
	// lo = ( -, -, -, x0 + x2 + x4 + x6 )
	const __m128 lo = sumDual;
	// hi = ( -, -, -, x1 + x3 + x5 + x7 )
	const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
	// sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
	const __m128 fin = _mm_max_ps(lo, hi);
	return _mm_cvtss_f32(fin);
    }
    static member_type reduce_max( type x, vmask_type m ) {
	// Replace inactive lanes by infinity,
	// then max using unmasked algorithm
	const __m256 inf = set1( -std::numeric_limits<float>::infinity() );
	const type b = _mm256_blendv_ps( inf, x, _mm256_castsi256_ps( m ) );
	return reduce_max( b );
    }
    static auto
    gather( const member_type *a, itype b, vmask_type mask ) {
	return _mm256_mask_i32gather_ps( setzero(), a, b, *(type*)&mask, W );
    }
    static auto
    gather( const member_type *a, itype b, vpair<vmask_type,vmask_type> mask ) {
	__m128i lo = avx2_convert_8i_4i( mask.a );
	__m128i hi = avx2_convert_8i_4i( mask.b );
	vmask_type mask32 = _mm256_castsi128_si256( lo );
	mask32 = _mm256_inserti128_si256( mask32, hi, 1 );
	return _mm256_mask_i32gather_ps( setzero(), a, b, *(type*)&mask32, W );
    }
    static auto
    gather( const member_type *a, itype b, mask_type mask ) {
#if __AVX512F__ && __AVX512VL__
	return _mm256_mmask_i32gather_ps( setzero(), mask, b, a, W );
#else // __AVX512F__ && __AVX512VL__
	return _mm256_mask_i32gather_ps(
	    setzero(), a, b, _mm256_castsi256_ps( asvector( mask ) ), W );
#endif // __AVX512F__ && __AVX512VL__
    }
    static auto
    gather( const member_type *a, itype b ) {
	return _mm256_i32gather_ps( a, b, W );
    }
    static void scatter( member_type *a, itype b, type c ) {
#if __AVX512VL__
	_mm256_i32scatter_ps( (void *)a, b, c, sizeof(member_type) );
	scatter( a, b, c );
#else
	for( size_t i=0; i < vlen; ++i )
	    a[int_traits::lane( b, i )] = lane( c, i );
#endif
    }
    static void scatter( member_type *a, itype b, type c, mask_type mask ) {
#if __AVX512VL__
	_mm256_mask_i32scatter_ps( (void *)a, mask, b, c, sizeof(member_type) );
#else
	for( size_t i=0; i < vlen; ++i ) {
	    if( mask & 1 )
		a[int_traits::lane( b, i )] = lane( c, i );
	    mask >>= 1;
	}
#endif
    }
    static void scatter( member_type *a, itype b, type c, vmask_type mask ) {
#if __AVX512VL__
	_mm256_mask_i32scatter_ps( (void *)a, asmask( mask ), b, c,
				   sizeof(member_type) );
#else
	for( size_t i=0; i < vlen; ++i )
	    if( int_traits::lane( mask, i ) )
		a[int_traits::lane( b, i )] = lane( c, i );
#endif
    }
    static void scatter( member_type *a, itype b, type c,
			 vpair<vmask_type,vmask_type> mask ) {
#if __AVX512VL__
	__m128i lo = avx2_convert_8i_4i( mask.a );
	__m128i hi = avx2_convert_8i_4i( mask.b );
	vmask_type mask32 = _mm256_castsi128_si256( lo );
	mask32 = _mm256_inserti128_si256( mask32, hi, 1 );
	_mm256_mask_i32scatter_ps( (void *)a, asmask( mask32 ), b, c,
				   sizeof(member_type) );
#else
	for( size_t i=0; i < vlen/2; ++i )
	    if( int_traits::lane( mask.a, i ) )
		a[int_traits::lane( b, i )] = lane( c, i );
	for( size_t i=vlen/2; i < vlen; ++i )
	    if( int_traits::lane( mask.b, i-vlen/2 ) )
		a[int_traits::lane( b, i )] = lane( c, i );
#endif
    }

    static __m128i avx2_convert_8i_4i( __m256i a ) {
#if __AVX512VL__
	return _mm256_cvtepi64_epi32( a );
#else
	// We choose an instruction sequence that does not require loading
	// shuffle masks. A single step would be possible with permutevar8x32
	// followed by cast to extract lower 128 bits, however, the load of
	// the shuffle mask is expensive (possible 7 cycles) on sky lake,
	// and will occupy a register to hold the temporary.
	// This conversion simply truncates the integers to 32 bits.
	const __m256i s = _mm256_shuffle_epi32( a, 0b10001000 );
	const __m256i z = target::avx2_bitwise::setzero();
	const __m256i l = _mm256_blend_epi32( z, s, 0b11000011 );
	const __m128i hi = _mm256_extracti128_si256( l, 1 );
	const __m128i lo = _mm256_castsi256_si128( l );
	return _mm_or_si128( hi, lo );
#endif
    }
};
#endif // __AVX2__

} // namespace target

#endif // GRAPTOR_TARGET_AVX2_4x8_H
