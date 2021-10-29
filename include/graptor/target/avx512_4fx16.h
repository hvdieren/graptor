// -*- c++ -*-
#ifndef GRAPTOR_TARGET_AVX512_4fx16_H
#define GRAPTOR_TARGET_AVX512_4fx16_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include "graptor/target/decl.h"

#if __SSE4_2__
#include "graptor/target/sse42_4fx4.h"
#endif // __SSE4_2__

#if __AVX512F__
#include "graptor/target/avx512_4x16.h"
#endif // __AVX512F__

alignas(64) extern const uint32_t avx512_4x16_evenodd_intlv_epi32_vl8[16];
alignas(64) extern const uint32_t avx512_4x16_evenodd_intlv_epi32_vl16[16];

alignas(64) extern const uint32_t avx512_4x16_evenodd_intlv_inv_epi32_vl8[16];

namespace target {

/***********************************************************************
 * AVX512 16 floats
 ***********************************************************************/
#if __AVX512F__
template<typename T = float>
struct avx512_4fx16 {
    static_assert( sizeof(T) == 4, 
		   "version of template class for 4-byte floats" );
public:
    static constexpr size_t W = 4;
    static constexpr size_t B = 8*W;
    static constexpr size_t vlen = 16;
    static constexpr size_t size = W * vlen;

    using member_type = T;
    using int_type = uint32_t;
    using type = __m512;
    using itype = __m512i;
    using vmask_type = __m512i;
    using mask_traits = mask_type_traits<vlen>;
    using mask_type = typename mask_traits::type;

    // using half_traits = avx2_8fx4<T>;
    using int_traits = avx512_4x16<int_type>;
    using mt_preferred = target::mt_mask;
    
    static type setzero() { return _mm512_setzero_ps(); }

    static type set1( member_type a ) { return _mm512_set1_ps( a ); }
    static type set_pair( __m256 hi, __m256 lo ) {
	return _mm512_insertf32x8( _mm512_castps256_ps512( lo ), hi, 1 );
    }

    static __m256 lower_half( type a ) {
	return _mm512_castps512_ps256( a );
    }
    static __m256 upper_half( type a ) {
	return _mm512_extractf32x8_ps( a, 1 );
    }

    static mask_type cmpeq( type a, type b, target::mt_mask ) {
	return _mm512_cmp_ps_mask( a, b, _CMP_EQ_OQ );
    }
    static mask_type cmpne( type a, type b, target::mt_mask ) {
	return _mm512_cmp_ps_mask( a, b, _CMP_NEQ_OQ );
    }
    static mask_type cmple( type a, type b, target::mt_mask ) {
	return _mm512_cmp_ps_mask( a, b, _CMP_LE_OQ );
    }
    static mask_type cmplt( type a, type b, target::mt_mask ) {
	return _mm512_cmp_ps_mask( a, b, _CMP_LT_OQ );
    }
    static mask_type cmpge( type a, type b, target::mt_mask ) {
	return _mm512_cmp_ps_mask( a, b, _CMP_GE_OQ );
    }
    static mask_type cmpgt( type a, type b, target::mt_mask ) {
	return _mm512_cmp_ps_mask( a, b, _CMP_GT_OQ );
    }

    static vmask_type cmpeq( type a, type b, mt_vmask ) {
	return asvector( cmpeq( a, b, mt_mask() ) );
    }
    static vmask_type cmpne( type a, type b, mt_vmask ) {
	return asvector( cmpne( a, b, mt_mask() ) );
    }
    static vmask_type cmpgt( type a, type b, mt_vmask ) {
	return asvector( cmpgt( a, b, mt_mask() ) );
    }
    static vmask_type cmpge( type a, type b, mt_vmask ) {
	return asvector( cmpge( a, b, mt_mask() ) );
    }
    static vmask_type cmplt( type a, type b, mt_vmask ) {
	return asvector( cmplt( a, b, mt_mask() ) );
    }
    static vmask_type cmple( type a, type b, mt_vmask ) {
	return asvector( cmple( a, b, mt_mask() ) );
    }

    static type add( type a, type b ) { return _mm512_add_ps( a, b ); }
    static type sub( type a, type b ) { return _mm512_sub_ps( a, b ); }
    static type add( type src, mask_type m, type a, type b ) {
	return _mm512_mask_add_ps( src, m, a, b );
    }
    static type mul( type a, type b ) { return _mm512_mul_ps( a, b ); }
    static type mul( type src, mask_type m, type a, type b ) {
	return _mm512_mask_mul_ps( src, m, a, b );
    }
    static type div( type a, type b ) { return _mm512_div_ps( a, b ); }

    static member_type lane( type a, int idx ) {
	using ht = sse42_4fx4<member_type>;
	
	switch( idx ) {
	case 0:  return ht::lane( _mm512_extractf32x4_ps( a, 0 ), 0 );
	case 1:  return ht::lane( _mm512_extractf32x4_ps( a, 0 ), 1 );
	case 2:  return ht::lane( _mm512_extractf32x4_ps( a, 0 ), 2 );
	case 3:  return ht::lane( _mm512_extractf32x4_ps( a, 0 ), 3 );
	case 4:  return ht::lane( _mm512_extractf32x4_ps( a, 1 ), 0 );
	case 5:  return ht::lane( _mm512_extractf32x4_ps( a, 1 ), 1 );
	case 6:  return ht::lane( _mm512_extractf32x4_ps( a, 1 ), 2 );
	case 7:  return ht::lane( _mm512_extractf32x4_ps( a, 1 ), 3 );
	case 8:  return ht::lane( _mm512_extractf32x4_ps( a, 2 ), 0 );
	case 9:  return ht::lane( _mm512_extractf32x4_ps( a, 2 ), 1 );
	case 10: return ht::lane( _mm512_extractf32x4_ps( a, 2 ), 2 );
	case 11: return ht::lane( _mm512_extractf32x4_ps( a, 2 ), 3 );
	case 12: return ht::lane( _mm512_extractf32x4_ps( a, 3 ), 0 );
	case 13: return ht::lane( _mm512_extractf32x4_ps( a, 3 ), 1 );
	case 14: return ht::lane( _mm512_extractf32x4_ps( a, 3 ), 2 );
	case 15: return ht::lane( _mm512_extractf32x4_ps( a, 3 ), 3 );
	default:
	    UNREACHABLE_CASE_STATEMENT;
	}
    }

    static type abs( type a ) { return _mm512_abs_ps( a ); }

    static member_type lane0( type a ) { return *(member_type *)&a; }
    static member_type lane1( type a ) { return *(((member_type *)&a)+1); }
    static member_type lane2( type a ) { return *(((member_type *)&a)+2); }
    static member_type lane3( type a ) { return *(((member_type *)&a)+3); }

    static type blendm( mask_type m, type a, type b ) {
	return blend( m, a, b );
    }
    static type blend( mask_type m, type a, type b ) {
	return _mm512_mask_blend_ps( m, a, b );
    }
    static type blendm( vmask_type m, type a, type b ) {
	return blend( m, a, b );
    }
    static type blend( vmask_type m, type a, type b ) {
	return _mm512_mask_blend_ps( int_traits::asmask( m ), a, b );
    }

    static type castfp( type a ) { return a; }
    static itype castint( type a ) { return _mm512_castps_si512( a ); }

    template<unsigned short PermVL>
    static type permute_evenodd( type a ) {
#if __AVX512VL__
	// Even/odd interleaving of the elements of a
	const uint32_t * shuf;
	// if constexpr ( PermVL == 4 )
	// shuf = avx512_4x16_evenodd_intlv_epi32_vl4;
	// else
	if constexpr ( PermVL == 8 )
	    shuf = avx512_4x16_evenodd_intlv_epi32_vl8;
	// else if constexpr ( PermVL == 16 )
	// shuf = avx512_4x16_evenodd_intlv_epi32_vl16;
	else
	    assert( 0 && "NYI" );

	const itype mask =
	    _mm512_load_si512( reinterpret_cast<const itype *>( shuf ) );
	const type p = _mm512_permutexvar_ps( mask, a );
	return p;
#endif // __AVX512VL__
	assert( 0 && "NYI" );
	return setzero();
    }

    template<unsigned short PermVL>
    static type permute_inv_evenodd( type a ) {
#if __AVX512VL__
	// Even/odd interleaving of the elements of a
	const uint32_t * shuf;
	// if constexpr ( PermVL == 4 )
	// shuf = avx512_4x16_evenodd_intlv_inv_epi32_vl4;
	// else
	if constexpr ( PermVL == 8 )
	    shuf = avx512_4x16_evenodd_intlv_inv_epi32_vl8;
	// else if constexpr ( PermVL == 16 )
	// shuf = avx512_4x16_evenodd_intlv_inv_epi32_vl16;
	else
	    assert( 0 && "NYI" );

	const itype mask =
	    _mm512_load_si512( reinterpret_cast<const itype *>( shuf ) );
	const type p = _mm512_permutexvar_ps( mask, a );
	return p;
#endif // __AVX512VL__
	assert( 0 && "NYI" );
	return setzero();
    }


    static type load( const member_type *a ) {
	return _mm512_load_ps( a );
    }
    static type loadu( const member_type *a ) {
	return _mm512_loadu_ps( a );
    }
    static void store( member_type *addr, type val ) {
	_mm512_store_ps( addr, val );
    }
    static void storeu( member_type *addr, type val ) {
	_mm512_storeu_ps( addr, val );
    }
    static type ntload( const member_type *a ) {
	return _mm512_castsi512_ps( _mm512_stream_load_si512( (__m512i *)a ) );
    }
    static void ntstore( member_type *a, type val ) {
	_mm512_stream_si512( (__m512i *)a, _mm512_castps_si512( val ) );
    }
    static member_type reduce_add( type val ) {
	return _mm512_reduce_add_ps( val );
    }
    static member_type reduce_add( type val, mask_type mask ) {
	return _mm512_mask_reduce_add_ps( mask, val );
    }
    static member_type reduce_min( type val ) {
	return _mm512_reduce_min_ps( val );
    }
    static member_type reduce_min( type val, mask_type mask ) {
	return _mm512_mask_reduce_min_ps( mask, val );
    }
    static member_type reduce_max( type val ) {
	return _mm512_reduce_max_ps( val );
    }
    static member_type reduce_max( type val, mask_type mask ) {
	return _mm512_mask_reduce_max_ps( mask, val );
    }
    static vmask_type asvector( mask_type mask ) {
	return int_traits::asvector( mask );
    }
    template<typename IdxT>
    static auto
    gather( member_type *a, IdxT b,
	    typename std::enable_if_t<sizeof(IdxT) == size> * = nullptr
	) {
	return _mm512_i32gather_ps( b, a, W );
    }
    static type
    gather( member_type *a, itype b, mask_type mask ) {
	return _mm512_mask_i32gather_ps( setzero(), mask, b, a, W );
    }
    static void
    scatter( member_type *a, itype b, type c ) {
	_mm512_i32scatter_ps( (void *)a, b, c, W );
    }
    static void
    scatter( member_type *a, itype b, type c, mask_type mask ) {
	_mm512_mask_i32scatter_ps( (void *)a, mask, b, c, W );
    }
};
#endif // __AVX512F__

} // namespace target

#endif // GRAPTOR_TARGET_AVX512_4fx16_H
