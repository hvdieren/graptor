// -*- c++ -*-
#ifndef GRAPTOR_TARGET_SSE42_4fx4_H
#define GRAPTOR_TARGET_SSE42_4fx4_H

#include <x86intrin.h>
#include <immintrin.h>

#include "graptor/target/decl.h"
#include "graptor/target/vt_recursive.h"
#include "graptor/target/sse42_4x4.h"

namespace target {

/***********************************************************************
 * Cases representable in 128 bits (SSEx)
 ***********************************************************************/
#if __SSE4_2__
template<typename T = float>
struct sse42_4fx4 {
    static_assert( sizeof(T) == 4, "size assumption" );

    using member_type = T;
    using int_type = uint32_t;
    using type = __m128;
    using itype = __m128i;
    using vmask_type = __m128i;

    using mask_traits = mask_type_traits<4>;
    using mask_type = typename mask_traits::type;

    // using half_traits = vector_type_traits<member_type,8>;
    // using recursive_traits = vt_recursive<member_type,4,16,half_traits>;
    using int_traits = sse42_4x4<int_type>;
    
    static constexpr size_t W = 4;
    static constexpr size_t B = 8*W;
    static constexpr size_t vlen = 4;
    static constexpr size_t size = W * vlen;

    static member_type lane( type a, int idx ) {
	switch( idx ) {
	case 0: return _mm_cvtss_f32( a );
	case 1: return _mm_cvtss_f32( _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 0, 0, 1)) );
	case 2: return _mm_cvtss_f32( _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 0, 0, 2)) );
	case 3: return _mm_cvtss_f32( _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 0, 0, 3)) );
	default:
	    assert( 0 && "should not get here" );
	}
    }
    static member_type lane0( type a ) { return lane( a, 0 ); }
    static member_type lane1( type a ) { return lane( a, 1 ); }
    static member_type lane2( type a ) { return lane( a, 2 ); }
    static member_type lane3( type a ) { return lane( a, 3 ); }

    static __m64 lower_half( type a ) {
	return _mm_cvtsi64_m64( _mm_extract_epi64( _mm_castps_si128( a ), 0 ) );
    }
    static __m64 upper_half( type a ) {
	return _mm_cvtsi64_m64( _mm_extract_epi64( _mm_castps_si128( a ), 1 ) );
    }
    static vpair<__m64,__m64> decompose( type a ) {
	return vpair<__m64,__m64>{ lower_half(a), upper_half(a) };
    }

    static type setzero() { return _mm_setzero_ps(); }
    static type set1( member_type a ) { return _mm_set1_ps( a ); }
    static type set( member_type a3, member_type a2,
		     member_type a1, member_type a0 ) {
	return _mm_set_ps( a3, a2, a1, a0 );
    }

    static vmask_type asvector( mask_type m ) {
	return int_traits::asvector( m );
    }
    static type blend( vmask_type m, type a, type b ) {
	return _mm_blendv_ps( a, b, _mm_castsi128_ps( m ) );
    }
    static type blend( mask_type m, type a, type b ) {
	return _mm_blendv_ps( a, b, _mm_castsi128_ps( asvector( m ) ) );
    }

#if __AVX__
    template<int flag>
    static vmask_type cmp_vmask( type a, type b ) {
	return _mm_castps_si128( _mm_cmp_ps( a, b, flag ) );
    }
    static vmask_type cmpne( type a, type b, mt_vmask ) {
	return cmp_vmask<_CMP_NEQ_OQ>( a, b );
    }
    static vmask_type cmpeq( type a, type b, mt_vmask ) {
	return cmp_vmask<_CMP_EQ_OQ>( a, b );
    }
    static vmask_type cmpgt( type a, type b, mt_vmask ) {
	return cmp_vmask<_CMP_GT_OQ>( a, b );
    }
    static vmask_type cmpge( type a, type b, mt_vmask ) {
	return cmp_vmask<_CMP_GE_OQ>( a, b );
    }
    static vmask_type cmplt( type a, type b, mt_vmask ) {
	return cmp_vmask<_CMP_LT_OQ>( a, b );
    }
    static vmask_type cmple( type a, type b, mt_vmask ) {
	return cmp_vmask<_CMP_LE_OQ>( a, b );
    }
#else
    static vmask_type cmpeq( type a, type b, mt_vmask ) {
	return _mm_castps_si128( _mm_cmpeq_ps( a, b ) );
    }
    static vmask_type cmpne( type a, type b, mt_vmask ) {
	return _mm_castps_si128( _mm_cmpneq_ps( a, b ) );
    }
    static vmask_type cmpgt( type a, type b, mt_vmask ) {
	return _mm_castps_si128( _mm_cmpgt_ps( a, b ) );
    }
    static vmask_type cmpge( type a, type b, mt_vmask ) {
	return _mm_castps_si128( _mm_cmpge_ps( a, b ) );
    }
    static vmask_type cmplt( type a, type b, mt_vmask ) {
	return _mm_castps_si128( _mm_cmplt_ps( a, b ) );
    }
    static vmask_type cmple( type a, type b, mt_vmask ) {
	return _mm_castps_si128( _mm_cmple_ps( a, b ) );
    }
#endif

#if __AVX__
#if __AVX512VL__
    template<int flag>
    static mask_type cmp_mask( type a, type b ) {
	return _mm_cmp_ps_mask( a, b, flag );
    }
#else
    template<int flag>
    static mask_type cmp_mask( type a, type b ) {
	auto r = _mm_cmp_ps( a, b, flag );
	return _mm_movemask_ps( r );
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
#endif


    static type abs( type a ) {
	// Not available; do it the hard way.
	__m128i m = int_traits::setone_shr1();
	__m128 i = _mm_and_ps( _mm_castsi128_ps( m ), a );
	return i;
    }

    static type add( type a, type b ) { return _mm_add_ps( a, b ); }
    static type add( type s, vmask_type m, type a, type b ) {
	return _mm_blendv_ps( s, add( a, b ), _mm_castsi128_ps( m ) );
    }
    static type mul( type a, type b ) { return _mm_mul_ps( a, b ); }
    static type mul( type s, vmask_type m, type a, type b ) {
	return _mm_blendv_ps( s, mul( a, b ), _mm_castsi128_ps( m ) );
    }
    static type div( type a, type b ) { return _mm_div_ps( a, b ); }
    static type sub( type a, type b ) { return _mm_sub_ps( a, b ); }

    static member_type reduce_add( type a ) {
	// return recursive_traits::reduce_add( decompose(a) );
	return lane0( a ) + lane1( a ) + lane2( a ) + lane3( a );
    }
    static member_type reduce_add( type a, vmask_type m ) {
#if __AVX512F__
	__m512i wm = _mm512_zextsi128_si512( m );
	__mmask8 mm = _mm512_cmpneq_epi32_mask( wm, _mm512_setzero_si512() );
	__m512 c = _mm512_insertf32x4( _mm512_setzero_ps(), a, 0 );
	return _mm512_mask_reduce_add_ps( mm, c );
#else
	member_type r = 0;
	if( int_traits::lane0( m ) ) r += lane0( a );
	if( int_traits::lane1( m ) ) r += lane1( a );
	if( int_traits::lane2( m ) ) r += lane2( a );
	if( int_traits::lane3( m ) ) r += lane3( a );
	return r;
#endif
    }

    static type castfp( type a ) { return a; }
    static itype castint( type a ) { return _mm_castps_si128( a ); }

    static type load( const member_type *a ) {
	return _mm_load_ps( (const member_type *)a );
    }
    static type loadu( const member_type *a ) {
	return _mm_loadu_ps( (const member_type *)a );
    }
    static void store( member_type *addr, type val ) {
	_mm_store_ps( (member_type *)addr, val );
    }
    static void storeu( member_type *addr, type val ) {
	_mm_storeu_ps( (member_type *)addr, val );
    }

    static type
    gather( const member_type *a, itype b ) {
#if __AVX2__
	return _mm_i32gather_ps( a, b, W );
#else
	using idx_traits = int_traits;
	// vector_type_traits_vl<typename int_type_of_size<sizeof(itype)/vlen>::type, vlen>;
	return set( a[idx_traits::lane3(b)], a[idx_traits::lane2(b)],
		    a[idx_traits::lane1(b)], a[idx_traits::lane0(b)] );
#endif
    }
    static type
    gather( const member_type *a, itype b, mask_type mask ) {
	// return _mm_mask_i32gather_epi32( setzero(), a, b, asvector(mask), W );
	assert( 0 && "NYI" );
	return setzero();
    }
/*
*/
    static type
    gather( const member_type *a, itype b, vmask_type vmask ) {
#if __AVX2__
	return _mm_mask_i32gather_ps( setzero(), a, b,
				      _mm_castsi128_ps( vmask ), W );
#else
	return set(
	    int_traits::lane3(vmask) ? a[int_traits::lane3(b)] : member_type(0),
	    int_traits::lane2(vmask) ? a[int_traits::lane2(b)] : member_type(0),
	    int_traits::lane1(vmask) ? a[int_traits::lane1(b)] : member_type(0),
	    int_traits::lane0(vmask) ? a[int_traits::lane0(b)] : member_type(0)
	    );
#endif
    }

    static void scatter( member_type *a, itype b, type c ) {
	a[int_traits::lane0(b)] = lane0(c);
	a[int_traits::lane1(b)] = lane1(c);
	a[int_traits::lane2(b)] = lane2(c);
	a[int_traits::lane3(b)] = lane3(c);
    }
    static void scatter( member_type *a, itype b, type c, vmask_type mask ) {
	if( int_traits::lane0(mask) ) a[int_traits::lane0(b)] = lane0(c);
	if( int_traits::lane1(mask) ) a[int_traits::lane1(b)] = lane1(c);
	if( int_traits::lane2(mask) ) a[int_traits::lane2(b)] = lane2(c);
	if( int_traits::lane3(mask) ) a[int_traits::lane3(b)] = lane3(c);
    }
    static void scatter( member_type *a, itype b, type c, mask_type mask ) {
	if( mask_traits::lane0(mask) ) a[int_traits::lane0(b)] = lane0(c);
	if( mask_traits::lane1(mask) ) a[int_traits::lane1(b)] = lane1(c);
	if( mask_traits::lane2(mask) ) a[int_traits::lane2(b)] = lane2(c);
	if( mask_traits::lane3(mask) ) a[int_traits::lane3(b)] = lane3(c);
    }
};
#endif // __SSE4_2__

} // namespace target

#endif //  GRAPTOR_TARGET_SSE42_4fx4_H
