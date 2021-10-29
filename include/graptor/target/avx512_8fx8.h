// -*- c++ -*-
#ifndef GRAPTOR_TARGET_AVX512_8fx8_H
#define GRAPTOR_TARGET_AVX512_8fx8_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include "graptor/target/decl.h"

#if __AVX512F__
#include "graptor/target/avx512_8x8.h"
#endif // __AVX512F__

#if __AVX2__
#include "graptor/target/avx2_4x8.h"
#endif // __AVX2__

namespace target {

/***********************************************************************
 * AVX512 8 double
 ***********************************************************************/
#if __AVX512F__
template<typename T = double>
struct avx512_8fx8 {
    static_assert( sizeof(T) == 8, 
		   "version of template class for 8-byte floats" );
public:
    using member_type = T;
    using int_type = uint64_t;
    using type = __m512d;
    using itype = __m512i;
    using vmask_type = __m512i;

    using mask_traits = mask_type_traits<8>;
    using mask_type = typename mask_traits::type;

    // using half_traits = avx2_8fx4<T>;
    using int_traits = avx512_8x8<int_type>;
    using mt_preferred = target::mt_mask;
    
    static constexpr size_t W = 8;
    static constexpr size_t B = 8*W;
    static constexpr size_t vlen = 8;
    static constexpr size_t size = W * vlen;

    static type setzero() { return _mm512_setzero_pd(); }
    static type setoneval() { return set1( 1.0 ); }

    static type set1( member_type a ) { return _mm512_set1_pd( a ); }

    static __m256d lower_half( type a ) {
	return _mm512_castpd512_pd256( a );
    }
    static __m256d upper_half( type a ) {
	return _mm512_extractf64x4_pd( a, 1 );
    }

    static mask_type asmask( vmask_type a ) {
	return int_traits::asmask( a );
    }

    static type abs( type a ) {
#if __MACOSX__ || __GNUC__ > 7 // exception case for system installation
	return _mm512_abs_pd( a );
#else
	return _mm512_abs_pd( _mm512_castpd_ps( a ) );
#endif
    }

    static type add( type a, type b ) { return _mm512_add_pd( a, b ); }
    static type add( type src, mask_type m, type a, type b ) {
	return _mm512_mask_add_pd( src, m, a, b );
    }
    static type add( type src, vmask_type m, type a, type b ) {
	return add( src, asmask( m ), a, b );
    }
    static type sub( type a, type b ) { return _mm512_sub_pd( a, b ); }
    static type mul( type a, type b ) { return _mm512_mul_pd( a, b ); }
    static type mul( type src, mask_type m, type a, type b ) {
	return _mm512_mask_mul_pd( src, m, a, b );
    }
    static type div( type a, type b ) { return _mm512_div_pd( a, b ); }
    static type min( type a, type b ) { return _mm512_min_pd( a, b ); }
    static type max( type a, type b ) { return _mm512_max_pd( a, b ); }

    static member_type lane( type a, int idx ) {
	// This is ugly - beware
	double tmp[vlen];
	// store( tmp, a );
	*(type *)tmp = a; // perhaps compiler can do something with this
	return tmp[idx];
    }

    static member_type lane0( type a ) { return *(member_type *)&a; }
    static member_type lane1( type a ) { return *(((member_type *)&a)+1); }
    static member_type lane2( type a ) { return *(((member_type *)&a)+2); }
    static member_type lane3( type a ) { return *(((member_type *)&a)+3); }

    static type blend( mask_type m, type a, type b ) {
	return _mm512_mask_blend_pd( m, a, b );
    }
    static type blendm( mask_type m, type a, type b ) {
	return blend( m, a, b );
    }
    static type blend( vmask_type m, type a, type b ) {
	return _mm512_mask_blend_pd( int_traits::asmask( m ), a, b );
    }
    static type blendm( vmask_type m, type a, type b ) {
	return blend( m, a, b );
    }

    static type castfp( type a ) { return a; }
    static itype castint( type a ) { return _mm512_castpd_si512( a ); }

    static type load( const member_type *a ) {
	return _mm512_load_pd( a );
    }
    static type loadu( const member_type *a ) {
	return _mm512_loadu_pd( a );
    }
    static void store( member_type *addr, type val ) {
	_mm512_store_pd( addr, val );
    }
    static void storeu( member_type *addr, type val ) {
	_mm512_storeu_pd( addr, val );
    }
    static member_type reduce_add( type val ) {
	return _mm512_reduce_add_pd( val );
    }
    static member_type reduce_min( type val ) {
	return _mm512_reduce_min_pd( val );
    }
    static member_type reduce_max( type val ) {
	return _mm512_reduce_max_pd( val );
    }
    static member_type reduce_add( type val, mask_type mask ) {
	return _mm512_mask_reduce_add_pd( mask, val );
    }
    static vmask_type asvector( mask_type mask ) {
	return int_traits::asvector( mask );
    }

    static mask_type cmpeq( type a, type b, target::mt_mask ) {
	return _mm512_cmp_pd_mask( a, b, _CMP_EQ_OQ );
    }
    static mask_type cmpne( type a, type b, target::mt_mask ) {
	return _mm512_cmp_pd_mask( a, b, _CMP_NEQ_OQ );
    }
    static mask_type cmple( type a, type b, target::mt_mask ) {
	return _mm512_cmp_pd_mask( a, b, _CMP_LE_OQ );
    }
    static mask_type cmplt( type a, type b, target::mt_mask ) {
	return _mm512_cmp_pd_mask( a, b, _CMP_LT_OQ );
    }
    static mask_type cmpge( type a, type b, target::mt_mask ) {
	return _mm512_cmp_pd_mask( a, b, _CMP_GE_OQ );
    }
    static mask_type cmpgt( type a, type b, target::mt_mask ) {
	return _mm512_cmp_pd_mask( a, b, _CMP_GT_OQ );
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

    
    static GG_INLINE auto
    gather( const member_type *a, itype b ) {
	return _mm512_i64gather_pd( b, a, W );
    }
    static GG_INLINE auto
    gather( const member_type *a, itype b, mask_type m ) {
	return _mm512_mask_i64gather_pd( setzero(), m, b, a, W );
    }
    static GG_INLINE auto
    gather( const member_type *a, itype b, vmask_type m ) {
	return _mm512_mask_i64gather_pd( setzero(), asmask( m ), b, a, W );
    }
    static type
    gather( const member_type *a, __m256i b ) {
	return _mm512_i32gather_pd( b, a, W );
    }
    static type
    gather( const member_type *a, __m256i b, mask_type m ) {
	return _mm512_mask_i32gather_pd( setzero(), m, b, a, W );
    }
    static GG_INLINE void
    scatter( member_type *a, itype b, type c ) {
	_mm512_i64scatter_pd( (void *)a, b, c, W );
    }
    static GG_INLINE void
    scatter( member_type *a, itype b, type c, mask_type m ) {
	_mm512_mask_i64scatter_pd( (void *)a, m, b, c, W );
    }
    static GG_INLINE void
    scatter( member_type *a, itype b, type c, vmask_type m ) {
	_mm512_mask_i64scatter_pd( (void *)a, asmask( m ), b, c, W );
    }
    static GG_INLINE void
    scatter( member_type *a, __m256i b, type c ) {
	_mm512_i32scatter_pd( (void *)a, b, c, W );
    }
    static GG_INLINE void
    scatter( member_type *a, __m256i b, type c, mask_type m ) {
	_mm512_mask_i32scatter_pd( (void *)a, m, b, c, W );
    }
/*
    static GG_INLINE void
    scatter( member_type *a, hitype b, type c ) {
	_mm512_i32scatter_pd( (void *)a, b, c, W );
    }
    static GG_INLINE void
    scatter( member_type *a, hitype b, type c, mask_type m ) {
	_mm512_mask_i32scatter_pd( (void *)a, m, b, c, W );
    }
    static GG_INLINE void
    scatter( member_type *a, hitype b, type c, vmask_type m ) {
	_mm512_mask_i32scatter_pd( (void *)a, asmask( m ), b, c, W );
    }
*/
};
#endif // __AVX512F__

} // namespace target

#endif // GRAPTOR_TARGET_AVX512_8fx8_H
