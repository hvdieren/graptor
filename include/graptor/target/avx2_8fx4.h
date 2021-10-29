// -*- c++ -*-
#ifndef GRAPTOR_TARGET_AVX2_8fx4_H
#define GRAPTOR_TARGET_AVX2_8fx4_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include "graptor/target/decl.h"

#if __AVX2__ // AVX2 implies SSE4.2
// #include "graptor/target/sse42_8fx2.h"
#include "graptor/target/avx2_8x4.h"
#endif // __AVX2__

namespace target {

/***********************************************************************
 * AVX2 8 float
 ***********************************************************************/
#if __AVX2__
template<typename T = double>
struct avx2_8fx4 {
    static_assert( sizeof(T) == 8, 
		   "version of template class for 8-byte floats" );
public:
    static constexpr size_t W = 8;
    static constexpr size_t B = 8*W;
    static constexpr size_t vlen = 4;
    static constexpr size_t size = W * vlen;

    using member_type = T;
    using int_type = uint64_t;
    using type = __m256d;
    using itype = __m256i;
    using vmask_type = __m256i;

    // using half_traits = sse42_8fx2<T>;
    using int_traits = avx2_8x4<int_type>;
    using mt_preferred = target::mt_vmask;
    
    using mask_traits = mask_type_traits<vlen>;
    using mask_type = typename mask_traits::type;

    static type set1( member_type a ) { return _mm256_set1_pd( a ); }
    static type setzero() { return _mm256_setzero_pd(); }

    static mask_type asmask( vmask_type vmask ) {
#if __AVX512F__
	__m512i wmask = _mm512_castsi256_si512( vmask );
	__m512i zero = _mm512_setzero_si512();
	mask_type m = _mm512_cmpneq_epi64_mask( wmask, zero );
	return m;
#elif __AVX__
	mask_type m = _mm256_movemask_pd( _mm256_castsi256_pd( vmask ) );
	return m;
	assert( 0 && "NYI" );
	return 0;
#endif
    }
    static mask_type asmask( __m128i vmask ) {
#if __AVX512F__
	__m512i wmask = _mm512_castsi128_si512( vmask );
	__m512i zero = _mm512_setzero_si512();
	mask_type m = _mm512_cmpneq_epi64_mask( wmask, zero );
	return m;
#elif __AVX__
	mask_type m = _mm_movemask_pd( _mm_castsi128_pd( vmask ) );
	return m;
	assert( 0 && "NYI" );
	return 0;
#endif
    }

    static type blend( vmask_type m, type a, type b ) {
	return _mm256_blendv_pd( a, b, _mm256_castsi256_pd( m ) );
    }
    static type blend( mask_type m, type a, type b ) {
	return _mm256_mask_blend_pd( m, a, b );
    }

    static type add( type a, type b ) { return _mm256_add_pd( a, b ); }
    static type add( type src, vmask_type m, type a, type b ) {
	return _mm256_blendv_pd( src, add( a, b ), _mm256_castsi256_pd( m ) );
    }
    static type add( type src, __m128i m, type a, type b ) {
#if __AVX512F__ && __AVX512VL__
	mask_type mm = asmask( m );
	return _mm256_mask_blend_pd( mm, src, add( a, b ) );
#else
	return _mm256_blendv_pd( src, add( a, b ),
				 _mm256_castsi256_pd(
				     _mm256_cvtepi32_epi64( m ) ) );
#endif
    }
    static type add( type src, mask_type m, type a, type b ) {
#if __AVX512F__ && __AVX512VL__
	return _mm256_mask_blend_pd( m, src, add( a, b ) );
#else
	return add( src, asvector( m ), a, b );
#endif
    }
    static type sub( type a, type b ) { return _mm256_sub_pd( a, b ); }
    static type mul( type a, type b ) { return _mm256_mul_pd( a, b ); }
    static type mul( type src, vmask_type m, type a, type b ) {
	return _mm256_blendv_pd( src, mul( a, b ), _mm256_castsi256_pd( m ) );
    }
    static type mul( type src, mask_type m, type a, type b ) {
	return mul( src, asvector( m ), a, b );
    }
    static type div( type a, type b ) { return _mm256_div_pd( a, b ); }
    static type min( type a, type b ) { return _mm256_min_pd( a, b ); }
    static type max( type a, type b ) { return _mm256_max_pd( a, b ); }

    static type abs( type a ) {
	// No _mm256_abs_pd( a, b ) routine exists
	itype mask = int_traits::srli( int_traits::setone(), 1 );
	return _mm256_and_pd( a, int_traits::castfp( mask ) );
    }

    static member_type lane0( type a ) { return *(member_type *)&a; }
    static member_type lane1( type a ) { return *(((member_type *)&a)+1); }
    static member_type lane2( type a ) { return *(((member_type *)&a)+2); }
    static member_type lane3( type a ) { return *(((member_type *)&a)+3); }
    static member_type lane( type a, unsigned l ) {
	return *(((member_type *)&a)+l);
    }

    static vmask_type cmpne( type a, type b, mt_vmask ) {
	auto r = _mm256_cmp_pd( a, b, _CMP_NEQ_OQ );
	return _mm256_castpd_si256( r );
    }
    static vmask_type cmpeq( type a, type b, mt_vmask ) {
	auto r = _mm256_cmp_pd( a, b, _CMP_EQ_OQ );
	return _mm256_castpd_si256( r );
    }
    static vmask_type cmpgt( type a, type b, mt_vmask ) {
	auto r = _mm256_cmp_pd( a, b, _CMP_GT_OQ );
	return _mm256_castpd_si256( r );
    }
    static vmask_type cmpge( type a, type b, mt_vmask ) {
	auto r = _mm256_cmp_pd( a, b, _CMP_GE_OQ );
	return _mm256_castpd_si256( r );
    }
    static vmask_type cmplt( type a, type b, mt_vmask ) {
	auto r = _mm256_cmp_pd( a, b, _CMP_LT_OQ );
	return _mm256_castpd_si256( r );
    }
    static vmask_type cmple( type a, type b, mt_vmask ) {
	auto r = _mm256_cmp_pd( a, b, _CMP_LE_OQ );
	return _mm256_castpd_si256( r );
    }

#if __AVX512VL__
    template<int flag>
    static mask_type cmp_mask( type a, type b ) {
	return _mm256_cmp_pd_mask( a, b, flag );
    }
#else
    template<int flag>
    static mask_type cmp_mask( type a, type b ) {
	auto r = _mm256_cmp_pd( a, b, flag );
	return _mm256_movemask_pd( r );
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

    static itype castint( type a ) { return _mm256_castpd_si256( a ); }
    static type castfp( type a ) { return a; }

    static type load( const member_type *a ) {
	return _mm256_load_pd( a );
    }
    static type loadu( const member_type *a ) {
	return _mm256_loadu_pd( a );
    }
    static void store( member_type *addr, type val ) {
	_mm256_store_pd( addr, val );
    }
    static void storeu( member_type *addr, type val ) {
	_mm256_storeu_pd( addr, val );
    }
    static member_type reduce_add( type val ) {
	type s = _mm256_hadd_pd( val, _mm256_permute2f128_pd( val, val, 1 ) );
	type t = _mm256_hadd_pd( s, s );
	return lane0( t );
    }
    static vmask_type asvector( mask_type mask ) {
	// Need to work with 8 32-bit integers as there is no 64-bit srai
	// in AVX2. Luckily, this works just as well.
	vmask_type vmask = _mm256_set1_epi32( (int)mask );
	const __m128i cnt = _mm_set_epi16( 28, 28, 29, 29, 30, 30, 31, 31 );
	vmask = _mm256_sll_epi32( vmask, cnt );
	return _mm256_srai_epi32( vmask, 31 );
    }
    static member_type reduce_add( type val, mask_type mask ) {
	return reduce_add( blend( mask, setzero(), val ) );
    }
    static member_type reduce_min( type a ) {
	member_type l0 = lane( a, 0 );
	member_type l1 = lane( a, 1 );
	member_type l2 = lane( a, 2 );
	member_type l3 = lane( a, 3 );
	member_type v0 = std::min( l0, l1 );
	member_type v1 = std::min( l2, l3 );
	return std::min( v0, v1 );
    }
    static member_type reduce_max( type a ) {
	member_type l0 = lane( a, 0 );
	member_type l1 = lane( a, 1 );
	member_type l2 = lane( a, 2 );
	member_type l3 = lane( a, 3 );
	member_type v0 = std::max( l0, l1 );
	member_type v1 = std::max( l2, l3 );
	return std::max( v0, v1 );
    }
    static type
    gather( const member_type *a, itype b ) {
	return _mm256_i64gather_pd( a, b, W );
    }
    static type
    gather( const member_type *a, itype b, vmask_type mask ) {
	return _mm256_mask_i64gather_pd( setzero(), a, b,
					 _mm256_castsi256_pd( mask ), W );
    }
#if __AVX512F__ && __AVX512VL__
    static type
    gather( const member_type *a, itype b, mask_type mask ) {
	return _mm256_mmask_i64gather_pd( setzero(), mask, b, a, W );
    }
#endif // __AVX512F__ && __AVX512VL__
    static type
    gather( const member_type *a, __m128i b ) {
	return _mm256_i32gather_pd( a, b, W );
    }
    static type
    gather( const member_type *a, __m128i b, __m128i mask ) {
	vmask_type wmask = _mm256_cvtepi32_epi64( mask );
	return _mm256_mask_i32gather_pd( setzero(), a, b,
					 _mm256_castsi256_pd( wmask ), W );
    }
    static void
    scatter( const member_type *a, itype b, type val ) {
	assert( 0 && "NYI" );
    }
    static void
    scatter( const member_type *a, itype b, type val, vmask_type mask ) {
	assert( 0 && "NYI" );
    }
#if __AVX512F__ && __AVX512VL__
    static void
    scatter( member_type *a, itype b, type val, mask_type mask ) {
	_mm256_mask_i64scatter_pd( a, mask, b, val, W );
    }
    static void
    scatter( member_type *a, __m128i b, type val, mask_type mask ) {
	_mm256_mask_i32scatter_pd( a, mask, b, val, W );
    }
#endif // __AVX512F__ && __AVX512VL__
    static void
    scatter( const member_type *a, __m128i b, type val ) {
	assert( 0 && "NYI" );
    }
    static void
    scatter( const member_type *a, __m128i b, type val, vmask_type mask ) {
	assert( 0 && "NYI" );
    }
};
#endif // __AVX2__

} // namespace target

#endif // GRAPTOR_TARGET_AVX2_8fx4_H
