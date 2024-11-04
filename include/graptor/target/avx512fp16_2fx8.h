// -*- c++ -*-
#ifndef GRAPTOR_TARGET_AVX512FP16_2fx8_H
#define GRAPTOR_TARGET_AVX512FP16_2fx8_H

#include <bit>

#include <x86intrin.h>
#include <immintrin.h>

#include "graptor/target/decl.h"
#include "graptor/target/vt_recursive.h"
#include "graptor/target/sse42_2x8.h"

namespace target {

/***********************************************************************
 * Cases representable in 128 bits (AVX512-FP16)
 ***********************************************************************/
#if __AVX512FP16__
template<typename T = float>
struct avx512fp16_2fx8 {
    static_assert( sizeof(T) == 2, "size assumption" );

    static constexpr size_t W = 2;
    static constexpr size_t B = 8*W;
    static constexpr size_t vlen = 8;
    static constexpr size_t size = W * vlen;

    using member_type = T;
    using int_type = uint16_t;
    using type = __m128h;
    using itype = __m128i;
    using vmask_type = __m128i;

    // using half_traits = avx512fp16_2fx4<T>;
    // using lo_half_traits = half_traits;
    // using hi_half_traits = half_traits;
    using int_traits = sse42_2x8<int_type>;

    using mask_traits = mask_type_traits<vlen>;
    using mask_type = typename mask_traits::type;
    using mt_preferred = target::mt_mask;
    using vmask_traits = sse42_2x8<uint16_t>;
    
    static member_type lane( type a, int idx ) {
	return std::bit_cast<member_type>( int_traits::lane( castint( a ), idx ) );
    }
    static member_type lane0( type a ) {
	return std::bit_cast<member_type>( int_traits::lane0( castint( a ) ) );
    }
    static member_type lane1( type a ) {
	return std::bit_cast<member_type>( int_traits::lane1( castint( a ) ) );
    }
    static member_type lane2( type a ) {
	return std::bit_cast<member_type>( int_traits::lane2( castint( a ) ) );
    }
    static member_type lane3( type a ) {
	return std::bit_cast<member_type>( int_traits::lane3( castint( a ) ) );
    }

    // static __m64 lower_half( type a ) { }
    // static __m64 upper_half( type a ) { }
    // static vpair<__m64,__m64> decompose( type a ) { }

    static type setzero() { return _mm_setzero_ph(); }
    static type set1( member_type a ) { return _mm_set1_ph( a ); }
    static type set1( type a ) {
	return _mm_castsi128_ph( _mm_broadcastw_epi16( castint( a ) ) ); }
    static type set( member_type a3, member_type a2,
		     member_type a1, member_type a0 ) {
	return _mm_set_ph( a3, a2, a1, a0 );
    }
    static type setl0( member_type a ) {
	// Also zeros lanes 1-7
	return _mm_set_sh( a );
    }

   // Set all lanes to the maximum lane
    static type set_max( type a ) {
#if 0
	__m128i ai = _mm_castph_si128( a );
	// ABCDEFGH -> EFGHABCD
	__m128i aj = _mm_shuffle_epi32( ai, 0b01001110 );
	// ABCDEFGH x EFGHABCD -> EAFBGCHD
	type a0 = _mm_castsi128_ph( _mm_unpacklo_epi16( aj, ai ) );
	// max:  AE AB CF BD EG CF GH DH
	type b0 = _mm_max_ph( a0, a );
	// swap: GH DH EG CF CF BD AE AB
	type a1 = _mm_castsi128_ph(
	    _mm_shuffle_epi32( _mm_castph_si128( b0 ), 0b00011011 ) );
	// max: AEGH ABDH CEFG BCDF CEFG BCDF AEGH ABDH
	type b1 = _mm_max_ph( a1, a );
	// cmp:  2,4  3,5  0,6  1,7  0,6  1,7  2,4  3,5
	// pairs: 1,7 vs 2,4   X..Y.YX.
	//        0,6 vs 3,5   .XY.Y..X
	__m128i b1i = _mm_castph_si128( b1 );
	// shuflo: AEGH ABDH CEFG BCDF | ABDH AEGH BCDF CEFG 
	__m128i b1l = _mm_shufflelo_epi16( b1i, 0b00011011 );
	// shufhi: BCDF CEFG ABDH AEGH | CEFG BCDF AEGH ABDH
	__m128i b1h = _mm_shufflehi_epi16( b1i, 0b00011011 );
	__m128h b1lh = _mm_castsi128_ph( b1l );
	__m128h b1hh = _mm_castsi128_ph( b1h );
	__m128h a2 = _mm_max_ph( b1lh, b1hh );
	return a2;
#else
	// This version is like a max_reduce, except it keeps the
	// value in the SSE registers until the broadcast, saving the
	// GPR vs SSE transfer latency twice.
	// ABCDEFGH -> EFGHABCD
	type a0 = _mm_castsi128_ph(
	    _mm_shuffle_epi32( _mm_castph_si128( a ), 0b01001110 ) );
	// max: AE BF CG DH AE BF CG DH
	type b0 = _mm_max_ph( a0, a );
	// swap: x x x x CG DH AE BF
	type a1 = _mm_castsi128_ph(
	    _mm_shuffle_epi32( _mm_castph_si128( b0 ), 0b0001 ) );
	// max: x x x x ACEG BDFH ACEG BDFH
	type b1 = _mm_max_ph( a1, a0 );
	// swap bottom two lanes
	type a2 = _mm_castsi128_ph(
	    _mm_shufflelo_epi16( _mm_castph_si128( b1 ), 0b0100 ) );
	// final max
	type b2 = _mm_max_ph( a2, a1 );
	// broadcast
	return set1( b2 );
#endif
    }

    static type blend( vmask_type m, type a, type b ) {
	return _mm_mask_blend_ph( int_traits::asmask( m ), a, b );
    }
    static type blend( mask_type m, type a, type b ) {
	return _mm_mask_blend_ph( m, a, b );
    }

    template<int flag>
    static mask_type cmp_mask( type a, type b ) {
	return _mm_cmp_ph_mask( a, b, flag );
    }
    template<int flag>
    static vmask_type cmp_vmask( type a, type b ) {
	return int_traits::asvector( _mm_cmp_ph_mask( a, b, flag ) );
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

    static type add( type a, type b ) { return _mm_add_ph( a, b ); }
    static type add( type z, mask_type m, type a, type b ) {
	return _mm_mask_add_ph( z, m, a, b );
    }
    static type mul( type a, type b ) { return _mm_mul_ph( a, b ); }
    static type div( type a, type b ) { return _mm_div_ph( a, b ); }
    static type sub( type a, type b ) { return _mm_sub_ph( a, b ); }

    static type mul0( type a, type b ) { return _mm_mul_sh( a, b ); }

    static type rsqrt( type a ) { return _mm_rsqrt_ph( a ); }

    static type castfp( type a ) { return a; }
    static itype castint( type a ) { return _mm_castph_si128( a ); }

    static type load( const member_type *a ) {
	return _mm_load_ph( (const member_type *)a );
    }
    static type loadu( const member_type *a ) {
	return _mm_loadu_ph( (const member_type *)a );
    }
    static void store( member_type *addr, type val ) {
	_mm_store_ph( (member_type *)addr, val );
    }
    static void storeu( member_type *addr, type val ) {
	_mm_storeu_ph( (member_type *)addr, val );
    }

#if 0
    // Gather/scatter will need to use encoding or defer to sse42_2x8
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
#endif
};
#endif // __AVX512FP16__

} // namespace target

#endif //  GRAPTOR_TARGET_AVX512FP16_2fx8_H
