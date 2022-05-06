// -*- c++ -*-
#ifndef GRAPTOR_TARGET_CUSTOMFP_21x24_H
#define GRAPTOR_TARGET_CUSTOMFP_21x24_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include <iostream>

#include "graptor/target/decl.h"

#if __AVX2__
#include "graptor/target/avx512_4x16.h"
#include "graptor/target/avx512_4fx16.h"
#include "graptor/target/avx2_8x4.h"
#include "graptor/target/customfp_21x3.h"
#endif // __AVX2__

alignas(64) extern const uint8_t customfp_21x24_cvt_shuffle[96];
alignas(64) extern const uint32_t customfp_21x24_cvt_shift[24];
alignas(64) extern const uint32_t customfp_21x24_cvt_mask147[8];
alignas(64) extern const uint32_t customfp_21x24_cvt_mask258[8];

namespace target {

/***********************************************************************
 * customfp<E,M> at bit width of 21, fitting three values in a 64-bit
 * integer.
 ***********************************************************************/
#if __AVX2__
template<unsigned short E, unsigned short M>
struct customfp_21x24 {
    static_assert( customfp<E,M>::bit_size == 21,
		   "version of template class for 21-bit customfp" );
public:
    using member_type = customfp<E,M>;
    using int_type = uint64_t;

    using enc_traits = typename vint_traits_select<int_type,sizeof(int_type)*8>::type;
    using cfp3_traits = customfp_21x3<E,M>;
    using fp_traits = typename vfp_traits_select<float,24*sizeof(float)>::type;
    
    using type = typename enc_traits::type;
    using em_type = typename member_type::int_type;
    using int_traits = typename vint_traits_select<uint32_t,sizeof(uint32_t)*24>::type;
    using itype = typename int_traits::type;
    
    using vmask_type = itype;
    using mask_type = __mmask32;

    static const size_t size = sizeof(type);
    static const size_t bit_size = member_type::bit_size;
    static const size_t vlen = 24;

    static_assert( size == sizeof(int_type)*vlen/3, "sanity check" );

    static member_type lane( type a, int idx ) {
	return cfp3_traits::lane( enc_traits::lane( a, idx/3 ), idx%3 );
    }
    static member_type lane0( type a ) { return lane( a, 0 ); }
    static member_type lane1( type a ) { return lane( a, 1 ); }
    static member_type lane2( type a ) { return lane( a, 2 ); }

    static type setzero() { return enc_traits::setzero(); }
    static type set1( member_type a ) {
	return enc_traits::set1( cfp3_traits::set1( a ) );
    }

#if 0
    static type set( member_type a2, member_type a1, member_type a0 ) {
	return a2.get() << (2*bit_size) | a1.get() << bit_size | a0.get();
    }

    static type blend( vmask_type m, type a, type b ) {
	return _mm256_blendv_pd( a, b, _mm256_castsi256_pd( m ) );
    }
#endif

    // Conversion could potentially be improved by vpshufbitqmb (Ice/Tiger Lake)
    static typename fp_traits::type cvt_to_float( type a ) {
	static_assert( E == 6 && M == 15, "defines dmask" );
#if __AVX512F__
	assert( 0 && "NYI" );
	return typename fp_traits::type();
#elif __AVX2__
	static_assert( std::is_same_v<vpair<__m256i,__m256i>, type>,
		       "customfp vector type check" );
	static_assert( std::is_same_v<vpair<vpair<__m256,__m256>,__m256>,
		       typename fp_traits::type>, "float vector type check" );
	// . is a zero/unused bit
	// a.a = | . 12 11 10 | . 9 8 7 | . 6 5 4 | . 3 2 1 |
	// a.b = | . 24 23 22 | . 21 20 19 | . 18 17 16 | . 15 14 13 |
	__m256i lo = a.a;
	__m256i hi = a.b;

	const __m256i one = _mm256_cmpeq_epi32( lo, lo );
	const __m256i maskEM = _mm256_srli_epi32( one, 32-(E+M) );
	const __m256i maskE = _mm256_srli_epi32( one, 32-((8-E)-1) );
	const __m256i exp = _mm256_slli_epi32( maskE, E+23 );
	
	// Need to have a systematic decoding process that is identical
	// between the top and bottom 128 bits (due to shuffle_epi8).
	// | . 12 11 10 | . 9 8 7 | . 6 5 4 | . 3 2 1 |
	// | . 24 23 22 | . 21 20 19 | . 18 17 16 | . 15 14 13 |
	// goal:
	// z0:  | 8 7 | 6 5 | 4 3 | 2 1 |
	// via: | _ 8 7 | 6 5 _ | _ 4 3 _ | _ 2 1 | shuffle_epi8
	//      | . 9 8 7 | . 6 5 4 | . 6 5 4 | . 3 2 1 | permute 4x64
	// z1:  | 16 15 | 14 13 | 12 11 | 10 9 |
	// via: | _ 16 _ 15 _ | _ 14 13 | 12 11 _ | _ 10 _ 9 _ | shuffle epi8
	//      | . 18 17 16 | . 15 14 13 | . 12 11 10 | . 9 8 7 | permute 2x128
	// z2:  | 24 23 | 22 21 | 20 19 | 18 17 |
	// via: | 24 23 _ | 22 21 _ | _ 20 19 | 18 17 _ |  shuffle_epi8
	//      | . 24 23 22 | . 21 20 19 | . 21 20 19 | . 18 17 16 | perm 4x64
	const __m256i * shuffle_cst
	    = reinterpret_cast<const __m256i *>( customfp_21x24_cvt_shuffle );
	const __m256i * shift_cst
	    = reinterpret_cast<const __m256i *>( customfp_21x24_cvt_shift );

	// b0 = | . 9 8 7 | . 6 5 4 | . 6 5 4 | . 3 2 1 |
	__m256i b0 = _mm256_permute4x64_epi64( lo, 0b10010100 );
	__m256i c0 = _mm256_shuffle_epi8(
	    b0, _mm256_load_si256( &shuffle_cst[0] ) );
	__m256i d0 = _mm256_srlv_epi32(
	    c0, _mm256_load_si256( &shift_cst[0] ) );
	__m256i e0 = _mm256_and_si256( d0, maskEM ); // redundant?
	__m256i f0 = _mm256_slli_epi32( e0, 23-M );
	__m256i g0 = _mm256_or_si256( f0, exp );
	__m256 h0 = _mm256_castsi256_ps( g0 );
	
	// b1 = | . 18 17 16 | . 15 14 13 | . 12 11 10 | . 9 8 7 |
	__m256i b1 = _mm256_permute2x128_si256( lo, hi, 0x21 );
	__m256i c1 = _mm256_shuffle_epi8(
	    b1, _mm256_load_si256( &shuffle_cst[1] ) );
	__m256i d1 = _mm256_srlv_epi32(
	    c1, _mm256_load_si256( &shift_cst[1] ) );
	__m256i e1 = _mm256_and_si256( d1, maskEM );
	__m256i f1 = _mm256_slli_epi32( e1, 23-M );
	__m256i g1 = _mm256_or_si256( f1, exp );
	__m256 h1 = _mm256_castsi256_ps( g1 );

	// b2 = | . 24 23 22 | . 21 20 19 | . 21 20 19 | . 18 17 16 |
	__m256i b2 = _mm256_permute4x64_epi64( hi, 0b11101001 );
	__m256i c2 = _mm256_shuffle_epi8(
	    b2, _mm256_load_si256( &shuffle_cst[2] ) );
	__m256i d2 = _mm256_srlv_epi32(
	    c2, _mm256_load_si256( &shift_cst[2] ) );
	__m256i e2 = _mm256_and_si256( d2, maskEM );
	__m256i f2 = _mm256_slli_epi32( e2, 23-M );
	__m256i g2 = _mm256_or_si256( f2, exp );
	__m256 h2 = _mm256_castsi256_ps( g2 );

	return vpair<vpair<__m256,__m256>,__m256>{
	    vpair<__m256,__m256>{ h0, h1 }, h2 };
#else
	assert( 0 && "NYI" );
#endif // __AVX512F__ / __AVX2__
    }
    static type cvt_to_cfp( typename fp_traits::type as ) {
	static_assert( E == 6 && M == 15, "defines dmask" );
	
#if __AVX512F__
	assert( 0 && "NYI" );
	return type(0);
#else
	// z0:  | 8 7 | 6 5 | 4 3 | 2 1 |
	// z1:  | 16 15 | 14 13 | 12 11 | 10 9 |
	// z2:  | 24 23 | 22 21 | 20 19 | 18 17 |
	__m256i z0 = _mm256_srli_epi32( _mm256_castps_si256( as.a.a ), 23-M );
	__m256i z1 = _mm256_srli_epi32( _mm256_castps_si256( as.a.b ), 23-M );
	__m256i z2 = _mm256_srli_epi32( _mm256_castps_si256( as.b ), 23-M );

	const __m256i one = _mm256_cmpeq_epi32( z0, z0 );
	const __m256i maskEM = _mm256_srli_epi32( one, 32-(E+M) );
	__m256i y0 = _mm256_and_si256( z0, maskEM );
	__m256i y1 = _mm256_and_si256( z1, maskEM );
	__m256i y2 = _mm256_and_si256( z2, maskEM );

	// using and mask:
	// x0m: |  8  _  |  _  5  |  _  _  |  2   _ |
	// x1m: |  _  _  | 14  _  |  _ 11  |  _   _ |
	// x2m: |  _ 23  |  _  _  | 20  _  |  _  17 |
	// x0r: |  _  7  |  _  _  |  4  _  |  _   1 |
	// x1r: | 16  _  |  _ 13  |  _  _  | 10   _ |
	// x2r: |  _  _  | 22  _  |  _ 19  |  _   _ |
	// x0l: |  _  _  |  6  _  |  _  3  |  _   _ |
	// x1l: |  _ 15  |  _  _  | 12  _  |  _   9 |
	// x2l: | 24  _  |  _ 21  |  _  _  | 18   _ |
	const __m256i mask258 = _mm256_load_si256(
	    reinterpret_cast<const __m256i *>( customfp_21x24_cvt_mask258 ) );
	const __m256i mask147 = _mm256_load_si256(
	    reinterpret_cast<const __m256i *>( customfp_21x24_cvt_mask147 ) );
	const __m256i mask36 = _mm256_bslli_epi128( mask258, 4 );

	__m256i x0m = _mm256_and_si256( y0, mask258 );
	__m256i x1m = _mm256_and_si256( y1, mask36 );
	__m256i x2m = _mm256_and_si256( y2, mask147 );
	__m256i x0r = _mm256_and_si256( y0, mask147 );
	__m256i x1r = _mm256_and_si256( y1, mask258 );
	__m256i x2r = _mm256_and_si256( y2, mask36 );
	__m256i x0l = _mm256_and_si256( y0, mask36 );
	__m256i x1l = _mm256_and_si256( y1, mask147 );
	__m256i x2l = _mm256_and_si256( y2, mask258 );

	// combine in fewer words with same actions using blend_epi32 (imm)
	// w0m: |  8  _  | 14  _  | 20  _  |  2   _ | (x0m and x1m and x2m)
	// w1m: |  _ 23  |  _  5  |  _ 11  |  _  17 | (x0m and x1m and x2m)
	// w0r: |  _  7  |  _ 13  |  _ 19  |  _   1 |
	// w1r: | 16  _  | 22  _  |  4  _  | 10   _ |
	// w0l: | 24  _  |  6  _  | 12  _  | 18   _ |
	// w1l: |  _ 15  |  _ 21  |  _  3  |  _   9 |
	__m256i w0m = _mm256_blend_epi32(
	    x0m, _mm256_blend_epi32( x1m, x2m, 0b00001100 ), 0b00111100 );
	__m256i w1m = _mm256_blend_epi32(
	    x0m, _mm256_blend_epi32( x1m, x2m, 0b11110011 ), 0b11001111 );
	__m256i w0r = _mm256_blend_epi32(
	    x0r, _mm256_blend_epi32( x1r, x2r, 0b00001100 ), 0b00111100 );
	__m256i w1r = _mm256_blend_epi32(
	    x0r, _mm256_blend_epi32( x1r, x2r, 0b00110000 ), 0b11110011 );
	__m256i w0l = _mm256_blend_epi32(
	    x0l, _mm256_blend_epi32( x1l, x2l, 0b11000011 ), 0b11001111 );
	__m256i w1l = _mm256_blend_epi32(
	    x0l, _mm256_blend_epi32( x1l, x2l, 0b00110000 ), 0b11110011 );
	
	// position within 64-bit word using sl/ri
	// v0m: |  8 | 14 | 20 |  2 | srli 11
	// v1m: | 23 |  5 | 11 | 17 | slli 21
	// v0r: |  7 | 13 | 19 |  1 | no-op
	// v1r: | 16 | 22 |  4 | 10 | srli 32
	// v0l: | 24 |  6 | 12 | 18 | slli 10
	// v1l: | 15 | 21 |  3 |  9 | slli 42
	__m256i v0m = _mm256_srli_epi64( w0m, 11 );
	__m256i v1m = _mm256_slli_epi64( w1m, 21 );
	__m256i v0r = w0r;
	__m256i v1r = _mm256_srli_epi64( w1r, 32 ); // TODO: address shift in shuffle usage
	__m256i v0l = _mm256_slli_epi64( w0l, 10 );
	__m256i v1l = _mm256_slli_epi64( w1l, 42 );

	// first reduction step
	// u0mr:  |  7,8  | 13,14 | 19,20 |  1,2  | or_si256 v0m v0r
	// u1ml:  | 23,24 |  5,6  | 11,12 | 17,18 | or_si256 v1m v0l
	// v1r: | 16 | 22 |  4 | 10 |
	// v1l: | 15 | 21 |  3 |  9 |
	__m256i u0mr = _mm256_or_si256( v0m, v0r );
	__m256i u1ml = _mm256_or_si256( v1m, v0l );

	// some practicalities using blend_epi32 (imm)
	// t0mr:  |  7,8  |  5,6  | 11,12 |  1,2  | (u0mr and u1ml)
	// t1ml:  | 23,24 | 13,14 | 19,20 | 17,18 | (u0mr and u1ml)
	__m256i t0mr = _mm256_blend_epi32( u0mr, u1ml, 0b00111100 );
	__m256i t1ml = _mm256_blend_epi32( u0mr, u1ml, 0b11000011 );
	// using shuffle (cheap)
	// _1r: | 22 | 16 | 10 |  4 |
	// _1l: | 21 | 15 |  9 |  3 |
	__m256i i1r = _mm256_shuffle_epi32( v1r, 0b01001110 );
	__m256i i1l = _mm256_shuffle_epi32( v1l, 0b01001110 );
	// using blend (cheap)
	// _1r: | 22 | 15 | 10 |  3 | (_1l and _1r)
	// _1l: | 21 | 16 |  9 |  4 | (_1l and _1r)
	__m256i j1r = _mm256_blend_epi32( i1r, i1l, 0b00110011 );
	__m256i j1l = _mm256_blend_epi32( i1r, i1l, 0b11001100 );
	// permute 2x128
	// _1l: |  9 |  4 | 21 | 16 | (_1l and _1r)
	__m256i k1l = _mm256_permute2x128_si256( j1l, j1l, 0x01 );
	// reblend:
	// t0lr: |  9 |  4 | 10 |  3 |
	// t1lr: | 22 | 15 | 21 | 16 |
	__m256i t0lr = _mm256_blend_epi32( j1r, k1l, 0b11110000 );
	__m256i t1lr = _mm256_blend_epi32( j1r, k1l, 0b00001111 );

	// second reduction step
	// s0: | . 9 8 7    | . 6 5 4    | . 12 11 10 | . 3 2 1    | (t0mr,t0lr)
	// s1: | . 24 23 22 | . 15 14 13 | . 21 20 19 | . 18 17 16 | (t1ml,t1lr)
	__m256i s0 = _mm256_or_si256( t0mr, t0lr );
	__m256i s1 = _mm256_or_si256( t1ml, t1lr );

	// finally permute 4x64

	// a.a = | . 12 11 10 | . 9 8 7 | . 6 5 4 | . 3 2 1 |
	// a.b = | . 24 23 22 | . 21 20 19 | . 18 17 16 | . 15 14 13 |
	__m256i r0 = _mm256_permute4x64_epi64( s0, 0b01111000 );
	__m256i r1 = _mm256_permute4x64_epi64( s1, 0b11010010 );

	return vpair<__m256i,__m256i>{ r0, r1 };
#endif
    }

#if 0
    static type add( type a, type b ) {
	auto af = cvt_to_float( a );
	auto bf = cvt_to_float( b );
	decltype(af) cf{ af.a + bf.a, af.b + bf.b };
	using traits = avx2_4fx8<float>;
	// std::cerr << "add: af[0]: " << traits::lane( af.a, 0 )
	// << " bf[0]: " << traits::lane( bf.a, 0 )
	// << " cf[0]: " << traits::lane( cf.a, 0 )
	// << "\n";
	return cvt_to_cfp( cf );
    }

    static type add( type src, vmask_type m, type a, type b ) {
	return _mm256_blendv_pd( src, add( a, b ), _mm256_castsi256_pd( m ) );
    }
    static type add( type src, mask_type m, type a, type b ) {
	return add( src, asvector( m ), a, b );
    }
    static type mul( type a, type b ) { return _mm256_mul_pd( a, b ); }
    static type mul( type src, vmask_type m, type a, type b ) {
	return _mm256_blendv_pd( src, mul( a, b ), _mm256_castsi256_pd( m ) );
    }
    static type mul( type src, mask_type m, type a, type b ) {
	return mul( src, asvector( m ), a, b );
    }
#endif

    // This is tricky - will likely need to know offsets
    static type load( const int_type *a ) {
	return enc_traits::load( reinterpret_cast<const int_type *>( a ) );
    }
    static type loadu( const int_type *a ) {
	return enc_traits::loadu( reinterpret_cast<const int_type *>( a ) );
    }
    static void store( int_type *addr, type val ) {
	return enc_traits::store( reinterpret_cast<int_type *>( addr ),
				  val );
    }
    static void storeu( int_type *addr, type val ) {
	return enc_traits::storeu( reinterpret_cast<int_type *>( addr ),
				   val );
    }

#if 0
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
	return reduce_add( blend( asvector( mask ), setzero(), val ) );
    }
    static type
    gather( const member_type *a, itype b ) {
	return _mm256_i64gather_pd( a, b, size );
    }
    static type
    gather( const member_type *a, itype b, vmask_type mask ) {
	return _mm256_mask_i64gather_pd( setzero(), a, b,
					 _mm256_castsi256_pd( mask ), size );
    }
    static type
    gather( const member_type *a, __m128i b ) {
	return _mm256_i32gather_pd( a, b, size );
    }
    static type
    gather( const member_type *a, __m128i b, __m128i mask ) {
	vmask_type wmask = _mm256_cvtepi32_epi64( mask );
	return _mm256_mask_i32gather_pd( setzero(), a, b,
					 _mm256_castsi256_pd( wmask ), size );
    }
    static void
    scatter( const member_type *a, itype b, type val ) {
	assert( 0 && "NYI" );
    }
    static void
    scatter( const member_type *a, itype b, type val, vmask_type mask ) {
	assert( 0 && "NYI" );
    }
    static void
    scatter( const member_type *a, __m128i b, type val ) {
	assert( 0 && "NYI" );
    }
    static void
    scatter( const member_type *a, __m128i b, type val, vmask_type mask ) {
	assert( 0 && "NYI" );
    }
#endif
};
#endif // __AVX2__

} // namespace target

#endif // GRAPTOR_TARGET_CUSTOMFP_21x24_H
