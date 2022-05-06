// -*- c++ -*-
#ifndef GRAPTOR_TARGET_CUSTOMFP_21x3_H
#define GRAPTOR_TARGET_CUSTOMFP_21x3_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include <iostream>

#include "graptor/target/decl.h"

#if __SSE4_2__
#include "graptor/target/sse42_4x4.h"
#include "graptor/target/sse42_4fx4.h"
#endif // __SSE4_2__

namespace target {

/***********************************************************************
 * customfp<E,M> at bit width of 21, fitting three values in a 64-bit
 * integer.
 ***********************************************************************/
#if __SSE4_2__
template<unsigned short E, unsigned short M>
struct customfp_21x3 {
    static_assert( customfp<E,M>::bit_size == 21,
		   "version of template class for 21-bit customfp" );
public:
    using member_type = customfp<E,M>;
    using int_type = uint64_t;
    using type = uint64_t;
    using em_type = typename member_type::int_type;
    using itype = __m128i;
    using vmask_type = __m128i;
    using mask_type = __mmask8;

    using int_traits = sse42_4x4<uint32_t>;
    using fp_traits = sse42_4fx4<float>;
    
    // static const size_t size = 8;
    static const size_t bit_size = member_type::bit_size;
    static const size_t vlen = 3;

    static member_type lane( type a, int idx ) {
	constexpr type mask = (((type)1) << bit_size) - 1;
	switch( idx ) {
	case 0: return member_type( em_type( a & mask ) );
	case 1: return member_type( em_type( ( a >> bit_size ) & mask ) );
	case 2: return member_type( em_type( ( a >> (2*bit_size) ) & mask ) );
	default:
	    assert( 0 && "should not get here" );
	}
    }
    static member_type lane0( type a ) { return lane( a, 0 ); }
    static member_type lane1( type a ) { return lane( a, 1 ); }
    static member_type lane2( type a ) { return lane( a, 2 ); }

    static type setzero() { return 0; }
    static type set1( member_type a ) {
	type x = a.get();
	return x << (2*bit_size) | x << bit_size | x;
    }
    static type set( member_type a2, member_type a1, member_type a0 ) {
	return a2.get() << (2*bit_size) | a1.get() << bit_size | a0.get();
    }
    static type setlane( type a, member_type b, int idx ) {
	constexpr type mask = (((type)1) << bit_size) - 1;
	return ( a & ~( mask << ( bit_size * idx ) ) )
	    | ( type(b.get()) << ( bit_size * idx ) );
    }

#if 0
    static type blend( vmask_type m, type a, type b ) {
	return _mm256_blendv_pd( a, b, _mm256_castsi256_pd( m ) );
    }
#endif

    // Conversion could potentially be improved by vpshufbitqmb (Ice/Tiger Lake)
    static __m128 cvt_to_float( type a ) {
	static_assert( E == 6 && M == 15, "defines dmask" );
	// a   = | 0(1b) s2(21b) s1(10b) | s1(11b) s0(21b) |
	constexpr uint64_t dmask = 0x1fffff001fffff00UL;
	// b01 = | 0(3b) s1(21b) 0(8b) | 0(3b) s0(21b) 0(8b) |
	uint64_t b01 = _pdep_u64( a, dmask );
	// b2  = | 0(32b) | 0(3b) s2(21b) 0(8b) |
	uint64_t b2 = ( a >> (2*(E+M)-(23-M)) ) & ~((uint64_t(1)<<(23-M))-1);

	// t   = | 0(32b) | 0(3b) s2(21b) 0(8b) | ... | ... |
	__m128i t = _mm_insert_epi64( _mm_cvtsi64_si128( b01 ), b2, 1 );

	// s   = | xxx | fp2 | fp1 | fp0 |
	__m128i e = int_traits::set1( ((uint32_t(1)<<(8-E-1))-1) << (E+23) );
	__m128i s = _mm_or_si128( t, e );
	return _mm_castsi128_ps( s );
    }
    static type cvt_to_cfp( __m128 a ) {
	static_assert( E == 6 && M == 15, "defines dmask" );
	__m128i b = _mm_castps_si128( a );
	uint64_t c01 = _mm_extract_epi64( b, 0 );
	uint64_t c2 = _mm_extract_epi64( b, 1 );

	constexpr uint64_t mask = (uint64_t(1) << (E+M)) - 1;
	uint64_t d2 = ( c2 >> (23-M) ) & mask;
	
	constexpr uint64_t dmask = 0x1fffff001fffff00UL;
	uint64_t d01 = _pext_u64( c01, dmask );

	uint64_t d012 = ( d2 << 42 ) | d01;

	return d012;
    }

    static type add( type a, type b ) {
	return cvt_to_cfp( _mm_add_ps( cvt_to_float( a ), cvt_to_float( b ) ) );
    }

#if 0
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
    static type load( const member_type *a ) {
	return *reinterpret_cast<const type *>( a );
    }
    static type loadu( const member_type *a ) {
	return *reinterpret_cast<const type *>( a );
    }
    static void store( member_type *addr, type val ) {
	*reinterpret_cast<type *>( addr ) = val;
    }
    static void storeu( member_type *addr, type val ) {
	*reinterpret_cast<type *>( addr ) = val;
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
#endif // __SSE4_2__

} // namespace target

#endif // GRAPTOR_TARGET_CUSTOMFP_21x3_H
