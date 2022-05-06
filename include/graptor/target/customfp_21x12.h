// -*- c++ -*-
#ifndef GRAPTOR_TARGET_CUSTOMFP_21x12_H
#define GRAPTOR_TARGET_CUSTOMFP_21x12_H

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

namespace target {

/***********************************************************************
 * customfp<E,M> at bit width of 21, fitting three values in a 64-bit
 * integer.
 ***********************************************************************/
#if __AVX2__
template<unsigned short E, unsigned short M>
struct customfp_21x12 {
    static_assert( customfp<E,M>::bit_size == 21,
		   "version of template class for 21-bit customfp" );
public:
    using member_type = customfp<E,M>;
    using int_type = uint64_t;
    using type = __m256i;
    using em_type = typename member_type::int_type;
    using mask_type = __mmask16;

    using enc_traits = avx2_8x4<uint64_t>;
    using cfp3_traits = customfp_21x3<E,M>;
    using fp_traits = typename vfp_traits_select<float,12*sizeof(float)>::type;

    using int_traits = typename vint_traits_select<uint32_t,sizeof(uint32_t)*12>::type;
    using itype = typename int_traits::type;
    using vmask_type = itype;
    
    static const size_t size = sizeof(type);
    static const size_t bit_size = member_type::bit_size;
    static const size_t vlen = 12;

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
    static vpair<__m256,__m256> cvt_to_float( type a ) {
	static_assert( E == 6 && M == 15, "defines dmask" );
	// Interpret 256-bit vector as 4 groups of 3 values each (with 1 unused
	// bit per group). Convert in 2 groups of 3.
	// a   = ... | 0(1b) s5(21b) s4(10b) | s4(11b) s3(21b)
	//           | 0(1b) s2(21b) s1(10b) | s1(11b) s0(21b) |

	// shift every 3k+0-th value in each 64-bit group to aligned position in
	// 2k+0-th 32-bit lane
	__m256i a0 = _mm256_slli_epi64( a, 23-M );
	// shift every 3k+1-th value in each 64-bit group to aligned position in
	// 2k+1-th 32-bit lane
	__m256i a1 = _mm256_slli_epi64( a, (E+M)-(8-E) );
	// shift every 3k+2-th value in each 64-bit group to aligned position in
	// 2k+1-th 32-bit lane
	__m256i a2 = _mm256_srli_epi64( a, 1+(8-E)-1 );

	// blend 2k+0-th lanes in a0 and 2k+1-th lanes in a1
	// sel1 mask is 8x 64 bits: 0x00000000ffffffff
	const __m256i one = _mm256_cmpeq_epi64( a, a );
	const __m256i sel1 = _mm256_srli_epi64( one, 32 );
	__m256i a01 = _mm256_blendv_epi8( a1, a0, sel1 );
	
	// mask out unneeded bits
	const __m256i mask
	    = _mm256_set1_epi32( ((uint32_t(1)<<(E+M))-1) << (23-M) );
	__m256i b01 = _mm256_and_si256( a01, mask );
	__m256i b2 = _mm256_and_si256( a2, mask );

	// set exponent bits
	__m256i e = _mm256_set1_epi32( ((uint32_t(1)<<(8-E-1))-1) << (E+23) );
	__m256i c01 = _mm256_or_si256( b01, e );
	__m256i c2 = _mm256_or_si256( b2, e );

	// _mm256_alignr_epi8
	// _mm256_permute4x64_epi64
	
	return vpair<__m256,__m256>{
	    _mm256_castsi256_ps( c01 ), _mm256_castsi256_ps( c2 ) };
    }
    static type cvt_to_cfp( vpair<__m256,__m256> as ) {
	static_assert( E == 6 && M == 15, "defines dmask" );
	__m256i a01 = _mm256_castps_si256( as.a );
	__m256i a2 = _mm256_castps_si256( as.b );

	// Mask required bits
	const __m256i one = _mm256_cmpeq_epi32( a01, a01 );
	const __m256i mlo = _mm256_srli_epi32( one, 32-(E+M) );
	const __m256i mask = _mm256_slli_epi32( mlo, (23-M) );
	const __m256i mask2 = _mm256_slli_epi64( one, 32 );
	__m256i b01 = _mm256_and_si256( a01, mask );
	__m256i b2 = _mm256_and_si256( a2, mask );

	// lanes 3k+0
	__m256i c0 = _mm256_srli_epi32( b01, (23-M) );
	// lanes 3k+1
	__m256i c1 = _mm256_srli_epi64( _mm256_and_si256( b01, mask2 ),
					(E+M)-(8-E) );
	// lanes 3k+2
	__m256i c2 = _mm256_slli_epi64( b2, 1+(8-E)-1 ); // -1: unused 64-th bit
	
	// blend 2k+0-th lanes in c0 and 2k+1-th lanes in c2
	// sel1 mask is 8x 64 bits: 0x00000000ffffffff
	const __m256i sel1 = _mm256_srli_epi64( one, 32 );
	__m256i d02 = _mm256_blendv_epi8( c2, c0, sel1 );

	// move c1 into place in d02
	__m256i e012 = _mm256_or_si256( d02, c1 );

	return e012;
    }

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
    static type load( const int_type *a ) {
	return _mm256_load_si256( reinterpret_cast<const type *>( a ) );
    }
    static type loadu( const int_type *a ) {
	return _mm256_loadu_si256( reinterpret_cast<const type *>( a ) );
    }
    static void store( int_type *addr, type val ) {
	_mm256_store_si256( reinterpret_cast<type *>( addr ), val );
    }
    static void storeu( int_type *addr, type val ) {
	_mm256_storeu_si256( reinterpret_cast<type *>( addr ), val );
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

#endif // GRAPTOR_TARGET_CUSTOMFP_21x12_H
