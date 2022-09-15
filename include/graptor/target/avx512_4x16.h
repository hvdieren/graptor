// -*- c++ -*-
#ifndef GRAPTOR_TARGET_AVX512_4x16_H
#define GRAPTOR_TARGET_AVX512_4x16_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include "graptor/target/decl.h"

#include "graptor/target/sse42_4x4.h"

alignas(64) extern const uint32_t avx512_4x16_evenodd_intlv_epi32_vl8[16];
alignas(64) extern const uint32_t avx512_4x16_evenodd_intlv_epi32_vl16[16];

namespace target {

/***********************************************************************
 * AVX512 16 4-byte integers
 ***********************************************************************/
#if __AVX512F__
template<typename T = uint32_t>
struct avx512_4x16 {
    static_assert( sizeof(T) == 4, 
		   "version of template class for 4-byte integers" );
public:
    static constexpr size_t W = 4;
    static constexpr size_t B = 8*W;
    static constexpr size_t vlen = 16;
    static constexpr size_t size = W * vlen;
    
    using member_type = T;
    using type = __m512i;
    using vmask_type = __m512i;
    using mask_traits = mask_type_traits<vlen>;
    using mask_type = typename mask_traits::type;
    using itype = __m512i;
    using int_type = uint32_t;

    using int_traits = avx512_4x16<int_type>;
    using mt_preferred = mt_mask;

    static member_type lane( type a, int idx ) {
	switch( idx ) {
	case 0: return (member_type) _mm_extract_epi32( _mm256_extracti128_si256( _mm512_extracti64x4_epi64( a, 0 ), 0 ), 0 );
	case 1: return (member_type) _mm_extract_epi32( _mm256_extracti128_si256( _mm512_extracti64x4_epi64( a, 0 ), 0 ), 1 );
	case 2: return (member_type) _mm_extract_epi32( _mm256_extracti128_si256( _mm512_extracti64x4_epi64( a, 0 ), 0 ), 2 );
	case 3: return (member_type) _mm_extract_epi32( _mm256_extracti128_si256( _mm512_extracti64x4_epi64( a, 0 ), 0 ), 3 );
	case 4: return (member_type) _mm_extract_epi32( _mm256_extracti128_si256( _mm512_extracti64x4_epi64( a, 0 ), 1 ), 0 );
	case 5: return (member_type) _mm_extract_epi32( _mm256_extracti128_si256( _mm512_extracti64x4_epi64( a, 0 ), 1 ), 1 );
	case 6: return (member_type) _mm_extract_epi32( _mm256_extracti128_si256( _mm512_extracti64x4_epi64( a, 0 ), 1 ), 2 );
	case 7: return (member_type) _mm_extract_epi32( _mm256_extracti128_si256( _mm512_extracti64x4_epi64( a, 0 ), 1 ), 3 );
	case 8: return (member_type) _mm_extract_epi32( _mm256_extracti128_si256( _mm512_extracti64x4_epi64( a, 1 ), 0 ), 0 );
	case 9: return (member_type) _mm_extract_epi32( _mm256_extracti128_si256( _mm512_extracti64x4_epi64( a, 1 ), 0 ), 1 );
	case 10: return (member_type) _mm_extract_epi32( _mm256_extracti128_si256( _mm512_extracti64x4_epi64( a, 1 ), 0 ), 2 );
	case 11: return (member_type) _mm_extract_epi32( _mm256_extracti128_si256( _mm512_extracti64x4_epi64( a, 1 ), 0 ), 3 );
	case 12: return (member_type) _mm_extract_epi32( _mm256_extracti128_si256( _mm512_extracti64x4_epi64( a, 1 ), 1 ), 0 );
	case 13: return (member_type) _mm_extract_epi32( _mm256_extracti128_si256( _mm512_extracti64x4_epi64( a, 1 ), 1 ), 1 );
	case 14: return (member_type) _mm_extract_epi32( _mm256_extracti128_si256( _mm512_extracti64x4_epi64( a, 1 ), 1 ), 2 );
	case 15: return (member_type) _mm_extract_epi32( _mm256_extracti128_si256( _mm512_extracti64x4_epi64( a, 1 ), 1 ), 3 );
	default:
	    UNREACHABLE_CASE_STATEMENT;
	}
    }
    static member_type lane0( type a ) { return lane( a, 0 ); }
    static member_type lane1( type a ) { return lane( a, 1 ); }
    static member_type lane2( type a ) { return lane( a, 2 ); }
    static member_type lane3( type a ) { return lane( a, 3 ); }
    static member_type lane4( type a ) { return lane( a, 4 ); }
    static member_type lane5( type a ) { return lane( a, 5 ); }
    static member_type lane6( type a ) { return lane( a, 6 ); }
    static member_type lane7( type a ) { return lane( a, 7 ); }
    static member_type lane8( type a ) { return lane( a, 8 ); }
    static member_type lane9( type a ) { return lane( a, 9 ); }
    static member_type lane10( type a ) { return lane( a, 10 ); }
    static member_type lane11( type a ) { return lane( a, 11 ); }
    static member_type lane12( type a ) { return lane( a, 12 ); }
    static member_type lane13( type a ) { return lane( a, 13 ); }
    static member_type lane14( type a ) { return lane( a, 14 ); }
    static member_type lane15( type a ) { return lane( a, 15 ); }

    static type setlane( type a, member_type b, int idx ) {
	switch( idx ) {
	case 0:
	case 1:
	case 2:
	case 3:
	{
	    __m128i c = _mm512_extracti32x4_epi32( a, 0 );
	    __m128i d = sse42_4x4<member_type>::setlane( c, b, idx%4 );
	    type e = _mm512_inserti32x4( a, d, 0 );
	    return e;
	}
	case 4:
	case 5:
	case 6:
	case 7:
	{
	    __m128i c = _mm512_extracti32x4_epi32( a, 1 );
	    __m128i d = sse42_4x4<member_type>::setlane( c, b, idx%4 );
	    type e = _mm512_inserti32x4( a, d, 1 );
	    return e;
	}
	case 8:
	case 9:
	case 10:
	case 11:
	{
	    __m128i c = _mm512_extracti32x4_epi32( a, 2 );
	    __m128i d = sse42_4x4<member_type>::setlane( c, b, idx%4 );
	    type e = _mm512_inserti32x4( a, d, 2 );
	    return e;
	}
	case 12:
	case 13:
	case 14:
	case 15:
	{
	    __m128i c = _mm512_extracti32x4_epi32( a, 3 );
	    __m128i d = sse42_4x4<member_type>::setlane( c, b, idx%4 );
	    type e = _mm512_inserti32x4( a, d, 3 );
	    return e;
	}
	default:
	    UNREACHABLE_CASE_STATEMENT;
	}
    }


    static __m256i lower_half( type a ) {
	return _mm512_castsi512_si256( a );
    }
    static __m256i upper_half( type a ) {
	return _mm512_extracti64x4_epi64( a, 1 );
    }

    static type setone() {
	// Recommended here:
	// https://stackoverflow.com/questions/45105164/set-all-bits-in-cpu-register-to-1-efficiently/45113467#45113467
	__m512i x;
	return _mm512_ternarylogic_epi32( x, x, x, 0xff );
    }
    static type setone_shr1() {
	return _mm512_srli_epi32( setone(), 1 );
    }
    static type setoneval() { // 0x00000001 repeated
	// http://agner.org/optimize/optimizing_assembly.pdf
	return _mm512_srli_epi32( setone(), 31 );
    }
    static type set_maskz( mask_type m, type a ) {
	return _mm512_maskz_mov_epi32( m, a );
    }
    
    static type set1( member_type a ) { return _mm512_set1_epi32( a ); }
    static itype set1inc( member_type a ) {
	return add( set1( a ), set1inc0() );
    }
    static itype set1inc0() {
	return int_traits::load(
	    static_cast<const int_type *>( &increasing_sequence_epi32[0] ) );
    }
/*
    static type set( member_type a7, member_type a6,
		     member_type a5, member_type a4,
		     member_type a3, member_type a2,
		     member_type a1, member_type a0 ) {
	return _mm512_set_epi64x( a7, a6, a5, a4, a3, a2, a1, a0 );
    }
*/
    static type setzero() { return _mm512_setzero_epi32(); }
    static type setl0( member_type a ) {
	// return (type)a;
	return _mm512_zextsi128_si512( _mm_cvtsi64_si128( (uint64_t)a ) );
    }
    static type set_pair( __m256i up, __m256i lo ) {
	return _mm512_inserti64x4( _mm512_castsi256_si512( lo ), up, 1 );
    }

    template<typename VecT2>
    static auto convert( VecT2 a ) { // TODO
	typedef vector_type_traits<
	    typename int_type_of_size<sizeof(VecT2)/vlen>::type,
	    sizeof(VecT2)> traits2;
	return set( traits2::lane7(a), traits2::lane6(a),
		    traits2::lane5(a), traits2::lane4(a),
		    traits2::lane3(a), traits2::lane2(a),
		    traits2::lane1(a), traits2::lane0(a) );
    }

    // Needs specialisations!
    // Casting here is to do a signed cast of a bitmask for logical masks
    template<typename T2>
    static auto convert_to( type a ) {
	typedef vector_type_traits<T2,sizeof(T2)*vlen> traits2;
	using Ty = typename std::make_signed<typename int_type_of_size<W>::type>::type;
	return traits2::set( (Ty)lane15(a), (Ty)lane14(a),
			     (Ty)lane13(a), (Ty)lane12(a),
			     (Ty)lane11(a), (Ty)lane10(a),
			     (Ty)lane9(a), (Ty)lane8(a),
			     (Ty)lane7(a), (Ty)lane6(a),
			     (Ty)lane5(a), (Ty)lane4(a),
			     (Ty)lane3(a), (Ty)lane2(a),
			     (Ty)lane1(a), (Ty)lane0(a) );
    }

    static type from_int( unsigned mask ) {
	auto ones = setone();
	auto zero = setzero();
	return _mm512_mask_blend_epi32( mask, zero, ones );
    }
    
    static mask_type movemask( vmask_type a ) {
#if __AVX512VL__ && __AVX512DQ__
	return _mm512_movepi32_mask( a );
#else
	vmask_type m = _mm512_srli_epi32( setone(), 1 );
	return _mm512_cmpgt_epu32_mask( a, m );
#endif
    }

    static mask_type asmask( vmask_type a ) { return movemask( a ); }
    static type asvector( mask_type mask ) {
	auto ones = setone();
	auto zero = setzero();
	return _mm512_mask_blend_epi32( mask, zero, ones );
    }

    static type blendm( mask_type m, type l, type r ) {
	return _mm512_mask_blend_epi32( m, l, r );
    }
    static type blend( mask_type m, type l, type r ) {
	return _mm512_mask_blend_epi32( m, l, r );
    }
    static type blend( vmask_type m, type l, type r ) {
	return blend( asmask( m ), l, r  );
    }
    static type bitblend( type m, type l, type r ) {
	if constexpr ( has_ternary ) {
	    // return ternary<0xe2>( r, m, l ); // TODO: one instr less
	    return ternary<0xac>( m, l, r );
	} else {
	    return bitwise_or( bitwise_and( m, r ),
			       bitwise_andnot( m, l ) );
	}
    }

    static type sra1( type a ) { return _mm512_srai_epi32( a, 1 ); }

    static type add( type a, type b ) { return _mm512_add_epi32( a, b ); }
    static type sub( type a, type b ) { return _mm512_sub_epi32( a, b ); }
    // static type mul( type a, type b ) { return _mm512_mul_epi32( a, b ); }
    // static type div( type a, type b ) { return _mm512_div_epi32( a, b ); }

    static type mulhi( type a, type b ) {
	// Multiply by halves
	type m0 = _mm512_mul_epi32( a, b );
	type as = _mm512_castps_si512(
	    _mm512_movehdup_ps( _mm512_castsi512_ps( a ) ) );
	type bs = _mm512_castps_si512(
	    _mm512_movehdup_ps( _mm512_castsi512_ps( b ) ) );
	type m1 = _mm512_mul_epi32( as, bs );
	type m0s = _mm512_castps_si512(
	    _mm512_movehdup_ps( _mm512_castsi512_ps( m0 ) ) );
	__mmask16 msk = 0xaaaa;
	return _mm512_mask_blend_epi32( msk, m0s, m1 );
    }

    static type mod( type a, type b ) {
	// Is this really a general (correct) remainder?
	// Based on https://stackoverflow.com/questions/70558346/generate-random-numbers-in-a-given-range-with-avx2-faster-than-svml-mm256-rem
	// Convert random bits into FP32 number in [ 1 .. 2 ) interval
	const type msk = srli( setone(), 7 ); // 0x7FFFFF, mantissa mask
	const type man = bitwise_and( a, msk );
	const __m512 one = _mm512_set1_ps( 1.0f );
	__m512 val = _mm512_or_ps( _mm512_castsi512_ps( man ), one );

	// Scale the number from [ 1 .. 2 ) into [ 0 .. range ),
	// the formula is ( val * range ) - range
	// where range = b
	// b > 0, so cvtepi32 / cvtepu32 should make no difference
	const __m512 rf = _mm512_cvtepi32_ps( b );
	val = _mm512_fmsub_ps( val, rf, rf );

	// Convert to integers
	// The instruction below always truncates towards 0 regardless of
	// MXCSR register.
	return _mm512_cvttps_epi32( val );
    }

    static type add( type src, mask_type m, type a, type b ) {
	return _mm512_mask_add_epi32( src, m, a, b );
    }
    static type add( type src, vmask_type m, type a, type b ) {
	return add( src, asmask( m ), a, b );
    }
    static type min( type a, type b ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm512_min_epi32( a, b );
	else
	    return _mm512_min_epu32( a, b );
    }
    static type max( type a, type b ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm512_max_epi32( a, b );
	else
	    return _mm512_max_epu32( a, b );
    }
    static type logical_andnot( type a, type b ) {
	return _mm512_andnot_si512( a, b );
    }
    static type logical_and( type a, type b ) {
	return _mm512_and_si512( a, b );
    }
    static type logical_or( type a, type b ) {
	return _mm512_or_si512( a, b );
    }
    static type bitwise_and( type a, type b ) {
	return _mm512_and_si512( a, b );
    }
    static type bitwise_andnot( type a, type b ) {
	return _mm512_andnot_si512( a, b );
    }
    static type bitwise_or( type a, type b ) {
	return _mm512_or_si512( a, b );
    }
    static type bitwise_xor( type a, type b ) {
	return _mm512_xor_si512( a, b );
    }
    static type bitwise_xnor( type a, type b ) {
	return bitwise_xor( bitwise_invert( a ), b );
    }
    static type bitwise_invert( type a ) {
	return _mm512_andnot_si512( a, setone() );
    }

    static mask_type cmpeq( type a, type b, mt_mask ) {
	return _mm512_cmpeq_epi32_mask( a, b );
    }
    static mask_type cmpne( type a, type b, mt_mask ) {
	return _mm512_cmpneq_epi32_mask( a, b );
    }
    static mask_type cmpgt( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm512_cmpgt_epi32_mask( a, b );
	else
	    return _mm512_cmpgt_epu32_mask( a, b );
    }
    static mask_type cmpge( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm512_cmpge_epi32_mask( a, b );
	else
	    return _mm512_cmpge_epu32_mask( a, b );
    }
    static mask_type cmplt( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm512_cmplt_epi32_mask( a, b );
	else
	    return _mm512_cmplt_epu32_mask( a, b );
    }
    static mask_type cmple( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm512_cmple_epi32_mask( a, b );
	else
	    return _mm512_cmple_epu32_mask( a, b );
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

    static bool cmpne( type a, type b, mt_bool ) {
	mask_type ne = cmpne( a, b, mt_mask() );
	return ! _kortestz_mask16_u8( ne, ne );
    }

    static member_type reduce_setif( type val ) {
	// Pick any, preferrably one that is not -1
	// Could also use min_epu32
	return _mm512_reduce_max_epi32( val );
    }
    static member_type reduce_setif( type val, mask_type mask ) {
	// Pick any, preferrably one that is not -1
	// Could also use min_epu32
	return _mm512_mask_reduce_max_epi32( mask, val );
    }
    static member_type reduce_add( type val ) {
	return _mm512_reduce_add_epi32( val );
    }
    static member_type reduce_add( type val, mask_type mask ) {
	return _mm512_mask_reduce_add_epi32( mask, val );
    }
    static member_type reduce_logicalor( type val ) {
	return member_type( _mm512_reduce_or_epi32( val ) );
    }
    static member_type reduce_logicalor( type val, mask_type mask ) {
	return _mm512_mask_reduce_or_epi32( mask, val );
    }
    static member_type reduce_bitwiseor( type val ) {
	return member_type( _mm512_reduce_or_epi32( val ) );
    }
    static member_type reduce_bitwiseor( type val, mask_type mask ) {
	return member_type( _mm512_mask_reduce_or_epi32( mask, val ) );
    }
    
    static member_type reduce_min( type val ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm512_reduce_min_epi32( val );
	else
	    return _mm512_reduce_min_epu32( val );
    }
    static member_type reduce_min( type val, mask_type mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm512_mask_reduce_min_epu32( mask, val );
	else
	    return _mm512_mask_reduce_min_epu32( mask, val );
    }
    static member_type reduce_max( type val ) {
	return _mm512_reduce_max_epi32( val );
    }
    static member_type reduce_max( type val, mask_type mask ) {
	return _mm512_mask_reduce_max_epi32( mask, val );
    }

    static type sllv( type a, __m256i b ) {
	return sllv( a, _mm512_cvtepi16_epi32( b ) );
    }
    static type sllv( type a, type b ) { return _mm512_sllv_epi32( a, b ); }
    static type srlv( type a, __m256i b ) {
	return srlv( a, _mm512_cvtepi16_epi32( b ) ); }
    static type srlv( type a, type b ) { return _mm512_srlv_epi32( a, b ); }
    static type sll( type a, __m128i b ) { return _mm512_sll_epi32( a, b ); }
    static type srl( type a, __m128i b ) { return _mm512_srl_epi32( a, b ); }
    static type sll( type a, long b ) {
	return _mm512_sll_epi32( a, _mm_cvtsi64_si128( b ) );
    }
    static type srl( type a, long b ) {
	return _mm512_srl_epi32( a, _mm_cvtsi64_si128( b ) );
    }
    static type slli( type a, unsigned int s ) {
	    return _mm512_slli_epi32( a, s );
    }
    static type srli( type a, unsigned int s ) {
	return _mm512_srli_epi32( a, s );
    }
    static type srai( type a, unsigned int s ) {
	    return _mm512_srai_epi32( a, s );
    }
    static type srav( type a, type s ) {
	    return _mm512_srav_epi32( a, s );
    }

    template<typename ReturnTy>
    static auto tzcnt( type a ) {
	__m512i zero = setzero();
	__m512i b = _mm512_sub_epi32( zero, a );
	__m512i c = _mm512_and_si512( a, b );
	__m512 f = _mm512_cvtepi32_ps( c ); // AVX512F
	__m512i g = _mm512_castps_si512( f );
	__m512i h = _mm512_srli_epi32( g, 23 );
	// __m512i bias = set1( 0x7f );
	__m512i bias = srli( setone(), 8*W-7 ); // 0x7f
	__m512i raw = _mm512_sub_epi32( h, bias );
	__m512i cnt = blendm( cmpeq( a, zero, mt_mask() ), raw, zero );
	if constexpr ( sizeof(ReturnTy) == W )
	    return cnt;
	else if constexpr ( sizeof(ReturnTy) == 2 ) {
	    return _mm512_cvtepi32_epi16( cnt ); // AVX512F
	} else {
	    assert( 0 && "NYI" );
	    return 0;
	}
    }

    template<typename ReturnTy>
    static auto lzcnt( type a ) {
#if __AVX512CD__
	type cnt = _mm512_lzcnt_epi32( a );
#else
	// AVX2: https://stackoverflow.com/questions/58823140/count-leading-zero-bits-for-each-element-in-avx2-vector-emulate-mm256-lzcnt-ep
	type cnt;
	assert( 0 && "NYI" );
#endif
	if constexpr ( sizeof(ReturnTy) == W )
	    return cnt;
	else if constexpr ( sizeof(ReturnTy) == 2 ) {
	    return _mm512_cvtepi32_epi16( cnt ); // AVX512F
	} else {
	    assert( 0 && "NYI" );
	    return 0;
	}
    }

    static __m512 castfp( type a ) { return _mm512_castsi512_ps( a ); }
    static type castint( type a ) { return a; }

    template<unsigned short PermVL>
    static type permute_evenodd( type a ) {
	// Even/odd interleaving of the elements of a
	const uint32_t * shuf;
/*
	if constexpr ( PermVL == 4 )
	    shuf = avx512_4x16_evenodd_intlv_epi32_vl4;
	    else */
	if constexpr ( PermVL == 8 )
	    shuf = avx512_4x16_evenodd_intlv_epi32_vl8;
	// else if constexpr ( PermVL == 16 ) -- ERROR
	    // shuf = avx512_4x16_evenodd_intlv_epi32_vl16;
	else
	    assert( 0 && "NYI" );

	const type mask =
	    _mm512_load_si512( reinterpret_cast<const type *>( shuf ) );
	const type p = _mm512_permutexvar_epi32( mask, a );
	return p;
    }

    static constexpr bool has_ternary = true;

    template<unsigned char imm8>
    static type ternary( type a, type b, type c ) {
	return _mm512_ternarylogic_epi32( a, b, c, imm8 );
    }
 
    static type load( const member_type * a ) {
	return _mm512_load_epi32( a );
    }
    static type loadu( const member_type * a ) {
	return _mm512_loadu_si512( a );
    }
    static void store( member_type * a, type val ) {
	_mm512_store_epi32( a, val );
    }
    static void storeu( member_type * a, type val ) {
	_mm512_storeu_si512( a, val );
    }

    static type ntload( const member_type * a ) {
	return _mm512_stream_load_si512( (__m512i *)a );
    }
    static void ntstore( member_type * a, type val ) {
	_mm512_stream_si512( (__m512i *)a, val );
    }
    
    template<unsigned short Scale>
    static type gather_w( const member_type *a, type b ) {
	return _mm512_i32gather_epi32( b, (const long long *)a, Scale );
    }
    template<unsigned short Scale>
    static type gather_w( const member_type *a, type b, mask_type mask ) {
	return _mm512_mask_i32gather_epi32( setzero(), mask, b, a, Scale );
    }
    template<unsigned short Scale>
    static type gather_w( const member_type *a, type b, vmask_type vmask ) {
	return gather_w<Scale>( a, b, asmask( vmask ) );
    }
    static type gather( const member_type *a, type b ) {
	return _mm512_i32gather_epi32( b, (const long long *)a, W );
    }
    static type gather( const member_type *a, type b, mask_type mask ) {
	return _mm512_mask_i32gather_epi32( setzero(), mask, b, a, W );
    }
    static type gather( const member_type *a, type b, vmask_type vmask ) {
	return gather( a, b, asmask( vmask ) );
    }
    static void scatter( member_type *a, itype b, type c ) {
	_mm512_i32scatter_epi32( (void *)a, b, c, W );
    }
    static void scatter( member_type *a, vpair<itype,itype> b, type c ) {
	_mm512_i64scatter_epi32( (void *)a, b.a, lower_half(c), W );
	_mm512_i64scatter_epi32( (void *)a, b.b, upper_half(c), W );
    }
    static void scatter( member_type *a, itype b, type c, mask_type mask ) {
	_mm512_mask_i32scatter_epi32( (void *)a, mask, b, c, W );
    }
    static void scatter( member_type *a, itype b, type c, vmask_type mask ) {
	scatter( a, b, c, asmask( mask ) );
    }
    class avx512f_epi32_extract_degree {
	type mask, shift;
	member_type degree_bits;

    public:
	avx512f_epi32_extract_degree( unsigned degree_bits_,
				      unsigned degree_shift )
	    : degree_bits( degree_bits_ ) {
	    member_type smsk
		= ( (member_type(1)<<degree_bits)-1 ) << degree_shift;
	    mask = set1( ~smsk );

	    type sh0 = set1inc0();
	    type sh2 = sh0;
	    for( int i=1; i < degree_bits; ++i )
		sh2 += sh0;
	    type sh3 = set1( degree_shift );
	    shift = sub( sh3, sh2 );
	}
	member_type extract_degree( type v ) const {
	    type vs = _mm512_andnot_si512( mask, v );
	    type bits = _mm512_srlv_epi32( vs, shift );
	    // member_type degree = reduce_add( bits );
	    /*
	    type s0 = _mm512_shuffle_epi32( bits, 0b00011110 );
	    type s1 = _mm512_or_si512( bits, s0 );
	    type s2 = _mm512_shuffle_epi32( s1, 0b00000001 );
	    type s3 = _mm512_or_si512( s1, s2 );
	    member_type degree = lane0( s3 ) | lane8( s3 );
	    */
	    member_type degree = _mm512_reduce_or_epi32( bits );
	    return degree;
	}
	type extract_source( type v ) const {
	    type x = _mm512_and_si512( mask, v );
	    // Now 'sign extend' from the dropped bit
	    type lx = _mm512_slli_epi32( x, degree_bits );
	    type rx = _mm512_srli_epi32( lx, degree_bits );
	    return rx;
	}
	type get_mask() const { return mask; }
    };
    class avx512f_epi32_extract_degree16 {
	type mask;

    public:
	avx512f_epi32_extract_degree16( unsigned degree_bits,
					unsigned degree_shift ) {
	    assert( degree_bits == 1 && "Specialised version for 1 bit" );
	    member_type smsk
		= ( (member_type(1)<<degree_bits)-1 ) << degree_shift;
	    mask = set1( ~smsk );

/*
	    type sh0 = set1inc0();
	    type sh2 = sh0;
	    for( int i=1; i < degree_bits; ++i )
		sh2 += sh0;
	    type sh3 = set1( degree_shift );
	    shift = sub( sh3, sh2 );
*/
	}
	member_type extract_degree( type v ) const {
	    // type vs = _mm512_andnot_si512( mask, v );
	    // type bits = _mm512_srlv_epi32( vs, shift );
	    // member_type degree = _mm512_reduce_or_epi32( bits );
	    // __mmask16 k = _mm512_cmpgt_epu32_mask( v, mask ); /* cmpge??*/
	    __mmask16 k = _mm512_movepi32_mask( v );
	    member_type degree = _cvtmask16_u32( k );
	    return degree;
	}
	type extract_source( type v ) const {
	    type x = _mm512_and_si512( mask, v );
	    // Now 'sign extend' from the dropped bit
	    // type lx = _mm512_slli_epi32( x, degree_bits );
	    // type rx = _mm512_srai_epi32( lx, degree_bits );
	    // return rx;
	    return x;
	}
	type get_mask() const { return mask; }
    };
#if GRAPTOR_EXTRACT_OPT
    static avx512f_epi32_extract_degree16
    create_extractor( unsigned degree_bits, unsigned degree_shift ) {
	return avx512f_epi32_extract_degree16( degree_bits, degree_shift );
    }
#else
    static avx512f_epi32_extract_degree
    create_extractor( unsigned degree_bits, unsigned degree_shift ) {
	return avx512f_epi32_extract_degree( degree_bits, degree_shift );
    }
#endif
};
#endif // __AVX512F__

} // namespace target

#endif // GRAPTOR_TARGET_AVX512_4x16_H
