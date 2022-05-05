// -*- c++ -*-
#ifndef GRAPTOR_TARGET_AVX2_4x8_H
#define GRAPTOR_TARGET_AVX2_4x8_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include "graptor/target/decl.h"
#include "graptor/target/vt_recursive.h"
#include "graptor/target/avx2_bitwise.h"
#include "graptor/target/avx512_4x16.h"
#include "graptor/target/sse42_4x4.h"
#include "graptor/target/mmx_1x8.h"

alignas(64) extern const uint32_t avx2_4x8_evenodd_intlv_epi32_vl4[8];

namespace target {

/***********************************************************************
 * AVX2 8 integers
 ***********************************************************************/
#if __AVX2__
template<typename T> // T = uint32_t
struct avx2_4x8 : public avx2_bitwise {
    static_assert( sizeof(T) == 4, 
		   "version of template class for 4-byte integers" );
public:
    using member_type = T;
    using type = __m256i;
    using vmask_type = __m256i;
    using itype = __m256i;
    using int_type = uint32_t;

    using mask_traits = mask_type_traits<8>;
    using mask_type = typename mask_traits::type;

    using mt_preferred = target::mt_vmask;

    // using half_traits = vector_type_int_traits<member_type,16>;
    using half_traits = sse42_4x4<member_type>;
    using recursive_traits = vt_recursive<member_type,4,32,half_traits>;
    using int_traits = avx2_4x8<int_type>;

    static constexpr unsigned short W = 4;
    static constexpr unsigned short B = 8 * W;
    static constexpr unsigned short vlen = 8;
    static constexpr unsigned short size = W * vlen;
    
    static member_type lane( type a, int idx ) {
	switch( idx ) {
	case 0: return (member_type) _mm256_extract_epi32( a, 0 );
	case 1: return (member_type) _mm256_extract_epi32( a, 1 );
	case 2: return (member_type) _mm256_extract_epi32( a, 2 );
	case 3: return (member_type) _mm256_extract_epi32( a, 3 );
	case 4: return (member_type) _mm256_extract_epi32( a, 4 );
	case 5: return (member_type) _mm256_extract_epi32( a, 5 );
	case 6: return (member_type) _mm256_extract_epi32( a, 6 );
	case 7: return (member_type) _mm256_extract_epi32( a, 7 );
	default:
	    assert( 0 && "should not get here" );
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

    static type setlane( type a, member_type b, int idx ) {
    	switch( idx ) {
	case 0: return _mm256_insert_epi32( a, b, 0 );
	case 1: return _mm256_insert_epi32( a, b, 1 );
	case 2: return _mm256_insert_epi32( a, b, 2 );
	case 3: return _mm256_insert_epi32( a, b, 3 );
	case 4: return _mm256_insert_epi32( a, b, 4 );
	case 5: return _mm256_insert_epi32( a, b, 5 );
	case 6: return _mm256_insert_epi32( a, b, 6 );
	case 7: return _mm256_insert_epi32( a, b, 7 );
	default:
	    assert( 0 && "should not get here" );
	}
    }

    static type setone_shr1() {
	return _mm256_srli_epi32( setone(), 1 );
    }
    static type setoneval() {
	// http://agner.org/optimize/optimizing_assembly.pdf
	type x;
	return _mm256_srli_epi32( _mm256_cmpeq_epi32( x, x ), 31 );
    }
    
    static type set1( member_type a ) { return _mm256_set1_epi32( a ); }
    static type set1inc( member_type a ) {
	return add( set1( a ), set1inc0() );
    }
    static type set1inc0() {
	return load(
	    reinterpret_cast<const member_type *>(
		&increasing_sequence_epi32[0] ) );
    }
    static type set( member_type a7, member_type a6,
		     member_type a5, member_type a4,
		     member_type a3, member_type a2,
		     member_type a1, member_type a0 ) {
	return _mm256_set_epi32( a7, a6, a5, a4, a3, a2, a1, a0 );
    }
    static type set( __m128i hi, __m128i lo ) {
	type vlo = _mm256_castsi128_si256( lo );
	type vb = _mm256_insertf128_si256( vlo, hi, 1 );
	return vb;
    }
    
    static type setl0( member_type a ) {
	// const member_type z( 0 );
	// return set( z, z, z, z, z, z, z, a );
	return _mm256_zextsi128_si256( _mm_cvtsi64_si128( (uint64_t)a ) );
    }

    // Logical mask - only test msb
    static bool is_all_false( type a ) { return is_zero( srli( a, B-1 ) ); }

    static type blend( vmask_type m, type l, type r ) {
	// see also https://stackoverflow.com/questions/40746656/howto-vblend
	// -for-32-bit-integer-or-why-is-there-no-mm256-blendv-epi32
	// for using FP blend.
	// We are assuming that the mask is all-ones in each field (not only
	// the highest bit is used), so this works without modifying the mask.
	return _mm256_blendv_epi8( l, r, m );
    }
    static type blend( mask_type m, type l, type r ) {
#if __AVX512F__ && __AVX512VL__
	return _mm256_mask_blend_epi32( m, l, r );
#else // __AVX512F__ && __AVX512VL__
	return blend( asvector( m ), l, r );
#endif // __AVX512F__ && __AVX512VL__
    }
    static type blendm( mask_type m, type l, type r ) {
	return blend( m, l, r );
    }
    static type blendm( vmask_type m, type l, type r ) {
	return blend( m, l, r );
    }
    static type blend( __m128i mask, type a, type b ) {
	return _mm256_blendv_epi8( a, b, _mm256_cvtepi16_epi32( mask ) );
    }
    static type bitblend( vmask_type m, type l, type r ) {
	if constexpr ( has_ternary ) {
	    // return ternary<0xe2>( r, m, l ); // TODO: one instr less
	    return ternary<0xca>( m, r, l );
	} else {
	    return bitwise_or( bitwise_and( m, r ),
			       bitwise_andnot( m, l ) );
	}
    }

/*
    static type logical_and( type a, type b ) { return _mm256_and_si256( a, b ); }
    static type logical_andnot( type a, type b ) { return _mm256_andnot_si256( a, b ); }
    static type logical_or( type a, type b ) { return _mm256_or_si256( a, b ); }
    static type logical_invert( type a ) { return _mm256_xor_si256( a, setone() ); }
    static type bitwise_and( type a, type b ) { return _mm256_and_si256( a, b ); }
    static type bitwise_andnot( type a, type b ) { return _mm256_andnot_si256( a, b ); }
    static type bitwise_or( type a, type b ) { return _mm256_or_si256( a, b ); }
    static type bitwise_xor( type a, type b ) { return _mm256_xor_si256( a, b ); }
    static type bitwise_xnor( type a, type b ) {
	return bitwise_xor( bitwise_invert( a ), b );
    }
    static type bitwise_invert( type a ) { return _mm256_xor_si256( a, setone() ); }
*/

    template<typename VecT2>
    static auto convert( VecT2 a ) {
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
	using Ty = typename std::make_signed<typename int_type_of_size<sizeof(member_type)>::type>::type;
	return traits2::set( (Ty)lane7(a), (Ty)lane6(a),
			     (Ty)lane5(a), (Ty)lane4(a),
			     (Ty)lane3(a), (Ty)lane2(a),
			     (Ty)lane1(a), (Ty)lane0(a) );
    }

    static type from_int( unsigned mask ) {
	return asvector( mask );
    }
    
    // template<typename T2>
    // static typename vector_type_traits<T2,sizeof(T2)*8>::vmask_type
    // asvector( vmask_type mask );

    static mask_type asmask( vmask_type a ) {
#if __AVX512VL__ && __AVX512DQ__
	return _mm256_movepi32_mask( a );
#elif __AVX512F__ && __AVX512VL__
	// The mask is necessary if we only want to pick up the highest bit
	vmask_type m = int_traits::slli( int_traits::setone(), 31 );
	auto am = _mm256_and_si256( a, m );
	return _mm256_cmpeq_epi32_mask( am, m );
#else
	return _mm256_movemask_ps( _mm256_castsi256_ps( a ) );
#endif
    }
    
    static vmask_type asvector( mask_type mask ) {
#if __AVX512F__ && __AVX512VL__
	return _mm256_mask_blend_epi32(
	    mask, int_traits::setzero(), int_traits::setone() );
#else // __AVX512F__ && __AVX512VL__
	vmask_type vmask = _mm256_set1_epi32( (int)mask );
	const itype inc = _mm256_loadu_si256(
	    reinterpret_cast<const itype *>( &increasing_sequence_epi32[1] ) );
	const itype cst = int_traits::slli(
	    int_traits::srli( int_traits::setone(), 31 ), 5 ); // broadcast 32
	const itype cnt = int_traits::sub( cst, inc );
	vmask_type sh = _mm256_sllv_epi32( vmask, cnt );
	vmask_type ex = _mm256_srai_epi32( sh, 31 ); // extend sign bit
	return ex;
#endif // __AVX512F__ && __AVX512VL__
    }


/*
    using vtraits = vector_type_traits_vl<T,8>;
    static type asvector(
	vmask_type mask // ,
	// typename std::enable_if<sizeof(T2)==sizeof(member_type)>::type * = nullptr
	) {
	return mask;
    }
*/

    static __m256 castfp( type a ) { return _mm256_castsi256_ps( a ); }

#if __AVX512F__ && __AVX512VL__
    static mask_type cmpgt( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm256_cmpgt_epi32_mask( a, b );
	else
	    return _mm256_cmpgt_epu32_mask( a, b );
    }
    static mask_type cmplt( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm256_cmplt_epi32_mask( a, b );
	else
	    return _mm256_cmplt_epu32_mask( a, b );
    }
    static mask_type cmpge( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm256_cmpge_epi32_mask( a, b );
	else
	    return _mm256_cmpge_epu32_mask( a, b );
    }
    static mask_type cmple( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm256_cmple_epi32_mask( a, b );
	else
	    return _mm256_cmple_epu32_mask( a, b );
    }
    static mask_type cmpeq( type a, type b, mt_mask ) {
	return _mm256_cmpeq_epi32_mask( a, b );
    }
    static mask_type cmpne( type a, type b, mt_mask ) {
	return _mm256_cmpneq_epi32_mask( a, b );
    }
#else
    static mask_type cmpgt( type a, type b, mt_mask ) {
	return asmask( cmpgt( a, b, mt_vmask() ) );
    }
    static mask_type cmpge( type a, type b, mt_mask ) {
	return asmask( cmpge( a, b, mt_vmask() ) );
    }
    static mask_type cmplt( type a, type b, mt_mask ) {
	return asmask( cmplt( a, b, mt_vmask() ) );
    }
    static mask_type cmple( type a, type b, mt_mask ) {
	return asmask( cmple( a, b, mt_vmask() ) );
    }
    static mask_type cmpeq( type a, type b, mt_mask ) {
	return asmask( cmpeq( a, b, mt_vmask() ) );
    }
    static mask_type cmpne( type a, type b, mt_mask ) {
	return asmask( cmpne( a, b, mt_vmask() ) );
    }
#endif
    static vmask_type cmpeq( type a, type b, mt_vmask ) {
	return _mm256_cmpeq_epi32( a, b );
    }
    static vmask_type cmpne( type a, type b, mt_vmask ) {
	return logical_invert( cmpeq( a, b, mt_vmask() ) );
    }
    static vmask_type cmpgt( type a, type b, mt_vmask ) {
	if constexpr ( std::is_signed_v<member_type> ) {
	    return _mm256_cmpgt_epi32( a, b );
	} else {
	    type one = slli( setone(), 8*W-1 );
	    type ax = add( a, one );
	    type bx = add( b, one );
	    return _mm256_cmpgt_epi32( ax, bx );
	}
    }
    static vmask_type cmpge( type a, type b, mt_vmask ) {
	if constexpr ( std::is_signed_v<member_type> ) {
	    return _mm256_or_si256( _mm256_cmpgt_epi32( a, b ),
				    _mm256_cmpeq_epi32( a, b ) );
	} else {
	    // This could be needlessly expensive for many comparisons
	    // where the top bit will never be set (e.g. VID)
	    // type ab = bitwise_xor( a, b );
	    // type flip = srli( ab, 8*W-1 );
	    // cmpgt -> xor with flip
	    type one = slli( setone(), 8*W-1 );
	    type ax = bitwise_xor( a, one );
	    type bx = bitwise_xor( b, one );
	    return _mm256_or_si256( _mm256_cmpgt_epi32( ax, bx ),
				    _mm256_cmpeq_epi32( a, b ) );
	}
    }
    static vmask_type cmplt( type a, type b, mt_vmask ) {
	return cmpgt( b, a, mt_vmask() );
    }
    static vmask_type cmple( type a, type b, mt_vmask ) {
	return cmpge( b, a, mt_vmask() );
    }

    static bool cmpeq( type a, type b, mt_bool ) {
	type e = _mm256_cmpeq_epi32( a, b );
	return _mm256_testc_si256( e, setone() );
    }

    static type add( type src, vmask_type m, type a, type b ) {
	type sum = add( a, b );
	return _mm256_blendv_epi8( src, sum, m );
    }
    static type add( type src, mask_type m, type a, type b ) {
	type sum = add( a, b );
	return _mm256_mask_blend_epi32( m, src, sum );
    }

    static type add( type a, type b ) { return _mm256_add_epi32( a, b ); }
    static type sub( type a, type b ) { return _mm256_sub_epi32( a, b ); }
    static type mul( type a, type b ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm256_mul_epi32( a, b );
	else
	    return _mm256_mul_epu32( a, b );
    }
    static vpair<type,type> divmod3( type a ) {
	// Based on
	// https://github.com/vectorclass/version2/blob/master/vectori128.h
	constexpr uint32_t d = 3;
	constexpr uint32_t L = 2; // _bit_scan_reverse(d - 1) + 1;    // ceil(log2(d))
	constexpr uint32_t L2 = 4; // uint32_t(L < 32 ? 1 << L : 0); // 2^L, overflow to 0 if L = 32
	constexpr uint32_t m = 0x55555556U; // 1 + uint32_t((uint64_t(L2 - d) << 32) / d); // multiplier
	constexpr uint32_t sh1 = 1;
	constexpr uint32_t sh2 = 1; // L - 1;
	const type multiplier = set1( m );
	// const __m128i shift1 = sse42_4x4<uint32_t>::setl0( sh1 ); // only one lane?
	// const __m128i shift2 = sse42_4x4<uint32_t>::setl0( sh2 ); // only one lane?

	// division algorithm
	type t1 = _mm256_mul_epu32( a, multiplier );
	type t2 = _mm256_srli_epi64( t1, 32 );
	type t3 = _mm256_srli_epi64( a, 32 );
	type t4 = _mm256_mul_epu32( t3, multiplier );
	type t7 = _mm256_blend_epi16( t2, t4, 0xCC );
	type t8 = _mm256_sub_epi32( a, t7 );
	// type t9 = _mm256_srl_epi32( t8, shift1 );
	type t9 = _mm256_srli_epi32( t8, sh1 );
	type t10 = _mm256_add_epi32( t7, t9 );
	// type t11 = _mm256_srl_epi32( t10, shift2 );
	type t11 = _mm256_srli_epi32( t10, sh2 );

	// modulus - note t12 != t10 as t11 drops out LSB;
	// could do t12 = t10 and ~sh2 also
	type t12 = _mm256_slli_epi32( t11, 1 );
	type t13 = _mm256_add_epi32( t11, t12 ); // t13 = 3 * t11
	type t14 = _mm256_sub_epi32( a, t13 );

	return vpair<type,type> { t11, t14 };
    }

    static type min( type a, type b ) {
	auto cmp = cmpgt( a, b, mt_vmask() );
	return blend( cmp, a, b );
    }
    static type max( type a, type b ) {
	auto cmp = cmpgt( a, b, mt_vmask() );
	return blend( cmp, b, a );
    }
    static bool cmpne( type a, type b, mt_bool ) {
	vmask_type ne = cmpne( a, b, mt_vmask() );
	return ! is_zero( ne );
    }
    static member_type reduce_setif( type val ) {
	for( short l=0; l < vlen; ++l ) {
	    member_type m = lane( val, l );
	    if( m != ~member_type(0) )
		return m;
	}
	return ~member_type(0);
    }
    static member_type reduce_add( type val ) {
	type s = _mm256_hadd_epi32( val, val );
	type t = _mm256_hadd_epi32( s, s );
	return lane0( t ) + lane4( t );
    }
    static member_type reduce_add( type val, mask_type mask ) {
	assert( 0 && "NYI" );
	return setzero(); // _mm256_reduce_add_epi64( val );
    }
    static member_type reduce_add( type val, vmask_type mask ) {
	type zval = _mm256_blendv_epi8( setzero(), val, mask );
	return reduce_add( zval );
    }
    static member_type reduce_logicalor( type val ) {
	return _mm256_movemask_epi8( val ) ? ~member_type(0) : member_type(0);
    }
    static member_type reduce_logicalor( type val, type mask ) {
	assert( 0 && "ERROR - need to only consider active lanes" );
	int v = _mm256_movemask_epi8( val );
	int m = _mm256_movemask_epi8( mask );
	return (!m | v) ? ~member_type(0) : member_type(0);
    }
    static member_type reduce_bitwiseor( type val ) {
	return half_traits::reduce_bitwiseor(
	    half_traits::bitwiseor( lower_half( val ), upper_half( val ) ) );
    }
    static member_type reduce_bitwiseor( type val, vmask_type m ) {
	return recursive_traits::reduce_bitwiseor(
	    decompose(val), int_traits::decompose(m) );
    }
    static member_type reduce_setif( type val, vmask_type m ) {
	return recursive_traits::reduce_setif(
	    decompose(val), int_traits::decompose(m) );
    }
    static member_type reduce_max( type val ) {
	return recursive_traits::reduce_max( decompose(val) );
    }
    static member_type reduce_max( type val, vmask_type m ) {
	return recursive_traits::reduce_max(
	    decompose(val), int_traits::decompose(m) );
    }
    
    static member_type reduce_min( type val ) {
	return recursive_traits::reduce_min( decompose(val) );
    }
    static member_type reduce_min( type val, vmask_type m ) {
	// using half_traits = vector_type_int_traits<member_type,sizeof(member_type)*vlen/2>;
	member_type ma = half_traits::reduce_min( lower_half( val ), lower_half( m ) );
	member_type mb = half_traits::reduce_min( upper_half( val ), upper_half( m ) );
	if( half_traits::is_zero( lower_half( m ) ) ) {
	    if( half_traits::is_zero( upper_half( m ) ) )
		return ~member_type(0);
	    return mb;
	}
	if( half_traits::is_zero( upper_half( m ) ) )
	    return ma;
	return std::min( ma, mb );
    }
    static type sllv( type a, type b ) { return _mm256_sllv_epi32( a, b ); }
    static type sllv( type a, __m128i b ) {
	return sllv( a, _mm256_cvtepi16_epi32( b ) );
    }
    static type srlv( type a, type b ) { return _mm256_srlv_epi32( a, b ); }
    static type srlv( type a, __m128i b ) {
	return srlv( a, _mm256_cvtepi16_epi32( b ) );
    }
    static type sll( type a, __m128i b ) { return _mm256_sll_epi32( a, b ); }
    static type srl( type a, __m128i b ) { return _mm256_srl_epi32( a, b ); }
    static type sll( type a, long b ) {
	return sll( a, _mm_cvtsi64_si128( b ) );
    }
    static type srl( type a, long b ) {
	return srl( a, _mm_cvtsi64_si128( b ) );
    }
    static type slli( type a, unsigned int s ) {
	    return _mm256_slli_epi32( a, s );
    }
    static type srli( type a, unsigned int s ) {
	return _mm256_srli_epi32( a, s );
    }
    static type srai( type a, unsigned int s ) {
	    return _mm256_srai_epi32( a, s );
    }

    template<typename ReturnTy = member_type>
    static auto tzcnt( type a ) {
	type zero = setzero();
	type b = _mm256_sub_epi32( zero, a );
	type c = _mm256_and_si256( a, b );
	__m256 f = _mm256_cvtepi32_ps( c ); // AVX
	type g = _mm256_castps_si256( f );
	type h = _mm256_srli_epi32( g, 23 );
	type bias = srli( setone(), 8*W-7 ); // 0x7f
	type raw = _mm256_sub_epi32( h, bias );
	type cnt = blendm( cmpeq( a, zero, mt_mask() ), raw, zero );
	if constexpr ( sizeof(ReturnTy) == W )
	    return cnt;
	else {
	    assert( 0 && "NYI" );
	    return 0;
	}
    }

    template<typename ReturnTy = member_type>
    static auto lzcnt( type a ) {
#if __AVX512VL__
	type v = _mm256_lzcnt_epi32( a );
#else
	// https://stackoverflow.com/questions/58823140/count-leading-zero-bits-for-each-element-in-avx2-vector-emulate-mm256-lzcnt-ep/58827596#58827596
	// prevent value from being rounded up to the next power of two
	type v = a;
	v = _mm256_andnot_si256(_mm256_srli_epi32(v, 8), v); // keep 8 MSB

	v = _mm256_castps_si256(_mm256_cvtepi32_ps(v)); // convert an integer to float
	v = _mm256_srli_epi32(v, 23); // shift down the exponent
	v = _mm256_subs_epu16(_mm256_set1_epi32(158), v); // undo bias
	v = _mm256_min_epi16(v, _mm256_set1_epi32(32)); // clamp at 32
#endif
	if constexpr ( sizeof(ReturnTy) == W )
	    return v;
	else {
	    assert( 0 && "NYI" );
	    return setzero();
	}
    }

    template<unsigned short PermuteVL>
    static type permute_evenodd( type a ) {
#if __AVX512VL__
	// Even/odd interleaving of the elements of a
	if constexpr ( PermuteVL == 4 ) {
	    const uint32_t * shuf = avx2_4x8_evenodd_intlv_epi32_vl4;
	    const type mask =
		_mm256_load_si256( reinterpret_cast<const type *>( shuf ) );
	    const type p = _mm256_permutevar8x32_epi32( a, mask );
	    return p;
	}
#endif // __AVX512VL__

	assert( 0 && "NYI" );
	return setzero();
    }

#if __AVX512VL__
    static constexpr bool has_ternary = true;
#else
    static constexpr bool has_ternary = false;
#endif

    template<unsigned char imm8>
    static type ternary( type a, type b, type c ) {
#if __AVX512VL__
	return _mm256_ternarylogic_epi32( a, b, c, imm8 );
#else
	assert( 0 && "NYI" );
	return setzero();
#endif
    }
    

    static member_type loads( const member_type * a, unsigned int off ) {
	return *(a+off);
    }
    static type loadu( const member_type * a, unsigned int off ) {
	return loadu( a+off );
    }
    static type loadu( const member_type * a ) {
	return _mm256_loadu_si256( (type *)a );
    }
    static type load( const member_type * a, unsigned int off ) {
	return load( a+off );
    }
    static type load( const member_type * a ) {
	return _mm256_load_si256( (type *)a );
    }
    static void store( member_type *addr, type val ) {
	_mm256_store_si256( (type *)addr, val );
    }
    static void storeu( member_type *addr, type val ) {
	_mm256_storeu_si256( (type *)addr, val );
    }

    static type ntload( const member_type * a ) {
	return _mm256_stream_load_si256( (__m256i *)a );
    }
    static void ntstore( member_type *addr, type val ) {
	_mm256_stream_si256( (__m256i *)addr, val );
    }

    template<unsigned short Scale>
    static type gather_w( const member_type *a, type b ) {
	return _mm256_i32gather_epi32( (const int *)a, b, Scale );
    }
    template<unsigned short Scale>
    static type gather_w( const member_type *a, itype b, mask_type mask ) {
#if __AVX512F__ && __AVX512VL__
	return _mm256_mmask_i32gather_epi32(
	    setzero(), mask, b, reinterpret_cast<const int *>( a ), Scale );
#else
	return _mm256_mask_i32gather_epi32(
	    setzero(), reinterpret_cast<const int *>( a ),
	    b, asvector(mask), Scale );
#endif
    }
    template<unsigned short Scale>
    static type gather_w( const member_type *a, itype b, vmask_type vmask ) {
	return _mm256_mask_i32gather_epi32( setzero(), (const int *)a, b, vmask, Scale );
    }

    static type gather( const member_type *a, itype b ) {
	return gather_w<W>( a, b );
    }
    static type gather( const member_type *a, itype b, mask_type mask ) {
	return gather_w<W>( a, b, mask );
    }
    static type gather( const member_type *a, itype b, vmask_type vmask ) {
	return gather_w<W>( a, b, vmask );
    }
    static type gather( const member_type *a, vpair<itype,itype> b,
			vpair<vmask_type,vmask_type> vmask );
    static type gather( const member_type *a, vpair<itype,itype> b );
#if !__AVX512F__
    static type gather( const member_type *a, itype b,
			vpair<vmask_type,vmask_type> vmask );
#endif


#if __AVX512VL__
    GG_INLINE
    static void scatter( member_type *a, __m512i b, type c ) {
	_mm512_i64scatter_epi32( (int *)a, b, c, W );
    }
#endif

    GG_INLINE
    static void scatter( member_type *a, vpair<itype,itype> b, type c ) {
#if __AVX512F__
	assert( 0 && "Use 512-bit scatter with mask" );
#else
	a[_mm256_extract_epi64(b.a,0)] = lane0(c);
	a[_mm256_extract_epi64(b.a,1)] = lane1(c);
	a[_mm256_extract_epi64(b.a,2)] = lane2(c);
	a[_mm256_extract_epi64(b.a,3)] = lane3(c);
	a[_mm256_extract_epi64(b.b,0)] = lane4(c);
	a[_mm256_extract_epi64(b.b,1)] = lane5(c);
	a[_mm256_extract_epi64(b.b,2)] = lane6(c);
	a[_mm256_extract_epi64(b.b,3)] = lane7(c);
#endif
    }
    
    GG_INLINE
    static void scatter( member_type *a, itype b, type c ) {
#if __AVX512F__
	assert( 0 && "Use 512-bit scatter with mask" );
#else
	a[int_traits::lane0(b)] = lane0(c);
	a[int_traits::lane1(b)] = lane1(c);
	a[int_traits::lane2(b)] = lane2(c);
	a[int_traits::lane3(b)] = lane3(c);
	a[int_traits::lane4(b)] = lane4(c);
	a[int_traits::lane5(b)] = lane5(c);
	a[int_traits::lane6(b)] = lane6(c);
	a[int_traits::lane7(b)] = lane7(c);
#endif
    }
    GG_INLINE
    static void scatter( member_type *a, itype b, type c, vmask_type mask ) {
#if __AVX512F__
	assert( 0 && "Use 512-bit scatter with mask" );
#else
	if( int_traits::lane0(mask) ) a[int_traits::lane0(b)] = lane0(c);
	if( int_traits::lane1(mask) ) a[int_traits::lane1(b)] = lane1(c);
	if( int_traits::lane2(mask) ) a[int_traits::lane2(b)] = lane2(c);
	if( int_traits::lane3(mask) ) a[int_traits::lane3(b)] = lane3(c);
	if( int_traits::lane4(mask) ) a[int_traits::lane4(b)] = lane4(c);
	if( int_traits::lane5(mask) ) a[int_traits::lane5(b)] = lane5(c);
	if( int_traits::lane6(mask) ) a[int_traits::lane6(b)] = lane6(c);
	if( int_traits::lane7(mask) ) a[int_traits::lane7(b)] = lane7(c);
#endif
    }
    static void
    scatter( member_type *a, vpair<itype,itype> b, type c, vmask_type mask ); 

    GG_INLINE
    static void scatter( member_type *a, itype b, type c, mask_type mask ) {
#if __AVX512F__
	assert( 0 && "Use 512-bit scatter with mask" );
#else
	if( mask_traits::lane0(mask) ) a[int_traits::lane0(b)] = lane0(c);
	if( mask_traits::lane1(mask) ) a[int_traits::lane1(b)] = lane1(c);
	if( mask_traits::lane2(mask) ) a[int_traits::lane2(b)] = lane2(c);
	if( mask_traits::lane3(mask) ) a[int_traits::lane3(b)] = lane3(c);
	if( mask_traits::lane4(mask) ) a[int_traits::lane4(b)] = lane4(c);
	if( mask_traits::lane5(mask) ) a[int_traits::lane5(b)] = lane5(c);
	if( mask_traits::lane6(mask) ) a[int_traits::lane6(b)] = lane6(c);
	if( mask_traits::lane7(mask) ) a[int_traits::lane7(b)] = lane7(c);
#endif
    }

#if 0
    template<_mm_hint HINT>
    GG_INLINE
    static void prefetch_gather( member_type *a, itype b ) {
	// No native support
#if __AVX512F__
	assert( 0 && "Use 512-bit prefetch gather with mask" );
#else
	// Just do one, otherwise too many instructions issued
	_mm_prefetch( &a[int_traits::lane0(b)], HINT );
#endif
    }
#endif

    class avx2_epi32_extract_degree {
	type mask, shift;
	member_type degree_bits;

    public:
	type get_mask() const { return mask; }
	avx2_epi32_extract_degree( unsigned degree_bits_,
				   unsigned degree_shift )
	    : degree_bits( degree_bits_ ) {
	    member_type smsk
		= ( (member_type(1)<<degree_bits)-1 ) << degree_shift;
	    mask = bitwise_invert( set1( smsk ) );

	    type sh0 = set1inc0();
	    type sh2 = sh0;
	    for( int i=1; i < degree_bits; ++i )
		sh2 += sh0;
	    type sh3 = set1( degree_shift );
	    shift = sub( sh3, sh2 );
	}
	member_type extract_degree( type v ) const {
	    type vs = _mm256_andnot_si256( mask, v );
	    type bits = _mm256_srlv_epi32( vs, shift );
	    // member_type degree = reduce_add( bits );
	    type s0 = _mm256_shuffle_epi32( bits, 0b00011110 );
	    type s1 = _mm256_or_si256( bits, s0 );
	    type s2 = _mm256_shuffle_epi32( s1, 0b00000001 );
	    type s3 = _mm256_or_si256( s1, s2 );
	    member_type degree = lane0( s3 ) | lane4( s3 );
	    return degree;
	}
	type extract_source( type v ) const {
	    type x = _mm256_and_si256( mask, v );
	    // Now 'sign extend' from the dropped bit
#if GRAPTOR_EXTRACT_OPT
	    // Leave the upper bits blank
	    type rx = x;
#else
#if __AVX2__
	    type lx = _mm256_slli_epi32( x, degree_bits );
	    type rx = _mm256_srli_epi32( lx, degree_bits );
#else
	    using half_type = __m128i;
	    half_type lx = lower_half( x );
	    half_type lxl = _mm_slli_epi32( lx, degree_bits );
	    half_type lxr = _mm_srai_epi32( lxl, degree_bits );
	    half_type ux = upper_half( x );
	    half_type uxl = _mm_slli_epi32( ux, degree_bits );
	    half_type uxr = _mm_srai_epi32( uxl, degree_bits );
	    // type rx = _mm256_set_m128i( uxr, lxr );
	    type rx = _mm256_castsi128_si256( lxr );
	    rx = _mm256_insertf128_si256( rx, uxr, 1 );
#endif
#endif
	    return rx;
	}
    };
#ifdef __AVX512F__
    class avx2_epi32_extract_degree8 {
	type mask;
	__m512i wmask;

    public:
	avx2_epi32_extract_degree8( unsigned degree_bits,
				    unsigned degree_shift ) {
	    assert( degree_bits == 1 && "Specialised version for 1 bit" );
	    assert( degree_bits + degree_shift == 8*sizeof(member_type) );
	    // member_type smsk
	    // = ( (member_type(1)<<degree_bits)-1 ) << degree_shift;
	    mask = setone_shr1(); // set1( ~smsk );
	    
	    wmask = avx512_4x16<member_type>::setone_shr1();
	    // _mm512_set1_epi32( ~smsk );
	}
	member_type extract_degree( type v ) const {
	    __m512i x = _mm512_castsi256_si512( v );
	    __mmask16 k = _mm512_cmpgt_epu32_mask( x, wmask );
	    member_type degree = _cvtmask16_u32( k ) & 255;
	    return degree;
	}
	type extract_source( type v ) const {
	    return _mm256_and_si256( mask, v );
	}
	type get_mask() const { return mask; }
    };
#endif // __AVX512F__
#if GRAPTOR_EXTRACT_OPT
#if __AVX512F__
    static avx2_epi32_extract_degree8
    create_extractor( unsigned degree_bits, unsigned degree_shift ) {
	return avx2_epi32_extract_degree8( degree_bits, degree_shift );
    }
#elif __AVX2__
    static avx2_epi32_extract_degree
    create_extractor( unsigned degree_bits, unsigned degree_shift ) {
	return avx2_epi32_extract_degree( degree_bits, degree_shift );
    }
#endif
#else
    static avx2_epi32_extract_degree
    create_extractor( unsigned degree_bits, unsigned degree_shift ) {
	return avx2_epi32_extract_degree( degree_bits, degree_shift );
    }
#endif
};

/*
template<>
template<>
inline
typename mmx_1x8<bool>::vmask_type
avx2_4x8<logical<4>>::asvector<bool>( vmask_type mask ) {
    assert( 0 && "Check" );
    // mask:  AAAA BBBB CCCC DDDD EEEE FFFF GGGG HHHH
    // Only interested to have 1 non-zero bit
    __m256i m0 = _mm256_srli_epi32( mask, 31 );
    // m0:  000A 000B 000C 000D 000E 000F 000G 000H
    __m256i m1 = _mm256_srli_si256( m0, 3 );
    // m1:  0000 00A0 00B0 00C0 00D0 00E0 00F0 00G0
    __m256i m2 = _mm256_or_si256( m1, m0 );
    // m2:  000A 00AB 00BC 00CD 00DE 00EF 00FG 00GH
    __m256i m3 = _mm256_srli_si256( m2, 10 );
    // m3:  0000 0000 0A00 AB00 0000 0000 DE00 EF00
    __m256i m4 = _mm256_or_si256( m3, m2 );
    // m4:  000A 00AB 0ABC ABCD 00DE 00EF DEFG EFGH
    return _mm_set_pi8( 0, 0, 0, 0, 0, 0, _mm256_extract_epi8( m4, 16 ),
			 _mm256_extract_epi8( m4, 0 ) );
// TODO: check
}
*/


#endif // __AVX2__

} // namespace target

#endif // GRAPTOR_TARGET_AVX2_4x8_H
