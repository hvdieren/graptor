// -*- c++ -*-
#ifndef GRAPTOR_TARGET_SSE42_4x4_H
#define GRAPTOR_TARGET_SSE42_4x4_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include "graptor/target/decl.h"
#if __MMX__
#include "graptor/target/mmx_4x2.h"
#endif
#if __SSE4_2__
#include "graptor/target/sse42_bitwise.h"
#endif

alignas(64) extern const uint8_t sse42_4x4_evenodd_intlv_epi32[16];
alignas(64) extern const uint32_t mm_cstoreu_select[64];
alignas(64) extern const uint32_t increasing_sequence_epi32[16];

namespace target {

/***********************************************************************
 * SSE4.2 4 integers
 ***********************************************************************/
#if __SSE4_2__
template<typename T = uint32_t>
struct sse42_4x4 : public sse42_bitwise {
    static_assert( sizeof(T) == 4, 
		   "version of template class for 4-byte integers" );
public:
    using member_type = T;
    using int_type = uint32_t;
    using type = __m128i;
    using itype = __m128i;
    using vmask_type = __m128i;

    using mask_traits = mask_type_traits<4>;
    using mask_type = typename mask_traits::type;
    using vmask_traits = sse42_4x4<uint32_t>;

    using mt_preferred = target::mt_vmask;

    using half_traits = mmx_4x2<T>;
    using lo_half_traits = half_traits;
    using hi_half_traits = half_traits;
    using int_traits = sse42_4x4<int_type>;
    
    static constexpr size_t W = 4;
    static constexpr size_t B = 8*W;
    static constexpr size_t vlen = 4;
    static constexpr size_t size = W * vlen;

    static member_type lane_permute( type a, int idx ) {
#if __AVX__
	type vidx = _mm_cvtsi32_si128( idx );
	__m128 af = _mm_castsi128_ps( a );
	__m128 pf = _mm_permutevar_ps( af, vidx );
	type perm = _mm_castps_si128( pf );
	return _mm_extract_epi32( perm, 0 );
#else
	assert( 0 && "NYI" );
	return 0;
#endif
    }
    static member_type lane_memory( type a, int idx ) {
	// This shorthand results in compilation errors
	// (observed with gcc 10.3.0)
	// return ((member_type*)(&a))[idx];
	member_type m[vlen];
	storeu( m, a );
	return m[idx];
    }
    static member_type lane_switch( type a, int idx ) {
	switch( idx ) {
	case 0: return (member_type) _mm_cvtsi128_si64( a );
	case 1: return (member_type) ( _mm_cvtsi128_si64( a ) >> 32 );
	case 2: return (member_type) _mm_extract_epi32( a, 2 );
	case 3: return (member_type) _mm_extract_epi32( a, 3 );
	default:
	    assert( 0 && "should not get here" );
	}
    }
    static member_type lane( type a, int idx ) {
	return lane_memory( a, idx );
    }
    static member_type lane0( type a ) { return _mm_cvtsi128_si64( a ); }
    static member_type lane1( type a ) { return lane( a, 1 ); }
    static member_type lane2( type a ) { return lane( a, 2 ); }
    static member_type lane3( type a ) { return lane( a, 3 ); }

    template<typename index_type>
    static type setlane( type a, member_type b, index_type idx ) {
	// requires literal constant
	if( idx == 0 )
	    return _mm_insert_epi32( a, b, 0 );
	else if( idx == 1 )
	    return _mm_insert_epi32( a, b, 1 );
	else if( idx == 2 )
	    return _mm_insert_epi32( a, b, 2 );
	else if( idx == 3 )
	    return _mm_insert_epi32( a, b, 3 );
	else
	    assert( 0 && "should not get here" );
	/*
	auto c = setl0( b );
	auto d = bslli( c, idx * W ); 
	auto e = setl0( 0xffffffffU );
	auto f = bslli( e, idx * W );
	auto g = blend( f, a, d );
	return g;
	*/
    }

    static type setone_shr1() {
	auto one = setone();
	return _mm_srli_epi32( one, 1 );
    }
    static type setoneval() {
	// http://agner.org/optimize/optimizing_assembly.pdf
	type x;
	return _mm_srli_epi32( _mm_cmpeq_epi32( x, x ), 31 );
    }
    
    static type set1( member_type a ) { return _mm_set1_epi32( a ); }
    static type set1inc( member_type a ) {
	return add( set1( a ), set1inc0() );
    }
    static type set1inc0() {
	return load(
	    static_cast<const member_type *>( &increasing_sequence_epi32[0] ) );
    }
    static type set( member_type a3, member_type a2,
		     member_type a1, member_type a0 ) {
	return _mm_set_epi32( a3, a2, a1, a0 );
    }
    static type setr( member_type a0, member_type a1,
		     member_type a2, member_type a3 ) {
	return _mm_set_epi32( a3, a2, a1, a0 );
    }
    static type setl0( member_type a ) {
	return _mm_cvtsi32_si128( a );
    }

    static type blendm( vmask_type m, type l, type r ) {
	// see also https://stackoverflow.com/questions/40746656/howto-vblend
	// -for-32-bit-integer-or-why-is-there-no-mm256-blendv-epi32
	// for using FP blend.
	// We are assuming that the mask is all-ones in each field (not only
	// the highest bit is used), so this works without modifying the mask.
	return _mm_blendv_epi8( l, r, m );
    }
    static type blendm( mask_type m, type l, type r ) {
	return blendm( asvector( m ), l, r ); 
    }

    static type bitblend( vmask_type m, type l, type r ) {
	if constexpr ( has_ternary ) {
	    return ternary<0xac>( m, l, r );
	} else {
	    return bitwise_or( bitwise_and( m, r ),
			       bitwise_andnot( m, l ) );
	}
    }

#if 0
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
#endif

    static type from_int( unsigned mask ) {
	return asvector( mask );
    }
    
    static mask_type movemask( vmask_type a ) {
#if __AVX512VL__ && __AVX512DQ__
	return _mm_movepi32_mask( a );
#else
	return _mm_movemask_ps( _mm_castsi128_ps( a ) );
#endif
    }

    static type asvector( mask_type mask ) {
	// Need to work with 8 32-bit integers as there is no 64-bit srai
	// in AVX2. Luckily, this works just as well.
#if __AVX2__
	// In AVX etc, blend requires a constant mask, so does not apply
	type vmask = _mm_set1_epi32( (int)mask );
	const __m128i cnt = _mm_set_epi32( 28, 29, 30, 31 );
	type vmask2 = _mm_sllv_epi32( vmask, cnt );
	return _mm_srai_epi32( vmask2, 31 );
#else
	// Use lookup table
	assert( 0 && "Looks like LUT is wrong - 16-bit fields recorded" );
	return _mm_load_si128(
	    (const type*)&movemask_lut_epi32[4*(0xf & (int)mask)] );
#endif
    }
    // template<typename T2>
    // static typename vector_type_traits<T2,sizeof(T2)*8>::vmask_type
    // asvector( vmask_type mask );

/*
    using vtraits = vector_type_traits_vl<T,8>;
    static type asvector(
	vmask_type mask // ,
	// typename std::enable_if<sizeof(T2)==sizeof(member_type)>::type * = nullptr
	) {
	return mask;
    }
*/

    static mask_type asmask( vmask_type mask ) {
#if __AVX512F__
	__m512i wmask = _mm512_castsi128_si512( mask );
	__m512i zero = _mm512_setzero_si512();
	return (mask_type)_mm512_cmpneq_epi32_mask( wmask, zero );
#else
	// 2 cycle latency, 1 CPI on SKL, delivers in GPR
	return _mm_movemask_ps( _mm_castsi128_ps( mask ) );
#endif
    }

    static vmask_type cmpeq( type a, type b, mt_vmask ) {
	return _mm_cmpeq_epi32( a, b );
    }
    static vmask_type cmpne( type a, type b, mt_vmask ) {
	return logical_invert( _mm_cmpeq_epi32( a, b ) );
    }
    static vmask_type cmpne( vmask_type m, type a, type b, mt_vmask ) {
	return logical_andnot( _mm_cmpeq_epi32( a, b ), m );
    }
    static vmask_type cmpgt( type a, type b, mt_vmask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm_cmpgt_epi32( a, b );
	else {
	    type one = slli( setone(), 8*W-1 );
	    type ax = bitwise_xor( a, one );
	    type bx = bitwise_xor( b, one );
	    return _mm_cmpgt_epi32( ax, bx );
	}
    }
    static vmask_type cmpge( type a, type b, mt_vmask ) {
	return logical_or( cmpgt( a, b, mt_vmask() ),
			   cmpeq( a, b, mt_vmask() ) );
    }
    static vmask_type cmplt( type a, type b, mt_vmask ) {
	return cmpgt( b, a, mt_vmask() );
    }
    static vmask_type cmple( type a, type b, mt_vmask ) {
	return cmpge( b, a, mt_vmask() );
    }
    static vmask_type cmpeq( vmask_type m, type a, type b, mt_vmask ) {
	return vmask_traits::logical_and( m, cmpeq( a, b, mt_vmask() ) );
    }
    static vmask_type cmplt( vmask_type m, type a, type b, mt_vmask ) {
	return vmask_traits::logical_and( m, cmplt( a, b, mt_vmask() ) );
    }
#if __AVX512VL__
    static mask_type cmpeq( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm_cmpeq_epi32_mask( a, b );
	else
	    return _mm_cmpeq_epu32_mask( a, b );
    }
    static mask_type cmpne( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm_cmpneq_epi32_mask( a, b );
	else
	    return _mm_cmpneq_epu32_mask( a, b );
    }
    static mask_type cmpgt( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm_cmpgt_epi32_mask( a, b );
	else
	    return _mm_cmpgt_epu32_mask( a, b );
    }
    static mask_type cmpge( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm_cmpge_epi32_mask( a, b );
	else
	    return _mm_cmpge_epu32_mask( a, b );
    }
    static mask_type cmplt( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm_cmplt_epi32_mask( a, b );
	else
	    return _mm_cmplt_epu32_mask( a, b );
    }
    static mask_type cmple( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm_cmple_epi32_mask( a, b );
	else
	    return _mm_cmple_epu32_mask( a, b );
    }
#else
    static mask_type cmpeq( type a, type b, mt_mask ) {
	return asmask( cmpeq( a, b, mt_vmask() ) );
    }
    static mask_type cmpne( type a, type b, mt_mask ) {
	return mask_traits::logical_invert(
	    asmask( cmpeq( a, b, mt_vmask() ) ) );
    }
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
#endif

    static bool cmpne( type a, type b, mt_bool ) {
	type m = cmpne( a, b, mt_vmask() );
	int z = _mm_testz_si128( m, m ); // z is 1 if m == 0
	return (bool)z;
    }

    static type add( type src, vmask_type m, type a, type b ) {
	return _mm_blendv_epi8( src, add( a, b ), m );
    }

    static type add( type a, type b ) { return _mm_add_epi32( a, b ); }
    static type adds( type a, type b ) {
	if constexpr ( std::is_signed_v<member_type> ) {
	    // Based on: https://stackoverflow.com/questions/29498824/add-saturate-32-bit-signed-ints-intrinsics
	    const __m128i int_max = _mm_set1_epi32( INT32_MAX );

	    // normal result (possibly wraps around)
	    const __m128i res = _mm_add_epi32( a, b );

	    // If result saturates, it has the same sign as both a and b
	    // shift sign to lowest bit
	    const __m128i sign_bit = _mm_srli_epi32( a, 31 );

#if defined(__AVX512VL__)
	    const __m128i overflow = _mm_ternarylogic_epi32( a, b, res, 0x42);
#else
	    const __m128i sign_xor = _mm_xor_si128( a, b );
	    const __m128i overflow =
		_mm_andnot_si128( sign_xor, _mm_xor_si128( a, res ) );
#endif

#if defined(__AVX512DQ__) && defined(__AVX512VL__)
	    return _mm_mask_add_epi32( res, _mm_movepi32_mask( overflow ),
				       int_max, sign_bit );
#else
	    const __m128i saturated = _mm_add_epi32( int_max, sign_bit );

#if defined(__SSE4_1__)
	    return
		_mm_castps_si128(
		    _mm_blendv_ps(
			_mm_castsi128_ps( res ),
			_mm_castsi128_ps( saturated ),
			_mm_castsi128_ps( overflow )
			)
		    );
#else
	    const __m128i overflow_mask = _mm_srai_epi32( overflow, 31 );
	    return
		_mm_or_si128(
		    _mm_and_si128( overflow_mask, saturated ),
		    _mm_andnot_si128( overflow_mask, res )
		    );
#endif
#endif
	} else {
	    assert( 0 && "NYI" );
	}
    }
    static type sub( type a, type b ) { return _mm_sub_epi32( a, b ); }
    static type mul( type a, type b ) { return _mm_mullo_epi32( a, b ); }

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
	type t1 = _mm_mul_epu32( a, multiplier );
	type t2 = _mm_srli_epi64( t1, 32 );
	type t3 = _mm_srli_epi64( a, 32 );
	type t4 = _mm_mul_epu32( t3, multiplier );
	type t7 = _mm_blend_epi16( t2, t4, 0xCC );
	type t8 = _mm_sub_epi32( a, t7 );
	// type t9 = _mm_srl_epi32( t8, shift1 );
	type t9 = _mm_srli_epi32( t8, sh1 );
	type t10 = _mm_add_epi32( t7, t9 );
	// type t11 = _mm_srl_epi32( t10, shift2 );
	type t11 = _mm_srli_epi32( t10, sh2 );

	// modulus - note t12 != t10 as t11 drops out LSB;
	// could do t12 = t10 and ~sh2 also
	type t12 = _mm_slli_epi32( t11, 1 );
	type t13 = _mm_add_epi32( t11, t12 ); // t13 = 3 * t11
	type t14 = _mm_sub_epi32( a, t13 );

	return vpair<type,type> { t11, t14 };
    }

    static type min( type a, type b ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm_min_epi32( a, b );
	else
	    return _mm_min_epu32( a, b );
    }
    static type logicalor_bool( type &a, type b ) {
	// cmp is the update mask. Important here is that if this mask is
	// not used by the caller, the compiler will be able to remove its
	// calculation from the code.
	auto cmp = _mm_andnot_si128(
	    _mm_cmpeq_epi32( b, setzero() ),
	    _mm_cmpeq_epi32( a, setzero() ) );
	// auto ones = set1(~member_type(0));
	// auto ones = setone();
	// a = _mm256_blendv_epi8( a, ones, cmp );
	a = _mm_or_si128( a, b ); // do not use cmp
	return cmp;
    }
    static type max( type a, type b ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm_max_epi32( a, b );
	else
	    return _mm_max_epu32( a, b );
    }
    static type blend( vmask_type mask, type a, type b ) {
	return _mm_blendv_epi8( a, b, mask );
    }
    static type blend( mask_type mask, type a, type b ) {
	return _mm_blendv_epi8( a, b, asvector( mask ) );
    }
    static member_type reduce_setif( type val ) {
	return lane0( val );
    }
    static member_type reduce_setif( type val, vmask_type m ) {
	if( int_traits::lane0( m ) )
	    return lane0( val );
	else if( int_traits::lane1( m ) )
	    return lane1( val );
	else if( int_traits::lane2( m ) )
	    return lane2( val );
	else if( int_traits::lane3( m ) )
	    return lane3( val );
	else
	    return std::numeric_limits<member_type>::max();
    }
    static member_type reduce_add( type val ) {
	type s = _mm_hadd_epi32( val, val );
	type t = _mm_hadd_epi32( s, s );
	return lane0( t );
    }
    static member_type reduce_add( type val, mask_type mask ) {
	assert( 0 && "NYI" );
	return setzero();
    }
    static member_type reduce_add( type val, vmask_type mask ) {
	// type zval = _mm_blendv_epi8( setzero(), val, mask );
	// return reduce_add( zval );
	// First filter out zeros, then add up all values
	type x = _mm_blendv_epi8( setzero(), val, mask );
	x = _mm_hadd_epi32( x, x );
	x = _mm_hadd_epi32( x, x );
	return lane0( x );
/*
	member_type s = 0;
	if( lane3( mask ) ) s += lane3( val );
	if( lane2( mask ) ) s += lane2( val );
	if( lane1( mask ) ) s += lane1( val );
	if( lane0( mask ) ) s += lane0( val );
	return s;
*/
    }
    static member_type reduce_logicalor( type val ) {
	return _mm_movemask_epi8( val ) ? ~member_type(0) : member_type(0);
    }
    static member_type reduce_logicalor( type val, type mask ) {
	int v = _mm_movemask_epi8( val );
	int m = _mm_movemask_epi8( mask );
	return (!m | v) ? ~member_type(0) : member_type(0);
    }
    static member_type reduce_bitwiseor( type val ) {
	uint64_t a = lower_half( val ) | upper_half( val );
	uint32_t b = a;
	uint32_t c = a >> 32;
	return b | c;
    }
    static member_type reduce_bitwiseor( type val, vmask_type m ) {
	member_type r = 0;
	if( int_traits::lane0( m ) )
	    r |= lane0( val );
	if( int_traits::lane1( m ) )
	    r |= lane1( val );
	if( int_traits::lane2( m ) )
	    r |= lane2( val );
	if( int_traits::lane3( m ) )
	    r |= lane3( val );
	return r;
    }
    static type bitwiseor( type a, type b ) {
	return _mm_or_si128( a, b );
    }
    
    static member_type reduce_max( type val ) {
	type sh = _mm_bsrli_si128( val, 8 );
	type max = _mm_max_epi32( val, sh );
	return std::max( lane0( max ), lane1( max ) );
    }
    static member_type reduce_min( type val ) {
	type sh = _mm_bsrli_si128( val, 8 );
	type min = _mm_min_epi32( val, sh );
	return std::min( lane0( min ), lane1( min ) );
/*
	auto rot = _mm256_permute4x64_epi64( val, 0b1110 ); // all in lo128 bits
	auto cmp = _mm256_cmpgt_epi64( val, rot );
	auto sel = _mm256_blendv_epi8( val, rot, cmp );
	auto vpr = _mm256_extracti128_si256( sel, 0 );
	auto vsh = _mm_srli_si128( vpr, 8 );
	auto cm2 = _mm_cmpgt_epi64( vpr, vsh );
	auto res = _mm_blendv_epi8( vpr, vsh, cm2 );
	return _mm_extract_epi64( res, 0 );
*/
	// assert( 0 && "TODO - optimize" );
/*
	return std::min( std::min( lane0(val), lane1(val) ),
			 std::min( lane2(val), lane3(val) ),
			 std::min( lane4(val), lane5(val) ),
			 std::min( lane6(val), lane7(val) ) );
*/
    }
    static member_type reduce_min( type val, vmask_type m ) {
	member_type r = std::numeric_limits<member_type>::max();
	if( int_traits::lane0( m ) && lane0( val ) < r )
	    r = lane0( val );
	if( int_traits::lane1( m ) && lane1( val ) < r )
	    r = lane1( val );
	if( int_traits::lane2( m ) && lane2( val ) < r )
	    r = lane2( val );
	if( int_traits::lane3( m ) && lane3( val ) < r )
	    r = lane3( val );
	return r;
    }
    static member_type reduce_max( type val, vmask_type m ) {
	member_type r = std::numeric_limits<member_type>::min();
	if( int_traits::lane0( m ) && lane0( val ) > r )
	    r = lane0( val );
	if( int_traits::lane1( m ) && lane1( val ) > r )
	    r = lane1( val );
	if( int_traits::lane2( m ) && lane2( val ) > r )
	    r = lane2( val );
	if( int_traits::lane3( m ) && lane3( val ) > r )
	    r = lane3( val );
	return r;
    }

    static type sllv( type a, type b ) { return _mm_sllv_epi32( a, b ); }
    static type srlv( type a, type b ) { return _mm_srlv_epi32( a, b ); }
    static type sll( type a, __m128i b ) { return _mm_sll_epi32( a, b ); }
    static type srl( type a, __m128i b ) { return _mm_srl_epi32( a, b ); }
    static type sll( type a, long b ) {
	return _mm_sll_epi32( a, _mm_cvtsi64_si128( b ) );
    }
    static type srl( type a, long b ) {
	return _mm_srl_epi32( a, _mm_cvtsi64_si128( b ) );
    }
    static type slli( type a, unsigned int s ) {
	    return _mm_slli_epi32( a, s );
    }
    static type srli( type a, unsigned int s ) {
	return _mm_srli_epi32( a, s );
    }
    static type srai( type a, unsigned int s ) {
	    return _mm_srai_epi32( a, s );
    }

    static type shuffle( type a, unsigned int p ) {
	return _mm_shuffle_epi32( a, p );
    }

    static auto castfp( type a ) { return _mm_castsi128_ps( a ); }
    static type castint( type a ) { return a; }

    template<typename ReturnTy = member_type>
    static auto lzcnt( type a ) {
#if __AVX512VL__
	type v = _mm_lzcnt_epi32( a );
#else
	// https://stackoverflow.com/questions/58823140/count-leading-zero-bits-for-each-element-in-avx2-vector-emulate-mm256-lzcnt-ep/58827596#58827596
	// prevent value from being rounded up to the next power of two
	type v = a;
	v = _mm_andnot_si128(_mm_srli_epi32(v, 8), v); // keep 8 MSB

	v = _mm_castps_si128(_mm_cvtepi32_ps(v)); // convert an integer to float
	v = _mm_srli_epi32(v, 23); // shift down the exponent
	v = _mm_subs_epu16(_mm_set1_epi32(158), v); // undo bias
	const type c32 = slli( setoneval(), 5 );
	v = _mm_min_epi16(v, c32); // clamp at 32
#endif
	if constexpr ( sizeof(ReturnTy) == W )
	    return v;
	else {
	    assert( 0 && "NYI" );
	    return setzero();
	}
    }

    static type permute_evenodd( type a ) {
	// Even/odd interleaving of the elements of a
	const type * shuf
	    = reinterpret_cast<const type *>( sse42_4x4_evenodd_intlv_epi32 );
	return _mm_shuffle_epi8( a, _mm_load_si128( shuf ) );
    }

    static type load( const member_type *a ) {
	return _mm_load_si128( (const type *)a );
    }
    static type loadu( const member_type *a ) {
	return _mm_loadu_si128( (const type *)a );
    }
    static void store( member_type *addr, type val ) {
	_mm_store_si128( (type *)addr, val );
    }
    static void storeu( member_type *addr, type val ) {
	_mm_storeu_si128( (type *)addr, val );
    }

    static type ntload( member_type *addr ) {
	return _mm_stream_load_si128( (type *)addr );
    }
    static void ntstore( member_type *addr, type val ) {
	_mm_stream_si128( (type *)addr, val );
    }

    static member_type * cstoreu_p( member_type *addr, mask_type m, type val ) {
	uint32_t im = ((uint32_t)m) & 0xf;
	type select = load( &mm_cstoreu_select[4*im] );
	type compress = _mm_castps_si128(
	    _mm_permutevar_ps( _mm_castsi128_ps( val ), select ) );
	storeu( addr, compress );
	return addr + _popcnt32( im );

#if 0
	if( mask_traits::is_zero( m ) )
	    return addr;
	if( mask_traits::is_ones( m ) ) {
	    storeu( addr, val );
	    return addr + 4;
	}
	
	uint32_t i = m;
	uint64_t l01 = _mm_cvtsi128_si64( val );
	if( ( i & 3 ) == 3 ) {
	    *reinterpret_cast<uint64_t*>( addr ) = l01;
	    addr += 2;
	} else if( i & 1 )
	    *addr++ = (member_type)l01;
	else if( i & 2 )
	    *addr++ = (member_type)( l01 >> 32 );

	uint64_t l23 = _mm_cvtsi128_si64( _mm_bsrli_si128( val, 8 ) );
	if( ( i & 12 ) == 12 ) {
	    *reinterpret_cast<uint64_t*>( addr ) = l23;
	    addr += 2;
	} else if( i & 4 )
	    *addr++ = (member_type)l23;
	else if( i & 8 )
	    *addr++ = (member_type)( l23 >> 32 );
/*
	if( i & 2 )
	    *addr++ = lane( val, 1 );
	if( i & 4 )
	    *addr++ = lane( val, 2 );
	if( i & 8 )
	    *addr++ = lane( val, 3 );
	    */
	return addr;
#endif
    }

    template<unsigned short Scale>
    static type
    gather_w( const member_type * a, itype b ) {
#if __AVX2__
	return _mm_i32gather_epi32( (const int *)a, b, Scale );
#else
	const char * p = reinterpret_cast<const char *>( a );
	return set(
	    reinterpret_cast<const member_type *>(
		p+Scale*int_traits::lane3(b) ),
	    reinterpret_cast<const member_type *>(
		p+Scale*int_traits::lane2(b) ),
	    reinterpret_cast<const member_type *>(
		p+Scale*int_traits::lane1(b) ),
	    reinterpret_cast<const member_type *>(
		p+Scale*int_traits::lane0(b) )
	    );
#endif
    }
    template<unsigned short Scale>
    static type
    gather_w( const member_type * a, itype b, itype vmask ) {
#if __AVX2__
	return _mm_mask_i32gather_epi32( setzero(), (const int *)a, b, vmask,
					 Scale );
#else
	const char * p = reinterpret_cast<const char *>( a );
	return set(
	    int_traits::lane3(vmask)
	    ? reinterpret_cast<const member_type *>(
		p+Scale*int_traits::lane3(b) ) : member_type(0),
	    int_traits::lane2(vmask)
	    ? reinterpret_cast<const member_type *>(
		p+Scale*int_traits::lane2(b) ) : member_type(0),
	    int_traits::lane1(vmask)
	    ? reinterpret_cast<const member_type *>(
		p+Scale*int_traits::lane1(b) ) : member_type(0),
	    int_traits::lane0(vmask)
	    ? reinterpret_cast<const member_type *>(
		p+Scale*int_traits::lane0(b) ) : member_type(0)
	    );
#endif
    }
    
#if __AVX2__
    template<unsigned short Scale>
    static type
    gather_w( const member_type * a, __m256i b, __m256i vmask );

    template<unsigned short Scale>
    static type
    gather_w( const member_type * a, __m256i b ) {
	return _mm256_i64gather_epi32( (const int *)a, b, Scale );
    }
#endif
    
    static type
    gather( const member_type *a, itype b ) {
#if __AVX2__
	return _mm_i32gather_epi32( (const int *)a, b, W );
#else
	return set( a[int_traits::lane3(b)], a[int_traits::lane2(b)],
		    a[int_traits::lane1(b)], a[int_traits::lane0(b)] );
#endif
    }
    static type
    gather( const member_type *a, itype b, mask_type mask ) {
	// return _mm_mask_i32gather_epi32( setzero(), a, b, asvector(mask), size );
	assert( 0 && "NYI" );
	return setzero();
    }
    static type
    gather( const member_type *a, itype b, vmask_type vmask ) {
#if __AVX2__
	return _mm_mask_i32gather_epi32( setzero(), (const int *)a, b, vmask, W );
#else
	return set(
	    int_traits::lane3(vmask) ? a[int_traits::lane3(b)] : member_type(0),
	    int_traits::lane2(vmask) ? a[int_traits::lane2(b)] : member_type(0),
	    int_traits::lane1(vmask) ? a[int_traits::lane1(b)] : member_type(0),
	    int_traits::lane0(vmask) ? a[int_traits::lane0(b)] : member_type(0)
	    );
#endif
    }
    static type
    gather( itype z, const member_type *a, itype b, vmask_type vmask ) {
#if __AVX2__
	return _mm_mask_i32gather_epi32( z, (const int *)a, b, vmask, W );
#else
	return set(
	    int_traits::lane3(vmask) ? a[int_traits::lane3(b)] : lane3(z),
	    int_traits::lane2(vmask) ? a[int_traits::lane2(b)] : lane2(z),
	    int_traits::lane1(vmask) ? a[int_traits::lane1(b)] : lane1(z),
	    int_traits::lane0(vmask) ? a[int_traits::lane0(b)] : lane0(z)
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
#if 0
    static member_type extract_degree( type v, unsigned degree_bits,
				       unsigned degree_shift ) {
	member_type smsk = ( (member_type(1)<<degree_bits)-1 ) << degree_shift;
	type msk = set1( smsk );
	type vs = _mm_and_si128( v, msk );

	member_type b0 = lane0( vs ) >> ( degree_shift - 0 * degree_bits );
	member_type b1 = lane1( vs ) >> ( degree_shift - 1 * degree_bits );
	member_type b2 = lane2( vs ) >> ( degree_shift - 2 * degree_bits );
	member_type b3 = lane3( vs ) >> ( degree_shift - 3 * degree_bits );

	return ( b0 | b1 ) | ( b2 | b3 );
    }
    static type extract_source( type v, unsigned degree_bits,
				unsigned degree_shift ) {
	// Written to reuse intermediate values from extract_degree()
	member_type smsk = ( (member_type(1)<<degree_bits)-1 ) << degree_shift;
	type msk = set1( smsk );
	type x = _mm_andnot_si128( msk, v );
	// Now 'sign extend' from the dropped bit
	type lx = _mm_slli_epi32( x, degree_bits );
	type rx = _mm_srai_epi32( lx, degree_bits );
	return rx;
    }
#endif
    class sse42_epi32_extract_degree {
	type mask, shift;
	member_type degree_bits;

    public:
	sse42_epi32_extract_degree( unsigned degree_bits_,
				    unsigned degree_shift )
	    : degree_bits( degree_bits_ ) {
	    member_type smsk
		= ( (member_type(1)<<degree_bits)-1 ) << degree_shift;
	    mask = set1( smsk );

	    type sh0 = set1inc0();
	    type sh2 = sh0;
	    for( int i=1; i < degree_bits; ++i )
		sh2 += sh0;
	    type sh3 = set1( degree_shift );
	    shift = sub( sh3, sh2 );
	}
	member_type extract_degree( type v ) const {
	    type vs = _mm_and_si128( v, mask );
	    type bits = _mm_srlv_epi32( vs, shift );
	    // member_type degree = reduce_add( bits );
	    type s0 = _mm_shuffle_epi32( bits, 0b00011110 );
	    type s1 = _mm_or_si128( bits, s0 );
	    type s2 = _mm_shuffle_epi32( s1, 0b00000001 );
	    type s3 = _mm_or_si128( s1, s2 );
	    member_type degree = lane0( s3 ) | lane2( s3 );
	    return degree;
	}
	type extract_source( type v ) const {
	    type x = _mm_andnot_si128( mask, v );
	    // Now 'sign extend' from the dropped bit
	    // type lx = _mm_slli_epi32( x, degree_bits );
	    // type rx = _mm_srai_epi32( lx, degree_bits );
	    return x;
	}
	type get_mask() const { return bitwise_invert( mask ); }
    };
    template<unsigned degree_bits, unsigned degree_shift>
    static sse42_epi32_extract_degree
    create_extractor() {
	return sse42_epi32_extract_degree( degree_bits, degree_shift );
    }
};
#endif // __SSE4_2__

} // namespace target

#endif // GRAPTOR_TARGET_SSE42_4x4_H
