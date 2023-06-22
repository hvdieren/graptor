// -*- c++ -*-
#ifndef GRAPTOR_TARGET_AVX2_8x4_H
#define GRAPTOR_TARGET_AVX2_8x4_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include "graptor/target/decl.h"

#if __AVX2__ // AVX2 implies SSE4.2
#include "graptor/target/avx2_bitwise.h"
#include "graptor/target/sse42_4x4.h" // for half-sized indices and masks
#include "graptor/target/sse42_8x2.h" // for half-traits
#include "graptor/target/sse42_1x4.h" // for tzcnt
#include "graptor/target/sse42_2x4.h" // for tzcnt
#include "graptor/target/mmx_2x4.h" // for tzcnt
#endif // __AVX2__

alignas(64) extern const uint64_t avx2_1x4_convert_to_8x4_lut[64];

namespace target {

/***********************************************************************
 * AVX2 8 float
 ***********************************************************************/
#if __AVX2__
template<typename T> // T = uint64_t
struct avx2_8x4 : public avx2_bitwise {
    static_assert( sizeof(T) == 8, 
		   "version of template class for 8-byte integers" );
public:
    static constexpr size_t W = 8;
    static constexpr size_t B = 8*W;
    static constexpr size_t vlen = 4;
    static constexpr size_t size = W * vlen;

    using member_type = T;
    using int_type = uint64_t;
    using type = __m256i;
    using itype = __m256i;
    using vmask_type = __m256i;

    using half_traits = sse42_8x2<T>;
    using lo_half_traits = half_traits;
    using hi_half_traits = half_traits;
    using int_traits = avx2_8x4<int_type>;

    using mask_traits = mask_type_traits<vlen>;
    using mask_type = typename mask_traits::type;
    using mt_preferred = target::mt_vmask;
    
/*
    static void print( std::ostream & os, type v ) {
	os << '(' << lane0(v) << ',' << lane1(v)
	   << ',' << lane2(v) << ',' << lane3(v) << ')';
    }
*/
    
    static member_type lane( type a, int idx ) {
	switch( idx ) {
	case 0: return (member_type) _mm256_extract_epi64( a, 0 );
	case 1: return (member_type) _mm256_extract_epi64( a, 1 );
	case 2: return (member_type) _mm256_extract_epi64( a, 2 );
	case 3: return (member_type) _mm256_extract_epi64( a, 3 );
	default:
	    assert( 0 && "should not get here" );
	}
    }
    static member_type lane0( type a ) { return lane( a, 0 ); }
    static member_type lane1( type a ) { return lane( a, 1 ); }
    static member_type lane2( type a ) { return lane( a, 2 ); }
    static member_type lane3( type a ) { return lane( a, 3 ); }

    static type setlane( type a, member_type b, int idx ) {
    	switch( idx ) {
	case 0: return _mm256_insert_epi64( a, b, 0 );
	case 1: return _mm256_insert_epi64( a, b, 1 );
	case 2: return _mm256_insert_epi64( a, b, 2 );
	case 3: return _mm256_insert_epi64( a, b, 3 );
	default:
	    assert( 0 && "should not get here" );
	}
    }

    // Logical mask - only test msb
    static bool is_all_false( type a ) { return is_zero( srli( a, B-1 ) ); }

    static type set1( member_type a ) { return _mm256_set1_epi64x( a ); }
    static itype set1inc( int_type a ) {
	return add( set1inc0(), _mm256_set1_epi64x( a ) );
    }
    static itype set1inc0() {
	return load(
	    reinterpret_cast<const member_type *>(
		&increasing_sequence_epi64[0] ) );
    }
    static type set( member_type a3, member_type a2,
		     member_type a1, member_type a0 ) {
	return _mm256_set_epi64x( a3, a2, a1, a0 );
    }
    static type setl0( member_type a ) {
	return _mm256_zextsi128_si256( _mm_cvtsi64_si128( a ) );
    }
    static type setoneval() {
	// http://agner.org/optimize/optimizing_assembly.pdf
	type x;
	return _mm256_srli_epi64( _mm256_cmpeq_epi64( x, x ), 63 );
    }

    template<typename VecT2>
    static auto convert( VecT2 a ) {
	typedef vector_type_traits<
	    typename int_type_of_size<sizeof(VecT2)/vlen>::type,
	    sizeof(VecT2)> traits2;
	return set( traits2::lane3(a), traits2::lane2(a),
		    traits2::lane1(a), traits2::lane0(a) );
    }

    // Needs specialisations!
    // Casting here is to do a signed cast of a bitmask for logical masks
    template<typename T2>
    static auto convert_to( type a ) {
	typedef vector_type_traits<T2,sizeof(T2)*vlen> traits2;
	using Ty = typename std::make_signed<typename int_type_of_size<sizeof(member_type)>::type>::type;
	return traits2::set( (Ty)lane3(a), (Ty)lane2(a), (Ty)lane1(a), (Ty)lane0(a) );
    }

    static type from_int( unsigned mask ) {
	return asvector( mask );
    }
    
    static type asvector( mask_type mask ) {
#if 0
	// Need to work with 8 32-bit integers as there is no 64-bit srai
	// in AVX2. Luckily, this works just as well.
	type vmask = _mm256_set1_epi32( (int)mask );
	const __m256i cnt = _mm256_set_epi32( 28, 28, 29, 29, 30, 30, 31, 31 );
	type vmask2 = _mm256_sllv_epi32( vmask, cnt );
	return _mm256_srai_epi32( vmask2, 31 );
#else
	// Use a lookup table. The table spans 8 64-byte cache lines.
	// Each "row" of the table contains vlen elements;
	unsigned idx = ( mask & 0xff ) * vlen;
	return load( reinterpret_cast<const member_type *>(
			 &avx2_1x4_convert_to_8x4_lut[idx] ) );
#endif
    }
    template<typename T2>
    static type asvector(
	vmask_type mask,
	typename std::enable_if<sizeof(T2)==sizeof(member_type)>::type * = nullptr ) {
	return mask;
    }

    static mask_type asmask( vmask_type a ) {
#if __AVX512VL__ && __AVX512DQ__
	return _mm256_movepi64_mask( a );
#else
	mask_type mm = _mm256_movemask_pd( _mm256_castsi256_pd( a ) );
	return mm;
#endif
    }

    static type add( type a, type b ) { return _mm256_add_epi64( a, b ); }
    static type sub( type a, type b ) { return _mm256_sub_epi64( a, b ); }
    // static type mul( type a, type b ) { return _mm256_mul_epi64( a, b ); }
    // static type div( type a, type b ) { return _mm256_div_epi64( a, b ); }
    
    static type add( type src, mask_type m, type a, type b ) {
	return blend( m, src, add( a, b ) );
    }
    static type add( type src, vmask_type m, type a, type b ) {
	return blend( m, src, add( a, b ) );
    }

    static type min( type a, type b ) {
	return blend( cmpgt( a, b, mt_vmask() ), a, b );
    }
    static type max( type a, type b ) {
	return blend( cmpgt( a, b, mt_vmask() ), b, a );
    }

    static type cmpeq( type a, type b, target::mt_vmask ) {
	return _mm256_cmpeq_epi64( a, b );
    }
    static type cmpne( type a, type b, target::mt_vmask ) {
	return ~_mm256_cmpeq_epi64( a, b );
    }
    static type cmpgt( type a, type b, target::mt_vmask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm256_cmpgt_epi64( a, b );
	else {
#if __AVX512VL__
	    return asvector( _mm256_cmpgt_epu64_mask( a, b ) );
#else
	    // This could be needlessly expensive for many comparisons
	    // where the top bit will never be set (e.g. VID)
	    // type ab = bitwise_xor( a, b );
	    // type flip = srli( ab, 8*W-1 );
	    // cmpgt -> xor with flip
	    type one = slli( setone(), 8*W-1 );
	    type ax = bitwise_xor( a, one );
	    type bx = bitwise_xor( b, one );
	    return _mm256_cmpgt_epi64( ax, bx );
#endif
	}
    }
    static type cmpge( type a, type b, target::mt_vmask ) {
	return logical_or( cmpgt( a, b, mt_vmask() ),
			   cmpeq( a, b, mt_vmask() ) );
    }
    static type cmplt( type a, type b, target::mt_vmask ) {
	return cmpgt( b, a, target::mt_vmask() );
    }
    static type cmple( type a, type b, target::mt_vmask ) {
	return cmpge( b, a, target::mt_vmask() );
    }

#if __AVX512F__ && __AVX512VL__
    static mask_type cmpgt( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm256_cmpgt_epi64_mask( a, b );
	else
	    return _mm256_cmpgt_epu64_mask( a, b );
    }
    static mask_type cmplt( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm256_cmplt_epi64_mask( a, b );
	else
	    return _mm256_cmplt_epu64_mask( a, b );
    }
    static mask_type cmpge( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm256_cmpge_epi64_mask( a, b );
	else
	    return _mm256_cmpge_epu64_mask( a, b );
    }
    static mask_type cmple( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm256_cmple_epi64_mask( a, b );
	else
	    return _mm256_cmple_epu64_mask( a, b );
    }
    static mask_type cmpeq( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm256_cmpeq_epi64_mask( a, b );
	else
	    return _mm256_cmpeq_epu64_mask( a, b );
    }
    static mask_type cmpne( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm256_cmpneq_epi64_mask( a, b );
	else
	    return _mm256_cmpneq_epu64_mask( a, b );
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

    static bool cmpne( type a, type b, target::mt_bool ) {
	return is_zero( cmpeq( a, b, target::mt_vmask() ) );
    }
    static bool cmpeq( type a, type b, target::mt_bool ) {
	return is_zero( cmpne( a, b, target::mt_vmask() ) );
    }

    static type blendm( vmask_type m, type l, type r ) {
	// see also https://stackoverflow.com/questions/40746656/howto-vblend
	// -for-32-bit-integer-or-why-is-there-no-mm256-blendv-epi32
	// for using FP blend.
	// We are assuming that the mask is all-ones in each field (not only
	// the highest bit is used), so this works without modifying the mask.
	return _mm256_blendv_epi8( l, r, m );
    }
    static type blend( vmask_type mask, type a, type b ) {
	return _mm256_blendv_epi8( a, b, mask );
    }
    static type blend( __m128i mask, type a, type b ) { // half-width mask
	return _mm256_blendv_epi8( a, b, _mm256_cvtepi8_epi16( mask ) );
    }
    static type blend( mask_type m, type a, type b ) {
#if __AVX512F__ && __AVX512VL__
	return _mm256_mask_blend_epi64( m, a, b );
#else
	return _mm256_blendv_epi8( a, b, asvector( m ) );
#endif
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

    static type iforz( vmask_type m, type a ) {
#if 0
#if __AVX512F__ && __AVX512VL__
	// Note: no 64-bit srai available
	vmask_type clean = _mm256_srai_epi64( m, 63 );
#else
	vmask_type half = _mm256_srai_epi32( m, 31 );
	vmask_type clean = _mm256_shuffle_epi32( half, 0b11110101 );
#endif
#else
	vmask_type clean = m;
#endif
	return bitwise_and( clean, a );
    }
    static type iforz( mask_type m, type a ) {
	return blend( m, setzero(), a );
    }

#if __AVX512VL__
    static constexpr bool has_ternary = true;
#else
    static constexpr bool has_ternary = false;
#endif

    template<unsigned char imm8>
    static type ternary( type a, type b, type c ) {
#if __AVX512VL__
	return _mm256_ternarylogic_epi64( a, b, c, imm8 );
#else
	assert( 0 && "NYI" );
	return setzero();
#endif
    }

    static uint32_t find_first( type v ) {
	return mask_traits::find_first( asmask( v ) );
    }
    static uint32_t find_first( type v, vmask_type m ) {
	return mask_traits::find_first( asmask( v ), asmask( m ) );
    }
    
    static type castint( type a ) { return a; }
    static __m256d castfp( type a ) { return _mm256_castsi256_pd( a ); }

    static member_type reduce_setif( type val ) {
	assert( 0 && "NYI" );
	return setzero(); // TODO
    }
    static member_type reduce_add( type val ) {
	// TODO: extract upper/lower half and add up using sse42_8x2::add and
	//       sse42_8x2::reduce_add
	// return lane0( val ) + lane1( val ) + lane2( val ) + lane3( val );
	return half_traits::reduce_add(
	    half_traits::add( lower_half( val ), upper_half( val ) ) );
    }
    static member_type reduce_add( type val, mask_type mask ) {
	assert( 0 && "NYI" );
	return setzero(); // _mm256_reduce_add_epi64( val );
    }
    static member_type reduce_logicalor( type val ) {
	return _mm256_movemask_epi8( val ) ? ~member_type(0) : member_type(0);
    }
    static member_type reduce_logicalor( type val, type mask ) {
	int v = _mm256_movemask_epi8( val );
	int m = _mm256_movemask_epi8( mask );
	return (!m | v) ? ~member_type(0) : member_type(0);
    }
    static member_type reduce_bitwiseor( type val ) {
	// return lane0( val ) | lane1( val ) | lane2( val ) | lane3( val );
	type c = _mm256_permute2x128_si256( val, val, 0x81 );
	type d = bitwise_or( c, val );
	return lane0( d ) | lane1( d );
    }
    static member_type reduce_bitwiseand( type val ) {
	return lane0( val ) & lane1( val ) & lane2( val ) & lane3( val );
    }
    
    static member_type reduce_min( type val ) {
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
	return std::min( std::min( lane0(val), lane1(val) ),
			 std::min( lane2(val), lane3(val) ) );
    }
    static member_type reduce_max( type val ) {
	return std::max( std::max( lane0(val), lane1(val) ),
			 std::max( lane2(val), lane3(val) ) );
    }

    static type sllv( type a, type b ) { return _mm256_sllv_epi64( a, b ); }
    static type srlv( type a, type b ) { return _mm256_srlv_epi64( a, b ); }
    static type sllv( type a, __m128i b ) {
	return _mm256_sllv_epi64( a, _mm256_cvtepi32_epi64( b ) );
    }
    static type srlv( type a, __m128i b ) {
	return _mm256_srlv_epi64( a, _mm256_cvtepi32_epi64( b ) );
    }
    static type sll( type a, long b ) {
	return _mm256_sll_epi64( a, _mm_cvtsi64_si128( b ) );
    }
    static type srl( type a, long b ) {
	return _mm256_srl_epi64( a, _mm_cvtsi64_si128( b ) );
    }
    static type slli( type a, int_type b ) { return _mm256_slli_epi64( a, b ); }
    static type srli( type a, int_type b ) { return _mm256_srli_epi64( a, b ); }
    static type srai( type a, int_type b ) {
#if __AVX512VL__
	return _mm256_srai_epi64( a, b );
#else
	if( b >= 32 ) {
	    // In absence of srai_epi64, swap 32-bit words within each
	    // 64-bit word, then shift using srai_epi32
	    type c = _mm256_shuffle_epi32( a, 0b10110001 );
	    return _mm256_srai_epi32( c, b-32 );
	}
	assert( 0 && "NYI" );
#endif
    }
    template<int_type b>
    static type bslli( type a ) {
	if constexpr ( b == 16 ) { // full __m128i lane
	    // Move lower 128-bit lane to upper lane, set lower lane to zero
	    auto c = _mm256_permute2x128_si256( a, a, 0x08 );
	    return c;
	} else {
	    assert( 0 && "NYI" );
	    /*
	      const auto one = setone();
	      const auto msk = _mm256_bsrli_epi128( one, b );
	      auto c = _mm256_bslli_epi128( a, b );
	      auto d = bitwise_andnot( msk, a );
	      auto e = _mm256_bsrli_epi128( d, 16-b );
	      auto f = _mm256_permute( e, a );
	      auto g = bitwise_or( msk, a );
	      return g;
	    */
	}
    }

    template<typename ReturnTy>
    static auto tzcnt( type a ) {
#if __AVX512VL__ && __AVX512DQ__
	// According to Bit Twiddling Hacks by conversion to floating-point
	__m256i b = _mm256_sub_epi64( setzero(), a );
	__m256i c = _mm256_and_si256( a, b );
	__m256d f = _mm256_cvtepi64_pd( c );
	__m256i g = _mm256_castpd_si256( f );
	__m256i h = _mm256_srli_epi64( g, 52 );
	__m256i bias = set1( 0x3ff );
	__m256i cnt = _mm256_sub_epi64( h, bias );
	if constexpr ( sizeof(ReturnTy) == W )
	    return cnt;
	else if constexpr ( sizeof(ReturnTy) == 4 ) {
	    return _mm256_cvtepi64_epi32( cnt ); // AVX512VL
	} else {
	    assert( 0 && "NYI" );
	    return 0;
	}
#elif __AVX__
	// We don't have access to a conversion from epi64 integers to
	// floating-point types.
	__m128i hi = upper_half( a );
	__m128i lo = lower_half( a );
	ReturnTy td = _tzcnt_u64( _mm_extract_epi64( hi, 1 ) );
	ReturnTy tc = _tzcnt_u64( _mm_extract_epi64( hi, 0 ) );
	ReturnTy tb = _tzcnt_u64( _mm_extract_epi64( lo, 1 ) );
	ReturnTy ta = _tzcnt_u64( _mm_extract_epi64( lo, 0 ) );
	
	if constexpr ( sizeof(ReturnTy) == W )
	    return set( td, tc, tb, ta );
	else if constexpr ( sizeof(ReturnTy) == 4 )
	    return sse42_4x4<ReturnTy>::set( td, tc, tb, ta );
	else if constexpr ( sizeof(ReturnTy) == 2 )
#if GRAPTOR_USE_MMX
	    return mmx_2x4<ReturnTy>::set( td, tc, tb, ta );
#else
	    return sse42_2x4<ReturnTy>::set( td, tc, tb, ta );
#endif
	else if constexpr ( sizeof(ReturnTy) == 1 )
	    return sse42_1x4<ReturnTy>::set( td, tc, tb, ta );
	else {
	    assert( 0 && "NYI" );
	    return 0;
	}
#endif
    }

    template<typename ReturnTy>
    static auto lzcnt( type a ) {
#if __AVX512VL__ && __AVX512DQ__
	type cnt = _mm256_lzcnt_epi64( a );
#elif __AVX2__
	// Count leading zeros in half-lanes
	type cnt2 = avx2_4x8<uint32_t>::lzcnt<uint32_t>( a );
	// Bottom-half lanes only count if top-half is 32
	// * top-half 32: bottom-half + top-half
	// * top-half lower: move top-half in place
	const type c1h = _mm256_srli_epi32( setone(), 31 );
	const type c32h = _mm256_slli_epi32( c1h, 5 );
	type cnt2hi = srli( cnt2, 32 );
	type mask = _mm256_cmpeq_epi32( cnt2hi, c32h );
	type cnt = add( bitwise_and( mask, cnt2 ), cnt2hi );
#endif
	if constexpr ( sizeof(ReturnTy) == W )
	    return cnt;
	else if constexpr ( sizeof(ReturnTy) == 4 ) {
	    return _mm256_cvtepi64_epi32( cnt ); // AVX512VL
	} else {
	    assert( 0 && "NYI" );
	    return 0;
	}
    }

    static auto popcnt( type v ) {
#if __AVX512VPOPCNTDQ__ && __AVX512VL__
	return _mm256_popcnt_epi64( v );
#else
	// source: https://arxiv.org/pdf/1611.07612.pdf
	__m256i lookup =
	    _mm256_setr_epi8( 0, 1, 1, 2, 1, 2, 2, 3,
			      1, 2, 2, 3, 2, 3, 3, 4,
			      0, 1, 1, 2, 1, 2, 2, 3,
			      1, 2, 2, 3, 2, 3, 3, 4 );
	__m256i low_mask = _mm256_set1_epi8( 0x0f );
	__m256i lo = _mm256_and_si256( v, low_mask );
	__m256i hi = _mm256_and_si256( _mm256_srli_epi32( v, 4 ), low_mask );
	__m256i popcnt1 = _mm256_shuffle_epi8( lookup, lo );
	__m256i popcnt2 = _mm256_shuffle_epi8( lookup, hi );
	__m256i total = _mm256_add_epi8( popcnt1, popcnt2 );
	return _mm256_sad_epu8( total, _mm256_setzero_si256() );
#endif
    }

    static type loadu( const member_type * a ) {
	return _mm256_loadu_si256( (type *)a );
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

    template<unsigned short Scale>
    static type
    gather_w( const member_type *a, itype b ) {
	return _mm256_i64gather_epi64( (const long long *)a, b, Scale );
    }
    template<unsigned short Scale>
    static type
    gather_w( const member_type *a, itype b, vmask_type mask ) {
	return _mm256_mask_i64gather_epi64( setzero(), reinterpret_cast<const long long *>( a ), b, mask, Scale );
    }
    template<unsigned short Scale>
    static type
    gather_w( const member_type *a, itype b, mask_type mask ) {
	return gather_w<Scale>( a, b, asvector( mask ) );
    }
    template<unsigned short Scale>
    static type
    gather_w( const member_type *a, __m128i b ) {
	return _mm256_i32gather_epi64( (const long long *)a, b, Scale );
    }
    template<unsigned short Scale>
    static type
    gather_w( const member_type *a, __m128i b, vmask_type vmask ) {
	return _mm256_mask_i32gather_epi64( setzero(), (const long long *)a, b, vmask, Scale );
    }
    template<unsigned short Scale>
    static type
    gather_w( const member_type *a, __m128i b, mask_type mask ) {
	return _mm256_mask_i32gather_epi64( setzero(), (const long long *)a, b, asvector( mask ), Scale );
    }
    template<unsigned short Scale>
    static type
    gather_w( const member_type *a, __m128i b, __m128i vmask ) {
#if __AVX512F__ && __AVX512VL__
	mask_type m = sse42_4x4<member_type>::asmask( vmask );
	return _mm256_mmask_i32gather_epi64( setzero(), m, b, (const long long *)a, Scale );
#else
	vmask_type wmask = _mm256_cvtepi32_epi64( vmask );
	return _mm256_mask_i32gather_epi64( setzero(), (const long long *)a, b, wmask, Scale );
#endif
    }

    static type
    gather( const member_type *a, itype b ) { return gather_w<W>( a, b ); }
    static type
    gather( const member_type *a, itype b, vmask_type mask ) {
	return gather_w<W>( a, b, mask );
    }
    static type
    gather( const member_type *a, itype b, mask_type mask ) {
	return gather_w,W>( a, b, asvector( mask ) );
    }
    static type
    gather( const member_type *a, __m128i b ) { return gather_w<W>( a, b ); }
    static type
    gather( const member_type *a, __m128i b, vmask_type vmask ) {
	return _mm256_mask_i32gather_epi64( setzero(), (const long long *)a, b, vmask, W );
    }
    static type
    gather( const member_type *a, __m128i b, mask_type mask ) {
	return gather_w<W>( a, b, asvector( mask ) );
    }

    static type
    gather( const member_type *a, __m128i b, __m128i vmask ) {
	__m256i wmask = _mm256_cvtepi32_epi64( vmask );
	return gather_w<W>( a, b, wmask );
    }

    static void scatter( member_type *a, itype b, type c ) {
#if __AVX512VL__
	_mm_i64scatter_epi64( a, b, c, W );
#else
	a[int_traits::lane0(b)] = lane0(c);
	a[int_traits::lane1(b)] = lane1(c);
	a[int_traits::lane2(b)] = lane2(c);
	a[int_traits::lane3(b)] = lane3(c);
#endif
    }
    static void scatter( member_type *a, itype b, type c, mask_type mask ) {
	if( mask & 1 ) a[int_traits::lane0(b)] = lane0(c);
	if( mask & 2 ) a[int_traits::lane1(b)] = lane1(c);
	if( mask & 4 ) a[int_traits::lane2(b)] = lane2(c);
	if( mask & 8 ) a[int_traits::lane3(b)] = lane3(c);
    }
    static void scatter( member_type *a, itype b, type c, vmask_type mask ) {
	if( int_traits::lane0(mask) ) a[int_traits::lane0(b)] = lane0(c);
	if( int_traits::lane1(mask) ) a[int_traits::lane1(b)] = lane1(c);
	if( int_traits::lane2(mask) ) a[int_traits::lane2(b)] = lane2(c);
	if( int_traits::lane3(mask) ) a[int_traits::lane3(b)] = lane3(c);
    }
    static void scatter( member_type *a, itype b, type c, __m128i mask ) {
	using mtraits = sse42_4x4<uint32_t>;
	if( mtraits::lane0(mask) ) a[int_traits::lane0(b)] = lane0(c);
	if( mtraits::lane1(mask) ) a[int_traits::lane1(b)] = lane1(c);
	if( mtraits::lane2(mask) ) a[int_traits::lane2(b)] = lane2(c);
	if( mtraits::lane3(mask) ) a[int_traits::lane3(b)] = lane3(c);
    }
    static void scatter( member_type *a, __m128i b, type c ) {
	using itraits = sse42_4x4<uint32_t>;
	a[itraits::lane0(b)] = lane0(c);
	a[itraits::lane1(b)] = lane1(c);
	a[itraits::lane2(b)] = lane2(c);
	a[itraits::lane3(b)] = lane3(c);
    }
    static void scatter( member_type *a, __m128i b, type c, mask_type mask ) {
	using itraits = sse42_4x4<uint32_t>;
	if( mask & 1 ) a[itraits::lane0(b)] = lane0(c);
	if( mask & 2 ) a[itraits::lane1(b)] = lane1(c);
	if( mask & 4 ) a[itraits::lane2(b)] = lane2(c);
	if( mask & 8 ) a[itraits::lane3(b)] = lane3(c);
    }
    static void scatter( member_type *a, __m128i b, type c, vmask_type mask ) {
	using itraits = sse42_4x4<uint32_t>;
	if( int_traits::lane0(mask) ) a[itraits::lane0(b)] = lane0(c);
	if( int_traits::lane1(mask) ) a[itraits::lane1(b)] = lane1(c);
	if( int_traits::lane2(mask) ) a[itraits::lane2(b)] = lane2(c);
	if( int_traits::lane3(mask) ) a[itraits::lane3(b)] = lane3(c);
    }
    static void scatter( member_type *a, __m128i b, type c, __m128i mask ) {
#if __AVX512VL__
	using itraits = sse42_4x4<uint32_t>;
	using mtraits = mask_type_traits<4>;
	__mmask8 m = _mm_movemask_ps( _mm_castsi128_ps( mask ) );
	_mm_mask_i32scatter_epi64( a, m, b, c, W );
#else
	using itraits = sse42_4x4<uint32_t>;
	if( itraits::lane0(mask) ) a[itraits::lane0(b)] = lane0(c);
	if( itraits::lane1(mask) ) a[itraits::lane1(b)] = lane1(c);
	if( itraits::lane2(mask) ) a[itraits::lane2(b)] = lane2(c);
	if( itraits::lane3(mask) ) a[itraits::lane3(b)] = lane3(c);
#endif
    }

    class avx2_epi64_extract_degree {
	type mask, shift, inv;
	member_type degree_bits;

    public:
	avx2_epi64_extract_degree( unsigned degree_bits_,
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

	    inv = bitwise_invert( mask );
	}
	member_type extract_degree( type v ) const {
	    type vs = _mm256_and_si256( v, mask );
	    type bits = _mm256_srlv_epi64( vs, shift );
	    // member_type degree = reduce_add( bits );
	    type s0 = _mm256_shuffle_epi32( bits, 0b11111010 );
	    type s1 = _mm256_or_si256( bits, s0 );
	    member_type degree = lane0( s1 ) | lane2( s1 );
	    return degree;
	}
	type extract_source( type v ) const {
	    type x = _mm256_andnot_si256( mask, v );
	    // Now 'sign extend' from the dropped bit
	    // type lx = _mm256_slli_epi64( x, degree_bits );
	    // type rx = _mm256_srai_epi64( lx, degree_bits );
	    // return rx;
	    // Match for 0x0...01...1
	    return blend( cmpeq( x, inv, mt_vmask() ), x, setone() );
	}
    };
    template<unsigned degree_bits, unsigned degree_shift>
    static avx2_epi64_extract_degree
    create_extractor() {
	return avx2_epi64_extract_degree( degree_bits, degree_shift );
    }
};

#if 0
template<>
template<>
auto vector_type_int_traits<logical<8>,32>::template convert_to<bool>( typename vector_type_int_traits<logical<8>,32>::type a ) {
    typedef vector_type_int_traits<bool,sizeof(bool)*vlen> traits2;
    // TODO: do != 0 comparison in vector mode twice,
    //       then extract (epi64 -> epi8)
    return traits2::set( !!lane3(a), !!lane2(a), !!lane1(a), !!lane0(a) );
}
#endif

#endif // __AVX2__

} // namespace target

#endif // GRAPTOR_TARGET_AVX2_8x4_H
