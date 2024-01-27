// -*- c++ -*-
#ifndef GRAPTOR_TARGET_SSE42_1x16_H
#define GRAPTOR_TARGET_SSE42_1x16_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include "graptor/target/decl.h"
#include "graptor/target/bitmask.h"

#if __MMX__
#include "graptor/target/mmx_1x8.h"
#endif

#if __AVX512F__
#include "graptor/target/avx512_4x16.h"
#endif

#if __AVX2__
#include "graptor/target/avx2_4x8.h"
#endif

#if __SSE4_2__
#include "graptor/target/sse42_bitwise.h"
#endif

alignas(64) extern const uint8_t movemask_lut_epi8[16*4];
alignas(64) extern const uint8_t increasing_sequence_epi8[64];

namespace target {

/***********************************************************************
 * SSE4.2 16 1-byte integers, or 8 1-byte integers (leaving upper half
 * unused)
 ***********************************************************************/
#if __SSE4_2__
template<unsigned short VL, typename T = uint8_t>
struct sse42_1xL : public sse42_bitwise {
    static_assert( sizeof(T) == 1, 
		   "version of template class for 1-byte integers" );
    static_assert( VL == 4 || VL == 8 || VL == 16,
		   "assumption on vector length" );
public:
    using member_type = T;
    using int_type = uint8_t;
    using type = __m128i;
    using itype = __m128i;
    using vmask_type = __m128i;

    using mt_preferred = target::mt_vmask;

    using mask_traits = mask_type_traits<VL>;
    using mask_type = typename mask_traits::type;

    using mask_traits16 = mask_type_traits<16>;
    using mask_type16 = typename mask_traits16::type;

    // using half_traits = mmx_1x8<T>;
    using half_traits = sse42_1xL<8,T>; // no half-traits if VL == 8 ...
    using lo_half_traits = half_traits;
    using hi_half_traits = half_traits;
    using int_traits = sse42_1xL<VL,int_type>;
    
    static constexpr unsigned short W = 1;
    static constexpr unsigned short B = 8*W;
    static constexpr unsigned short vlen = VL;
    static constexpr unsigned short size = W * vlen;

    static member_type lane( type a, int idx ) {
	switch( idx ) {
	case 0: return (member_type) _mm_extract_epi8( a, 0 );
	case 1: return (member_type) _mm_extract_epi8( a, 1 );
	case 2: return (member_type) _mm_extract_epi8( a, 2 );
	case 3: return (member_type) _mm_extract_epi8( a, 3 );
	case 4: return (member_type) _mm_extract_epi8( a, 4 );
	case 5: return (member_type) _mm_extract_epi8( a, 5 );
	case 6: return (member_type) _mm_extract_epi8( a, 6 );
	case 7: return (member_type) _mm_extract_epi8( a, 7 );
	case 8: return (member_type) _mm_extract_epi8( a, 8 );
	case 9: return (member_type) _mm_extract_epi8( a, 9 );
	case 10: return (member_type) _mm_extract_epi8( a, 10 );
	case 11: return (member_type) _mm_extract_epi8( a, 11 );
	case 12: return (member_type) _mm_extract_epi8( a, 12 );
	case 13: return (member_type) _mm_extract_epi8( a, 13 );
	case 14: return (member_type) _mm_extract_epi8( a, 14 );
	case 15: return (member_type) _mm_extract_epi8( a, 15 );
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

    static type setoneval() {
	// Agner Fog optimization manual
	return _mm_abs_epi8( setone() );
    }
    
    static type set1( member_type a ) { return _mm_set1_epi8( a ); }
    static type set1inc( member_type a ) {
	return add( set1( a ), set1inc0() );
    }
    static type set1inc0() {
	return load(
	    static_cast<const member_type *>( &increasing_sequence_epi8[0] ) );
    }
    static type setl0( member_type a ) {
	return _mm_set1_epi16( (uint16_t)a );
    }
    static type setl1( member_type a ) {
	return _mm_set1_epi16( ((short)a)<<8 );
    }
    static type set( member_type a15, member_type a14,
		     member_type a13, member_type a12,
		     member_type a11, member_type a10,
		     member_type a9, member_type a8,
		     member_type a7, member_type a6,
		     member_type a5, member_type a4,
		     member_type a3, member_type a2,
		     member_type a1, member_type a0 ) {
	return _mm_set_epi8( a15, a14, a13, a12, a11, a10, a9, a8,
			     a7, a6, a5, a4, a3, a2, a1, a0 );
    }
    static type set( member_type a7, member_type a6,
		     member_type a5, member_type a4,
		     member_type a3, member_type a2,
		     member_type a1, member_type a0 ) {
	return _mm_set_epi8( 0, 0, 0, 0, 0, 0, 0, 0,
			     a7, a6, a5, a4, a3, a2, a1, a0 );
    }
    static type set( member_type a3, member_type a2,
		     member_type a1, member_type a0 ) {
	uint32_t u3 = a3;
	uint32_t u2 = a2;
	uint32_t u1 = a1;
	uint32_t u0 = a0;
	uint32_t hi = ( a3 << B ) | a2;
	uint32_t lo = ( a1 << B ) | a0;
	uint32_t qw = ( hi << (2*B) ) | lo;
	return _mm_cvtsi32_si128( qw );
    }

    static type set_pair( __m64 hi, __m64 lo ) {
	return _mm_set_epi64( hi, lo );
    }

    static type set_pair( type hi, type lo ) {
	if constexpr( VL == 16 ) {
	    // Use lowest half of hi and lo
	    // type rot = _mm_shuffle_epi32( hi, 0b01000100 );
	    // type p = _mm_blend_epi32( lo, rot, 0b1100 );
	    // return p;
	    return _mm_unpacklo_epi64( lo, hi );
	} else
	    assert( 0 && "NYI - should not occur" );
    }

    static type set_pair( int hi, int lo ) {
	if constexpr ( VL == 16 ) {
	    static_assert( W == 1, "intended only for VL == 8, W==1" );
	} else {
	    return _mm_set_epi32( 0, 0, hi, lo );
	}
    }

    static type set_pair( vpair<vpair<member_type,member_type>,
			  vpair<member_type,member_type>> hi,
			  vpair<vpair<member_type,member_type>,
			  vpair<member_type,member_type>> lo ) {
	if constexpr ( VL == 16 ) {
	    static_assert( W == 1, "intended only for VL == 8, W==1" );
	} else {
	    return _mm_set_epi32( 0, 0,
				  *reinterpret_cast<const int *>( &hi ),
				  *reinterpret_cast<const int *>( &lo ) );
	}
    }

#if GRAPTOR_USE_MMX
    static __m64 lower_half( type a ) {
	return (__m64)_mm_extract_epi64( a, 0 );
    }
    static __m64 upper_half( type a ) {
	return (__m64)_mm_extract_epi64( a, 1 );
    }
#else // GRAPTOR_USE_MMX
    static __m128i lower_half( type a ) {
	type z = setzero();
	return _mm_blend_epi32( z, a, VL == 16 ? 0b0011 : 0b0001 );
    }
    static __m128i upper_half( type a ) {
	type z = setzero();
	type s = _mm_bsrli_si128( a, VL == 16 ? 8 : 4 );
	return _mm_blend_epi32( z, s, VL == 16 ? 0b0011 : 0b0001 );
    }
#endif // GRAPTOR_USE_MMX

    // Logical mask - only test msb
    static bool is_all_false( type a ) {
	if constexpr ( VL == 8 ) {
	    uint64_t b = _mm_extract_epi64( a, 0 );
	    return ( b & 0x8080808080808080ull ) == 0;
	} else
	    assert( 0 && "NYI" );
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
#if __AVX512VL__ && __AVX512BW__
	return _mm_mask_blend_epi8( extendVL( m ), l, r );
#elif __AVX512BW__
	return _mm512_castsi512_si128(
	    _mm512_mask_blend_epi8( m,
				    _mm512_castsi128_si512( l ),
				    _mm512_castsi128_si512( r ) ) );
#else
	return blendm( asvector( m ), l, r );
#endif
    }

    static uint32_t find_first( type v ) {
	// Needs to return vlen when all lanes of v are zero. Happens because
	// of the bitwise inversion of the mask, which puts default 0 bits
	// to 1. As long as vlen < 32
	return mask_traits::find_first( asmask( v ) );
    }
    static uint32_t find_first( type v, vmask_type m ) {
	// Should return 8 when all lanes of v are zero due to mask
	// argument
	return mask_traits::find_first( asmask( v ), asmask( m ) );
    }
    
    static type blend( mask_type m, type l, type r ) {
	return blendm( m, l, r );
    }
    static type blend( vmask_type m, type l, type r ) {
	return blendm( m, l, r );
    }

    static type from_int( unsigned mask ) {
	return asvector( mask );
    }
    
    static mask_type movemask( vmask_type a ) {
#if __AVX512VL__ && __AVX512BW__
	return _mm_movepi8_mask( a ); // k register
#else
	return _mm_movemask_epi8( a ); // GP register
#endif
    }

    static type asvector( mask_type mask ) {
#if __AVX512VL__ && __AVX512BW__
	return _mm_mask_blend_epi8( extendVL( mask ), setzero(), setone() );
#else
	// Use lookup table
	if constexpr ( VL == 16 ) {
	    uint32_t p0 = *(const uint32_t*)
		&movemask_lut_epi8[4*(0xf & (int)mask)];
	    uint32_t p1 = *(const uint32_t*)
		&movemask_lut_epi8[4*(0xf & (int)(mask>>4))];
	    uint32_t p2 = *(const uint32_t*)
		&movemask_lut_epi8[4*(0xf & (int)(mask>>8))];
	    uint32_t p3 = *(const uint32_t*)
		&movemask_lut_epi8[4*(0xf & (int)(mask>>12))];

	    return _mm_set_epi32( p3, p2, p1, p0 );
	} else {
	    uint32_t p0 = *(const uint32_t*)
		&movemask_lut_epi8[4*(0xf & (int)mask)];
	    uint32_t p1 = *(const uint32_t*)
		&movemask_lut_epi8[4*(0xf & (int)(mask>>4))];

	    return _mm_set_epi32( 0, 0, p1, p0 );
	}
#endif
    }

    static mask_type asmask( vmask_type mask ) {
	return movemask( mask );
    }

    static type add( type src, vmask_type m, type a, type b ) {
	return _mm_blendv_epi8( src, add( a, b ), m );
    }
    static type add( type src, mask_type m, type a, type b ) {
	return _mm_mask_blend_epi8( m, src, add( a, b ) );
    }

    static type add( type a, type b ) { return _mm_add_epi8( a, b ); }
    static type adds( type a, type b ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm_add_epi8( a, b );
	else
	    return _mm_adds_epu8( a, b );
    }
    static type sub( type a, type b ) { return _mm_sub_epi8( a, b ); }
    static type min( type a, type b ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm_min_epi8( a, b );
	else
	    return _mm_min_epu8( a, b );
    }
    static type max( type a, type b ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm_max_epi8( a, b );
	else
	    return _mm_max_epu8( a, b );
    }

    static type pruneVL( type a ) {
	return a;
/*
	if constexpr ( VL == 16 )
	    return a;
	else {
	    type z = setzero();
	    return _mm_blend_epi32( z, a, 0b0011 );
	}
*/
    }
    static mask_type pruneVL( mask_type16 a ) {
	// Assumes cast will clear higher bits or imply invalidity
	// return a;
	mask_type r;
	__asm__ __volatile__ ( "" : "=Yk"(r) : "0"(a) : );
	return r;
    }
    static mask_type16 extendVL( mask_type a ) {
	if constexpr( VL == 16 )
	    return a;
	else {
	    mask_type16 r;
	    __asm__ __volatile__ ( "" : "=Yk"(r) : "0"(a) : );
	    mask_type16 m = (unsigned short)0x00ff;
	    auto rm = mask_traits16::logical_and( r, m );
	    return rm;
	}
    }

    static vmask_type cmpeq( type a, type b, target::mt_vmask ) {
	return pruneVL( _mm_cmpeq_epi8( a, b ) );
    }
    static vmask_type cmpne( type a, type b, target::mt_vmask ) {
	return pruneVL( logical_invert( _mm_cmpeq_epi8( a, b ) ) );
    }
    static vmask_type cmpgt( type a, type b, target::mt_vmask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return pruneVL( _mm_cmpgt_epi8( a, b ) );
	else {
	    type one = set1( 0x80 );
	    type ax = bitwise_xor( a, one );
	    type bx = bitwise_xor( b, one );
	    return pruneVL( _mm_cmpgt_epi8( ax, bx ) );
	}
    }
    static vmask_type cmpge( type a, type b, target::mt_vmask ) {
	return logical_or( cmpgt( a, b, mt_vmask() ),
			   cmpeq( a, b, mt_vmask() ) );
    }
    static vmask_type cmplt( type a, type b, target::mt_vmask ) {
	return cmpgt( b, a, mt_vmask() );
    }
    static vmask_type cmple( type a, type b, target::mt_vmask ) {
	return cmpge( b, a, mt_vmask() );
    }

#if __AVX512VL__ && __AVX512BW__
    static mask_type cmpeq( type a, type b, target::mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return pruneVL( _mm_cmpeq_epi8_mask( a, b ) );
	else
	    return pruneVL( _mm_cmpeq_epu8_mask( a, b ) );
    }
    static mask_type cmpne( type a, type b, target::mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return pruneVL( _mm_cmpneq_epi8_mask( a, b ) );
	else
	    return pruneVL( _mm_cmpneq_epu8_mask( a, b ) );
    }
    static mask_type cmpgt( type a, type b, target::mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return pruneVL( _mm_cmpgt_epi8_mask( a, b ) );
	else
	    return pruneVL( _mm_cmpgt_epu8_mask( a, b ) );
    }
    static mask_type cmpge( type a, type b, target::mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return pruneVL( _mm_cmpge_epi8_mask( a, b ) );
	else
	    return pruneVL( _mm_cmpge_epu8_mask( a, b ) );
    }
    static mask_type cmplt( type a, type b, target::mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return pruneVL( _mm_cmplt_epi8_mask( a, b ) );
	else
	    return pruneVL( _mm_cmplt_epu8_mask( a, b ) );
    }
    static mask_type cmple( type a, type b, target::mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return pruneVL( _mm_cmple_epi8_mask( a, b ) );
	else
	    return pruneVL( _mm_cmple_epu8_mask( a, b ) );
    }
#else // __AVX512VL__ && __AVX512BW__
    static mask_type cmpeq( type a, type b, target::mt_mask ) {
	return asmask( cmpeq( a, b, mt_vmask() ) );
    }
    static mask_type cmpne( type a, type b, target::mt_mask ) {
	return asmask( cmpne( a, b, mt_vmask() ) );
    }
    static mask_type cmpgt( type a, type b, target::mt_mask ) {
	return asmask( cmpgt( a, b, mt_vmask() ) );
    }
    static mask_type cmpge( type a, type b, target::mt_mask ) {
	return asmask( cmpge( a, b, mt_vmask() ) );
    }
    static mask_type cmplt( type a, type b, target::mt_mask ) {
	return asmask( cmplt( a, b, mt_vmask() ) );
    }
    static mask_type cmple( type a, type b, target::mt_mask ) {
	return asmask( cmple( a, b, mt_vmask() ) );
    }
#endif // __AVX512VL__ && __AVX512BW__

    static mask_type cmpneg( type a, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return asmask( a );
	else
	    return mask_traits::setzero();
    }
    static vmask_type cmpneg( type a, mt_vmask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return asvector( asmask( a ) );
	else
	    return setzero();
    }
    
    static member_type reduce_setif( type val ) {
	assert( 0 && "NYI" );
	return setzero();
    }
    static member_type reduce_add( type val ) {
	assert( VL == 16 && "NYI - only correct VL16" );
	type p = pruneVL( val );
	type s = _mm_hadd_epi16( p, p );
	type t = _mm_hadd_epi16( s, s );
	return lane0( t ) + lane1( t );
    }
    static member_type reduce_add( type val, mask_type mask ) {
	assert( 0 && "NYI" );
	return setzero();
    }
    static member_type reduce_add( type val, vmask_type mask ) {
	type x = _mm_blendv_epi8( setzero(), val, mask );
	return reduce_add( x );
    }
    static member_type reduce_logicalor( type val ) {
	if constexpr ( VL == 16 )
	    return _mm_movemask_epi8( val ) ? ~member_type(0) : member_type(0);
	else
	    return ( _mm_movemask_epi8( val ) & 0xff )
		? ~member_type(0) : member_type(0);
    }
    static member_type reduce_logicalor( type val, type mask ) {
	int v = _mm_movemask_epi8( val );
	int m = _mm_movemask_epi8( mask );
	if constexpr ( VL == 16 )
	    return (!m | v) ? ~member_type(0) : member_type(0);
	else
	    return ( (!m | v) & 0xff ) ? ~member_type(0) : member_type(0);
    }
    static member_type reduce_logicaland( type val ) {
	   static_assert( vlen < 32, "word width limit" );
	   return member_type(
	       (uint32_t)asmask( val ) == ( uint32_t(1) << vlen ) - 1 );
    }
    static member_type reduce_logicaland( type val, type mask ) {
	// if mask then true if val; if not mask, then true
	// mask => val, or !mask || val, then reduce
	return reduce_logicaland( logical_or( logical_invert( mask ), val ) );
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
	assert( 0 && "TODO - optimize" );
/*
	return std::min( std::min( lane0(val), lane1(val) ),
			 std::min( lane2(val), lane3(val) ),
			 std::min( lane4(val), lane5(val) ),
			 std::min( lane6(val), lane7(val) ) );
*/
    }

    static member_type reduce_max( type val ) {
#if __AVX512F__
	assert( 0 && "NYI -- ignore upper lanes using mask" );
	if constexpr ( std::is_signed_v<member_type> ) {
	    __m512i w = _mm512_cvtepi8_epi32( val );
	    return (member_type)_mm512_reduce_max_epi32( w );
	} else {
	    __m512i w = _mm512_cvtepu8_epi32( val );
	    return (member_type)_mm512_reduce_max_epu32( w );
	}
#else
	auto by2 = max( val, _mm_bslli_si128( val, 1 ) );
	auto by4 = max( by2, _mm_bslli_si128( by2, 2 ) );
	auto by8 = max( by4, _mm_bslli_si128( by4, 4 ) );
	if constexpr ( VL == 16 ) {
	    auto fin = max( by8, _mm_bslli_si128( by8, 8 ) );
	    return lane0( fin );
	} else
	    return lane0( by8 );
#endif
    }
    static member_type reduce_max( type val, mask_type mask ) {
#if __AVX512F__
	assert( 0 && "NYI -- ignore upper lanes using mask" );
	if constexpr ( std::is_signed_v<member_type> ) {
	    __m512i w = _mm512_cvtepi8_epi32( val );
	    return (member_type)_mm512_mask_reduce_max_epi32( mask, w );
	} else {
	    __m512i w = _mm512_cvtepu8_epi32( val );
	    return (member_type)_mm512_mask_reduce_max_epu32( mask, w );
	}
#else
	assert( 0 && "NYI" );
#endif
    }

    static type srli( type a, unsigned int sh ) {
	auto b = _mm_srli_epi32( a, sh );
	auto m = set1( (member_type)((1<<(W*8-sh))-1) );
	auto c = _mm_and_si128( b, m );
	return c;
    }
    static type srlv( type a, type sh ) {
	assert( 0 && "NYI" );
    }

    static type load( const member_type *a ) {
	if constexpr ( VL == 16 )
	    return _mm_load_si128( reinterpret_cast<const type *>( a ) );
	else
	    return _mm_loadu_si128( reinterpret_cast<const type *>( a ) );
    }
    static type loadu( const member_type *a ) {
	return _mm_loadu_si128( reinterpret_cast<const type *>( a ) );
    }
    static void store( member_type *addr, type val ) {
	if constexpr ( VL == 16 )
	    _mm_store_si128( (type *)addr, val );
	else
	    *reinterpret_cast<__m64 *>( addr ) = _mm_movepi64_pi64( val );
    }
    static void storeu( member_type *addr, type val ) {
	if constexpr ( VL == 16 )
	    _mm_storeu_si128( (type *)addr, val );
	else
	    *reinterpret_cast<__m64 *>( addr ) = _mm_movepi64_pi64( val );
    }
    static type ntload( const member_type *a ) {
	if constexpr ( VL == 16 )
	    return _mm_stream_load_si128( (__m128i*)const_cast<member_type *>(a) );
	else
	    assert( 0 && "NYI" );
    }
    static void ntstore( const member_type *a, type val ) {
	if constexpr ( VL == 16 )
	    return _mm_stream_si128( (__m128i*)const_cast<member_type *>(a), val );
	else
	    assert( 0 && "NYI" );
    }
#if __AVX512F__
    static type gather( const member_type *a, __m512i b, __mmask16 m ) {
	using it = avx512_4x16<uint32_t>;
	auto g = it::template gather_w<W>(
	    reinterpret_cast<const uint32_t *>( a ), b, m );
	return _mm512_cvtepi32_epi8( g );
    }
    static type gather( const member_type *a, __m512i b ) {
	using it = avx512_4x16<uint32_t>;
	auto g = it::template gather_w<W>(
	    reinterpret_cast<const uint32_t *>( a ), b );
	return _mm512_cvtepi32_epi8( g );
    }
    static type gather( const member_type *a, __m256i b, __mmask8 m ) {
	return gather( a, _mm512_castsi256_si512( b ), (__mmask16)m );
    }
    static type gather( const member_type *a, __m256i b ) {
	if constexpr ( VL == 8 ) {
	    // Cast leaves upper bits undefined, set mask
	    return gather( a, _mm512_castsi256_si512( b ), (__mmask16)0xff );
	}
	assert( 0 && "NYI -- VL16" );
    }
#else
    // Assumes __AVX2__
    static type gather( const member_type *a,
			typename avx2_4x8<uint32_t>::type b,
			typename avx2_4x8<uint32_t>::type m );
    static type gather( const member_type *a,
			typename vt_recursive<uint32_t,4,64,
			avx2_4x8<uint32_t>>::type b,
			typename vt_recursive<uint32_t,4,64,
			avx2_4x8<uint32_t>>::type m );
    // Assumes IdxT must be vpair
    template<typename IdxT>
    static type gather( const member_type *a, IdxT b );
#endif
    static type gather( const member_type *a, itype b ) {
	assert( 0 && "NYI" );
#if __AVX2__ && 0
	return _mm_i32gather_epi32( (const int *)a, b, size );
#else
	return set( a[int_traits::lane7(b)], a[int_traits::lane6(b)],
		    a[int_traits::lane5(b)], a[int_traits::lane4(b)],
		    a[int_traits::lane3(b)], a[int_traits::lane2(b)],
		    a[int_traits::lane1(b)], a[int_traits::lane0(b)] );
#endif
    }
    static type gather( const member_type *a, itype b, mask_type mask ) {
	// return _mm_mask_i32gather_epi32( setzero(), a, b, asvector(mask), size );
	assert( 0 && "NYI" );
	return setzero();
    }
/*
*/
    static type
    gather( const member_type *a, itype b, vmask_type vmask ) {
	assert( 0 && "NYI" );
#if __AVX2__
	assert( 0 && "NYI" );
	return _mm_mask_i32gather_epi32( setzero(), (const int *)a, b, vmask, size );
#else
	constexpr member_type zero = member_type(0);
	return set(
	    int_traits::lane15(vmask) ? a[int_traits::lane15(b)] : zero,
	    int_traits::lane14(vmask) ? a[int_traits::lane14(b)] : zero,
	    int_traits::lane13(vmask) ? a[int_traits::lane13(b)] : zero,
	    int_traits::lane12(vmask) ? a[int_traits::lane12(b)] : zero,
	    int_traits::lane11(vmask) ? a[int_traits::lane11(b)] : zero,
	    int_traits::lane10(vmask) ? a[int_traits::lane10(b)] : zero,
	    int_traits::lane9(vmask) ? a[int_traits::lane9(b)] : zero,
	    int_traits::lane8(vmask) ? a[int_traits::lane8(b)] : zero,
	    int_traits::lane7(vmask) ? a[int_traits::lane7(b)] : zero,
	    int_traits::lane6(vmask) ? a[int_traits::lane6(b)] : zero,
	    int_traits::lane5(vmask) ? a[int_traits::lane5(b)] : zero,
	    int_traits::lane4(vmask) ? a[int_traits::lane4(b)] : zero,
	    int_traits::lane3(vmask) ? a[int_traits::lane3(b)] : zero,
	    int_traits::lane2(vmask) ? a[int_traits::lane2(b)] : zero,
	    int_traits::lane1(vmask) ? a[int_traits::lane1(b)] : zero,
	    int_traits::lane0(vmask) ? a[int_traits::lane0(b)] : zero
	    );
#endif
    }

    static void scatter( member_type *a, itype b, type c ) {
	assert( 0 && "NYI" );
	a[int_traits::lane0(b)] = lane0(c);
	a[int_traits::lane1(b)] = lane1(c);
	a[int_traits::lane2(b)] = lane2(c);
	a[int_traits::lane3(b)] = lane3(c);
    }
    static void scatter( member_type *a, vpair<__m256i,__m256i> b, type c ) {
	assert( 0 && "NYI" );
    }
    static void scatter( member_type *a, itype b, type c, vmask_type mask ) {
	assert( 0 && "NYI" );
	if( int_traits::lane0(mask) ) a[int_traits::lane0(b)] = lane0(c);
	if( int_traits::lane1(mask) ) a[int_traits::lane1(b)] = lane1(c);
	if( int_traits::lane2(mask) ) a[int_traits::lane2(b)] = lane2(c);
	if( int_traits::lane3(mask) ) a[int_traits::lane3(b)] = lane3(c);
    }
    static void scatter( member_type *a, itype b, type c, mask_type mask ) {
	assert( 0 && "NYI" );
	if( mask_traits::lane0(mask) ) a[int_traits::lane0(b)] = lane0(c);
	if( mask_traits::lane1(mask) ) a[int_traits::lane1(b)] = lane1(c);
	if( mask_traits::lane2(mask) ) a[int_traits::lane2(b)] = lane2(c);
	if( mask_traits::lane3(mask) ) a[int_traits::lane3(b)] = lane3(c);
    }

    static void scatter( member_type *a, vpair<__m256i,__m256i> b,
			 type c, vmask_type mask );
};

template<typename T = uint8_t>
struct sse42_1x16 : public sse42_1xL<16,T> { };
    
#endif // __SSE4_2__

} // namespace target

#endif // GRAPTOR_TARGET_SSE42_1x16_H
