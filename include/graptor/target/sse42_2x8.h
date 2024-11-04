// -*- c++ -*-
#ifndef GRAPTOR_TARGET_SSE42_2x8_H
#define GRAPTOR_TARGET_SSE42_2x8_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include "graptor/target/decl.h"

#if __AVX2__
#include "graptor/target/avx2_4x8.h"
#include "graptor/target/avx2_8x4.h"
#endif // __AVX2__

#if __SSE4_2__
#include "graptor/target/sse42_bitwise.h"
#endif // __SSE4_2__

#if __AVX512F__
#include "graptor/target/avx512_8x8.h"
#endif

alignas(64) extern const uint8_t conversion_4fx8_cfp16x8_shuffle[32];
alignas(64) extern const uint8_t conversion_2x8_1x8_shuffle[32];
alignas(64) extern const uint16_t increasing_sequence_epi16[16];
alignas(64) extern const uint16_t movemask_lut_epi16[16*4];

namespace target {

/***********************************************************************
 * SSE4.2 8 short integers
 ***********************************************************************/
#if __SSE4_2__
template<unsigned short VL, typename T = uint16_t>
struct sse42_2xL : public sse42_bitwise {
    static_assert( sizeof(T) == 2, 
		   "version of template class for 2-byte integers" );
public:
    using member_type = T;
    using int_type = uint16_t;
    using type = __m128i;
    using itype = __m128i;
    using vmask_type = __m128i;

    using mask_traits = mask_type_traits<VL>;
    using mask_type = typename mask_traits::type;

    using mask_traits8 [[deprecated("needed?")]] = mask_type_traits<8>;
    using mask_type8 [[deprecated("needed?")]] = typename mask_traits8::type;

    // using half_traits = sse??_2x4<T>;
    using int_traits = sse42_2xL<VL,int_type>;
    using mt_preferred = mt_vmask;
    
    static constexpr unsigned short W = 2;
    static constexpr unsigned short B = 8 * W;
    static constexpr unsigned short vlen = VL;
    static constexpr unsigned short size = W * vlen;
    
    static member_type lane( type a, int idx ) {
	switch( idx ) {
	case 0: return member_type( (unsigned short)_mm_extract_epi16( a, 0 ) );
	case 1: return member_type( (unsigned short)_mm_extract_epi16( a, 1 ) );
	case 2: return member_type( (unsigned short)_mm_extract_epi16( a, 2 ) );
	case 3: return member_type( (unsigned short)_mm_extract_epi16( a, 3 ) );
	case 4: return member_type( (unsigned short)_mm_extract_epi16( a, 4 ) );
	case 5: return member_type( (unsigned short)_mm_extract_epi16( a, 5 ) );
	case 6: return member_type( (unsigned short)_mm_extract_epi16( a, 6 ) );
	case 7: return member_type( (unsigned short)_mm_extract_epi16( a, 7 ) );
	default: UNREACHABLE_CASE_STATEMENT;
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

    static type setoneval() {
	// http://agner.org/optimize/optimizing_assembly.pdf
	type x;
	return _mm_srli_epi16( _mm_cmpeq_epi16( x, x ), 15 );
    }
    
    static type set1( member_type a ) {
	// Casting twice is done for customfp class. Cast to uint16_t is
	// inferrable, but cast to int16_t is ambiguous.
	return _mm_set1_epi16(
	    static_cast<short int>( static_cast<unsigned short int>( a ) ) );
    }
    static type set1inc( member_type a ) {
	return add( set1( a ), set1inc0() );
    }
    static type set1inc0() {
	return load(
	    reinterpret_cast<const member_type *>( &increasing_sequence_epi16[0] ) );
    }
    static type setl0( member_type a ) {
	// return set( 0, 0, 0, 0, 0, 0, 0, a );
	return _mm_cvtsi64_si128( (uint64_t)a );
    }
    static type setl1( member_type a ) {
	return set( 0, 0, 0, 0, 0, 0, a, 0 );
    }
    static type set( member_type a7, member_type a6,
		     member_type a5, member_type a4,
		     member_type a3, member_type a2,
		     member_type a1, member_type a0 ) {
	return _mm_set_epi16(
	    (unsigned short)a7, (unsigned short)a6, (unsigned short)a5,
	    (unsigned short)a4, (unsigned short)a3, (unsigned short)a2,
	    (unsigned short)a1, (unsigned short)a0 );
    }
    static type set( member_type a3, member_type a2,
		     member_type a1, member_type a0 ) {
	return _mm_set_epi16(
	    (unsigned short)0, (unsigned short)0, (unsigned short)0,
	    (unsigned short)0, (unsigned short)a3, (unsigned short)a2,
	    (unsigned short)a1, (unsigned short)a0 );
    }

    static type set_pair( __m64 hi, __m64 lo ) {
	return _mm_set_epi64( hi, lo );
    }

    static type set_pair( type hi, type lo ) {
	if constexpr( VL == 8 ) {
	    // Use lowest half of hi and lo
	    return _mm_unpacklo_epi64( lo, hi );
	} else
	    assert( 0 && "NYI - should not occur" );
    }

    static __m64 lower_half( type a ) { return (__m64)_mm_extract_epi64( a, 0 ); }
    static __m64 upper_half( type a ) { return (__m64)_mm_extract_epi64( a, 1 ); }

    // Logical mask - only test msb
    static bool is_all_false( type a ) { return is_zero( srli( a, B-1 ) ); }

    static type bitblend( type m, type l, type r ) {
	if constexpr ( has_ternary )
	    return ternary<0xac>( m, l, r );
	else
	    return bitwise_or( bitwise_and( m, r ),
			       bitwise_andnot( m, l ) );
    }

    static type blendm( mask_type mask, type a, type b ) {
	return blend( mask, a, b );
    }
    static type blend( mask_type mask, type a, type b ) {
#if __AVX512VL__
	return _mm_mask_blend_epi16( mask, a, b );
#else
	return blend( asvector( mask ), a, b );
#endif
    }

    static type blend( type mask, type a, type b ) {
	return _mm_blendv_epi8( a, b, mask );
    }
    
    static type blendm( vmask_type m, type l, type r ) {
	// see also https://stackoverflow.com/questions/40746656/howto-vblend
	// -for-32-bit-integer-or-why-is-there-no-mm256-blendv-epi32
	// for using FP blend.
	// We are assuming that the mask is all-ones in each field (not only
	// the highest bit is used), so this works without modifying the mask.
	return _mm_blendv_epi8( l, r, m );
    }

    static type from_int( unsigned mask ) {
	return asvector( mask );
    }
    
    static mask_type movemask( vmask_type a ) {
#if __AVX512VL__ && __AVX512BW__
	return _mm_movepi16_mask( a );
#elif __AVX2__
	// Extract one bit per byte, then extract the bits in odd positions
        // unsigned int m8 = _mm_movemask_epi8( a );
        // static constexpr unsigned sel = 0xaa;
        // unsigned int m16 = _pext_u32( m8, sel );
        // return m16;
        // An alternative is to shuffle the bytes into position
        // using _mm_shuffle_epi8 (1 cycle latency) followed by
        // _mm_movemask_epi8 (1 cycle), however, this keeps a register occupied
        // and the load of the shuffle mask takes more cycles than pext.
	const type shuf = _mm_load_si128(
	    reinterpret_cast<const __m128i*>( conversion_2x8_1x8_shuffle ) );
	type b = _mm_shuffle_epi8( a, shuf );
	unsigned int m = _mm_movemask_epi8( b );
	return m;
#else
	assert( 0 && "NYI" );
#endif
    }

    static type asvector( mask_type mask ) {
	// Need to work with 8 32-bit integers as there is no 64-bit srai
	// in AVX2. Luckily, this works just as well.
#if __AVX512VL__ && __AVX512BW__
	return _mm_mask_blend_epi16( mask, setzero(), setone() );
	// return _mm_movm_epi16( mask );
#else
/* _mm_sllv_epi16 is also __AVX512VL__ && __AVX512BW__, but _mm_srai_epi16 is
   SSE2
	// In AVX etc, blend requires a constant mask, so does not apply
	type vmask = _mm_set1_epi16( (int)mask );
	const __m128i cnt = _mm_set_epi16( 8, 9, 10, 11, 12, 13, 14, 15 );
	type vmask2 = _mm_sllv_epi16( vmask, cnt );
	return _mm_srai_epi16( vmask2, 15 );
*/
	static_assert( vlen == 8, "specialised code" );
	mask_type lomask = mask & 15;
	mask_type himask = ( mask >> 4 ) & 15;
	type lo = loadu( (const member_type *)&movemask_lut_epi16[4 * lomask] );
	type hi = loadu( (const member_type *)&movemask_lut_epi16[4 * himask] );
	return set_pair( hi, lo );
#endif
    }
    template<typename T2>
    static typename vector_type_traits<T2,sizeof(T2)*8>::vmask_type
    asvector( vmask_type mask );

    static mask_type asmask( vmask_type mask ) {
	return movemask( mask );
/*
#if __AVX512VL__ && __AVX512BW__
	vmask_type m = slli( int_traits::setone(), 15 );
	auto am = _mm_and_si128( mask, m );
	return _mm_cmpeq_epi16_mask( am, m );
#endif
	assert( 0 && "NYI" );
*/
    }

    static vmask_type cmpeq( type a, type b, mt_vmask ) {
	return _mm_cmpeq_epi16( a, b );
    }
    static vmask_type cmpne( type a, type b, mt_vmask ) {
	return logical_invert( _mm_cmpeq_epi16( a, b ) );
    }
    static vmask_type cmpgt( type a, type b, target::mt_vmask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm_cmpgt_epi16( a, b );
	else {
	    type one = slli( setone(), 8*W-1 );
	    type ax = bitwise_xor( a, one );
	    type bx = bitwise_xor( b, one );
	    return _mm_cmpgt_epi16( ax, bx );
	}
    }
    static vmask_type cmpge( type a, type b, target::mt_vmask ) {
	return logical_or( cmpgt( a, b, mt_vmask() ),
			   cmpeq( a, b, mt_vmask() ) );
    }
    static vmask_type cmplt( type a, type b, target::mt_vmask ) {
	return cmpgt( b, a, target::mt_vmask() );
    }
    static vmask_type cmple( type a, type b, target::mt_vmask ) {
	return cmpge( b, a, target::mt_vmask() );
    }
#if __AVX512VL__ && __AVX512BW__
    static mask_type cmpeq( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm_cmpeq_epi16_mask( a, b );
	else
	    return _mm_cmpeq_epu16_mask( a, b );
    }
    static mask_type cmpne( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm_cmpneq_epi16_mask( a, b );
	else
	    return _mm_cmpneq_epu16_mask( a, b );
    }
    static mask_type cmpgt( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm_cmpgt_epi16_mask( a, b );
	else
	    return _mm_cmpgt_epu16_mask( a, b );
    }
    static mask_type cmpge( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm_cmpge_epi16_mask( a, b );
	else
	    return _mm_cmpge_epu16_mask( a, b );
    }
    static mask_type cmplt( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm_cmplt_epi16_mask( a, b );
	else
	    return _mm_cmplt_epu16_mask( a, b );
    }
    static mask_type cmple( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm_cmple_epi16_mask( a, b );
	else
	    return _mm_cmple_epu16_mask( a, b );
    }
#else
    static mask_type cmpeq( type a, type b, mt_mask ) {
	return asmask( cmpeq( a, b, mt_vmask() ) );
    }
    static mask_type cmpne( type a, type b, mt_mask ) {
	return asmask( cmpne( a, b, mt_vmask() ) );
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


    static bool cmpeq( type a, type b, mt_bool ) {
	type e = _mm_cmpeq_epi64( a, b );
	return _mm_test_all_ones( e );
    }

    static type add( type src, vmask_type m, type a, type b ) {
	return _mm_blendv_epi8( src, add( a, b ), m );
    }

    static type add( type a, type b ) {
	static_assert( !is_customfp_v<member_type>, "need special operation" );
	return _mm_add_epi16( a, b );
    }
    static type sub( type a, type b ) {
	static_assert( !is_customfp_v<member_type>, "need special operation" );
	return _mm_sub_epi16( a, b );
    }
    static type min( type a, type b ) {
	static_assert( !is_customfp_v<member_type>, "need special operation" );
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm_min_epi16( a, b );
	else
	    return _mm_min_epu16( a, b );
    }
    static type max( type a, type b ) {
	static_assert( !is_customfp_v<member_type>, "need special operation" );
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm_max_epi16( a, b );
	else
	    return _mm_max_epu16( a, b );
    }
    static member_type reduce_setif( type val ) {
	assert( 0 && "NYI" );
	return setzero();
    }
    static member_type reduce_add( type val ) {
	static_assert( !is_customfp_v<member_type>, "need special operation" );
	type s = _mm_hadd_epi16( val, val );
	type t = _mm_hadd_epi16( s, s );
	return lane0( t );
    }
    static member_type reduce_add( type val, mask_type mask ) {
	static_assert( !is_customfp_v<member_type>, "need special operation" );
	assert( 0 && "NYI" );
	return setzero();
    }
    static member_type reduce_add( type val, vmask_type mask ) {
	static_assert( !is_customfp_v<member_type>, "need special operation" );
	// type zval = _mm_blendv_epi8( setzero(), val, mask );
	// return reduce_add( zval );
	// First filter out zeros, then add up all values
	type x = _mm_blendv_epi8( setzero(), val, mask );
	x = _mm_hadd_epi16( x, x );
	x = _mm_hadd_epi16( x, x );
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
	static_assert( !is_customfp_v<member_type>, "need special operation" );
	return _mm_movemask_epi8( val ) ? ~member_type(0) : member_type(0);
    }
    static member_type reduce_logicalor( type val, type mask ) {
	static_assert( !is_customfp_v<member_type>, "need special operation" );
	int v = _mm_movemask_epi8( val );
	int m = _mm_movemask_epi8( mask );
	return (!m | v) ? ~member_type(0) : member_type(0);
    }
    
    static member_type reduce_min( type val ) {
	static_assert( !is_customfp_v<member_type>, "need special operation" );
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
	__mmask16 mask = 0xffffU;
	if constexpr ( std::is_signed_v<member_type> ) {
	    __m512i w = _mm512_castsi256_si512( _mm256_cvtepi16_epi32( val ) );
	    return (member_type)_mm512_mask_reduce_max_epi32( mask, w );
	} else {
	    __m512i w = _mm512_castsi256_si512( _mm256_cvtepu16_epi32( val ) );
	    return (member_type)_mm512_mask_reduce_max_epu32( mask, w );
	}
#else
	type y = max( val, _mm_bsrli_si128( val, 8 ) );
	y = max( y, _mm_bsrli_si128( y, 4 ) );
	y = max( y, _mm_bsrli_si128( y, 2 ) );
	return lane0( y );
#endif
    }
    static member_type reduce_max( type val, mask_type mask ) {
#if __AVX512F__
	if constexpr ( std::is_signed_v<member_type> ) {
	    __m512i w = _mm512_castsi256_si512( _mm256_cvtepi16_epi32( val ) );
	    return (member_type)_mm512_mask_reduce_max_epi32( mask, w );
	} else {
	    __m512i w = _mm512_castsi256_si512( _mm256_cvtepu16_epi32( val ) );
	    return (member_type)_mm512_mask_reduce_max_epu32( mask, w );
	}
#else
	assert( 0 && "NYI" );
#endif
    }

#if __AVX512F__ && __AVX512BW__
    static type sllv( type a, type b ) { return _mm_sllv_epi16( a, b ); }
    static type srlv( type a, type b ) { return _mm_srlv_epi16( a, b ); }
#endif
    static type sll( type a, __m128i b ) { return _mm_sll_epi16( a, b ); }
    static type srl( type a, __m128i b ) { return _mm_srl_epi16( a, b ); }
    static type sll( type a, long b ) {
	return sll( a, _mm_cvtsi64_si128( b ) );
    }
    static type srl( type a, long b ) {
	return srl( a, _mm_cvtsi64_si128( b ) );
    }
    static type slli( type a, unsigned int s ) {
	return _mm_slli_epi16( a, s );
    }
    static type srli( type a, unsigned int s ) {
	return _mm_srli_epi16( a, s );
    }
    static type srai( type a, unsigned int s ) {
	return _mm_srai_epi16( a, s );
    }

    template<unsigned int m>
    static type shufflelo( type a ) {
	return _mm_shufflelo_epi16( a, m );
    }

    // Don't know what 2-byte floats would look like, so keep binary pattern.
    static auto castfp( type a ) { return a; }
    static type castint( type a ) { return a; }

#if __AVX512VL__ && __AVX512F__
    static constexpr bool has_ternary = true;
#else
    static constexpr bool has_ternary = false;
#endif

    // The assumption is that the _epi32 variant of ternarylogic is correct
    // for _epi16 data as well
    template<unsigned char imm8>
    static type ternary( type a, type b, type c ) {
#if __AVX512VL__ && __AVX512F__
	return _mm_ternarylogic_epi32( a, b, c, imm8 );
#else
	assert( 0 && "NYI" );
	return setzero();
#endif
    }
    
    
    static type loadu( const member_type * a ) {
	return _mm_loadu_si128( reinterpret_cast<const type *>(a) );
    }
    static type load( const member_type * a ) {
	return _mm_load_si128( reinterpret_cast<const type *>(a) );
    }
    static void storeu( member_type * a, type v ) {
	_mm_storeu_si128( reinterpret_cast<type *>(a), v );
    }
    static void store( member_type * a , type v ) {
	_mm_store_si128( reinterpret_cast<type *>(a), v );
    }
    static type
    gather( member_type *a, itype b ) {
	return set( a[int_traits::lane7(b)], a[int_traits::lane6(b)],
		    a[int_traits::lane5(b)], a[int_traits::lane4(b)],
		    a[int_traits::lane3(b)], a[int_traits::lane2(b)],
		    a[int_traits::lane1(b)], a[int_traits::lane0(b)] );
    }
    static type
    gather( member_type *a, itype b, mask_type mask ) {
	assert( 0 && "NYI" );
	return setzero();
    }

    static type
    gather( member_type *a, itype b, vmask_type vmask ) {
	assert( 0 && "NYI" );
	return setzero();
    }
#if __AVX2__
    static type
    gather( member_type *a, typename avx2_4x8<uint32_t>::type b ) {
	assert( 0 && "NYI - need to use AVX2 gather" );
	using itraits = avx2_4x8<uint32_t>;
	return set( a[itraits::lane7(b)], a[itraits::lane6(b)],
		    a[itraits::lane5(b)], a[itraits::lane4(b)],
		    a[itraits::lane3(b)], a[itraits::lane2(b)],
		    a[itraits::lane1(b)], a[itraits::lane0(b)] );
    }
    static type
    gather( member_type *a,
	    typename avx2_4x8<uint32_t>::type b,
	    typename avx2_4x8<uint32_t>::type c ) {
	// Divide indices by 2, load 32-bit values
	// const __m256i b2 = avx2_4x8<uint32_t>::srli( b, 1 );

	// By using gather with scale of W==2, the 32-bit loads are unaligned
	// but the required 16-words end up in the lower-half of the
	// gathered 32-bit words.
	const __m256i w =
	    _mm256_mask_i32gather_epi32( avx2_4x8<uint32_t>::setzero(),
					 (const int *)a, b, c, W );

	// Compact
#if __AVX512F__ && __AVX512VL__
	__m128i f = _mm256_cvtepi32_epi16( w );
#else
	const __m256i ctrl = _mm256_load_si256(
	    reinterpret_cast<const __m256i*>( conversion_4fx8_cfp16x8_shuffle ) );
	__m256i e = _mm256_shuffle_epi8( w, ctrl );
	__m128i ehi = _mm256_extractf128_si256( e, 1 );
	__m128i elo = _mm256_castsi256_si128( e );
	__m128i f = _mm_or_si128( ehi, elo ); // note the mask is tuned for this
#endif

	return f;
    }

    static type
    gather( member_type *a,
	    typename avx2_4x8<uint32_t>::type b,
	    mask_type c ) {
	// Divide indices by 2, load 32-bit values
	// const __m256i b2 = avx2_4x8<uint32_t>::srli( b, 1 );

	// By using gather with scale of W==2, the 32-bit loads are unaligned
	// but the required 16-words end up in the lower-half of the
	// gathered 32-bit words.
	const __m256i w =
	    _mm256_mmask_i32gather_epi32( avx2_4x8<uint32_t>::setzero(),
					  c, b, (const int *)a, W );

	// Compact
#if __AVX512F__ && __AVX512VL__
	__m128i f = _mm256_cvtepi32_epi16( w );
#else
	const __m256i ctrl = _mm256_load_si256(
	    reinterpret_cast<const __m256i*>( conversion_4fx8_cfp16x8_shuffle ) );
	__m256i e = _mm256_shuffle_epi8( w, ctrl );
	__m128i ehi = _mm256_extractf128_si256( e, 1 );
	__m128i elo = _mm256_castsi256_si128( e );
	__m128i f = _mm_or_si128( ehi, elo ); // note the mask is tuned for this
#endif

	return f;
    }
#endif // __AVX2__
#if __AVX512F__
    static type
    gather( member_type *a, typename avx512_8x8<uint64_t>::type b ) {
	using itraits = vector_type_traits<uint64_t,vlen>;
	return set( a[itraits::lane7(b)], a[itraits::lane6(b)],
		    a[itraits::lane5(b)], a[itraits::lane4(b)],
		    a[itraits::lane3(b)], a[itraits::lane2(b)],
		    a[itraits::lane1(b)], a[itraits::lane0(b)] );
    }
#elif __AVX2__
    static type
    gather( member_type *a,
	    typename vt_recursive<uint64_t,8,64,avx2_8x4<uint64_t>>::type b ) {
	using itraits = vt_recursive<uint64_t,8,64,avx2_8x4<uint64_t>>;
	return set( a[itraits::lane(b,7)], a[itraits::lane(b,6)],
		    a[itraits::lane(b,5)], a[itraits::lane(b,4)],
		    a[itraits::lane(b,3)], a[itraits::lane(b,2)],
		    a[itraits::lane(b,1)], a[itraits::lane(b,0)] );
    }
#endif // __AVX512F__

#if 0
    template<typename IdxT>
    static typename std::enable_if<sizeof(IdxT) == size*vlen, type>::type
    vgather( member_type *a, IdxT b,
	     typename vector_type_traits_vl<typename int_type_of_size<sizeof(IdxT)/vlen>::type, vlen>::vmask_type vmask ) {
	assert( 0 && "NYI" );
	return _mm_mask_i32gather_epi32( setzero(), (const int *)a, b, vmask, size );
    }
    template<typename U, typename IdxT>
    static type vgather_scale( U *addr, IdxT idx, IdxT mask ) {
	assert( 0 && "NYI" );
	__m128i zero = _mm_setzero_si128();
	__m128i g = _mm_mask_i32gather_epi32( zero, (const int *)addr, idx, mask, sizeof(U) );
	__m128i filter = _mm_set1_epi32( (member_type(1)<<sizeof(U))-1 );
	__m128i gf = g & filter;
	if( std::is_same<U,bool>::value /*&& is_logical<T>::value*/ ) // convert bool to logical<T>
	    return ~_mm_cmpeq_epi32( gf, zero );
	else
	    return gf;
    }
#endif
    static void scatter( member_type *a, itype b, type c ) {
	a[int_traits::lane0(b)] = lane0(c);
	a[int_traits::lane1(b)] = lane1(c);
	a[int_traits::lane2(b)] = lane2(c);
	a[int_traits::lane3(b)] = lane3(c);
	a[int_traits::lane4(b)] = lane4(c);
	a[int_traits::lane5(b)] = lane5(c);
	a[int_traits::lane6(b)] = lane6(c);
	a[int_traits::lane7(b)] = lane7(c);
    }
    template<typename IdxT>
    static void scatter( member_type *a,
			 IdxT b,
			 type c,
			 vmask_type mask ) {
	assert( 0 && "NYI" );
	if( int_traits::lane0(mask) ) a[int_traits::lane0(b)] = lane0(c);
	if( int_traits::lane1(mask) ) a[int_traits::lane1(b)] = lane1(c);
	if( int_traits::lane2(mask) ) a[int_traits::lane2(b)] = lane2(c);
	if( int_traits::lane3(mask) ) a[int_traits::lane3(b)] = lane3(c);
    }
    template<typename IdxT>
    static void scatter( member_type *a,
			 IdxT b,
			 type c,
			 mask_type mask ) {
	assert( 0 && "NYI" );
	if( mask_traits::lane0(mask) ) a[int_traits::lane0(b)] = lane0(c);
	if( mask_traits::lane1(mask) ) a[int_traits::lane1(b)] = lane1(c);
	if( mask_traits::lane2(mask) ) a[int_traits::lane2(b)] = lane2(c);
	if( mask_traits::lane3(mask) ) a[int_traits::lane3(b)] = lane3(c);
    }

#if __AVX2__
    static void
    scatter( member_type *a, typename avx2_4x8<uint32_t>::type b, type c ) {
	using itraits = avx2_4x8<uint32_t>;
	a[itraits::lane0(b)] = lane0(c);
	a[itraits::lane1(b)] = lane1(c);
	a[itraits::lane2(b)] = lane2(c);
	a[itraits::lane3(b)] = lane3(c);
	a[itraits::lane4(b)] = lane4(c);
	a[itraits::lane5(b)] = lane5(c);
	a[itraits::lane6(b)] = lane6(c);
	a[itraits::lane7(b)] = lane7(c);
    }
#endif // __AVX2__
#if __AVX512F__
    static void
    scatter( member_type *a, typename avx512_8x8<uint64_t>::type b, type c ) {
	using itraits = vector_type_traits_vl<uint64_t,vlen>;
	a[itraits::lane0(b)] = lane0(c);
	a[itraits::lane1(b)] = lane1(c);
	a[itraits::lane2(b)] = lane2(c);
	a[itraits::lane3(b)] = lane3(c);
	a[itraits::lane4(b)] = lane4(c);
	a[itraits::lane5(b)] = lane5(c);
	a[itraits::lane6(b)] = lane6(c);
	a[itraits::lane7(b)] = lane7(c);
    }
#elif __AVX2__
    static void
    scatter( member_type *a,
	     typename vt_recursive<uint64_t,8,64,avx2_8x4<uint64_t>>::type b,
	     type c ) {
	using itraits = vt_recursive<uint64_t,8,64,avx2_8x4<uint64_t>>;
	a[itraits::lane(b,0)] = lane0(c);
	a[itraits::lane(b,1)] = lane1(c);
	a[itraits::lane(b,2)] = lane2(c);
	a[itraits::lane(b,3)] = lane3(c);
	a[itraits::lane(b,4)] = lane4(c);
	a[itraits::lane(b,5)] = lane5(c);
	a[itraits::lane(b,6)] = lane6(c);
	a[itraits::lane(b,7)] = lane7(c);
    }
#endif // __AVX512F__

    
    static member_type extract_degree( type v, unsigned degree_bits,
				       unsigned degree_shift ) {
	assert( 0 && "NYI" );
#if 0
	member_type smsk = ( (member_type(1)<<degree_bits)-1 ) << degree_shift;
	type msk = set1( smsk );
	type vs = _mm_and_si128( v, msk );

	member_type b0 = lane0( vs ) >> ( degree_shift - 0 * degree_bits );
	member_type b1 = lane1( vs ) >> ( degree_shift - 1 * degree_bits );
	member_type b2 = lane2( vs ) >> ( degree_shift - 2 * degree_bits );
	member_type b3 = lane3( vs ) >> ( degree_shift - 3 * degree_bits );

	return ( b0 | b1 ) | ( b2 | b3 );
#endif
    }
    static type extract_source( type v, unsigned degree_bits,
				unsigned degree_shift ) {
	assert( 0 && "NYI" );
#if 0
	// Written to reuse intermediate values from extract_degree()
	member_type smsk = ( (member_type(1)<<degree_bits)-1 ) << degree_shift;
	type msk = set1( smsk );
	type x = _mm_andnot_si128( msk, v );
	// Now 'sign extend' from the dropped bit
	type lx = _mm_slli_epi32( x, degree_bits );
	type rx = _mm_srai_epi32( lx, degree_bits );
	return rx;
#endif
    }
};

template<typename T = uint8_t>
struct sse42_2x8 : public sse42_2xL<8,T> { };
    
#endif // __SSE4_2__

} // namespace target

#endif // GRAPTOR_TARGET_SSE42_2x8_H
