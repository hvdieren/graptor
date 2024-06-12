// -*- c++ -*-
#ifndef GRAPTOR_TARGET_AVX2_2x16_H
#define GRAPTOR_TARGET_AVX2_2x16_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include "graptor/target/decl.h"
#include "graptor/target/avx2_bitwise.h"
#include "graptor/target/sse42_2x8.h"
#include "graptor/target/vt_recursive.h"

#if __AVX512F__
#include "graptor/target/avx512_4x16.h"
#endif

alignas(64) extern const uint8_t conversion_2x8_1x8_shuffle[32];

namespace target {

/***********************************************************************
 * AVX2 8 integers
 ***********************************************************************/
#if __AVX2__
template<typename T = uint16_t>
struct avx2_2x16 : public avx2_bitwise {
    static_assert( sizeof(T) == 2, 
		   "version of template class for 2-byte integers" );
public:
    using member_type = T;
    using type = __m256i;
    using vmask_type = __m256i;
    using itype = __m256i;
    using int_type = uint16_t;

    using mask_traits = mask_type_traits<16>;
    using mask_type = typename mask_traits::type;
    using vmask_traits = avx2_2x16<uint16_t>;

    using mt_preferred = target::mt_vmask;

    // using half_traits = vector_type_int_traits<member_type,16>;
    using half_traits = sse42_2x8<member_type>;
    using recursive_traits = vt_recursive<member_type,2,32,half_traits>;
    using int_traits = avx2_2x16<int_type>;
    
    static constexpr size_t W = 2;
    static constexpr size_t B = 8*W;
    static constexpr size_t vlen = 16;
    static constexpr size_t size = W * vlen;
    
    static member_type lane( type a, int idx ) {
	switch( idx ) {
	case 0: return (member_type) _mm256_extract_epi16( a, 0 );
	case 1: return (member_type) _mm256_extract_epi16( a, 1 );
	case 2: return (member_type) _mm256_extract_epi16( a, 2 );
	case 3: return (member_type) _mm256_extract_epi16( a, 3 );
	case 4: return (member_type) _mm256_extract_epi16( a, 4 );
	case 5: return (member_type) _mm256_extract_epi16( a, 5 );
	case 6: return (member_type) _mm256_extract_epi16( a, 6 );
	case 7: return (member_type) _mm256_extract_epi16( a, 7 );
	case 8: return (member_type) _mm256_extract_epi16( a, 8 );
	case 9: return (member_type) _mm256_extract_epi16( a, 9 );
	case 10: return (member_type) _mm256_extract_epi16( a, 10 );
	case 11: return (member_type) _mm256_extract_epi16( a, 11 );
	case 12: return (member_type) _mm256_extract_epi16( a, 12 );
	case 13: return (member_type) _mm256_extract_epi16( a, 13 );
	case 14: return (member_type) _mm256_extract_epi16( a, 14 );
	case 15: return (member_type) _mm256_extract_epi16( a, 15 );
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

    static type setone_shr1() {
	return _mm256_srli_epi16( setone(), 1 );
    }
    static type setoneval() {
	// http://agner.org/optimize/optimizing_assembly.pdf
	type x;
	return _mm256_srli_epi16( _mm256_cmpeq_epi32( x, x ), 15 );
    }
    
    static type set1( member_type a ) {
	return _mm256_set1_epi16( (unsigned short)a ); // in case of customfp
    }
    static type set1inc( member_type a ) {
	return add( set1( a ), set1inc0() );
    }
    static type set1inc0() {
	return load(
	    reinterpret_cast<const member_type *>( &increasing_sequence_epi16[0] ) );
    }
/*
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
*/
    
    static type setl0( member_type a ) {
	// const member_type z( 0 );
	// return set( z, z, z, z, z, z, z, a );
	return _mm256_zextsi128_si256( _mm_cvtsi64_si128( (uint64_t)a ) );
    }

    static type blendm( vmask_type m, type l, type r ) {
	// see also https://stackoverflow.com/questions/40746656/howto-vblend
	// -for-32-bit-integer-or-why-is-there-no-mm256-blendv-epi32
	// for using FP blend.
	// We are assuming that the mask is all-ones in each field (not only
	// the highest bit is used), so this works without modifying the mask.
	return _mm256_blendv_epi8( l, r, m );
    }
    static type blendm( mask_type m, type l, type r ) {
	return blendm( asvector( m ), l, r );
    }

    static type from_int( unsigned mask ) {
	return asvector( mask );
    }
    
    static mask_type movemask( vmask_type a ) {
#if __AVX512VL__ && __AVX512BW__
	return _mm256_movepi16_mask( a );
#else
	// vmask_type mask = srli( setone(), 1 );
	// return _mm256_cmpgt_epu16_mask( a, mask );
	assert( 0 && "NYI" );
#endif
    }

    static type asvector( mask_type mask ) {
#if __AVX512BW__
	return _mm256_mask_blend_epi16( mask, setzero(), setone() );
#endif
	assert( 0 && "NYI" );
	return type();
    }
    template<typename T2>
    static typename vector_type_traits<T2,sizeof(T2)*8>::vmask_type
    asvector( vmask_type mask );

    static mask_type asmask( vmask_type mask ) {
#if __AVX512F__ && __AVX512VL__ && __AVX512BW__
	return _mm256_movepi16_mask( mask );
#else
	// Extract one bit per byte, then extract the bits in odd positions
	// The alternative, shuffling bytes before movemask_epi8, is less
	// interesting here as the bytes aren't easily shuffled across SSE lanes
        unsigned int m8 = _mm256_movemask_epi8( mask );
        static constexpr unsigned sel = 0xaaaaaaaa;
        unsigned int m16 = _pext_u32( m8, sel );
        return m16;
#endif
    }

    // There is no native 16-bit FP type
    // static __m256 castfp( type a ) { ... }

    static type add( type src, vmask_type m, type a, type b ) {
	type sum = add( a, b );
	return _mm256_blendv_epi8( src, sum, m );
    }

    static type add( type a, type b ) { return _mm256_add_epi16( a, b ); }
    static type sub( type a, type b ) { return _mm256_sub_epi16( a, b ); }
    // static type mul( type a, type b ) { return _mm256_mul_epi16( a, b ); }

    static type min( type a, type b ) {
#if __AVX512VL__ && __AVX512BW__
	auto cmp = cmpgt( a, b, target::mt_mask() );
	return blend( cmp, a, b );
#else
	auto cmp = cmpgt( a, b, target::mt_vmask() );
	return blend( cmp, a, b );
#endif
    }
    static type max( type a, type b ) {
#if __AVX512VL__ && __AVX512BW__
	auto cmp = cmpgt( a, b, target::mt_mask() );
	return blend( cmp, b, a );
#else
	auto cmp = cmpgt( a, b, target::mt_vmask() );
	return blend( cmp, b, a );
#endif
    }

    static vmask_type cmpeq( type a, type b, mt_vmask ) {
	return _mm256_cmpeq_epi16( a, b );
    }
    static vmask_type cmpne( type a, type b, mt_vmask ) {
	return ~cmpeq( a, b, mt_vmask() );
    }
    static vmask_type cmpge( type a, type b, mt_vmask tag ) {
	return logical_or( cmpgt( a, b, tag ), cmpeq( a, b, tag ) );
    }
    static vmask_type cmple( type a, type b, mt_vmask tag ) {
	return logical_or( cmplt( a, b, tag ), cmpeq( a, b, tag ) );
    }
    static vmask_type cmplt( type a, type b, mt_vmask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm256_cmpgt_epi16( b, a );
	else {
#if __AVX512F__ && __AVX512BW__
	    return asvector( _mm256_cmpgt_epu16_mask( b, a ) );
#else
	    type one = slli( setone(), 8*W-1 );
	    type ax = bitwise_xor( a, one );
	    type bx = bitwise_xor( b, one );
	    return _mm256_cmpgt_epi16( bx, ax );
#endif
	}
    }
    static vmask_type cmpgt( type a, type b, mt_vmask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm256_cmpgt_epi16( a, b );
	else {
#if __AVX512F__ && __AVX512BW__
	    return asvector( _mm256_cmpgt_epu16_mask( a, b ) );
#else
	    type one = slli( setone(), 8*W-1 );
	    type ax = bitwise_xor( a, one );
	    type bx = bitwise_xor( b, one );
	    return _mm256_cmpgt_epi16( ax, bx );
#endif
	}
    }

#if __AVX512VL__ && __AVX512BW__
    static mask_type cmpgt( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm256_cmpgt_epi16_mask( a, b );
	else
	    return _mm256_cmpgt_epu16_mask( a, b );
    }
    static mask_type cmplt( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm256_cmplt_epi16_mask( a, b );
	else
	    return _mm256_cmplt_epu16_mask( a, b );
    }
    static mask_type cmpge( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm256_cmpge_epi16_mask( a, b );
	else
	    return _mm256_cmpge_epu16_mask( a, b );
    }
    static mask_type cmple( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm256_cmple_epi16_mask( a, b );
	else
	    return _mm256_cmple_epu16_mask( a, b );
    }
    static mask_type cmpeq( type a, type b, mt_mask ) {
	return _mm256_cmpeq_epi16_mask( a, b );
    }
    static mask_type cmpne( type a, type b, mt_mask ) {
	return _mm256_cmpneq_epi16_mask( a, b );
    }
    static mask_type cmpne( mask_type m, type a, type b, mt_mask ) {
	return _mm256_mask_cmpneq_epi16_mask( m, a, b );
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
    static mask_type cmpge( mask_type m, type a, type b, mt_mask ) {
	return mask_traits::logical_and( m, cmpge( a, b, mt_mask() ) );
    }
    static mask_type cmplt( mask_type m, type a, type b, mt_mask ) {
	return mask_traits::logical_and( m, cmplt( a, b, mt_mask() ) );
    }
    static mask_type cmpeq( mask_type m, type a, type b, mt_mask ) {
	return mask_traits::logical_and( m, cmpeq( a, b, mt_mask() ) );
    }
    static mask_type cmpne( mask_type m, type a, type b, mt_mask ) {
	return mask_traits::logical_and( m, cmpne( a, b, mt_mask() ) );
    }
    static mask_type cmpge( vmask_type m, type a, type b, mt_mask ) {
	return asmask( cmpge( m, a, b, mt_vmask() ) );
    }
    static mask_type cmplt( vmask_type m, type a, type b, mt_mask ) {
	return asmask( cmplt( m, a, b, mt_vmask() ) );
    }
#endif

    static vmask_type cmpeq( mask_type m, type a, type b, mt_vmask ) {
	return cmpeq( asvector( m ), a, b, mt_vmask() );
    }
    static vmask_type cmpeq( vmask_type m, type a, type b, mt_vmask ) {
	return vmask_traits::logical_and( m, cmpeq( a, b, mt_vmask() ) );
    }
    static vmask_type cmpge( vmask_type m, type a, type b, mt_vmask ) {
	return vmask_traits::logical_and( m, cmpge( a, b, mt_vmask() ) );
    }
    static vmask_type cmplt( vmask_type m, type a, type b, mt_vmask ) {
	return vmask_traits::logical_and( m, cmplt( a, b, mt_vmask() ) );
    }

    static bool cmpne( type a, type b, mt_bool ) {
	vmask_type ne = cmpne( a, b, mt_vmask() );
	return ! is_zero( ne );
    }
    static type blend( mask_type mask, type a, type b ) {
#if __AVX512VL__ && __AVX512BW__
	return _mm256_mask_blend_epi16( mask, a, b );
#else
	return _mm256_blendv_epi8( a, b, asvector( mask ) );
#endif
    }
    static type blend( vmask_type mask, type a, type b ) {
	return _mm256_blendv_epi8( a, b, mask );
    }
    static type blend( __m128i mask, type a, type b ) {
	return _mm256_blendv_epi8( a, b, _mm256_cvtepi8_epi16( mask ) );
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
	return half_traits::reduce_add(
	    half_traits::add( lower_half( val ), upper_half( val ) ) );
	return member_type();
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
	assert( 0 && "NYI" );
	return member_type();
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
	assert( 0 && "NYI" );
	return member_type();
    }
    static type sll( type a, __m128i b ) { return _mm256_sll_epi16( a, b ); }
    static type srl( type a, __m128i b ) { return _mm256_srl_epi16( a, b ); }
    static type slli( type a, unsigned int s ) {
	    return _mm256_slli_epi16( a, s );
    }
    static type srli( type a, unsigned int s ) {
	return _mm256_srli_epi16( a, s );
    }
    static type srai( type a, unsigned int s ) {
	return _mm256_srai_epi16( a, s );
    }
    static type sllv( type a, type sh ) {
	return _mm256_sllv_epi16( a, sh );
    }
    static type srlv( type a, type sh ) {
	return _mm256_srlv_epi16( a, sh );
    }

    static type castfp( type a ) { return a; } // 16-bit float is customfp
    static type castint( type a ) { return a; }

    static mask_type intersect( type a, const member_type * b ) {
	// This code is claimed to be faster than the vp2intersect instruction
	// https://arxiv.org/pdf/2112.06342.pdf
	mask_type m00 = cmpne( a, set1( b[0] ), mt_mask() );
	mask_type m01 = cmpne( a, set1( b[1] ), mt_mask() );
	mask_type m02 = cmpne( a, set1( b[2] ), mt_mask() );

	mask_type m03 = cmpne( m00, a, set1( b[3] ), mt_mask() );
	mask_type m04 = cmpne( m01, a, set1( b[4] ), mt_mask() );
	mask_type m05 = cmpne( m02, a, set1( b[5] ), mt_mask() );
	mask_type m06 = cmpne( m03, a, set1( b[6] ), mt_mask() );
	mask_type m07 = cmpne( m04, a, set1( b[7] ), mt_mask() );
	mask_type m08 = cmpne( m05, a, set1( b[8] ), mt_mask() );
	mask_type m09 = cmpne( m06, a, set1( b[9] ), mt_mask() );
	mask_type m10 = cmpne( m07, a, set1( b[10] ), mt_mask() );
	mask_type m11 = cmpne( m08, a, set1( b[11] ), mt_mask() );
	mask_type m12 = cmpne( m09, a, set1( b[12] ), mt_mask() );
	mask_type m13 = cmpne( m10, a, set1( b[13] ), mt_mask() );
	mask_type m14 = cmpne( m11, a, set1( b[14] ), mt_mask() );
	mask_type m15 = cmpne( m12, a, set1( b[15] ), mt_mask() );

	return mask_traits::logical_invert(
	    mask_traits::logical_and(
		m13, mask_traits::logical_and( m14, m15 ) ) );
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

    static type gather( const member_type *a, type b ) {
	assert( 0 && "NYI" );
	return type();
	// return _mm256_i32gather_epi32( (const int *)a, b, W );
    }
#if __AVX512F__
    static type gather( member_type *a, __m512i b ) {
	using it = avx512_4x16<uint32_t>;
	auto g = it::template gather_w<W>(
	    reinterpret_cast<const uint32_t *>( a ), b );
	return _mm512_cvtepi32_epi16( g );
    }
    static type gather( member_type *a, __m512i b, __mmask16 m ) {
	using it = avx512_4x16<uint32_t>;
	auto g = it::template gather_w<W>(
	    reinterpret_cast<const uint32_t *>( a ), b, m );
	return _mm512_cvtepi32_epi16( g );
    }
#else
#endif
    GG_INLINE
    static type gather( member_type *a, vpair<itype,itype> b ) {
	// This version uses 4-byte integers
	__m128i lo = half_traits::gather( a, b.a );
	__m128i hi = half_traits::gather( a, b.b );
	return set_pair( hi, lo );
    }
    static type gather( const member_type *a, type b, mask_type mask ) {
	assert( 0 && "NYI" );
	return type();
	// return _mm256_mask_i32gather_epi32(
	    // setzero(), reinterpret_cast<const int *>( a ),
	    // b, asvector(mask), W );
    }
    static type gather( const member_type *a, __m512i b, mask_type mask ) {
	assert( 0 && "NYI" );
	return type();
	// return _mm256_mask_i32gather_epi32(
	    // setzero(), reinterpret_cast<const int *>( a ),
	    // b, asvector(mask), W );
    }
    GG_INLINE
    static type gather( member_type *a, vpair<itype,itype> b, mask_type mask ) {
	// This version uses 4-byte integers
	__m128i lo = half_traits::gather( a, b.a, mask_traits::lower_half( mask ) );
	__m128i hi = half_traits::gather( a, b.b, mask_traits::upper_half( mask ) );
	return set_pair( hi, lo );
    }
    static type gather( const member_type *a, itype b, vmask_type vmask ) {
	assert( 0 && "NYI" );
	return type();
	// return _mm256_mask_i32gather_epi32( setzero(), (const int *)a, b, vmask, W );
    }
#if __AVX512F__
    GG_INLINE
    static type gather( member_type *a, __m512i b, __m512i vmask ) {
	// This version uses 4-byte integers
	assert( 0 && "NYI" );
    }
#endif
    GG_INLINE
    static type gather( member_type *a, vpair<itype,itype> b,
			vpair<itype,itype> vmask ) {
	// This version uses 4-byte integers
	__m128i lo = half_traits::gather( a, b.a, vmask.a );
	__m128i hi = half_traits::gather( a, b.b, vmask.b );
	return set_pair( hi, lo );
    }

    GG_INLINE
    static void scatter( member_type *a, itype b, type c ) {
#if __AVX512F__
	assert( 0 && "Use 512-bit scatter with mask" );
#else
	assert( 0 && "NYI" ); // decompose by halves (just one extract)
#endif
    }
#if __AVX512F__
    GG_INLINE
    static void scatter( member_type *a, __m512i b, type c ) {
	// This version uses 4-byte integers
	assert( 0 && "NYI" );
    }
#endif
    GG_INLINE
    static void scatter( member_type *a, vpair<itype,itype> b, type c ) {
	// This version uses 4-byte integers
	half_traits::scatter( a, b.a, lower_half( c ) );
	half_traits::scatter( a, b.b, upper_half( c ) );
    }
    GG_INLINE
    static void scatter( member_type *a, itype b, type c, vmask_type mask ) {
#if __AVX512F__
	assert( 0 && "Use 512-bit scatter with mask" );
#else
	assert( 0 && "NYI" ); // decompose by halves (just one extract)
#endif
    }
    GG_INLINE
    static void scatter( member_type *a, itype b, type c, mask_type mask ) {
#if __AVX512F__
	assert( 0 && "Use 512-bit scatter with mask" );
#else
	assert( 0 && "NYI" ); // decompose by halves (just one extract)
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
};

#endif // __AVX2__

} // namespace target

#endif // GRAPTOR_TARGET_AVX2_2x16_H
