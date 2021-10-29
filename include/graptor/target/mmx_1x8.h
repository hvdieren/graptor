// -*- c++ -*-
#ifndef GRAPTOR_TARGET_MMX_1x8_H
#define GRAPTOR_TARGET_MMX_1x8_H

#if GRAPTOR_USE_MMX

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include "graptor/target/decl.h"

#include "graptor/target/mmx_bitwise.h"
#include "graptor/target/avx2_4x8.h"

namespace target {

/***********************************************************************
 * MMX 8 byte-sized integers
 * This is poorly tested. Conversion between %xmm and %mm registers are
 * slow, support for mask-based operations is weak, generally not
 * promising.
 ***********************************************************************/
#if __MMX__
template<typename T = uint8_t>
struct mmx_1x8 : public mmx_bitwise {
    static_assert( sizeof(T) == 1, 
		   "version of template class for 1-byte integers" );
public:
    static const size_t W = 1;
    static const size_t vlen = 8;
    static const size_t size = W * vlen;

    using member_type = T;
    using type = __m64;
    using vmask_type = __m64;
    using itype = __m64;
    using int_type = uint8_t;

    using mask_traits = mask_type_traits<8>;
    using mask_type = typename mask_traits::type;

    // using half_traits = vector_type_int_traits<member_type,16>;
    // using recursive_traits = vt_recursive<member_type,1,32,half_traits>;
    using int_traits = mmx_1x8<int_type>;

    // Methods to switch easily between sse42_1x8 and mmx_1x8
    static uint64_t asint( type a ) {
	return _mm_cvtm64_si64( a );
    }
    
    static type set1( member_type a ) {
	return _mm_set1_pi8( a );
    }
    static type set( member_type a7, member_type a6,
		     member_type a5, member_type a4,
		     member_type a3, member_type a2,
		     member_type a1, member_type a0 ) {
	return _mm_set_pi8( a7, a6, a5, 4, a3, a2, a1, a0 );
    }
    static type set1inc( member_type a ) {
	return type(0x0706050403020100ULL) + set1(a);
    }

    static member_type lane( type a, int idx ) {
	int x;
	switch( idx/2 ) {
	case 0:
	    x = _mm_extract_pi16( a, 0 );
	    break;
	case 1:
	    x = _mm_extract_pi16( a, 1 );
	    break;
	case 2:
	    x = _mm_extract_pi16( a, 2 );
	    break;
	case 3:
	    x = _mm_extract_pi16( a, 3 );
	    break;
	}
	return static_cast<member_type>( ( (idx&1) ? x >> 8 : x ) & 0xff );
    }
/*
    static member_type lane0( type a ) { return (member_type)(a & 0xff); }
    static member_type lane1( type a ) { return (member_type)((a >> 8) & 0xff); }
    static member_type lane2( type a ) { return (member_type)((a >> 16) & 0xff); }
    static member_type lane3( type a ) { return (member_type)((a >> 24) & 0xff); }
*/

    static type load( member_type *addr ) {
	return *(type *)addr;
    }
    static type loadu( member_type *addr ) {
	return *(type *)addr;
    }
    static void store( member_type *addr, type val ) {
	*(type *)addr = val;
    }
    static void storeu( member_type *addr, type val ) {
	*(type *)addr = val;
    }
    
    static mask_type asmask( vmask_type mask ) {
	return _mm_movemask_pi8( mask );
    }
    static vmask_type asvector( mask_type mask ) {
#if __AVX512VL__ && __AVX512BW__
	using it = sse42_4x4<uint32_t>;
	auto b = _mm_mask_blend_epi8( mask, it::setzero(), it::setone() );
	return _mm_cvtsi64_m64( _mm_cvtsi128_si64( b ) );
#else
	assert( 0 && "NYI" );
#endif
    }
    
    static vmask_type cmpeq( type a, type b, mt_vmask ) {
	return _mm_cmpeq_pi8( a, b );
    }
    static vmask_type cmpne( type a, type b, mt_vmask ) {
	return logical_invert( cmpeq( a, b, mt_vmask() ) );
    }
    static vmask_type cmpgt( type a, type b, mt_vmask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm_cmpgt_pi8( a, b );
	else {
	    type one = set1( 0x80 );
	    type ax = bitwise_xor( a, one );
	    type bx = bitwise_xor( b, one );
	    return _mm_cmpgt_pi8( ax, bx );
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

    static mask_type cmpeq( type a, type b, mt_mask ) {
	return asmask( cmpeq( a, b, mt_vmask() ) );
    }
    static mask_type cmpne( type a, type b, mt_mask ) {
	assert( 0 && "Fails unnit test" );
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

    static type blendm( vmask_type m, type l, type r ) {
	return _mm_or_si64( _mm_andnot_si64( m, l ),
			    _mm_and_si64( m, r ) );
    }
    static type blendm( mask_type m, type l, type r ) {
	return blendm( asvector( m ), l, r );
    }
    static type blend( vmask_type m, type l, type r ) {
	return blendm( m, l, r );
    }
    static type blend( mask_type m, type l, type r ) {
	return blendm( m, l, r );
    }
    
    static type srli( type a, unsigned short sh ) {
	auto b = _mm_srli_pi32( a, sh );
	auto m = set1( (member_type)((1<<(W*8-sh))-1) );
	auto c = _mm_and_si64( b, m );
	return c;
    }

    static type srlv( type a, type sh ) {
	assert( 0 && "NYI" );
    }

    static type add( type a, type b ) {
	return _mm_add_pi8( a, b );
    }
    static type sub( type a, type b ) {
	return _mm_sub_pi8( a, b );
    }
    static type min( type a, type b ) {
#if __SSE__
	if constexpr ( !std::is_signed_v<member_type> )
	    return _mm_min_pu8( a, b );
#endif
	return blend( cmpgt( a, b, mt_vmask() ), a, b );
    }
    static type max( type a, type b ) {
#if __SSE__
	if constexpr ( !std::is_signed_v<member_type> )
	    return _mm_max_pu8( a, b );
#endif
	return blend( cmpgt( a, b, mt_vmask() ), b, a );
    }

    static member_type reduce_max( type a ) {
	assert( 0 && "NYI" );
    }

    static mask_type movemask( vmask_type a ) {
	return _mm_movemask_pi8( a );
    }
/*
    static mask_type packbyte( type a ) {
	a = a & 0x01010101;
	a |= a >> 14;
	a |= a >> 7;
	return a & 0xf;
	// return ( a | (a >> 14) | (a >> 7) | (a >> 21) ) & 0xf;
    }
    // TODO: need proper support for bool's that are all ones
    static __m256i expandmask8( type a ) { // a are boolean bytes
	__m128i ones = _mm_cvtsi32_si128( 0x01010101 );
	__m128i x = _mm_cvtsi32_si128( ~a ); // load inverse in vector (lo half)
	__m128i y = _mm_add_epi8( ones, x ); // convert 0/1 to all 0 or all 1
	return _mm256_cvtepi8_epi64( y );
    }
*/
    static member_type reduce_setif( type val ) {
	assert( 0 && "NYI" );
	return setzero(); // TODO
    }
    static member_type reduce_add( type val ) {
	assert( 0 && "NYI" );
	return setzero(); // _mm256_reduce_add_epi64( val );
    }
    static member_type reduce_add( type val, mask_type mask ) {
	assert( 0 && "NYI" );
	return setzero(); // _mm256_reduce_add_epi64( val );
    }

    static type convert_4_1( __m256i a ) {
	auto c = convert_4b_1b( a );
	return _mm_cvtsi64_m64( _mm_cvtsi128_si64( c ) );
    }
    
    static type gather( const member_type *a, __m256i b ) {
	using it = avx2_4x8<uint32_t>;
	__m256i g = it::template gather_w<W>(
	    reinterpret_cast<const uint32_t *>( a ), b );
	return convert_4_1( g );
    }
    static type gather( const member_type *a, __m256i b, __m256i m ) {
	using it = avx2_4x8<uint32_t>;
	__m256i g = it::template gather_w<W>(
	    reinterpret_cast<const uint32_t *>( a ), b, m );
	return convert_4_1( g );
    }
    static type gather( const member_type *a, __m256i b, mask_type m ) {
	using it = avx2_4x8<uint32_t>;
	__m256i g = it::template gather_w<W>(
	    reinterpret_cast<const uint32_t *>( a ), b, m );
	return convert_4_1( g );
    }

    template<typename IdxT>
    static void scatter( member_type *a, IdxT b, type c ) {
	assert( 0 && "NYI" );
    }
};
#endif // __MMX__

} // namespace target

#else // GRAPTOR_USE_MMX

#include "graptor/target/sse42_1x8.h"

#endif // GRAPTOR_USE_MMX

#endif // GRAPTOR_TARGET_MMX_1x8_H
