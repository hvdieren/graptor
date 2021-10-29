// -*- c++ -*-
#ifndef GRAPTOR_TARGET_AVX512_2x32_H
#define GRAPTOR_TARGET_AVX512_2x32_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include "graptor/target/decl.h"

#if __AVX2__
#include "graptor/target/avx2_2x16.h"
#endif

namespace target {

/***********************************************************************
 * AVX512 16 4-byte integers
 ***********************************************************************/
#if __AVX512F__
template<typename T = uint16_t>
struct avx512_2x32 {
    static_assert( sizeof(T) == 2, 
		   "version of template class for 2-byte integers" );
public:
    static constexpr size_t W = 2;
    static constexpr size_t B = 8*W;
    static constexpr size_t vlen = 32;
    static constexpr size_t size = W * vlen;
    
    using member_type = T;
    using type = __m512i;
    using vmask_type = __m512i;
    using mask_traits = mask_type_traits<vlen>;
    using mask_type = typename mask_traits::type;
    using itype = __m512i;
    using int_type = uint16_t;
    using mt_preferred = mt_mask;

    using int_traits = avx512_2x32<int_type>;
    using half_traits = avx2_2x16<member_type>;
    using recursive_traits = vt_recursive<member_type,2,32,half_traits>;

    static member_type lane( type a, int idx ) {
	if( idx < 16 )
	    return half_traits::lane( lower_half( a ), idx );
	else
	    return half_traits::lane( upper_half( a ), idx-16 );
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

    static __m256i lower_half( type a ) {
	return _mm512_castsi512_si256( a );
    }
    static __m256i upper_half( type a ) {
	return _mm512_extracti64x4_epi64( a, 1 );
    }
    static vpair<__m256i,__m256i> decompose( type a ) {
	return vpair<__m256i,__m256i>{ lower_half(a), upper_half(a) };
    }
    static type set_pair( vpair<__m256i,__m256i> p ) {
	return set_pair( p.b, p.a );
    }
    static type set_pair( __m256i up, __m256i lo ) {
	return _mm512_inserti64x4( _mm512_castsi256_si512( lo ), up, 1 );
    }

    static type setone() {
	// Recommended here:
	// https://stackoverflow.com/questions/45105164/set-all-bits-in-cpu-register-to-1-efficiently/45113467#45113467
	__m512i x;
	return _mm512_ternarylogic_epi32( x, x, x, 0xff );
    }
    static type setone_shr1() {
	return srli( setone(), 1 );
    }
    static type setoneval() { // 0x00000001 repeated
	// http://agner.org/optimize/optimizing_assembly.pdf
	return srli( setone(), 31 );
    }
    static type set_maskz( mask_type m, type a ) {
	return _mm512_maskz_mov_epi16( m, a );
    }
    
    static type set1( member_type a ) { return _mm512_set1_epi16( a ); }
    static itype set1inc( member_type a ) {
	return add( set1( a ), set1inc0() );
    }
    static itype set1inc0() {
	return int_traits::load(
	    static_cast<const int_type *>( &increasing_sequence_epi16[0] ) );
    }
    static type setzero() { return _mm512_setzero_epi32(); }
    static type setl0( member_type a ) {
	// return (type)a;
	return _mm512_zextsi128_si512( _mm_cvtsi64_si128( (uint64_t)a ) );
    }

    template<typename VecT2>
    static auto convert( VecT2 a ) { // TODO
	assert( 0 && "NYI" );
	/*
	typedef vector_type_traits<
	    typename int_type_of_size<sizeof(VecT2)/vlen>::type,
	    sizeof(VecT2)> traits2;
	return set( traits2::lane7(a), traits2::lane6(a),
		    traits2::lane5(a), traits2::lane4(a),
		    traits2::lane3(a), traits2::lane2(a),
		    traits2::lane1(a), traits2::lane0(a) );
	*/
    }

    // Needs specialisations!
    // Casting here is to do a signed cast of a bitmask for logical masks
    template<typename T2>
    static auto convert_to( type a ) {
	assert( 0 && "NYI" );
/*
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
*/
    }

    static type from_int( unsigned mask ) {
	auto ones = setone();
	auto zero = setzero();
	return _mm512_mask_blend_epi16( mask, zero, ones );
    }
    
    static mask_type movemask( vmask_type a ) {
#if __AVX512BW__
	return _mm512_movepi16_mask( a );
#else
	// vmask_type mask = srli( setone(), 1 );
	// return _mm512_cmpgt_epu16_mask( a, mask );
	assert( 0 && "NYI" );
#endif
    }

    static mask_type asmask( vmask_type a ) { return movemask( a ); }
    static type asvector( mask_type mask ) {
	auto ones = setone();
	auto zero = setzero();
	return _mm512_mask_blend_epi16( mask, zero, ones );
    }

    static type blendm( mask_type m, type l, type r ) {
	return _mm512_mask_blend_epi16( m, l, r );
    }
    static type blend( mask_type m, type l, type r ) {
	return _mm512_mask_blend_epi16( m, l, r );
    }
    static type blend( vmask_type m, type l, type r ) {
	return blend( asmask( m ), l, r  );
    }
    static type bitblend( vmask_type m, type l, type r ) {
	if constexpr ( has_ternary ) {
	    return ternary<0xac>( m, l, r );
	} else {
	    return bitwise_or( bitwise_and( m, r ),
			       bitwise_andnot( m, l ) );
	}
    }

    static type sra1( type a ) { return _mm512_srai_epi16( a, 1 ); }

    static type add( type a, type b ) { return _mm512_add_epi16( a, b ); }
    static type sub( type a, type b ) { return _mm512_sub_epi16( a, b ); }
    // static type mul( type a, type b ) { return _mm512_mul_epi16( a, b ); }
    // static type div( type a, type b ) { return _mm512_div_epi32( a, b ); }

    static type add( type src, mask_type m, type a, type b ) {
	return _mm512_mask_add_epi16( src, m, a, b );
    }
    static type add( type src, vmask_type m, type a, type b ) {
	return add( src, asmask( m ), a, b );
    }
    static type min( type a, type b ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm512_min_epi16( a, b );
	else
	    return _mm512_min_epu16( a, b );
    }
    static type max( type a, type b ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm512_max_epi16( a, b );
	else
	    return _mm512_max_epu16( a, b );
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
	return _mm512_cmpeq_epi16_mask( a, b );
    }
    static mask_type cmpne( type a, type b, mt_mask ) {
	return _mm512_cmpneq_epi16_mask( a, b );
    }
    static bool cmpne( type a, type b, mt_bool ) {
	mask_type ne = cmpne( a, b, mt_mask() );
	return ! _kortestz_mask32_u8( ne, ne );
    }
    static mask_type cmpgt( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm512_cmpgt_epi16_mask( a, b );
	else
	    return _mm512_cmpgt_epu16_mask( a, b );
    }
    static mask_type cmpge( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm512_cmpge_epi16_mask( a, b );
	else
	    return _mm512_cmpge_epu16_mask( a, b );
    }
    static mask_type cmplt( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm512_cmplt_epi16_mask( a, b );
	else
	    return _mm512_cmplt_epu16_mask( a, b );
    }
    static mask_type cmple( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm512_cmple_epi16_mask( a, b );
	else
	    return _mm512_cmple_epu16_mask( a, b );
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

/*
    static member_type reduce_setif( type val ) {
	// Pick any, preferrably one that is not -1
	// Could also use min_epu64
	return _mm512_reduce_max_epi16( val );
    }
    static member_type reduce_setif( type val, mask_type mask ) {
	// Pick any, preferrably one that is not -1
	// Could also use min_epu64
	return _mm512_mask_reduce_max_epi16( mask, val );
    }
    static member_type reduce_add( type val ) {
	return _mm512_reduce_add_epi16( val );
    }
    static member_type reduce_add( type val, mask_type mask ) {
	return _mm512_mask_reduce_add_epi16( mask, val );
    }
    static member_type reduce_logicalor( type val ) {
	return member_type( _mm512_reduce_or_epi16( val ) );
    }
    static member_type reduce_logicalor( type val, mask_type mask ) {
	return _mm512_mask_reduce_or_epi16( mask, val );
    }
    static member_type reduce_bitwiseor( type val ) {
	return member_type( _mm512_reduce_or_epi16( val ) );
    }
    static member_type reduce_bitwiseor( type val, mask_type mask ) {
	return member_type( _mm512_mask_reduce_or_epi16( mask, val ) );
    }
    
    static member_type reduce_min( type val ) {
	return _mm512_reduce_min_epi16( val );
    }
    static member_type reduce_min( type val, mask_type mask ) {
	return _mm512_mask_reduce_min_epi16( mask, val );
    }
*/

    static member_type reduce_max( type val ) {
	auto lo = lower_half( val );
	auto hi = upper_half( val );
	return half_traits::reduce_max( half_traits::max( lo, hi ) );
    }
    static member_type reduce_max( type val, mask_type mask ) {
	auto lo = lower_half( val );
	auto hi = upper_half( val );
	auto mlo = mask_traits::lower_half( mask );
	auto mhi = mask_traits::upper_half( mask );
	return half_traits::reduce_max(
	    half_traits::blend(
		half_traits::mask_traits::logical_or(
		    half_traits::mask_traits::logical_invert( mhi ),
		    half_traits::mask_traits::logical_and(
			mlo, 
			half_traits::cmpge( lo, hi, mt_mask() ) ) ),
		hi, lo ),
	    half_traits::mask_traits::logical_or( mlo, mhi ) );
    }

    static type sllv( type a, type b ) { return _mm512_sllv_epi16( a, b ); }
    static type srlv( type a, type b ) { return _mm512_srlv_epi16( a, b ); }
    static type sll( type a, __m128i b ) { return _mm512_sll_epi16( a, b ); }
    static type srl( type a, __m128i b ) { return _mm512_srl_epi16( a, b ); }
    static type sll( type a, long b ) {
	return _mm512_sll_epi16( a, _mm_cvtsi64_si128( b ) );
    }
    static type srl( type a, long b ) {
	return _mm512_srl_epi16( a, _mm_cvtsi64_si128( b ) );
    }
    static type slli( type a, unsigned int s ) {
	    return _mm512_slli_epi16( a, s );
    }
    static type srli( type a, unsigned int s ) {
	return _mm512_srli_epi16( a, s );
    }
    static type srai( type a, unsigned int s ) {
	    return _mm512_srai_epi16( a, s );
    }
    static type srav( type a, type s ) {
	    return _mm512_srav_epi16( a, s );
    }

    template<typename ReturnTy>
    static auto tzcnt( type a ) {
	assert( 0 && "NYI" );
    }

    static __m512 castfp( type a ) { assert( 0 && "NYI" ); }
    static type castint( type a ) { return a; }

    template<unsigned short PermVL>
    static type permute_evenodd( type a ) {
	assert( 0 && "NYI" );
    }

    static constexpr bool has_ternary = true;

    template<unsigned char imm8>
    static type ternary( type a, type b, type c ) {
	return _mm512_ternarylogic_epi32( a, b, c, imm8 );
    }
 
    static type load( const member_type * a ) {
	return _mm512_load_si512( a );
    }
    static type loadu( const member_type * a ) {
	return _mm512_loadu_si512( a );
    }
    static void store( member_type * a, type val ) {
	_mm512_store_si512( a, val ); // alignment?
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
    
#if 0
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
#endif
    template<typename VIdx>
    static type gather( const member_type *a, VIdx b ) {
	return set_pair( recursive_traits::gather( a, b ) );
    }
    template<typename VIdx>
    static type gather( const member_type *a, VIdx b, mask_type mask ) {
	return set_pair( recursive_traits::gather( a, b, mask ) );
    }
    template<typename VIdx>
    static type gather( const member_type *a, VIdx b, vmask_type vmask ) {
	return set_pair( recursive_traits::gather( a, b, vmask ) );
    }
#if 0
    static void scatter( member_type *a, type b, type c ) {
	_mm512_i32scatter_epi32( (void *)a, b, c, W );
    }
    static void scatter( member_type *a, type b, type c, mask_type mask ) {
	_mm512_mask_i32scatter_epi32( (void *)a, mask, b, c, W );
    }
    static void scatter( member_type *a, type b, type c, vmask_type mask ) {
	scatter( a, b, c, asmask( mask ) );
    }
#endif
};
#endif // __AVX512F__

} // namespace target

#endif // GRAPTOR_TARGET_AVX512_2x32_H
