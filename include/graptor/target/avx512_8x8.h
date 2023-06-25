// -*- c++ -*-
#ifndef GRAPTOR_TARGET_AVX512_8x8_H
#define GRAPTOR_TARGET_AVX512_8x8_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include "graptor/target/decl.h"
#include "graptor/target/avx512_bitwise.h"

#if __AVX512F__ // AVX2 implies SSE4.2
// #include "graptor/target/sse42_4x4.h" // for half-sized indices
#endif // __AVX2__

namespace target {

/***********************************************************************
 * AVX512 8 long int
 ***********************************************************************/
#if __AVX512F__
template<typename T = uint64_t>
struct avx512_8x8 : public avx512_bitwise {
    static_assert( sizeof(T) == 8, 
		   "version of template class for 8-byte integers" );
public:
    using member_type = T;
    using int_type = uint64_t;
    using type = __m512i;
    using itype = __m512i;
    using vmask_type = __m512i;

    using mask_traits = mask_type_traits<8>;
    using mask_type = typename mask_traits::type;

    using half_traits = vector_type_traits<member_type,32>;
    using recursive_traits = vt_recursive<member_type,8,64,half_traits>;
    using int_traits = avx512_8x8<int_type>;
    using mt_preferred = target::mt_mask;
    
    static constexpr size_t W = 8;
    static constexpr size_t B = 8*W;
    static constexpr size_t vlen = 8;
    static constexpr size_t size = W * vlen;

/*
    static void print( std::ostream & os, type v ) {
	os << '(' << lane0(v) << ',' << lane1(v)
	   << ',' << lane2(v) << ',' << lane3(v)
	   << ',' << lane4(v) << ',' << lane5(v)
	   << ',' << lane6(v) << ',' << lane7(v) << ')';
    }
*/
    
    static member_type lane( type a, int idx ) {
	switch( idx ) {
	case 0: return (member_type) _mm_extract_epi64( _mm256_extracti128_si256( _mm512_extracti64x4_epi64( a, 0 ), 0 ), 0 );
	case 1: return (member_type) _mm_extract_epi64( _mm256_extracti128_si256( _mm512_extracti64x4_epi64( a, 0 ), 0 ), 1 );
	case 2: return (member_type) _mm_extract_epi64( _mm256_extracti128_si256( _mm512_extracti64x4_epi64( a, 0 ), 1 ), 0 );
	case 3: return (member_type) _mm_extract_epi64( _mm256_extracti128_si256( _mm512_extracti64x4_epi64( a, 0 ), 1 ), 1 );
	case 4: return (member_type) _mm_extract_epi64( _mm256_extracti128_si256( _mm512_extracti64x4_epi64( a, 1 ), 0 ), 0 );
	case 5: return (member_type) _mm_extract_epi64( _mm256_extracti128_si256( _mm512_extracti64x4_epi64( a, 1 ), 0 ), 1 );
	case 6: return (member_type) _mm_extract_epi64( _mm256_extracti128_si256( _mm512_extracti64x4_epi64( a, 1 ), 1 ), 0 );
	case 7: return (member_type) _mm_extract_epi64( _mm256_extracti128_si256( _mm512_extracti64x4_epi64( a, 1 ), 1 ), 1 );
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

    static type setoneval() { // 0x0000000000000001 repeated
	// http://agner.org/optimize/optimizing_assembly.pdf
	return _mm512_srli_epi64( setone(), 63 );
    }
    static type set_maskz( mask_type m, type a ) {
	return _mm512_maskz_mov_epi64( m, a );
    }
    
    static type set1( member_type a ) { return _mm512_set1_epi64( a ); }
    static type set1inc( member_type a ) {
	return add( set1( a ),
		    _mm512_set_epi64( 7, 6, 5, 4, 3, 2, 1, 0 ) );
    }
    static type set( member_type a7, member_type a6,
		     member_type a5, member_type a4,
		     member_type a3, member_type a2,
		     member_type a1, member_type a0 ) {
	return _mm512_set_epi64x( a7, a6, a5, a4, a3, a2, a1, a0 );
    }
    static type setl0( member_type a ) {
	return _mm512_zextsi128_si512( _mm_cvtsi64_si128( a ) );
    }

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
	auto ones = setone();
	auto zero = setzero();
	return _mm512_mask_blend_epi64( mask, zero, ones );
    }
    
    static mask_type asmask( vmask_type a ) {
#if __AVX512DQ__
	return _mm512_movepi64_mask( a );
#else
	vmask_type m = srli( setone(), 1 );
	return _mm512_cmpgt_epu64_mask( a, m );
#endif
    }

    static type asvector( mask_type mask ) {
	auto ones = setone();
	auto zero = setzero();
	return _mm512_mask_blend_epi64( mask, zero, ones );
    }

    static mask_type cmpeq( type a, type b, mt_mask ) {
	return _mm512_cmpeq_epi64_mask( a, b );
    }
    static mask_type cmpne( type a, type b, mt_mask ) {
	return _mm512_cmpneq_epi64_mask( a, b );
    }
    static mask_type cmpgt( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm512_cmpgt_epi64_mask( a, b );
	else
	    return _mm512_cmpgt_epu64_mask( a, b );
    }
    static mask_type cmpge( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm512_cmpge_epi64_mask( a, b );
	else
	    return _mm512_cmpge_epu64_mask( a, b );
    }
    static mask_type cmplt( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm512_cmplt_epi64_mask( a, b );
	else
	    return _mm512_cmplt_epu64_mask( a, b );
    }
    static mask_type cmple( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm512_cmple_epi64_mask( a, b );
	else
	    return _mm512_cmple_epu64_mask( a, b );
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


    static type add( type src, mask_type m, type a, type b ) {
	return _mm512_mask_add_epi64( src, m, a, b );
    }
    static type add( type src, vmask_type m, type a, type b ) {
	return _mm512_mask_add_epi64( src, asmask(m), a, b );
    }

    static type add( type a, type b ) { return _mm512_add_epi64( a, b ); }
    static type sub( type a, type b ) { return _mm512_sub_epi64( a, b ); }
    // static type mul( type a, type b ) { return _mm512_mul_epi64( a, b ); }
    // static type div( type a, type b ) { return _mm512_div_epi64( a, b ); }

    static type min( type a, type b ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm512_min_epi64( a, b );
	else
	    return _mm512_min_epu64( a, b );
    }

    static type max( type a, type b ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm512_max_epi64( a, b );
	else
	    return _mm512_max_epu64( a, b );
    }

    static bool cmpne( type a, type b, target::mt_bool ) {
	mask_type ne = cmpne( a, b, target::mt_mask() );
	return ! _kortestz_mask8_u8( ne, ne );
    }
    static type blend( vmask_type m, type l, type r ) {
	return _mm512_mask_blend_epi64( asmask( m ), l, r );
    }
    static type blendm( mask_type m, type l, type r ) {
	return _mm512_mask_blend_epi64( m, l, r );
    }
    static type blend( mask_type m, type l, type r ) { return blendm( m, l, r ); }
    static type bitblend( type m, type l, type r ) {
	if constexpr ( has_ternary ) {
	    return ternary<0xac>( m, l, r );
	} else {
	    return bitwise_or( bitwise_and( m, r ),
			       bitwise_andnot( m, l ) );
	}
    }
    static type iforz( vmask_type m, type a ) {
	return bitwise_and( m, a );
    }
    static type iforz( mask_type m, type a ) {
	return blend( m, setzero(), a );
    }

    static constexpr bool has_ternary = true;

    template<unsigned char imm8>
    static type ternary( type a, type b, type c ) {
	return _mm512_ternarylogic_epi64( a, b, c, imm8 );
    }

    static type castint( type a ) { return a; }
    static __m512d castfp( type a ) { return _mm512_castsi512_pd( a ); }

    static member_type reduce_setif( type val ) {
	// Pick any, preferrably one that is not -1
	// Could also use min_epu64
	return _mm512_reduce_max_epi64( val );
    }
    static member_type reduce_add( type val ) {
	return _mm512_reduce_add_epi64( val );
    }
    static member_type reduce_add( type val, mask_type mask ) {
	return _mm512_mask_reduce_add_epi64( mask, val );
    }
    static member_type reduce_logicalor( type val ) {
	return member_type( _mm512_reduce_or_epi64( val ) );
    }
    static member_type reduce_logicalor( type val, mask_type mask ) {
	return _mm512_mask_reduce_or_epi64( mask, val );
    }
    static member_type reduce_bitwiseor( type val ) {
	return member_type( _mm512_reduce_or_epi64( val ) );
    }
    
    static member_type reduce_min( type val ) {
	return _mm512_reduce_min_epi64( val );
    }
    static member_type reduce_min( type val, mask_type mask ) {
	return _mm512_mask_reduce_min_epi64( mask, val );
    }

    static type sllv( type a, type b ) { return _mm512_sllv_epi64( a, b ); }
    static type sllv( type a, __m128i b ) {
	return sllv( a, _mm512_cvtepi8_epi32( b ) ); }
    static type srlv( type a, type b ) { return _mm512_srlv_epi64( a, b ); }
    static type sll( type a, __m128i b ) { return _mm512_sll_epi64( a, b ); }
    static type srl( type a, __m128i b ) { return _mm512_srl_epi64( a, b ); }
    static type sll( type a, long b ) {
	return _mm512_sll_epi64( a, _mm_cvtsi64_si128( b ) );
    }
    static type srl( type a, long b ) {
	return _mm512_srl_epi64( a, _mm_cvtsi64_si128( b ) );
    }
    static type sllv( type a, __m256i b ) {
	return sllv( a, _mm512_cvtepi32_epi64( b ) );
    }
    static type srlv( type a, __m256i b ) {
	return srlv( a, _mm512_cvtepi32_epi64( b ) );
    }
    static type slli( type a, unsigned int s ) {
	    return _mm512_slli_epi64( a, s );
    }
    static type srli( type a, unsigned int s ) {
	return _mm512_srli_epi64( a, s );
    }
    static type srai( type a, unsigned int s ) {
	    return _mm512_srai_epi64( a, s );
    }
    static type srav( type a, type s ) {
	    return _mm512_srav_epi64( a, s );
    }

    template<typename ReturnTy>
    static auto tzcnt( type a ) {
	__m512i zero = setzero();
	__m512i b = _mm512_sub_epi64( zero, a );
	__m512i c = _mm512_and_si512( a, b );
	__m512d f = _mm512_cvtepi64_pd( c ); // AVX512DQ
	__m512i g = _mm512_castpd_si512( f );
	__m512i h = _mm512_srli_epi64( g, 52 );
	// __m512i bias = set1( 0x3ff );
	__m512i bias = srli( setone(), 8*W-10 ); // 0x3ff
	__m512i raw = _mm512_sub_epi64( h, bias );
	__m512i cnt = blendm( cmpeq( a, zero, mt_mask() ), raw, zero );

#if 1
	__m512i chk = bitwise_and(
	    a, sub( sllv( setoneval(), cnt ), setoneval() ) );
	__mmask8 allz = cmpeq( chk, setzero(), mt_mask() );
	assert( _kortestz_mask8_u8( allz, allz ) == 0 && "error tzcnt" );
#endif
	
	if constexpr ( sizeof(ReturnTy) == W ) {
	    return cnt;
	} else if constexpr ( sizeof(ReturnTy) == 4 ) {
	    return _mm512_cvtepi64_epi32( cnt ); // AVX512F
	} else if constexpr ( sizeof(ReturnTy) == 2 ) {
	    return _mm512_cvtepi64_epi16( cnt ); // AVX512F
	} else if constexpr ( sizeof(ReturnTy) == 1 ) {
	    auto r = _mm512_cvtepi64_epi8( cnt ); // AVX512F
#if GRAPTOR_USE_MMX
	    return _mm_cvtsi64_m64( _mm_cvtsi128_si64( r ) );
#else
	    return r;
#endif
	} else {
	    assert( 0 && "NYI" );
	    return 0;
	}
    }

    template<typename ReturnTy>
    static auto lzcnt( type a ) {
#if __AVX512CD__
	type cnt = _mm512_lzcnt_epi64( a );
#else
	type cnt;
	assert( 0 && "NYI" );
#endif
	if constexpr ( sizeof(ReturnTy) == W )
	    return cnt;
	else if constexpr ( sizeof(ReturnTy) == 4 )
	    return _mm512_cvtepi64_epi32( cnt ); // AVX512F
	else if constexpr ( sizeof(ReturnTy) == 2 ) {
	    return _mm512_cvtepi64_epi16( cnt ); // AVX512F
	} else {
	    assert( 0 && "NYI" );
	    return 0;
	}
    }

    static auto popcnt( type v ) {
#if __AVX512VPOPCNTDQ__ && __AVX512VL__
	return _mm512_popcnt_epi64( v );
#else
	// source: https://arxiv.org/pdf/1611.07612.pdf
	type lookup =
	    _mm512_set_epi8( 4, 3, 3, 2, 3, 2, 2, 1,
			     3, 2, 2, 1, 2, 1, 1, 0,
			     4, 3, 3, 2, 3, 2, 2, 1,
			     3, 2, 2, 1, 2, 1, 1, 0,
			     4, 3, 3, 2, 3, 2, 2, 1,
			     3, 2, 2, 1, 2, 1, 1, 0,
			     4, 3, 3, 2, 3, 2, 2, 1,
			     3, 2, 2, 1, 2, 1, 1, 0 );
	type low_mask = _mm512_set1_epi8( 0x0f );
	type lo = _mm512_and_si512( v, low_mask );
	type hi = _mm512_and_si512( _mm512_srli_epi32( v, 4 ), low_mask );
	type popcnt1 = _mm512_shuffle_epi8( lookup, lo );
	type popcnt2 = _mm512_shuffle_epi8( lookup, hi );
	type total = _mm512_add_epi8( popcnt1, popcnt2 );
	return _mm512_sad_epu8( total, setzero() );
#endif
    }
    
    static type load( const member_type *a ) {
	return _mm512_load_si512( (const type *)a );
    }
    static type loadu( const member_type *a ) {
	return _mm512_loadu_si512( (const type *)a );
    }
    static void store( member_type *addr, type val ) {
	_mm512_store_si512( (type *)addr, val );
    }
    static void storeu( member_type *addr, type val ) {
	_mm512_storeu_si512( (type *)addr, val );
    }

    static type ntload( const member_type * a ) {
	return _mm512_stream_load_si512( (__m512i *)a );
    }
    static void ntstore( member_type * a, type val ) {
	_mm512_stream_si512( (__m512i *)a, val );
    }
    
    static type
    gather( const member_type *a, itype b ) {
	return _mm512_i64gather_epi64( b, (const long long *)a, W );
    }
    static type
    gather( const member_type *a, itype b, mask_type mask ) {
	return _mm512_mask_i64gather_epi64( setzero(), mask, b, a, W );
    }
    static type
    gather( const member_type *a, itype b, vmask_type mask ) {
	return _mm512_mask_i64gather_epi64( setzero(), asmask( mask ), b, a, W );
    }
    template<unsigned short Scale>
    static type
    gather_w( const member_type *a, itype b ) {
	return _mm512_i64gather_epi64( b, (const long long *)a, Scale );
    }
    template<unsigned short Scale>
    static type
    gather_w( const member_type *a, itype b, mask_type mask ) {
	return _mm512_mask_i64gather_epi64( setzero(), mask, b, a, Scale );
    }
    template<unsigned short Scale>
    static type
    gather_w( const member_type *a, itype b, vmask_type mask ) {
	return _mm512_mask_i64gather_epi64( setzero(), asmask( mask ), b, a, Scale );
    }
    static type
    gather( const member_type *a, __m256i b ) {
	return _mm512_i32gather_epi64( b, (const long long *)a, W );
    }
    static type
    gather( const member_type *a, __m256i b, mask_type mask ) {
	return _mm512_mask_i32gather_epi64( setzero(), mask, b, a, W );
    }
    static type
    gather( const member_type *a, __m256i b, __m256i mask ) {
	return _mm512_mask_i32gather_epi64( setzero(), half_traits::asmask(mask), b, a, W );
    }
    template<unsigned short Scale>
    static type
    gather_w( const member_type *a, __m256i b ) {
	return _mm512_i32gather_epi64( b, (const long long *)a, Scale );
    }
    template<unsigned short Scale>
    static type
    gather_w( const member_type *a, __m256i b, mask_type mask ) {
	return _mm512_mask_i32gather_epi64( setzero(), mask, b, a, Scale );
    }
    template<unsigned short Scale>
    static type
    gather_w( const member_type *a, __m256i b, __m256i mask ) {
	return _mm512_mask_i32gather_epi64( setzero(), half_traits::asmask(mask), b, a, Scale );
    }
    static void scatter( member_type *a, type b, type c ) {
	_mm512_i64scatter_epi64( (void *)a, b, c, W );
    }
    static void scatter( member_type *a, type b, type c, mask_type mask ) {
	_mm512_mask_i64scatter_epi64( (void *)a, mask, b, c, W );
    }
    static void scatter( member_type *a, type b, type c, vmask_type mask ) {
	_mm512_mask_i64scatter_epi64( (void *)a, asmask( mask ), b, c, W );
    }
    static void scatter( member_type *a, __m256i b, type c ) {
	_mm512_i32scatter_epi64( (void *)a, b, c, W );
    }
    template<typename IdxT>
    static typename std::enable_if<sizeof(IdxT)==sizeof(type)/2>::type
    scatter( member_type *a, IdxT b, type c, mask_type mask ) {
	_mm512_mask_i32scatter_epi64( (void *)a, mask, b, c, W );
    }
    template<typename IdxT>
    static typename std::enable_if<sizeof(IdxT)==sizeof(type)/2>::type
    scatter( member_type *a, IdxT b, type c, vmask_type mask ) {
	_mm512_mask_i32scatter_epi64( (void *)a, asmask(mask), b, c, W );
    }
};
#endif // __AVX512F__

} // namespace target

#endif // GRAPTOR_TARGET_AVX512_8x8_H
