// -*- c++ -*-
#ifndef GRAPTOR_TARGET_SSE42_8fx2_H
#define GRAPTOR_TARGET_SSE42_8fx2_H

#include <x86intrin.h>
#include <immintrin.h>

#include "graptor/target/decl.h"
#include "graptor/target/vt_recursive.h"

namespace target {

/***********************************************************************
 * Cases representable in 128 bits (SSEx)
 ***********************************************************************/
#if __SSE4_2__
template<typename T = float>
struct sse42_8fx2 {
    static_assert( sizeof(T) == 8, "size assumption" );

    using member_type = T;
    using int_type = uint64_t;
    using type = __m128d;
    using itype = __m128i;
    using vmask_type = __m128i;

    using mtraits = mask_type_traits<2>;
    using mask_type = typename mtraits::type;

    // using half_traits = vector_type_traits<member_type,8>;
    // using recursive_traits = vt_recursive<member_type,4,16,half_traits>;
    using int_traits = sse42_8x2<int_type>;
    
    static constexpr size_t W = 8;
    static constexpr size_t B = 8*W;
    static constexpr size_t vlen = 2;
    static constexpr size_t size = W * vlen;

    static member_type lane( type a, int idx ) {
	const member_type *m = reinterpret_cast<const member_type *>( &a );
	switch( idx ) {
	case 0: return m[0];
	case 1: return m[1];
	default:
	    assert( 0 && "should not get here" );
	}
    }
    static member_type lane0( type a ) { return lane( a, 0 ); }
    static member_type lane1( type a ) { return lane( a, 1 ); }

    static type set( member_type a1, member_type a0 ) {
	return _mm_set_pd( a1, a0 );
    }

    static __m64 lower_half( type a ) {
	return _mm_cvtsi64_m64( _mm_extract_epi64( _mm_castpd_si128( a ), 0 ) );
    }
    static __m64 upper_half( type a ) {
	return _mm_cvtsi64_m64( _mm_extract_epi64( _mm_castpd_si128( a ), 1 ) );
    }
    static vpair<__m64,__m64> decompose( type a ) {
	return vpair<__m64,__m64>{ lower_half(a), upper_half(a) };
    }

    static type setzero() { return _mm_setzero_pd(); }
    static type set1( member_type a ) { return _mm_set1_pd( a ); }

    static vmask_type asvector( mask_type m ) {
	return int_traits::asvector( m );
    }
    static type blend( vmask_type m, type a, type b ) {
	return _mm_blendv_pd( a, b, _mm_castsi128_pd( m ) );
    }
    static type blend( mask_type m, type a, type b ) {
	return _mm_blendv_pd( a, b, _mm_castsi128_pd( asvector( m ) ) );
    }

    static vmask_type cmpgt( type a, type b, mt_vmask ) {
	return _mm_castpd_si128( _mm_cmpgt_pd( a, b ) );
    }

/*
    static type abs( type a ) {
	// Not available; do it the hard way.
	__m128i m = int_traits::setone_shr1();
	__m128 i = _mm_and_pd( _mm_castsi128_pd( m ), a );
	return i;
    }
*/

    static type add( type a, type b ) { return _mm_add_pd( a, b ); }
    static type add( type s, vmask_type m, type a, type b ) {
	return _mm_blendv_pd( s, add( a, b ), _mm_castsi128_pd( m ) );
    }
    static type mul( type a, type b ) { return _mm_mul_pd( a, b ); }
    static type mul( type s, vmask_type m, type a, type b ) {
	return _mm_blendv_pd( s, mul( a, b ), _mm_castsi128_pd( m ) );
    }
    static type div( type a, type b ) { return _mm_div_pd( a, b ); }
    static type sub( type a, type b ) { return _mm_sub_pd( a, b ); }

    static member_type reduce_add( type a ) {
	return lane0( a ) + lane1( a );
    }
    static member_type reduce_add( type a, vmask_type m ) {
#if __AVX512F__
	__m512i wm = _mm512_zextsi128_si512( m );
	__mmask8 mm = _mm512_cmpneq_epi32_mask( wm, _mm512_setzero_si512() );
	__m512 c = _mm512_insertf32x4( _mm512_setzero_ps(),
				       _mm_castpd_ps( a ), 0 );
	return _mm512_mask_reduce_add_pd( mm, _mm512_castps_pd( c ) );
#else
	member_type r = 0;
	if( int_traits::lane0( m ) ) r += lane0( a );
	if( int_traits::lane1( m ) ) r += lane1( a );
	return r;
#endif
    }

    static type load( const member_type *a ) {
	return _mm_load_pd( (const member_type *)a );
    }
    static type loadu( const member_type *a ) {
	return _mm_loadu_pd( (const member_type *)a );
    }
    static void store( member_type *addr, type val ) {
	_mm_store_pd( (member_type *)addr, val );
    }
    static void storeu( member_type *addr, type val ) {
	_mm_storeu_pd( (member_type *)addr, val );
    }

    static type
    gather( const member_type *a, itype b ) {
#if __AVX2__
	return _mm_i32gather_pd( a, b, W );
#else
	using idx_traits = int_traits;
	// vector_type_traits_vl<typename int_type_of_size<sizeof(itype)/vlen>::type, vlen>;
	return set( a[idx_traits::lane1(b)], a[idx_traits::lane0(b)] );
#endif
    }
    static type gather( const member_type *a, __m64 b ) {
	uint32_t b0 = _m_to_int( b );
	uint32_t b1 = _m_to_int64( b ) >> 32;
	return set( a[b1], a[b0] );
    }
    static type
    gather( const member_type *a, itype b, mask_type mask ) {
	// return _mm_mask_i32gather_epi32( setzero(), a, b, asvector(mask), size );
	assert( 0 && "NYI" );
	return setzero();
    }
/*
*/
    static type
    gather( const member_type *a, itype b, vmask_type vmask ) {
#if __AVX2__
	return _mm_mask_i32gather_pd( setzero(), a, b,
				      _mm_castsi128_pd( vmask ), W );
#else
	return set(
	    int_traits::lane1(vmask) ? a[int_traits::lane1(b)] : member_type(0),
	    int_traits::lane0(vmask) ? a[int_traits::lane0(b)] : member_type(0)
	    );
#endif
    }

    static void scatter( member_type *a, itype b, type c ) {
	a[int_traits::lane0(b)] = lane0(c);
	a[int_traits::lane1(b)] = lane1(c);
    }
    static void scatter( member_type *a, __m64 b, type c ) {
	uint32_t b0 = _m_to_int( b );
	uint32_t b1 = _m_to_int64( b ) >> 32;
	a[b0] = lane0(c);
	a[b1] = lane1(c);
    }
    static void scatter( member_type *a, itype b, type c, vmask_type mask ) {
	if( int_traits::lane0(mask) ) a[int_traits::lane0(b)] = lane0(c);
	if( int_traits::lane1(mask) ) a[int_traits::lane1(b)] = lane1(c);
    }
    static void scatter( member_type *a, itype b, type c, mask_type mask ) {
	if( mtraits::lane0(mask) ) a[int_traits::lane0(b)] = lane0(c);
	if( mtraits::lane1(mask) ) a[int_traits::lane1(b)] = lane1(c);
    }
};
#endif // __SSE4_2__

} // namespace target

#endif //  GRAPTOR_TARGET_SSE42_8fx2_H
