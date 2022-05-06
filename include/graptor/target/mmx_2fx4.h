// -*- c++ -*-
#ifndef GRAPTOR_TARGET_MMX_2Fx4_H
#define GRAPTOR_TARGET_MMX_2Fx4_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include "graptor/target/decl.h"
#include "graptor/target/mmx_2x4.h"
#include "graptor/target/sse42_4x4.h"

namespace target {

/***********************************************************************
 * MMX 4 2-byte floating-point numbers
 * This is poorly tested
 ***********************************************************************/
#if __MMX__
template<typename T>
struct mmx_2fx4 {
    static_assert( sizeof(T) == 2, 
		   "version of template class for 2-byte floats" );
public:
    static constexpr unsigned short W = 2;
    static constexpr unsigned short vlen = 4;
    static constexpr unsigned short size = W * vlen;

    using member_type = T;
    using type = __m64;
    using vmask_type = __m64;
    using itype = __m64;
    using int_type = uint16_t;

    using mtraits = mask_type_traits<4>;
    using mask_type = typename mtraits::type;

    // using half_traits = vector_type_int_traits<member_type,16>;
    // using recursive_traits = vt_recursive<member_type,1,32,half_traits>;
    using int_traits = mmx_2x4<int_type>;
    
    static type set1( member_type a ) {
	if constexpr ( is_customfp_v<member_type> )
	    return _mm_set1_pi16( static_cast<unsigned short int>( a ) );
	else
	    return _mm_set1_pi16( a );
    }
    static type set( member_type a3, member_type a2,
		     member_type a1, member_type a0 ) {
	if constexpr ( is_customfp_v<member_type> )
	    return _mm_set_pi16(
		static_cast<unsigned short>( a3 ),
		static_cast<unsigned short>( a2 ),
		static_cast<unsigned short>( a1 ),
		static_cast<unsigned short>( a0 ) );
	else
	    return _mm_set_pi16( a3, a2, a1, a0 );
    }
    static type setzero() { return _mm_setzero_si64(); }
    static type setone( member_type a ) { set( 0xff, 0xff, 0xff, 0xff ); }
    static bool is_zero( type x ) { return _mm_cvtm64_si64( x ) == 0ULL; }

    static member_type lane( type a, int idx ) {
	if constexpr ( is_customfp_v<member_type> )
	    return member_type( (typename member_type::int_type)(
				    (_mm_cvtm64_si64(a) >> idx*16) & 0xffUL ) );
	else
	    assert( 0 && "NYI" );
    }
    static member_type lane0( type a ) { return lane( a, 0 ); }
    static member_type lane1( type a ) { return lane( a, 1 ); }
    static member_type lane2( type a ) { return lane( a, 2 ); }
    static member_type lane3( type a ) { return lane( a, 3 ); }

    static type add( type a, type b ) {
	assert( 0 && "NYI" );
    }

    static type castfp( type a ) { return a; }
    static type castint( type a ) { return a; }

    static type load( const member_type *addr ) {
	return *(const type *)addr;
    }
    static type loadu( const member_type *addr ) {
	return *(const type *)addr;
    }
    static void store( member_type *addr, type val ) {
	*(type *)addr = val;
    }
    static void storeu( member_type *addr, type val ) {
	*(type *)addr = val;
    }

    template<typename VecT2>
    static auto convert( VecT2 a ) {
	assert( 0 && "NYI" );
	return setzero();
    }

    // Casting here is to do a signed cast of a bitmask for logical masks
    template<typename T2>
    static auto asvector( type a );
    
    static mask_type movemask( vmask_type a ) {
	assert( 0 && "NYI" );
    }

    template<typename IdxT>
    static type gather( member_type *a, IdxT b ) {
	assert( 0 && "NYI" );
	return setzero();
    }

    static type gather( member_type *a, __m128i b, __m128i vmask ) {
#if __AVX2__ && __AVX512F__ && __AVX512VL__
	__m128i g = _mm_mask_i32gather_epi32(
	    sse42_4x4<uint32_t>::setzero(), (const int *)a, b, vmask, W );
	__m128i h = _mm_cvtepi32_epi16( g );
	return _mm_extract_epi64( h, 0 );
#else
	using wint_traits = sse42_4x4<uint32_t>;
	const char * p = reinterpret_cast<const char *>( a );
	member_type zero;
	if constexpr ( is_customfp_v<member_type> )
	    zero = member_type( typename member_type::int_type(0) );
	else
	    zero = 0;
	return set(
	    wint_traits::lane3(vmask)
	    ? *reinterpret_cast<const member_type *>(
		p + W * wint_traits::lane3(b) ) : zero,
	    wint_traits::lane2(vmask)
	    ? *reinterpret_cast<const member_type *>(
		p + W * wint_traits::lane2(b) ) : zero,
	    wint_traits::lane1(vmask)
	    ? *reinterpret_cast<const member_type *>(
		p + W * wint_traits::lane1(b) ) : zero,
	    wint_traits::lane0(vmask)
	    ? *reinterpret_cast<const member_type *>(
		p + W * wint_traits::lane0(b) ) : zero
	    );
#endif
    }

    static type gather( member_type *a, __m128i b, mask_type mask ) {
#if __AVX2__ && __AVX512F__ && __AVX512VL__
	__m128i g = _mm_mmask_i32gather_epi32(
	    sse42_4x4<uint32_t>::setzero(), mask, b, (const int *)a, W );
	__m128i h = _mm_cvtepi32_epi16( g );
	return _mm_extract_epi64( h, 0 );
#else
	using wint_traits = sse42_4x4<uint32_t>;
	const char * p = reinterpret_cast<const char *>( a );
	member_type zero;
	if constexpr ( is_customfp_v<member_type> )
	    zero = member_type( typename member_type::int_type(0) );
	else
	    zero = 0;
	return set(
	    mtraits::lane3(mask)
	    ? *reinterpret_cast<const member_type *>(
		p + W * wint_traits::lane3(b) ) : zero,
	    mtraits::lane2(mask)
	    ? *reinterpret_cast<const member_type *>(
		p + W * wint_traits::lane2(b) ) : zero,
	    mtraits::lane1(mask)
	    ? *reinterpret_cast<const member_type *>(
		p + W * wint_traits::lane1(b) ) : zero,
	    mtraits::lane0(mask)
	    ? *reinterpret_cast<const member_type *>(
		p + W * wint_traits::lane0(b) ) : zero
	    );
#endif
    }


    template<typename IdxT>
    static void scatter( member_type *a, IdxT b, type c ) {
	assert( 0 && "NYI" );
    }
};
#endif // __MMX__

} // namespace target

#endif // GRAPTOR_TARGET_MMX_2Fx4_H
