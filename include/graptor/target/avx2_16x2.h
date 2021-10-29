// -*- c++ -*-
#ifndef GRAPTOR_TARGET_AVX2_16x2_H
#define GRAPTOR_TARGET_AVX2_16x2_H

#include <x86intrin.h>
#include <immintrin.h>

#include "graptor/longint.h"
#include "graptor/target/decl.h"

#include "graptor/target/mmx_4x2.h"

namespace target {

/***********************************************************************
 * Cases representable in 256 bits (AVX2)
 ***********************************************************************/
#if __AVX2__
template<typename T = longint<16>>
struct avx2_16x2 {
    static_assert( sizeof(T) == 16, "size assumption" );
    static constexpr unsigned short W = 16;
    static constexpr unsigned short vlen = 2;
    static constexpr unsigned short size = W * vlen;
    using member_type = T;
    using type = __m256i;
    using itype = __m64;
    using vmask_type = __m256i;
    using mask_type = typename mask_type_traits<2>::type;

    using int_type = uint32_t;
    using int_traits = mmx_4x2<int_type>;
    using mt_preferred = target::mt_vmask;

    static type setzero() { return _mm256_setzero_si256(); }
    static type setone() {
	auto zero = setzero();
	return _mm256_cmpeq_epi32( zero, zero );
    }
    static type set1( member_type a ) {
    	__m128i v = a.get();
	type x = _mm256_castsi128_si256( v );
	return _mm256_inserti128_si256( x, v, 1 );
    }
    static type set( type a ) {
	return a;
    }

    static member_type lane( type a, int l ) {
	switch( l ) {
	case 0: return lane0( a );
	case 1: return lane1( a );
	}
    }
    static member_type lane0( type a ) {
	// return member_type( _mm256_extracti128_si256( a, 0 ) );
	return member_type( _mm256_castsi256_si128( a ) );
    }
    static member_type lane1( type a ) {
	return member_type( _mm256_extracti128_si256( a, 1 ) );
    }

    static mask_type asmask( vmask_type v ) {
#if 0
	// ones = 111...111
	__m128i x;
	__m128i ones = _mm_cmpeq_epi64( x, x );
	// hi2 = 10..010..0
	__m128i hi2 = _mm_slli_epi64( ones, 63 );
	// hi = 10.......0
	__m128i hi = _mm_bslli_si128( hi2, 8 );
	__m128i l0 = lane0( v ).get();
	__m128i l1 = lane1( v ).get();
	mask_type m0 = _mm_testz_si128( l0, hi ) ? 0 : 1;
	mask_type m1 = _mm_testz_si128( l1, hi ) ? 0 : 1;
	return ( m1 << 1 ) | m0;
#else
	// vr = ( 0...0 B1 0...0 B0 ) where B1 and B0 are the top bytes
	// of the two 128-bit lanes in v
	vmask_type vr = _mm256_srli_si256( v, 15 );
	// v0 = ( 0....0 x 0....0 y ) where x and y are the top bits of interest
	vmask_type v0 = _mm256_srli_epi64( v, 7 );
	__m128i l0 = lane0( v ).get();
	__m128i l1 = lane1( v ).get();
	mask_type m0 = _mm_testz_si128( l0, l0 ) ? 0 : 1;
	mask_type m1 = _mm_testz_si128( l1, l1 ) ? 0 : 1;
	return ( m1 << 1 ) | m0;
#endif
    }
    
    static mask_type cmpne( type l, type r, mt_mask ) {
	return asmask( cmpne( l, r, mt_vmask() ) );
    }
    static mask_type cmpeq( type l, type r, mt_mask ) {
	return asmask( cmpeq( l, r, mt_vmask() ) );
    }
    static vmask_type cmpne( type l, type r, mt_vmask ) {
	return ~cmpeq( l, r, mt_vmask() );
    }
    static vmask_type cmpeq( type l, type r, mt_vmask ) {
	// Check using 4 lanes of 64 bits each
	vmask_type eq = _mm256_cmpeq_epi64( l, r );
	// Check that both pairs of 128 bit lanes are all one
	// One 128 bit part: ( q1, q0 )
	// Calculate: ( q1, q0 ) AND ( q0, q1 ) (true if both true)
	// Full vector: ( q3, q2, q1, q0 ) -> ( q2, q3, q0, q1 )
	vmask_type sh = _mm256_shuffle_epi32( eq, 0b01001110 );
	return _mm256_and_si256( eq, sh );
    }

    static type bitwise_and( type l, type r ) {
	return _mm256_and_si256( l, r );
    }
    static type bitwise_or( type l, type r ) {
	return _mm256_or_si256( l, r );
    }
    static type bitwise_invert( type a ) {
	return _mm256_xor_si256( a, setone() );
    }
    static type logical_and( type l, type r ) {
	return _mm256_and_si256( l, r );
    }
    static type logical_or( type l, type r ) {
	return _mm256_or_si256( l, r );
    }
    static type logical_invert( type a ) {
	return _mm256_xor_si256( a, setone() );
    }

    static member_type reduce_bitwiseor( type a ) {
	return member_type( lane0( a ) | lane1( a ) );
    }
    static member_type reduce_logicalor( type a ) {
	return member_type( lane0(a) | lane1(a) );
    }

    static type loadu( const member_type * addr ) {
	return _mm256_load_si256( reinterpret_cast<const type *>( addr ) );
    }
    static type load( const member_type * addr ) {
	return _mm256_load_si256( reinterpret_cast<const type *>( addr ) );
    }
    static void storeu( member_type * addr, type val ) {
	_mm256_store_si256( reinterpret_cast<type *>( addr ), val );
    }
    static void store( member_type * addr, type val ) {
	_mm256_store_si256( reinterpret_cast<type *>( addr ), val );
    }

    static type gather( const member_type * addr, __m64 idx ) {
	// idx = ( i1, i0 ), 2 4-byte indices
	// ridx = ( x, x, i1, i0 )
	__m128i ridx = _mm_set_epi64( idx, idx );
	// uidx = ( i1, i1, i0, i0 )
	__m128i uidx = _mm_unpacklo_epi32( ridx, ridx );
	// idx4 = ( i1+1, i1, i0+1, i0 )
	__m128i idx4 = uidx + _mm_set_epi32( 1, 0, 1, 0 );
	type g = _mm256_i32gather_epi64(
	    reinterpret_cast<const long long*>(addr), idx4, W/2 );
	return g;
    }
    static type gather( const member_type * addr, __m128i idx ) {
	// idx = ( i1, i0 ), 2 8-byte indices
	__m256i ridx = _mm256_castsi128_si256( idx );
	// uidx = ( i1, i1, i0, i0 )
	__m256i uidx = _mm256_permute4x64_epi64( ridx, 0b01010000 );
	// idx4 = ( i1+1, i1, i0+1, i0 )
	__m256i idx4 = uidx + _mm256_set_epi64x( 1, 0, 1, 0 );
	type g = _mm256_i64gather_epi64(
	    reinterpret_cast<const long long*>(addr), idx4, W/2 );
	return g;
    }
    static type gather( const member_type * addr, __m64 idx, __m64 mask ) {
	// idx = ( i1, i0 ), 2 4-byte indices
	// ridx = ( x, x, i1, i0 )
	__m128i ridx = _mm_movpi64_epi64( idx );
	// uidx = ( i1, i1, i0, i0 )
	__m128i uidx = _mm_unpacklo_epi32( ridx, ridx );
	// idx4 = ( i1+1, i1, i0+1, i0 )
	__m128i idx4 = uidx + _mm_set_epi32( 1, 0, 1, 0 );
	// mask = ( m1, m0 ), 2 4-byte masks
	// need: ( m1, m1, m1, m1, m0, m0, m0, m0 )
	// rmask = ( x, x, m1, m0 )
	__m128i rmask = _mm_movpi64_epi64( mask );
	// umask = ( m1, m1, m0, m0 )
	__m128i umask = _mm_unpacklo_epi32( rmask, rmask );
	// cmask = ( x, x, x, x, m1, m1, m0, m0 )
	__m256i cmask = _mm256_castsi128_si256( umask );
	// mask4 = ( m1, m1, m1, m1, m0, m0, m0, m0 )
	__m256i mask4 = _mm256_permute4x64_epi64( cmask, 0b01010000 );
	type g = _mm256_mask_i32gather_epi64(
	    setzero(), reinterpret_cast<const long long*>(addr),
	    idx4, mask4, W/2 );
	return g;
    }
    static type gather( const member_type * addr, __m64 idx, mask_type mask ) {
	using itraits = mmx_4x2<uint32_t>;
	const __m128i * maddr = reinterpret_cast<const __m128i *>( addr );
	type r = setzero();
	if( mask & 1 ) {
	    __m128i data = _mm_load_si128( maddr + itraits::lane0( idx ) );
	    r = _mm256_castsi128_si256( data );
	}
	if( mask & 2 ) {
	    __m128i data = _mm_load_si128( maddr + itraits::lane1( idx ) );
	    r = _mm256_inserti128_si256( r, data, 1 );
	}
	return r;
    }
    static type gather( const member_type * addr, __m128i idx, __m128i mask ) {
	// idx = ( i1, i0 ), 2 8-byte indices
	__m256i ridx = _mm256_castsi128_si256( idx );
	// uidx = ( i1, i1, i0, i0 )
	__m256i uidx = _mm256_permute4x64_epi64( ridx, 0b01010000 );
	// idx4 = ( i1+1, i1, i0+1, i0 )
	__m256i idx4 = uidx + _mm256_set_epi64x( 1, 0, 1, 0 );
	// Now the same process for mask, bar the addition
	__m256i rmask = _mm256_castsi128_si256( mask );
	__m256i mask4 = _mm256_permute4x64_epi64( rmask, 0b01010000 );
	type g = _mm256_mask_i64gather_epi64(
	    setzero(), reinterpret_cast<const long long*>(addr),
	    idx4, mask4, W/2 );
	return g;
    }
    template<typename IdxT>
    static void scatter( const member_type * addr, IdxT idx, type val ) {
	assert( 0 && "NYI" );
    }
    template<typename IdxT, typename MaskT>
    static void scatter( const member_type * addr, IdxT idx, type val, MaskT mask ) {
	assert( 0 && "NYI" );
    }
};
#endif // __AVX2__

} // namespace target

#endif //  GRAPTOR_TARGET_AVX2_16x2_H
