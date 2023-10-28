// -*- c++ -*-
#ifndef GRAPTOR_TARGET_SSE42_8x2_H
#define GRAPTOR_TARGET_SSE42_8x2_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include "graptor/target/decl.h"

#include "graptor/target/mmx_4x2.h"
#include "graptor/target/sse42_bitwise.h"
#include "graptor/target/scalar_int.h"

alignas(64) extern const uint64_t increasing_sequence_epi64[16];

namespace target {

/***********************************************************************
 * SSE4.2 2 long integers
 ***********************************************************************/
#if __SSE4_2__
template<typename T = uint64_t>
struct sse42_8x2 : public sse42_bitwise {
    static_assert( sizeof(T) == 8, 
		   "version of template class for 8-byte integers" );
public:
    using member_type = T;
    using int_type = uint64_t;
    using type = __m128i;
    using itype = __m128i;
    using vmask_type = __m128i;

    using mtraits = mask_type_traits<2>;
    using mask_type = typename mtraits::type;

    using half_traits = scalar_int<T>;
    using int_traits = sse42_8x2<int_type>;

    static constexpr unsigned short W = 8;
    static constexpr unsigned short vlen = 2;
    static constexpr unsigned short size = W * vlen;

    static member_type lane_permute( type a, int idx ) {
#if __AVX__
	type vidx = _mm_cvtsi32_si128( idx<<1 );
	__m128d ad = _mm_castsi128_pd( a );
	__m128d pd = _mm_permutevar_pd( ad, vidx );
	type perm = _mm_castpd_si128( pd );
	return _mm_extract_epi64( perm, 0 );
#else
	assert( 0 && "NYI" );
	return 0;
#endif
    }
    static member_type lane_memory( type a, int idx ) {
	member_type m[vlen];
	storeu( m, a );
	return m[idx];
    }
    static member_type lane_switch( type a, int idx ) {
	switch( idx ) {
	case 0: return (member_type) _mm_extract_epi64( a, 0 );
	case 1: return (member_type) _mm_extract_epi64( a, 1 );
	default:
	    assert( 0 && "should not get here" );
	}
    }
    static member_type lane( type a, int idx ) {
	// With only two lanes, the switch variant is marginally faster
	// than the others on AMD EPYC 7702.
	return lane_switch( a, idx );
    }
    static member_type lane0( type a ) {
	return _mm_cvtsi128_si64( a );
    }
    static member_type lane1( type a ) { return lane( a, 1 ); }

    static type setzero() { return _mm_setzero_si128(); }
    static type setone() { // 0xffffffffffffffff repeated
	auto zero = setzero();
	return _mm_cmpeq_epi64( zero, zero );
    }
    static type setoneval() { // 0x0000000000000001 repeated
	// http://agner.org/optimize/optimizing_assembly.pdf
	return _mm_srli_epi64( setone(), 63 );
    }
    
    static type set1( member_type a ) { return _mm_set1_epi64x( a ); }
    static type setl0( member_type a ) { return _mm_cvtsi64_si128( a ); }
    static type set1inc( member_type a ) {
	return add( set1( a ), set1inc0() );
    }
    static type set1inc0() {
	return load(
	    static_cast<const member_type *>( &increasing_sequence_epi64[0] ) );
    }
    static type set( member_type a1, member_type a0 ) {
	return _mm_set_epi64x( a1, a0 );
    }
    static type set_pair( member_type a1, member_type a0 ) {
	return set( a1, a0 );
    }

    static member_type lower_half( type a ) { return lane0( a ); }
    static member_type upper_half( type a ) { return lane1( a ); }

    static type blendm( vmask_type m, type l, type r ) {
	// see also https://stackoverflow.com/questions/40746656/howto-vblend
	// -for-32-bit-integer-or-why-is-there-no-mm256-blendv-epi32
	// for using FP blend.
	// We are assuming that the mask is all-ones in each field (not only
	// the highest bit is used), so this works without modifying the mask.
	return _mm_blendv_epi8( l, r, m );
    }

#if 0
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
#endif

    static type from_int( unsigned mask ) {
	return asvector( mask );
    }
    
    static mask_type movemask( vmask_type a ) {
#if __AVX512VL__ && __AVX512DQ__
	return _mm_movepi64_mask( a );
#else
	return _mm_movemask_pd( *(__m128d*)&a );
#endif
    }

    static type asvector( mask_type mask ) {
	// Need to work with 8 32-bit integers as there is no 64-bit srai
	// in AVX2. Luckily, this works just as well.
	type vmask = _mm_set1_epi32( (int)mask );
	const __m128i cnt = _mm_set_epi32( 30, 30, 31, 31 );
	type vmask2 = _mm_sllv_epi32( vmask, cnt );
	return _mm_srai_epi32( vmask2, 31 );
    }

    static type slli( type a, const int_type b ) {
	return _mm_slli_epi64( a, b );
    }
    static type sll( type a, const int_type b ) {
	return _mm_sll_epi64( a, _mm_cvtsi64_si128( b ) );
    }

/*
    template<typename T2>
    static typename vector_type_traits<T2,sizeof(T2)*8>::vmask_type
    asvector( vmask_type mask );
*/

    // static type asmask( vmask_type mask ) { return mask; }
    static mask_type asmask( vmask_type mask ) {
	return _mm_movemask_pd( _mm_castsi128_pd( mask ) );
    }

    static vmask_type cmpgt( type a, type b, target::mt_vmask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm_cmpgt_epi64( a, b );
	else {
	    type one = slli( setone(), 8*W-1 );
	    type ax = bitwise_xor( a, one );
	    type bx = bitwise_xor( b, one );
	    return _mm_cmpgt_epi64( ax, bx );
	}
    }
    static vmask_type cmplt( type a, type b, target::mt_vmask ) {
	return cmpgt( b, a, target::mt_vmask() );
    }

    static type add( type src, vmask_type m, type a, type b ) {
	return _mm_blendv_epi8( src, add( a, b ), m );
    }
    static type add( type src, __m64 m, type a, type b ) {
	__m128i wm = _mm_cvtepi32_epi64( a );
	return add( src, wm, a, b );
    }

    static type add( type a, type b ) { return _mm_add_epi64( a, b ); }
    static type sub( type a, type b ) { return _mm_sub_epi64( a, b ); }
    static type min( type a, type b ) {
#if __AVX512VL__
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm_min_epi64( a, b );
	else
	    return _mm_min_epu64( a, b );
#else
	return blend( cmpgt( a, b, mt_vmask() ), a, b );
#endif
    }
    static type max( type a, type b ) {
#if __AVX512VL__
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm_max_epi64( a, b );
	else
	    return _mm_max_epu64( a, b );
#else
	return blend( cmpgt( a, b, mt_vmask() ), b, a );
#endif
    }
    static type cmpeq( type a, type b, mt_vmask ) {
	return _mm_cmpeq_epi64( a, b );
    }
    static type cmpne( type a, type b, mt_vmask ) {
	return ~_mm_cmpeq_epi64( a, b );
    }
    static mask_type cmpne( type a, type b, mt_mask ) {
#if __AVX512DQ__
	return _mm_cmpeq_epi64_mask( a, b );
#else
	return asmask( cmpne( a, b, mt_vmask() ) );
#endif
    }
    static bool cmpne( type a, type b, mt_bool ) {
	type vcmp = cmpeq( a, b, mt_vmask() );
	return !is_zero( vcmp );
    }
    static bool cmpeq( type a, type b, mt_bool ) {
	type vcmp = cmpeq( a, b, mt_vmask() );
	return asmask( vcmp ) == mtraits::setone();
    }
    static type blend( type mask, type a, type b ) {
	return _mm_blendv_epi8( a, b, mask );
    }
    static type blend( mask_type mask, type a, type b ) {
	return blend( asvector( mask ), a, b );
    }
    static member_type reduce_setif( type val ) {
	assert( 0 && "NYI" );
	return setzero();
    }
    static member_type reduce_add( type val ) {
	static_assert( vlen == 2, "using vlen == 2" );
	return lane0( val ) + lane1( val );
    }
    static member_type reduce_add( type val, mask_type mask ) {
	assert( 0 && "NYI" );
	return setzero();
    }
    static member_type reduce_add( type val, vmask_type mask ) {
	// type zval = _mm_blendv_epi8( setzero(), val, mask );
	// return reduce_add( zval );
	// First filter out zeros, then add up all values
	member_type s = 0;
	if( lane1( mask ) ) s += lane1( val );
	if( lane0( mask ) ) s += lane0( val );
	return s;
    }
    static member_type reduce_logicalor( type val ) {
	return _mm_movemask_epi8( val ) ? ~member_type(0) : member_type(0);
    }
    static member_type reduce_logicalor( type val, type mask ) {
	int v = _mm_movemask_epi8( val );
	int m = _mm_movemask_epi8( mask );
	return (!m | v) ? ~member_type(0) : member_type(0);
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

    static type load( const member_type *a ) {
	return _mm_load_si128( (const type *)a );
    }
    static type loadu( const member_type *a ) {
	return _mm_loadu_si128( (const type *)a );
    }
    static void store( member_type *addr, type val ) {
	_mm_store_si128( (type *)addr, val );
    }
    static void storeu( member_type *addr, type val ) {
	_mm_storeu_si128( (type *)addr, val );
    }

    static type ntload( member_type *addr ) {
	return _mm_stream_load_si128( (type *)addr );
    }
    static void ntstore( member_type *addr, type val ) {
	_mm_stream_si128( (type *)addr, val );
    }

    static type gather( const member_type *a, itype b ) {
#if __AVX2__
	return _mm_i64gather_epi64( (const long long int *)a, b, size );
#else
	return set( a[int_traits::lane1(b)], a[int_traits::lane0(b)] );
#endif
    }
    static type gather( const member_type *a, itype b, mask_type mask ) {
#if __AVX2__
	assert( 0 && "NYI" );
	return setzero();
#else
	member_type ra = mtraits::lane0( mask )
	    ? *(member_type*)( a+int_traits::lane0(b) ): (member_type)0;
	member_type rb = mtraits::lane1( mask )
	    ? *(member_type*)( a+int_traits::lane1(b) ): (member_type)0;
	return set( rb, ra );
#endif
    }
    static type gather( const member_type *a, __m64 b, mask_type mask ) {
	using half_itraits = mmx_4x2<uint32_t>;
#if __AVX2__
	assert( 0 && "NYI" );
	return setzero();
#else
	member_type ra = mtraits::lane0( mask )
	    ? *(member_type*)( a+half_traits::lane0(b) ): (member_type)0;
	member_type rb = mtraits::lane1( mask )
	    ? *(member_type*)( a+half_traits::lane1(b) ): (member_type)0;
	return set( rb, ra );
#endif
    }
/*
*/
    static type gather( const member_type *a, itype b, vmask_type vmask ) {
#if __AVX2__
	return _mm_mask_i64gather_epi64( setzero(), (const long long int *)a, b, vmask, size );
#else
	return set(
	    int_traits::lane1(vmask) ? a[int_traits::lane1(b)] : member_type(0),
	    int_traits::lane0(vmask) ? a[int_traits::lane0(b)] : member_type(0)
	    );
#endif
    }

    static type gather( const member_type *a, __m64 b ) {
	using half_itraits = mmx_4x2<uint32_t>;
	return set( *(member_type *)( a + half_itraits::lane1( b ) ),
		    *(member_type *)( a + half_itraits::lane0( b ) ) );
    }
    static type gather( const member_type *a, __m64 b, vmask_type vmask ) {
#if __AVX2__
	return _mm_mask_i32gather_epi64(
	    setzero(), (const long long int *)a,
	    _mm_cvtsi64_si128( _mm_cvtm64_si64( b ) ), vmask, size );
#else
	using half_itraits = mmx_4x2<uint32_t>;
	return set(
	    int_traits::lane1(vmask) ? a[half_itraits::lane1(b)] : member_type(0),
	    int_traits::lane0(vmask) ? a[half_itraits::lane0(b)] : member_type(0)
	    );
#endif
    }
    static type gather( const member_type *a, __m64 b, __m64 vmask ) {
#if __AVX2__
	__m128i m = _mm_cvtsi64_si128( _mm_cvtm64_si64( vmask ) );
	__m128i v = _mm_unpacklo_epi32( m, m );
	return _mm_mask_i32gather_epi64(
	    setzero(), (const long long int *)a,
	    _mm_cvtsi64_si128( _mm_cvtm64_si64( b ) ), v, size );
#else
	using half_itraits = mmx_4x2<uint32_t>;
	return set(
	    half_traits::lane1(vmask) ? a[half_itraits::lane1(b)] : member_type(0),
	    half_traits::lane0(vmask) ? a[half_itraits::lane0(b)] : member_type(0)
	    );
#endif
    }
    static type gather( const member_type *a, itype b, __m64 vmask ) {
	using half_itraits = mmx_4x2<uint32_t>;
	return set(
	    half_itraits::lane1(vmask) ? a[int_traits::lane1(b)] : member_type(0),
	    half_itraits::lane0(vmask) ? a[int_traits::lane0(b)] : member_type(0)
	    );
    }

    static void scatter( member_type *a, itype b, type c ) {
	a[int_traits::lane0(b)] = lane0(c);
	a[int_traits::lane1(b)] = lane1(c);
    }
    static void scatter( member_type *a, __m64 b, type c ) {
	using half_itraits = mmx_4x2<uint32_t>;
	a[half_itraits::lane0(b)] = lane0(c);
	a[half_itraits::lane1(b)] = lane1(c);
    }
    static void scatter( member_type *a, itype b, type c, vmask_type mask ) {
/*
	using traits2 = vector_type_int_traits<IdxT,vlen*sizeof(IdxT)>;
	using traits3 = vector_type_int_traits<member_type,vlen*sizeof(member_type)>;

	if( traits3::lane0(mask) ) a[traits2::lane0(b)] = lane0(c);
	if( traits3::lane1(mask) ) a[traits2::lane1(b)] = lane1(c);
	if( traits3::lane2(mask) ) a[traits2::lane2(b)] = lane2(c);
	if( traits3::lane3(mask) ) a[traits2::lane3(b)] = lane3(c);
*/
	assert( 0 && "NYI" );
    }
    static void scatter( member_type *a, __m64 b, type c, vmask_type mask ) {
	using half_itraits = mmx_4x2<uint32_t>;
	if( int_traits::lane0( mask ) ) a[half_itraits::lane0(b)] = lane0(c);
	if( int_traits::lane1( mask ) ) a[half_itraits::lane1(b)] = lane1(c);
    }
    static void scatter( const member_type *a, itype b, type c, __m64 vmask ) {
	using half_itraits = mmx_4x2<uint32_t>;
	if( half_itraits::lane0( vmask ) ) a[int_traits::lane0(b)] = lane0(c);
	if( half_itraits::lane1( vmask ) ) a[int_traits::lane1(b)] = lane1(c);
    }
    
    static member_type extract_degree( type v, unsigned degree_bits,
				       unsigned degree_shift ) {
/*
	member_type smsk = ( (member_type(1)<<degree_bits)-1 ) << degree_shift;
	type msk = set1( smsk );
	type vs = _mm_and_si128( v, msk );

	member_type b0 = lane0( vs ) >> ( degree_shift - 0 * degree_bits );
	member_type b1 = lane1( vs ) >> ( degree_shift - 1 * degree_bits );
	member_type b2 = lane2( vs ) >> ( degree_shift - 2 * degree_bits );
	member_type b3 = lane3( vs ) >> ( degree_shift - 3 * degree_bits );

	return ( b0 | b1 ) | ( b2 | b3 );
*/
	assert( 0 && "NYI" );
	return setzero();
    }
    static type extract_source( type v, unsigned degree_bits,
				unsigned degree_shift ) {
/*
	// Written to reuse intermediate values from extract_degree()
	member_type smsk = ( (member_type(1)<<degree_bits)-1 ) << degree_shift;
	type msk = set1( smsk );
	type x = _mm_andnot_si128( msk, v );
	// Now 'sign extend' from the dropped bit
	type lx = _mm_slli_epi32( x, degree_bits );
	type rx = _mm_srai_epi32( lx, degree_bits );
	return rx;
*/
	assert( 0 && "NYI" );
	return setzero();
    }
};
#endif // __SSE4_2__

} // namespace target

#endif // GRAPTOR_TARGET_SSE42_8x2_H
