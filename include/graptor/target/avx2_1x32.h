// -*- c++ -*-
#ifndef GRAPTOR_TARGET_AVX2_1x32_H
#define GRAPTOR_TARGET_AVX2_1x32_H

#include <x86intrin.h>
#include <immintrin.h>
#include <cstdint>

#include "graptor/target/decl.h"
#include "graptor/target/vt_recursive.h"
// #include "graptor/target/sse42_1x16.h"

#if __AVX2__
#include "graptor/target/avx2_bitwise.h"
#include "graptor/target/avx2_4x8.h"
#include "graptor/target/avx2_2x16.h"
#include "graptor/target/sse42_1x16.h"
#endif // __AVX2__

namespace target {

/***********************************************************************
 * AVX2 32 byte-sized integers
 * This is poorly tested
 ***********************************************************************/
#if __AVX2__ // most code removed due to lack of testing
template<typename T = uint8_t>
struct avx2_1x32 : public avx2_bitwise {
    static_assert( sizeof(T) == 1, 
		   "version of template class for 1-byte integers" );
public:
    static constexpr size_t W = 1;
    static constexpr size_t B = 8*W;
    static constexpr size_t vlen = 32;
    static constexpr unsigned short size = W * vlen;

    using member_type = T;
    using type = __m256i;
    using vmask_type = __m256i;
    using itype = __m256i;
    using int_type = uint8_t;

    using mask_traits = mask_type_traits<32>;
    using mask_type = typename mask_traits::type;

    using mt_preferred = target::mt_vmask;

    using half_traits = sse42_1x16<member_type>;
    // using recursive_traits = vt_recursive<member_type,1,32,half_traits>;
    using int_traits = avx2_1x32<int_type>;

    static member_type lane( type a, int idx ) {
	switch( idx ) {
	case 0: return (member_type) _mm256_extract_epi8( a, 0 );
	case 1: return (member_type) _mm256_extract_epi8( a, 1 );
	case 2: return (member_type) _mm256_extract_epi8( a, 2 );
	case 3: return (member_type) _mm256_extract_epi8( a, 3 );
	case 4: return (member_type) _mm256_extract_epi8( a, 4 );
	case 5: return (member_type) _mm256_extract_epi8( a, 5 );
	case 6: return (member_type) _mm256_extract_epi8( a, 6 );
	case 7: return (member_type) _mm256_extract_epi8( a, 7 );
	case 8: return (member_type) _mm256_extract_epi8( a, 8 );
	case 9: return (member_type) _mm256_extract_epi8( a, 9 );
	case 10: return (member_type) _mm256_extract_epi8( a, 10 );
	case 11: return (member_type) _mm256_extract_epi8( a, 11 );
	case 12: return (member_type) _mm256_extract_epi8( a, 12 );
	case 13: return (member_type) _mm256_extract_epi8( a, 13 );
	case 14: return (member_type) _mm256_extract_epi8( a, 14 );
	case 15: return (member_type) _mm256_extract_epi8( a, 15 );
	case 16: return (member_type) _mm256_extract_epi8( a, 16 );
	case 17: return (member_type) _mm256_extract_epi8( a, 17 );
	case 18: return (member_type) _mm256_extract_epi8( a, 18 );
	case 19: return (member_type) _mm256_extract_epi8( a, 19 );
	case 20: return (member_type) _mm256_extract_epi8( a, 20 );
	case 21: return (member_type) _mm256_extract_epi8( a, 21 );
	case 22: return (member_type) _mm256_extract_epi8( a, 22 );
	case 23: return (member_type) _mm256_extract_epi8( a, 23 );
	case 24: return (member_type) _mm256_extract_epi8( a, 24 );
	case 25: return (member_type) _mm256_extract_epi8( a, 25 );
	case 26: return (member_type) _mm256_extract_epi8( a, 26 );
	case 27: return (member_type) _mm256_extract_epi8( a, 27 );
	case 28: return (member_type) _mm256_extract_epi8( a, 28 );
	case 29: return (member_type) _mm256_extract_epi8( a, 29 );
	case 30: return (member_type) _mm256_extract_epi8( a, 30 );
	case 31: return (member_type) _mm256_extract_epi8( a, 31 );
	default:
	    assert( 0 && "should not get here" );
	}
    }
    
    static type set1( member_type a ) { return _mm256_set1_epi8( a ); }
    static type set1inc0() {
	return _mm256_set_epi64x( 0x1f1e1d1c1b1a1918u, 0x1716151413121110u,
				  0x0f0e0d0c0b0a0908u, 0x0706050403020100u );
    }
    static type set1inc( member_type a ) {
	return add( set1inc0(), _mm256_set1_epi8( a ) );
    }

    static mask_type asmask( vmask_type a ) {
#if __AVX512VL__ && __AVX512BW__
	return _mm256_movepi8_mask( a );
#elif __AVX512F__ && __AVX512VL__
	// The mask is necessary if we only want to pick up the highest bit
	using traits16 = avx2_2x16<uint16_t>;
	vmask_type me = traits16::slli( traits16::setone(), 15 );
	vmask_type mo = traits16::srli( me, 8 );
	vmask_type m = bitwise_or( me, mo );
	auto am = bitwise_and( a, m );
	return _mm256_cmpeq_epi8_mask( am, m );
#else
	return _mm256_movemask_epi8( a );
#endif
    }

    static type blendm( mask_type m, type l, type r ) {
	return blend( m, l, r );
    }
    static type blendm( vmask_type m, type l, type r ) {
	return blend( m, l, r );
    }
    static type blend( mask_type m, type l, type r ) {
#if __AVX512VL__ && __AVX512BW__
	return _mm256_mask_blend_epi8( m, l, r );
#else
	assert( 0 && "NYI" );
#endif
    }
    static type blend( vmask_type m, type l, type r ) {
	return _mm256_blendv_epi8( l, r, m );
    }

    static vmask_type cmpeq( type a, type b, mt_vmask ) {
	return _mm256_cmpeq_epi8( a, b );
    }
    static vmask_type cmpne( type a, type b, mt_vmask ) {
	return bitwise_invert( cmpeq( a, b, mt_vmask() ) );
    }
    static vmask_type cmpgt( type a, type b, mt_vmask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm256_cmpgt_epi8( a, b );
	else {
	    type one = set1( 0x80 );
	    type ax = bitwise_xor( a, one );
	    type bx = bitwise_xor( b, one );
	    return _mm256_cmpgt_epi8( ax, bx );
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
	return logical_or( cmplt( a, b, mt_vmask() ),
			   cmpeq( a, b, mt_vmask() ) );
    }

#if __AVX512VL__ && __AVX512BW__
    static mask_type cmpeq( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm256_cmpeq_epi8_mask( a, b );
	else
	    return _mm256_cmpeq_epu8_mask( a, b );
    }
    static mask_type cmpne( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm256_cmpneq_epi8_mask( a, b );
	else
	    return _mm256_cmpneq_epu8_mask( a, b );
    }
    static mask_type cmpgt( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm256_cmpgt_epi8_mask( a, b );
	else
	    return _mm256_cmpgt_epu8_mask( a, b );
    }
    static mask_type cmpge( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm256_cmpge_epi8_mask( a, b );
	else
	    return _mm256_cmpge_epu8_mask( a, b );
    }
    static mask_type cmplt( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm256_cmplt_epi8_mask( a, b );
	else
	    return _mm256_cmplt_epu8_mask( a, b );
    }
    static mask_type cmple( type a, type b, mt_mask ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm256_cmple_epi8_mask( a, b );
	else
	    return _mm256_cmple_epu8_mask( a, b );
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
	return is_zero( bitwise_xor( a, b ) );
    }
    static bool cmpne( type a, type b, mt_bool ) {
	return !cmpeq( a, b, mt_bool() );
    }

    static type srli( type a, unsigned int sh ) {
	auto b = _mm256_srli_epi32( a, sh );
	auto m = set1( (member_type)((1<<(W*8-sh))-1) );
	auto c = _mm256_and_si256( b, m );
	return c;
    }
    static type srlv( type a, type sh ) {
	assert( 0 && "NYI" );
    }

    static type add( type a, type b ) {
	return _mm256_add_epi8( a, b );
    }
    static type sub( type a, type b ) {
	return _mm256_sub_epi8( a, b );
    }

    static type min( type a, type b ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm256_min_epi8( a, b );
	else
	    return _mm256_min_epu8( a, b );
    }
    static type max( type a, type b ) {
	if constexpr ( std::is_signed_v<member_type> )
	    return _mm256_max_epi8( a, b );
	else
	    return _mm256_max_epu8( a, b );
    }

    static member_type reduce_max( type val ) {
	auto lo = lower_half( val );
	auto hi = upper_half( val );
	return half_traits::reduce_max(
	    half_traits::blend(
		half_traits::cmpge( lo, hi, mt_mask() ), hi, lo ) );
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

    static type load( const member_type * a ) {
	return _mm256_load_si256( (const type *)a );
    }
    static type loadu( const member_type * a ) {
	return _mm256_loadu_si256( (const type *)a );
    }
    static void store( member_type *addr, type val ) {
	_mm256_store_si256( (type *)addr, val );
    }
    static void storeu( member_type *addr, type val ) {
	_mm256_storeu_si256( (type *)addr, val );
    }
    
    template<typename IdxT>
    static type gather( const member_type *a, IdxT b ) {
#if __AVX512F__
	using wt = avx512_4x16<uint32_t>;
	using idxty = int_type_of_size_t<sizeof(IdxT)/vlen>;
	using it = vector_type_traits_vl<idxty,vlen>;
	auto g = wt::template gather_w<W>(
	    reinterpret_cast<const uint32_t *>( a ),
	    it::lower_half( b ) );
	auto h = wt::template gather_w<W>(
	    reinterpret_cast<const uint32_t *>( a ),
	    it::upper_half( b ) );
	auto i = _mm512_cvtepi32_epi8( g );
	auto j = _mm512_cvtepi32_epi8( h );
	return set_pair( j, i );
#else
	assert( 0 && "NYI" );
#endif
    }
    template<typename IdxT>
    static type gather( const member_type *a, IdxT b, mask_type m ) {
#if __AVX512F__
	using wt = avx512_4x16<uint32_t>;
	using idxty = int_type_of_size_t<sizeof(IdxT)/vlen>;
	using it = vector_type_traits_vl<idxty,vlen>;
	auto g = wt::template gather_w<W>(
	    reinterpret_cast<const uint32_t *>( a ),
	    it::lower_half( b ), mask_traits::lower_half( m ) );
	auto h = wt::template gather_w<W>(
	    reinterpret_cast<const uint32_t *>( a ),
	    it::upper_half( b ), mask_traits::upper_half( m ) );
	auto i = _mm512_cvtepi32_epi8( g );
	auto j = _mm512_cvtepi32_epi8( h );
	return set_pair( j, i );
#else
	assert( 0 && "NYI" );
#endif
    }
    template<typename IdxT>
    static type gather( const member_type *a, IdxT b, IdxT m ) {
#if __AVX512F__
	using wt = avx512_4x16<uint32_t>;
	using idxty = int_type_of_size_t<sizeof(IdxT)/vlen>;
	using it = vector_type_traits_vl<idxty,vlen>;
	auto g = wt::template gather_w<W>(
	    reinterpret_cast<const uint32_t *>( a ),
	    it::lower_half( b ), it::lower_half( m ) );
	auto h = wt::template gather_w<W>(
	    reinterpret_cast<const uint32_t *>( a ),
	    it::upper_half( b ), it::upper_half( m ) );
	auto i = _mm512_cvtepi32_epi8( g );
	auto j = _mm512_cvtepi32_epi8( h );
	return set_pair( j, i );
#else
	using wt = sse42_1x16<member_type>;
	auto g = wt::gather( a, b.a, m.a ); // gather 1x16 bytes
	auto h = wt::gather( a, b.b, m.b ); // gather 1x16 bytes
	return set_pair( h, g );
#endif
    }

};
#endif // __AVX2__

} // namespace target

#endif // GRAPTOR_TARGET_AVX2_1x32_H
