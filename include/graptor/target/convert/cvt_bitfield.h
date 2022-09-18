// -*- c++ -*-
#ifndef GRAPTOR_TARGET_CONVERT_BITFIELD_H
#define GRAPTOR_TARGET_CONVERT_BITFIELD_H

#include <x86intrin.h>
#include <immintrin.h>
#include <type_traits>

#include "graptor/target/bitfield.h"

namespace conversion {

// Use signed integers in auxiliary width conversions when allowed, as
// unsigned conversions in AVX2 and earlier can be inefficient.

template<unsigned short B, typename U, unsigned short VL>
struct bf_conversion_traits<bitfield<B>, U, VL> {
    using src_traits = target::bitfield_24_byte<B, B*VL/8>;
    using dst_traits = vector_type_traits_vl<U, VL>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	using T = bitfield<B>;
	
	if constexpr ( T::bits*VL <= 8*dst_traits::W ) {
	    // Broadcast the bit pattern to all lanes, shift and mask out
	    // relevant bits
/* delivers only the lsb
	    auto b = dst_traits::set1( U( a ) );
	    const auto inc = dst_traits::set1inc0();
	    const auto sh = dst_traits::slli( inc, ilog2(T::bits) );
	    auto c = dst_traits::srlv( b, sh );
	    const auto m = dst_traits::srli(
		dst_traits::setone(), 8*dst_traits::W-T::bits );
	    auto d = dst_traits::bitwise_and( c, m );
	    return d;
*/
#if __AVX2__ && !__AVX512VL__
	    if constexpr ( is_logical_v<U> && sizeof(U) == 8
			   && T::bits*VL <= 32 ) {
		// Work-around for missing AVX2 64-bit srai
		// As values are either all 0 or all 1, replicate
		// successive half-width lanes with the same value
		using tr = vector_type_traits_vl<uint32_t,VL*2>;
		auto b = tr::set1( uint32_t( a ) );
		const auto all = tr::setone();
		const auto one = tr::srli( all, 8*tr::W-1 );
		const auto inc = tr::set1inc0();
		const auto sh = tr::bitwise_andnot( one, inc );
		const auto w = tr::set1( uint32_t( 8*tr::W-T::bits ) );
		const auto osh = tr::sub( w, sh );
		auto c = tr::sllv( b, osh );
		auto d = tr::srai( c, 8*tr::W-1 );
		return d;
	    } else
#endif
	    {
		auto b = dst_traits::set1( U( a ) );
		const auto inc = dst_traits::set1inc0();
		const auto sh = dst_traits::slli( inc, ilog2(T::bits) );
		const auto w = dst_traits::set1( U( 8*dst_traits::W-T::bits ) );
		const auto osh = dst_traits::sub( w, sh );
		auto c = dst_traits::sllv( b, osh );
		typename dst_traits::type d;
		if constexpr ( is_logical_v<U> )
		    d = dst_traits::srai( c, 8*dst_traits::W-T::bits );
		else
		    d = dst_traits::srli( c, 8*dst_traits::W-T::bits );
		return d;
	    }
	}

	else if constexpr ( dst_traits::W > 1 && VL > 1 ) {
	    // First scale integer
	    if constexpr ( is_logical_v<U> ) {
		auto b = bf_conversion_traits<T,logical<1>,VL>::convert( a );
		return int_conversion_traits<logical<1>,U,VL>::convert( b );
	    } else {
		auto b = bf_conversion_traits<T,uint8_t,VL>::convert( a );
		return int_conversion_traits<uint8_t,U,VL>::convert( b );
	    }
	}

	else if constexpr ( T::bits == 2 && VL == 8 && dst_traits::W == 1 ) {
	    // Take 16 bits (compacted) and scatter into 8 lanes of 1 byte
	    long long unsigned int mask = is_logical_v<U>
		? 0xc0c0c0c0c0c0c0c0ULL : 0x0303030303030303ULL;
	    auto b = _pdep_u64( (uint64_t)a, mask );
#if GRAPTOR_USE_MMX
	    return _mm_cvtsi64_m64( b );
#else
	    // Top bits cleared
	    return _mm_cvtsi64_si128( b );
#endif
	}

	else if constexpr ( T::bits == 2 && VL == 16 && dst_traits::W == 1 ) {
	    // Take 16 bits (compacted) and scatter into 8 lanes of 1 byte
	    long long unsigned int mask = is_logical_v<U>
		? 0xc0c0c0c0c0c0c0c0ULL : 0x0303030303030303ULL;
	    auto w0 = _pdep_u64( a, mask );
	    auto w1 = _pdep_u64( a >> 16, mask );
	    return _mm_set_epi64x( w1, w0 );
	}

	else if constexpr ( T::bits == 2 && VL == 32 && dst_traits::W == 1 ) {
	    // Take 64 bits (compacted) and scatter into 32 lanes of 1 byte
	    long long unsigned int mask = is_logical_v<U>
		? 0xc0c0c0c0c0c0c0c0ULL : 0x0303030303030303ULL;
	    auto w0 = _pdep_u64( a, mask );
	    auto w1 = _pdep_u64( a >> 16, mask );
	    auto w2 = _pdep_u64( a >> 32, mask );
	    auto w3 = _pdep_u64( a >> 48, mask );
	    return _mm256_set_epi64x( w3, w2, w1, w0 );
	}

	else if constexpr ( T::bits == 2 && dst_traits::W == 1
#if __AVX512F__
		       && VL >= 64
#elif __AVX2__
		       && VL > 32
#elif __SSE4_2__
		       && VL > 16
#endif
	    ) {
	    auto lo = bf_conversion_traits<T,U,VL/2>::convert(
		src_traits::lower_half( a ) );
	    auto hi = bf_conversion_traits<T,U,VL/2>::convert(
		src_traits::upper_half( a ) );
	    return dst_traits::set_pair( hi, lo );
	}
	
	assert( 0 && "NYI" );
    }
};

template<typename T, unsigned short B, unsigned short VL>
struct bf_conversion_traits<T, bitfield<B>, VL> {
    using src_traits = vector_type_traits_vl<T, VL>;
    using dst_traits = target::bitfield_24_byte<B, B*VL/8>;

    static typename dst_traits::type convert( typename src_traits::type a ) {
	using U = bitfield<B>;
	if constexpr ( src_traits::W > 1 ) {
	    if constexpr ( is_logical_v<T> || is_logical_v<U> ) {
		auto b = int_conversion_traits<T,logical<1>,VL>::convert( a );
		return bf_conversion_traits<logical<1>,U,VL>::convert( b );
	    } else {
		auto b = int_conversion_traits<T,uint8_t,VL>::convert( a );
		return bf_conversion_traits<uint8_t,U,VL>::convert( b );
	    }
	}
	
	if constexpr ( U::bits == 1 && 0 ) {
	    // Pick top bit if logical
	    long long unsigned int mask = is_logical_v<T>
		? 0x8080808080808080ULL : 0x0101010101010101ULL;

	    if constexpr ( VL == 8 && src_traits::W == 1 ) {
#if __AVX512VL__ && __AVX512BW__
		auto b = a;
		if constexpr ( !is_logical_v<T> )
		    // shift bottom bit to top position
		    b = src_traits::slli( a, src_traits::B-1 );
		auto mm = src_traits::slli( src_traits::setone(),
					    src_traits::B-1 );
		auto c = src_traits::cmpge( b, mm, target::mt_mask() );
		return static_cast<typename dst_traits::type>( c );
#else
		uint64_t m = mask;
		uint64_t b = src_traits::asint( a );
		uint64_t c = _pext_u64( b, m );
		return static_cast<typename dst_traits::type>( c );
#endif
	    }
	}
	
	if constexpr ( U::bits == 2 ) {
	    // Pick top bits
	    long long unsigned int mask = is_logical_v<T>
		? 0xc0c0c0c0c0c0c0c0ULL : 0x0303030303030303ULL;

	    if constexpr ( VL == 8 && src_traits::W == 1 ) {
		uint64_t m = mask;
		uint64_t b = src_traits::asint( a );
		uint64_t c = _pext_u64( b, m );
		return static_cast<typename dst_traits::type>( c );
	    }
	    if constexpr ( VL == 16 && src_traits::W == 1 ) {
		uint64_t w0 = _pext_u64( _mm_extract_epi64( a, 0 ), mask );
		uint64_t w1 = _pext_u64( _mm_extract_epi64( a, 1 ), mask );
		auto c = ( w1 << 16 ) | w0;
		return static_cast<typename dst_traits::type>( c );
	    }
	    if constexpr ( VL == 32 && src_traits::W == 1 ) {
#if 1
		uint64_t w0 = _pext_u64( _mm256_extract_epi64( a, 0 ), mask );
		uint64_t w1 = _pext_u64( _mm256_extract_epi64( a, 1 ), mask );
		uint64_t w2 = _pext_u64( _mm256_extract_epi64( a, 2 ), mask );
		uint64_t w3 = _pext_u64( _mm256_extract_epi64( a, 3 ), mask );
		auto c = ( w3 << 48 ) | ( w2 << 32 ) | ( w1 << 16 ) | w0;
		return static_cast<typename dst_traits::type>( c );
#else
		auto shuf = ;
		auto shift = _mm256_load_si256( shuffle_encoding_bitfield_sllv_2bx32 );
		auto b = _mm256_shuffle_epi8( a, shuf );
		auto c = _mm256_sllv_epi32( b, shift );
		auto d = _mm256_hadd_epi16( c, c );
		auto e = _mm256_hadd_epi16( d, d );
		auto f = _mm256_cvtepi32_epi8( e );
		return static_cast<typename dst_traits::type>( f );
#endif
	    }
	    if constexpr ( VL == 64 && src_traits::W == 1 ) {
		auto lo = bf_conversion_traits<T,U,VL/2>::convert(
		    src_traits::lower_half( a ) );
		auto hi = bf_conversion_traits<T,U,VL/2>::convert(
		    src_traits::upper_half( a ) );
		return dst_traits::set_pair( hi, lo );
	    }
	    if constexpr ( VL >= 128 && src_traits::W == 1 ) {
		auto lo = bf_conversion_traits<T,U,VL/2>::convert(
		    src_traits::lower_half( a ) );
		auto hi = bf_conversion_traits<T,U,VL/2>::convert(
		    src_traits::upper_half( a ) );
		return dst_traits::set_pair( hi, lo );
	    }
	}
	
	assert( 0 && "NYI" );
    }
};

} // namespace conversion

#endif // GRAPTOR_TARGET_CONVERT_BITFIELD_H
