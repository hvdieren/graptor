// -*- c++ -*-
#ifndef GRAPTOR_TARGET_CONVERT_2I_8I_H
#define GRAPTOR_TARGET_CONVERT_2I_8I_H

#include <x86intrin.h>
#include <immintrin.h>

namespace conversion {

#if __AVX2__
namespace {

static __m256i avx2_convert_2i_8i( __m128i a ) {
#if __AVX2__
    return _mm256_cvtepi16_epi64( a );
#else
    assert( 0 && "NYI" );
#endif
}

static __m256i avx2_convert_2u_8i( __m128i a ) {
#if __AVX2__
    return _mm256_cvtepu16_epi64( a );
#else
    assert( 0 && "NYI" );
#endif
}

static __m256i avx2_convert_2i_8i( __m64 a ) {
#if __SSE4_2__
    auto b = _mm_cvtsi64_si128( _mm_cvtm64_si64( a ) );
    return avx2_convert_2i_8i( b );
#else
    assert( 0 && "NYI" );
#endif
}

static __m256i avx2_convert_2u_8i( __m64 a ) {
#if __SSE4_2__
    auto b = _mm_cvtsi64_si128( _mm_cvtm64_si64( a ) );
    return avx2_convert_2u_8i( b );
#else
    assert( 0 && "NYI" );
#endif
}

} // namespace anonymous

template<>
struct int_conversion_traits<int16_t, int64_t, 4> {
    static __m256i convert( __m64 a ) {
	return avx2_convert_2i_8i( a );
    }
    static __m256i convert( __m128i a ) {
	return avx2_convert_2i_8i( a );
    }
};

template<>
struct int_conversion_traits<int16_t, uint64_t, 4> {
    static __m256i convert( __m64 a ) {
	return avx2_convert_2i_8i( a );
    }
    static __m256i convert( __m128i a ) {
	return avx2_convert_2i_8i( a );
    }
};

template<>
struct int_conversion_traits<uint16_t, int64_t, 4> {
    static __m256i convert( __m64 a ) {
	return avx2_convert_2u_8i( a );
    }
    static __m256i convert( __m128i a ) {
	return avx2_convert_2u_8i( a );
    }
};

template<>
struct int_conversion_traits<uint16_t, uint64_t, 4> {
    static __m256i convert( __m64 a ) {
	return avx2_convert_2u_8i( a );
    }
    static __m256i convert( __m128i a ) {
	return avx2_convert_2u_8i( a );
    }
};
#endif

#if __AVX512F__
namespace {

static __m512i avx512_convert_2i_8i( __m128i a ) {
    return _mm512_cvtepi16_epi64( a );
}

static __m512i avx512_convert_2u_8i( __m128i a ) {
    return _mm512_cvtepu16_epi64( a );
}

} // namespace anonymous

template<>
struct int_conversion_traits<int16_t, int64_t, 8> {
    static __m512i convert( __m128i a ) {
	return avx512_convert_2i_8i( a );
    }
};

template<>
struct int_conversion_traits<int16_t, uint64_t, 8> {
    static __m512i convert( __m128i a ) {
	return avx512_convert_2i_8i( a );
    }
};

template<>
struct int_conversion_traits<uint16_t, int64_t, 8> {
    static __m512i convert( __m128i a ) {
	return avx512_convert_2u_8i( a );
    }
};

template<>
struct int_conversion_traits<uint16_t, uint64_t, 8> {
    static __m512i convert( __m128i a ) {
	return avx512_convert_2u_8i( a );
    }
};
#endif


} // namespace conversion

#endif // GRAPTOR_TARGET_CONVERT_2I_8I_H
