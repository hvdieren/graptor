// -*- c++ -*-
#ifndef GRAPTOR_TARGET_IMPL_SSE42_4x4_H
#define GRAPTOR_TARGET_IMPL_SSE42_4x4_H

#include "graptor/target/sse42_4x4.h"

namespace target {

#if __AVX2__
template<typename T>
template<unsigned short Scale>
typename sse42_4x4<T>::type
sse42_4x4<T>::gather_w( const typename sse42_4x4<T>::member_type * a,
			__m256i b, __m256i vmask ) {
    auto m = conversion::int_conversion_traits<logical<8>,logical<4>,sse42_4x4<T>::vlen>
	::convert( vmask );
    return _mm256_mask_i64gather_epi32(
	setzero(), (const int *)a, b, m, Scale );
}
#endif // __AVX2__

} // namespace target

#endif // GRAPTOR_TARGET_IMPL_SSE42_4x4_H

