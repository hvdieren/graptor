// -*- c++ -*-
#ifndef GRAPTOR_LONGINT_IMPL_H
#define GRAPTOR_LONGINT_IMPL_H

#include "graptor/longint.h"
#include "graptor/target/vt_longint.h"

/***********************************************************************
 * Implementations
 ***********************************************************************/

#ifdef __SSE4_2__

inline
bool longint<16>::operator == ( self_type a ) const {
    return target::sse42_16x1<longint<16>>
	::cmpeq( get(), a.get(), target::mt_bool() );
}

inline
bool longint<16>::operator != ( self_type a ) const {
    return target::sse42_16x1<longint<16>>
	::cmpne( get(), a.get(), target::mt_bool() );
}

#endif // __SSE4_1__

#ifdef __AVX2__

inline
bool longint<32>::operator == ( self_type a ) const {
    __m256i m = _mm256_cmpeq_epi64( get(), a.get() );
    int z = _mm256_testz_si256( m, m ); // z is 1 if m == 0
    return (bool)z;
}

inline
bool longint<32>::operator != ( self_type a ) const {
    return ! this->operator == ( a );
}

#endif // __SSE4_1__

#endif // GRAPTOR_LONGINT_IMPL_H
