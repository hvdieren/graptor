// -*- c++ -*-

#ifndef GRAPTOR_DSL_SIMD_VECTOR_H
#define GRAPTOR_DSL_SIMD_VECTOR_H

#include "graptor/simd/simd.h"

template<unsigned short W, unsigned short VL>
using simd_mask = simd::mask<W, VL>;

template<unsigned short W, typename I, unsigned short VL>
using simd_mask_ref = simd::mask_ref<W, I, VL>;

template<typename T, unsigned short VL>
using simd_vector = simd::vector<T, VL>;

template<typename T, typename I, unsigned short VL>
using simd_vector_ref = simd::vector_ref<T, I, VL>;

using simd::nomask;
using simd::layout_t;
using simd::lo_unknown;
using simd::lo_constant;
using simd::lo_linear;
using simd::lo_linalgn;


#endif // GRAPTOR_DSL_SIMD_VECTOR_H
