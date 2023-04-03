// -*- c++ -*-
#ifndef GRAPTOR_TARGET_BITMASK_H
#define GRAPTOR_TARGET_BITMASK_H

namespace target {

/***********************************************************************
 * Bit masks
 ***********************************************************************/
template<unsigned short VL>
struct mask_type_traits;

template<typename T>
struct is_mask_type_traits : std::false_type { };

template<unsigned short VL>
struct is_mask_type_traits<mask_type_traits<VL>> : std::true_type { };

} // namespace target

// Default definition
#include "graptor/target/bitmask_other.h"

#include "graptor/target/bitmask_1.h"
#include "graptor/target/bitmask_2.h"
#include "graptor/target/bitmask_4.h"
#if __AVX512F__
#include "graptor/target/bitmask_avx512_8.h"
#include "graptor/target/bitmask_avx512_16.h"
#else
#include "graptor/target/bitmask_8.h"
#include "graptor/target/bitmask_16.h"
#endif // __AVX512F__

#if __AVX512F__ && __AVX512BW__
#include "graptor/target/bitmask_avx512_32.h"
#include "graptor/target/bitmask_avx512_64.h"
#else
#include "graptor/target/bitmask_32.h"
#endif // __AVX512F__ && __AVX512BW__

#include "graptor/target/bitmask_128.h"
#if __AVX2__
#include "graptor/target/bitmask_256.h"
#endif

using target::mask_type_traits;

#endif // GRAPTOR_TARGET_BITMASK_H
