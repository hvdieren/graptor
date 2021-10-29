// -*- c++ -*-
#ifndef GRAPTOR_TARGET_CONVERT_SIGN_H
#define GRAPTOR_TARGET_CONVERT_SIGN_H

#include <x86intrin.h>
#include <immintrin.h>

namespace conversion {

#define _CVT_TEMPLATE_DEF__(T,U) 	 	 	 	 	\
    template<unsigned short VL> 	 	 	 	 	\
    struct int_conversion_traits<T, U, VL> {	 	 	 	\
	using src_traits = vector_type_traits_vl<T, VL>;		\
	using dst_traits = vector_type_traits_vl<U, VL>;		\
									\
	static typename dst_traits::type				\
	convert( typename src_traits::type a ) {			\
	    return a;							\
	}								\
    }

_CVT_TEMPLATE_DEF__(int8_t, uint8_t);
_CVT_TEMPLATE_DEF__(uint8_t, int8_t);

_CVT_TEMPLATE_DEF__(int16_t, uint16_t);
_CVT_TEMPLATE_DEF__(uint16_t, int16_t);

_CVT_TEMPLATE_DEF__(int32_t, uint32_t);
_CVT_TEMPLATE_DEF__(uint32_t, int32_t);

_CVT_TEMPLATE_DEF__(int64_t, uint64_t);
_CVT_TEMPLATE_DEF__(uint64_t, int64_t);

#undef _CVT_TEMPLATE_DEF__

} // namespace conversion

#endif // GRAPTOR_TARGET_CONVERT_SIGN_H
