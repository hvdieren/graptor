// -*- c++ -*-
#ifndef GRAPTOR_FRONTIER_TYPE_H
#define GRAPTOR_FRONTIER_TYPE_H

#include "graptor/simd/decl.h"

// Pre-declarations to avoid taking in full header file
template<typename StoredTy, typename Enable = void>
struct array_encoding;

template<unsigned short Bits>
struct array_encoding_bit;

template<unsigned short W, bool MSB, typename Enable = void>
struct array_encoding_msb;

enum class frontier_type {
    ft_true = 0,
    ft_unbacked = 1,
    ft_bool = 2,
    ft_bit = 3,
    ft_logical1 = 4,
    ft_logical2 = 5,
    ft_logical4 = 6,
    ft_logical8 = 7,
    ft_sparse = 8,
    ft_bit2 = 9,
    ft_msb4 = 10,
    ft_N = 11
};

extern const char * frontier_type_names[static_cast<std::underlying_type_t<frontier_type>>( frontier_type::ft_N )+1];

inline std::ostream & operator << ( std::ostream & os, frontier_type fr ) {
    int ifr = (int) fr;
    if( ifr >= 0 && ifr < (int)frontier_type::ft_N )
	return os << frontier_type_names[ifr];
    else
	return os << frontier_type_names[(int)frontier_type::ft_N];
}

// Default information
template<frontier_type ftype_, unsigned short VL, typename Enable = void>
struct frontier_params; /* {
    static constexpr frontier_type ftype = ftype_;
    using type = void;
    }; */

template<unsigned short VL_>
struct frontier_params<frontier_type::ft_bool,VL_> {
    static constexpr unsigned short W = 1;
    static constexpr unsigned short VL = VL_;
    static constexpr frontier_type ftype = frontier_type::ft_bool;
    using type = bool;
    using data_type = simd::ty<bool,VL>;
    using mask_type = simd::detail::mask_bool_traits;
    using encoding = array_encoding<type>;
};

template<>
struct frontier_params<frontier_type::ft_bit,0> {
    static constexpr unsigned short W = 0;
    static constexpr unsigned short VL = 0;
    static constexpr frontier_type ftype = frontier_type::ft_bit;
    using type = unsigned char;
    using data_type = simd::ty<void,VL>;
    using mask_type = simd::detail::mask_bit_traits<VL>;
    using encoding = array_encoding<type>;
};

template<unsigned short VL_>
struct frontier_params<frontier_type::ft_bit2,VL_> {
    static constexpr unsigned short W = 0;
    static constexpr unsigned short VL = VL_;
    static constexpr frontier_type ftype = frontier_type::ft_bit2;
    using type = unsigned char;
    using data_type = simd::ty<bitfield<2>,VL>;
    using mask_type = simd::detail::mask_bit_logical_traits<2,VL>;
    using encoding = array_encoding_bit<1>;
};

template<unsigned short VL_>
struct frontier_params<frontier_type::ft_bit,VL_,std::enable_if_t<VL_ != 0>> {
    static constexpr unsigned short W = 0;
    static constexpr unsigned short VL = VL_;
    static constexpr frontier_type ftype = frontier_type::ft_bit;
    using type = typename mask_type_traits<1>::type;
    using data_type = simd::ty<void,VL>;
    using mask_type = simd::detail::mask_bit_traits<VL>;
    using encoding = array_encoding_bit<2>;
};

template<unsigned short VL_>
struct frontier_params<frontier_type::ft_logical1,VL_> {
    static constexpr unsigned short W = 1;
    static constexpr unsigned short VL = VL_;
    static constexpr frontier_type ftype = frontier_type::ft_logical1;
    using type = logical<1>;
    using data_type = simd::detail::mask_logical_traits<W,VL>;
    using mask_type = data_type;
    using encoding = array_encoding<type>;
};

template<unsigned short VL_>
struct frontier_params<frontier_type::ft_logical2,VL_> {
    static constexpr unsigned short W = 2;
    static constexpr unsigned short VL = VL_;
    static constexpr frontier_type ftype = frontier_type::ft_logical2;
    using type = logical<2>;
    using data_type = simd::detail::mask_logical_traits<W,VL>;
    using mask_type = data_type;
    using encoding = array_encoding<type>;
};

template<unsigned short VL_>
struct frontier_params<frontier_type::ft_logical4,VL_> {
    static constexpr unsigned short W = 4;
    static constexpr unsigned short VL = VL_;
    static constexpr frontier_type ftype = frontier_type::ft_logical4;
    using type = logical<4>;
    using data_type = simd::detail::mask_logical_traits<W,VL>;
    using mask_type = data_type;
    using encoding = array_encoding<type>;
};

template<unsigned short VL_>
struct frontier_params<frontier_type::ft_logical8,VL_> {
    static constexpr unsigned short W = 8;
    static constexpr unsigned short VL = VL_;
    static constexpr frontier_type ftype = frontier_type::ft_logical8;
    using type = logical<8>;
    using data_type = simd::detail::mask_logical_traits<W,VL>;
    using mask_type = data_type;
    using encoding = array_encoding<type>;
};

template<unsigned short VL_>
struct frontier_params<frontier_type::ft_msb4,VL_> {
    static constexpr unsigned short W = 4;
    static constexpr unsigned short VL = VL_;
    static constexpr frontier_type ftype = frontier_type::ft_msb4;
    using type = logical<4>;
    using data_type = simd::detail::mask_logical_traits<W,VL>;
    using mask_type = data_type;
    using encoding = array_encoding_msb<W,true>;
};

enum frontier_mode {
    fm_all_true = 0,
    fm_reduction = 1,
    fm_calculate = 2,
    fm_N = 3
};

// Pre-declaration
class frontier;

extern const char * frontier_mode_names[fm_N+1];

inline std::ostream & operator << ( std::ostream & os, frontier_mode fr ) {
    int ifr = (int) fr;
    if( ifr >= 0 && ifr < (int)fm_N )
	return os << frontier_mode_names[ifr];
    else
	return os << frontier_mode_names[(int)fm_N];
}

#endif // GRAPTOR_FRONTIER_TYPE_H
