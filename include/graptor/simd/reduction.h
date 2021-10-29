// -*- c++ -*-

#define REDUCTION_CAT_NX(a,b) a ## b
#define REDUCTION_CAT(a,b) REDUCTION_CAT_NX(a,b)
#define reduction_name(n) REDUCTION_CAT(reduce_, n)
#define REDUCTION_NAME reduction_name(REDUCTION_OP)

template<typename Tr, simd::layout_t Layout1>
auto REDUCTION_NAME( simd::detail::vec<Tr,Layout1> v ) {
    using Tr1 = simd::detail::vdata_traits<typename Tr::member_type,1>;
    return simd::detail::vec<Tr1,simd::lo_constant>(
	Tr::traits::REDUCTION_NAME( v.data() ) );
}

template<typename Tr, simd::layout_t Layout1>
auto REDUCTION_NAME( simd::detail::vec<Tr,Layout1> v,
		     simd::nomask<Tr::VL> ) {
    using Tr1 = simd::detail::vdata_traits<typename Tr::member_type,1>;
    return simd::detail::vec<Tr1,simd::lo_constant>(
	Tr::traits::REDUCTION_NAME( v.data() ) );
}

template<typename Tr, simd::layout_t Layout1>
auto REDUCTION_NAME( simd::detail::vec<Tr,Layout1> v,
		     simd::mask<Tr::W,Tr::VL> m ) {
    using Tr1 = simd::detail::vdata_traits<typename Tr::member_type,1>;
    return simd::detail::vec<Tr1,simd::lo_constant>(
	Tr::traits::REDUCTION_NAME( v.data(), m.get() ) );
}
