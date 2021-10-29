// -*- c++ -*-
#ifndef GRAPTOR_TARGET_IMPL_AVX2_4x8_H
#define GRAPTOR_TARGET_IMPL_AVX2_4x8_H

#include "graptor/target/avx2_4x8.h"
#include "graptor/target/sse42_4x4.h"

namespace target {

#if __AVX2__
template<typename T>
typename avx2_4x8<T>::type avx2_4x8<T>::gather(
    const typename avx2_4x8<T>::member_type *a,
    vpair<typename avx2_4x8<T>::itype,typename avx2_4x8<T>::itype> b,
    vpair<typename avx2_4x8<T>::vmask_type,typename avx2_4x8<T>::vmask_type> vmask )
{
    using ht = sse42_4x4<member_type>;
    auto lo = ht::template gather_w<W>( a, b.a, vmask.a );
    auto hi = ht::template gather_w<W>( a, b.b, vmask.b );
    return set_pair( hi, lo );
}

template<typename T>
typename avx2_4x8<T>::type avx2_4x8<T>::gather(
    const typename avx2_4x8<T>::member_type *a,
    vpair<typename avx2_4x8<T>::itype,typename avx2_4x8<T>::itype> b )
{
    using ht = sse42_4x4<member_type>;
    auto lo = ht::template gather_w<W>( a, b.a );
    auto hi = ht::template gather_w<W>( a, b.b );
    return set_pair( hi, lo );
}
template<typename T>
void avx2_4x8<T>::scatter( typename avx2_4x8<T>::member_type *a,
			   vpair<typename avx2_4x8<T>::itype,
			         typename avx2_4x8<T>::itype> b,
			   typename avx2_4x8<T>::type c,
			   typename avx2_4x8<T>::vmask_type mask ) {
#if __AVX512F__
	assert( 0 && "Use 512-bit scatter with mask" );
#else
	using dhalf_traits = avx2_8x4<uint64_t>;
	using dint_traits = vt_recursive<uint64_t,8,64,dhalf_traits>;
	if( int_traits::lane0(mask) ) a[dint_traits::lane(b,0)] = lane0(c);
	if( int_traits::lane1(mask) ) a[dint_traits::lane(b,1)] = lane1(c);
	if( int_traits::lane2(mask) ) a[dint_traits::lane(b,2)] = lane2(c);
	if( int_traits::lane3(mask) ) a[dint_traits::lane(b,3)] = lane3(c);
	if( int_traits::lane4(mask) ) a[dint_traits::lane(b,4)] = lane4(c);
	if( int_traits::lane5(mask) ) a[dint_traits::lane(b,5)] = lane5(c);
	if( int_traits::lane6(mask) ) a[dint_traits::lane(b,6)] = lane6(c);
	if( int_traits::lane7(mask) ) a[dint_traits::lane(b,7)] = lane7(c);
#endif
}

#endif // __AVX2__

} // namespace target

#endif // GRAPTOR_TARGET_IMPL_AVX2_4x8_H

