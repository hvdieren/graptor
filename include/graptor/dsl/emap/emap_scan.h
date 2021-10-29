// -*- c++ -*-
#ifndef GRAPTOR_DSL_EMAP_EMAP_SCAN_H
#define GRAPTOR_DSL_EMAP_EMAP_SCAN_H

template<unsigned short VL, typename fVID, typename RExpr, typename RFExpr>
__attribute__((always_inline))
static inline void emap_scan(
    const partitioner & part,
    RExpr pvop0,
    RFExpr pvopf0 ) {
    int np = part.get_num_partitions();
    for( int p=1; p < np; ++p ) {
	auto pp = simd::template create_constant<simd::ty<fVID,1>>( VL*(fVID)p );
	auto m = expr::create_value_map2<VL,VL,expr::vk_pid>( pp );
	expr::cache<> c;
	expr::evaluate( c, m, pvop0 );
    }

    auto pp = simd::template create_zero<simd::ty<fVID,1>>();
    auto m = expr::create_value_map2<VL,VL,expr::vk_pid>( pp );
    expr::cache<> c;
    expr::evaluate( c, m, pvopf0 );
}

template<unsigned short VL, typename fVID, typename Environment,
	 typename RExpr, typename RFExpr>
__attribute__((always_inline))
static inline void emap_scan(
    const Environment & env,
    const partitioner & part,
    RExpr pvop0,
    RFExpr pvopf0 ) {
    int np = part.get_num_partitions();
    for( int p=1; p < np; ++p ) {
	auto pp = simd::template create_constant<simd::ty<fVID,1>>( VL*(fVID)p );
	auto m = expr::create_value_map2<VL,VL,expr::vk_pid>( pp );
	expr::cache<> c;
	env.evaluate( c, m, pvop0 );
    }

    auto pp = simd::template create_zero<simd::ty<fVID,1>>();
    auto m = expr::create_value_map2<VL,VL,expr::vk_pid>( pp );
    expr::cache<> c;
    env.evaluate( c, m, pvopf0 );
}

#endif // GRAPTOR_DSL_EMAP_EMAP_SCAN_H
