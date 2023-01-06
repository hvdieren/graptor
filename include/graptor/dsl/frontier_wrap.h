// -*- c++ -*-
#ifndef GRAPTOR_DSL_FRONTIER_WRAP_H
#define GRAPTOR_DSL_FRONTIER_WRAP_H

#include "graptor/frontier.h"
#include "graptor/dsl/ast.h"

/*======================================================================*
 * Auxiliaries for DSL 	 	 	 	 	 	 	*
 *======================================================================*/

// Actively used in edgemap in older versions of the edge traversal code
template<typename Operator,typename StoreTy,typename OperateTy>
struct wrap_frontier_read2 {
    expr::array_ro<StoreTy, VID, expr::aid_frontier_old_2,
		   array_encoding<StoreTy>, false> frontier;
    Operator op;

    static constexpr bool is_scan = Operator::is_scan;
    
    wrap_frontier_read2( const StoreTy *frontier_, Operator op_ )
	: frontier( (StoreTy *) frontier_ ), // TODO
	  op( op_ ) {
	// assert( frontier_ );
    }
    
    // This works well for vector sources, but what about scalars?
    // In those cases we would like to use a conditional
    // In AVX-512, we have a register mask and we might want to use a
    // conditional as well in case all lanes are false
    template<typename VIDSrc, typename VIDDst>
    auto relax( VIDSrc s, VIDDst d ) {
	using Tr = typename simd::detail::vdata_traits<OperateTy,VIDDst::VL>; 
	return op.relax(
	    add_mask( remove_mask( s ),
		      // Might also use cvt_to_kmask as identical now
		      expr::make_unop_cvt_to_mask<Tr>( frontier[s] ) ),
	    d );
    }

    template<typename VIDDst>
    auto active( VIDDst d ) {
	return op.active( d );
    }

    template<typename VIDType>
    auto operator() ( VIDType vid ) {
	// wrap_frontier_read2 is also used by vertexmap
	using Tr = typename simd::detail::vdata_traits<OperateTy,VIDType::VL>; 
	static_assert( VIDType::VL == 1 || simd::detail::is_mask_bit_traits<Tr>::value, "test" );
	return op(
	    add_mask( remove_mask( vid ),
		      // Might also use cvt_to_kmask as identical now
		      expr::make_unop_cvt_to_mask<Tr>( frontier[vid] ) ) );
    }

    template<typename VIDDst>
    auto vertexop( VIDDst d ) {
	return op.vertexop( d );
    }
};

// Used in one instance in vertexmap where replacement by wrap_frontier_read_m
// is unclear due to lack of knowledge of VL
template<typename Operator,typename StoreTy>
struct wrap_frontier_read3 {
    expr::array_ro<StoreTy, VID, expr::aid_frontier_old_2,
		   array_encoding<StoreTy>,  false> frontier;
    Operator op;

    static constexpr bool is_scan = Operator::is_scan;
    
    wrap_frontier_read3( const StoreTy *frontier_, Operator op_ )
	: frontier( (StoreTy *) frontier_ ), // TODO
	  op( op_ ) {
	// assert( frontier_ );
    }
    
    // This works well for vector sources, but what about scalars?
    // In those cases we would like to use a conditional
    // In AVX-512, we have a register mask and we might want to use a
    // conditional as well in case all lanes are false
    template<typename VIDSrc, typename VIDDst>
    auto relax( VIDSrc s, VIDDst d ) {
	return op.relax(
	    add_mask( remove_mask( s ),
		      expr::make_unop_cvt_to_pmask<StoreTy>( frontier[s] ) ),
	    d );
    }

    template<typename VIDDst>
    auto active( VIDDst d ) {
	return op.active( d );
    }

    template<typename VIDType>
    auto operator() ( VIDType vid ) {
	return op(
	    add_mask( remove_mask( vid ),
		      expr::make_unop_cvt_to_pmask<StoreTy>( frontier[vid] ) )
	    );
    }

    template<typename VIDDst>
    auto vertexop( VIDDst d ) {
	return op.vertexop( d );
    }
};


// Version operating on k-mask
// Used frequently in edgemap in older, scalar versions of edge traversal
template<typename Operator,typename StoreTy>
struct wrap_frontier_readf {
    expr::array_ro<StoreTy, VID, expr::aid_frontier_old_f,
		   array_encoding<StoreTy>,  false> frontier;
    Operator op;

    static constexpr bool is_scan = Operator::is_scan;
    
    wrap_frontier_readf( StoreTy *frontier_, Operator op_ )
	: frontier( frontier_ ), op( op_ ) { }
    
    // This works well for vector sources, but what about scalars?
    // In those cases we would like to use a conditional
    // In AVX-512, we have a register mask and we might want to use a
    // conditional as well in case all lanes are false
    template<typename VIDSrc, typename VIDDst>
    auto relax( VIDSrc s, VIDDst d ) {
	using Tr = typename simd::detail::vdata_traits<StoreTy,VIDDst::VL>; 
	static_assert( VIDSrc::VL == 1 || simd::detail::is_mask_bit_traits<Tr>::value, "test" );
	return op.relax(
	    add_mask( remove_mask( s ),
		      expr::make_unop_cvt_to_mask<Tr>( frontier[s] ) ),
	    d );
    }

    template<typename VIDDst>
    auto active( VIDDst d ) {
	return op.active( d );
    }

    template<typename VIDDst>
    auto vertexop( VIDDst d ) {
	return op.vertexop( d );
    }
};

template<typename Operator>
struct wrap_is_active {
    Operator op;

    static constexpr bool is_scan = Operator::is_scan;
    
    wrap_is_active( Operator op_ )
	: op( op_ ) { }
    
    template<typename VIDSrc, typename VIDDst>
    auto relax( VIDSrc s, VIDDst d ) {
	return op.relax( s, add_mask( d, op.active( d ) ) );
    }

    template<typename VIDDst>
    auto active( VIDDst d ) {
	return op.active( d );
    }

    template<typename VIDDst>
    auto vertexop( VIDDst d ) {
	return op.vertexop( d );
    }
};

#if 0
// Used frequently in edgemap in older, scalar versions of edge traversal
template<typename Operator, typename StoreTy, typename OperateTy>
struct wrap_frontier_update2 {
    expr::array_ro<StoreTy, VID, expr::aid_frontier_new_2,
		   array_encoding<StoreTy>,  false> frontier;
    Operator op;

    static constexpr bool is_scan = Operator::is_scan;
    
    wrap_frontier_update2( StoreTy *frontier_, Operator op_ )
	: frontier( frontier_ ), op( op_ ) { }
    
    // This works well for vector sources, but what about scalars?
    // In those cases we would like to use a conditional
    // In AVX-512, we have a register mask and we might want to use a
    // conditional as well in case all lanes are false
    template<typename VIDSrc, typename VIDDst>
    auto relax( VIDSrc s, VIDDst d ) {
	return frontier[d]
	    |= expr::make_unop_cvt_to_bool<OperateTy,StoreTy>( op.relax( s, d ) );
    }

    template<typename VIDDst>
    auto active( VIDDst d ) {
	return op.active( d );
    }

    template<typename VIDDst>
    auto vertexop( VIDDst d ) {
	return op.vertexop( d );
    }
};

// Specialisation for cases where we do not need a conversion operation
// Used frequently in edgemap in older, scalar versions of edge traversal
template<typename Operator, typename StoreTy>
struct wrap_frontier_update2<Operator,StoreTy,StoreTy> {
    expr::array_ro<StoreTy, VID, expr::aid_frontier_new_2b,
		   array_encoding<StoreTy>,  false> frontier;
    Operator op;

    static constexpr bool is_scan = Operator::is_scan;
    
    wrap_frontier_update2( StoreTy *frontier_, Operator op_ )
	: frontier( frontier_ ), op( op_ ) { }
    
    // This works well for vector sources, but what about scalars?
    // In those cases we would like to use a conditional
    // In AVX-512, we have a register mask and we might want to use a
    // conditional as well in case all lanes are false
    template<typename VIDSrc, typename VIDDst>
    auto relax( VIDSrc s, VIDDst d ) {
	return frontier[d] |= op.relax( s, d );
    }

    template<typename VIDDst>
    auto active( VIDDst d ) {
	return op.active( d );
    }

    template<typename VIDDst>
    auto vertexop( VIDDst d ) {
	return op.vertexop( d );
    }
};
#endif

// Version working on AVX-512 flags register
// Used frequently in edgemap in older, scalar versions of edge traversal
template<typename Operator, typename StoreTy>
struct wrap_frontier_updatef {
    expr::array_ro<StoreTy, VID, expr::aid_frontier_new_f,
		   array_encoding<StoreTy>,  false> frontier;
    Operator op;

    static constexpr bool is_scan = Operator::is_scan;
    
    wrap_frontier_updatef( StoreTy *frontier_, Operator op_ )
	: frontier( frontier_ ), op( op_ ) { }
    
    // This works well for vector sources, but what about scalars?
    // In those cases we would like to use a conditional
    // In AVX-512, we have a register mask and we might want to use a
    // conditional as well in case all lanes are false
    template<typename VIDSrc, typename VIDDst>
    auto relax( VIDSrc s, VIDDst d ) {
	return frontier[d]
	    |= expr::make_unop_cvt_to_vector<StoreTy>( op.relax( s, d ) );
	// TODO: want to write this for CSxSIMD, but then also need to change
	//       rewrite_caches to make the cache for an array with logicalor
	//       reduction be a kmask (only in the case of AVX-512).
	//       In the case of COOSIMD, there is no cache, so it should retain
	//       the conversion from mask to vector (unless if we have
	//       bitwise scatters?
	// |= op.relax( s, d );
    }

    template<typename VIDDst>
    auto active( VIDDst d ) {
	return op.active( d );
    }

    template<typename VIDDst>
    auto vertexop( VIDDst d ) {
	return op.vertexop( d );
    }
};

// Version working on AVX-512 flags register
// Key difference is in the use of bitarray_ro instead of array_ro and
// adding detail in the indexing of the bitarray_ro.
template<typename Operator, frontier_type ftype, unsigned short VL>
struct wrap_frontier_update_m {
    using StoreTy = typename frontier_params<ftype,VL>::type;
    using array_ty = typename expr::array_select<ftype,StoreTy,VID,expr::aid_frontier_new_f,array_encoding<StoreTy>>::type;
    // array_ty<StoreTy,VID, expr::aid_frontier_new_f> frontier;
    array_ty frontier;
    Operator op;

    static constexpr bool is_scan = Operator::is_scan;
    static constexpr bool is_idempotent = Operator::is_idempotent;
    
    wrap_frontier_update_m( StoreTy *frontier_, Operator op_ )
	: frontier( frontier_ ), op( op_ ) { }
    wrap_frontier_update_m( unsigned char *frontier_, Operator op_ )
	: frontier( reinterpret_cast<StoreTy *>( frontier_ ) ), op( op_ ) { }
    
    // This works well for vector sources, but what about scalars?
    // In those cases we would like to use a conditional
    // In AVX-512, we have a register mask and we might want to use a
    // conditional as well in case all lanes are false
    template<typename VIDSrc, typename VIDDst>
    auto relax( VIDSrc s, VIDDst d ) {
	// using Tr = typename simd::detail::mask_bit_traits<VIDDst::VL>; 
	using Tr = typename frontier_params<ftype,VL>::data_type;
	return frontier[d]
	    |= // expr::make_unop_cvt_to_mask<Tr>(
	    op.relax( s, d ); // );
	// TODO: want to write this for CSxSIMD, but then also need to change
	//       rewrite_caches to make the cache for an array with logicalor
	//       reduction be a kmask (only in the case of AVX-512).
	//       In the case of COOSIMD, there is no cache, so it should retain
	//       the conversion from mask to vector (unless if we have
	//       bitwise scatters?
	// |= op.relax( s, d );
    }

    template<typename VIDDst>
    auto active( VIDDst d ) {
	return op.active( d );
    }

    template<typename VIDDst>
    auto update( VIDDst d ) {
	return op.update( d );
    }

    template<typename VIDDst>
    auto different( VIDDst d ) {
	return op.different( d );
    }

    template<typename VIDDst>
    auto vertexop( VIDDst d ) {
	return op.vertexop( d );
    }
};

// TODO: this class should be merged into the wrap_scan_filter class
// in vertexmap.h
template<typename GraphType, typename Operator, frontier_type ftype,
	 unsigned short VL>
struct wrap_filter {
    using self_type = wrap_filter<GraphType,Operator,ftype,VL>;
    using StoreTy = typename frontier_params<ftype,VL>::type;
    using array_ty =
	typename expr::array_select<ftype,StoreTy,VID,
				    expr::aid_frontier_new_f,
				    array_encoding<StoreTy>,
				    false,true>::type;
    const GraphType & G;
    array_ty a_f;
    frontier & f;
    Operator op;

    static constexpr bool is_scan = Operator::is_scan;
    static constexpr bool is_idempotent = Operator::is_idempotent;
    
    wrap_filter( const GraphType & G_, frontier & f_, Operator op_ )
	: G( G_ ), a_f( reinterpret_cast<StoreTy *>( f_.getDense<ftype>() ) ),
	  f( f_ ), op( op_ ) { }

    template<typename VIDDst>
    auto operator() ( VIDDst vid ) {
	VID * nactvp = f.nActiveVerticesPtr();
	*nactvp = 0;
	expr::array_ro<VID,VID,expr::aid_frontier_nactv> a_v( nactvp );

	EID * nactep = f.nActiveEdgesPtr();
	*nactep = 0;
	expr::array_ro<EID,VID,expr::aid_frontier_nacte> a_e( nactep );

	expr::array_ro<VID,VID,expr::aid_graph_degree>
	    a_d( const_cast<VID *>( G.getOutDegree() ) );

	auto val = op( vid );
	using dTr = typename std::decay_t<decltype(val)>::data_type;
	using Tr = typename dTr::prefmask_traits;

	return expr::let<expr::aid_frontier_a>(
	    expr::make_unop_switch_to_vector( val ),
	    [&]( auto a ) {
		auto z = expr::zero_val(vid);
		return expr::make_seq(
		    a_v[z] += expr::add_predicate(
			expr::constant_val_one( a_v[z] ),
			expr::make_unop_cvt_to_mask<Tr>( a ) ),
		    a_e[z] += expr::add_predicate(
			expr::cast<EID>( a_d[vid] ),
			expr::make_unop_cvt_to_mask<Tr>( a ) ),
		    a_f[vid] = expr::cast<StoreTy>( a ) // a is vector, not mask
		    );
	    }
	    );
    }

    template<typename map_type0>
    struct ptrset {
	using map_type1 = typename Operator::ptrset<map_type0>::map_type;
	using map_type2 =
	    typename expr::ast_ptrset::ptrset_pointer<
	    expr::aid_frontier_new_f,StoreTy,map_type1>::map_type;
	using map_type3 =
	    typename expr::ast_ptrset::ptrset_pointer<
	    expr::aid_frontier_nacte, EID, map_type2>::map_type;
	using map_type4 =
	    typename expr::ast_ptrset::ptrset_pointer<
	    expr::aid_frontier_nactv, VID, map_type3>::map_type;
	using map_type =
	    typename expr::ast_ptrset::ptrset_pointer<
	    expr::aid_graph_degree, VID, map_type4>::map_type;

	template<typename MapTy>
	static void initialize( MapTy & map, const self_type & op ) {
	    Operator::template ptrset<map_type0>::initialize( map, op.op );
	    expr::ast_ptrset::ptrset_pointer<
		expr::aid_frontier_new_f,
		typename frontier_params<ftype,VL>::type,
		map_type1
		>::initialize( map, op.f.template getDense<ftype>() );
	    expr::ast_ptrset::ptrset_pointer<
		expr::aid_frontier_nacte, EID, map_type2
		>::initialize( map, op.f.nActiveEdgesPtr() );
	    expr::ast_ptrset::ptrset_pointer<
		expr::aid_frontier_nactv, VID, map_type3
		>::initialize( map, op.f.nActiveVerticesPtr() );
	    expr::ast_ptrset::ptrset_pointer<
		expr::aid_graph_degree, VID, map_type4
		>::initialize( map, const_cast<VID *>( op.G.getOutDegree() ) );
	}
    };
};

// Version operating on k-mask
template<typename Operator, frontier_type ftype, unsigned short VL>
struct wrap_frontier_read_m {
    using self_type = wrap_frontier_read_m<Operator,ftype,VL>;
    using StoreTy = typename frontier_params<ftype,VL>::type;
    using array_ty = typename expr::array_select<ftype,StoreTy,VID,expr::aid_frontier_old_f,array_encoding<StoreTy>,false,true>::type;
    array_ty frontier;
    Operator op;

    // static constexpr bool is_scan = Operator::is_scan;
    // static constexpr bool is_idempotent = Operator::is_idempotent;
    
    template<typename T>
    wrap_frontier_read_m( T *frontier_, Operator op_,
			  std::enable_if_t<std::is_same<T,unsigned char>::value
			  && !std::is_same<T,StoreTy>::value> * = nullptr )
	: frontier( reinterpret_cast<StoreTy *>( frontier_ ) ), op( op_ ) { }
    wrap_frontier_read_m( StoreTy *frontier_, Operator op_ )
	: frontier( frontier_ ), op( op_ ) { }
    
    // This works well for vector sources, but what about scalars?
    // In those cases we would like to use a conditional
    // In AVX-512, we have a register mask and we might want to use a
    // conditional as well in case all lanes are false
    template<typename VIDSrc, typename VIDDst>
    auto relax( VIDSrc s, VIDDst d ) {
	// using Tr = typename simd::detail::vdata_traits<StoreTy,VIDDst::VL>; 
	// using Tr = typename frontier_params<ftype,VL>::data_type;
	using Tr = simd::detail::mask_preferred_traits_type<typename VIDSrc::type, VIDSrc::VL>;
	return
	    expr::set_mask(
		expr::get_mask_cond( s ),
		expr::set_mask(
		    expr::make_unop_cvt_to_mask<Tr>( frontier[remove_mask( s )] ),
		    op.relax( remove_mask( s ), d ) ) );
    }

    template<typename VIDDst>
    auto active( VIDDst d ) {
	return op.active( d );
    }

    template<typename VIDDst>
    auto update( VIDDst d ) {
	return op.update( d );
    }

    template<typename VIDDst>
    auto different( VIDDst d ) {
	return op.different( d );
    }

    template<typename VIDDst>
    auto vertexop( VIDDst d ) {
	return op.vertexop( d );
    }

    template<typename VIDType>
    auto operator() ( VIDType vid ) {
	using Tr = simd::detail::mask_preferred_traits_type<typename VIDType::type, VIDType::VL>;
	return expr::set_mask(
	    expr::get_mask_cond( vid ),
	    expr::set_mask(
		expr::make_unop_cvt_to_mask<Tr>( frontier[remove_mask( vid )] ),
		op( remove_mask( vid ) ) ) );
    }

    template<typename map_type0>
    struct ptrset {
	using map_type1 = typename Operator::ptrset<map_type0>::map_type;
	using map_type =
	    typename expr::ast_ptrset::ptrset_pointer<
	    expr::aid_frontier_old_f,StoreTy,map_type1>::map_type;

	template<typename MapTy>
	static void initialize( MapTy & map, const self_type & op ) {
	    Operator::template ptrset<map_type0>::initialize( map, op.op );
	    expr::ast_ptrset::ptrset_pointer<
		expr::aid_frontier_old_f, StoreTy, map_type1
		>::initialize( map, op.frontier.ptr() );
	}
    };
};

template<typename Operator, unsigned short VL>
struct wrap_frontier_read_m<Operator,frontier_type::ft_bit2,VL> {
    using self_type = wrap_frontier_read_m<Operator,frontier_type::ft_bit2,VL>;
    using StoreTy = typename frontier_params<frontier_type::ft_bit2,VL>::type;
    using array_ty = typename expr::array_select<frontier_type::ft_bit2,bitfield<2>,VID,expr::aid_frontier_old_f,array_encoding_bit<2>,false,true>::type;
    array_ty frontier;
    Operator op;

    // static constexpr bool is_scan = Operator::is_scan;
    // static constexpr bool is_idempotent = Operator::is_idempotent;
    
    template<typename T>
    wrap_frontier_read_m( T *frontier_, Operator op_,
			  std::enable_if_t<std::is_same<T,unsigned char>::value
			  && !std::is_same<T,StoreTy>::value> * = nullptr )
	: frontier( reinterpret_cast<StoreTy *>( frontier_ ) ), op( op_ ) { }
    wrap_frontier_read_m( StoreTy *frontier_, Operator op_ )
	: frontier( frontier_ ), op( op_ ) { }
    
    // This works well for vector sources, but what about scalars?
    // In those cases we would like to use a conditional
    // In AVX-512, we have a register mask and we might want to use a
    // conditional as well in case all lanes are false
    template<typename VIDSrc, typename VIDDst>
    auto relax( VIDSrc s, VIDDst d ) {
	using Tr = typename simd::detail::mask_bit_logical_traits<2,VIDSrc::VL>;
	return set_mask(
	    get_mask_cond( s ),
	    set_mask(
		expr::make_unop_cvt_to_mask<Tr>( frontier[remove_mask(s)] ),
		op.relax( remove_mask( s ), d ) ) );
    }

    template<typename VIDDst>
    auto active( VIDDst d ) {
	return op.active( d );
    }

    template<typename VIDDst>
    auto update( VIDDst d ) {
	return op.update( d );
    }

    template<typename VIDDst>
    auto different( VIDDst d ) {
	return op.different( d );
    }

    template<typename VIDDst>
    auto vertexop( VIDDst d ) {
	return op.vertexop( d );
    }

    template<typename VIDType>
    auto operator() ( VIDType vid ) {
	using Tr = typename simd::detail::mask_bit_logical_traits<2,VIDType::VL>;
	return set_mask(
	    get_mask_cond( vid ),
	    set_mask(
		expr::make_unop_cvt_to_mask<Tr>( frontier[remove_mask(vid)] ),
		op( remove_mask( vid ) ) ) );
    }

    template<typename map_type0>
    struct ptrset {
	using map_type1 = typename Operator::ptrset<map_type0>::map_type;
	using map_type =
	    typename expr::ast_ptrset::ptrset_pointer<
	    expr::aid_frontier_old_f,StoreTy,map_type1>::map_type;

	template<typename MapTy>
	static void initialize( MapTy & map, const self_type & op ) {
	    Operator::template ptrset<map_type0>::initialize( map, op.op );
	    expr::ast_ptrset::ptrset_pointer<
		expr::aid_frontier_old_f, StoreTy, map_type1
		>::initialize( map, op.frontier.ptr() );
	}
    };
};


/***********************************************************************
 * PlaceHolder operator
 ***********************************************************************/
struct operator_getmask {
    static constexpr bool is_scan = false;
    static constexpr bool is_idempotent = true;
    
    template<typename VIDSrc, typename VIDDst>
    auto relax( VIDSrc s, VIDDst d ) {
	return expr::get_mask_cond( d );
    }

    template<typename VIDDst>
    auto active( VIDDst d ) {
	return expr::get_mask_cond( d );
    }

    template<typename VIDDst>
    auto vertexop( VIDDst d ) {
	return expr::get_mask_cond( d );
    }

    template<typename VIDType>
    auto operator() ( VIDType vid ) {
	return expr::get_mask_cond( vid );
    }
};

/***********************************************************************
 * Append profiling code to track utilization
 ***********************************************************************/
template<typename Operator, unsigned short VL>
struct wrap_utilization {
    using ProfTy = typename int_type_of_size<sizeof(VID)>::type;
    expr::array_ro<ProfTy,VID,expr::aid_utilization_active,
		   array_encoding<ProfTy>,false> nactive;
    expr::array_ro<ProfTy,VID,expr::aid_utilization_vectors,
		   array_encoding<ProfTy>,false> vectors;
    Operator op;

    static constexpr bool is_scan = true; // adding scan
    static constexpr bool is_idempotent = Operator::is_idempotent;
    
    wrap_utilization( ProfTy *nactive_, ProfTy *vectors_, Operator op_ )
	: nactive( nactive_ ), vectors( vectors_ ), op( op_ ) { }
    
    template<typename VIDSrc, typename VIDDst>
    auto relax( VIDSrc s, VIDDst d ) {
	auto e = op.relax( s, d );
	return expr::make_seq( analyze( s, d, e ), e );
    }

    template<typename VIDDst>
    auto active( VIDDst d ) {
	return op.active( d );
    }

    template<typename VIDDst>
    auto vertexop( VIDDst d ) {
	return op.vertexop( d );
	// auto e = op.vertexop( d );
	// return expr::make_seq( analyze( d, e ), e );
    }

    template<typename VIDType>
    auto operator() ( VIDType vid ) {
	return op( vid );
	// auto e = op( vid );
	// return expr::make_seq( analyze( vid, e ), e );
    }

private:
    template<typename E1, typename E2>
    auto combine_mask( E1 e1, E2 e2 ) {
	return e1 && e2;
    }
    template<typename E>
    auto combine_mask( E e, expr::noop ) {
	return e;
    }
    template<typename E>
    auto combine_mask( expr::noop, E e ) {
	return e;
    }
    auto combine_mask( expr::noop, expr::noop ) {
	return expr::noop();
    }
    template<typename MaskTy>
    auto analyze( MaskTy m ) {
	using Tr1 = simd::ty<VID,1>;
	using Tr = simd::ty<ProfTy,VL>;
	auto z = expr::make_unop_incseq<VL>(
	    // expr::value<Tr1,expr::vk_any>( VL ) *
	    expr::value<Tr1,expr::vk_pid>() );
	auto c0 = expr::value<Tr,expr::vk_zero>();
	auto c1 = expr::value<Tr,expr::vk_cstone>();
	return expr::make_seq(
	    expr::make_redop( nactive[z],
			      expr::iif( m, c0, c1 ),
			      expr::redop_add<true>() ),
	    expr::make_redop( vectors[z], c1, expr::redop_add<true>() ) );
    }
    auto analyze( expr::noop m ) {
	using Tr1 = simd::ty<VID,1>;
	using Tr = simd::ty<ProfTy,VL>;
	auto z = expr::make_unop_incseq<VL>(
	    // expr::value<Tr1,expr::vk_any>( VL ) *
	    expr::value<Tr1,expr::vk_pid>() );
	auto c1 = expr::value<Tr,expr::vk_cstone>();
	return expr::make_seq(
	    expr::make_redop( nactive[z], c1, expr::redop_add<true>() ),
	    expr::make_redop( vectors[z], c1, expr::redop_add<true>() ) );
    }
    template<typename VIDSrc, typename VIDDst, typename Expr>
    auto analyze( VIDSrc s, VIDDst d, Expr e ) {
	auto m = combine_mask( expr::get_mask( s ), expr::get_mask( d ) );
	return analyze( m );
    }
    template<typename VIDSrc, typename VIDDst>
    auto analyze( VIDSrc s, VIDDst d, expr::noop e ) {
	return e; // don't count lanes if no code is executed
    }
    template<typename VIDTy, typename Expr>
    auto analyze( VIDTy v, Expr e ) {
	auto m = expr::get_mask( v );
	return analyze( m );
    }
    template<typename VIDTy>
    auto analyze( VIDTy v, expr::noop e ) {
	return e; // don't count lanes if no code is executed
    }
};


/***********************************************************************
 * Append a vertexmap type operation to calculate the active frontier
 ***********************************************************************/
template<frontier_type new_fr_type,
	 short iAID, short nfAID, short avAID, short aeAID,
	 unsigned short VL, typename Operator>
class OpFrontierActiveCount {
    using fr_type = typename frontier_params<new_fr_type,VL>::type;
    expr::array_ro<VID,VID,iAID,array_encoding<VID>,false> degree;
    expr::array_ro<fr_type,VID,nfAID,array_encoding<fr_type>,false> new_frontier;
    expr::array_ro<VID,VID,avAID,array_encoding<VID>,false> nactv;
    expr::array_ro<EID,VID,aeAID,array_encoding<EID>,false> nacte;

public:
    OpFrontierActiveCount( const GraphCSx & G, frontier & newf,
			   VID * _nactv, EID *_nacte, Operator op_ )
	: degree( const_cast<VID *>( G.getDegree() ) ),
	new_frontier( newf.getDense<new_fr_type>() ),
	nactv( _nactv ), nacte( _nacte ) { }

    static constexpr bool is_scan = true;

    template<typename VIDType>
    auto operator() ( VIDType vid ) {
	// TODO: vid + [1]  is considered to be of lo_unknown type and results
	//       in a gather. It should be possible to note it is lo_linear
	//       resulting in an unaligned load.
	using ty = simd::detail::mask_preferred_traits_width<
	    frontier_params<new_fr_type,VIDType::VL>::W,
	    frontier_params<new_fr_type,VIDType::VL>::VL>;
	auto mask = expr::make_unop_cvt_to_mask<ty>( new_frontier[vid] );
        return expr::make_seq(
		nacte[expr::zero_val(vid)] +=
		expr::add_mask( expr::make_unop_cvt_type<EID>( degree[vid] ),
				mask ),
		nactv[expr::zero_val(vid)] +=
		expr::add_mask( expr::constant_val_one(vid), mask ) );
    }
};

template<short iAID, short nfAID, short avAID, short aeAID,
	 unsigned short VL, typename Operator>
class OpFrontierActiveCount<frontier_type::ft_unbacked,iAID,nfAID,avAID,aeAID,VL,Operator> {
    expr::array_ro<VID,VID,iAID,array_encoding<VID>,false> degree;
    expr::array_ro<VID,VID,avAID,array_encoding<VID>,false> nactv;
    expr::array_ro<EID,VID,aeAID,array_encoding<EID>,false> nacte;
    Operator op;

public:
    OpFrontierActiveCount( const GraphCSx & G, frontier & newf,
			   VID * _nactv, EID *_nacte, Operator op_ )
	: degree( const_cast<VID *>( G.getDegree() ) ),
	  nactv( _nactv ), nacte( _nacte ), op( op_ ) { }

    static constexpr bool is_scan = true;

    template<typename VIDType>
    auto operator() ( VIDType vid ) {
	auto mask = op.update( vid );
        return expr::make_seq(
		nacte[expr::zero_val(vid)] +=
		expr::add_mask( expr::make_unop_cvt_type<EID>( degree[vid] ),
				mask ),
		nactv[expr::zero_val(vid)] +=
		expr::add_mask( expr::constant_val_one(vid), mask ) );
    }
};

// TODO: getting rid of distinction between array_ro and bitarray_ro would
//       make code more compact here and in other places
template<short iAID, short nfAID, short avAID, short aeAID,
	 unsigned short VL, typename Operator>
class OpFrontierActiveCount<frontier_type::ft_bit,iAID,nfAID,avAID,aeAID,VL,Operator> {
    using fr_type = typename frontier_params<frontier_type::ft_bit,VL>::type;
    expr::array_ro<VID,VID,iAID,array_encoding<VID>,false> degree;
    expr::bitarray_ro<fr_type,VID,nfAID> new_frontier;
    expr::array_ro<VID,VID,avAID,array_encoding<VID>,false> nactv;
    expr::array_ro<EID,VID,aeAID,array_encoding<EID>,false> nacte;

public:
    OpFrontierActiveCount( const GraphCSx & G, frontier & newf,
			   VID * _nactv, EID *_nacte, Operator op_ )
	: degree( const_cast<VID *>( G.getDegree() ) ),
	  new_frontier( reinterpret_cast<fr_type *>( newf.template getDense<frontier_type::ft_bit>() ) ),
	  nactv( _nactv ), nacte( _nacte ) { }

    static constexpr bool is_scan = true;

    template<typename VIDType>
    auto operator() ( VIDType vid ) {
	// TODO: vid + [1]  is considered to be of lo_unknown type and results
	//       in a gather. It should be possible to note it is lo_linear
	//       resulting in an unaligned load.
	// TODO: remove hardcoded VID width of 4 in ft_logical4
	using Tr = typename simd::detail::mask_bit_traits<VIDType::VL>; 
	auto mask = expr::make_unop_cvt_to_mask<Tr>( new_frontier[vid] );
        return expr::make_seq(
		nacte[expr::zero_val(vid)] +=
		expr::add_mask( expr::make_unop_cvt_type<EID>( degree[vid] ),
				mask ),
		nactv[expr::zero_val(vid)] +=
		expr::add_mask( expr::constant_val_one(vid), mask ) );
    }
};


template<frontier_type new_fr_type, unsigned short VL,
	 typename Operator, typename Executor>
auto schedule_active_count(
    Operator op, Executor exec, const partitioner & part,
    const GraphCSx & CSR, frontier & new_frontier,
    typename std::enable_if<Operator::new_frontier != fm_all_true>::type *
    = nullptr ) {
    return exec.vertex_scan_unsafe(
	OpFrontierActiveCount<
	new_fr_type,
	expr::aid_graph_degree,
	expr::aid_frontier_new,
	expr::aid_frontier_nactv,
	expr::aid_frontier_nacte,
	VL,Operator>( // Assume vertexmap is not vectorised
	    CSR, new_frontier,
	    new_frontier.nActiveVerticesPtr(),
	    new_frontier.nActiveEdgesPtr(), op ) );
}

template<frontier_type new_fr_type, unsigned short VL,
	 typename Operator, typename Executor>
auto schedule_active_count(
    Operator op, Executor exec, const partitioner & part,
    const GraphCSx & CSR, frontier & new_frontier,
    typename std::enable_if<Operator::new_frontier == fm_all_true>::type *
    = nullptr ) {
    return exec;	
}

/***********************************************************************
 * Append the update operation from the edgemap operation/struct to
 * a vertexmap style execution
 ***********************************************************************/
template<typename Operator, frontier_type ftype, unsigned short VL>
struct wrap_frontier_store {
    using fr_type = typename frontier_params<ftype,VL>::type;
    expr::array_ro<fr_type, VID, expr::aid_frontier_new_store,
		   array_encoding<fr_type>, false> frontier;
    Operator op;

    static constexpr bool is_scan = Operator::is_scan;
    
    wrap_frontier_store( fr_type *frontier_, Operator op_ )
	: frontier( frontier_ ), op( op_ ) { }
    
    template<typename VIDDst>
    auto update( VIDDst d ) {
	return frontier[d]
	    = expr::make_unop_cvt_to_vector<fr_type>( op.update( d ) );
    }

    template<typename VIDDst>
    auto vertexop( VIDDst d ) {
	return op.vertexop( d );
    }
};

template<short nfAID, frontier_type NewFrontierTy, typename Operator,
	 unsigned short VL>
class OpFrontierCalculate {
    using fr_type = typename frontier_params<NewFrontierTy,VL>::type;
    expr::array_ro<fr_type,VID,nfAID,array_encoding<fr_type>,false> new_frontier;
    Operator op;

public:
    OpFrontierCalculate( frontier & newf, Operator _op )
	: new_frontier( newf.template getDense<NewFrontierTy>() ), op( _op ) { }

    static constexpr bool is_scan = false;

    template<typename VIDType>
    auto operator() ( VIDType vid ) {
	// TODO: for every VID or only for those active in old_frontier?
	// return sop.update( vid );
	return new_frontier[vid]
	    = expr::make_unop_cvt_to_vector<fr_type>( op.update( vid ) );
    }
};

template<short nfAID, typename Operator, unsigned short VL>
class OpFrontierCalculate<nfAID,frontier_type::ft_unbacked,Operator,VL> {
public:
    OpFrontierCalculate( frontier & newf, Operator _op ) { }

    static constexpr bool is_scan = false;

    template<typename VIDType>
    auto operator() ( VIDType vid ) {
	return expr::noop();
    }
};

template<short nfAID, typename Operator, unsigned short VL>
class OpFrontierCalculate<nfAID,frontier_type::ft_bit,Operator,VL> {
    using StoreTy = typename frontier_params<frontier_type::ft_bit,VL>::type;
    expr::bitarray_ro<StoreTy,VID,nfAID> new_frontier;
    Operator op;

public:
    OpFrontierCalculate( frontier & newf, Operator _op )
	: new_frontier( reinterpret_cast<StoreTy *>( newf.template getDense<frontier_type::ft_bit>() ) ), op( _op ) { }

    static constexpr bool is_scan = false;

    template<typename VIDType>
    auto operator() ( VIDType vid ) {
	// TODO: for every VID or only for those active in old_frontier?
	// return sop.update( vid );
	return new_frontier[vid]
	    = expr::make_unop_cvt_to_mask<typename simd::detail::mask_bit_traits<VIDType::VL>>(
		op.update( vid ) );
    }
};

template<short nfAID, frontier_type NewFrontierTy,
	 unsigned short VL, typename Operator, typename Executor>
auto schedule_calculate_frontier(
    const partitioner & part,
    Operator op, Executor exec,
    frontier & new_frontier,
    typename std::enable_if<Operator::new_frontier == fm_calculate>::type *
    = nullptr ) {
    return exec.vertex_map_unsafe(
	OpFrontierCalculate<nfAID,NewFrontierTy,Operator,VL>( new_frontier, op ) );
}

template<short nfAID, frontier_type NewFrontierTy,
	 unsigned short VL, typename Operator, typename Executor>
auto schedule_calculate_frontier(
    const partitioner & part,
    Operator op, Executor exec,
    frontier & new_frontier,
    typename std::enable_if<Operator::new_frontier != fm_calculate>::type *
    = nullptr ) {
    return exec;
}

/***********************************************************************
 * Calculate a sparse frontier
 ***********************************************************************/
template<typename Operator>
void calculate_sparse_frontier(
    frontier & new_frontier, const partitioner & part, Operator op ) {
    // TODO: vectorize using AVX512CD compress instruction
    const int np = part.get_num_partitions();

    new_frontier.toSparse( part );
    VID * sparse = new_frontier.getSparse();

    using Ty = simd::ty<VID,1>;
    auto vid = expr::value<Ty,expr::vk_vid>();
    auto vexpr0 = op.update( vid );
    auto vexpr = expr::rewrite_mask_main( vexpr0 );

    expr::cache<> c;

    // Avoid cache thrashing
    constexpr unsigned block_size = 64;
    constexpr unsigned mult = ( block_size + sizeof(VID) - 1 ) / sizeof(VID);

    VID * nactvp = new VID[(np+1)*mult];
    std::fill( &nactvp[0], &nactvp[(np+1)*mult], (VID)0 );

    // 1. count the number of active vertices in each partition
    map_partitionL( part, [&]( int p ) {
	    VID s = part.start_of( p );
	    VID e = part.end_of( p );
	    VID a = 0;

	    for( VID v=s; v < e; ++v ) {
		auto vv = simd::create_scalar( v );
		auto m = expr::create_value_map_new<1>(
		    expr::create_entry<expr::vk_vid>( vv ) );
		if( expr::evaluate_bool( c, m, vexpr ) )
		    ++a;
	    }
	    nactvp[p*mult] = a;
	} );

#if 0
    std::cerr << "nactv per p:";
    for( int p=0; p < np; ++p )
	std::cerr << ' ' << nactvp[p*mult];
    std::cerr << "\n";
#endif

    // 2. aggregate
    VID mov = 0;
    for( int p=0; p < np; ++p ) {
	VID tmp = nactvp[p*mult];
	nactvp[p*mult] = mov;
	mov += tmp;
    }
    VID nactv = nactvp[np*mult] = mov;
    if( nactv != new_frontier.nActiveVertices() ) {
	std::cerr << "active frontier mismatch: nactv=" << nactv
		  << " new_frontier: v: " << new_frontier.nActiveVertices()
		  << " e: " << new_frontier.nActiveEdges()
		  << "\n";
	abort();
    }

    for( VID v=0; v < nactv; ++v )
	sparse[v] = ~(VID)0;

    // 3. store indices
    map_partitionL( part, [&]( int p ) {
	    VID s = part.start_of( p );
	    VID e = part.end_of( p );
	    VID sidx = nactvp[p*mult];

	    for( VID v=s; v < e; ++v ) {
		auto vv = simd::create_scalar( v );
		auto m = expr::create_value_map_new<1>(
		    expr::create_entry<expr::vk_vid>( vv ) );
		if( expr::evaluate_bool( c, m, vexpr ) )
		    sparse[sidx++] = v;
	    }
	    assert( sidx == nactvp[(p+1)*mult] );
	} );

    // 4. clean up
    delete[] nactvp;
}

/***********************************************************************
 * Calculating the new frontier
 ***********************************************************************/
template<unsigned short VL, frontier_type NewFrontierTy, typename Operator>
static __attribute__((noinline)) void calculate_frontier(
    frontier & new_frontier, VID from, VID to, Operator op ) {
    // Infer frontier for the partition [from,to)

    wrap_frontier_store<Operator,NewFrontierTy,VL>
	sop( new_frontier.getDense<NewFrontierTy>(), op );

    auto vexpr0 = sop.update( expr::value<simd::ty<VID,VL>,expr::vk_dst>() );
    auto vexpr = expr::rewrite_mask_main( vexpr0 );
    
    simd::vec<simd::ty<VID, VL>,simd::lo_constant> step( (VID)VL );
    // unaligned unless we have guarantees from graph partitioning
    auto dst = simd_vector<VID, VL>::template s_set1inc<false>( from );

    tuple<> c;

    logical<sizeof(VID)> * farr = new_frontier.getDenseL<sizeof(VID)>();

    VID v = from;
    for( ; v < to; v += VL ) {
	// auto m = expr::value_map_dst<VID, VL, VL>( dst );
	auto m = expr::create_value_map2<VL,VL,expr::vk_dst>( dst );
	expr::evaluate( c, m, vexpr );
	dst += step;
    }

    if( v < to ) { // remainder loop
	auto sexpr0 = sop.update( expr::value<simd::ty<VID,1>,expr::vk_dst>() );
	auto sexpr = expr::rewrite_mask_main( sexpr0 );

	for( ; v < to; v++ ) {
	    auto dst = simd::create_scalar( v );
	    // auto m = expr::value_map_dst<VID, 1, 1>( dst );
	    auto m = expr::create_value_map2<VL,VL,expr::vk_dst>( dst );
	    expr::evaluate( c, m, sexpr );
	}
    }
}

template<unsigned short VL, frontier_type NewFrontierTy, typename Operator>
static void calculate_frontier_and_activ(
    const GraphCSx & G, frontier & new_frontier,
    VID from, VID to, Operator op ) {
    // Infer frontier for the partition [from,to)
    // and simultaneously calculate the number of active vertices and edges

    wrap_frontier_store<Operator,NewFrontierTy,VL>
	sop( new_frontier.getDense<NewFrontierTy>(), op );

    auto vexpr0 = sop.update( expr::value<simd::ty<VID,VL>,expr::vk_dst>() );
    auto vexpr = expr::rewrite_mask_main( vexpr0 );
    
    // simd_vector<VID, VL> step( (VID)VL );
    auto step = simd::template create_constant<simd::ty<VID,VL>>( (VID)VL );
    // unaligned unless we have guarantees from graph partitioning
    auto dst = simd_vector<VID, VL>::template s_set1inc<false>( from );

    tuple<> c;

    logical<sizeof(VID)> * farr = new_frontier.getDenseL<sizeof(VID)>();

    // simd_vector<VID, VL> nactv( (VID)0 );
    // simd_vector<EID, VL> nacte( (EID)0 );
    // simd_vector<VID, VL> vzero( (VID)0 );
    // simd_vector<VID, VL> vone( (VID)1 );
    simd::vec<simd::ty<VID, VL>,simd::lo_unknown> nactv( (VID)0 );
    simd::vec<simd::ty<EID, VL>,simd::lo_unknown> nacte( (EID)0 );
    auto vzero = simd::template create_zero<simd::ty<VID, VL>>();
    auto vone = simd::template create_one<simd::ty<VID, VL>>();

    // We won't change this, but simd_vector_ref not set up for constant refs
    const EID * index = const_cast<EID *>( G.getIndex() );

    VID v = from;
    for( ; v < to; v += VL ) {
	// auto m = expr::value_map_dst<VID, VL, VL>( dst );
	auto m = expr::create_value_map2<VL,VL,expr::vk_dst>( dst );
	auto r = expr::evaluate( c, m, vexpr );
	// r should be vector of logical<VID> x VL
	static_assert( r.W == 0, "there should be no mask" );
	nactv += blend( r.value(), vzero, vone );
	simd_vector_ref<EID, VID, VL> idx1( index, dst+vone );
	simd_vector_ref<EID, VID, VL> idx( index, dst );
	nacte += blend( r.value(), vzero, idx1.load() - idx.load() );
	dst += step;
    }

    VID nactvs = nactv.reduce_add();
    EID nactes = nacte.reduce_add();

    if( v < to ) { // remainder loop
	auto sexpr0 = sop.update( expr::value<simd::ty<VID,1>,expr::vk_dst>() );
	auto sexpr = expr::rewrite_mask_main( sexpr0 );

	for( ; v < to; v++ ) {
	    auto dst = simd::create_scalar( v );
	    // simd_vector<VID, 1> dst( v );
	    // auto m = expr::value_map_dst<VID, 1, 1>( dst );
	    auto m = expr::create_value_map2<1,1,expr::vk_dst>( dst );
	    auto r = expr::evaluate( c, m, sexpr );
	    if( r.value().data() ) {
		nactvs++;
		nacte += index[v+1] - index[v];
	    }
	}
    }

    new_frontier.setActiveCounts( nactvs, nactes );
}


template<typename GraphTypeCSR, typename Operator>
static void calculate_active_count( GraphTypeCSR & GA, frontier & new_frontier, Operator op ) {
    if( new_frontier.isSparse() ) {
	if( op.new_frontier != fm_all_true )
	    new_frontier.calculateActiveCounts( GA );
    } else {
	switch( op.new_frontier ) {
	case fm_all_true: break;
	case fm_reduction:
	case fm_calculate:
	    new_frontier.calculateActiveCounts( GA.getCSR() );
	    break;
	}
    }
}

/***********************************************************************
 * Active-set edgemap iteration
 ***********************************************************************/
template<typename Operator, short ofAID, frontier_type ftype, unsigned short VL>
struct OpActiveSetCheck {
    using fr_type = typename frontier_params<ftype,VL>::type;
    using mask_type = typename frontier_params<ftype,VL>::mask_type;

    static_assert( ftype != frontier_type::ft_bit,
		   "ft_bit needs different array type" );

    expr::array_intl<fr_type,VID,ofAID> m_active_set;
    Operator m_op;

    static constexpr frontier_mode new_frontier = Operator::new_frontier;
    static constexpr bool is_scan = Operator::is_scan;
    static constexpr bool is_idempotent = Operator::is_idempotent;
    static constexpr bool may_omit_frontier_rd = Operator::may_omit_frontier_rd;
    static constexpr bool may_omit_frontier_wr = Operator::may_omit_frontier_wr;
    static constexpr bool new_frontier_dense = Operator::new_frontier_dense;
    
    OpActiveSetCheck( Operator op, frontier & f )
	: m_op( op ), m_active_set( f.template getDense<ftype>() ) { }

    template<typename VIDSrc, typename VIDDst>
    auto relax( VIDSrc s, VIDDst d ) {
	auto actv = expr::make_unop_cvt_to_mask<mask_type>( m_active_set[d] );
	if constexpr ( may_omit_frontier_rd )
	    return m_op.relax( s, d );
	else
	    return m_op.relax( s, expr::add_mask( d, actv ) );
    }

    template<typename VIDDst>
    auto active( VIDDst d ) {
	auto actv = expr::make_unop_cvt_to_mask<mask_type>( m_active_set[d] );
	return m_op.active( d ) && actv;
    }

    template<typename VIDDst>
    auto update( VIDDst d ) {
	return m_op.update( d );
    }

    template<typename VIDDst>
    auto different( VIDDst d ) {
	return m_op.different( d );
    }

    template<typename VIDDst>
    auto vertexop( VIDDst d ) {
	return m_op.vertexop( d );
    }
};

template<short ofAID, frontier_type ftype, unsigned short VL, typename Operator>
auto wrap_active_set_check( Operator op, frontier & aset ) {
    return OpActiveSetCheck<Operator,ofAID,ftype,VL>( op, aset );
}

#endif // GRAPTOR_DSL_FRONTIER_WRAP_H
